import os
import re
import pdfplumber
import pandas as pd
from nltk.tokenize import sent_tokenize

PDF_DIR = "data/pdfs"
OUT_MASTER = "data/master_sentences.csv"

# crude helpers -----------------------------------------------------

def looks_like_header_or_footer(line: str) -> bool:
    line_stripped = line.strip()
    if not line_stripped:
        return False
    # page numbers like "1", "- 1 -", "Page 2 of 7"
    if re.fullmatch(r"-?\s*\d+\s*-?", line_stripped):
        return True
    if re.search(r"Page\s+\d+\s+of\s+\d+", line_stripped, re.IGNORECASE):
        return True
    # common boilerplate terms – you can extend this list
    header_keywords = [
        "inter-american development bank",
        "idb", "tc abstract", "project document", "technical cooperation document"
    ]
    lower = line_stripped.lower()
    if any(k in lower for k in header_keywords) and len(line_stripped) < 80:
        return True
    return False


def clean_inline_refs(text: str) -> str:
    # remove [1], [23], (Annex 1) – keep simple
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(see\s+table\s+\d+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(annex\s+\w+\)", "", text, flags=re.IGNORECASE)
    return text


def is_header_paragraph(p: str) -> bool:
    stripped = p.strip()

    # Reject paragraphs that are too long to be headers
    if len(stripped) > 40:
        return False

    # True headers are usually 1–5 words max
    word_count = len(stripped.split())
    if word_count <= 5:
        # patterns like "I. Background", "II. Description"
        if re.match(r"^[IVXLC]+\.\s+[A-Za-z]", stripped):
            return True
        # fully uppercase short titles
        if stripped.isupper():
            return True
        # short section-like titles ending with :
        if stripped.endswith(":"):
            return True

    return False



def is_bullet_paragraph(p: str) -> bool:
    stripped = p.lstrip()
    # bullets or numbered list
    if stripped.startswith(("•", "-", "–", "—")):
        return True
    if re.match(r"^\d+[\).\s]", stripped):  # "1.", "2)", "3 "
        return True
    return False


def infer_document_type(fname: str) -> str:
    lower = fname.lower()
    if "abstract" in lower:
        return "TC_ABSTRACT"
    if "project document" in lower:
        return "PROJECT_DOCUMENT"
    if "disclosure" in lower:
        return "DISCLOSURE"
    if "cclip" in lower:
        return "CCLIP"
    return "OTHER"


def infer_sector_guess(fname: str, full_text: str) -> str:
    lower_all = (fname + " " + full_text[:3000]).lower()
    if any(w in lower_all for w in ["water", "sanitation", "drainage", "wastewater"]):
        return "Water and Sanitation"
    if any(w in lower_all for w in ["urban", "housing", "city", "cities"]):
        return "Urban development and housing"
    # fallback
    return "Unknown"


def infer_country_guess(full_text: str) -> str:
    # extremely simple heuristic based on patterns like "Country: X" or "Country/Region: X"
    m = re.search(r"Country/Region:\s*([A-Za-z\s]+)", full_text)
    if m:
        return m.group(1).strip()
    m = re.search(r"Country:\s*([A-Za-z\s]+)", full_text)
    if m:
        return m.group(1).strip()
    return "Unknown"

# main --------------------------------------------------------------

rows = []

for fname in os.listdir(PDF_DIR):
    if not fname.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_DIR, fname)
    print(f"Processing {pdf_path}...")
    all_text_lines = []

    # extract page text, line by line
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            for line in txt.splitlines():
                if not looks_like_header_or_footer(line):
                    all_text_lines.append(line)

    # join lines into paragraphs (split on blank lines)
    paragraphs = []
    buf = []
    for line in all_text_lines:
        if line.strip():
            buf.append(line.strip())
        else:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
    if buf:
        paragraphs.append(" ".join(buf))

    full_text_for_meta = "\n".join(paragraphs)
    document_type = infer_document_type(fname)
    sector_guess = infer_sector_guess(fname, full_text_for_meta)
    country_guess = infer_country_guess(full_text_for_meta)

    sentence_id = 0
    for para in paragraphs:
        para = clean_inline_refs(para).strip()
        if not para:
            continue

        if is_header_paragraph(para):
            # keep header as a single unit
            rows.append({
                "doc_id": fname,
                "sentence_id": sentence_id,
                "sentence_text": para,
                "sentence_type": "header",
                "document_type": document_type,
                "sector_guess": sector_guess,
                "country_guess": country_guess,
            })
            sentence_id += 1

        elif is_bullet_paragraph(para):
            # bullet / list paragraph kept as one block
            rows.append({
                "doc_id": fname,
                "sentence_id": sentence_id,
                "sentence_text": para,
                "sentence_type": "bullet",
                "document_type": document_type,
                "sector_guess": sector_guess,
                "country_guess": country_guess,
            })
            sentence_id += 1

        else:
            # normal paragraph → split into sentences (strict)
            for sent in sent_tokenize(para):
                sent = sent.strip()
                if not sent:
                    continue
                rows.append({
                    "doc_id": fname,
                    "sentence_id": sentence_id,
                    "sentence_text": sent,
                    "sentence_type": "normal",
                    "document_type": document_type,
                    "sector_guess": sector_guess,
                    "country_guess": country_guess,
                })
                sentence_id += 1

# build dataframe and save
df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv(OUT_MASTER, index=False)
print(f"Saved master dataset to {OUT_MASTER}, {len(df)} rows")
