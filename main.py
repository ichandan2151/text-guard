import os
import io
import re
from typing import List, Optional

import requests
import PyPDF2
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI(title="TextGuard Processor", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later to your Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Config / environment
# -------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "model")
PORT = int(os.environ.get("PORT", "8080"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")  # e.g. https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "documents")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print(
        "WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set. "
        "Service will run, but DB integration is disabled."
    )
    REST_BASE = None
else:
    REST_BASE = f"{SUPABASE_URL.rstrip('/')}/rest/v1"
    print("Supabase env vars present ✅ DB integration enabled.")

# -------------------------
# Load model once on startup
# -------------------------
print(f"Loading model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print("Model loaded ✅")

# -------------------------
# Pydantic models
# -------------------------
class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: str
    score: float
    probs: List[float]


class ProcessDocumentRequest(BaseModel):
    document_id: str
    storage_path: str  # path inside the Supabase bucket
    file_name: Optional[str] = None
    project_name: Optional[str] = None


class ProcessDocumentResponse(BaseModel):
    document_id: str
    total_chunks: int
    risk_chunks: int
    no_risk_chunks: int
    risk_score: float
    storage_url: str


# -------------------------
# Helpers: PDF → text
# -------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def extract_sections(full_text: str):
    """
    Try to split the document into sections with Roman numeral headings:
    e.g. "II. Objective and Justification"
    """
    pattern = re.compile(r"(?m)^\s*([IVX]+)\.\s+([^\n]+)")
    sections = []

    for match in pattern.finditer(full_text):
        sec_roman = match.group(1)
        sec_title = match.group(2).strip()
        start = match.start()
        sections.append({"roman": sec_roman, "title": sec_title, "start": start})

    # Optional: special case for "V. Executing Agency..." if needed
    if not any(s["roman"] == "V" for s in sections):
        m = re.search(r"V\.\s+Executing Agency and Execution Structure", full_text)
        if m:
            sections.append(
                {
                    "roman": "V",
                    "title": "Executing Agency and Execution Structure",
                    "start": m.start(),
                }
            )

    if not sections:
        # Fallback: treat whole document as one section
        return [
            {
                "roman": "I",
                "title": "Full document",
                "start": 0,
                "text": full_text,
            }
        ]

    sections = sorted(sections, key=lambda s: s["start"])

    # Slice full text for each section
    for i, sec in enumerate(sections):
        start = sec["start"]
        end = sections[i + 1]["start"] if i + 1 < len(sections) else len(full_text)
        sec["text"] = full_text[start:end].strip()

    return sections


def chunk_section_text(section_text: str, max_words: int = 180) -> List[str]:
    # Remove the first line (heading) if present
    parts = section_text.split("\n", 1)
    if len(parts) == 2:
        _, body = parts
    else:
        body = section_text

    body = body.replace("\n", " ")
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", body)

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sent in sentences:
        if not sent:
            continue
        w = len(sent.split())
        if current and current_words + w > max_words:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_words = w
        else:
            current.append(sent)
            current_words += w

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def build_chunks_for_document(
    doc_id: str, full_text: str, max_words: int = 180
) -> List[dict]:
    sections = extract_sections(full_text)
    rows: List[dict] = []

    for sec in sections:
        section_text = sec["text"]
        section_id = f"sec{sec['roman']}"
        section_title = sec["title"]

        chunk_texts = chunk_section_text(section_text, max_words=max_words)

        for idx, chunk_text in enumerate(chunk_texts):
            rows.append(
                {
                    "document_id": doc_id,
                    "section_id": section_id,
                    "section_title": section_title,
                    "section_text": section_text,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    # labels filled after model inference
                    "risk_label": "",
                    "risk_type": "",
                    "policy_codes": "",
                }
            )

    return rows


# -------------------------
# Helpers: model inference
# -------------------------
def classify_chunks_in_place(chunk_rows: List[dict], batch_size: int = 16):
    """
    Run BERT on each chunk_text and fill risk_label / risk stats in-place.
    Assumes model.config.id2label has 0->"no_risk", 1->"risk".
    """
    if not chunk_rows:
        return 0, 0

    texts = [row["chunk_text"] for row in chunk_rows]
    id2label = model.config.id2label

    risk_chunks = 0
    no_risk_chunks = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        preds = probs.argmax(axis=-1)

        for j, (pred_id, prob_vec) in enumerate(zip(preds, probs)):
            row = chunk_rows[i + j]
            label = id2label.get(int(pred_id), str(int(pred_id)))

            row["risk_label"] = label
            row["risk_type"] = ""  # you can extend later
            row["policy_codes"] = ""  # you can extend later

            if label == "risk":
                risk_chunks += 1
            else:
                no_risk_chunks += 1

    return risk_chunks, no_risk_chunks


# -------------------------
# Helpers: Supabase HTTP
# -------------------------
def supabase_headers():
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY not set")
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def insert_chunks(rows: List[dict]):
    if not REST_BASE:
        print("REST_BASE not set – skipping chunk insert.")
        return

    if not rows:
        return

    url = f"{REST_BASE}/chunks"  # ✅ no '?minimal'
    headers = supabase_headers()

    resp = requests.post(url, json=rows, headers=headers, timeout=60)
    if not resp.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to insert chunks: {resp.status_code} {resp.text}",
        )


def update_document_summary(
    document_id: str,
    file_name: Optional[str],
    storage_path: str,
    project_name: Optional[str],
    total_chunks: int,
    risk_chunks: int,
    no_risk_chunks: int,
    storage_url: str,
):
    if not REST_BASE:
        print("REST_BASE not set – skipping document update.")
        return

    if total_chunks == 0:
        risk_score = 0.0
    else:
        risk_score = risk_chunks / float(total_chunks)

    url = f"{REST_BASE}/documents?id=eq.{document_id}"  # ✅ no '?minimal'
    headers = supabase_headers()
    payload = {
        "file_name": file_name,
        "storage_path": storage_path,
        "document_link": storage_url,
        "project_name": project_name,
        "total_chunks": total_chunks,
        "risk_chunks": risk_chunks,
        "no_risk_chunks": no_risk_chunks,
        "risk_score": risk_score,
    }

    resp = requests.patch(url, json=payload, headers=headers, timeout=30)
    if not resp.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update document: {resp.status_code} {resp.text}",
        )

    return risk_score


# -------------------------
# FastAPI endpoints
# -------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "healthy"}


@app.post("/classify", response_model=ClassifyResponse)
def classify_text(req: ClassifyRequest):
    inputs = tokenizer(
        req.text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_id = int(probs.argmax())
    id2label = model.config.id2label
    label = id2label.get(pred_id, str(pred_id))

    return ClassifyResponse(
        label=label,
        score=float(probs[pred_id]),
        probs=probs.tolist(),
    )


@app.post("/process_document", response_model=ProcessDocumentResponse)
def process_document(req: ProcessDocumentRequest):
    """
    1. Download PDF from Supabase storage (public bucket).
    2. Extract text & chunk.
    3. Run BERT on each chunk.
    4. Insert rows into 'chunks' table.
    5. Update 'documents' row with summary stats.
    """

    if not SUPABASE_URL or not SUPABASE_BUCKET:
        raise HTTPException(
            status_code=500,
            detail="Supabase config missing on server.",
        )

    # 1) Build public URL to the file
    storage_url = (
        f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/"
        f"{SUPABASE_BUCKET}/{req.storage_path.lstrip('/')}"
    )

    # 2) Download the PDF
    pdf_resp = requests.get(storage_url, timeout=60)
    if not pdf_resp.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download PDF from storage: {pdf_resp.status_code}",
        )

    full_text = extract_text_from_pdf_bytes(pdf_resp.content)

    # 3) Chunk
    chunk_rows = build_chunks_for_document(req.document_id, full_text, max_words=140)

    # 4) Classify chunks
    risk_chunks, no_risk_chunks = classify_chunks_in_place(chunk_rows)
    total_chunks = len(chunk_rows)

    # 5) Insert chunks into DB
    insert_chunks(chunk_rows)

    # 6) Update document summary
    risk_score = update_document_summary(
        document_id=req.document_id,
        file_name=req.file_name,
        storage_path=req.storage_path,
        project_name=req.project_name,
        total_chunks=total_chunks,
        risk_chunks=risk_chunks,
        no_risk_chunks=no_risk_chunks,
        storage_url=storage_url,
    )

    return ProcessDocumentResponse(
        document_id=req.document_id,
        total_chunks=total_chunks,
        risk_chunks=risk_chunks,
        no_risk_chunks=no_risk_chunks,
        risk_score=float(risk_score),
        storage_url=storage_url,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
