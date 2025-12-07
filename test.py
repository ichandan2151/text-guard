import pdfplumber

path = "data/pdfs/TC abstract DR-T1318.pdf"
with pdfplumber.open(path) as pdf:
    print([len(p.extract_text() or "") for p in pdf.pages])
