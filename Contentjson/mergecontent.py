import json
from pathlib import Path

# 1. Map each file to a document prefix used in chunk_id
files = {
    "content_BL-T1168.json": "BL-T1168",
    "content_BR-T1688.json": "BR-T1688",
    "content_RG-T4735.json": "RG-T4735",
    "content_TC_87565.json": "TC_87565",
    "content_RG-T4603.json": "RG-T4603",
}

merged = []

for filename, doc_prefix in files.items():
    path = Path(filename)
    if not path.exists():
        print(f"WARNING: {filename} not found, skipping.")
        continue

    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    # make sure chunks is a list
    if isinstance(chunks, dict):
        chunks = [chunks]

    # reassign chunk_id to document-based ids
    for i, obj in enumerate(chunks, start=1):
        obj["chunk_id"] = f"{doc_prefix}-{i}"
        merged.append(obj)

# 2. Save merged content
with open("merged_content.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Merged {len(merged)} chunks into merged_content.json")
