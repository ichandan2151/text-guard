import json
import random
from pathlib import Path

# ---------- CONFIG ----------
INPUT_FILES = [
    ("content_transport.json", "transport"),
    ("content_urbandevHousing.json", "urban_housing"),
    ("content_WaterSanitation.json", "water_sanitation"),
]

OUTPUT_DATASET = "dataset.jsonl"
OUTPUT_TRAIN = "train.jsonl"
OUTPUT_VAL = "val.jsonl"
OUTPUT_TEST = "test.jsonl"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # test will be 1 - train - val
RANDOM_SEED = 42
# ----------------------------


def load_chunks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    all_records = []

    for filename, sector in INPUT_FILES:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        chunks = load_chunks(path)

        if not isinstance(chunks, list):
            raise ValueError(f"{path} is not a JSON array.")

        for ch in chunks:
            record = {
                "sector": sector,
                "chunk_id": ch.get("chunk_id"),
                "chunk_title": ch.get("chunk_title"),
                "text": ch.get("chunk"),
                "labels": ch.get("policy_references", []),
                "risk_level": ch.get("risk_level"),
                "compliance_status": ch.get("compliance_status"),
            }
            all_records.append(record)

    print(f"Total records loaded: {len(all_records)}")

    # ---------- WRITE FULL DATASET ----------
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as out_f:
        for rec in all_records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- TRAIN / VAL / TEST SPLIT ----------
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)

    n = len(all_records)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    n_test = n - n_train - n_val

    train_data = all_records[:n_train]
    val_data = all_records[n_train:n_train + n_val]
    test_data = all_records[n_train + n_val:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    def write_split(fname, data):
        with open(fname, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_split(OUTPUT_TRAIN, train_data)
    write_split(OUTPUT_VAL, val_data)
    write_split(OUTPUT_TEST, test_data)

    print("Wrote:", OUTPUT_DATASET, OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST)


if __name__ == "__main__":
    main()
