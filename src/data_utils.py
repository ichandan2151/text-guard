import json
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def prepare_dataset(jsonl_path):
    samples = load_jsonl(jsonl_path)
    texts = [s["text"] for s in samples]
    labels = [s["labels"] for s in samples]  # list of lists
    return texts, labels

def fit_label_binarizer(train_labels):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(train_labels)
    return mlb, Y
