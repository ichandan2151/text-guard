import os
import joblib
import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertForSequenceClassification

# ---------- Paths ----------
BASELINE_DIR = "models/sentence_baseline"
TRANSFORMER_DIR = "models/transformer_finetuned"

# ---------- Load sentence baseline ----------
def load_sentence_baseline():
    clf = joblib.load(os.path.join(BASELINE_DIR, "classifier.joblib"))
    mlb = joblib.load(os.path.join(BASELINE_DIR, "label_binarizer.joblib"))
    encoder = SentenceTransformer(os.path.join(BASELINE_DIR, "encoder"))
    return encoder, clf, mlb

# ---------- Load transformer ----------
def load_transformer():
    tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMER_DIR)
    model = BertForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    mlb = joblib.load(os.path.join(TRANSFORMER_DIR, "label_binarizer.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, mlb, device

# ---------- Prediction helpers ----------

def predict_baseline(text, encoder, clf, mlb, threshold=0.3):
    # Encode text
    emb = encoder.encode([text])  # shape (1, hidden_dim)
    # Predict multi-label probabilities (decision_function or predict_proba)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(emb)[0]  # shape (num_labels,)
    else:
        # Fallback for models without predict_proba
        scores = clf.decision_function(emb)[0]
        # Map scores to 0-1 via sigmoid
        probs = 1 / (1 + np.exp(-scores))

    # Apply threshold
    binary = (probs >= threshold).astype(int)

    # Map to label names
    labels_on = [label for label, flag in zip(mlb.classes_, binary) if flag == 1]

    return labels_on, probs

def predict_transformer(text, tokenizer, model, mlb, device, threshold=0.3):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape (1, num_labels)

    probs = torch.sigmoid(logits).cpu().numpy()[0]
    binary = (probs >= threshold).astype(int)

    labels_on = [label for label, flag in zip(mlb.classes_, binary) if flag == 1]

    return labels_on, probs

# ---------- Main demo ----------

def main():
    # 1) Load models
    print("Loading baseline model...")
    encoder, clf, mlb_baseline = load_sentence_baseline()

    print("Loading transformer model...")
    tokenizer, model, mlb_transformer, device = load_transformer()

    # 2) Example paragraph (you can change this or read from input)
    example_text = """
    The program aims to improve urban water and drainage systems, 
    strengthen institutional capacity for utility management, 
    and support climate resilience and flood risk management in mid-sized cities.
    """
    print("\n=== Input paragraph ===")
    print(example_text.strip())
    print()

    # 3) Predict with baseline
    print("=== Sentence Baseline Predictions (threshold=0.3) ===")
    baseline_labels, baseline_probs = predict_baseline(
        example_text, encoder, clf, mlb_baseline, threshold=0.3
    )
    if baseline_labels:
        for lbl in baseline_labels:
            idx = list(mlb_baseline.classes_).index(lbl)
            print(f"- {lbl} (prob ~ {baseline_probs[idx]:.3f})")
    else:
        print("(No labels predicted)")

    # 4) Predict with transformer
    print("\n=== Transformer Predictions (threshold=0.3) ===")
    transformer_labels, transformer_probs = predict_transformer(
        example_text, tokenizer, model, mlb_transformer, device, threshold=0.3
    )
    if transformer_labels:
        for lbl in transformer_labels:
            idx = list(mlb_transformer.classes_).index(lbl)
            print(f"- {lbl} (prob ~ {transformer_probs[idx]:.3f})")
    else:
        print("(No labels predicted)")

if __name__ == "__main__":
    main()
