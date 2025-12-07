# src/train_transformer.py

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data_utils import prepare_dataset  # <--- you already have this


# ---------- 1. Simple PyTorch Dataset wrapper ----------

class PolicyDataset(Dataset):
    def __init__(self, texts, labels_multi_hot, tokenizer, max_length=512):
        """
        texts: list[str]
        labels_multi_hot: np.ndarray of shape (N, num_labels) with 0/1
        """
        self.texts = texts
        self.labels = labels_multi_hot.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        # For multi-label classification, labels should be float
        item["labels"] = torch.tensor(labels, dtype=torch.float32)
        return item


# ---------- 2. Main training function ----------

def main():
    # --- Load datasets from JSONL ---
    train_texts, train_labels = prepare_dataset("data/train.jsonl")
    val_texts,   val_labels   = prepare_dataset("data/val.jsonl")
    test_texts,  test_labels  = prepare_dataset("data/test.jsonl")

    # --- Same label frequency filtering as baseline ---
    # Count label frequencies in TRAIN only
    label_counter = Counter()
    for labs in train_labels:
        label_counter.update(labs)

    print(f"Total distinct labels in train: {len(label_counter)}")

    # Keep only labels that appear at least twice in train
    min_freq = 2
    kept_labels = {label for label, count in label_counter.items() if count >= min_freq}
    print(f"Keeping {len(kept_labels)} labels with freq >= {min_freq}")

    def filter_split(texts, labels):
        new_texts = []
        new_labels = []
        for t, labs in zip(texts, labels):
            filtered = [l for l in labs if l in kept_labels]
            if filtered:
                new_texts.append(t)
                new_labels.append(filtered)
        return new_texts, new_labels

    train_texts, train_labels = filter_split(train_texts, train_labels)
    val_texts,   val_labels   = filter_split(val_texts, val_labels)
    test_texts,  test_labels  = filter_split(test_texts, test_labels)

    print(f"After filtering: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # --- MultiLabelBinarizer with fixed class order (same as baseline idea) ---
    mlb = MultiLabelBinarizer(classes=sorted(kept_labels))
    mlb.fit(train_labels)

    Y_train = mlb.transform(train_labels)
    Y_val   = mlb.transform(val_labels)
    Y_test  = mlb.transform(test_labels)

    num_labels = Y_train.shape[1]
    print(f"Num labels (after filtering): {num_labels}")

    # ---------- 3. Init tokenizer / model ----------
    model_name = "bert-base-uncased"  # you can change to legal-bert etc. later if you want
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    # Tell HF this is multi-label (BCEWithLogits)
    model.config.problem_type = "multi_label_classification"

    # ---------- 4. Build Datasets ----------
    train_dataset = PolicyDataset(train_texts, Y_train, tokenizer)
    val_dataset   = PolicyDataset(val_texts,   Y_val,   tokenizer)
    test_dataset  = PolicyDataset(test_texts,  Y_test,  tokenizer)

    # ---------- 5. Define metrics ----------
    def compute_metrics(p):
        """
        p.predictions: (N, num_labels)
        p.label_ids:  (N, num_labels)
        """
        logits = p.predictions
        labels = p.label_ids

        # Apply sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-logits))
        # Threshold at 0.5
        preds = (probs >= 0.5).astype(int)

        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
        }

    # ---------- 6. TrainingArguments ----------
    training_args = TrainingArguments(
        output_dir="models/transformer_checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        num_train_epochs=8,              # small dataset, a few epochs is fine
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        remove_unused_columns=False,     # IMPORTANT for custom Dataset
    )

    # ---------- 7. Trainer ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # ---------- 8. Train ----------
    trainer.train()

    # ---------- 9. Final evaluation on all splits ----------
    print("\n=== Final Evaluation ===")
    train_metrics = trainer.evaluate(train_dataset)
    print(f"Train metrics: {train_metrics}")

    val_metrics = trainer.evaluate(val_dataset)
    print(f"Val metrics:   {val_metrics}")

    test_metrics = trainer.evaluate(test_dataset)
    print(f"Test metrics:  {test_metrics}")

    # ---------- 10. Save model + tokenizer + label binarizer ----------
    out_dir = "models/transformer"
    os.makedirs(out_dir, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    import joblib
    joblib.dump(mlb, os.path.join(out_dir, "label_binarizer.joblib"))

    print(f"\nSaved transformer model + tokenizer + label binarizer to {out_dir}")


if __name__ == "__main__":
    main()
