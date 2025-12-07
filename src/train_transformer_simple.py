# src/train_transformer_simple.py

import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW      # <--- FIXED HERE

from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from data_utils import prepare_dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels_multi_hot):
        self.texts = texts
        self.labels = labels_multi_hot

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch, tokenizer, max_length=256):
    texts = [b[0] for b in batch]
    labels = np.stack([b[1] for b in batch])  # (batch, num_labels)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(labels, dtype=torch.float32)

    enc["labels"] = labels
    return enc


def compute_pos_weight(Y_train):
    """
    Compute pos_weight per label for BCEWithLogitsLoss to handle imbalance:
    pos_weight_j = (#neg / #pos) for each label j
    """
    # Y_train: (N, L) numpy array
    pos_counts = Y_train.sum(axis=0)        # (L,)
    neg_counts = Y_train.shape[0] - pos_counts
    # avoid division by zero
    pos_weight = np.ones_like(pos_counts, dtype=np.float32)
    for j in range(len(pos_counts)):
        if pos_counts[j] > 0:
            pos_weight[j] = neg_counts[j] / pos_counts[j]
        else:
            pos_weight[j] = 1.0
    return pos_weight


def main():
    # 1) Load raw data
    train_texts, train_labels = prepare_dataset("data/train.jsonl")
    val_texts,   val_labels   = prepare_dataset("data/val.jsonl")
    test_texts,  test_labels  = prepare_dataset("data/test.jsonl")

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # 2) Fit label binarizer on TRAIN labels only
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)
    Y_train = mlb.transform(train_labels)
    Y_val   = mlb.transform(val_labels)
    Y_test  = mlb.transform(test_labels)

    num_labels = Y_train.shape[1]
    print(f"Num labels (after binarization): {num_labels}")

    # 3) Compute pos_weight for BCE
    pos_weight_np = compute_pos_weight(Y_train)
    print("Sample of pos_weight:", pos_weight_np[:10])
    # convert to tensor later on correct device

    # 4) Build datasets + dataloaders
    train_dataset = TextDataset(train_texts, Y_train)
    val_dataset   = TextDataset(val_texts,   Y_val)
    test_dataset  = TextDataset(test_texts,  Y_test)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def make_loader(dataset, batch_size, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )

    train_loader = make_loader(train_dataset, batch_size=4, shuffle=True)
    val_loader   = make_loader(val_dataset,   batch_size=4, shuffle=False)
    test_loader  = make_loader(test_dataset,  batch_size=4, shuffle=False)

    # 5) Model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 6) Loss + optimizer
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs = 5           # a bit more epochs
    best_val_f1 = 0.0
    best_state_dict = None

    # Global threshold for turning probabilities -> labels
    THRESHOLD = 0.3

    def eval_split(dataloader, split_name):
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop("labels").to(device)
                outputs = model(**{k: v.to(device) for k, v in batch.items()})
                logits = outputs.logits
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
        Y_pred = (probs >= THRESHOLD).astype(int)

        f1_micro = f1_score(all_labels, Y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(all_labels, Y_pred, average="macro", zero_division=0)
        print(f"{split_name} F1-micro: {f1_micro:.3f}, F1-macro: {f1_macro:.3f}")
        return f1_micro, f1_macro, all_labels, Y_pred

    # 7) Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} train loss: {avg_loss:.4f}")

        # Evaluate on val
        val_f1_micro, val_f1_macro, _, _ = eval_split(val_loader, "Val")

        # Early stopping-style: keep best epoch
        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            best_state_dict = model.state_dict()
            print(f"New best val micro-F1: {best_val_f1:.3f} (saving state)")

    # 8) Load best model (if we found any improvement)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\nLoaded best model with Val F1-micro = {best_val_f1:.3f}")
    else:
        print("\nWarning: no improvement on Val F1; using last epoch model.")

    # 9) Final evaluation on train/val/test with best model
    print("\n=== Final Evaluation (best epoch) ===")
    train_f1_micro, train_f1_macro, train_y_true, train_y_pred = eval_split(train_loader, "Train")
    val_f1_micro, val_f1_macro, val_y_true, val_y_pred         = eval_split(val_loader,   "Val")
    test_f1_micro, test_f1_macro, test_y_true, test_y_pred     = eval_split(test_loader,  "Test")

    print("\nTest classification report:")
    print(classification_report(
        test_y_true,
        test_y_pred,
        target_names=mlb.classes_,
        zero_division=0
    ))

    # 10) Save model + tokenizer + label binarizer
    out_dir = "models/transformer_finetuned"
    os.makedirs(out_dir, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Save label binarizer
    import joblib
    joblib.dump(mlb, os.path.join(out_dir, "label_binarizer.joblib"))

    print(f"\nSaved fine-tuned transformer model to {out_dir}")
    print(f"Best Val F1-micro: {best_val_f1:.3f} with threshold={THRESHOLD}")

if __name__ == "__main__":
    main()
