# src/train_sentence_baseline.py

from data_utils import prepare_dataset
from sentence_transformers import SentenceTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import joblib
import os
import numpy as np


def filter_dataset(texts, label_lists, keep_labels):
    """Keep only labels in keep_labels; drop samples that become label-less."""
    new_texts = []
    new_labels = []
    for t, labs in zip(texts, label_lists):
        labs_f = [l for l in labs if l in keep_labels]
        if labs_f:  # keep only if at least one label remains
            new_texts.append(t)
            new_labels.append(labs_f)
    return new_texts, new_labels

def proba_to_multi_hot(Y_proba, threshold=0.3, min_labels=1):
    """
    Convert probability matrix (n_samples, n_classes) to multi-hot predictions.
    - Use a probability threshold.
    - Ensure at least `min_labels` labels per sample by forcing top-k if needed.
    """
    Y_pred = (Y_proba >= threshold).astype(int)

    for i in range(Y_pred.shape[0]):
        if Y_pred[i].sum() < min_labels:
            # force the top-k labels by probability
            top_idx = np.argsort(Y_proba[i])[-min_labels:]
            Y_pred[i, top_idx] = 1

    return Y_pred

def main():
    # 1) Load raw data
    train_texts, train_labels = prepare_dataset("data/train.jsonl")
    val_texts,   val_labels   = prepare_dataset("data/val.jsonl")
    test_texts,  test_labels  = prepare_dataset("data/test.jsonl")

    # ------------------------------------------------------------------
    # 2) FIX BASELINE: clean label space (remove super-rare labels)
    # ------------------------------------------------------------------
    # Count label frequencies on TRAIN ONLY
    all_train_labels = [lab for labs in train_labels for lab in labs]
    label_counts = Counter(all_train_labels)

    # you can tweak this; 2 is a reasonable start for such a small dataset
    min_freq = 2
    keep_labels = {lbl for lbl, c in label_counts.items() if c >= min_freq}

    print(f"Total distinct labels in train: {len(label_counts)}")
    print(f"Keeping {len(keep_labels)} labels with freq >= {min_freq}")

    # Filter all splits using the same label set
    train_texts, train_labels = filter_dataset(train_texts, train_labels, keep_labels)
    val_texts,   val_labels   = filter_dataset(val_texts,   val_labels,   keep_labels)
    test_texts,  test_labels  = filter_dataset(test_texts,  test_labels,  keep_labels)

    print(
        f"After filtering: "
        f"Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}"
    )

    # ------------------------------------------------------------------
    # 3) Labels → multi-hot with a consistent MultiLabelBinarizer
    # ------------------------------------------------------------------
    mlb = MultiLabelBinarizer(classes=sorted(list(keep_labels)))
    mlb.fit(train_labels)  # defines class order

    Y_train = mlb.transform(train_labels)
    Y_val   = mlb.transform(val_labels)
    Y_test  = mlb.transform(test_labels)

    print(f"Num labels (after filtering): {len(mlb.classes_)}")

    # ------------------------------------------------------------------
    # 4) Encode texts with sentence transformer
    # ------------------------------------------------------------------
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = encoder.encode(train_texts, show_progress_bar=True)
    X_val   = encoder.encode(val_texts,   show_progress_bar=True)
    X_test  = encoder.encode(test_texts,  show_progress_bar=True)

    # ------------------------------------------------------------------
    # 5) Train classifier
    # ------------------------------------------------------------------
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=200, n_jobs=-1)
    )
    clf.fit(X_train, Y_train)

    # Decision threshold (optional tweak)
    threshold = 0.5

    def eval_split(X, Y_true, split_name):
        Y_proba = clf.predict_proba(X)
        Y_pred = proba_to_multi_hot(Y_proba, threshold=0.3, min_labels=1)

        f1_micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
        print(f"{split_name} F1-micro: {f1_micro:.3f}, F1-macro: {f1_macro:.3f}")
        return Y_pred


    print("\n=== Evaluation ===")
    eval_split(X_train, Y_train, "Train")
    eval_split(X_val,   Y_val,   "Val")
    Y_test_pred = eval_split(X_test,  Y_test,  "Test")

    # Optional: per-label report on test
    print("\nTest classification report:")
    print(classification_report(Y_test, Y_test_pred, target_names=mlb.classes_))

    # ------------------------------------------------------------------
    # 6) Save models for later use in Flask
    # ------------------------------------------------------------------
    out_dir = "models/sentence_baseline"
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(clf, os.path.join(out_dir, "classifier.joblib"))
    joblib.dump(mlb, os.path.join(out_dir, "label_binarizer.joblib"))
    encoder.save(os.path.join(out_dir, "encoder"))

    print(f"\nSaved baseline model to {out_dir}")


if __name__ == "__main__":
    main()
