#!/usr/bin/env python3
"""
train_nlp_models.py
====================
Trains text classification models to predict assay-level columns
that cannot be fetched directly from APIs.

Training data:  Final_updated_Dataset_v2025_11-12.xlsx
                (rows with both assay_description and the target column populated)

Output:         nlp_models/  — one .joblib model file per column

Currently trains:
  - assay_kit  (21 unique values, ~2,678 training rows)

Usage:
    python3 train_nlp_models.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
DB_PATH     = SCRIPT_DIR.parent.parent / "Final ODO Dataset_v2026-06-10.xlsx"
MODELS_DIR  = SCRIPT_DIR / "nlp_models"
MODELS_DIR.mkdir(exist_ok=True)

# Columns to train (input_col → target_col)
NLP_TARGETS = {
    "assay_kit": "assay_description",
}

# Minimum samples per class to include that class
MIN_SAMPLES_PER_CLASS = 10


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------

def train_classifier(df: pd.DataFrame, text_col: str, label_col: str) -> tuple:
    """
    Train TF-IDF + Random Forest classifier.
    Returns (pipeline, accuracy, report).
    """
    sub = df.dropna(subset=[text_col, label_col]).copy()
    sub[label_col] = sub[label_col].astype(str).str.strip()

    # Remove rare classes
    counts = sub[label_col].value_counts()
    valid_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    sub = sub[sub[label_col].isin(valid_classes)]

    print(f"  Training on {len(sub)} rows, {len(valid_classes)} classes")
    if len(sub) < 50:
        print(f"  Too few samples, skipping.")
        return None, None, None

    X = sub[text_col].fillna("").tolist()
    y = sub[label_col].tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
        if min(pd.Series(y).value_counts()) >= 2 else None
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, zero_division=0)

    return pipe, acc, report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NLP Model Training")
    print("=" * 60)

    print(f"\nLoading DB from {DB_PATH.name} …")
    db = pd.read_excel(DB_PATH, engine="openpyxl")
    print(f"  {len(db)} rows loaded")

    results = {}

    for label_col, text_col in NLP_TARGETS.items():
        print(f"\n[{label_col}]  (text source: {text_col})")

        pipe, acc, report = train_classifier(db, text_col, label_col)
        if pipe is None:
            continue

        model_path = MODELS_DIR / f"{label_col}.joblib"
        joblib.dump(pipe, model_path)
        print(f"  Accuracy (test): {acc:.4f}")
        print(f"  Saved → {model_path.name}")
        print("\n  Per-class report:")
        for line in report.splitlines():
            print("    " + line)

        results[label_col] = acc

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for col, acc in results.items():
        print(f"  {col:<40}  Accuracy={acc:.4f}")

    print(f"\nAll models saved to: {MODELS_DIR}")
    print("Done.\n")
