#!/usr/bin/env python3
"""
train_qikprop_models.py
========================
Trains Random Forest models to predict QikProp physicochemical properties
directly from molecular structure (RDKit descriptors).

Training data:  Final_updated_Dataset_v2025_11-12.xlsx  (DB molecules with
                known QikProp values computed by Schrödinger QikProp)
Output:         qikprop_models/  — one .joblib model file per property

Usage:
    python3 train_qikprop_models.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
DB_PATH     = SCRIPT_DIR.parent.parent / "Final ODO Dataset_v2026-06-10.xlsx"
MODELS_DIR  = SCRIPT_DIR / "qikprop_models"
MODELS_DIR.mkdir(exist_ok=True)

QIKPROP_TARGETS = [
    "qikprop_sasa",
    "qikprop_fisa",
    "qikprop_donor_hb",
    "qikprop_accpt_hb",
    "qikprop_qplog_pw",
    "qikprop_qplog_po/w",
    "qikprop_qplogs",
    "qikprop_qplog_khsa",
    "qikprop_percent_human_oral_absorption",
    # qikprop_dipole excluded — requires quantum chemistry
]

# Descriptor names computed by RDKit (all 2D, no conformer needed)
DESC_NAMES = [name for name, _ in Descriptors.descList]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_descriptors(smiles: str) -> np.ndarray | None:
    """Return a 1-D array of all RDKit 2D descriptors, or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        vals = [func(mol) for _, func in Descriptors.descList]
        return np.array(vals, dtype=float)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset():
    print(f"Loading DB from {DB_PATH.name} …")
    db = pd.read_excel(DB_PATH, engine="openpyxl")

    # Keep only columns we need; deduplicate on SMILES
    cols_needed = ["rdkit_canonical_smiles"] + QIKPROP_TARGETS
    sub = (
        db[cols_needed]
        .dropna(subset=["rdkit_canonical_smiles"])
        .drop_duplicates("rdkit_canonical_smiles")
        .reset_index(drop=True)
    )
    print(f"Unique SMILES with qikprop data: {len(sub)}")

    print("Computing RDKit descriptors (this may take a few minutes) …")
    X_rows, y_rows, valid_idx = [], [], []
    for i, row in sub.iterrows():
        feats = compute_descriptors(row["rdkit_canonical_smiles"])
        if feats is not None:
            X_rows.append(feats)
            y_rows.append(row[QIKPROP_TARGETS].values.astype(float))
            valid_idx.append(i)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(sub)} molecules processed …")

    X = np.array(X_rows)
    Y = np.array(y_rows)
    print(f"Dataset: {X.shape[0]} molecules × {X.shape[1]} descriptors")
    return X, Y, DESC_NAMES, QIKPROP_TARGETS


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(X, Y, target_names):
    results = {}

    for j, tgt in enumerate(target_names):
        y = Y[:, j]

        # Keep only rows where target is not NaN
        mask = ~np.isnan(y)
        Xj, yj = X[mask], y[mask]
        if len(yj) < 50:
            print(f"  [{tgt}] too few samples ({len(yj)}), skipping")
            continue

        print(f"\n  [{tgt}]  n={len(yj)}")

        # Train / test split (80 / 20)
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xj, yj, test_size=0.20, random_state=42
        )

        # Pipeline: impute → scale → Random Forest
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   RandomForestRegressor(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )),
        ])

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        print(f"    Test  R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  (n_test={len(y_te)})")

        # Save model
        model_path = MODELS_DIR / f"{tgt.replace('/', '_')}.joblib"
        joblib.dump(pipe, model_path)
        print(f"    Saved → {model_path.name}")

        results[tgt] = {"r2": r2, "mae": mae, "rmse": rmse, "n": len(yj)}

    return results


# ---------------------------------------------------------------------------
# Compare with linear baseline on same test data
# ---------------------------------------------------------------------------

def compare_with_linear(X, Y, target_names):
    """Quick R² comparison between ML model and old linear calibration."""
    from rdkit.Chem import Descriptors as D

    # Indices of descriptors we need for linear formulas
    desc_map = {n: i for i, n in enumerate(DESC_NAMES)}
    idx_logp = desc_map.get("MolLogP")
    idx_tpsa = desc_map.get("TPSA")
    idx_mw   = desc_map.get("MolWt")
    idx_rb   = desc_map.get("NumRotatableBonds")
    idx_ar   = desc_map.get("NumAromaticRings")
    idx_hbd  = desc_map.get("NumHDonors")
    idx_hba  = desc_map.get("NumHAcceptors")

    print("\n" + "="*70)
    print(f"{'Property':<45} {'Linear R²':>9} {'RF R²':>9}  {'Winner'}")
    print("="*70)

    for j, tgt in enumerate(target_names):
        y = Y[:, j]
        mask = ~np.isnan(y)
        Xj, yj = X[mask], y[mask]
        if len(yj) < 50:
            continue

        _, X_te, _, y_te = train_test_split(Xj, yj, test_size=0.20, random_state=42)

        # Linear baseline predictions
        logp = X_te[:, idx_logp]
        tpsa = X_te[:, idx_tpsa]
        mw   = X_te[:, idx_mw]
        rb   = X_te[:, idx_rb]
        ap   = X_te[:, idx_ar]
        hbd  = X_te[:, idx_hbd]
        hba  = X_te[:, idx_hba]
        esol = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*ap

        linear_map = {
            "qikprop_sasa":                         0.9317 * X_te[:, idx_logp] * 0 + 220.04,  # placeholder
            "qikprop_fisa":                         0.9290 * tpsa + 69.3221,
            "qikprop_donor_hb":                     hbd.astype(float),
            "qikprop_accpt_hb":                     hba.astype(float),
            "qikprop_qplog_pw":                     0.0979*tpsa - 0.2414*logp + 8.204,
            "qikprop_qplog_po/w":                   0.8267*logp - 0.0582,
            "qikprop_qplogs":                       0.6261*esol + 0.3987,
            "qikprop_qplog_khsa":                   0.3970*logp - 1.1860,
            "qikprop_percent_human_oral_absorption": np.clip(109.0 - 0.345*tpsa, 0, 100),
        }

        model_path = MODELS_DIR / f"{tgt.replace('/', '_')}.joblib"
        if not model_path.exists():
            continue

        pipe = joblib.load(model_path)
        rf_pred  = pipe.predict(X_te)
        lin_pred = linear_map.get(tgt)

        rf_r2  = r2_score(y_te, rf_pred)  if lin_pred is not None else float("nan")
        lin_r2 = r2_score(y_te, lin_pred) if lin_pred is not None else float("nan")
        winner = "RF ✓" if rf_r2 > lin_r2 else "Linear"
        print(f"  {tgt:<43} {lin_r2:>9.4f} {rf_r2:>9.4f}  {winner}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  QikProp ML Model Training")
    print("="*60)

    X, Y, desc_names, target_names = build_dataset()

    print("\nTraining Random Forest models …")
    results = train_and_evaluate(X, Y, target_names)

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    for tgt, m in results.items():
        print(f"  {tgt:<45}  R²={m['r2']:.4f}  MAE={m['mae']:.4f}  n={m['n']}")

    compare_with_linear(X, Y, target_names)

    print("\nAll models saved to:", MODELS_DIR)
    print("Done.\n")
