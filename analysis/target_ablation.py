"""
Task 5 vehicle: fingerprint-only RandomForest, full-target vs
opioid-receptor-only, on the bipartite pipeline's temporal split
(gnn/processed_temporal_2048fp.pt — the 43-target graph).

Reuses compute_metrics / RF_PARAMS / SEEDS from baselines/run_baselines.py
unmodified (imported, not reimplemented) so the methodology is identical to
the already-reported full-target number (bipartite fingerprint_rf,
R2 = 0.1797 +- 0.0023, from baselines/baseline_results.csv).

"Opioid-receptor rows only" = the 27 target_keys analysis/target_inventory.py
classifies as MOR/DOR/KOR/NOP (any species, including chembl-id and
name-fallback duplicate keys for the same receptor) — i.e. every row on-target
for one of the four canonical opioid receptor types. Excluded: the 12
ambiguous/multi-subtype opioid groupings (e.g. "Opioid receptors; mu &
delta") and the 4 non-opioid off-targets (2 dopamine, 2 neurotensin
receptors). See target_inventory.py's OPIOID_RECEPTOR_PATTERNS for the exact
classification rule and its known limitation (a couple of mixed-pharmacology
names, e.g. "mu/kappa opioid receptor", get bucketed by whichever pattern
matches first — flagged in target_inventory.csv, not hidden).

No GNN training. No tuning: same RF_PARAMS/SEEDS as run_baselines.py, no
knobs changed here.

Run:
    conda run -n odo python3 analysis/target_ablation.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baselines.run_baselines import (  # noqa: E402
    BIPARTITE_GRAPH_PATH, N_NON_FP_COMPOUND_COLS, RF_PARAMS, SEEDS,
    compute_metrics,
)
from analysis.target_inventory import (  # noqa: E402
    EXCEL_PATH, classify_receptor, safe, slugify, target_key as _target_key,
)

EXACT_QUALIFIER_MATCH_NOTE = (
    "receptor membership is defined at the TARGET level (chembl_target_id / "
    "name-fallback key), not per-row — every activity row for a receptor-"
    "classified target is included, exact or censored, consistent with how "
    "gnn/dataset.py's train/test masks are built downstream of this graph."
)


def build_target_key_order() -> list[str]:
    """Reproduce gnn/preprocess_bipartite_graph.py:build_target_table's
    target_key list in its exact first-appearance order (deterministic,
    same row order as pd.read_excel), so it lines up with the integer
    target-node indices already baked into processed_temporal_2048fp.pt."""
    raw = pd.read_excel(EXCEL_PATH, sheet_name="Full Dataset")
    keys = raw.apply(_target_key, axis=1)
    ordered = pd.unique(keys.dropna())
    return list(ordered)


def receptor_target_keys() -> set[str]:
    raw = pd.read_excel(EXCEL_PATH, sheet_name="Full Dataset")
    raw["target_key"] = raw.apply(_target_key, axis=1)
    raw = raw[raw["target_key"].notna()]
    names = raw.groupby("target_key")["target_name"].apply(
        lambda s: s.dropna().mode().iloc[0] if s.dropna().size else "Unknown"
    )
    return {k for k, name in names.items() if classify_receptor(name) is not None}


def load_split():
    from gnn.dataset import build_dataset
    data, _ = build_dataset(verbose=False, graph_path=BIPARTITE_GRAPH_PATH)
    et = data[("compound", "activity", "target")]
    compound_x = data["compound"].x.numpy()
    n_targets = data["target"].num_nodes

    train_mask = (et.train_mask | et.val_mask).numpy()
    test_mask = et.test_mask.numpy()
    edge_index = et.edge_index.numpy()
    edge_label = et.edge_label.numpy()

    fp_width = compound_x.shape[1] - N_NON_FP_COMPOUND_COLS
    fp = compound_x[:, :fp_width]
    compound_idx_all = edge_index[0]
    target_idx_all = edge_index[1]

    return {
        "n_targets": n_targets,
        "fp": fp,
        "compound_idx": compound_idx_all,
        "target_idx": target_idx_all,
        "y": edge_label,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


def build_X(fp, compound_idx, target_idx, n_targets):
    oh = np.zeros((len(target_idx), n_targets), dtype=np.float32)
    oh[np.arange(len(target_idx)), target_idx] = 1.0
    return np.concatenate([fp[compound_idx].astype(np.float32), oh], axis=1)


def run_rf(split, row_mask, label):
    from sklearn.ensemble import RandomForestRegressor

    train_idx = np.where(split["train_mask"] & row_mask)[0]
    test_idx = np.where(split["test_mask"] & row_mask)[0]

    X_train = build_X(split["fp"], split["compound_idx"][train_idx],
                       split["target_idx"][train_idx], split["n_targets"])
    y_train = split["y"][train_idx]
    X_test = build_X(split["fp"], split["compound_idx"][test_idx],
                      split["target_idx"][test_idx], split["n_targets"])
    y_test = split["y"][test_idx]

    print(f"\n=== {label}: train n={len(train_idx)}, test n={len(test_idx)} ===")
    per_seed = []
    for seed in SEEDS:
        model = RandomForestRegressor(random_state=seed, **RF_PARAMS)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = compute_metrics(y_test, pred)
        per_seed.append(m)
        print(f"  seed={seed}: RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
              f"r={m['pearson_r']:.4f} R2={m['r2']:.4f}")

    agg = {}
    for key in ["rmse", "mae", "pearson_r", "r2"]:
        vals = np.array([m[key] for m in per_seed])
        agg[f"{key}_mean"] = float(vals.mean())
        agg[f"{key}_sd"] = float(vals.std(ddof=1))
    print(f"  mean+-sd(n={len(SEEDS)},ddof=1): "
          f"RMSE={agg['rmse_mean']:.4f}+-{agg['rmse_sd']:.4f} "
          f"R2={agg['r2_mean']:.4f}+-{agg['r2_sd']:.4f}")
    return agg, len(train_idx), len(test_idx)


def main():
    print(EXACT_QUALIFIER_MATCH_NOTE)
    target_order = build_target_key_order()
    rec_keys = receptor_target_keys()
    rec_idx_set = {i for i, k in enumerate(target_order) if k in rec_keys}
    print(f"\n{len(rec_keys)} receptor target_keys -> {len(rec_idx_set)} graph indices "
          f"(out of {len(target_order)} total)")

    split = load_split()
    assert split["n_targets"] == len(target_order), (
        f"target count mismatch: graph has {split['n_targets']}, "
        f"pandas reconstruction has {len(target_order)} — key order assumption failed, "
        f"do not trust the ablation below"
    )
    # Sanity-check the reconstructed order against a known row count (CHEMBL233 / MOR, human).
    chembl233_idx = target_order.index("CHEMBL233")
    n_chembl233 = int((split["target_idx"] == chembl233_idx).sum())
    print(f"Sanity check: CHEMBL233 (human MOR) graph index {chembl233_idx} "
          f"has {n_chembl233} edges (target_inventory.py reported 7339)")
    assert n_chembl233 == 7339, "target_key ordering does not match the graph — aborting"

    all_mask = np.ones(len(split["y"]), dtype=bool)
    receptor_mask = np.isin(split["target_idx"], list(rec_idx_set))

    full_agg, full_ntr, full_nte = run_rf(split, all_mask, "Full target set (43 targets)")
    rec_agg, rec_ntr, rec_nte = run_rf(split, receptor_mask,
                                        f"Opioid-receptor-only ({len(rec_idx_set)} targets)")

    print("\n" + "=" * 70)
    print(f"{'Config':<35}{'n_train':>10}{'n_test':>10}{'R2 mean':>10}{'R2 sd':>10}")
    print("=" * 70)
    print(f"{'Full (43 targets)':<35}{full_ntr:>10}{full_nte:>10}"
          f"{full_agg['r2_mean']:>10.4f}{full_agg['r2_sd']:>10.4f}")
    print(f"{'Receptor-only (' + str(len(rec_idx_set)) + ' targets)':<35}{rec_ntr:>10}{rec_nte:>10}"
          f"{rec_agg['r2_mean']:>10.4f}{rec_agg['r2_sd']:>10.4f}")
    print("=" * 70)

    diff = rec_agg["r2_mean"] - full_agg["r2_mean"]
    pooled_sd = max(full_agg["r2_sd"], rec_agg["r2_sd"])
    verdict = (
        "helps (receptor-only R2 clearly above full-set R2, beyond seed noise)"
        if diff > 2 * pooled_sd else
        "hurts (receptor-only R2 clearly below full-set R2, beyond seed noise)"
        if diff < -2 * pooled_sd else
        "is within noise (difference is smaller than ~2x the seed-to-seed sd)"
    )
    print(f"\nR2 difference (receptor-only - full) = {diff:+.4f}; "
          f"largest per-config sd = {pooled_sd:.4f} -> {verdict}")
    print(
        "Caveat: this compares R2 on two DIFFERENT test populations (all test "
        "rows vs. the receptor-only subset of test rows), not the same rows "
        "scored by two models — a change here reflects some mix of (a) "
        "training-signal dilution from off-target rows and (b) the "
        "receptor-only test subset simply being an easier/harder population "
        "on its own. Both numbers use the same temporal cutoff and the same "
        "fixed RF config; nothing was tuned to produce either number."
    )


if __name__ == "__main__":
    main()
