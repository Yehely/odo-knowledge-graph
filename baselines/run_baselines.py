"""
Baseline models for the ODO GNN's temporal-split test set — global mean,
per-target mean, and a fingerprint-only RandomForest (ECFP4 + target one-hot,
no graph structure, no message passing).

Two GNN pipelines exist in this repo, on two DIFFERENT temporal splits:

  "hetero"    hetero_gnn/processed_hetero_graph.pt
              train: doc_year < 2015 | test: doc_year >= 2015, no upper cap.
              This is the split described in the baseline request, and it is
              what produced the "0.26 heterogeneous" reference number.

  "bipartite" gnn/processed_temporal_2048fp.pt
              gnn/preprocess_bipartite_graph.py's own "temporal" split is
              actually train <= 2015 / test 2016-2020 (capped) — its CLI
              help text says so outright. This does NOT match "train before
              2015, test 2015 and later": years 2021+ are excluded from test
              and fall back into train. It is included anyway, clearly
              labeled, because the "0.22 bipartite ensemble" reference number
              was produced under THIS split, not a 2015-onward one. Do not
              read the bipartite numbers below as being on the same split as
              the hetero numbers.

Both splits' train/test membership and target-variable (pChEMBL) values are
loaded via the GNN codebase's own build_dataset() functions
(gnn.dataset.build_dataset / hetero_gnn.dataset.build_dataset) — the exact
same saved split tensors and mask-combination logic the GNNs themselves use
for their reported metrics. Nothing about the split is re-derived here.

Fingerprints are NOT regenerated with RDKit. Both pipelines store the ECFP4
bits as the leading columns of data["compound"].x inside their saved graph
file (Morgan bits first, then 12 numeric descriptor columns — verified by
reading gnn/features.py and hetero_gnn/preprocess.py). This script slices
those columns back out, so results are bound to whatever fingerprints the
GNN itself was trained on.

Prerequisites (run once, in the odo conda env — NOT done by this script):
    conda run -n odo python3 hetero_gnn/run_preprocess.py
        (only if hetero_gnn/processed_hetero_graph.pt does not already exist)
    conda run -n odo python3 gnn/preprocess_bipartite_graph.py --split temporal --fp-bits 2048
        (needed for the bipartite section; the file is not checked into git)

Run:
    conda run -n odo python3 baselines/run_baselines.py
"""
import importlib.metadata
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BIPARTITE_GRAPH_PATH = os.path.join(ROOT, "gnn", "processed_temporal_2048fp.pt")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SEEDS = [42, 123, 456, 789, 1024]  # reused from gnn/train_ensemble.py's SEEDS
RF_PARAMS = {"n_estimators": 500, "n_jobs": 4}  # random_state set per seed below
N_NON_FP_COMPOUND_COLS = 12  # ADMET/QikProp numeric columns appended after the
                              # Morgan bits in both pipelines (verified: 10
                              # ADMET cols + max_phase + is_radiolabeled in
                              # gnn/features.py; 12-column COMPOUND_NUMERIC_COLS
                              # in hetero_gnn/config.py)

# Reference numbers: previously reported, NOT re-run here.
GNN_REFERENCE = {
    "bipartite_ensemble": {
        "source": "gnn/ensemble_test_results.txt (on-disk, verified)",
        "split": "bipartite (train<=2015 / test 2016-2020, capped)",
        "rmse": 1.2328, "mae": 0.9765, "pearson_r": 0.5277, "r2": 0.2159,
    },
    "heterogeneous": {
        "source": "hetero_gnn/test_results.txt (on-disk, verified)",
        "split": "hetero (train<2015 / test>=2015, no cap)",
        "rmse": 1.2312, "mae": 0.9755, "pearson_r": 0.5396, "r2": 0.2580,
    },
    "attention": {
        "source": "reports/generate_final_report.py hardcoded literal — "
                   "UNVERIFIED, no surviving results file for this run "
                   "(see results/seed_variance/summary.md)",
        "split": "hetero (train<2015 / test>=2015, no cap), presumed — same "
                 "code path as 'heterogeneous' row with POOLING_MODE=attention",
        "rmse": 1.1889, "mae": 0.9150, "pearson_r": 0.5910, "r2": 0.3081,
    },
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score

    rmse = math.sqrt(float(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    if len(y_true) > 2:
        r, _ = pearsonr(y_pred, y_true)
        r2 = r2_score(y_true, y_pred)
    else:
        r, r2 = float("nan"), float("nan")
    return {"rmse": rmse, "mae": mae, "pearson_r": float(r), "r2": float(r2)}


def _extract(compound_x: np.ndarray, n_targets: int, edge_index: np.ndarray,
             edge_label: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray,
             meta: dict, name: str) -> dict:
    fp_width = compound_x.shape[1] - N_NON_FP_COMPOUND_COLS
    if fp_width != 2048:
        print(
            f"  [WARNING] {name}: detected fingerprint width {fp_width} "
            f"(compound_x has {compound_x.shape[1]} columns, assuming "
            f"{N_NON_FP_COMPOUND_COLS} trailing non-fingerprint columns), "
            f"not the expected 2048. Check MORGAN_BITS in gnn/config.py / "
            f"hetero_gnn/config.py and how the graph file was built."
        )
    fp = compound_x[:, :fp_width]
    compound_idx_all = edge_index[0]
    target_idx_all = edge_index[1]

    def gather(mask: np.ndarray) -> dict:
        idx = np.where(mask)[0]
        return {
            "fp": fp[compound_idx_all[idx]],
            "target_idx": target_idx_all[idx],
            "y": edge_label[idx],
            "idx": idx,  # row indices into this split's full edge arrays (edge_index/edge_label)
        }

    return {
        "name": name,
        "n_targets": n_targets,
        "fp_width": fp_width,
        "train": gather(train_mask),
        "test": gather(test_mask),
        "meta": meta,
    }


def load_hetero_split() -> dict | None:
    try:
        from hetero_gnn.dataset import build_dataset
    except Exception as e:
        print(f"  [SKIP] hetero split: could not import hetero_gnn.dataset ({e})")
        return None
    try:
        data, meta = build_dataset(verbose=False)
    except FileNotFoundError as e:
        print(f"  [SKIP] hetero split: {e}")
        return None

    et = data[("compound", "binds_to", "target")]
    compound_x = data["compound"].x.numpy()
    n_targets = data["target"].num_nodes

    train_mask = (et.train_exact_mask | et.val_exact_mask).numpy()
    test_mask = et.test_exact_mask.numpy()

    return _extract(
        compound_x, n_targets, et.edge_index.numpy(), et.edge_label.numpy(),
        train_mask, test_mask, meta, name="hetero",
    )


def load_bipartite_split() -> dict | None:
    try:
        from gnn.dataset import build_dataset
    except Exception as e:
        print(f"  [SKIP] bipartite split: could not import gnn.dataset ({e})")
        return None
    try:
        data, meta = build_dataset(verbose=False, graph_path=BIPARTITE_GRAPH_PATH)
    except FileNotFoundError as e:
        print(f"  [SKIP] bipartite split: {e}")
        print(
            "    Generate it first with:\n"
            "    conda run -n odo python3 gnn/preprocess_bipartite_graph.py "
            "--split temporal --fp-bits 2048"
        )
        return None

    et = data[("compound", "activity", "target")]
    compound_x = data["compound"].x.numpy()
    n_targets = data["target"].num_nodes

    train_mask = (et.train_mask | et.val_mask).numpy()
    test_mask = et.test_mask.numpy()

    return _extract(
        compound_x, n_targets, et.edge_index.numpy(), et.edge_label.numpy(),
        train_mask, test_mask, meta, name="bipartite",
    )


def global_mean_baseline(split: dict) -> tuple[dict, dict]:
    y_train = split["train"]["y"]
    y_test = split["test"]["y"]
    pred = np.full_like(y_test, fill_value=y_train.mean())
    return compute_metrics(y_test, pred), {"n_test": len(y_test)}


def per_target_mean_baseline(split: dict) -> tuple[dict, dict]:
    y_train = split["train"]["y"]
    t_train = split["train"]["target_idx"]
    y_test = split["test"]["y"]
    t_test = split["test"]["target_idx"]

    global_mean = float(y_train.mean())
    target_means = {
        int(t): float(y_train[t_train == t].mean()) for t in np.unique(t_train)
    }

    pred = np.empty_like(y_test)
    n_fallback = 0
    for i, t in enumerate(t_test):
        m = target_means.get(int(t))
        if m is None:
            pred[i] = global_mean
            n_fallback += 1
        else:
            pred[i] = m

    return compute_metrics(y_test, pred), {
        "n_test": len(y_test),
        "n_fallback_to_global": n_fallback,
        "n_targets_seen_in_train": len(target_means),
    }


def fingerprint_rf_baseline(split: dict, seeds: list[int]) -> tuple[dict, dict]:
    from sklearn.ensemble import RandomForestRegressor

    n_targets = split["n_targets"]

    def build_X(part: dict) -> np.ndarray:
        oh = np.zeros((len(part["target_idx"]), n_targets), dtype=np.float32)
        oh[np.arange(len(part["target_idx"])), part["target_idx"]] = 1.0
        return np.concatenate([part["fp"].astype(np.float32), oh], axis=1)

    X_train = build_X(split["train"])
    y_train = split["train"]["y"]
    X_test = build_X(split["test"])
    y_test = split["test"]["y"]

    n_unseen_target_test = len(
        set(split["test"]["target_idx"].tolist()) - set(split["train"]["target_idx"].tolist())
    )

    per_seed = []
    for seed in seeds:
        model = RandomForestRegressor(random_state=seed, **RF_PARAMS)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = compute_metrics(y_test, pred)
        m["seed"] = seed
        per_seed.append(m)
        print(f"    seed={seed}: RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
              f"r={m['pearson_r']:.4f} R2={m['r2']:.4f}")

        pred_path = os.path.join(OUT_DIR, f"predictions_{split['name']}_seed{seed}.npz")
        np.savez(
            pred_path,
            pred=pred,
            y_true=y_test,
            row_idx=split["test"]["idx"],  # indices into this split's full edge arrays
        )

    agg = {}
    for key in ["rmse", "mae", "pearson_r", "r2"]:
        vals = np.array([m[key] for m in per_seed])
        agg[f"{key}_mean"] = float(vals.mean())
        agg[f"{key}_sd"] = float(vals.std(ddof=1))

    return agg, {
        "n_test": len(y_test),
        "seeds": seeds,
        "rf_params": {**RF_PARAMS, "random_state": "per-seed, see seeds"},
        "n_targets_unseen_in_train": n_unseen_target_test,
        "per_seed": per_seed,
    }


def run_split(split: dict, rows: list, seeds: list[int]):
    name = split["name"]
    print(f"\n=== {name} split — train n={len(split['train']['y'])}, "
          f"test n={len(split['test']['y'])}, fp_width={split['fp_width']} ===")

    m, info = global_mean_baseline(split)
    print(f"  global_mean      : RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
          f"r={m['pearson_r']:.4f} R2={m['r2']:.4f}")
    rows.append({"split": name, "baseline": "global_mean", **m,
                 "rmse_sd": "", "mae_sd": "", "pearson_r_sd": "", "r2_sd": "",
                 "n_test": info["n_test"], "notes": ""})

    m, info = per_target_mean_baseline(split)
    print(f"  per_target_mean  : RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
          f"r={m['pearson_r']:.4f} R2={m['r2']:.4f} "
          f"(fallback-to-global: {info['n_fallback_to_global']}/{info['n_test']} "
          f"test rows, target unseen in train)")
    rows.append({"split": name, "baseline": "per_target_mean", **m,
                 "rmse_sd": "", "mae_sd": "", "pearson_r_sd": "", "r2_sd": "",
                 "n_test": info["n_test"],
                 "notes": f"fallback_to_global_mean={info['n_fallback_to_global']} rows "
                          f"(target unseen in the {info['n_targets_seen_in_train']}-target "
                          f"training set)"})

    print(f"  fingerprint_rf   : {len(seeds)} seeds {seeds}")
    agg, info = fingerprint_rf_baseline(split, seeds)
    print(f"    mean+-sd(n={len(seeds)},ddof=1): "
          f"RMSE={agg['rmse_mean']:.4f}+-{agg['rmse_sd']:.4f} "
          f"MAE={agg['mae_mean']:.4f}+-{agg['mae_sd']:.4f} "
          f"r={agg['pearson_r_mean']:.4f}+-{agg['pearson_r_sd']:.4f} "
          f"R2={agg['r2_mean']:.4f}+-{agg['r2_sd']:.4f}")
    rows.append({
        "split": name, "baseline": "fingerprint_rf",
        "rmse": agg["rmse_mean"], "mae": agg["mae_mean"],
        "pearson_r": agg["pearson_r_mean"], "r2": agg["r2_mean"],
        "rmse_sd": agg["rmse_sd"], "mae_sd": agg["mae_sd"],
        "pearson_r_sd": agg["pearson_r_sd"], "r2_sd": agg["r2_sd"],
        "n_test": info["n_test"],
        "notes": f"RandomForestRegressor({RF_PARAMS}, random_state=seed), "
                 f"seeds={seeds}, ECFP4({split['fp_width']}-bit)+target-one-hot "
                 f"({split['n_targets']} targets), "
                 f"{info['n_targets_unseen_in_train']} test targets unseen in train",
    })


def write_csv(rows: list, path: str):
    import csv
    fieldnames = ["split", "baseline", "rmse", "mae", "pearson_r", "r2",
                  "rmse_sd", "mae_sd", "pearson_r_sd", "r2_sd", "n_test", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def write_summary(rows: list, splits_run: list[str], path: str):
    lines = []
    lines.append("# Baseline results — temporal split(s)\n")
    lines.append(
        "Two different temporal splits are in play; they are **not** the same "
        "split, and their numbers are not directly comparable to each other. "
        "See `baselines/run_baselines.py` module docstring for the full "
        "explanation of why.\n"
    )

    baseline_labels = {
        "global_mean": "Global mean",
        "per_target_mean": "Per-target mean",
        "fingerprint_rf": "Fingerprint-only RF (ECFP4 + target one-hot)",
    }

    if "hetero" in splits_run:
        lines.append(
            "\n## Hetero split (train: year < 2015, test: year >= 2015, no cap)\n"
        )
        lines.append(
            "This is the split described in the baseline request, and the split "
            "the heterogeneous-GNN reference rows below were evaluated on.\n"
        )
        lines.append("| Baseline | RMSE | MAE | Pearson r | R² | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            if r["split"] != "hetero":
                continue
            sd_note = ""
            if r["rmse_sd"] != "":
                sd_note = f" (±{_fmt(r['rmse_sd'])} RMSE sd, ±{_fmt(r['r2_sd'])} R² sd across seeds)"
            lines.append(
                f"| {baseline_labels[r['baseline']]} | {_fmt(r['rmse'])} | "
                f"{_fmt(r['mae'])} | {_fmt(r['pearson_r'])} | {_fmt(r['r2'])} | "
                f"n_test={r['n_test']}{sd_note}. {r['notes']} |"
            )
        ref = GNN_REFERENCE["heterogeneous"]
        lines.append(
            f"| Heterogeneous GNN (previously reported, not re-run here) | "
            f"{_fmt(ref['rmse'])} | {_fmt(ref['mae'])} | {_fmt(ref['pearson_r'])} | "
            f"{_fmt(ref['r2'])} | source: {ref['source']} |"
        )
        ref = GNN_REFERENCE["attention"]
        lines.append(
            f"| \"Attention\" GNN variant (previously reported, not re-run here, "
            f"**unverified** — see below) | {_fmt(ref['rmse'])} | {_fmt(ref['mae'])} | "
            f"{_fmt(ref['pearson_r'])} | {_fmt(ref['r2'])} | source: {ref['source']} |"
        )

    if "bipartite" in splits_run:
        lines.append(
            "\n## Bipartite split (train: year <= 2015, test: year 2016-2020, "
            "capped — NOT the split described in the baseline request)\n"
        )
        lines.append(
            "Included only because the 0.22 bipartite-ensemble reference number "
            "was produced on this split. Do not compare these rows to the hetero "
            "table above as if they were on the same test set.\n"
        )
        lines.append("| Baseline | RMSE | MAE | Pearson r | R² | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            if r["split"] != "bipartite":
                continue
            sd_note = ""
            if r["rmse_sd"] != "":
                sd_note = f" (±{_fmt(r['rmse_sd'])} RMSE sd, ±{_fmt(r['r2_sd'])} R² sd across seeds)"
            lines.append(
                f"| {baseline_labels[r['baseline']]} | {_fmt(r['rmse'])} | "
                f"{_fmt(r['mae'])} | {_fmt(r['pearson_r'])} | {_fmt(r['r2'])} | "
                f"n_test={r['n_test']}{sd_note}. {r['notes']} |"
            )
        ref = GNN_REFERENCE["bipartite_ensemble"]
        lines.append(
            f"| Bipartite GNN ensemble (previously reported, not re-run here) | "
            f"{_fmt(ref['rmse'])} | {_fmt(ref['mae'])} | {_fmt(ref['pearson_r'])} | "
            f"{_fmt(ref['r2'])} | source: {ref['source']} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_run_log(path: str, start_time: float, splits_run: list[str]):
    pkgs = {}
    for pkg in ["torch", "scikit-learn", "numpy", "pandas", "scipy", "torch-geometric", "rdkit"]:
        try:
            pkgs[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkgs[pkg] = "NOT INSTALLED in this environment"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "UNKNOWN (git rev-parse failed)"

    runtime_s = time.time() - start_time

    lines = [
        "ODO baselines — run log",
        f"Commit           : {commit}",
        f"Splits run       : {splits_run}",
        f"Seeds (RF)       : {SEEDS}",
        f"Runtime          : {runtime_s:.1f}s",
        "Package versions :",
    ]
    for pkg, ver in pkgs.items():
        lines.append(f"  {pkg:15s} {ver}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    start_time = time.time()
    rows: list = []
    splits_run: list[str] = []

    hetero = load_hetero_split()
    if hetero is not None:
        run_split(hetero, rows, SEEDS)
        splits_run.append("hetero")

    bipartite = load_bipartite_split()
    if bipartite is not None:
        run_split(bipartite, rows, SEEDS)
        splits_run.append("bipartite")

    if not rows:
        print("\nNo splits could be loaded — nothing to write. See [SKIP] messages above.")
        return

    csv_path = os.path.join(OUT_DIR, "baseline_results.csv")
    write_csv(rows, csv_path)
    print(f"\nWrote {csv_path}")

    summary_path = os.path.join(OUT_DIR, "summary.md")
    write_summary(rows, splits_run, summary_path)
    print(f"Wrote {summary_path}")

    log_path = os.path.join(OUT_DIR, "run_log.txt")
    write_run_log(log_path, start_time, splits_run)
    print(f"Wrote {log_path}")

    # Task 4 — read the result honestly.
    for r in rows:
        if r["baseline"] == "fingerprint_rf" and r["split"] == "hetero":
            fp_r2 = r["r2"]
            ref_r2 = GNN_REFERENCE["heterogeneous"]["r2"]
            ref_att_r2 = GNN_REFERENCE["attention"]["r2"]
            print(
                f"\nTask 4 — fingerprint-only RF R² = {fp_r2:.2f} vs. "
                f"heterogeneous GNN R² = {ref_r2:.2f} "
                f"(and unverified attention-variant R² = {ref_att_r2:.2f}) "
                f"on the SAME hetero test set. "
                + ("The fingerprint-only baseline is AT OR ABOVE the GNN reference(s) "
                   "on this split — that is the finding; it is not being softened here."
                   if fp_r2 >= ref_r2 else
                   "The fingerprint-only baseline is below the GNN reference(s) on this "
                   "split, i.e. the graph is adding measurable signal here.")
            )


if __name__ == "__main__":
    main()
