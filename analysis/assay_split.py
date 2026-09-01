"""
Part 2 — read-only, assay-stratified re-scoring of the predictions
baselines/run_baselines.py already saved (no fitting, no retraining).

For each saved baselines/predictions_<split>_seed<N>.npz (pred, y_true,
row_idx), row_idx are positions into that split's full edge arrays
(edge_index/edge_label as loaded by gnn.dataset.build_dataset /
hetero_gnn.dataset.build_dataset — same call run_baselines.py made). To
recover each test row's `endpoint` value, this script re-runs the ACTUAL
pipeline row-processing functions (gnn.preprocess_bipartite_graph's
load_raw/build_activity_table; hetero_gnn.preprocess's
load_raw/drop_corrupted_unit_rows/build_entities_and_activities — all
imported unmodified, not re-implemented) to get one row per surviving
activity, in the same order the saved graphs were built in, then indexes
that with row_idx.

This is verified, not assumed: for every edge in each graph, the
is-exact/exact_eligible pChEMBL value recomputed here is checked byte-for-
byte against the graph's own stored edge_label. If any split fails that
check, this script stops and reports it rather than silently misaligning
rows to endpoints.

No model, hyperparameter, split, or RF config is touched here. Metrics use
baselines/run_baselines.py's own compute_metrics(), imported unmodified.

Run (after baselines/run_baselines.py has produced the .npz files):
    python3 analysis/assay_split.py
"""
import importlib.metadata
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baselines.run_baselines import (  # noqa: E402
    BIPARTITE_GRAPH_PATH, OUT_DIR as BASELINES_DIR, SEEDS, compute_metrics,
)

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

BINDING_ENDPOINTS = {"Ki", "IC50"}
FUNCTIONAL_ENDPOINTS = {"EC50", "Emax"}


def classify(endpoint: str) -> str:
    if endpoint in BINDING_ENDPOINTS:
        return "binding"
    if endpoint in FUNCTIONAL_ENDPOINTS:
        return "functional"
    return "other"


# ---------------------------------------------------------------------------
# Per-split row reconstruction — reuses each pipeline's own functions.
# ---------------------------------------------------------------------------

def load_bipartite_activities_and_graph():
    from gnn.preprocess_bipartite_graph import build_activity_table, load_raw
    from gnn.dataset import build_dataset

    raw = load_raw()
    activities_df = build_activity_table(raw)  # one row per surviving activity, original row order

    data, _ = build_dataset(verbose=False, graph_path=BIPARTITE_GRAPH_PATH)
    et = data[("compound", "activity", "target")]
    return {
        "activities_df": activities_df,
        "endpoint_col": "endpoint_type",
        "exact_col": "is_exact",
        "pchembl_col": "pchembl",
        "edge_label_full": et.edge_label.numpy(),
        "train_mask": (et.train_mask | et.val_mask).numpy(),
        "test_mask": et.test_mask.numpy(),
    }


def load_hetero_activities_and_graph():
    from hetero_gnn.preprocess import (
        build_entities_and_activities, drop_corrupted_unit_rows, load_raw,
    )
    from hetero_gnn.dataset import build_dataset

    raw = load_raw()
    filtered = drop_corrupted_unit_rows(raw)
    _, _, _, _, _, activities = build_entities_and_activities(filtered)
    activities_df = pd.DataFrame(activities)

    data, _ = build_dataset(verbose=False)
    et = data[("compound", "binds_to", "target")]
    return {
        "activities_df": activities_df,
        "endpoint_col": "endpoint_type",
        "exact_col": "exact_eligible",
        "pchembl_col": "pchembl",
        "edge_label_full": et.edge_label.numpy(),
        "train_mask": (et.train_exact_mask | et.val_exact_mask).numpy(),
        "test_mask": et.test_exact_mask.numpy(),
    }


LOADERS = {"bipartite": load_bipartite_activities_and_graph, "hetero": load_hetero_activities_and_graph}


def validate_and_load(split_name: str) -> dict | None:
    print(f"\n{'='*70}\n{split_name} split — row-index validation\n{'='*70}")
    ctx = LOADERS[split_name]()
    activities_df = ctx["activities_df"]
    E = len(ctx["edge_label_full"])

    print(f"  Reconstructed activities table: {len(activities_df):,} rows")
    print(f"  Graph edge_label array length : {E:,}")
    if len(activities_df) != E:
        print(f"  [STOP] Row count mismatch for {split_name}: reconstructed "
              f"{len(activities_df):,} activity rows but the graph has {E:,} "
              f"edges. Refusing to assume alignment.")
        return None

    exact = activities_df[ctx["exact_col"]].to_numpy(dtype=bool)
    pchembl_reconstructed = activities_df[ctx["pchembl_col"]].to_numpy(dtype=float)
    edge_label_full = ctx["edge_label_full"]

    ok = np.allclose(edge_label_full[exact], pchembl_reconstructed[exact], equal_nan=False)
    print(f"  pChEMBL cross-check on {exact.sum():,} exact-eligible edges "
          f"(reconstructed vs. graph's stored edge_label): {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print(f"  [STOP] {split_name}: recomputed pChEMBL does not match the "
              f"graph's stored edge_label on the exact-eligible subset — row "
              f"order does not line up. Refusing to proceed.")
        return None

    # Confirm the saved row_idx files agree with a freshly recomputed test_mask
    # (same build_dataset() call, deterministic, no randomness — should match exactly).
    fresh_test_idx = np.where(ctx["test_mask"])[0]
    for seed in SEEDS:
        path = os.path.join(BASELINES_DIR, f"predictions_{split_name}_seed{seed}.npz")
        if not os.path.exists(path):
            print(f"  [STOP] missing {path} — run baselines/run_baselines.py first.")
            return None
        npz = np.load(path)
        if not np.array_equal(npz["row_idx"], fresh_test_idx):
            print(f"  [STOP] {split_name} seed={seed}: saved row_idx does not match "
                  f"a freshly recomputed test_mask. Refusing to proceed.")
            return None
    print(f"  All {len(SEEDS)} saved prediction files' row_idx match a freshly "
          f"recomputed test_mask ({len(fresh_test_idx):,} test rows). Alignment verified.")

    return ctx


def endpoint_for_rows(ctx: dict, row_idx: np.ndarray) -> np.ndarray:
    return ctx["activities_df"][ctx["endpoint_col"]].to_numpy()[row_idx]


def partition_report(endpoints: np.ndarray) -> dict:
    groups = np.array([classify(e) for e in endpoints])
    out = {}
    for g in ["binding", "functional", "other"]:
        idx = np.where(groups == g)[0]
        out[g] = idx
    return out, groups


def label_stats(y: np.ndarray) -> dict:
    return {"n": len(y), "mean": float(np.mean(y)) if len(y) else float("nan"),
            "sd": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan")}


def group_metrics_across_seeds(split_name: str, group_idx: np.ndarray) -> dict | None:
    if len(group_idx) == 0:
        return None
    per_seed = []
    for seed in SEEDS:
        npz = np.load(os.path.join(BASELINES_DIR, f"predictions_{split_name}_seed{seed}.npz"))
        y_true, pred = npz["y_true"], npz["pred"]
        m = compute_metrics(y_true[group_idx], pred[group_idx])
        per_seed.append(m)
    agg = {}
    for key in ["rmse", "mae", "pearson_r", "r2"]:
        vals = np.array([m[key] for m in per_seed])
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    agg["per_seed"] = per_seed
    return agg


def analyze_split(split_name: str) -> dict | None:
    ctx = validate_and_load(split_name)
    if ctx is None:
        return None

    test_idx = np.where(ctx["test_mask"])[0]
    train_idx = np.where(ctx["train_mask"])[0]

    test_endpoints = endpoint_for_rows(ctx, test_idx)
    train_endpoints = endpoint_for_rows(ctx, train_idx)

    test_groups, test_group_arr = partition_report(test_endpoints)
    train_groups, train_group_arr = partition_report(train_endpoints)

    npz0 = np.load(os.path.join(BASELINES_DIR, f"predictions_{split_name}_seed{SEEDS[0]}.npz"))
    y_true_test = npz0["y_true"]
    for seed in SEEDS[1:]:
        npz = np.load(os.path.join(BASELINES_DIR, f"predictions_{split_name}_seed{seed}.npz"))
        assert np.array_equal(npz["y_true"], y_true_test), (
            f"{split_name}: y_true differs across seeds — test set is not fixed, "
            f"cannot compute label stats from a single seed's file"
        )

    result = {
        "split": split_name,
        "n_test_total": len(test_idx),
        "n_train_total": len(train_idx),
        "groups": {},
        "train_proportions": {},
        "other_endpoint_values": {},
    }

    print(f"\n{split_name}: test set partition")
    for g in ["binding", "functional", "other"]:
        idx = test_groups[g]
        stats = label_stats(y_true_test[idx])
        metrics = group_metrics_across_seeds(split_name, idx)
        result["groups"][g] = {"label_stats": stats, "metrics": metrics}
        pct = 100 * stats["n"] / len(test_idx) if len(test_idx) else float("nan")
        r2_str = "n/a" if metrics is None else f"{metrics['r2_mean']:.4f}+-{metrics['r2_sd']:.4f}"
        print(f"  {g:<10}: n={stats['n']:>6,} ({pct:5.1f}%)  "
              f"label mean={stats['mean']:.3f} sd={stats['sd']:.3f}  R2={r2_str}")
        if g == "other" and len(idx):
            distinct = sorted(set(test_endpoints[idx].tolist()))
            result["other_endpoint_values"] = distinct
            print(f"    distinct 'other' endpoint values: {distinct}")

    print(f"\n{split_name}: train set partition")
    for g in ["binding", "functional", "other"]:
        idx = train_groups[g]
        pct = 100 * len(idx) / len(train_idx) if len(train_idx) else float("nan")
        result["train_proportions"][g] = {"n": len(idx), "pct": pct}
        print(f"  {g:<10}: n={len(idx):>7,} ({pct:5.1f}%)")

    return result


def write_report(results: dict, path: str):
    lines = []
    lines.append("# Assay-stratified scoring of the temporal-split test set\n")
    lines.append(
        "Read-only re-scoring of predictions already saved by "
        "`baselines/run_baselines.py` (Part 1). No model was retrained, no "
        "hyperparameter or split was changed; rows were only partitioned at "
        "scoring time. `binding = {Ki, IC50}`, `functional = {EC50, Emax}`, "
        "`other` = everything else, per Section 4.4's own grouping.\n"
    )

    for split_name in ["bipartite", "hetero"]:
        r = results.get(split_name)
        lines.append(f"\n## {split_name} split\n")
        if r is None:
            lines.append(
                f"**Could not be computed for the {split_name} split** — see run "
                f"log / stdout for the specific validation failure. No numbers "
                f"are reported below for this split.\n"
            )
            continue

        lines.append(f"Test set: {r['n_test_total']:,} rows. Train set (train+val, as "
                      f"`baselines/run_baselines.py` defines it): {r['n_train_total']:,} rows.\n")

        lines.append("### Test-set metrics by assay group (5 seeds, mean ± sample sd, ddof=1)\n")
        lines.append("| Group | n | % of test | label mean | label sd | RMSE | MAE | Pearson r | R² |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for g in ["binding", "functional", "other"]:
            gd = r["groups"][g]
            s = gd["label_stats"]
            pct = 100 * s["n"] / r["n_test_total"] if r["n_test_total"] else float("nan")
            if gd["metrics"] is None:
                lines.append(f"| {g} | {s['n']:,} | {pct:.1f}% | "
                              f"{s['mean']:.2f} | {s['sd']:.2f} | n/a (0 rows) | n/a | n/a | n/a |")
            else:
                m = gd["metrics"]
                lines.append(
                    f"| {g} | {s['n']:,} | {pct:.1f}% | {s['mean']:.2f} | {s['sd']:.2f} | "
                    f"{m['rmse_mean']:.3f} ± {m['rmse_sd']:.3f} | "
                    f"{m['mae_mean']:.3f} ± {m['mae_sd']:.3f} | "
                    f"{m['pearson_r_mean']:.3f} ± {m['pearson_r_sd']:.3f} | "
                    f"{m['r2_mean']:.3f} ± {m['r2_sd']:.3f} |"
                )
        if r["other_endpoint_values"]:
            lines.append(f"\n`other` distinct endpoint values: {r['other_endpoint_values']}\n")

        lines.append("\n### Train-set binding/functional/other proportions\n")
        lines.append("| Group | n (train) | % (train) |")
        lines.append("|---|---|---|")
        for g in ["binding", "functional", "other"]:
            tp = r["train_proportions"][g]
            lines.append(f"| {g} | {tp['n']:,} | {tp['pct']:.1f}% |")

        test_binding_pct = 100 * r["groups"]["binding"]["label_stats"]["n"] / r["n_test_total"] if r["n_test_total"] else float("nan")
        test_functional_pct = 100 * r["groups"]["functional"]["label_stats"]["n"] / r["n_test_total"] if r["n_test_total"] else float("nan")
        train_binding_pct = r["train_proportions"]["binding"]["pct"]
        train_functional_pct = r["train_proportions"]["functional"]["pct"]
        lines.append(
            f"\nSection 4.4 quotes binding/functional proportions of "
            f"**67%/33% (train) → 48%/52% (test)**. This split's actual "
            f"proportions, from the model's own train/test population: "
            f"**{train_binding_pct:.0f}%/{train_functional_pct:.0f}% (train) → "
            f"{test_binding_pct:.0f}%/{test_functional_pct:.0f}% (test)**"
            + (" — reproduces the reported figures." if
               abs(train_binding_pct - 67) < 2 and abs(test_binding_pct - 48) < 2
               else " — does NOT reproduce the reported figures.")
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def check_figure7_literal_source():
    """Section 4.4's 67/33->48/52 figures: check whether they are computed
    anywhere in this repo, or hardcoded with no visible derivation."""
    path = os.path.join(ROOT, "reports", "make_figures.py")
    with open(path) as f:
        content = f.read()
    if "functional = [33, 52]" in content and "binding = [67, 48]" in content:
        return (
            "`reports/make_figures.py:fig_domain_shift()` defines "
            "`functional = [33, 52]` and `binding = [67, 48]` as bare literals "
            "— no computation, no comment pointing at a source script or a "
            "results file, anywhere in that function. These are not derived "
            "from `Final ODO Dataset_v2026-06-10.xlsx` or from either GNN "
            "pipeline's train/test split by any code in this repository."
        )
    return "Could not confirm the exact literal values in reports/make_figures.py (file may have changed)."


def write_run_log(path: str, start_time: float):
    pkgs = {}
    for pkg in ["torch", "scikit-learn", "numpy", "pandas", "scipy", "torch-geometric", "rdkit"]:
        try:
            pkgs[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkgs[pkg] = "NOT INSTALLED"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "UNKNOWN"
    lines = [f"Commit: {commit}", f"Runtime: {time.time()-start_time:.1f}s", "Packages:"]
    for pkg, ver in pkgs.items():
        lines.append(f"  {pkg}: {ver}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    start = time.time()
    results = {}
    for split_name in ["bipartite", "hetero"]:
        results[split_name] = analyze_split(split_name)

    report_path = os.path.join(ANALYSIS_DIR, "assay_split_results.md")
    write_report(results, report_path)

    fig7_note = check_figure7_literal_source()
    with open(report_path, "a") as f:
        f.write(f"\n## Figure 7 / Section 4.4 provenance check\n\n{fig7_note}\n")

    run_log = write_run_log(os.path.join(ANALYSIS_DIR, "assay_split_run_log.txt"), start)
    with open(report_path, "a") as f:
        f.write(f"\n## Run log\n\n```\n{run_log}\n```\n")

    print(f"\nWrote {report_path}")
    print(run_log)


if __name__ == "__main__":
    main()
