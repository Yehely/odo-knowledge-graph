"""
Ensemble training for OpioidGNN.

Trains N_MODELS independent models with different random seeds, then averages
their test-set predictions. Ensemble averaging reduces variance and typically
improves out-of-distribution (temporal) generalisation.

Usage (from the repo root):
    conda run -n odo python3 gnn/train_ensemble.py
"""
import argparse
import math
import os
import sys

# repo root (this file lives in gnn/), so `gnn` resolves as a sibling package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from gnn.config import CHECKPOINT_DIR
from gnn.dataset import build_dataset
from gnn.model import build_model
from gnn.train import train

ROOT = os.path.dirname(os.path.abspath(__file__))
N_MODELS = 5
SEEDS = [42, 123, 456, 789, 1024]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensemble_metrics(preds: list[np.ndarray], target: np.ndarray) -> dict:
    avg = np.mean(np.stack(preds, axis=0), axis=0)
    mse = float(np.mean((avg - target) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(avg - target)))
    r, _ = pearsonr(avg, target) if len(avg) > 2 else (float("nan"), None)
    r2 = r2_score(target, avg) if len(avg) > 2 else float("nan")
    return {"rmse": rmse, "mae": mae, "pearson_r": r, "r2": r2, "predictions": avg}


def _graph_path(split: str, fp_bits: int) -> str:
    return os.path.join(ROOT, f"processed_{split}_{fp_bits}fp.pt")


def _auto_preprocess(split: str, fp_bits: int) -> str:
    path = _graph_path(split, fp_bits)
    if not os.path.exists(path):
        print(f"  Graph file not found: {os.path.basename(path)}")
        print("  Running preprocessing …")
        import subprocess
        cmd = [
            sys.executable, os.path.join(ROOT, "preprocess_bipartite_graph.py"),
            "--split", split, "--fp-bits", str(fp_bits),
        ]
        subprocess.run(cmd, check=True)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble training for OpioidGNN."
    )
    parser.add_argument(
        "--split",
        choices=["temporal", "random", "compound_random"],
        default="temporal",
    )
    parser.add_argument(
        "--fp-bits", type=int, choices=[512, 1024, 2048], default=2048,
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ODO GNN — Ensemble Training")
    print(f"  Models: {N_MODELS}  |  Seeds: {SEEDS}")
    print(f"  Split: {args.split}  |  FP bits: {args.fp_bits}")
    print("=" * 60)

    graph_path = _auto_preprocess(args.split, args.fp_bits)
    data, _ = build_dataset(verbose=True, graph_path=graph_path)
    et = ("compound", "activity", "target")
    edge_dim = data[et].edge_attr.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_mask = data[et].test_mask
    true_labels = data[et].edge_label[test_mask].numpy().astype(float)

    individual_results = []
    all_test_preds = []

    for i, seed in enumerate(SEEDS):
        print(f"\n--- Model {i + 1}/{N_MODELS}  (seed={seed}) ---")
        set_seed(seed)

        model = build_model(data, edge_dim=edge_dim)
        ckpt_name = f"ensemble_seed{seed}.pt"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

        model, history, device = train(model, data, verbose=True)

        # Save this model's checkpoint under a seed-specific name
        src = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, ckpt_path)

        # Collect test predictions
        model.eval()
        data_dev = data.to(device)
        with torch.no_grad():
            ei = data_dev[et].edge_index[:, test_mask.to(device)]
            ea = data_dev[et].edge_attr[test_mask.to(device)]
            preds = model(data_dev, edge_index=ei, edge_attr=ea)
        all_test_preds.append(preds.cpu().numpy().astype(float))
        individual_results.append(history["test"])

    # ── Ensemble metrics ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INDIVIDUAL MODEL RESULTS (test set)")
    print("=" * 60)
    for i, (seed, res) in enumerate(zip(SEEDS, individual_results)):
        print(
            f"  Seed {seed:5d} | RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}"
            f"  r={res['pearson_r']:.4f}  R²={res['r2']:.4f}"
        )

    print("\n" + "=" * 60)
    print("  ENSEMBLE RESULTS (averaged predictions)")
    print("=" * 60)
    ens = ensemble_metrics(all_test_preds, true_labels)
    print(f"  RMSE     : {ens['rmse']:.4f}")
    print(f"  MAE      : {ens['mae']:.4f}")
    print(f"  Pearson r: {ens['pearson_r']:.4f}")
    print(f"  R²       : {ens['r2']:.4f}")

    results_path = os.path.join(ROOT, "ensemble_test_results.txt")
    with open(results_path, "w") as f:
        f.write("ODO GNN Ensemble — Test Set Results\n")
        f.write(f"Models: {N_MODELS}  Seeds: {SEEDS}\n\n")
        f.write("Individual models:\n")
        for seed, res in zip(SEEDS, individual_results):
            f.write(
                f"  seed={seed}: RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}"
                f"  r={res['pearson_r']:.4f}  R²={res['r2']:.4f}\n"
            )
        f.write("\nEnsemble:\n")
        f.write(f"  RMSE     : {ens['rmse']:.4f}\n")
        f.write(f"  MAE      : {ens['mae']:.4f}\n")
        f.write(f"  Pearson r: {ens['pearson_r']:.4f}\n")
        f.write(f"  R²       : {ens['r2']:.4f}\n")
    print(f"\n  Results saved → {results_path}")


if __name__ == "__main__":
    main()
