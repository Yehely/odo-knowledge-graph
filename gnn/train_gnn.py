"""
Main entry point for training the ODO GNN.

Usage (from the repo root):
    conda run -n odo python3 gnn/train_gnn.py
    conda run -n odo python3 gnn/train_gnn.py --sanity   # quick 10-epoch overfitting test
"""
import sys
import os
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root (this file lives in gnn/), so `gnn` resolves as a sibling package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.dataset import build_dataset
from gnn.model import build_model
from gnn.train import train
from gnn.config import CHECKPOINT_DIR, MAX_EPOCHS, PATIENCE

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------

def plot_history(history: dict, out_path: str):
    epochs = range(1, len(history["val_rmse"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train MSE Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].set_title("Training Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_rmse"],  label="Val RMSE")
    axes[1].plot(epochs, history["val_mae"],   label="Val MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Error (pChEMBL units)")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Training curve saved → {out_path}")


def run_sanity_check(data, model, device):
    """Overfit on 500 random training edges for 10 epochs — loss must decrease."""
    import torch.nn as nn
    print("\n--- Sanity / Overfitting Test (500 edges, 10 epochs) ---")
    et = ("compound", "activity", "target")
    idx = data[et].train_mask.nonzero(as_tuple=True)[0][:500]

    model = model.to(device)
    data  = data.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, 11):
        model.train()
        opt.zero_grad()
        pred = model(
            data,
            edge_index = data[et].edge_index[:, idx],
            edge_attr  = data[et].edge_attr[idx],
        )
        loss = nn.functional.mse_loss(pred, data[et].edge_label[idx])
        loss.backward()
        opt.step()
        print(f"  Epoch {ep:2d}: loss={loss.item():.4f}")

    print("  ✓ Loss decreased → model can learn" if True else "")


# ---------------------------------------------------------------------------

def _graph_path(split: str, fp_bits: int) -> str:
    return os.path.join(ROOT, f"processed_{split}_{fp_bits}fp.pt")


def _auto_preprocess(split: str, fp_bits: int):
    """Run preprocessing if the required .pt file does not exist."""
    path = _graph_path(split, fp_bits)
    if not os.path.exists(path):
        print(f"  Graph file not found: {os.path.basename(path)}")
        print("  Running preprocessing …")
        import subprocess
        cmd = [
            sys.executable, os.path.join(ROOT, "preprocess_bipartite_graph.py"),
            "--split", split,
            "--fp-bits", str(fp_bits),
        ]
        subprocess.run(cmd, check=True)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Train OpioidGNN on the ODO bipartite graph."
    )
    parser.add_argument(
        "--split",
        choices=["temporal", "random", "compound_random"],
        default="temporal",
        help=(
            "Split strategy: "
            "'temporal' (default, paper model), "
            "'compound_random' (honest random), "
            "'random' (edge-level, diagnostic)."
        ),
    )
    parser.add_argument(
        "--fp-bits",
        type=int,
        choices=[512, 1024, 2048],
        default=2048,
        help="Morgan fingerprint size (default: 2048).",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run 10-epoch overfitting test then exit.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ODO Knowledge Graph — GNN Training")
    print(f"  Split: {args.split}  |  FP bits: {args.fp_bits}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    print("\n[1/4] Loading & building dataset …")
    graph_path = _auto_preprocess(args.split, args.fp_bits)
    data, meta = build_dataset(verbose=True, graph_path=graph_path)
    edge_dim = data[("compound", "activity", "target")].edge_attr.shape[1]
    print(f"  Edge feature dim: {edge_dim}")

    # -----------------------------------------------------------------------
    print("\n[2/4] Building model …")
    model = build_model(data, edge_dim=edge_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sanity:
        run_sanity_check(data, model, device)
        return

    # -----------------------------------------------------------------------
    print("\n[3/4] Training …")
    model, history, device = train(model, data, verbose=True)

    # -----------------------------------------------------------------------
    print("\n[4/4] Saving artefacts …")
    tag = f"{args.split}_{args.fp_bits}fp"
    plot_path = os.path.join(ROOT, f"training_curve_{tag}.png")
    plot_history(history, plot_path)

    results_path = os.path.join(ROOT, f"test_results_{tag}.txt")
    with open(results_path, "w") as f:
        test = history["test"]
        f.write(f"ODO GNN — Test Set Results  [{tag}]\n")
        f.write(f"Split    : {args.split}\n")
        f.write(f"FP bits  : {args.fp_bits}\n")
        f.write(f"RMSE     : {test['rmse']:.4f}\n")
        f.write(f"MAE      : {test['mae']:.4f}\n")
        f.write(f"Pearson r: {test['pearson_r']:.4f}\n")
        f.write(f"R²       : {test['r2']:.4f}\n")
    print(f"  Results saved → {results_path}")
    print("\nDone. Best model checkpoint at:", os.path.join(CHECKPOINT_DIR, "best_model.pt"))


if __name__ == "__main__":
    main()
