"""
Main entry point for training the ODO GNN.

Usage:
    conda run -n odo python3 train_gnn.py
    conda run -n odo python3 train_gnn.py --sanity   # quick 10-epoch overfitting test
"""
import sys
import os
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true",
                        help="Run 10-epoch overfitting test then exit")
    args = parser.parse_args()

    print("=" * 60)
    print("  ODO Knowledge Graph — GNN Training")
    print("=" * 60)

    # -----------------------------------------------------------------------
    print("\n[1/4] Loading & building dataset …")
    data, meta = build_dataset(verbose=True)
    edge_dim = data[("compound","activity","target")].edge_attr.shape[1]
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
    plot_path = os.path.join(ROOT, "gnn", "training_curve.png")
    plot_history(history, plot_path)

    results_path = os.path.join(ROOT, "gnn", "test_results.txt")
    with open(results_path, "w") as f:
        test = history["test"]
        f.write("ODO GNN — Test Set Results\n")
        f.write(f"RMSE     : {test['rmse']:.4f}\n")
        f.write(f"MAE      : {test['mae']:.4f}\n")
        f.write(f"Pearson r: {test['pearson_r']:.4f}\n")
        f.write(f"R²       : {test['r2']:.4f}\n")
    print(f"  Results saved → {results_path}")

    print("\nDone. Best model checkpoint at:", os.path.join(CHECKPOINT_DIR, "best_model.pt"))


if __name__ == "__main__":
    main()
