"""
Main entry point for training the heterogeneous 5-node-type ODO GNN.

Usage:
    conda run -n odo python3 hetero_gnn/run_train.py
    conda run -n odo python3 hetero_gnn/run_train.py --sanity   # quick 10-epoch overfitting test
"""
import argparse
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hetero_gnn.dataset import build_dataset, BINDS_TO
from hetero_gnn.model import build_model
from hetero_gnn.train import train
from hetero_gnn.config import (
    ASSAY_EMB_DIM, ASSAY_EMB_DIM_RANGE, CHECKPOINT_DIR, DOCUMENT_JOURNAL_EMB_DIM,
    DOCUMENT_JOURNAL_EMB_DIM_RANGE, GRAPH_PATH, MODEL_SYSTEM_EMB_DIM,
    MODEL_SYSTEM_EMB_DIM_RANGE, PKG_DIR,
)
from hetero_gnn.preprocess import main as run_preprocess


def plot_history(history: dict, out_path: str):
    epochs = range(1, len(history["val_rmse"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train MSE Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].set_title("Training Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_rmse"], label="Val RMSE")
    axes[1].plot(epochs, history["val_mae"], label="Val MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Error (pKi units)")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Training curve saved → {out_path}")


def run_sanity_check(data, model, device):
    """Overfit on 500 random training edges for 10 epochs — loss must decrease.

    Uses the real training objective (combined_loss, §1) over whichever of
    the 500 edges are loss-eligible (exact/hinge-floor/hinge-ceiling).
    """
    from hetero_gnn.train import combined_loss
    print("\n--- Sanity / Overfitting Test (500 edges, 10 epochs) ---")
    et = data[BINDS_TO]
    loss_eligible = et.train_exact_mask | et.train_hinge_floor_mask | et.train_hinge_ceiling_mask
    idx = loss_eligible.nonzero(as_tuple=True)[0][:500]

    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for ep in range(1, 11):
        model.train()
        opt.zero_grad()
        pred = model(
            data,
            edge_index=et.edge_index[:, idx],
            edge_attr=et.edge_attr[idx],
        )
        loss = combined_loss(
            pred, et.edge_label[idx],
            et.train_exact_mask[idx], et.train_hinge_floor_mask[idx], et.train_hinge_ceiling_mask[idx],
        )
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"  Epoch {ep:2d}: loss={loss.item():.4f}")

    if losses[-1] < losses[0]:
        print("  ✓ Loss decreased → model can learn")
    else:
        print("  ✗ Loss did not decrease — investigate before full training")


def main():
    parser = argparse.ArgumentParser(description="Train HeteroOpioidGNN on the ODO heterogeneous graph.")
    parser.add_argument("--sanity", action="store_true", help="Run 10-epoch overfitting test then exit.")
    parser.add_argument(
        "--assay-emb-dim", type=int, default=ASSAY_EMB_DIM,
        help=f"Assay identity-embedding size (§5 search range {ASSAY_EMB_DIM_RANGE}); "
             f"default {ASSAY_EMB_DIM}. Plug in search_hparams.py's winning value here.",
    )
    parser.add_argument(
        "--model-system-emb-dim", type=int, default=MODEL_SYSTEM_EMB_DIM,
        help=f"Model System identity-embedding size (§5 search range {MODEL_SYSTEM_EMB_DIM_RANGE}); "
             f"default {MODEL_SYSTEM_EMB_DIM}.",
    )
    parser.add_argument(
        "--document-journal-emb-dim", type=int, default=DOCUMENT_JOURNAL_EMB_DIM,
        help=f"Document journal-embedding size (§5 search range {DOCUMENT_JOURNAL_EMB_DIM_RANGE}); "
             f"default {DOCUMENT_JOURNAL_EMB_DIM}.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ODO Heterogeneous Knowledge Graph — GNN Training")
    print("=" * 60)

    print("\n[1/4] Loading & building dataset …")
    if not os.path.exists(GRAPH_PATH):
        print(f"  Graph file not found: {os.path.basename(GRAPH_PATH)}")
        print("  Running preprocessing …")
        run_preprocess()
    data, meta = build_dataset(verbose=True)
    edge_dim = data[BINDS_TO].edge_attr.shape[1]
    print(f"  Edge feature dim: {edge_dim}")

    print("\n[2/4] Building model …")
    print(f"  assay_emb_dim={args.assay_emb_dim}  model_system_emb_dim={args.model_system_emb_dim}  "
          f"document_journal_emb_dim={args.document_journal_emb_dim}")
    model = build_model(
        data, edge_dim=edge_dim,
        assay_emb_dim=args.assay_emb_dim,
        model_system_emb_dim=args.model_system_emb_dim,
        document_journal_emb_dim=args.document_journal_emb_dim,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sanity:
        run_sanity_check(data, model, device)
        return

    print("\n[3/4] Training …")
    model, history, device = train(model, data, verbose=True)

    print("\n[4/4] Saving artefacts …")
    plot_path = os.path.join(PKG_DIR, "training_curve.png")
    plot_history(history, plot_path)

    results_path = os.path.join(PKG_DIR, "test_results.txt")
    with open(results_path, "w") as f:
        test = history["test"]
        f.write("ODO Heterogeneous GNN — Test Set Results\n")
        f.write(f"RMSE     : {test['rmse']:.4f}\n")
        f.write(f"MAE      : {test['mae']:.4f}\n")
        f.write(f"Pearson r: {test['pearson_r']:.4f}\n")
        f.write(f"R²       : {test['r2']:.4f}\n")
    print(f"  Results saved → {results_path}")
    print("\nDone. Best model checkpoint at:", os.path.join(CHECKPOINT_DIR, "best_model.pt"))


if __name__ == "__main__":
    main()
