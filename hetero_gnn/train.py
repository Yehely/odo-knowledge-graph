"""
Training loop for HeteroOpioidGNN.

§1/§9: loss has two components, both restricted to endpoint=='Ki' edges with
a resolvable qualifier:
  1. Exact MSE       — qualifier=='=', label in [MIN_PCHEMBL, MAX_PCHEMBL].
  2. Censored hinge   — one-sided, branched on the *raw* qualifier:
       raw '<'/'<=' (floor)   : (max(0, y - pred))^2
       raw '>'/'>=' (ceiling) : (max(0, pred - y))^2
Total = mean(exact terms) + CENSORED_LOSS_WEIGHT * mean(hinge terms).
Non-Ki and unrecoverable/'~'-qualifier edges contribute to neither term but
stay in the graph for message passing (already true by construction — see
preprocess.py).

Reported RMSE/MAE/Pearson-r/R² (and early-stopping / checkpoint selection)
use the exact-labelled subset only — those are the only edges with a real
point-value ground truth to score standard regression metrics against; a
one-sided bound isn't a "point" a normal error metric can be computed on.

Functions:
  combined_loss(pred, edge_label, exact_mask, floor_mask, ceiling_mask, weight) -> loss tensor
  train_epoch(model, data, optimizer, device)  -> float (mean train loss)
  evaluate(model, data, split, device)          -> dict of metrics
  train(model, data)                             -> trained model + history
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from hetero_gnn.config import (
    BATCH_SIZE, CENSORED_LOSS_WEIGHT, CHECKPOINT_DIR, LR, LR_PATIENCE, MAX_EPOCHS,
    PATIENCE, WEIGHT_DECAY,
)

BINDS_TO = ("compound", "binds_to", "target")


# ---------------------------------------------------------------------------
# §1 combined loss
# ---------------------------------------------------------------------------

def combined_loss(
    pred: torch.Tensor,
    edge_label: torch.Tensor,
    exact_mask: torch.Tensor,
    hinge_floor_mask: torch.Tensor,
    hinge_ceiling_mask: torch.Tensor,
    hinge_weight: float = CENSORED_LOSS_WEIGHT,
) -> torch.Tensor:
    zero = torch.zeros((), device=pred.device)

    loss_exact = (
        ((pred[exact_mask] - edge_label[exact_mask]) ** 2).mean()
        if exact_mask.any() else zero
    )

    floor_terms = torch.clamp(edge_label[hinge_floor_mask] - pred[hinge_floor_mask], min=0) ** 2
    ceiling_terms = torch.clamp(pred[hinge_ceiling_mask] - edge_label[hinge_ceiling_mask], min=0) ** 2
    hinge_terms = torch.cat([floor_terms, ceiling_terms])
    loss_censored = hinge_terms.mean() if hinge_terms.numel() > 0 else zero

    return loss_exact + hinge_weight * loss_censored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    p = pred.detach().cpu().numpy().astype(float)
    t = target.detach().cpu().numpy().astype(float)
    mse = float(np.mean((p - t) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(p - t)))
    r, _ = pearsonr(p, t) if len(p) > 2 else (float("nan"), None)
    r2 = r2_score(t, p) if len(p) > 2 else float("nan")
    return {"rmse": rmse, "mae": mae, "pearson_r": r, "r2": r2}


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, data, optimizer, device):
    model.train()
    et = data[BINDS_TO]

    train_idx = (
        et.train_exact_mask | et.train_hinge_floor_mask | et.train_hinge_ceiling_mask
    ).nonzero(as_tuple=True)[0]
    n_edges = train_idx.shape[0]
    perm = torch.randperm(n_edges)
    total_loss = 0.0
    n_batches = max(1, n_edges // BATCH_SIZE)

    for b in range(n_batches):
        batch_idx = train_idx[perm[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]]
        ei = et.edge_index[:, batch_idx]
        ea = et.edge_attr[batch_idx].to(device)
        lbls = et.edge_label[batch_idx].to(device)
        exact_m = et.train_exact_mask[batch_idx].to(device)
        floor_m = et.train_hinge_floor_mask[batch_idx].to(device)
        ceil_m = et.train_hinge_ceiling_mask[batch_idx].to(device)

        optimizer.zero_grad()
        pred = model(data, edge_index=ei.to(device), edge_attr=ea)
        loss = combined_loss(pred, lbls, exact_m, floor_m, ceil_m)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data, split: str, device):
    """split: 'train' | 'val' | 'test'."""
    model.eval()
    et = data[BINDS_TO]
    exact_mask = getattr(et, f"{split}_exact_mask")
    floor_mask = getattr(et, f"{split}_hinge_floor_mask")
    ceil_mask = getattr(et, f"{split}_hinge_ceiling_mask")
    loss_mask = exact_mask | floor_mask | ceil_mask

    edge_index = et.edge_index[:, loss_mask].to(device)
    edge_attr = et.edge_attr[loss_mask].to(device)
    edge_label = et.edge_label[loss_mask].to(device)
    exact_m = exact_mask[loss_mask].to(device)
    floor_m = floor_mask[loss_mask].to(device)
    ceil_m = ceil_mask[loss_mask].to(device)

    pred = model(data, edge_index=edge_index, edge_attr=edge_attr)
    combined = combined_loss(pred, edge_label, exact_m, floor_m, ceil_m).item()

    # Standard regression metrics on the exact-labelled subset only — the
    # only edges with a real point-value ground truth (see module docstring).
    m = _metrics(pred[exact_m], edge_label[exact_m])
    m["combined_loss"] = combined
    return m


# ---------------------------------------------------------------------------
# Full training routine
# ---------------------------------------------------------------------------

def train(model, data, verbose: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"  Device: {device}")

    data = data.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=0.5, min_lr=1e-6,
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    no_improve = 0
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    history = {
        "train_loss": [], "val_rmse": [], "val_mae": [], "val_r": [], "val_r2": [],
        "val_combined_loss": [],
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, data, optimizer, device)
        val = evaluate(model, data, "val", device)

        scheduler.step(val["rmse"])
        history["train_loss"].append(loss)
        history["val_rmse"].append(val["rmse"])
        history["val_mae"].append(val["mae"])
        history["val_r"].append(val["pearson_r"])
        history["val_r2"].append(val["r2"])
        history["val_combined_loss"].append(val["combined_loss"])

        if val["rmse"] < best_val_rmse:
            best_val_rmse = val["rmse"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d} | train_loss={loss:.4f} | "
                f"val_RMSE={val['rmse']:.4f}  MAE={val['mae']:.4f}  "
                f"r={val['pearson_r']:.4f}  R²={val['r2']:.4f}  "
                f"combined={val['combined_loss']:.4f}"
                f"  lr={lr_now:.2e}"
            )

        if no_improve >= PATIENCE:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch}, RMSE={best_val_rmse:.4f})")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test = evaluate(model, data, "test", device)

    if verbose:
        print("\n  === TEST SET RESULTS (exact-labelled edges) ===")
        print(f"  RMSE     : {test['rmse']:.4f}")
        print(f"  MAE      : {test['mae']:.4f}")
        print(f"  Pearson r: {test['pearson_r']:.4f}")
        print(f"  R²       : {test['r2']:.4f}")
        print(f"  Combined loss (exact + hinge, all loss-eligible edges): {test['combined_loss']:.4f}")

    history["test"] = test
    history["best_epoch"] = best_epoch
    return model, history, device
