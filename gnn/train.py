"""
Training loop for OpioidGNN.

Functions:
  asymmetric_loss(pred, target)               -> scalar loss tensor
  train_epoch(model, data, optimizer, device) -> float (mean train loss)
  evaluate(model, data, mask_name, device)    -> dict of metrics
  train(model, data)                          -> trained model + history
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

from gnn.config import (
    BATCH_SIZE, CHECKPOINT_DIR, LR, LR_PATIENCE, MAX_EPOCHS, PATIENCE,
    WEIGHT_DECAY,
)


# ---------------------------------------------------------------------------
# Asymmetric loss (as specified in Progress Report 5)
# ---------------------------------------------------------------------------

# pChEMBL >= 7 corresponds to Ki < 100 nM (pharmacologically active).
# Under-prediction of these potent compounds is penalised ALPHA times harder.
_HIGH_AFFINITY_THRESHOLD = 7.0
_ALPHA = 3.0


def asymmetric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = _ALPHA,
    threshold: float = _HIGH_AFFINITY_THRESHOLD,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    MSE-based asymmetric loss with optional per-sample temporal weighting.

    High-affinity under-predictions (target >= threshold, pred < target) are
    penalised alpha times more. sample_weights multiplies each edge's loss
    before averaging (used for temporal weighting: recent edges > old edges).
    """
    error = pred - target  # negative = under-prediction
    sq_err = error ** 2

    high_affinity = (target >= threshold).float()
    under_predict = (error < 0).float()
    affinity_weight = 1.0 + (alpha - 1.0) * high_affinity * under_predict

    loss_per_sample = affinity_weight * sq_err
    if sample_weights is not None:
        loss_per_sample = loss_per_sample * sample_weights
    return loss_per_sample.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(data, device):
    return data.to(device)


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
    et = ("compound", "activity", "target")

    train_idx = data[et].train_mask.nonzero(as_tuple=True)[0]
    n_edges = train_idx.shape[0]
    perm = torch.randperm(n_edges)
    total_loss = 0.0
    n_batches = max(1, n_edges // BATCH_SIZE)

    for b in range(n_batches):
        batch_idx = train_idx[perm[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]]
        ei = data[et].edge_index[:, batch_idx]
        ea = data[et].edge_attr[batch_idx].to(device)
        lbls = data[et].edge_label[batch_idx].to(device)

        optimizer.zero_grad()
        pred = model(data, edge_index=ei.to(device), edge_attr=ea)
        sw = data[et].sample_weight[batch_idx].to(device)
        loss = asymmetric_loss(pred, lbls, sample_weights=sw)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data, mask_name: str, device):
    model.eval()
    et = ("compound", "activity", "target")
    mask = getattr(data[et], mask_name)

    edge_index = data[et].edge_index[:, mask].to(device)
    edge_attr = data[et].edge_attr[mask].to(device)
    edge_label = data[et].edge_label[mask].to(device)

    pred = model(data, edge_index=edge_index, edge_attr=edge_attr)
    # Report MSE loss for evaluation (comparable across runs)
    loss = nn.functional.mse_loss(pred, edge_label).item()
    m = _metrics(pred, edge_label)
    m["loss"] = loss
    return m


# ---------------------------------------------------------------------------
# Full training routine
# ---------------------------------------------------------------------------

def train(model, data, verbose: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"  Device: {device}")

    data = _to_device(data, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=0.5, min_lr=1e-6,
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    no_improve = 0
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    history = {
        "train_loss": [], "val_rmse": [], "val_mae": [],
        "val_r": [], "val_r2": [],
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, data, optimizer, device)
        val = evaluate(model, data, "val_mask", device)

        scheduler.step(val["rmse"])
        history["train_loss"].append(loss)
        history["val_rmse"].append(val["rmse"])
        history["val_mae"].append(val["mae"])
        history["val_r"].append(val["pearson_r"])
        history["val_r2"].append(val["r2"])

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
                f"r={val['pearson_r']:.4f}  R²={val['r2']:.4f}"
                f"  lr={lr_now:.2e}"
            )

        if no_improve >= PATIENCE:
            if verbose:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(best: {best_epoch}, RMSE={best_val_rmse:.4f})"
                )
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test = evaluate(model, data, "test_mask", device)

    if verbose:
        print("\n  === TEST SET RESULTS ===")
        print(f"  RMSE     : {test['rmse']:.4f}")
        print(f"  MAE      : {test['mae']:.4f}")
        print(f"  Pearson r: {test['pearson_r']:.4f}")
        print(f"  R²       : {test['r2']:.4f}")

    history["test"] = test
    return model, history, device
