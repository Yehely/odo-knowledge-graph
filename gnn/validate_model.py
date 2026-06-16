"""
Pre-training validation suite for OpioidGNN.

Runs 6 checks and prints a clear PASS/FAIL report:
  1. Data integrity       — shapes, missing values, split balance
  2. Feature sanity       — fingerprint density, ADMET ranges
  3. Naive baseline       — RMSE of mean-predictor (lower bound to beat)
  4. Forward pass         — shapes are correct, no NaN/Inf
  5. Gradient flow        — every parameter receives a gradient
  6. Overfitting capacity — loss drops on 100 samples over 30 epochs

Usage:
    conda run -n odo python3 -m gnn.validate_model
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.dataset import build_dataset
from gnn.model import build_model

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
INFO = "\033[94m  INFO\033[0m"

results = {}


def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"{tag}  [{name}]  {detail}")
    results[name] = ok


# ============================================================
# Load data once
# ============================================================
print("\n=== ODO GNN — Pre-Training Validation ===\n")
print("Loading data …")

data, meta = build_dataset(verbose=False)
et = ("compound", "activity", "target")
edge_dim = data[et].edge_attr.shape[1]
model = build_model(data, edge_dim=edge_dim)
device = torch.device("cpu")


# ============================================================
# 1. DATA INTEGRITY
# ============================================================
print("\n--- Check 1: Data Integrity ---")

n_comp  = data["compound"].num_nodes
n_targ  = data["target"].num_nodes
n_edges = data[et].edge_index.shape[1]
n_train = int(data[et].train_mask.sum())
n_val   = int(data[et].val_mask.sum())
n_test  = int(data[et].test_mask.sum())
n_covered = n_train + n_val + n_test

check("edge coverage",
      n_covered == n_edges,
      f"train={n_train:,} + val={n_val:,} + test={n_test:,} = {n_covered:,}  (total={n_edges:,})")

train_set = set(data[et].train_mask.nonzero(as_tuple=True)[0].tolist())
val_set   = set(data[et].val_mask.nonzero(as_tuple=True)[0].tolist())
test_set  = set(data[et].test_mask.nonzero(as_tuple=True)[0].tolist())
check("split disjoint",
      not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set),
      "no edge appears in two splits")

labels = data[et].edge_label
check("label range",
      float(labels.min()) >= 2.0 and float(labels.max()) <= 15.0,
      f"pChEMBL ∈ [{labels.min():.2f}, {labels.max():.2f}]")

m_tr = float(labels[data[et].train_mask].mean())
m_va = float(labels[data[et].val_mask].mean())
m_te = float(labels[data[et].test_mask].mean())
check("split balance",
      abs(m_tr - m_va) < 0.5 and abs(m_tr - m_te) < 0.5,
      f"pChEMBL mean — train={m_tr:.3f}  val={m_va:.3f}  test={m_te:.3f}")

nan_c = torch.isnan(data["compound"].x).any().item()
nan_t = torch.isnan(data["target"].x).any().item()
nan_e = torch.isnan(data[et].edge_attr).any().item()
check("no NaN in features",
      not (nan_c or nan_t or nan_e),
      f"compound_NaN={nan_c}  target_NaN={nan_t}  edge_NaN={nan_e}")


# ============================================================
# 2. FEATURE SANITY
# ============================================================
print("\n--- Check 2: Feature Sanity ---")

fp = data["compound"].x[:, :2048]
bit_density = float(fp.mean())
check("fingerprint density",
      0.01 < bit_density < 0.25,
      f"mean bit density = {bit_density:.4f}  (expected ~0.03–0.15)")

nonzero_fps = int((fp.sum(dim=1) > 0).sum())
check("fingerprints non-zero",
      nonzero_fps / n_comp > 0.90,
      f"{nonzero_fps:,}/{n_comp:,} = {100*nonzero_fps/n_comp:.1f}% of compounds have non-zero fingerprint")

mw_col = data["compound"].x[:, 2048]
check("ADMET feature spread",
      float(mw_col.std()) > 0.1,
      f"normalised MW std = {float(mw_col.std()):.3f}  (> 0.1 means scaler is working)")

ep_cols = data[et].edge_attr[:, :7]
ep_hit  = int((ep_cols.sum(dim=0) > 0).sum())
check("edge endpoint types",
      ep_hit >= 2,
      f"{ep_hit}/7 endpoint-type slots have at least one positive")


# ============================================================
# 3. NAIVE BASELINE
# ============================================================
print("\n--- Check 3: Naive Baseline ---")

train_labels = labels[data[et].train_mask]
val_labels   = labels[data[et].val_mask]
test_labels  = labels[data[et].test_mask]
train_mean   = float(train_labels.mean())

val_rmse_baseline  = float(torch.sqrt(((val_labels  - train_mean) ** 2).mean()))
test_rmse_baseline = float(torch.sqrt(((test_labels - train_mean) ** 2).mean()))

print(f"{INFO}  Mean predictor (train mean = {train_mean:.3f})")
print(f"{INFO}    Val  RMSE = {val_rmse_baseline:.4f}  ← model must beat this")
print(f"{INFO}    Test RMSE = {test_rmse_baseline:.4f}  ← model must beat this")
results["baseline_val_rmse"]  = val_rmse_baseline
results["baseline_test_rmse"] = test_rmse_baseline


# ============================================================
# 4. FORWARD PASS
# ============================================================
print("\n--- Check 4: Forward Pass ---")

model.eval()
data_d = data.to(device)
model  = model.to(device)

with torch.no_grad():
    try:
        preds = model(data_d)
        check("output shape",   preds.shape == (n_edges,),
              f"expected [{n_edges}], got {list(preds.shape)}")
        check("no NaN output",  not torch.isnan(preds).any().item(),
              f"NaN count = {torch.isnan(preds).sum().item()}")
        check("no Inf output",  not torch.isinf(preds).any().item(),
              f"Inf count = {torch.isinf(preds).sum().item()}")
        check("output std > 0", float(preds.std()) > 0,
              f"pred std = {float(preds.std()):.4f}")
        # After output-bias init the initial predictions should be near the training mean
        pred_mean = float(preds.mean())
        check("output near label mean",
              abs(pred_mean - train_mean) < 2.0,
              f"pred mean={pred_mean:.3f}  label mean={train_mean:.3f}")
    except Exception as e:
        check("forward pass exception", False, str(e))


# ============================================================
# 5. GRADIENT FLOW
# ============================================================
print("\n--- Check 5: Gradient Flow ---")

model.train()
preds = model(data_d)
loss  = nn.functional.mse_loss(preds, data_d[et].edge_label)
loss.backward()

no_grad   = [n for n, p in model.named_parameters() if p.grad is None]
zero_grad = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() == 0]
ok_grad   = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() > 0]

check("all params have gradient",
      len(no_grad) == 0,
      f"{len(ok_grad)} OK, {len(zero_grad)} zero-grad, {len(no_grad)} no-grad")
if no_grad:
    print(f"  ⚠  No-gradient params: {no_grad[:5]}")
if zero_grad:
    print(f"  ⚠  Zero-gradient params: {zero_grad[:5]}")

model.zero_grad()


# ============================================================
# 6. OVERFITTING CAPACITY (30 epochs on 100 samples)
# ============================================================
print("\n--- Check 6: Overfitting Capacity (100 edges, 30 epochs) ---")

model.train()
idx100 = data_d[et].train_mask.nonzero(as_tuple=True)[0][:100]
ei100  = data_d[et].edge_index[:, idx100]
ea100  = data_d[et].edge_attr[idx100]
lbl100 = data_d[et].edge_label[idx100]

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for _ in range(30):
    opt.zero_grad()
    p = model(data_d, edge_index=ei100, edge_attr=ea100)
    l = nn.functional.mse_loss(p, lbl100)
    l.backward()
    opt.step()
    losses.append(l.item())

loss_drop  = (losses[0] - losses[-1]) / max(losses[0], 1e-9)
rmse_final = math.sqrt(losses[-1])
baseline   = results.get("baseline_val_rmse", 1.5)
overfit_ok = loss_drop > 0.40 and rmse_final < baseline

check("overfitting capacity",
      overfit_ok,
      f"loss: {losses[0]:.3f} → {losses[-1]:.3f}  "
      f"(drop={100*loss_drop:.1f}%,  RMSE={rmse_final:.3f}  baseline={baseline:.3f})")
loss_every5 = "  ".join(f"ep{i+1}={losses[i]:.2f}" for i in [0, 4, 9, 19, 29])
print(f"  {INFO}  {loss_every5}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 50)
print("  VALIDATION SUMMARY")
print("=" * 50)

bool_results = {k: v for k, v in results.items() if isinstance(v, bool)}
passed = sum(bool_results.values())
total  = len(bool_results)

for name, val in bool_results.items():
    print(f"  {PASS if val else FAIL}  {name}")

print(f"\n  Baseline RMSE to beat → val={results['baseline_val_rmse']:.4f}  "
      f"test={results['baseline_test_rmse']:.4f}")
print(f"\n  {passed}/{total} checks passed")

if passed == total:
    print("\n  \033[92m✓ Model is ready for full training.\033[0m")
    print("  Run:  conda run -n odo python3 train_gnn.py")
else:
    print("\n  \033[91m✗ Fix failing checks before running full training.\033[0m")
