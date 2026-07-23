"""
Inference: predict pChEMBL for arbitrary compound SMILES against all targets.

Usage:
    from gnn.predict import load_predictor, predict_smiles

    predictor = load_predictor()
    df = predict_smiles(predictor, "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5")
    print(df.sort_values("pchembl_pred", ascending=False))
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd

from gnn.config import CHECKPOINT_DIR, MORGAN_BITS, MORGAN_RADIUS, EXP_SETTINGS, ENDPOINT_TYPES, QUALIFIERS
from gnn.features import CompoundFeatureBuilder, build_target_features, _one_hot
from gnn.dataset import build_dataset
from gnn.model import build_model

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False


# ---------------------------------------------------------------------------

def load_predictor(ckpt_path: str | None = None):
    """
    Load the trained model and dataset metadata.

    Returns a dict with: model, data, meta, device
    """
    data, meta = build_dataset(verbose=False)
    model = build_model(data, edge_dim=data[("compound","activity","target")].edge_attr.shape[1])

    ckpt = ckpt_path or os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint found at {ckpt}. Run train_gnn.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    data = data.to(device)
    model = model.to(device)

    return dict(model=model, data=data, meta=meta, device=device)


def predict_smiles(predictor: dict, smiles: str) -> pd.DataFrame:
    """
    Predict pChEMBL for a single SMILES against all 55 targets.

    Returns a DataFrame with columns: target_name, species, pchembl_pred
    sorted by prediction descending.
    """
    model   = predictor["model"]
    data    = predictor["data"]
    meta    = predictor["meta"]
    device  = predictor["device"]
    builder: CompoundFeatureBuilder = meta["feature_builder"]
    targets_df = meta["targets_df"]

    # Build a fake compound row
    compound_row = pd.DataFrame([{
        "uri":    "odod:query_compound",
        "smiles": smiles,
        "mw": None, "alogp": None, "ro5_violations": None,
        "max_phase": 0.0, "is_radiolabeled": 0.0,
        "qp_logpow": None, "qp_logs": None, "qp_donor_hb": None,
        "qp_acceptor_hb": None, "qp_logkhsa": None,
        "human_oral_absorption": None, "qp_sasa": None,
    }])
    compound_x = builder.transform(compound_row).to(device)   # [1, F_c]

    n_targets = len(targets_df)

    # Build edge_index: compound 0 → each target 0..N_t-1
    src = torch.zeros(n_targets, dtype=torch.long, device=device)
    dst = torch.arange(n_targets, dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)   # [2, N_t]

    # Default edge features: Ki, "=", "in vitro"
    ep_oh  = _one_hot("Ki", ENDPOINT_TYPES)
    q_oh   = _one_hot("=", QUALIFIERS)
    set_oh = _one_hot("in vitro", EXP_SETTINGS)
    edge_attr_row = np.concatenate([ep_oh, q_oh, set_oh])
    edge_attr = torch.from_numpy(
        np.tile(edge_attr_row, (n_targets, 1)).astype(np.float32)
    ).to(device)

    # We need to temporarily override compound embeddings in the data object
    # Strategy: append the query compound as node N_c, run inference
    orig_x     = data["compound"].x
    orig_nodes = data["compound"].num_nodes

    data["compound"].x         = torch.cat([orig_x, compound_x], dim=0)
    data["compound"].num_nodes = orig_nodes + 1

    # Shift edge_index source to point to the new node
    query_idx = torch.tensor([[orig_nodes]], dtype=torch.long, device=device).expand(1, n_targets)
    edge_index_shifted = torch.stack([query_idx.squeeze(0), dst], dim=0)

    with torch.no_grad():
        preds = model(data, edge_index=edge_index_shifted, edge_attr=edge_attr)

    # Restore
    data["compound"].x         = orig_x
    data["compound"].num_nodes = orig_nodes

    results = pd.DataFrame({
        "target_uri":    targets_df["uri"].values,
        "target_name":   targets_df["name"].values,
        "species":       targets_df["species"].values,
        "pchembl_pred":  preds.cpu().numpy(),
    })
    return results.sort_values("pchembl_pred", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Morphine SMILES
    morphine = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
    print(f"Loading model …")
    predictor = load_predictor()
    print(f"Predicting for morphine …\n")
    df = predict_smiles(predictor, morphine)
    print(df.to_string(index=False))
