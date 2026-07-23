"""
Inference: predict pKi for an arbitrary compound SMILES against every target
already in the graph.

A query compound has no real experimental history (no Assay connections), so
its hop-2 "pooled Assay context" mean-aggregates over zero neighbours and
correctly degrades to a zero vector (verified: torch_geometric.utils.scatter
returns 0, not NaN, for empty groups) — the prediction then rests on its
intrinsic structure/ADMET encoding alone, exactly as it should for a novel
compound with no recorded assay history.

Usage:
    from hetero_gnn.predict import load_predictor, predict_smiles

    predictor = load_predictor()
    df = predict_smiles(predictor, "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5")
    print(df.sort_values("pki_pred", ascending=False))
"""
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hetero_gnn.config import (
    BINDING_SITES, CHECKPOINT_DIR, COMPOUND_BUILDER_PATH, COMPOUND_NUMERIC_COLS,
    ENDPOINT_TYPES, PHARM_ROLES, QUALIFIERS, TARGET_REFERENCE_PATH,
)
from hetero_gnn.dataset import build_dataset, BINDS_TO
from hetero_gnn.model import build_model
from hetero_gnn.preprocess import _one_hot
import numpy as np


def load_predictor(ckpt_path: str | None = None) -> dict:
    """Load the trained model, dataset, and fitted compound feature builder.

    Returns a dict with: model, data, meta, device, compound_builder.
    """
    data, meta = build_dataset(verbose=False)
    model = build_model(data, edge_dim=data[BINDS_TO].edge_attr.shape[1])

    ckpt = ckpt_path or os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint found at {ckpt}. Run run_train.py first.")
    if not os.path.exists(COMPOUND_BUILDER_PATH):
        raise FileNotFoundError(
            f"No fitted compound feature builder at {COMPOUND_BUILDER_PATH}. "
            "Run run_preprocess.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    data = data.to(device)
    model = model.to(device)
    compound_builder = torch.load(COMPOUND_BUILDER_PATH, weights_only=False)

    return dict(model=model, data=data, meta=meta, device=device, compound_builder=compound_builder)


def predict_smiles(predictor: dict, smiles: str) -> pd.DataFrame:
    """
    Predict pKi for a single SMILES against every target node in the graph,
    assuming a default measurement context (Ki, '=', in vitro-typical —
    i.e. an all-"other"/empty edge feature vector except endpoint='Ki' and
    qualifier='=', which is the only context a not-yet-measured compound can
    plausibly be scored under).

    Returns a DataFrame with columns: target_idx, pki_pred, sorted descending.
    """
    model = predictor["model"]
    data = predictor["data"]
    device = predictor["device"]
    builder = predictor["compound_builder"]

    # CompoundFeatureBuilder.transform reads COMPOUND_NUMERIC_COLS by name;
    # a query compound has no ADMET data, so every numeric column is left
    # missing and gets imputed to the (fitted) training column means.
    compound_row = pd.DataFrame([{"smiles": smiles, **{c: float("nan") for c in COMPOUND_NUMERIC_COLS}}])
    compound_x = builder.transform(compound_row).to(device)   # [1, F_c]

    n_targets = data["target"].num_nodes

    orig_x = data["compound"].x
    orig_nodes = data["compound"].num_nodes
    data["compound"].x = torch.cat([orig_x, compound_x], dim=0)
    data["compound"].num_nodes = orig_nodes + 1

    query_idx = torch.full((n_targets,), orig_nodes, dtype=torch.long, device=device)
    dst = torch.arange(n_targets, dtype=torch.long, device=device)
    edge_index = torch.stack([query_idx, dst], dim=0)

    ep_oh = _one_hot("Ki", ENDPOINT_TYPES, other_bucket=True)
    q_oh = _one_hot("=", QUALIFIERS, other_bucket=True)
    role_oh = _one_hot(None, PHARM_ROLES, other_bucket=False)
    site_oh = _one_hot(None, BINDING_SITES, other_bucket=False)
    edge_attr_row = np.concatenate([ep_oh, q_oh, role_oh, site_oh])
    edge_attr = torch.from_numpy(
        np.tile(edge_attr_row, (n_targets, 1)).astype(np.float32)
    ).to(device)

    with torch.no_grad():
        preds = model(data, edge_index=edge_index, edge_attr=edge_attr)

    data["compound"].x = orig_x
    data["compound"].num_nodes = orig_nodes

    results = pd.DataFrame({
        "target_idx": np.arange(n_targets),
        "pki_pred": preds.cpu().numpy(),
    })
    if os.path.exists(TARGET_REFERENCE_PATH):
        ref = pd.read_csv(TARGET_REFERENCE_PATH, index_col="target_idx")
        results = results.join(ref[["name", "taxonomy"]], on="target_idx")
    return results.sort_values("pki_pred", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    morphine = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
    print("Loading model …")
    predictor = load_predictor()
    print("Predicting for morphine …\n")
    df = predict_smiles(predictor, morphine)
    print(df.to_string(index=False))
