"""
Load processed_bipartite_graph.pt and adapt it for the training pipeline.

The preprocessed file uses two separate mask conventions:
  train_mask  — supervision mask: True where pChEMBL is exact and in-range
  split_train / split_val / split_test — temporal partition masks

This module combines them so downstream code (train.py, model.py) sees the
familiar train_mask / val_mask / test_mask interface, where each mask selects
edges that are both in the correct temporal partition AND have a valid label.

Message passing always uses ALL edges (including censored ones) to maximise
graph context; the loss is computed only on the supervised subset.
"""
import os

import torch
from torch_geometric.data import HeteroData

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH = os.path.join(ROOT, "processed_bipartite_graph.pt")


def build_dataset(verbose: bool = True):
    """
    Load the preprocessed bipartite graph and return (data, meta).

    data["compound"].x          FloatTensor [N_c, 2060]
    data["target"].x            FloatTensor [N_t,   13]
    data["target"].node_idx     LongTensor  [N_t]        for learned embedding
    data[et].edge_index         LongTensor  [2, E]
    data[et].edge_attr          FloatTensor [E, 16]
    data[et].edge_label         FloatTensor [E]          pChEMBL (0 for censored)
    data[et].train_mask         BoolTensor  [E]          temporal-train ∩ supervised
    data[et].val_mask           BoolTensor  [E]          temporal-val  ∩ supervised
    data[et].test_mask          BoolTensor  [E]          temporal-test ∩ supervised
    """
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(
            f"Preprocessed graph not found: {GRAPH_PATH}\n"
            "Run:  conda run -n odo python3 preprocess_bipartite_graph.py"
        )

    if verbose:
        print(f"  Loading {os.path.basename(GRAPH_PATH)} …")

    data = torch.load(GRAPH_PATH, weights_only=False)
    et = ("compound", "activity", "target")

    # Target embedding index (0 … N_t-1) required by OpioidGNN
    data["target"].node_idx = torch.arange(
        data["target"].num_nodes, dtype=torch.long
    )

    # Supervision mask from preprocessing (True = exact pChEMBL label)
    supervision = data[et].train_mask

    # Replace NaN labels with 0 — masked out by train/val/test_mask anyway
    labels = data[et].edge_label.clone()
    labels[torch.isnan(labels)] = 0.0
    data[et].edge_label = labels

    # Combine temporal split with supervision into the standard mask names
    data[et].train_mask = data[et].split_train & supervision
    data[et].val_mask   = data[et].split_val   & supervision
    data[et].test_mask  = data[et].split_test  & supervision

    # Per-edge temporal weights: recent experiments weighted up to 2×, oldest 0.5×
    years = data[et].doc_year.clone()
    valid = ~torch.isnan(years)
    weights = torch.ones(years.shape[0])
    min_y, max_y = 1977.0, 2015.0
    weights[valid] = 0.5 + 1.5 * (years[valid] - min_y) / (max_y - min_y)
    weights = weights.clamp(0.5, 2.0)
    data[et].sample_weight = weights

    if verbose:
        E = data[et].edge_index.shape[1]
        print(f"  Compounds   : {data['compound'].num_nodes:,}")
        print(f"  Targets     : {data['target'].num_nodes}")
        print(f"  Total edges : {E:,}  (all experiments, incl. censored)")
        print(f"  Train edges : {data[et].train_mask.sum().item():,}  (supervised)")
        print(f"  Val   edges : {data[et].val_mask.sum().item():,}  (supervised)")
        print(f"  Test  edges : {data[et].test_mask.sum().item():,}  (supervised)")

    meta = {
        "graph_path":  GRAPH_PATH,
        "cutoff_year": 2015,
    }
    return data, meta


def get_edge_split(data: HeteroData, mask_name: str):
    """Return (edge_index, edge_attr, edge_label) for a given split mask."""
    et = ("compound", "activity", "target")
    mask = getattr(data[et], mask_name)
    return (
        data[et].edge_index[:, mask],
        data[et].edge_attr[mask],
        data[et].edge_label[mask],
    )


if __name__ == "__main__":
    data, meta = build_dataset(verbose=True)
    print("\nHeteroData summary:")
    print(data)
