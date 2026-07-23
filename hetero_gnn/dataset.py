"""
Load processed_hetero_graph.pt and expose the train/val/test interface used
by train.py / predict.py.

§1 defines two loss components (exact MSE, censored one-sided hinge), each
restricted to endpoint=='Ki' with a resolvable qualifier. preprocess.py
stores the three base eligibility masks (exact_mask, hinge_floor_mask,
hinge_ceiling_mask) un-intersected with the temporal split; this module
combines each with split_train/split_val/split_test into the 9 masks
train.py consumes:

    {train,val,test}_exact_mask     -> MSE term
    {train,val,test}_hinge_floor_mask    -> one-sided hinge, floor
    {train,val,test}_hinge_ceiling_mask  -> one-sided hinge, ceiling

Message passing always sees ALL binds_to + structural edges; only the loss
computation is gated by these masks.
"""
import os
import sys

import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hetero_gnn.config import GRAPH_PATH, TEMPORAL_CUTOFF_YEAR

BINDS_TO = ("compound", "binds_to", "target")
_SPLITS = ("train", "val", "test")
_COMPONENTS = ("exact_mask", "hinge_floor_mask", "hinge_ceiling_mask")


def build_dataset(verbose: bool = True, graph_path: str | None = None):
    """Load the preprocessed heterogeneous graph and return (data, meta)."""
    path = graph_path or GRAPH_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessed graph not found: {path}\n"
            "Run:  conda run -n odo python3 hetero_gnn/run_preprocess.py"
        )

    if verbose:
        print(f"  Loading {os.path.basename(path)} …")

    data = torch.load(path, weights_only=False)
    et = data[BINDS_TO]

    # Replace NaN labels with 0 — always masked out by the loss masks anyway
    labels = et.edge_label.clone()
    labels[torch.isnan(labels)] = 0.0
    et.edge_label = labels

    split_mask = {"train": et.split_train, "val": et.split_val, "test": et.split_test}
    for split in _SPLITS:
        for component in _COMPONENTS:
            setattr(et, f"{split}_{component}", split_mask[split] & getattr(et, component))

    if verbose:
        E = et.edge_index.shape[1]
        print(f"  Compounds     : {data['compound'].num_nodes:,}")
        print(f"  Targets       : {data['target'].num_nodes}")
        print(f"  Assays        : {data['assay'].num_nodes:,}")
        print(f"  Model systems : {data['model_system'].num_nodes:,}")
        print(f"  Documents     : {data['document'].num_nodes:,}")
        print(f"  Total binds_to edges : {E:,}  (all activities, incl. censored/context-only)")
        for split in _SPLITS:
            n_exact = getattr(et, f"{split}_exact_mask").sum().item()
            n_floor = getattr(et, f"{split}_hinge_floor_mask").sum().item()
            n_ceil = getattr(et, f"{split}_hinge_ceiling_mask").sum().item()
            print(f"  {split:5s} edges : exact={n_exact:,}  hinge_floor={n_floor:,}  hinge_ceiling={n_ceil:,}")

    meta = {"graph_path": path, "cutoff_year": TEMPORAL_CUTOFF_YEAR}
    return data, meta


def get_edge_split(data: HeteroData, split: str):
    """Return (edge_index, edge_attr, edge_label, exact_mask, floor_mask,
    ceiling_mask) for a given split ('train' | 'val' | 'test'), restricted
    to loss-eligible edges (exact | hinge_floor | hinge_ceiling)."""
    et = data[BINDS_TO]
    mask = (
        getattr(et, f"{split}_exact_mask")
        | getattr(et, f"{split}_hinge_floor_mask")
        | getattr(et, f"{split}_hinge_ceiling_mask")
    )
    return (
        et.edge_index[:, mask],
        et.edge_attr[mask],
        et.edge_label[mask],
        getattr(et, f"{split}_exact_mask")[mask],
        getattr(et, f"{split}_hinge_floor_mask")[mask],
        getattr(et, f"{split}_hinge_ceiling_mask")[mask],
    )


if __name__ == "__main__":
    data, meta = build_dataset(verbose=True)
    print("\nHeteroData summary:")
    print(data)
