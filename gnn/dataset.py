"""
Build a PyTorch Geometric HeteroData object from the ODO knowledge graph.

Split strategy: temporal (by publication year)
  Train : activities from papers published before TEMPORAL_CUTOFF_YEAR (< 2015)
            → ~79% of the corpus; a random VAL_FRAC_OF_TRAIN subset is held out for val
  Val   : random 10% of the pre-cutoff activities
  Test  : activities from papers published in TEMPORAL_CUTOFF_YEAR or later (>= 2015)
            → ~21% of the corpus (the "future" the model must generalise to)
  No-year: activities with no publication year are assigned to train.

Graph schema:
  Node types: "compound", "target"
  Edge type:  ("compound", "activity", "target")    — directed
              ("target",   "rev_activity", "compound") — reverse for message passing
"""
import os
import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import SEED, TEMPORAL_CUTOFF_YEAR, VAL_FRAC_OF_TRAIN
from gnn.data_loader import load_all
from gnn.features import (
    CompoundFeatureBuilder,
    build_edge_features,
    build_target_features,
)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(verbose: bool = True):
    """
    Load data, engineer features, and return a HeteroData object with splits.

    Returns
    -------
    data : HeteroData
        data["compound"].x                               FloatTensor [N_c, F_c]
        data["target"].x                                 FloatTensor [N_t, F_t_cat]
        data["target"].node_idx                          LongTensor  [N_t]
        data[et].edge_index                              LongTensor  [2, E]
        data[et].edge_attr                               FloatTensor [E, F_e]
        data[et].edge_label                              FloatTensor [E]  pChEMBL
        data[et].train_mask / val_mask / test_mask       BoolTensor  [E]
        data[et].doc_year                                FloatTensor [E]  (NaN = unknown)
    meta : dict
        compound_uri_to_idx, target_uri_to_idx,
        compounds_df, targets_df, activities_df,
        feature_builder, cutoff_year
    """
    compounds_df, targets_df, activities_df = load_all()

    if verbose:
        print(f"  Compounds in activities: {len(compounds_df):,}")
        print(f"  Targets   in activities: {len(targets_df):,}")
        print(f"  Activities (filtered)  : {len(activities_df):,}")

    # Index maps URI -> integer index
    compound_uri_to_idx = {uri: i for i, uri in enumerate(compounds_df["uri"])}
    target_uri_to_idx   = {uri: i for i, uri in enumerate(targets_df["uri"])}

    # -----------------------------------------------------------------------
    # Node features
    # -----------------------------------------------------------------------
    feat_builder = CompoundFeatureBuilder()
    compound_x   = feat_builder.fit_transform(compounds_df)   # [N_c, F_c]

    target_cat, target_idx, target_types, target_species = build_target_features(
        targets_df
    )  # [N_t, F_t], [N_t]

    if verbose:
        print(f"  Compound feature dim : {compound_x.shape[1]}")
        print(f"  Target   feature dim : {target_cat.shape[1]}")
        print(f"    targetType  vocab  : {target_types}")
        print(f"    targetSpecies vocab: {target_species}")

    # -----------------------------------------------------------------------
    # Edge features & labels
    # -----------------------------------------------------------------------
    edge_index, edge_attr, edge_label = build_edge_features(
        activities_df, compound_uri_to_idx, target_uri_to_idx
    )
    E = edge_index.shape[1]

    if verbose:
        print(f"  Edge (activity) count: {E:,}")
        print(f"  Edge feature dim     : {edge_attr.shape[1]}")

    # -----------------------------------------------------------------------
    # Temporal split
    # -----------------------------------------------------------------------
    doc_years = activities_df["doc_year"].values  # float or NaN

    # Test: year >= cutoff  |  pre-cutoff or unknown: candidate for train/val
    test_mask_np  = np.array(
        [y is not None and not np.isnan(y) and y >= TEMPORAL_CUTOFF_YEAR
         for y in doc_years],
        dtype=bool,
    )
    pre_cutoff_idx = np.where(~test_mask_np)[0]   # indices that are NOT test

    # Split pre-cutoff randomly into train and val
    rng    = np.random.default_rng(SEED)
    perm   = rng.permutation(len(pre_cutoff_idx))
    n_val  = max(1, int(len(pre_cutoff_idx) * VAL_FRAC_OF_TRAIN))

    val_pos   = set(pre_cutoff_idx[perm[:n_val]].tolist())
    train_pos = set(pre_cutoff_idx[perm[n_val:]].tolist())

    train_mask_np = np.array([i in train_pos for i in range(E)], dtype=bool)
    val_mask_np   = np.array([i in val_pos   for i in range(E)], dtype=bool)

    train_mask = torch.from_numpy(train_mask_np)
    val_mask   = torch.from_numpy(val_mask_np)
    test_mask  = torch.from_numpy(test_mask_np)

    # Store doc_year as a tensor (NaN for unknowns)
    doc_year_tensor = torch.tensor(
        [float(y) if (y is not None and not np.isnan(y)) else float("nan")
         for y in doc_years],
        dtype=torch.float32,
    )

    if verbose:
        n_no_year = int((~test_mask_np & ~np.array(
            [y is not None and not np.isnan(y) for y in doc_years]
        )).sum())
        print(f"\n  Temporal split (cutoff = {TEMPORAL_CUTOFF_YEAR}):")
        print(f"    Train edges : {train_mask.sum().item():,}")
        print(f"    Val   edges : {val_mask.sum().item():,}")
        print(f"    Test  edges : {test_mask.sum().item():,}  (year >= {TEMPORAL_CUTOFF_YEAR})")
        if n_no_year:
            print(f"    No-year     : {n_no_year:,}  → assigned to train")

        # Year range sanity
        train_years = [y for y, m in zip(doc_years, train_mask_np)
                       if m and y is not None and not np.isnan(y)]
        test_years  = [y for y, m in zip(doc_years, test_mask_np)
                       if m and y is not None and not np.isnan(y)]
        if train_years:
            print(f"    Train year range: {int(min(train_years))}–{int(max(train_years))}")
        if test_years:
            print(f"    Test  year range: {int(min(test_years))}–{int(max(test_years))}")

    # -----------------------------------------------------------------------
    # Assemble HeteroData
    # -----------------------------------------------------------------------
    data = HeteroData()

    data["compound"].x         = compound_x
    data["compound"].num_nodes = len(compounds_df)

    data["target"].x         = target_cat
    data["target"].node_idx  = target_idx
    data["target"].num_nodes = len(targets_df)

    et = ("compound", "activity", "target")
    data[et].edge_index  = edge_index
    data[et].edge_attr   = edge_attr
    data[et].edge_label  = edge_label
    data[et].doc_year    = doc_year_tensor
    data[et].train_mask  = train_mask
    data[et].val_mask    = val_mask
    data[et].test_mask   = test_mask

    # Reverse edges so targets can send messages back to compounds
    rev_et = ("target", "rev_activity", "compound")
    data[rev_et].edge_index = edge_index.flip(0)
    data[rev_et].edge_attr  = edge_attr

    meta = dict(
        compound_uri_to_idx = compound_uri_to_idx,
        target_uri_to_idx   = target_uri_to_idx,
        compounds_df        = compounds_df,
        targets_df          = targets_df,
        activities_df       = activities_df,
        feature_builder     = feat_builder,
        cutoff_year         = TEMPORAL_CUTOFF_YEAR,
        target_types        = target_types,
        target_species      = target_species,
    )
    return data, meta


# ---------------------------------------------------------------------------
# Edge-level split helper
# ---------------------------------------------------------------------------

def get_edge_split(data: HeteroData, mask_name: str):
    """Return (edge_index, edge_attr, edge_label) for a given split mask."""
    et = ("compound", "activity", "target")
    mask = getattr(data[et], mask_name)
    return (
        data[et].edge_index[:, mask],
        data[et].edge_attr[mask],
        data[et].edge_label[mask],
    )


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data, meta = build_dataset(verbose=True)
    print("\nHeteroData summary:")
    print(data)
