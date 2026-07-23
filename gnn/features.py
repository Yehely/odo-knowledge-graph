"""
Feature engineering for the ODO GNN.

  build_compound_features(compounds_df) → FloatTensor [N_c, F_c]
  build_target_features(targets_df)    → (FloatTensor [N_t, F_t_cat],
                                          LongTensor  [N_t]  target_idx for embedding)
  build_edge_features(activities_df,
                      compound_index, target_index) → FloatTensor [N_e, F_e]

Compound feature layout:
  [0 : MORGAN_BITS]           Morgan fingerprint (ECFP4, 2048-bit)
  [MORGAN_BITS : +10]         ADMET numeric (StandardScaler-normalised)
  [-2]                        maxPhase / 4  (normalised 0-1)
  [-1]                        isRadiolabeled (0/1)

Target feature layout (categorical one-hots only; embedding added in model):
  [0 : 4]   targetType one-hot
  [4 : 17]  targetSpecies one-hot

Edge feature layout:
  [0 : 7]   endpointType one-hot
  [7 : 13]  qualifier one-hot
  [13: 16]  experimentalSetting one-hot
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gnn.config import (
    MORGAN_BITS, MORGAN_RADIUS, ADMET_COLS,
    ENDPOINT_TYPES, QUALIFIERS, EXP_SETTINGS,
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False
    print("[WARNING] RDKit not available — Morgan fingerprints will be zero vectors.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _morgan(smiles: str | None) -> np.ndarray:
    """Return 2048-bit Morgan fingerprint as float array, or zeros if invalid."""
    fp = np.zeros(MORGAN_BITS, dtype=np.float32)
    if not _RDKIT or not smiles:
        return fp
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fp
    bit_vec = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    fp[:] = np.frombuffer(bit_vec.ToBitString().encode(), dtype=np.uint8) - ord("0")
    return fp


def _one_hot(value: str, categories: list[str]) -> np.ndarray:
    vec = np.zeros(len(categories), dtype=np.float32)
    v = str(value).strip().lower() if value else "other"
    for i, cat in enumerate(categories):
        if cat.lower() == v:
            vec[i] = 1.0
            return vec
    vec[-1] = 1.0  # "other" bucket is always last
    return vec


# ---------------------------------------------------------------------------
# Compound features
# ---------------------------------------------------------------------------

_ADMET_RAW = [
    "mw", "alogp", "ro5_violations",
    "qp_logpow", "qp_logs", "qp_donor_hb", "qp_acceptor_hb",
    "qp_logkhsa", "human_oral_absorption", "qp_sasa",
]

class CompoundFeatureBuilder:
    """Fits a StandardScaler on ADMET columns; call build() to get the tensor.

    Missing ADMET values are imputed with the column mean of the training set
    (fit_transform), so N/A compounds appear at the centre of each feature
    distribution rather than at an arbitrary value like 0.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._col_means: np.ndarray | None = None

    def fit_transform(self, compounds_df: pd.DataFrame) -> torch.FloatTensor:
        raw = compounds_df[_ADMET_RAW].values.astype(np.float64)
        # Compute column means from non-NaN training values for imputation
        self._col_means = np.nanmean(raw, axis=0)
        # Impute NaN with column mean before scaling
        nan_mask = np.isnan(raw)
        raw[nan_mask] = np.take(self._col_means, np.where(nan_mask)[1])
        admet_norm = self.scaler.fit_transform(raw).astype(np.float32)
        return self._assemble(compounds_df, admet_norm)

    def transform(self, compounds_df: pd.DataFrame) -> torch.FloatTensor:
        raw = compounds_df[_ADMET_RAW].values.astype(np.float64)
        # Impute with training means
        nan_mask = np.isnan(raw)
        raw[nan_mask] = np.take(self._col_means, np.where(nan_mask)[1])
        admet_norm = self.scaler.transform(raw).astype(np.float32)
        return self._assemble(compounds_df, admet_norm)

    def _assemble(self, df: pd.DataFrame, admet_norm: np.ndarray) -> torch.FloatTensor:
        n = len(df)
        fingerprints = np.stack([_morgan(s) for s in df["smiles"]], axis=0)  # [N, 2048]
        max_phase    = (df["max_phase"].fillna(0.0).values / 4.0).astype(np.float32).reshape(-1, 1)
        radiolabeled = df["is_radiolabeled"].fillna(0.0).values.astype(np.float32).reshape(-1, 1)

        feats = np.concatenate([fingerprints, admet_norm, max_phase, radiolabeled], axis=1)
        return torch.from_numpy(feats)


# ---------------------------------------------------------------------------
# Target features  (vocabularies built from actual data, not hardcoded)
# ---------------------------------------------------------------------------

def _build_vocab(series: "pd.Series") -> list:
    """
    Sorted list of unique non-null values, with 'other'/'Unknown' always last.
    Used so the one-hot encoding covers exactly what is in the data.
    """
    vals = sorted(v for v in series.dropna().unique()
                  if v not in ("other", "Unknown", "unspecified"))
    vals.append("other")   # catch-all bucket (unspecified → other via _one_hot)
    return vals


def build_target_features(targets_df: pd.DataFrame):
    """
    Build one-hot target features from the *actual* values present in targets_df.
    Vocabularies are derived from the data, never hardcoded, so no species or
    type is silently dropped into the 'other' bucket.

    Returns:
        cat_feats  : FloatTensor [N_t, F_t]  — one-hot type + species
        name_idx   : LongTensor  [N_t]       — 0..N_t-1 for learned embedding
        target_types   : list[str]           — vocabulary used for type
        target_species : list[str]           — vocabulary used for species
    """
    target_types   = _build_vocab(targets_df["type"])
    target_species = _build_vocab(targets_df["species"])

    rows = []
    for _, row in targets_df.iterrows():
        type_oh    = _one_hot(row["type"],    target_types)
        species_oh = _one_hot(row["species"], target_species)
        rows.append(np.concatenate([type_oh, species_oh]))

    cat_feats = torch.from_numpy(np.stack(rows).astype(np.float32))
    name_idx  = torch.arange(len(targets_df), dtype=torch.long)
    return cat_feats, name_idx, target_types, target_species


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------

def build_edge_features(
    activities_df: pd.DataFrame,
    compound_uri_to_idx: dict,
    target_uri_to_idx: dict,
):
    """
    Build edge_index, edge_attr, and edge_label (pChEMBL) tensors.

    Returns:
        edge_index : LongTensor  [2, E]  — (compound_idx, target_idx) per activity
        edge_attr  : FloatTensor [E, 16] — endpoint type + qualifier + exp setting
        edge_label : FloatTensor [E]     — pChEMBL values
        valid_mask : boolean array [E]   — True if both compound + target present
    """
    n = len(activities_df)
    src_idx = np.full(n, -1, dtype=np.int64)
    dst_idx = np.full(n, -1, dtype=np.int64)
    attrs   = []
    labels  = np.empty(n, dtype=np.float32)

    for i, row in activities_df.iterrows():
        c_idx = compound_uri_to_idx.get(row["compound_uri"], -1)
        t_idx = target_uri_to_idx.get(row["target_uri"],   -1)
        src_idx[i] = c_idx
        dst_idx[i] = t_idx

        ep_oh   = _one_hot(row["endpoint_type"], ENDPOINT_TYPES)
        qual_oh = _one_hot(row["qualifier"],     QUALIFIERS)
        set_oh  = _one_hot(row["exp_setting"],   EXP_SETTINGS)
        attrs.append(np.concatenate([ep_oh, qual_oh, set_oh]))
        labels[i] = float(row["pchembl"])

    attrs = np.stack(attrs).astype(np.float32)

    valid = (src_idx >= 0) & (dst_idx >= 0)
    edge_index = torch.from_numpy(np.stack([src_idx[valid], dst_idx[valid]], axis=0))
    edge_attr  = torch.from_numpy(attrs[valid])
    edge_label = torch.from_numpy(labels[valid])
    return edge_index, edge_attr, edge_label
