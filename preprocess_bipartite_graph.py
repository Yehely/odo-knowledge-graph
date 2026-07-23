"""
preprocess_bipartite_graph.py
─────────────────────────────
Standalone preprocessing pipeline: reads the ODO Excel dataset directly and
outputs a PyTorch Geometric HeteroData bipartite graph object ready for a
GraphSAGE model.

Run:
    conda run -n odo python3 preprocess_bipartite_graph.py

Output: processed_bipartite_graph.pt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Graph schema
  Node types : 'compound'  [N_c, 2060]
               'target'    [N_t,   13]   (+ 32-dim learned embedding in model)
  Edge type  : ('compound', 'activity', 'target')  [E, 16]
  Rev. edges : ('target', 'rev_activity', 'compound')  for bidirectional MP

Compound feature layout  (2060 dims):
  [0    : 2048]  Morgan fingerprint  ECFP4, radius=2
  [2048 : 2058]  10 ADMET properties (StandardScaler-normalised; NaN → column mean)
  [2058]         maxPhase / 4        (0–1 normalised)
  [2059]         isRadiolabeled      (0/1 binary)

Target feature layout  (13 static dims):
  [0 : 4]   targetType one-hot   (4 fixed categories)
  [4 : 13]  targetSpecies one-hot (9 fixed categories)
  The remaining 32-dim learned embedding is initialised inside the model.

Edge feature layout  (16 dims):
  [0  : 7]   endpointType one-hot   Ki / IC50 / EC50 / Inhibition / Activity / Binding / other
  [7  : 13]  qualifier one-hot      = / < / > / <= / >= / other
  [13 : 16]  experimentalSetting OHE in vitro / in vivo / other

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Censored-data masking strategy
  All experiments are kept as edges to preserve full graph topology.

  data[et].train_mask[i] = True   iff qualifier == '=' AND pChEMBL ∈ [2, 15]
                         = False  for censored (>, <, >=, <=) or missing pChEMBL

  data[et].edge_label[i] = pChEMBL value  for labelled edges
                          = NaN           for censored/missing edges
  Training loops use train_mask to gate the supervised loss.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temporal split  (by document/experiment publication year)
  split_train: doc_year <  2015  (or no year → assigned to train)
  split_val  : random 10 % of the train partition
  split_test : 2015 ≤ doc_year ≤ 2020
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False
    print(
        "[WARNING] RDKit not available — Morgan fingerprints will be zero vectors.",
        file=sys.stderr,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ROOT        = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH  = os.path.join(ROOT, "Final ODO Dataset_v2026-06-10.xlsx")
OUTPUT_PATH = os.path.join(ROOT, "processed_bipartite_graph.pt")

MORGAN_BITS   = 512    # reduced from 2048 to curb memorization
MORGAN_RADIUS = 2
SPLIT_MODE    = "compound_random"  # "temporal" | "random" | "compound_random"

# Internal DataFrame column names for the 10 ADMET properties
ADMET_COLS = [
    "mw", "alogp", "ro5_violations",
    "qp_logpow", "qp_logs", "qp_donor_hb", "qp_acceptor_hb",
    "qp_logkhsa", "human_oral_absorption", "qp_sasa",
]

# Fixed one-hot vocabularies — last entry is always the "other" catch-all bucket.
# Sizes: 4 + 9 = 13 static target dims as per architectural spec.
TARGET_TYPES = [
    "SINGLE PROTEIN",
    "PROTEIN COMPLEX",
    "PROTEIN FAMILY",
    "other",           # catch-all for SELECTIVITY GROUP, ORGANISM, etc.
]

TARGET_SPECIES = [
    "Homo sapiens",
    "Mus musculus",
    "Rattus norvegicus",
    "Cavia porcellus",
    "Ovis aries",
    "Sus scrofa",
    "Chlorocebus sabaeus",
    "Bos taurus",
    "other",           # catch-all for any unlisted species
]

# Edge feature vocabularies (7 + 6 + 3 = 16 dims)
ENDPOINT_TYPES = ["Ki", "IC50", "EC50", "Inhibition", "Activity", "Binding", "other"]
QUALIFIERS     = ["=", "<", ">", "<=", ">=", "other"]
EXP_SETTINGS   = ["in vitro", "in vivo", "other"]

MIN_PCHEMBL       = 2.0
MAX_PCHEMBL       = 15.0
TEMPORAL_CUTOFF   = 2016   # first year of test partition (train ≤ 2015)
TEST_MAX_YEAR     = 2020   # last year of test partition (inclusive)
VAL_FRAC_OF_TRAIN = 0.10
SEED              = 42


# ──────────────────────────────────────────────────────────────────────────────
# Low-level utilities
# ──────────────────────────────────────────────────────────────────────────────

def safe(val) -> str | None:
    """Return str(val) if not blank/NaN/None, else None."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(text).strip())


def to_float(val) -> float:
    """Parse val to float; return NaN on failure."""
    v = safe(val)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def one_hot(value: str | None, categories: list[str]) -> np.ndarray:
    """One-hot encode value against categories; unknown values → last bucket."""
    vec = np.zeros(len(categories), dtype=np.float32)
    v   = str(value).strip() if value else "other"
    for i, cat in enumerate(categories):
        if cat.lower() == v.lower():
            vec[i] = 1.0
            return vec
    vec[-1] = 1.0   # "other" is always last
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Raw data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Read the ODO Excel workbook and return the 'Full Dataset' sheet."""
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")
    print(f"Reading {os.path.basename(EXCEL_PATH)} …")
    df = pd.read_excel(EXCEL_PATH, sheet_name="Full Dataset")
    print(f"  {len(df):,} rows × {len(df.columns)} columns loaded")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Entity-table builders
# ──────────────────────────────────────────────────────────────────────────────

def build_compound_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by ChEMBL compound ID → InChIKey fallback.
    Returns one row per unique compound with SMILES + all feature columns.
    """
    seen: set[str] = set()
    rows = []

    for _, row in raw.iterrows():
        cid  = safe(row.get("chembl_compound_id"))
        ikey = safe(row.get("rdkit_library_standard_inchi_key"))
        key  = cid or (f"inchikey_{ikey}" if ikey else None)
        if not key or key in seen:
            continue
        seen.add(key)

        rl = safe(row.get("reference_radiolabeled_molecular_entity"))
        rows.append({
            "compound_key":      key,
            "smiles":            safe(row.get("rdkit_canonical_smiles")),
            # 10 ADMET properties (mapped to internal names used by ADMET_COLS)
            "mw":                to_float(row.get("rdkit_molecular_weight")),
            "alogp":             to_float(row.get("chembl_alogp")),
            "ro5_violations":    to_float(row.get("chembl_#ro5_violations")),
            "qp_logpow":         to_float(row.get("qikprop_qplog_po/w")),
            "qp_logs":           to_float(row.get("qikprop_qplogs")),
            "qp_donor_hb":       to_float(row.get("qikprop_donor_hb")),
            "qp_acceptor_hb":    to_float(row.get("qikprop_accpt_hb")),
            "qp_logkhsa":        to_float(row.get("qikprop_qplog_khsa")),
            "human_oral_absorption": to_float(row.get("qikprop_percent_human_oral_absorption")),
            "qp_sasa":           to_float(row.get("qikprop_sasa")),
            # Clinical / radiolabeling features
            "max_phase":         to_float(row.get("chembl_molecule_max_phase")),
            "is_radiolabeled":   1.0 if rl and rl.lower() in ("true", "1", "yes", "t") else 0.0,
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  {len(df):,} unique compounds")
    return df


def build_target_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by ChEMBL target ID → target name fallback.
    Returns one row per unique target with type and species columns.
    """
    seen: set[str] = set()
    rows = []

    for _, row in raw.iterrows():
        tcid  = safe(row.get("chembl_target_id"))
        tname = safe(row.get("target_name"))
        key   = tcid or (slugify(tname) if tname else None)
        if not key or key in seen:
            continue
        seen.add(key)

        taxon = safe(row.get("ncbi_target_taxonomy"))
        if taxon:
            parts   = taxon.strip().split()
            species = (
                parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])
                if len(parts) >= 2 else taxon.strip()
            )
        else:
            species = "other"

        rows.append({
            "target_key": key,
            "name":        tname or "Unknown",
            "type":        safe(row.get("target_type")) or "other",
            "species":     species,
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  {len(df):,} unique targets")
    return df


def build_activity_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per experiment — ALL activities, including censored/missing pChEMBL.

    Columns added:
      is_exact : bool   True iff qualifier == '=' and pChEMBL in [MIN, MAX]
      pchembl  : float  numeric value for exact rows; NaN for censored/missing
    """
    rows = []

    for _, row in raw.iterrows():
        cid  = safe(row.get("chembl_compound_id"))
        ikey = safe(row.get("rdkit_library_standard_inchi_key"))
        c_key = cid or (f"inchikey_{ikey}" if ikey else None)

        tcid  = safe(row.get("chembl_target_id"))
        tname = safe(row.get("target_name"))
        t_key = tcid or (slugify(tname) if tname else None)

        if not c_key or not t_key:
            continue

        qual = safe(row.get("endpoint_qualifier")) or ""
        pch  = to_float(row.get("pchembl_value"))

        is_exact = (
            qual == "="
            and not np.isnan(pch)
            and MIN_PCHEMBL <= pch <= MAX_PCHEMBL
        )

        rows.append({
            "compound_key":  c_key,
            "target_key":    t_key,
            "pchembl":       pch if is_exact else float("nan"),
            "qualifier":     qual,
            "endpoint_type": safe(row.get("endpoint")) or "other",
            "exp_setting":   safe(row.get("bao_experimental_setting")) or "other",
            "doc_year":      to_float(row.get("document_year")),
            "is_exact":      is_exact,
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    n_exact = int(df["is_exact"].sum())
    print(
        f"  {len(df):,} activities  "
        f"({n_exact:,} exact / {len(df) - n_exact:,} censored or missing pChEMBL)"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Compound feature engineering  →  [N_c, 2060]
# ──────────────────────────────────────────────────────────────────────────────

def _morgan_fp(smiles: str | None) -> np.ndarray:
    """Return MORGAN_BITS-bit ECFP4 as float32; zero vector for invalid/missing SMILES."""
    fp = np.zeros(MORGAN_BITS, dtype=np.float32)
    if not _RDKIT or not smiles:
        return fp
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fp
    bv  = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    fp[:] = np.frombuffer(bv.ToBitString().encode(), dtype=np.uint8) - ord("0")
    return fp


def build_compound_features(compounds_df: pd.DataFrame) -> torch.FloatTensor:
    """
    Returns FloatTensor [N_c, MORGAN_BITS + 12].
    NaN ADMET values are imputed with the column mean before StandardScaler.
    All-NaN columns (degenerate case) fall back to 0 after imputation.
    """
    n = len(compounds_df)
    print(f"  Computing Morgan fingerprints for {n:,} compounds …")
    fps = np.stack([_morgan_fp(s) for s in compounds_df["smiles"]], axis=0)   # [N, MORGAN_BITS]

    raw_admet  = compounds_df[ADMET_COLS].values.astype(np.float64)           # [N, 10]
    col_means  = np.nanmean(raw_admet, axis=0)
    col_means  = np.nan_to_num(col_means, nan=0.0)   # guard against all-NaN columns
    nan_pos    = np.where(np.isnan(raw_admet))
    raw_admet[nan_pos] = np.take(col_means, nan_pos[1])

    scaler     = StandardScaler()
    admet_norm = scaler.fit_transform(raw_admet).astype(np.float32)            # [N, 10]

    max_phase  = (compounds_df["max_phase"].fillna(0.0).values / 4.0).astype(np.float32)
    is_radio   = compounds_df["is_radiolabeled"].fillna(0.0).values.astype(np.float32)

    x = np.concatenate(
        [fps, admet_norm, max_phase[:, None], is_radio[:, None]], axis=1
    )  # [N, MORGAN_BITS + 12]

    expected = MORGAN_BITS + 12
    assert x.shape == (n, expected), f"Compound feature shape mismatch: {x.shape} (expected {expected})"
    return torch.from_numpy(x)


# ──────────────────────────────────────────────────────────────────────────────
# Target feature engineering  →  [N_t, 13]
# ──────────────────────────────────────────────────────────────────────────────

def build_target_features(targets_df: pd.DataFrame) -> torch.FloatTensor:
    """
    Returns FloatTensor [N_t, 13]:
      [0:4]  targetType one-hot  (TARGET_TYPES, 4 categories)
      [4:13] targetSpecies one-hot (TARGET_SPECIES, 9 categories)

    The 32-dim learned embedding will be concatenated inside the model.
    """
    rows = []
    for _, row in targets_df.iterrows():
        rows.append(np.concatenate([
            one_hot(row["type"],    TARGET_TYPES),
            one_hot(row["species"], TARGET_SPECIES),
        ]))
    x = np.stack(rows).astype(np.float32)
    assert x.shape == (len(targets_df), 13), f"Target feature shape mismatch: {x.shape}"
    return torch.from_numpy(x)


# ──────────────────────────────────────────────────────────────────────────────
# Edge tensor builder  →  [E, 16]
# ──────────────────────────────────────────────────────────────────────────────

def build_edge_tensors(
    activities_df: pd.DataFrame,
    compound_key_to_idx: dict[str, int],
    target_key_to_idx:   dict[str, int],
) -> tuple[
    torch.LongTensor,   # edge_index  [2, E]
    torch.FloatTensor,  # edge_attr   [E, 16]
    torch.FloatTensor,  # edge_label  [E]   NaN for censored
    torch.BoolTensor,   # train_mask  [E]   True = ground-truth label available
    np.ndarray,         # doc_years   [E]   float, NaN for unknown
]:
    """
    Build edge tensors for ALL activities (including censored ones).

    train_mask[i] = True  iff the edge has a clean, exact pChEMBL label.
    edge_label[i] = NaN   for censored / missing edges (ignored during training).
    """
    src, dst, attrs, labels, masks, years = [], [], [], [], [], []

    for _, row in activities_df.iterrows():
        c_idx = compound_key_to_idx.get(row["compound_key"], -1)
        t_idx = target_key_to_idx.get(row["target_key"],   -1)
        if c_idx < 0 or t_idx < 0:
            continue

        src.append(c_idx)
        dst.append(t_idx)
        attrs.append(np.concatenate([
            one_hot(row["endpoint_type"], ENDPOINT_TYPES),
            one_hot(row["qualifier"],     QUALIFIERS),
            one_hot(row["exp_setting"],   EXP_SETTINGS),
        ]))
        labels.append(float(row["pchembl"]))   # NaN for non-exact rows
        masks.append(bool(row["is_exact"]))
        years.append(float(row["doc_year"]))

    if not src:
        raise RuntimeError("No valid edges found — check that compound/target keys align.")

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.from_numpy(np.stack(attrs).astype(np.float32))
    edge_label = torch.tensor(labels, dtype=torch.float32)
    train_mask = torch.tensor(masks,  dtype=torch.bool)
    doc_years  = np.array(years, dtype=np.float64)

    assert edge_attr.shape[1] == 16, f"Edge attr width mismatch: {edge_attr.shape[1]}"
    return edge_index, edge_attr, edge_label, train_mask, doc_years


# ──────────────────────────────────────────────────────────────────────────────
# Temporal split
# ──────────────────────────────────────────────────────────────────────────────

def make_temporal_splits(
    doc_years: np.ndarray,
    compound_src: np.ndarray | None = None,
) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    """
    Returns (split_train, split_val, split_test) BoolTensors of shape [E].

    SPLIT_MODE="temporal":
      Test  : doc_year in [TEMPORAL_CUTOFF, TEST_MAX_YEAR]
      Val   : random VAL_FRAC_OF_TRAIN fraction of non-test edges
      Train : all remaining non-test edges  (unknown year → train)

    SPLIT_MODE="random":
      80/10/10 random split across all edges (ignores year) — edge-level.
      WARNING: same compound can appear in both train and test.

    SPLIT_MODE="compound_random":
      70/10/20 split at COMPOUND level — each compound appears in exactly
      one partition. No data leakage between train and test compounds.
    """
    E = len(doc_years)
    rng = np.random.default_rng(SEED)

    if SPLIT_MODE == "compound_random":
        assert compound_src is not None, "compound_src required for compound_random split"
        n_compounds = int(compound_src.max()) + 1
        perm = rng.permutation(n_compounds)
        n_test = int(n_compounds * 0.20)
        n_val  = int(n_compounds * 0.10)
        test_compounds  = set(perm[:n_test].tolist())
        val_compounds   = set(perm[n_test:n_test + n_val].tolist())
        train_compounds = set(perm[n_test + n_val:].tolist())

        test_np  = np.array([c in test_compounds  for c in compound_src], dtype=bool)
        val_np   = np.array([c in val_compounds   for c in compound_src], dtype=bool)
        train_np = np.array([c in train_compounds for c in compound_src], dtype=bool)
        return (
            torch.from_numpy(train_np),
            torch.from_numpy(val_np),
            torch.from_numpy(test_np),
        )

    if SPLIT_MODE == "random":
        perm   = rng.permutation(E)
        n_test = int(E * 0.20)
        n_val  = int(E * 0.10)
        test_np  = np.zeros(E, dtype=bool)
        val_np   = np.zeros(E, dtype=bool)
        train_np = np.zeros(E, dtype=bool)
        test_np[perm[:n_test]]                    = True
        val_np[perm[n_test:n_test + n_val]]       = True
        train_np[perm[n_test + n_val:]]           = True
        return (
            torch.from_numpy(train_np),
            torch.from_numpy(val_np),
            torch.from_numpy(test_np),
        )

    # --- temporal split (default) ---
    test_np = np.array(
        [not np.isnan(y) and TEMPORAL_CUTOFF <= y <= TEST_MAX_YEAR for y in doc_years],
        dtype=bool,
    )

    pre_idx = np.where(~test_np)[0]
    perm    = rng.permutation(len(pre_idx))
    n_val   = max(1, int(len(pre_idx) * VAL_FRAC_OF_TRAIN))

    val_set   = set(pre_idx[perm[:n_val]].tolist())
    train_set = set(pre_idx[perm[n_val:]].tolist())

    train_np = np.array([i in train_set for i in range(E)], dtype=bool)
    val_np   = np.array([i in val_set   for i in range(E)], dtype=bool)

    return (
        torch.from_numpy(train_np),
        torch.from_numpy(val_np),
        torch.from_numpy(test_np),
    )


# ──────────────────────────────────────────────────────────────────────────────
# HeteroData assembly
# ──────────────────────────────────────────────────────────────────────────────

def assemble_heterodata(
    compound_x:     torch.FloatTensor,
    target_x:       torch.FloatTensor,
    edge_index:     torch.LongTensor,
    edge_attr:      torch.FloatTensor,
    edge_label:     torch.FloatTensor,
    train_mask:     torch.BoolTensor,    # supervision mask (ground-truth availability)
    split_train:    torch.BoolTensor,    # temporal train partition
    split_val:      torch.BoolTensor,    # temporal val partition
    split_test:     torch.BoolTensor,    # temporal test partition
    doc_year_tensor: torch.FloatTensor,
) -> HeteroData:
    """
    Wraps all tensors into a PyG HeteroData object.

    Key tensors on ('compound', 'activity', 'target'):
      .edge_index  [2, E]   compound_idx → target_idx
      .edge_attr   [E, 16]  experiment features
      .edge_label  [E]      pChEMBL (NaN for censored/missing edges)
      .train_mask  [E]      True = exact ground-truth label present
      .split_train [E]      True = temporal train partition
      .split_val   [E]      True = temporal val partition
      .split_test  [E]      True = temporal test partition (year ≥ 2015)
      .doc_year    [E]      publication year (NaN if unknown)
    """
    data = HeteroData()

    data["compound"].x         = compound_x
    data["compound"].num_nodes = compound_x.shape[0]

    data["target"].x         = target_x
    data["target"].num_nodes = target_x.shape[0]

    et = ("compound", "activity", "target")
    data[et].edge_index  = edge_index
    data[et].edge_attr   = edge_attr
    data[et].edge_label  = edge_label
    data[et].train_mask  = train_mask
    data[et].split_train = split_train
    data[et].split_val   = split_val
    data[et].split_test  = split_test
    data[et].doc_year    = doc_year_tensor

    # Reverse edges so targets can aggregate messages from compounds
    rev_et = ("target", "rev_activity", "compound")
    data[rev_et].edge_index = edge_index.flip(0)
    data[rev_et].edge_attr  = edge_attr

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> HeteroData:
    print("=" * 60)
    print("  ODO Bipartite Graph — Preprocessing Pipeline")
    print("=" * 60)

    # ── Step 1: Load raw Excel ─────────────────────────────────────────────
    raw = load_raw()

    # ── Step 2: Build entity tables ────────────────────────────────────────
    print("\n[1/5] Building entity tables …")
    compounds_df  = build_compound_table(raw)
    targets_df    = build_target_table(raw)
    activities_df = build_activity_table(raw)

    # Cross-filter: retain only activities whose compound AND target are in the tables.
    # This is a safety check; in practice all should be present after the loops above.
    c_keys        = set(compounds_df["compound_key"])
    t_keys        = set(targets_df["target_key"])
    activities_df = activities_df[
        activities_df["compound_key"].isin(c_keys) &
        activities_df["target_key"].isin(t_keys)
    ].reset_index(drop=True)

    # ── Step 3: Node features ──────────────────────────────────────────────
    print("\n[2/5] Engineering node features …")
    compound_key_to_idx = {k: i for i, k in enumerate(compounds_df["compound_key"])}
    target_key_to_idx   = {k: i for i, k in enumerate(targets_df["target_key"])}

    compound_x = build_compound_features(compounds_df)
    target_x   = build_target_features(targets_df)
    print(f"  compound_x  shape: {tuple(compound_x.shape)}")
    print(f"  target_x    shape: {tuple(target_x.shape)}")

    # ── Step 4: Edge tensors ───────────────────────────────────────────────
    print("\n[3/5] Building edge tensors …")
    edge_index, edge_attr, edge_label, train_mask, doc_years = build_edge_tensors(
        activities_df, compound_key_to_idx, target_key_to_idx
    )
    E = edge_index.shape[1]
    print(f"  Total edges : {E:,}")
    print(f"  edge_attr   : {tuple(edge_attr.shape)}")
    print(f"  train_mask  : {train_mask.sum().item():,} exact-labelled / {E:,} total")

    # ── Step 5: Temporal split ─────────────────────────────────────────────
    print("\n[4/5] Computing temporal split …")
    compound_src_np = edge_index[0].numpy()
    split_train, split_val, split_test = make_temporal_splits(
        doc_years, compound_src=compound_src_np
    )

    known     = doc_years[~np.isnan(doc_years)]
    train_yrs = doc_years[split_train.numpy().astype(bool) & ~np.isnan(doc_years)]
    test_yrs  = doc_years[split_test.numpy().astype(bool)  & ~np.isnan(doc_years)]
    print(f"  Cutoff: year < {TEMPORAL_CUTOFF} → train/val  |  "
          f"year {TEMPORAL_CUTOFF}–{TEST_MAX_YEAR} → test")
    if len(train_yrs): print(f"  Train year range : {int(train_yrs.min())}–{int(train_yrs.max())}")
    if len(test_yrs):  print(f"  Test  year range : {int(test_yrs.min())}–{int(test_yrs.max())}")
    print(f"  Split counts  →  "
          f"train: {split_train.sum().item():,}  "
          f"val: {split_val.sum().item():,}  "
          f"test: {split_test.sum().item():,}")
    n_no_year = int(np.isnan(doc_years).sum())
    if n_no_year:
        print(f"  (No year: {n_no_year:,} edges → assigned to train)")

    # Supervised labels within the train split
    eff = int((split_train & train_mask).sum().item())
    print(f"  Effective supervised train labels : {eff:,}")

    doc_year_tensor = torch.tensor(
        [y if not np.isnan(y) else float("nan") for y in doc_years],
        dtype=torch.float32,
    )

    # ── Step 6: Assemble & save ────────────────────────────────────────────
    print("\n[5/5] Assembling HeteroData object …")
    data = assemble_heterodata(
        compound_x, target_x,
        edge_index, edge_attr, edge_label,
        train_mask,
        split_train, split_val, split_test,
        doc_year_tensor,
    )

    torch.save(data, OUTPUT_PATH)
    print(f"\n  Saved → {OUTPUT_PATH}")
    print("\nHeteroData summary:")
    print(data)
    print("\n" + "=" * 60)
    print("  Preprocessing complete.")
    print("=" * 60)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build ODO bipartite graph from Excel dataset."
    )
    parser.add_argument(
        "--split",
        choices=["temporal", "random", "compound_random"],
        default="temporal",
        help=(
            "Split strategy: "
            "'temporal' = train≤2015 / test 2016-2020 (paper model); "
            "'compound_random' = 70/10/20 compound-level split (no leakage); "
            "'random' = 80/10/10 edge-level split (diagnostic only)."
        ),
    )
    parser.add_argument(
        "--fp-bits",
        type=int,
        choices=[512, 1024, 2048],
        default=2048,
        help="Morgan fingerprint bit size (default: 2048).",
    )
    args = parser.parse_args()

    # Override module-level constants before main() uses them
    globals()["SPLIT_MODE"] = args.split
    globals()["MORGAN_BITS"] = args.fp_bits
    globals()["OUTPUT_PATH"] = os.path.join(
        ROOT, f"processed_{args.split}_{args.fp_bits}fp.pt"
    )

    main()
