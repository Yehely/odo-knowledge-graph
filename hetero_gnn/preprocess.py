"""
preprocess.py
──────────────
Excel → PyTorch Geometric HeteroData for the 5-node-type heterogeneous GNN
described in HETEROGENEOUS_GNN_ARCHITECTURE.md.

Run:
    conda run -n odo python3 hetero_gnn/run_preprocess.py

Output: hetero_gnn/processed_hetero_graph.pt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Graph schema
  Node types : 'compound'      [N_c, 2060]
               'target'        [N_t,   13]  (+ 32-dim learned embedding in model)
               'assay'         node_idx only (search-range 32-64-dim identity embedding in model, §5)
               'model_system'  node_idx only (search-range 8-24-dim identity embedding in model, §5)
               'document'      [N_d, 3] + journal_idx (search-range 4-12-dim journal embedding in model, §5)

  Label edge : ('compound', 'binds_to', 'target')            — one per activity row
  Structural : ('compound', 'tested_in', 'assay')             + reverse
               ('assay', 'tests_target', 'target')            + reverse
               ('assay', 'has_document', 'document')          + reverse
               ('assay', 'has_model_system', 'model_system')  + reverse
  (structural edges are de-duplicated to unique pairs; 'binds_to' is one
   edge per surviving raw row, per §1: "every row builds a binds_to edge")

  binds_to also carries 3 loss-eligibility masks (§1):
    exact_mask         — qualifier=='=', label in [MIN_PCHEMBL, MAX_PCHEMBL] -> MSE
    hinge_floor_mask    — raw qualifier '<'/'<=' -> one-sided hinge, floor
    hinge_ceiling_mask  — raw qualifier '>'/'>=' -> one-sided hinge, ceiling
  (mutually exclusive; non-Ki edges and unrecoverable/'~' qualifiers get all
   three False but still carry an edge_label/edge_attr and stay in the graph)

See hetero_gnn/README.md for the full rationale behind every design decision
(this docstring only summarises the schema).
"""
import hashlib
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

from hetero_gnn.config import (
    ASSAY_EMB_DIM, BINDING_SITES, COMPOUND_BUILDER_PATH, COMPOUND_NUMERIC_COLS,
    DOCUMENT_JOURNAL_EMB_DIM, EDGE_ATTR_DIM, ENDPOINT_TYPES, EXCEL_PATH, GRAPH_PATH,
    HINGE_CEILING_QUALIFIERS, HINGE_FLOOR_QUALIFIERS, LOSS_ENDPOINT, LOSS_QUALIFIER,
    MAX_PCHEMBL, MIN_PCHEMBL, MODEL_SYSTEM_EMB_DIM, MORGAN_BITS, MORGAN_RADIUS,
    N_TARGET_TAXONOMY_CATS, N_TARGET_TYPE_CATS, PHARM_ROLES, QUALIFIERS, SEED,
    TARGET_REFERENCE_PATH, TEMPORAL_CUTOFF_YEAR, UNIT_TO_MOLAR, VAL_FRAC_OF_TRAIN,
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


# ──────────────────────────────────────────────────────────────────────────────
# Low-level utilities (mirrors build_kg.py / preprocess_bipartite_graph.py)
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
    v = safe(val)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def short_hash(*parts, n: int = 16) -> str:
    h = hashlib.sha1("|".join("" if p is None else str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:n]


def _is_missing(value) -> bool:
    """True for None *and* float NaN — pandas silently turns None into NaN
    when a DataFrame is built from a list of dicts with a mixed-type
    column, so both forms have to be treated as "missing" everywhere a
    value is pulled back out of a DataFrame/Series."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _one_hot(value: str | None, categories: list[str], other_bucket: bool) -> np.ndarray:
    """One-hot encode value against categories.

    other_bucket=True  : unmatched/missing -> last category (an explicit
                          catch-all already present in `categories`).
    other_bucket=False : unmatched/missing -> all-zero vector (no catch-all
                          slot; used where "not recorded" must stay
                          distinguishable from "recorded but rare").
    """
    vec = np.zeros(len(categories), dtype=np.float32)
    if _is_missing(value):
        if other_bucket:
            vec[-1] = 1.0
        return vec
    v = str(value).strip().lower()
    for i, cat in enumerate(categories):
        if cat.lower() == v:
            vec[i] = 1.0
            return vec
    if other_bucket:
        vec[-1] = 1.0
    return vec


def top_n_vocab(values, n: int) -> list[str]:
    """The n most frequent distinct non-null values, by descending frequency."""
    from collections import Counter
    counts = Counter(v for v in values if not _is_missing(v))
    return [v for v, _ in counts.most_common(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Raw data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Read the ODO Excel workbook and return the 'Full Dataset' sheet.

    Column names are whitespace-normalised on load: the source file has one
    column literally named 'bao_assay _format_l2_id' (stray space before
    '_format'); collapsing internal whitespace to '_' fixes this and is a
    no-op for every other column.
    """
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")
    print(f"Reading {os.path.basename(EXCEL_PATH)} …")
    df = pd.read_excel(EXCEL_PATH, sheet_name="Full Dataset")
    df.columns = [re.sub(r"\s+", "_", c.strip()) for c in df.columns]
    print(f"  {len(df):,} rows × {len(df.columns)} columns loaded")
    return df


def drop_corrupted_unit_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """§1/§4: drop rows with a corrupted/implausible unit_of_measurement.

    These are Ki-endpoint rows with no pchembl_value and no usable unit —
    either literally 'M' (e.g. a recorded value of 168 would mean 168
    Molar — physically implausible at Ki scale) or entirely missing so the
    scale can't be recovered at all. Reproduces exactly the 9 rows §4 calls
    out (verified against the source workbook: 4 rows with unit=='M',
    5 with unit missing, all endpoint=='Ki', all pchembl_value NaN).
    """
    corrupted = (
        (raw["endpoint"] == LOSS_ENDPOINT)
        & raw["pchembl_value"].isna()
        & raw["endpoint_value"].notna()
        & (raw["unit_of_measurement"].isna() | (raw["unit_of_measurement"] == "M"))
    )
    n = int(corrupted.sum())
    print(f"  Dropping {n} rows with corrupted/implausible unit_of_measurement (§1/§4)")
    return raw.loc[~corrupted].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entity key resolution — one resolver per node type, each with a fallback
# chain so (almost) every row can be placed in the graph (§1: maximal
# inclusion; only the 9 corrupted rows above are dropped entirely).
# ──────────────────────────────────────────────────────────────────────────────

def compound_key_of(row) -> str | None:
    """§2.A primary key: pubchem_cid (always the tested compound, never the
    parent). Falls back to the standard InChIKey (0% missing) when
    pubchem_cid itself is absent (~2.9% of rows)."""
    cid = safe(row.get("pubchem_cid"))
    if cid is not None:
        return f"cid_{cid}"
    ikey = safe(row.get("rdkit_library_standard_inchi_key"))
    return f"inchikey_{ikey}" if ikey else None


def target_key_of(row) -> str | None:
    """§2.B primary key: uniprot_protein_id. Falls back to a slugified
    target_name (~4.1% of rows lack a UniProt id)."""
    uid = safe(row.get("uniprot_protein_id"))
    if uid is not None:
        return f"uniprot_{uid}"
    tname = safe(row.get("target_name"))
    return f"tname_{slugify(tname)}" if tname else None


def assay_key_of(row) -> str | None:
    """§2.C primary key: chembl_assay_id. ~9% of rows lack one; for those,
    fall back to a content hash of the assay's static descriptive columns
    (format/method/bioassay/setting + target + document) so rows that
    describe the *same* unlabelled assay still merge into one node instead
    of exploding into one singleton per row — mirrors build_kg.py's
    content-addressed Activity URI convention (see CLAUDE.md)."""
    aid = safe(row.get("chembl_assay_id"))
    if aid is not None:
        return f"assay_{aid}"
    parts = [
        safe(row.get("bao_assay_format")), safe(row.get("bao_assay_method")),
        safe(row.get("bao_bioassay_1")), safe(row.get("bao_bioassay_2")),
        safe(row.get("bao_bioassay_type")), safe(row.get("bao_experimental_setting")),
        safe(row.get("subcellular_format")), safe(row.get("target_name")),
        safe(row.get("pubmed_id")) or safe(row.get("document_doi")),
    ]
    return "assay_h" + short_hash(*parts)


def document_key_of(row) -> str:
    """§2.E primary key chain: pubmed_id -> document_doi -> patent_id ->
    shared 'unknown_document' bucket (never fails to resolve)."""
    pmid = safe(row.get("pubmed_id"))
    if pmid is not None:
        return f"pmid_{pmid}"
    doi = safe(row.get("document_doi"))
    if doi is not None:
        return f"doi_{doi.lower()}"
    pat = safe(row.get("patent_id"))
    if pat is not None:
        return f"patent_{pat}"
    return "unknown_document"


def _ms_part(v) -> str | None:
    v = safe(v)
    return v.strip().lower() if v else None


def model_system_key_of(row) -> str:
    """§2.D primary key: 'derived from cell line / tissue / organism
    identity'. Built as a composite of the normalised cell/tissue/animal
    columns; rows with none of these specified (generic / not reported)
    share one 'unspecified_model_system' bucket node."""
    parts = [
        _ms_part(row.get("cell_type_name")), _ms_part(row.get("cellosaurus_cell_line_id")),
        _ms_part(row.get("tissue_name")), _ms_part(row.get("ncit_animal_model")),
        _ms_part(row.get("ncit_animal_animal_model_strain")), _ms_part(row.get("ncbi_tissue_taxonomy")),
    ]
    if all(p is None for p in parts):
        return "unspecified_model_system"
    return "ms_" + "|".join(p or "_" for p in parts)


# ──────────────────────────────────────────────────────────────────────────────
# Feature normalisers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_taxonomy(raw) -> str | None:
    """Canonicalise species-name casing ('Homo Sapiens' -> 'Homo sapiens');
    composite multi-species entries (e.g. 'Mus musculus / Cavia porcellus')
    and the literal 'unspecified' both map to the 'unspecified' bucket."""
    v = safe(raw)
    if v is None:
        return None
    if v.strip().lower() == "unspecified" or "/" in v:
        return "unspecified"
    parts = v.strip().split()
    if len(parts) >= 2:
        return parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])
    return v.strip()


def normalize_journal(raw) -> str | None:
    """Fold punctuation/whitespace variants of the same journal together
    ('J. Med. Chem.' / 'J Med Chem' -> the same normalised key)."""
    v = safe(raw)
    if v is None:
        return None
    return re.sub(r"[.\s]+", " ", v).strip().lower()


def normalize_pharm_role(raw) -> str | None:
    v = safe(raw)
    if v is None:
        return None
    v = v.strip().lower()
    if v == "agonist":
        return "agonist"
    if v == "antagonist":
        return "antagonist"
    return "other"   # partial agonist, inverse agonist, positive allosteric modulator, ...


def normalize_binding_site(raw) -> str | None:
    v = safe(raw)
    if v is None:
        return None
    v = v.strip().lower()
    if v == "high affinity":
        return "high_affinity"
    if v == "low affinity":
        return "low_affinity"
    return None


def derive_pchembl(pchembl_raw, endpoint_value_raw, unit_raw) -> float:
    """§1 label derivation (value only — the qualifier is never rewritten).

    If pchembl_value is present, use it as-is. Otherwise, derive from
    endpoint_value + unit_of_measurement when the unit is a pure molar
    concentration (nM or uM only — '%', 'mg kg-1', etc. are not
    concentration units and can't be converted).

    The raw endpoint_qualifier is used, unmodified, everywhere downstream
    (edge_attr, the exact-loss mask, and the censored-hinge floor/ceiling
    split in §1.2) — the qualifier no longer needs to be flipped as a
    separate step: `raw_qualifier in {'<','<='}` already means "true pKi
    is above this derived value" (a floor) and `raw_qualifier in {'>','>='}`
    already means "true pKi is below this derived value" (a ceiling),
    which is exactly the §1.2 hinge-loss branch — flipping the *symbol*
    per the old step 3 and then re-deriving floor/ceiling from the flipped
    symbol is equivalent to just reading the direction off the original
    symbol directly.
    """
    pchembl = to_float(pchembl_raw)
    if not math.isnan(pchembl):
        return pchembl

    ev = to_float(endpoint_value_raw)
    unit = safe(unit_raw)
    if math.isnan(ev) or unit not in UNIT_TO_MOLAR:
        return float("nan")

    value_m = ev * UNIT_TO_MOLAR[unit]
    if value_m <= 0:
        return float("nan")

    return -math.log10(value_m)


# ──────────────────────────────────────────────────────────────────────────────
# Single combined pass: entity tables + activity list
# ──────────────────────────────────────────────────────────────────────────────

def build_entities_and_activities(raw: pd.DataFrame):
    """
    One scan over all surviving rows. For each row, resolves all 5 entity
    keys, registers each entity's feature snapshot on first encounter
    (mirrors build_kg.py's "seen" dedup convention — first sighting wins,
    per CLAUDE.md), and appends one activity record (used to build the
    `binds_to` edge and the 4 structural edge types).

    Rows missing a resolvable compound_key or target_key are skipped
    entirely (extremely rare given the fallback chains above) — there is no
    valid compound-target activity to record without both.
    """
    compounds: dict[str, dict] = {}
    targets: dict[str, dict] = {}
    assays: dict[str, None] = {}
    documents: dict[str, dict] = {}
    model_systems: dict[str, None] = {}
    activities: list[dict] = []
    n_skipped = 0

    for _, row in raw.iterrows():
        c_key = compound_key_of(row)
        t_key = target_key_of(row)
        if c_key is None or t_key is None:
            n_skipped += 1
            continue

        a_key = assay_key_of(row)
        d_key = document_key_of(row)
        m_key = model_system_key_of(row)

        if c_key not in compounds:
            compounds[c_key] = {
                "compound_key": c_key,
                "smiles": safe(row.get("rdkit_canonical_smiles")),
                **{col: to_float(row.get(col)) for col in COMPOUND_NUMERIC_COLS},
            }

        if t_key not in targets:
            targets[t_key] = {
                "target_key": t_key,
                "name": safe(row.get("target_name")) or t_key,
                "type": safe(row.get("target_type")),
                "taxonomy": normalize_taxonomy(row.get("ncbi_target_taxonomy")),
            }

        assays.setdefault(a_key, None)
        model_systems.setdefault(m_key, None)

        if d_key not in documents:
            documents[d_key] = {
                "document_key": d_key,
                "journal_raw": safe(row.get("document_journal")),
                "year": to_float(row.get("document_year")),
                "is_patent": 1.0 if safe(row.get("patent_id")) is not None else 0.0,
                "is_unknown": 1.0 if d_key == "unknown_document" else 0.0,
            }

        pchembl = derive_pchembl(
            row.get("pchembl_value"), row.get("endpoint_value"), row.get("unit_of_measurement"),
        )
        qualifier_raw = safe(row.get("endpoint_qualifier")) or ""
        endpoint_type = safe(row.get("endpoint")) or "other"
        is_ki = endpoint_type == LOSS_ENDPOINT
        has_value = not math.isnan(pchembl)

        # §1: two loss components, both endpoint=='Ki' only. Non-Ki edges and
        # unrecoverable/'~' qualifiers get none of the three flags below —
        # they still get a binds_to edge and participate fully in message
        # passing, they just never contribute a loss term (§1, §9).
        exact_eligible = (
            is_ki and qualifier_raw == LOSS_QUALIFIER and has_value
            and MIN_PCHEMBL <= pchembl <= MAX_PCHEMBL
        )
        hinge_floor_eligible = is_ki and has_value and qualifier_raw in HINGE_FLOOR_QUALIFIERS
        hinge_ceiling_eligible = is_ki and has_value and qualifier_raw in HINGE_CEILING_QUALIFIERS

        activities.append({
            "compound_key": c_key, "target_key": t_key, "assay_key": a_key,
            "document_key": d_key, "model_system_key": m_key,
            "endpoint_type": endpoint_type,
            "qualifier_raw": qualifier_raw,
            "pharm_role": normalize_pharm_role(row.get("compound_pharmacological_role")),
            "binding_site": normalize_binding_site(row.get("chembl_binding_site_description")),
            "pchembl": pchembl,
            "exact_eligible": exact_eligible,
            "hinge_floor_eligible": hinge_floor_eligible,
            "hinge_ceiling_eligible": hinge_ceiling_eligible,
            "doc_year": documents[d_key]["year"],   # canonical per-document year, not the raw row's own value
        })

    if n_skipped:
        print(f"  Skipped {n_skipped} rows with no resolvable compound/target key")

    compounds_df = pd.DataFrame(compounds.values()).reset_index(drop=True)
    targets_df = pd.DataFrame(targets.values()).reset_index(drop=True)
    documents_df = pd.DataFrame(documents.values()).reset_index(drop=True)
    assay_keys = list(assays.keys())
    model_system_keys = list(model_systems.keys())

    print(f"  {len(compounds_df):,} unique compounds")
    print(f"  {len(targets_df):,} unique targets")
    print(f"  {len(assay_keys):,} unique assays")
    print(f"  {len(documents_df):,} unique documents")
    print(f"  {len(model_system_keys):,} unique model systems")
    print(f"  {len(activities):,} activities (binds_to edges)")

    return compounds_df, targets_df, assay_keys, documents_df, model_system_keys, activities


# ──────────────────────────────────────────────────────────────────────────────
# Compound features  →  [N_c, MORGAN_BITS + 12]   (§2.A, §5)
# ──────────────────────────────────────────────────────────────────────────────

def _morgan_fp(smiles: str | None) -> np.ndarray:
    fp = np.zeros(MORGAN_BITS, dtype=np.float32)
    if not _RDKIT or _is_missing(smiles):
        return fp
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fp
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    fp[:] = np.frombuffer(bv.ToBitString().encode(), dtype=np.uint8) - ord("0")
    return fp


class CompoundFeatureBuilder:
    """Fits a StandardScaler on the 12 numeric columns; NaN imputed with the
    column mean before scaling (so N/A compounds sit at each feature
    distribution's centre rather than an arbitrary value). Kept as a class
    (not a free function) so predict.py can reuse the exact fitted scaler
    for query compounds at inference time."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._col_means: np.ndarray | None = None

    def fit_transform(self, compounds_df: pd.DataFrame) -> torch.FloatTensor:
        raw = compounds_df[COMPOUND_NUMERIC_COLS].values.astype(np.float64)
        self._col_means = np.nan_to_num(np.nanmean(raw, axis=0), nan=0.0)
        nan_pos = np.where(np.isnan(raw))
        raw[nan_pos] = np.take(self._col_means, nan_pos[1])
        numeric_norm = self.scaler.fit_transform(raw).astype(np.float32)
        return self._assemble(compounds_df, numeric_norm)

    def transform(self, compounds_df: pd.DataFrame) -> torch.FloatTensor:
        raw = compounds_df[COMPOUND_NUMERIC_COLS].values.astype(np.float64)
        nan_pos = np.where(np.isnan(raw))
        raw[nan_pos] = np.take(self._col_means, nan_pos[1])
        numeric_norm = self.scaler.transform(raw).astype(np.float32)
        return self._assemble(compounds_df, numeric_norm)

    def _assemble(self, compounds_df: pd.DataFrame, numeric_norm: np.ndarray) -> torch.FloatTensor:
        n = len(compounds_df)
        print(f"  Computing Morgan fingerprints for {n:,} compounds …")
        fps = np.stack([_morgan_fp(s) for s in compounds_df["smiles"]], axis=0)
        x = np.concatenate([fps, numeric_norm], axis=1)
        expected = MORGAN_BITS + len(COMPOUND_NUMERIC_COLS)
        assert x.shape == (n, expected), f"Compound feature shape mismatch: {x.shape} (expected {(n, expected)})"
        return torch.from_numpy(x)


# ──────────────────────────────────────────────────────────────────────────────
# Target features  →  [N_t, 13] categorical + node_idx for the 32-dim
# uniprot_protein_id embedding (added in the model)   (§2.B, §5)
# ──────────────────────────────────────────────────────────────────────────────

def build_target_features(targets_df: pd.DataFrame):
    target_types = top_n_vocab(targets_df["type"], N_TARGET_TYPE_CATS)
    taxonomy_vocab = top_n_vocab(targets_df["taxonomy"], N_TARGET_TAXONOMY_CATS)

    rows = []
    for _, row in targets_df.iterrows():
        type_oh = _one_hot(row["type"], target_types, other_bucket=False)
        tax_oh = _one_hot(row["taxonomy"], taxonomy_vocab, other_bucket=False)
        rows.append(np.concatenate([type_oh, tax_oh]))

    cat_feats = torch.from_numpy(np.stack(rows).astype(np.float32))
    node_idx = torch.arange(len(targets_df), dtype=torch.long)
    assert cat_feats.shape[1] == N_TARGET_TYPE_CATS + N_TARGET_TAXONOMY_CATS
    return cat_feats, node_idx, target_types, taxonomy_vocab


# ──────────────────────────────────────────────────────────────────────────────
# Document features  →  journal_idx [N_d] (for the model's journal
# embedding) + scalars [N_d, 3] = is_patent, norm_year, is_unknown  (§2.E, §5)
# ──────────────────────────────────────────────────────────────────────────────

def build_document_features(documents_df: pd.DataFrame):
    journal_vocab = top_n_vocab(
        (normalize_journal(j) for j in documents_df["journal_raw"]),
        DOCUMENT_JOURNAL_EMB_DIM * 100,  # generous cap; real data has 19 raw variants (~13 after normalisation)
    )
    # journal embedding table reserves index len(journal_vocab) for "other/unknown journal"
    vocab_index = {j: i for i, j in enumerate(journal_vocab)}
    other_idx = len(journal_vocab)
    journal_idx = np.array([
        vocab_index.get(normalize_journal(j), other_idx) for j in documents_df["journal_raw"]
    ], dtype=np.int64)

    years = documents_df["year"].values.astype(np.float64)
    known = ~np.isnan(years)
    mean_year = float(years[known].mean()) if known.any() else 2000.0
    filled = np.where(known, years, mean_year)
    y_min, y_max = float(filled.min()), float(filled.max())
    norm_year = ((filled - y_min) / max(y_max - y_min, 1e-6)).astype(np.float32)

    is_patent = documents_df["is_patent"].values.astype(np.float32)
    is_unknown = documents_df["is_unknown"].values.astype(np.float32)
    scalars = np.stack([is_patent, norm_year, is_unknown], axis=1).astype(np.float32)

    return torch.from_numpy(journal_idx), torch.from_numpy(scalars), journal_vocab, (y_min, y_max, mean_year)


# ──────────────────────────────────────────────────────────────────────────────
# binds_to edge tensors  →  [E, 16]   (§3, §5)
# ──────────────────────────────────────────────────────────────────────────────

def build_binds_to_edges(activities: list[dict], compound_key_to_idx: dict, target_key_to_idx: dict):
    src, dst, attrs, labels = [], [], [], []
    exact, hinge_floor, hinge_ceiling, years = [], [], [], []

    for a in activities:
        src.append(compound_key_to_idx[a["compound_key"]])
        dst.append(target_key_to_idx[a["target_key"]])

        ep_oh = _one_hot(a["endpoint_type"], ENDPOINT_TYPES, other_bucket=True)
        q_oh = _one_hot(a["qualifier_raw"], QUALIFIERS, other_bucket=True)
        role_oh = _one_hot(a["pharm_role"], PHARM_ROLES, other_bucket=False)
        site_oh = _one_hot(a["binding_site"], BINDING_SITES, other_bucket=False)
        attrs.append(np.concatenate([ep_oh, q_oh, role_oh, site_oh]))

        labels.append(a["pchembl"])   # NaN for unlabelled — kept as NaN, resolved in dataset.py
        exact.append(a["exact_eligible"])
        hinge_floor.append(a["hinge_floor_eligible"])
        hinge_ceiling.append(a["hinge_ceiling_eligible"])
        years.append(a["doc_year"])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.from_numpy(np.stack(attrs).astype(np.float32))
    edge_label = torch.tensor(labels, dtype=torch.float32)
    exact_mask = torch.tensor(exact, dtype=torch.bool)
    hinge_floor_mask = torch.tensor(hinge_floor, dtype=torch.bool)
    hinge_ceiling_mask = torch.tensor(hinge_ceiling, dtype=torch.bool)
    doc_years = np.array(years, dtype=np.float64)

    assert edge_attr.shape[1] == EDGE_ATTR_DIM, f"Edge attr width mismatch: {edge_attr.shape[1]}"
    assert not (hinge_floor_mask & hinge_ceiling_mask).any(), "floor/ceiling masks must be mutually exclusive"
    assert not (exact_mask & hinge_floor_mask).any(), "exact/hinge masks must be mutually exclusive"
    assert not (exact_mask & hinge_ceiling_mask).any(), "exact/hinge masks must be mutually exclusive"
    return edge_index, edge_attr, edge_label, exact_mask, hinge_floor_mask, hinge_ceiling_mask, doc_years


# ──────────────────────────────────────────────────────────────────────────────
# Structural edges (de-duplicated pairs) for message passing   (§7)
# ──────────────────────────────────────────────────────────────────────────────

def build_structural_edges(
    activities: list[dict],
    compound_key_to_idx: dict, target_key_to_idx: dict, assay_key_to_idx: dict,
    document_key_to_idx: dict, model_system_key_to_idx: dict,
) -> dict[str, torch.LongTensor]:
    """Each raw row implies 4 structural (context) links; de-duplicated to
    unique pairs so mean-aggregation (§7) averages over *unique* neighbours,
    not once per repeated row."""
    ca_pairs, at_pairs, ad_pairs, am_pairs = set(), set(), set(), set()

    for a in activities:
        c = compound_key_to_idx[a["compound_key"]]
        t = target_key_to_idx[a["target_key"]]
        asy = assay_key_to_idx[a["assay_key"]]
        d = document_key_to_idx[a["document_key"]]
        m = model_system_key_to_idx[a["model_system_key"]]
        ca_pairs.add((c, asy))
        at_pairs.add((asy, t))
        ad_pairs.add((asy, d))
        am_pairs.add((asy, m))

    def to_edge_index(pairs: set) -> torch.LongTensor:
        if not pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        arr = np.array(sorted(pairs), dtype=np.int64).T
        return torch.from_numpy(arr)

    return {
        "compound_assay": to_edge_index(ca_pairs),
        "assay_target": to_edge_index(at_pairs),
        "assay_document": to_edge_index(ad_pairs),
        "assay_model_system": to_edge_index(am_pairs),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Temporal split (§9, Open Decision #1)
# ──────────────────────────────────────────────────────────────────────────────

def make_temporal_split(doc_years: np.ndarray):
    """train: year < TEMPORAL_CUTOFF_YEAR (or unknown year) | val: random
    VAL_FRAC_OF_TRAIN of that | test: year >= TEMPORAL_CUTOFF_YEAR."""
    E = len(doc_years)
    rng = np.random.default_rng(SEED)

    known = ~np.isnan(doc_years)
    test_mask = known & (doc_years >= TEMPORAL_CUTOFF_YEAR)
    pre_idx = np.where(~test_mask)[0]

    perm = rng.permutation(len(pre_idx))
    n_val = max(1, int(len(pre_idx) * VAL_FRAC_OF_TRAIN))
    val_idx = pre_idx[perm[:n_val]]
    train_idx = pre_idx[perm[n_val:]]

    train_mask = np.zeros(E, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(E, dtype=bool)
    val_mask[val_idx] = True

    return torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)


# ──────────────────────────────────────────────────────────────────────────────
# HeteroData assembly
# ──────────────────────────────────────────────────────────────────────────────

def assemble_heterodata(
    compound_x, target_cat_x, target_node_idx,
    n_assays: int, n_model_systems: int,
    document_journal_idx, document_scalars, journal_vocab_size: int,
    structural_edges: dict,
    binds_to_edge_index, binds_to_edge_attr, binds_to_edge_label,
    exact_mask, hinge_floor_mask, hinge_ceiling_mask,
    split_train, split_val, split_test, doc_year_tensor,
) -> HeteroData:
    data = HeteroData()

    data["compound"].x = compound_x
    data["compound"].num_nodes = compound_x.shape[0]

    data["target"].x = target_cat_x
    data["target"].node_idx = target_node_idx
    data["target"].num_nodes = target_cat_x.shape[0]

    data["assay"].node_idx = torch.arange(n_assays, dtype=torch.long)
    data["assay"].num_nodes = n_assays

    data["model_system"].node_idx = torch.arange(n_model_systems, dtype=torch.long)
    data["model_system"].num_nodes = n_model_systems

    data["document"].journal_idx = document_journal_idx
    data["document"].x = document_scalars
    data["document"].num_nodes = document_scalars.shape[0]
    # Embedding-table size (vocab + 1 "other/unknown journal" slot). Passed
    # in directly from the vocab itself — deriving it from journal_idx.max()
    # would silently under-size the table if the "other" bucket happened not
    # to be used by any document in a given run of the data.
    data["document"].journal_vocab_size = journal_vocab_size

    et = ("compound", "binds_to", "target")
    data[et].edge_index = binds_to_edge_index
    data[et].edge_attr = binds_to_edge_attr
    data[et].edge_label = binds_to_edge_label
    # §1 loss eligibility (base masks, *not* yet intersected with the
    # temporal split — dataset.py combines each of these three with
    # split_train/val/test into the final 9 masks train.py consumes).
    data[et].exact_mask = exact_mask
    data[et].hinge_floor_mask = hinge_floor_mask
    data[et].hinge_ceiling_mask = hinge_ceiling_mask
    data[et].split_train = split_train
    data[et].split_val = split_val
    data[et].split_test = split_test
    data[et].doc_year = doc_year_tensor

    def add_structural(name: str, src_type: str, dst_type: str, edge_index: torch.LongTensor):
        data[(src_type, name, dst_type)].edge_index = edge_index
        data[(dst_type, f"rev_{name}", src_type)].edge_index = edge_index.flip(0)

    add_structural("tested_in", "compound", "assay", structural_edges["compound_assay"])
    add_structural("tests_target", "assay", "target", structural_edges["assay_target"])
    add_structural("has_document", "assay", "document", structural_edges["assay_document"])
    add_structural("has_model_system", "assay", "model_system", structural_edges["assay_model_system"])

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> HeteroData:
    print("=" * 60)
    print("  ODO Heterogeneous Graph — Preprocessing Pipeline")
    print("=" * 60)

    raw = load_raw()
    raw = drop_corrupted_unit_rows(raw)

    print("\n[1/5] Building entity tables + activities …")
    (compounds_df, targets_df, assay_keys, documents_df, model_system_keys,
     activities) = build_entities_and_activities(raw)

    compound_key_to_idx = {k: i for i, k in enumerate(compounds_df["compound_key"])}
    target_key_to_idx = {k: i for i, k in enumerate(targets_df["target_key"])}
    assay_key_to_idx = {k: i for i, k in enumerate(assay_keys)}
    document_key_to_idx = {k: i for i, k in enumerate(documents_df["document_key"])}
    model_system_key_to_idx = {k: i for i, k in enumerate(model_system_keys)}

    print("\n[2/5] Engineering node features …")
    compound_builder = CompoundFeatureBuilder()
    compound_x = compound_builder.fit_transform(compounds_df)
    torch.save(compound_builder, COMPOUND_BUILDER_PATH)
    print(f"  Fitted CompoundFeatureBuilder saved → {COMPOUND_BUILDER_PATH} (reused by predict.py)")
    target_cat_x, target_node_idx, target_types, taxonomy_vocab = build_target_features(targets_df)
    targets_df[["target_key", "name", "type", "taxonomy"]].to_csv(TARGET_REFERENCE_PATH, index_label="target_idx")
    print(f"  Target reference table saved → {TARGET_REFERENCE_PATH} (for predict.py)")
    document_journal_idx, document_scalars, journal_vocab, year_range = build_document_features(documents_df)
    print(f"  compound_x       shape: {tuple(compound_x.shape)}")
    print(f"  target_x         shape: {tuple(target_cat_x.shape)}  (+ {32}-dim identity embedding in model)")
    print(f"  assay            {len(assay_keys):,} nodes (+ {ASSAY_EMB_DIM}-dim identity embedding by default, search range 32-64)")
    print(f"  model_system     {len(model_system_keys):,} nodes (+ {MODEL_SYSTEM_EMB_DIM}-dim identity embedding by default, search range 8-24)")
    print(f"  document_scalars shape: {tuple(document_scalars.shape)}  (+ {DOCUMENT_JOURNAL_EMB_DIM}-dim journal embedding by default (search range 4-12), vocab={len(journal_vocab)})")
    print(f"  target types    : {target_types}")
    print(f"  target taxonomy : {taxonomy_vocab}")
    print(f"  journal vocab   : {journal_vocab}")

    print("\n[3/5] Building binds_to + structural edges …")
    (binds_to_edge_index, binds_to_edge_attr, binds_to_edge_label,
     exact_mask, hinge_floor_mask, hinge_ceiling_mask, doc_years) = \
        build_binds_to_edges(activities, compound_key_to_idx, target_key_to_idx)
    structural_edges = build_structural_edges(
        activities, compound_key_to_idx, target_key_to_idx, assay_key_to_idx,
        document_key_to_idx, model_system_key_to_idx,
    )
    E = binds_to_edge_index.shape[1]
    print(f"  binds_to edges          : {E:,}")
    print(f"  exact (MSE) eligible    : {exact_mask.sum().item():,}")
    print(f"  hinge-floor eligible    : {hinge_floor_mask.sum().item():,}  (raw qualifier < or <=)")
    print(f"  hinge-ceiling eligible  : {hinge_ceiling_mask.sum().item():,}  (raw qualifier > or >=)")
    n_loss_eligible = int((exact_mask | hinge_floor_mask | hinge_ceiling_mask).sum().item())
    print(f"  total loss-eligible     : {n_loss_eligible:,} / {E:,}  "
          f"({E - n_loss_eligible:,} message-passing-only: non-Ki or unrecoverable/'~' qualifier)")
    for name, ei in structural_edges.items():
        print(f"  {name:20s} edges : {ei.shape[1]:,}")

    print("\n[4/5] Computing temporal split …")
    split_train, split_val, split_test = make_temporal_split(doc_years)
    known = doc_years[~np.isnan(doc_years)]
    print(f"  Cutoff: year < {TEMPORAL_CUTOFF_YEAR} → train/val  |  year ≥ {TEMPORAL_CUTOFF_YEAR} → test")
    if len(known):
        print(f"  Year range        : {int(known.min())}–{int(known.max())}")
    print(f"  Split counts  →  train: {split_train.sum().item():,}  "
          f"val: {split_val.sum().item():,}  test: {split_test.sum().item():,}")
    eff_exact = int((split_train & exact_mask).sum().item())
    eff_hinge = int((split_train & (hinge_floor_mask | hinge_ceiling_mask)).sum().item())
    print(f"  Train split → exact: {eff_exact:,}  censored/hinge: {eff_hinge:,}")

    doc_year_tensor = torch.tensor(
        [y if not np.isnan(y) else float("nan") for y in doc_years], dtype=torch.float32,
    )

    print("\n[5/5] Assembling HeteroData object …")
    data = assemble_heterodata(
        compound_x, target_cat_x, target_node_idx,
        len(assay_keys), len(model_system_keys),
        document_journal_idx, document_scalars, len(journal_vocab) + 1,
        structural_edges,
        binds_to_edge_index, binds_to_edge_attr, binds_to_edge_label,
        exact_mask, hinge_floor_mask, hinge_ceiling_mask,
        split_train, split_val, split_test, doc_year_tensor,
    )

    torch.save(data, GRAPH_PATH)
    print(f"\n  Saved → {GRAPH_PATH}")
    print("\nHeteroData summary:")
    print(data)
    print("\n" + "=" * 60)
    print("  Preprocessing complete.")
    print("=" * 60)
    return data


if __name__ == "__main__":
    main()
