"""
Reconciles Table 1's entity counts (13,297 compounds / 73 targets / 4,631
assays / 1,069 documents / 148 model systems / 37,353 activities) against
the source spreadsheet, using pandas directly — no GNN code imported, no
GraphDB required.

Two identity-key pairs are tested per entity, taken verbatim from the
pipelines (see analysis/target_inventory.md Task 1 for the full
side-by-side):

  "chembl" rule  (kg/build_kg.py == gnn/preprocess_bipartite_graph.py
                  for compound/target):
      compound key      = chembl_compound_id, else inchikey_<InChIKey>
      target key         = chembl_target_id,  else slug(target_name)
      assay key          = chembl_assay_id ONLY; rows lacking it get no
                            assay node (kg/build_kg.py:build_assay returns
                            None, contributing nothing to the count)
      document key       = pmid (int-normalised) -> document_doi ->
                            chembl_document_id; rows with none of the
                            three get no document node (returns None)
      model_system key   = requires ncit_model_system_id or
                            slug(ncit_model_system) as a "base"; rows
                            lacking both get no model_system node
                            (returns None). When present, composited with
                            cell/tissue/organism identity columns.

  "hetero" rule  (hetero_gnn/preprocess.py):
      compound key       = cid_<pubchem_cid>, else inchikey_<InChIKey>
      target key          = uniprot_<uniprot_protein_id>, else
                             tname_<slug(target_name)>
      assay key           = assay_<chembl_assay_id>, else a content hash
                             ("assay_h...") of BAO format/method/bioassay/
                             setting + subcellular_format + target_name +
                             pubmed_id/document_doi — never None
      document key        = pmid_<pmid>, else doi_<doi.lower()>, else
                             patent_<patent_id>, else the shared bucket
                             "unknown_document" — never None
      model_system key    = composite of cell_type_name,
                             cellosaurus_cell_line_id, tissue_name,
                             ncit_animal_model,
                             ncit_animal_animal_model_strain,
                             ncbi_tissue_taxonomy (ncit_model_system is
                             NOT used); rows with none of the six fields
                             share the bucket "unspecified_model_system"
                             — never None
      + drops 9 rows with a corrupted/implausible unit_of_measurement
        (Ki endpoint, no pchembl_value, unit missing or literally "M")
        before any of the above dedup runs.

Agreement check (assay/document/model_system, done by inspection before
any counting):
  - assay:        DISAGREE. chembl rule drops rows without
                   chembl_assay_id; hetero rule always assigns a key via
                   its content-hash fallback.
  - document:      DISAGREE. Fallback chain differs (chembl_document_id
                   vs patent_id) and null-handling differs (chembl rule
                   drops; hetero rule buckets into "unknown_document").
  - model_system:  DISAGREE, and more fundamentally so — the chembl rule
                   requires ncit_model_system/_id as its base identity;
                   the hetero rule never reads that column at all and
                   keys purely off cell/tissue/organism fields.

Read-only: only reads the .xlsx and hetero_gnn/processed_hetero_graph.pt,
writes nothing. Run:
    python3 analysis/reconcile_counts.py
"""
import hashlib
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(ROOT, "Final ODO Dataset_v2026-06-10.xlsx")
GRAPH_PT_PATH = os.path.join(ROOT, "hetero_gnn", "processed_hetero_graph.pt")

REPORTED = {
    "compounds": 13_297, "targets": 73, "assays": 4_631,
    "documents": 1_069, "model_systems": 148, "activities": 37_353,
}
BIPARTITE_PRINTED = {"compounds": 13_396, "targets": 43, "activities": 37_362}


def safe(val):
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def slugify(text):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(text).strip())


def load_raw():
    df = pd.read_excel(EXCEL_PATH, sheet_name="Full Dataset")
    df.columns = [re.sub(r"\s+", "_", c.strip()) for c in df.columns]
    return df


def drop_corrupted_unit_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Verbatim logic from hetero_gnn/preprocess.py:drop_corrupted_unit_rows."""
    corrupted = (
        (raw["endpoint"] == "Ki")
        & raw["pchembl_value"].isna()
        & raw["endpoint_value"].notna()
        & (raw["unit_of_measurement"].isna() | (raw["unit_of_measurement"] == "M"))
    )
    return raw.loc[~corrupted].reset_index(drop=True), int(corrupted.sum())


def chembl_compound_key(row):
    cid = safe(row.get("chembl_compound_id"))
    if cid:
        return cid
    ikey = safe(row.get("rdkit_library_standard_inchi_key"))
    return f"inchikey_{ikey}" if ikey else None


def chembl_target_key(row):
    tcid = safe(row.get("chembl_target_id"))
    if tcid:
        return tcid
    tname = safe(row.get("target_name"))
    return slugify(tname) if tname else None


def hetero_compound_key(row):
    cid = safe(row.get("pubchem_cid"))
    if cid:
        return f"cid_{cid}"
    ikey = safe(row.get("rdkit_library_standard_inchi_key"))
    return f"inchikey_{ikey}" if ikey else None


def hetero_target_key(row):
    uid = safe(row.get("uniprot_protein_id"))
    if uid:
        return f"uniprot_{uid}"
    tname = safe(row.get("target_name"))
    return f"tname_{slugify(tname)}" if tname else None


def short_hash(*parts, n=16):
    """Verbatim logic from hetero_gnn/preprocess.py:short_hash."""
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:n]


def chembl_assay_key(row):
    """Verbatim logic from kg/build_kg.py:build_assay's key derivation.
    Rows with no chembl_assay_id get NO assay node at all (build_assay
    returns None), so this yields None for them rather than a fallback."""
    return safe(row.get("chembl_assay_id"))


def hetero_assay_key(row):
    """Verbatim logic from hetero_gnn/preprocess.py:assay_key_of."""
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


def chembl_document_key(row):
    """Verbatim logic from kg/build_kg.py:build_document's key derivation.
    Rows with no pmid/doi/chembl_document_id get NO document node
    (build_document returns None), so this yields None for them."""
    pmid = safe(row.get("pubmed_id"))
    if pmid:
        try:
            pmid = str(int(float(pmid)))
        except (ValueError, OverflowError):
            pass
    doi = safe(row.get("document_doi"))
    chdoc = safe(row.get("chembl_document_id"))
    return pmid or doi or chdoc


def hetero_document_key(row):
    """Verbatim logic from hetero_gnn/preprocess.py:document_key_of."""
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


def chembl_model_system_key(row):
    """Verbatim logic from kg/build_kg.py:build_model_system's key
    derivation. Rows with neither ncit_model_system_id nor
    ncit_model_system get NO model_system node (returns None)."""
    ms_val = safe(row.get("ncit_model_system"))
    ms_id = safe(row.get("ncit_model_system_id"))
    base = ms_id or (slugify(ms_val) if ms_val else None)
    if not base:
        return None
    cell_key = (safe(row.get("cellosaurus_cell_line_id"))
                or safe(row.get("clo_cell_line_id"))
                or safe(row.get("cell_line_name")) or "")
    tissue_key = safe(row.get("bto_tissue_id")) or safe(row.get("tissue_name")) or ""
    org_key = (safe(row.get("ncbi_target_taxonomy_id"))
               or safe(row.get("ncbi_target_taxonomy")) or "")
    return f"{base}_{slugify(cell_key)}_{slugify(tissue_key)}_{slugify(org_key)}"


def _ms_part(v):
    """Verbatim logic from hetero_gnn/preprocess.py:_ms_part."""
    v = safe(v)
    return v.strip().lower() if v else None


def hetero_model_system_key(row):
    """Verbatim logic from hetero_gnn/preprocess.py:model_system_key_of.
    Note: does NOT read ncit_model_system/_id at all."""
    parts = [
        _ms_part(row.get("cell_type_name")), _ms_part(row.get("cellosaurus_cell_line_id")),
        _ms_part(row.get("tissue_name")), _ms_part(row.get("ncit_animal_model")),
        _ms_part(row.get("ncit_animal_animal_model_strain")), _ms_part(row.get("ncbi_tissue_taxonomy")),
    ]
    if all(p is None for p in parts):
        return "unspecified_model_system"
    return "ms_" + "|".join(p or "_" for p in parts)


def load_graph_pt_counts():
    """Cross-check: node counts baked into the actual built hetero graph."""
    import torch
    data = torch.load(GRAPH_PT_PATH, weights_only=False)
    return {
        "assays": int(data["assay"].num_nodes),
        "documents": int(data["document"].num_nodes),
        "model_systems": int(data["model_system"].num_nodes),
    }


def main():
    raw = load_raw()
    n_raw = len(raw)
    print(f"Raw sheet: {n_raw:,} rows\n")

    filtered, n_dropped = drop_corrupted_unit_rows(raw)
    print(f"hetero-rule corrupt-row filter drops {n_dropped} rows "
          f"({n_raw} -> {len(filtered)})\n")

    print(f"pubchem_cid missing        : {raw['pubchem_cid'].apply(lambda v: safe(v) is None).mean()*100:.1f}% of rows")
    print(f"uniprot_protein_id missing : {raw['uniprot_protein_id'].apply(lambda v: safe(v) is None).mean()*100:.1f}% of rows")
    print(f"chembl_compound_id missing : {raw['chembl_compound_id'].apply(lambda v: safe(v) is None).mean()*100:.1f}% of rows")
    print(f"chembl_target_id missing   : {raw['chembl_target_id'].apply(lambda v: safe(v) is None).mean()*100:.1f}% of rows\n")

    def n_unique(df, keyfn):
        keys = df.apply(keyfn, axis=1)
        return keys.dropna().nunique(), keys

    n_c_chembl, _ = n_unique(raw, chembl_compound_key)
    n_t_chembl, _ = n_unique(raw, chembl_target_key)
    n_a_chembl, _ = n_unique(raw, chembl_assay_key)
    n_d_chembl, _ = n_unique(raw, chembl_document_key)
    n_m_chembl, _ = n_unique(raw, chembl_model_system_key)

    n_c_hetero, _ = n_unique(filtered, hetero_compound_key)
    n_t_hetero, _ = n_unique(filtered, hetero_target_key)
    n_a_hetero, _ = n_unique(filtered, hetero_assay_key)
    n_d_hetero, _ = n_unique(filtered, hetero_document_key)
    n_m_hetero, _ = n_unique(filtered, hetero_model_system_key)

    cols = ["compounds", "targets", "assays", "documents", "model_systems", "activities"]
    chembl_row = {
        "compounds": n_c_chembl, "targets": n_t_chembl, "assays": n_a_chembl,
        "documents": n_d_chembl, "model_systems": n_m_chembl, "activities": n_raw,
    }
    hetero_row = {
        "compounds": n_c_hetero, "targets": n_t_hetero, "assays": n_a_hetero,
        "documents": n_d_hetero, "model_systems": n_m_hetero, "activities": len(filtered),
    }

    def fmt_row(label, row):
        cells = "".join(f"{row[c]:>14,}" for c in cols)
        return f"{label:<38}{cells}"

    print("=" * 122)
    header = "".join(f"{c:>14}" for c in cols)
    print(f"{'Rule':<38}{header}")
    print("=" * 122)
    print(fmt_row("chembl-id rule, all rows", chembl_row))
    print(fmt_row("hetero rule, post-filter", hetero_row))
    print("=" * 122)
    print(fmt_row("Table 1 (reported)", REPORTED))
    bipartite_full = {**BIPARTITE_PRINTED, "assays": None, "documents": None, "model_systems": None}
    bp_cells = "".join(
        f"{bipartite_full[c]:>14,}" if bipartite_full[c] is not None else f"{'n/a':>14}"
        for c in cols
    )
    print(f"{'gnn/preprocess_bipartite_graph.py print':<38}{bp_cells}")
    print("=" * 122)

    print("\nCross-check against hetero_gnn/processed_hetero_graph.pt (built graph node counts):")
    try:
        graph_counts = load_graph_pt_counts()
        for k, v in graph_counts.items():
            match = "== Table 1" if v == REPORTED[k] else "!= Table 1"
            print(f"  {k:<16}: {v:>8,}  ({match}, Table 1 says {REPORTED[k]:,})")
    except Exception as e:
        graph_counts = {}
        print(f"  could not load {GRAPH_PT_PATH}: {e}")

    print("\nMatch check (compounds/targets/activities, existing):")
    print(f"  chembl-id rule   == bipartite printed counts? "
          f"{(n_c_chembl, n_t_chembl, n_raw) == tuple(BIPARTITE_PRINTED.values())}")
    print(f"  hetero rule == Table 1 reported (compounds/targets/activities)? "
          f"{(n_c_hetero, n_t_hetero, len(filtered)) == (REPORTED['compounds'], REPORTED['targets'], REPORTED['activities'])}")

    print("\nMatch check (assays/documents/model_systems, new):")
    for name, key in [("assays", "assays"), ("documents", "documents"), ("model_systems", "model_systems")]:
        chembl_val = chembl_row[key]
        hetero_val = hetero_row[key]
        reported = REPORTED[key]
        chembl_hit = chembl_val == reported
        hetero_hit = hetero_val == reported
        if hetero_hit:
            verdict = "hetero rule reproduces it"
        elif chembl_hit:
            verdict = "chembl rule reproduces it"
        else:
            verdict = "NEITHER rule reproduces it"
        print(f"  {name:<14}: chembl={chembl_val:,}  hetero={hetero_val:,}  "
              f"reported={reported:,}  -> {verdict}")


if __name__ == "__main__":
    main()
