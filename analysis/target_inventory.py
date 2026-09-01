"""
Task 4 vehicle: builds the per-target inventory for the 43-target identity
rule (chembl_target_id -> slug(target_name) fallback — the rule used by
kg/build_kg.py and gnn/preprocess_bipartite_graph.py, i.e. the target set the
bipartite GNN actually trained/evaluated on).

For each of the 43 targets: name, UniProt ID(s) observed, organism, total
activity-row count, exact-qualifier row count, and temporal-test-set row
count (bipartite split: doc_year in [2016, 2020], same window
gnn/preprocess_bipartite_graph.py uses — see TEMPORAL_CUTOFF/TEST_MAX_YEAR
there).

Read-only: only reads the .xlsx. Prints the full table to stdout.
Run:
    python3 analysis/target_inventory.py
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(ROOT, "Final ODO Dataset_v2026-06-10.xlsx")

MIN_PCHEMBL, MAX_PCHEMBL = 2.0, 15.0
TEMPORAL_CUTOFF, TEST_MAX_YEAR = 2016, 2020  # bipartite pipeline's actual test window


def safe(val):
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def slugify(text):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(text).strip())


def target_key(row):
    tcid = safe(row.get("chembl_target_id"))
    if tcid:
        return tcid
    tname = safe(row.get("target_name"))
    return slugify(tname) if tname else None


def species_of(row):
    taxon = safe(row.get("ncbi_target_taxonomy"))
    if not taxon:
        return "other"
    parts = taxon.strip().split()
    if len(parts) >= 2:
        return parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])
    return taxon.strip()


def build_inventory():
    df = pd.read_excel(EXCEL_PATH, sheet_name="Full Dataset")
    df.columns = [re.sub(r"\s+", "_", c.strip()) for c in df.columns]

    df["target_key"] = df.apply(target_key, axis=1)
    df = df[df["target_key"].notna()].copy()

    qual = df["endpoint_qualifier"].apply(safe).fillna("")
    pch = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["is_exact"] = (qual == "=") & pch.notna() & (pch >= MIN_PCHEMBL) & (pch <= MAX_PCHEMBL)

    doc_year = pd.to_numeric(df["document_year"], errors="coerce")
    df["is_test"] = doc_year.between(TEMPORAL_CUTOFF, TEST_MAX_YEAR)

    df["species"] = df.apply(species_of, axis=1)
    df["uniprot"] = df["uniprot_protein_id"].apply(safe)

    rows = []
    for key, g in df.groupby("target_key", sort=False):
        name = g["target_name"].apply(safe).dropna()
        name = name.mode().iloc[0] if len(name) else "Unknown"
        species_vals = g["species"].value_counts()
        species = species_vals.index[0] if len(species_vals) else "other"
        uniprots = sorted(set(g["uniprot"].dropna()))
        rows.append({
            "target_key": key,
            "name": name,
            "uniprot": ";".join(uniprots) if uniprots else "(none)",
            "n_uniprot_distinct": len(uniprots),
            "species": species,
            "n_rows": len(g),
            "n_exact": int(g["is_exact"].sum()),
            "n_test": int(g["is_test"].sum()),
        })

    inv = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    return inv, df


OPIOID_RECEPTOR_PATTERNS = {
    "MOR": r"\bmu[- ]?type\b|\bmu[- ]?opioid\b|OPRM1",
    "DOR": r"\bdelta[- ]?type\b|\bdelta[- ]?opioid\b|OPRD1",
    "KOR": r"\bkappa[- ]?type\b|\bkappa[- ]?opioid\b|OPRK1",
    "NOP": r"nociceptin|orphanin|opioid[- ]receptor[- ]like|OPRL1",
}


def classify_receptor(name: str) -> str | None:
    for label, pat in OPIOID_RECEPTOR_PATTERNS.items():
        if re.search(pat, name, flags=re.IGNORECASE):
            return label
    return None


def main():
    inv, df = build_inventory()
    inv["receptor"] = inv["name"].apply(classify_receptor)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(inv.to_string(index=False))

    print(f"\nTotal targets: {len(inv)}")
    print(f"Total activity rows across all targets: {inv['n_rows'].sum():,} "
          f"(raw sheet has {len(df):,} target-resolvable rows)")

    receptor_rows = inv[inv["receptor"].notna()]
    print(f"\nOpioid-receptor-labeled targets (any species): {len(receptor_rows)}")
    print(receptor_rows[["target_key", "name", "species", "uniprot", "n_rows", "n_exact", "n_test"]].to_string(index=False))

    total_rows = inv["n_rows"].sum()
    total_test = inv["n_test"].sum()
    rec_rows = receptor_rows["n_rows"].sum()
    rec_test = receptor_rows["n_test"].sum()
    print(f"\nReceptor-labeled share of all activity rows : {rec_rows:,}/{total_rows:,} = {rec_rows/total_rows*100:.1f}%")
    print(f"Receptor-labeled share of temporal test rows : {rec_test:,}/{total_test:,} = {rec_test/total_test*100:.1f}%")

    print(f"\nTargets with <50 rows: {(inv['n_rows'] < 50).sum()}")
    print(f"Targets with <10 rows: {(inv['n_rows'] < 10).sum()}")

    inv.to_csv(os.path.join(os.path.dirname(__file__), "target_inventory.csv"), index=False)
    print(f"\nWrote {os.path.join(os.path.dirname(__file__), 'target_inventory.csv')}")


if __name__ == "__main__":
    main()
