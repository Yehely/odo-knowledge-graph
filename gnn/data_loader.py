"""
Extract compound, target, and activity data from the ODO TTL files.

Returns three DataFrames:
  compounds_df  — one row per compound, with SMILES + ADMET features
  targets_df    — one row per target, with type + species
  activities_df — one row per activity (qualifier="=" + valid pChEMBL only)

Usage:
    python -m gnn.data_loader          # prints summary stats
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

from gnn.config import (
    TTL_COMPOUNDS, TTL_ACTIVITIES, TTL_TARGETS, TTL_ASSAYS, TTL_DOCUMENTS,
    EXACT_QUALIFIER, MIN_PCHEMBL, MAX_PCHEMBL,
)

ODO  = Namespace("http://odo-project.org/ontology#")
ODOD = Namespace("http://odo-project.org/data#")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(node):
    """Return string value of an rdflib term, or None."""
    if node is None:
        return None
    s = str(node)
    return s if s.strip() else None


def _float(node):
    try:
        return float(str(node))
    except (TypeError, ValueError):
        return None


def _load_graph(*paths):
    g = Graph()
    for p in paths:
        if os.path.exists(p):
            g.parse(p, format="turtle")
        else:
            print(f"[WARNING] TTL file not found: {p}", file=sys.stderr)
    return g


# ---------------------------------------------------------------------------
# Compounds
# ---------------------------------------------------------------------------

_COMPOUND_QUERY = """
PREFIX odo:  <http://odo-project.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?uri ?smiles ?mw ?alogp ?ro5 ?maxphase ?radiolabeled
       ?qp_logpow ?qp_logs ?qp_donor ?qp_acceptor
       ?qp_logkhsa ?oral_abs ?qp_sasa
WHERE {
  ?uri a odo:Compound .
  OPTIONAL { ?uri odo:smiles            ?smiles }
  OPTIONAL { ?uri odo:molecularWeight   ?mw }
  OPTIONAL { ?uri odo:alogP             ?alogp }
  OPTIONAL { ?uri odo:ro5Violations     ?ro5 }
  OPTIONAL { ?uri odo:maxPhase          ?maxphase }
  OPTIONAL { ?uri odo:isRadiolabeled    ?radiolabeled }
  OPTIONAL { ?uri odo:qpLogPow          ?qp_logpow }
  OPTIONAL { ?uri odo:qpLogS            ?qp_logs }
  OPTIONAL { ?uri odo:qpDonorHB         ?qp_donor }
  OPTIONAL { ?uri odo:qpAcceptorHB      ?qp_acceptor }
  OPTIONAL { ?uri odo:qpLogKhsa         ?qp_logkhsa }
  OPTIONAL { ?uri odo:humanOralAbsorption ?oral_abs }
  OPTIONAL { ?uri odo:qpSASA            ?qp_sasa }
}
"""


def load_compounds(g: Graph) -> pd.DataFrame:
    rows = []
    for r in g.query(_COMPOUND_QUERY):
        rows.append({
            "uri":               str(r.uri),
            "smiles":            _val(r.smiles),
            "mw":                _float(r.mw),
            "alogp":             _float(r.alogp),
            "ro5_violations":    _float(r.ro5),
            "max_phase":         _float(r.maxphase),
            "is_radiolabeled":   1.0 if _val(r.radiolabeled) in ("true", "1", "yes", "t") else 0.0,
            "qp_logpow":         _float(r.qp_logpow),
            "qp_logs":           _float(r.qp_logs),
            "qp_donor_hb":       _float(r.qp_donor),
            "qp_acceptor_hb":    _float(r.qp_acceptor),
            "qp_logkhsa":        _float(r.qp_logkhsa),
            "human_oral_absorption": _float(r.oral_abs),
            "qp_sasa":           _float(r.qp_sasa),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset="uri").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

_TARGET_QUERY = """
PREFIX odo:  <http://odo-project.org/ontology#>

SELECT DISTINCT ?uri ?name ?type ?species
WHERE {
  ?uri a odo:Target .
  OPTIONAL { ?uri odo:targetName    ?name }
  OPTIONAL { ?uri odo:targetType    ?type }
  OPTIONAL { ?uri odo:targetSpecies ?species }
}
"""


def load_targets(g: Graph) -> pd.DataFrame:
    rows = []
    for r in g.query(_TARGET_QUERY):
        rows.append({
            "uri":     str(r.uri),
            "name":    _val(r.name)    or "Unknown",
            "type":    _val(r.type)    or "Unknown",
            "species": _val(r.species) or "Unknown",
        })
    df = pd.DataFrame(rows).drop_duplicates(subset="uri").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

_ACTIVITY_QUERY = """
PREFIX odo:  <http://odo-project.org/ontology#>

SELECT ?uri ?compound ?target ?pchembl ?qualifier ?endpoint_type ?exp_setting ?doc_year
WHERE {
  ?uri a odo:Activity ;
       odo:hasCompound ?compound ;
       odo:hasTarget   ?target .
  OPTIONAL { ?uri odo:pchemblValue         ?pchembl }
  OPTIONAL { ?uri odo:endpointQualifier    ?qualifier }
  OPTIONAL { ?uri odo:endpointType         ?endpoint_type }
  OPTIONAL { ?uri odo:hasAssay             ?assay_uri .
             ?assay_uri odo:experimentalSetting ?exp_setting }
  OPTIONAL { ?uri odo:publishedIn          ?doc .
             ?doc odo:documentYear         ?doc_year }
}
"""


def load_activities(g: Graph) -> pd.DataFrame:
    rows = []
    for r in g.query(_ACTIVITY_QUERY):
        rows.append({
            "uri":            str(r.uri),
            "compound_uri":   str(r.compound),
            "target_uri":     str(r.target),
            "pchembl":        _float(r.pchembl),
            "qualifier":      _val(r.qualifier) or "",
            "endpoint_type":  _val(r.endpoint_type) or "other",
            "exp_setting":    _val(r.exp_setting) or "other",
            "doc_year":       _float(r.doc_year),   # None if no year
        })
    df = pd.DataFrame(rows)

    # Keep only exact measurements with valid pChEMBL
    df = df[df["qualifier"] == EXACT_QUALIFIER].copy()
    df = df.dropna(subset=["pchembl"])
    df = df[(df["pchembl"] >= MIN_PCHEMBL) & (df["pchembl"] <= MAX_PCHEMBL)]
    df = df.drop_duplicates(subset="uri").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all():
    """
    Load and return (compounds_df, targets_df, activities_df).

    Activities are filtered to qualifier="=" and valid pChEMBL range.
    Only compounds and targets that appear in the filtered activities are kept.
    """
    print("Loading TTL files …")
    g_compounds  = _load_graph(TTL_COMPOUNDS)
    g_targets    = _load_graph(TTL_TARGETS)
    g_activities = _load_graph(TTL_ACTIVITIES)
    g_assays     = _load_graph(TTL_ASSAYS)
    g_documents  = _load_graph(TTL_DOCUMENTS)

    # Merge assay + document info for experimentalSetting and doc_year lookup
    g_act_full = g_activities + g_assays + g_documents

    print("  Extracting compounds …")
    compounds_df = load_compounds(g_compounds)

    print("  Extracting targets …")
    targets_df = load_targets(g_targets)

    print("  Extracting activities …")
    activities_df = load_activities(g_act_full)

    # Filter to only compounds/targets that appear in activities
    active_compounds = set(activities_df["compound_uri"])
    active_targets   = set(activities_df["target_uri"])

    compounds_df = compounds_df[compounds_df["uri"].isin(active_compounds)].reset_index(drop=True)
    targets_df   = targets_df[targets_df["uri"].isin(active_targets)].reset_index(drop=True)

    # Drop activities whose compound or target was removed (shouldn't happen, safety)
    compound_set = set(compounds_df["uri"])
    target_set   = set(targets_df["uri"])
    activities_df = activities_df[
        activities_df["compound_uri"].isin(compound_set) &
        activities_df["target_uri"].isin(target_set)
    ].reset_index(drop=True)

    return compounds_df, targets_df, activities_df


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    compounds_df, targets_df, activities_df = load_all()

    print(f"\n{'='*50}")
    print(f"Compounds : {len(compounds_df):,}  (with SMILES: {compounds_df['smiles'].notna().sum():,})")
    print(f"Targets   : {len(targets_df):,}")
    print(f"Activities: {len(activities_df):,}  (qualifier='=', valid pChEMBL)")
    print(f"\npChEMBL stats:")
    print(activities_df["pchembl"].describe().to_string())
    print(f"\nEndpoint type distribution:")
    print(activities_df["endpoint_type"].value_counts().to_string())
    print(f"\nTarget species distribution:")
    print(targets_df["species"].value_counts().to_string())
    print(f"\nSample activities:")
    print(activities_df.head(5).to_string())
