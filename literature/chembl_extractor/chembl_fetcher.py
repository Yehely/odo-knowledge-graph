#!/usr/bin/env python3
"""
ChEMBL Automated Data Fetcher  –  v2 (improved accuracy)
=========================================================
Fetches molecule, activity, assay, and target data from ChEMBL,
PubChem, UniProt, InterPro, OLS4 (BTO/PRO) and PubMed.
Produces an Excel file matching the full 131-column dataset format.

Usage:
    python chembl_fetcher.py --ids CHEMBL25 CHEMBL59
    python chembl_fetcher.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python chembl_fetcher.py --file molecules.txt --target CHEMBL233
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdFreeSASA import CalcSASA, classifyAtoms

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
BASE_URL      = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
UNIPROT_BASE  = "https://rest.uniprot.org/uniprotkb"
INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"
OLS4_BASE     = "https://www.ebi.ac.uk/ols4/api"
PUBMED_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

HEADERS             = {"Accept": "application/json"}
RATE_LIMIT_DELAY    = 0.25   # seconds between calls


# ---------------------------------------------------------------------------
# Generic HTTP helpers
# ---------------------------------------------------------------------------

def _get(url, params=None, data=None, method="GET", retries=3, timeout=20):
    """Robust HTTP GET/POST with retry and rate-limit handling."""
    for attempt in range(retries):
        try:
            if method == "POST":
                r = requests.post(url, params=params, data=data,
                                  headers=HEADERS, timeout=timeout)
            else:
                r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def api_get(endpoint, params=None):
    r = _get(f"{BASE_URL}/{endpoint}", params=params)
    return r.json() if r else None


def api_get_all(endpoint, params=None, limit=1000):
    if params is None:
        params = {}
    params["limit"]  = limit
    params["offset"] = 0
    all_results = []
    while True:
        data = api_get(endpoint, params)
        if data is None:
            break
        keys = [k for k in data if k != "page_meta"]
        if not keys:
            break
        all_results.extend(data[keys[0]])
        if data.get("page_meta", {}).get("next") is None:
            break
        params["offset"] += limit
        time.sleep(RATE_LIMIT_DELAY)
    return all_results


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
_assay_cache    = {}
_target_cache   = {}
_doc_cache      = {}
_cell_cache     = {}
_bto_cache      = {}
_interpro_cache = {}
_uniprot_cache  = {}


# ---------------------------------------------------------------------------
# ChEMBL fetchers
# ---------------------------------------------------------------------------

def fetch_molecule(chembl_id):
    return api_get(f"molecule/{chembl_id}.json")

def fetch_activities(chembl_id):
    return api_get_all("activity.json", params={"molecule_chembl_id": chembl_id})

def fetch_assay(assay_id):
    return api_get(f"assay/{assay_id}.json")

def fetch_target(target_id):
    return api_get(f"target/{target_id}.json")

def fetch_document(doc_id):
    return api_get(f"document/{doc_id}.json")

def fetch_cell_line(cell_chembl_id):
    return api_get(f"cell_line/{cell_chembl_id}.json")

def resolve_smiles_to_chembl(smiles):
    data = api_get("molecule.json",
                   params={"molecule_structures__canonical_smiles": smiles})
    if data and data.get("molecules"):
        return data["molecules"][0].get("molecule_chembl_id")
    return None

def get_assay(assay_id):
    if assay_id not in _assay_cache:
        _assay_cache[assay_id] = fetch_assay(assay_id)
        time.sleep(RATE_LIMIT_DELAY)
    return _assay_cache[assay_id]

def get_target(target_id):
    if target_id not in _target_cache:
        _target_cache[target_id] = fetch_target(target_id)
        time.sleep(RATE_LIMIT_DELAY)
    return _target_cache[target_id]

def get_document(doc_id):
    if doc_id not in _doc_cache:
        _doc_cache[doc_id] = fetch_document(doc_id)
        time.sleep(RATE_LIMIT_DELAY)
    return _doc_cache[doc_id]

def get_cell_line(cell_chembl_id):
    if cell_chembl_id not in _cell_cache:
        _cell_cache[cell_chembl_id] = fetch_cell_line(cell_chembl_id)
        time.sleep(RATE_LIMIT_DELAY)
    return _cell_cache[cell_chembl_id]


# ---------------------------------------------------------------------------
# PubChem helpers
# ---------------------------------------------------------------------------

def get_pubchem_data(inchikey):
    """Get PubChem CID + IUPAC name from an InChIKey."""
    if not inchikey:
        return None, None
    try:
        # Step 1: InChIKey → CID
        r_cid = _get(f"{PUBCHEM_BASE}/compound/inchikey/{inchikey}/cids/JSON", timeout=15)
        if not r_cid:
            return None, None
        cids = r_cid.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None, None
        cid = cids[0]
        # Step 2: CID → IUPAC name
        r_name = _get(f"{PUBCHEM_BASE}/compound/cid/{cid}/property/IUPACName/JSON",
                      timeout=15)
        if r_name:
            props = r_name.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            return cid, props.get("IUPACName")
        return cid, None
    except Exception:
        pass
    return None, None


def get_pubchem_parent_structure(stereo_inchi, stereo_inchikey):
    """
    Get the non-stereospecific (flat) parent structure from PubChem.
    Uses the flat InChI (stereo layers stripped) to look up the parent CID,
    then retrieves InChIKey, InChI, MolecularFormula, MolecularWeight.
    Returns a dict (may be empty on failure).
    """
    if not stereo_inchi:
        return {}
    flat_inchi = strip_stereo_from_inchi(stereo_inchi)
    try:
        # POST flat InChI → parent CID
        r_cid = _get(f"{PUBCHEM_BASE}/compound/inchi/cids/JSON",
                     data={"inchi": flat_inchi},
                     method="POST", timeout=20)
        if not r_cid:
            return {}
        cids = r_cid.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return {}
        cid = cids[0]
        # CID → properties
        r_props = _get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}"
            "/property/InChI,InChIKey,MolecularFormula,MolecularWeight/JSON",
            timeout=15)
        if r_props:
            props = r_props.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            return props
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# PubMed helper
# ---------------------------------------------------------------------------

def get_pubmed_abstract(pmid):
    """Fetch PubMed abstract text for NLP extraction."""
    if not pmid:
        return ""
    try:
        r = _get(f"{PUBMED_BASE}/efetch.fcgi",
                 params={"db": "pubmed", "id": str(int(pmid)),
                         "rettype": "abstract", "retmode": "text"},
                 timeout=15)
        return r.text if r else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# UniProt helper
# ---------------------------------------------------------------------------

def get_uniprot_data(uniprot_id):
    """Fetch UniProt entry: returns dict with PRO ID, G-protein coupling info, etc."""
    if not uniprot_id:
        return {}
    if uniprot_id in _uniprot_cache:
        return _uniprot_cache[uniprot_id]
    result = {}
    try:
        r = _get(f"{UNIPROT_BASE}/{uniprot_id}.json", timeout=15)
        if r:
            data = r.json()
            xrefs = data.get("uniProtKBCrossReferences", [])
            # PRO cross-reference
            for x in xrefs:
                if x.get("database") == "PRO":
                    result["pro_id"] = x.get("id")   # e.g. "PR:P32300"
                    break
            # InterPro cross-references (full set with EntryName)
            result["interpro_xrefs"] = [
                {"id": x.get("id"), "name": x.get("properties", [{}])[0].get("value", "")}
                for x in xrefs if x.get("database") == "InterPro"
            ]
            # Comments – look for Gi/Gq/Gs coupling
            for c in data.get("comments", []):
                text = str(c)
                if any(kw in text.lower() for kw in ("gi", "gq", "g-protein", "adenylate cyclase")):
                    result["gprotein_comment"] = text[:500]
                    break
            # Recommended protein name
            prot_names = (data.get("proteinDescription", {})
                              .get("recommendedName", {})
                              .get("fullName", {}))
            if isinstance(prot_names, dict):
                result["recommended_name"] = prot_names.get("value", "")
            elif isinstance(prot_names, str):
                result["recommended_name"] = prot_names
    except Exception:
        pass
    _uniprot_cache[uniprot_id] = result
    time.sleep(RATE_LIMIT_DELAY)
    return result


# ---------------------------------------------------------------------------
# InterPro helper
# ---------------------------------------------------------------------------

_pro_name_cache = {}

def _get_pro_id_from_protein_name(protein_name):
    """
    Look up canonical PRO ontology ID (e.g. PR:000001579) via OLS4 using
    the protein's common name. Returns None on failure.
    """
    if not protein_name:
        return None
    key = protein_name.lower().strip()
    if key in _pro_name_cache:
        return _pro_name_cache[key]
    try:
        r = _get(f"{OLS4_BASE}/search",
                 params={"q": protein_name, "ontology": "pr", "rows": 10},
                 timeout=15)
        if r:
            docs = r.json().get("response", {}).get("docs", [])
            for d in docs:
                obo_id = d.get("obo_id", "")
                label  = d.get("label", "")
                # Prefer species-independent entries (no "(species)" suffix)
                if obo_id.startswith("PR:") and not re.search(r"\(.*?\)", label):
                    _pro_name_cache[key] = obo_id
                    time.sleep(RATE_LIMIT_DELAY)
                    return obo_id
    except Exception:
        pass
    _pro_name_cache[key] = None
    return None


def get_interpro_name(ipr_id):
    """Return the human-readable entry name for an InterPro ID (lowercased)."""
    if not ipr_id:
        return None
    if ipr_id in _interpro_cache:
        return _interpro_cache[ipr_id]
    try:
        r = _get(f"{INTERPRO_BASE}/entry/interpro/{ipr_id}/",
                 params={"format": "json"}, timeout=10)
        if r:
            name = (r.json().get("metadata", {})
                            .get("name", {})
                            .get("name", ""))
            result = name.lower() if name else None
            _interpro_cache[ipr_id] = result
            time.sleep(RATE_LIMIT_DELAY)
            return result
    except Exception:
        pass
    _interpro_cache[ipr_id] = None
    return None


# ---------------------------------------------------------------------------
# BTO (BRENDA Tissue Ontology) lookup via OLS4
# ---------------------------------------------------------------------------

def get_bto_id(tissue_name):
    """Return BTO:XXXXXXX ID for a tissue name via OLS4 search."""
    if not tissue_name:
        return None
    key = tissue_name.lower().strip()
    if key in _bto_cache:
        return _bto_cache[key]
    try:
        r = _get(f"{OLS4_BASE}/search",
                 params={"q": key, "ontology": "bto",
                         "exact": "true", "type": "class"},
                 timeout=10)
        if r:
            docs = r.json().get("response", {}).get("docs", [])
            for d in docs:
                label = (d.get("label") or "").lower()
                if label == key:
                    obo_id = d.get("obo_id", "")   # e.g. "BTO:0000620"
                    _bto_cache[key] = obo_id
                    time.sleep(RATE_LIMIT_DELAY)
                    return obo_id
    except Exception:
        pass
    _bto_cache[key] = None
    return None


# ---------------------------------------------------------------------------
# SMILES / InChI stereo stripping
# ---------------------------------------------------------------------------

_STEREO_ATOM_RE = re.compile(r'\[([A-Z][a-z]?)@@?H?\]')

def strip_stereo_from_smiles(smiles):
    """
    Remove stereochemistry marks from a SMILES string to produce a flat
    (non-stereospecific) canonical SMILES.
    """
    if not smiles:
        return smiles
    s = smiles
    # Replace stereogenic atoms: [C@@H] -> C, [C@H] -> C, [C@@] -> [C], [C@] -> [C]
    s = re.sub(r'\[C@@H\]', 'C', s)
    s = re.sub(r'\[C@H\]',  'C', s)
    s = re.sub(r'\[C@@\]',  '[C]', s)
    s = re.sub(r'\[C@\]',   '[C]', s)
    s = re.sub(r'\[N@@H\]', 'N', s)
    s = re.sub(r'\[N@H\]',  'N', s)
    s = re.sub(r'\[N@@\]',  '[N]', s)
    s = re.sub(r'\[N@\]',   '[N]', s)
    s = re.sub(r'\[S@@\]',  '[S]', s)
    s = re.sub(r'\[S@\]',   '[S]', s)
    # Remove remaining @ marks
    s = s.replace('@@', '').replace('@', '')
    # Remove E/Z bond stereo
    s = s.replace('/', '').replace('\\', '')
    return s


def strip_stereo_from_inchi(inchi):
    """Remove /t /m /s stereo layers from a standard InChI string."""
    if not inchi:
        return inchi
    return re.sub(r'/[tms][^/]*', '', inchi)


# ---------------------------------------------------------------------------
# NLP / pattern-based extraction from assay descriptions
# ---------------------------------------------------------------------------

# Pharmacological role patterns (ordered: most specific first)
ROLE_PATTERNS = [
    (re.compile(r'\binverse\s+agonist\b', re.I),
     'inverse agonist',  'http://purl.obolibrary.org/obo/CHEBI_90847'),
    (re.compile(r'\bpartial\s+agonist\b', re.I),
     'partial agonist',  'http://purl.obolibrary.org/obo/CHEBI_77971'),
    (re.compile(r'\bagonist\b',           re.I),
     'agonist',          'http://purl.obolibrary.org/obo/CHEBI_48705'),
    (re.compile(r'\bantagonist\b',        re.I),
     'antagonist',       'http://purl.obolibrary.org/obo/CHEBI_48706'),
    (re.compile(r'\binhibitor\b|\binhibit(ion|ory|ing)\b', re.I),
     'inhibitor',        'http://purl.obolibrary.org/obo/CHEBI_35222'),
    (re.compile(r'\bactivator\b',         re.I),
     'activator',        'http://purl.obolibrary.org/obo/CHEBI_35224'),
    (re.compile(r'\bmodulator\b',         re.I),
     'modulator',        'http://purl.obolibrary.org/obo/CHEBI_50846'),
    (re.compile(r'\bblocker\b',           re.I),
     'blocker',          'http://purl.obolibrary.org/obo/CHEBI_67105'),
]

CHEBI_ROLE_MAP = {r: u for _, r, u in ROLE_PATTERNS}

# Tissue patterns: regex → standard tissue name
TISSUE_PATTERNS = [
    (re.compile(r'\bvas\s+deferens\b', re.I), 'vas deferens'),
    (re.compile(r'\bileum\b',          re.I), 'ileum'),
    (re.compile(r'\batri(um|a)\b',     re.I), 'cardiac atrium'),
    (re.compile(r'\baort(a|ic)\b',     re.I), 'aorta'),
    (re.compile(r'\bcolon\b',          re.I), 'colon'),
    (re.compile(r'\bbladder\b',        re.I), 'urinary bladder'),
    (re.compile(r'\bjejunum\b',        re.I), 'jejunum'),
    (re.compile(r'\bduodenum\b',       re.I), 'duodenum'),
    (re.compile(r'\bhippocampus\b',    re.I), 'hippocampus'),
    (re.compile(r'\bcerebral\s+cortex\b', re.I), 'cerebral cortex'),
    (re.compile(r'\bcortex\b',         re.I), 'cortex'),
    (re.compile(r'\bbrain\b',          re.I), 'brain'),
    (re.compile(r'\bliver\b|\bhepatic\b', re.I), 'liver'),
    (re.compile(r'\bkidney\b|\brenal\b',  re.I), 'kidney'),
    (re.compile(r'\blung\b',           re.I), 'lung'),
    (re.compile(r'\bheart\b',          re.I), 'heart'),
    (re.compile(r'\bspleen\b',         re.I), 'spleen'),
    (re.compile(r'\btestis\b|\btesticular\b', re.I), 'testis'),
    (re.compile(r'\buterus\b|\buterine\b',    re.I), 'uterus'),
    (re.compile(r'\bstomach\b|\bgastric\b',   re.I), 'stomach'),
    (re.compile(r'\bpancreas\b',       re.I), 'pancreas'),
    (re.compile(r'\badrenal\b',        re.I), 'adrenal gland'),
    (re.compile(r'\bplatelet\b',       re.I), 'blood platelet'),
]

# Assay method patterns → dict of BAO fields
ASSAY_METHOD_PATTERNS = [
    (re.compile(r'electrically[\s-](induced|stimulated)\s+contraction', re.I), {
        'bao_bioassay_1':          'smooth muscle contraction assay',
        'bao_bioasssay_1_id':      'http://www.bioassayontology.org/bao#BAO_0013039',
        'bao_assay_method':        'organ bath method',
        'bao_assay_method_id':     'http://www.bioassayontology.org/bao#BAO_0050007',
        'physical_detection_method':    'isometric tension recording method',
        'physical_detection_method_id': 'http://www.bioassayontology.org/bao#BAO_0050017',
        'odo_assay_endpoint_description': 'physiological enpoint',
    }),
    (re.compile(r'\[35S\]GTP.{0,5}S|GTP.{0,3}gamma.{0,3}S\s+binding', re.I), {
        'bao_bioassay_1':          'G protein activation assay',
        # bao_bioasssay_1_id is null in DB for [35S]GTPgammaS assays
        'bao_bioassay_2':          'radioligand binding assay',
        'bao_bioasssay_2_id':      'http://www.bioassayontology.org/bao#BAO_0002776',
        'bao_assay_method':        'radioligand binding method',
        'bao_assay_method_id':     'http://www.bioassayontology.org/bao#BAO_0002776',
        'physical_detection_method':    'scintillation counting',
        'physical_detection_method_id': 'http://www.bioassayontology.org/bao#BAO_0000405',
        'odo_assay_endpoint_description': 'biochemical endpoint',
    }),
    (re.compile(
        r'radioligand\s+binding|competitive\s+binding|'
        r'displacement.{0,30}(?:binding|assay)|'
        r'\[(?:3H|125I|14C)\].{0,30}(?:displacement|binding\s+affinity)|'
        r'binding\s+affinity.{0,30}\[(?:3H|125I|14C)\]',
        re.I), {
        'bao_bioassay_1':          'radioligand binding assay',
        'bao_bioasssay_1_id':      'http://www.bioassayontology.org/bao#BAO_0002776',
        'bao_assay_method':        'radioligand binding method',
        'bao_assay_method_id':     'http://www.bioassayontology.org/bao#BAO_0002776',
        'physical_detection_method':    'scintillation counting',
        'physical_detection_method_id': 'http://www.bioassayontology.org/bao#BAO_0000405',
        'odo_assay_endpoint_description': 'binding endpoint',
    }),
    (re.compile(r'\bHTRF\b|homogeneous\s+time.?resolved\s+fluorescence', re.I), {
        'bao_assay_method':        'homogeneous time-resolved fluorescence method',
        'bao_assay_method_id':     'http://www.bioassayontology.org/bao#BAO_0000125',
        'physical_detection_method':    'time-resolved fluorescence measurement',
        'physical_detection_method_id': 'http://www.bioassayontology.org/bao#BAO_0002966',
        'odo_assay_endpoint_description': 'biochemical endpoint',
    }),
    (re.compile(r'\bcAMP\b|cyclic\s+AMP', re.I), {
        'bao_bioassay_1':          'cAMP assay',
        'bao_bioasssay_1_id':      'http://www.bioassayontology.org/bao#BAO_0002993',
        'odo_assay_endpoint_description': 'biochemical endpoint',
    }),
    (re.compile(r'\bELISA\b', re.I), {
        'bao_assay_method':        'enzyme-linked immunosorbent assay',
        'bao_assay_method_id':     'http://www.bioassayontology.org/bao#BAO_0000019',
        'odo_assay_endpoint_description': 'biochemical endpoint',
    }),
    (re.compile(r'\bfluorescence\b|\bFRET\b', re.I), {
        'bao_assay_method':        'fluorescence method',
        'bao_assay_method_id':     'http://www.bioassayontology.org/bao#BAO_0000009',
        'physical_detection_method':    'fluorescence intensity measurement',
        'physical_detection_method_id': 'http://www.bioassayontology.org/bao#BAO_0002993',
        'odo_assay_endpoint_description': 'biochemical endpoint',
    }),
]

# Standardized ODO assay names
ODO_ASSAY_PATTERNS = [
    (re.compile(r'guinea\s+pig\s+ileum',  re.I),
     'Electrically-stimulated guinea pig ileum (GPI) assay'),
    (re.compile(r'mouse\s+vas\s+deferens', re.I),
     'Electrically-stimulated mouse vas deferens (MVD) assay'),
    (re.compile(r'rat\s+vas\s+deferens',  re.I),
     'Electrically-stimulated rat vas deferens (RVD) assay'),
    (re.compile(r'rabbit\s+vas\s+deferens', re.I),
     'Electrically-stimulated rabbit vas deferens assay'),
    (re.compile(r'binding\s+affinity|affinity.{0,20}radioligand', re.I),
     'Affinity radioligand binding assay'),
    (re.compile(r'radioligand\s+binding|displacement.{0,30}binding', re.I),
     'radioligand binding assay'),
    (re.compile(r'\[35S\]GTP',             re.I),
     '[35S]GTPgammaS radioligand binding assay'),
    (re.compile(r'cAMP\s+accumulation',    re.I),
     'cAMP accumulation assay'),
    (re.compile(r'\bcAMP\b',               re.I),
     'cAMP assay'),
]

# Correct BAO experimental-setting URIs (verified against DB)
# BAO_0020007 = in vitro cell-based; BAO_0020008 = in vitro cell-free; BAO_0020009 = in vivo; BAO_0020006 = ex vivo
BAO_SETTING_MAP = {
    "In vitro":           ("in vitro", "http://www.bioassayontology.org/bao#BAO_0020007"),
    "In vitro cell-free": ("in vitro", "http://www.bioassayontology.org/bao#BAO_0020008"),
    "In vivo":            ("in vivo",  "http://www.bioassayontology.org/bao#BAO_0020009"),
    "Ex vivo":            ("ex vivo",  "http://www.bioassayontology.org/bao#BAO_0020006"),
}

# Assay format codes and labels
BAO_FORMAT_LABELS = {
    "tissue-based format":   "http://www.bioassayontology.org/bao#BAO_0000221",
    "cell-based format":     "http://www.bioassayontology.org/bao#BAO_0000219",
    "cell-free format":      "http://www.bioassayontology.org/bao#BAO_0000366",
    "organism-based format": "http://www.bioassayontology.org/bao#BAO_0000218",
    "single protein format": "http://www.bioassayontology.org/bao#BAO_0000357",
}

# Format → experimental setting key
FORMAT_TO_SETTING = {
    "tissue-based format":   "Ex vivo",
    "cell-based format":     "In vitro",
    "cell-free format":      "In vitro cell-free",
    "organism-based format": "In vivo",
    "single protein format": "In vitro",
}

# BAO bioassay type mapping: ChEMBL assay_type_description (lower) → (label, BAO_URI)
# NOTE: "binding" uses "binding type" convention (BAO_0002989 is newer, verified in DB).
#       "functional" keeps no "type" suffix (older convention, as stored in DB).
BAO_TYPE_MAP = {
    "binding":    ("binding type", "http://www.bioassayontology.org/bao#BAO_0002989"),
    "functional": ("functional",   "http://www.bioassayontology.org/bao#BAO_0000010"),
    "admet":      ("admet",        "http://www.bioassayontology.org/bao#BAO_0000011"),
    "toxicity":   ("toxicity",     "http://www.bioassayontology.org/bao#BAO_0000012"),
}

# NCIT model system: format → (label, ncit_id)
NCIT_MODEL_SYSTEM_MAP = {
    "cell-based format":     ("cell-line",  "C16403"),
    "tissue-based format":   ("tissue",     "C12801"),
    "organism-based format": ("organism",   "C14250"),
    "cell-free format":      ("cell-line",  "C16403"),
}

# Source description map
SOURCE_MAP = {
    1:  "Scientific Literature",
    2:  "GSK Published Kinase Inhibitor Set",
    7:  "PubChem BioAssays",
    9:  "DrugMatrix",
    11: "FDA Approval Packages",
    12: "Manually Added Small Molecules",
    13: "Patent Bioactivity Data",
    26: "USP Dictionary of USAN and International Drug Names",
    35: "Open TG-GATEs",
    36: "MMV Malaria Box",
    37: "BindingDB",
    38: "TP-search TransPorter DataBase",
    39: "Deposited Supplementary Bioactivity Data",
    40: "ChEMBL",
    41: "Curation",
    42: "CO-ADD Antimicrobial Screening Data",
}

# PANTHER family → DTO GPCR category
# Based on Drug Target Ontology (DTO) classification
PANTHER_TO_DTO = {
    "PTHR24229": ("peptidic GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390004"),
    "PTHR10661": ("peptidic GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390004"),
    "PTHR10480": ("peptidic GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390004"),
    "PTHR10168": ("aminergic GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390005"),
    "PTHR10184": ("aminergic GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390005"),
    "PTHR18893": ("aminergic GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390005"),
    "PTHR10165": ("aminergic GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390005"),
    "PTHR24245": ("lipid GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390006"),
    "PTHR10257": ("orphan GPCR",
                  "http://www.drugtargetontology.org/dto/DTO_02390009"),
}

# InterPro category → G-protein coupling (NCIT)
# Maps broad receptor subfamily to Gi/Gq/Gs coupling
INTERPRO_TO_GPROTEIN = {
    "IPR001418": ("G(i) Alpha", "C19275"),   # Opioid receptor → Gi
    "IPR000321": ("G(i) Alpha", "C19275"),   # Delta opioid receptor → Gi
    "IPR001172": ("G(i) Alpha", "C19275"),   # Mu opioid receptor → Gi
    "IPR000276": None,                        # Generic GPCR → can't determine
}

# Organism scientific name → common name (for ncit_animal_model)
ORGANISM_COMMON_NAME = {
    "Cavia porcellus":            "guinea pig",
    "Rattus norvegicus":          "rat",
    "Mus musculus":               "mouse",
    "Homo sapiens":               "human",
    "Oryctolagus cuniculus":      "rabbit",
    "Sus scrofa":                 "pig",
    "Canis lupus familiaris":     "dog",
    "Felis catus":                "cat",
    "Macaca mulatta":             "rhesus monkey",
    "Macaca fascicularis":        "cynomolgus monkey",
    "Mesocricetus auratus":       "hamster",
    "Meriones unguiculatus":      "gerbil",
}

# Route of administration extraction patterns (in order of specificity)
_ROUTE_PATTERNS = [
    (re.compile(r'intracerebroventricular', re.I), "intracerebroventricular"),
    (re.compile(r'intracerebral',           re.I), "intracerebral"),
    (re.compile(r'intrathecal',             re.I), "intrathecal"),
    (re.compile(r'intramuscular|i\.m\.',    re.I), "intramuscular"),
    (re.compile(r'intraperitoneal|i\.p\.',  re.I), "intraperitoneal"),
    (re.compile(r'intravenous|i\.v\.',      re.I), "intravenous"),
    (re.compile(r'subcutaneous|s\.c\.',     re.I), "subcutaneous"),
    (re.compile(r'\boral(?:ly)?\b|p\.o\.', re.I), "oral"),
    (re.compile(r'intranasal',              re.I), "intranasal"),
]


def extract_route_of_administration(desc):
    """Extract route of administration from assay description."""
    if not desc:
        return None
    for pattern, label in _ROUTE_PATTERNS:
        if pattern.search(desc):
            return label
    return None


# Target name normalization
_TARGET_NAME_RE = [
    (re.compile(r'-type\s+(opioid)',    re.I), r' \1'),
    (re.compile(r'-type\s+(receptor)',  re.I), r' \1'),
    (re.compile(r'\s+(receptor)\s+type\b', re.I), r' \1'),
]


def normalize_target_name(name):
    """Normalize ChEMBL target name to standard short form."""
    if not name:
        return name
    for pattern, repl in _TARGET_NAME_RE:
        name = pattern.sub(repl, name)
    return name


# ---------------------------------------------------------------------------
# NLP extraction functions
# ---------------------------------------------------------------------------

def extract_role_from_text(text):
    """Extract pharmacological role (name, chebi_id) from any text."""
    if not text:
        return None, None
    for pattern, role, chebi in ROLE_PATTERNS:
        if pattern.search(text):
            return role, chebi
    return None, None


def extract_tissue_from_text(text):
    """Extract tissue name from assay description."""
    if not text:
        return None
    for pattern, tissue in TISSUE_PATTERNS:
        if pattern.search(text):
            return tissue
    return None


def classify_assay_methods(description):
    """
    Classify assay methods from description.
    Returns a dict with bao_bioassay_1, bao_assay_method, physical_detection_method, etc.
    """
    result = {}
    if not description:
        return result
    for pattern, fields in ASSAY_METHOD_PATTERNS:
        if pattern.search(description):
            result.update(fields)
            break
    return result


def classify_odo_assay(description):
    """Return a standardized ODO assay name from description."""
    if not description:
        return None
    for pattern, name in ODO_ASSAY_PATTERNS:
        if pattern.search(description):
            return name
    return None


def normalize_journal_abbrev(journal):
    """
    Normalize ChEMBL journal abbreviation to match DB format.
    The DB is inconsistent (some journals have periods, some don't).
    Use a known mapping for common journals; pass others through unchanged.
    """
    if not journal:
        return journal
    _MAP = {
        "J Med Chem":              "J. Med. Chem.",
        "Bioorg Med Chem Lett":    "Bioorg. Med. Chem. Lett.",
        "Bioorg Med Chem":         "Bioorg. Med. Chem.",
        "J Pharmacol Exp Ther":    "J. Pharmacol. Exp. Ther.",
        "Eur J Pharmacol":         "Eur. J. Pharmacol.",
        "Br J Pharmacol":          "Br. J. Pharmacol.",
        "Life Sci":                "Life Sci.",
        "Neurosci Lett":           "Neurosci. Lett.",
        "Mol Pharmacol":           "Mol. Pharmacol.",
        "J Pharm Sci":             "J. Pharm. Sci.",
        "J Nat Prod":              "J. Nat. Prod.",
        "J Biol Chem":             "J. Biol. Chem.",
        "Eur J Med Chem":          "Eur. J. Med. Chem.",
        "J Neurochem":             "J. Neurochem.",
        "J Chem Inf Model":        "J. Chem. Inf. Model.",
    }
    return _MAP.get(journal, journal)


# Radiolabeled reference compound: [3H]-DAMGO, [125I]-Naloxone, [35S]-GTPgammaS
_RADIOLABEL_RE = re.compile(
    r'(\[(?:3H|14C|125I|35S|32P|131I|3H|11C)\][\s\-–]?\w+(?:\w+)*)',
    re.I
)

def extract_radiolabeled_entity(text):
    """Extract radiolabeled reference ligand from assay description.
    Normalizes to DB format with hyphen: [3H]-DAMGO."""
    if not text:
        return None
    m = _RADIOLABEL_RE.search(text)
    if m:
        entity = m.group(1).strip()
        # Ensure hyphen between isotope bracket and ligand name (DB format)
        entity = re.sub(r'(\])\s*(\w)', r'\1-\2', entity)
        return entity
    return None


def classify_assay_format(assay, description=""):
    """
    Determine the correct BAO assay format label from context.
    ChEMBL's bao_format field is unreliable (often returns generic BAO_0000019).
    Priority order:
      -1. In vivo context (description starts with 'In vivo' or ChEMBL says organism-based)
       0. NLP override for cell-free (membrane/radioligand assays done on cell membranes)
       1. Explicit tissue in ChEMBL record
       2. Explicit cell in ChEMBL record
       3. NLP keywords
       4. Organism-based (assay_type == A)
       5. Default: single protein
    """
    if assay is None:
        return None

    _TISSUE_ORGANS = (r'brain|liver|kidney|gut|ileum|lung|spleen|spinal\s+cord|'
                      r'cerebral|cortex\b|cerebellum|striatum|heart|adrenal|bladder|'
                      r'hippocampus|hypothalamus|jejunum|duodenum|stomach|testis|'
                      r'uterus|trachea|diaphragm|aorta|atrium|colon|vas\s+deferens')

    # -1. In vivo detection: description starts with "In vivo" OR ChEMBL bao_label is
    #     "organism-based format". For in vivo tissue assays (injected into live animal,
    #     brain/tissue collected afterward) DB uses "tissue-based format" + "in vivo" setting.
    bao_label = (assay.get("bao_label") or "").lower()
    if description and re.match(r'^\s*In\s+vivo\b', description, re.I):
        if re.search(_TISSUE_ORGANS, description, re.I):
            return "tissue-based format"   # tissue used in in vivo assay
        return "organism-based format"
    if "organism" in bao_label:
        if description and re.search(_TISSUE_ORGANS, description, re.I):
            return "tissue-based format"
        return "organism-based format"

    # 0. NLP override: membrane/cell-free or [35S]GTP indicators trump explicit cell records
    #    These assays use cell membranes even if ChEMBL records a cell line.
    if description:
        early_cellfree_kw = (r'cell\s+membrane[s]?\b|membrane\s+preparation[s]?\b|'
                             r'membrane\s+fraction[s]?\b|broken\s+cell[s]?\b|cell.free|'
                             r'\[35[sS]\]GTP|\[3[hH]\].*?(?:displacement|binding\s+affinity)|'
                             r'binding\s+affinity.*?\[3[hH]\]')
        if re.search(early_cellfree_kw, description, re.I):
            return "cell-free format"

    # 1. Explicit tissue in ChEMBL assay record
    if assay.get("assay_tissue") or assay.get("tissue_chembl_id"):
        return "tissue-based format"

    # 2. Explicit cell in ChEMBL assay record
    if (assay.get("assay_cell_type") or assay.get("cell_chembl_id")):
        return "cell-based format"

    # 3. NLP from description (remaining cases)
    if description:
        desc_lower = description.lower()
        # Tissue keywords
        tissue_kw = (r'ileum|vas deferens|aorta|atrium|colon|hippocampus|'
                     r'cortex\b|cerebellum|striatum|hypothalamus|spinal cord|'
                     r'jejunum|duodenum|stomach|kidney|liver|lung|heart|spleen|'
                     r'testis|uterus|adrenal|trachea|bladder|diaphragm')
        if re.search(tissue_kw, desc_lower):
            return "tissue-based format"
        # Cell line keywords
        cell_kw = (r'\bcell\s+line\b|CHO|HEK|COS|NIH.?3T3|MDCK|PC12|'
                   r'SH-SY5Y|NG108|neuroblastoma|transfect')
        if re.search(cell_kw, desc_lower):
            return "cell-based format"

    # 4. Organism-based (in vivo, no tissue/cell)
    assay_type = (assay.get("assay_type") or "").upper()
    if assay_type == "A":
        return "organism-based format"

    # 5. Default: single protein
    return "single protein format"


# ---------------------------------------------------------------------------
# QikProp estimation via RDKit + ML models
# ---------------------------------------------------------------------------

_MODELS_DIR   = Path(__file__).parent / "qikprop_models"
_QIKPROP_MODELS: dict | None = None   # lazy-loaded cache

_QIKPROP_TARGETS = [
    "qikprop_sasa",
    "qikprop_fisa",
    "qikprop_donor_hb",
    "qikprop_accpt_hb",
    "qikprop_qplog_pw",
    "qikprop_qplog_po/w",
    "qikprop_qplogs",
    "qikprop_qplog_khsa",
    "qikprop_percent_human_oral_absorption",
]

# R² on hold-out test set (20 %), trained on 12,905 molecules
_MODEL_R2 = {
    "qikprop_sasa":                          0.952,
    "qikprop_fisa":                          0.943,
    "qikprop_donor_hb":                      0.966,
    "qikprop_accpt_hb":                      0.973,
    "qikprop_qplog_pw":                      0.973,
    "qikprop_qplog_po/w":                    0.917,
    "qikprop_qplogs":                        0.799,
    "qikprop_qplog_khsa":                    0.939,
    "qikprop_percent_human_oral_absorption":  0.944,
}


def _load_qikprop_models() -> dict:
    """Load trained Random Forest models from disk (once, then cached)."""
    global _QIKPROP_MODELS
    if _QIKPROP_MODELS is not None:
        return _QIKPROP_MODELS
    models = {}
    for tgt in _QIKPROP_TARGETS:
        path = _MODELS_DIR / f"{tgt.replace('/', '_')}.joblib"
        if path.exists():
            models[tgt] = joblib.load(path)
    _QIKPROP_MODELS = models
    return models


def compute_qikprop(smiles: str) -> dict:
    """Predict QikProp-like physicochemical properties from SMILES.

    Primary method: Random Forest models trained on 12,905 molecules from
    Final_updated_Dataset_v2025_11-12.xlsx using 217 RDKit 2D descriptors.
    Test-set R² ranges from 0.80 (QPlogS) to 0.97 (HBA, QPlogPw).

    Fallback (if model files not found): calibrated linear formulas.

    qikprop_dipole is not predicted — requires quantum-mechanical calculation.
    """
    empty = {
        "qikprop_dipole":                        None,
        "qikprop_sasa":                          None,
        "qikprop_fisa":                          None,
        "qikprop_donor_hb":                      None,
        "qikprop_accpt_hb":                      None,
        "qikprop_qplog_pw":                      None,
        "qikprop_qplog_po/w":                    None,
        "qikprop_qplogs":                        None,
        "qikprop_qplog_khsa":                    None,
        "qikprop_percent_human_oral_absorption":  None,
    }
    if not smiles:
        return empty

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty

    # ── Compute all 217 RDKit 2D descriptors ─────────────────────────────
    feats = np.array(
        [[func(mol) for _, func in Descriptors.descList]],
        dtype=float,
    )

    # ── Try ML models (primary path) ─────────────────────────────────────
    models = _load_qikprop_models()
    if models:
        result = {"qikprop_dipole": None}
        for tgt in _QIKPROP_TARGETS:
            if tgt in models:
                try:
                    val = float(models[tgt].predict(feats)[0])
                    result[tgt] = round(val, 3)
                except Exception:
                    result[tgt] = None
            else:
                result[tgt] = None
        return result

    # ── Fallback: calibrated linear formulas (used if model files missing) ──
    logp = Descriptors.MolLogP(mol)
    mw   = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ap   = rdMolDescriptors.CalcNumAromaticRings(mol)
    esol = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*ap
    return {
        "qikprop_dipole":                        None,
        "qikprop_sasa":                          None,
        "qikprop_fisa":                          round(0.9290*tpsa + 69.3221, 2),
        "qikprop_donor_hb":                      round(hbd, 3),
        "qikprop_accpt_hb":                      round(hba, 3),
        "qikprop_qplog_pw":                      round(0.0979*tpsa - 0.2414*logp + 8.204, 3),
        "qikprop_qplog_po/w":                    round(0.8267*logp - 0.0582, 3),
        "qikprop_qplogs":                        round(0.6261*esol + 0.3987, 3),
        "qikprop_qplog_khsa":                    round(0.3970*logp - 1.1860, 3),
        "qikprop_percent_human_oral_absorption":  round(max(0.0, min(100.0, 109.0-0.345*tpsa)), 1),
    }


# ---------------------------------------------------------------------------
# Main extraction functions
# ---------------------------------------------------------------------------

def extract_molecule_fields(mol):
    """Extract molecule-level fields including parent structure (non-stereo)."""
    props    = mol.get("molecule_properties")  or {}
    structs  = mol.get("molecule_structures")  or {}
    hierarchy = mol.get("molecule_hierarchy") or {}

    # --- Parent molecule lookup ---
    parent_id  = hierarchy.get("parent_chembl_id")
    parent_mol = None
    if parent_id and parent_id != mol.get("molecule_chembl_id"):
        parent_mol = fetch_molecule(parent_id)
        time.sleep(RATE_LIMIT_DELAY)

    parent_structs = ((parent_mol or {}).get("molecule_structures") or {})
    parent_props   = ((parent_mol or {}).get("molecule_properties")  or {})

    # Prefer parent structures; fall back to molecule's own structures
    canonical = structs.get("canonical_smiles") or ""
    parent_smiles_stereo = (parent_structs.get("canonical_smiles")
                            or canonical)

    # --- Parent structure: use stereo SMILES/InChI from ChEMBL (DB stores with stereo) ---
    stereo_inchi    = parent_structs.get("standard_inchi")     or structs.get("standard_inchi")
    stereo_inchikey = parent_structs.get("standard_inchi_key") or structs.get("standard_inchi_key")
    # PubChem is only used for formula/weight of the parent (desalted) compound
    parent_pc   = get_pubchem_parent_structure(stereo_inchi, stereo_inchikey)

    qikprop = compute_qikprop(canonical)

    return {
        "chembl_compound_id":                    mol.get("molecule_chembl_id"),
        "rdkit_library_standard_inchi":          structs.get("standard_inchi"),
        "rdkit_library_standard_inchi_key":      structs.get("standard_inchi_key"),
        "rdkit_canonical_smiles":                canonical,
        "rdkit_molecular_foruLa":                props.get("full_molformula"),
        "rdkit_molecular_weight":                props.get("full_mwt"),
        "rdkit_parent_structure_smiles":         parent_smiles_stereo,
        "rdkit_parent_structure_inchi":          stereo_inchi,
        "rdkit_parent_structure_inchi_key":      stereo_inchikey,
        "rdkit_parent_structure_molecular_formula": (
            parent_pc.get("MolecularFormula")
            or parent_props.get("full_molformula")
            or props.get("full_molformula")),
        "rdkit_parent_structure_molecular_weight": (
            parent_pc.get("MolecularWeight")
            or parent_props.get("mw_freebase")
            or props.get("mw_freebase")),
        "chembl_molecule_max_phase":   mol.get("max_phase", 0) if mol.get("max_phase") is not None else 0,
        "chembl_#ro5_violations":      props.get("num_ro5_violations"),
        "chembl_alogp":                props.get("alogp"),
        "chembl_chemical_entity_name": mol.get("pref_name"),
        "chembl_chemical_entity_key":  None,   # manually curated
        **qikprop,
    }


# ---------------------------------------------------------------------------
# NLP / text-based prediction for assay columns (previously always empty)
# ---------------------------------------------------------------------------

_NLP_MODELS_DIR = Path(__file__).parent / "nlp_models"
_NLP_MODELS: dict | None = None   # lazy-loaded cache

# Columns filled by this module (used for Excel blue highlighting)
NLP_PREDICTED_COLS: set[str] = {
    "assay_kit",
    "bao_reference_compound",
    "dose_reference_compound",
    "ncit_route_of_administration_id",
    "mi_database_citation_id",
}

# Known reference compounds (from DB top values) — searched in assay description
# Ordered longest-first to avoid partial matches (e.g. "U50,488H" before "U50,488")
_REFERENCE_COMPOUNDS = [
    "Nociceptin/Orphanin FQ",
    "Salvinorin A",
    "hydromorphone",
    "naltrexone",
    "bremazocine",
    "U50,488H", "U50488H",
    "U69,593",  "U69593",
    "U50,488",  "U50488",  "U50 488",
    "SNC80",    "SNC 80",
    "naloxone",
    "morphine",
    "DPDPE",
    "DAMGO",
    "TAPP",
    "Naltrindole",
]

# Route of administration → NCIT ID (from NCIT ontology)
_ROUTE_TO_NCIT_ID: dict[str, str] = {
    "oral":                    "C38288",
    "intravenous":             "C38276",
    "subcutaneous":            "C38299",
    "intraperitoneal":         "C38261",
    "intramuscular":           "C28161",
    "intracerebral":           "C79839",
    "intrathecal":             "C38253",
    "intracerebroventricular": "C79799",
    "intranasal":              "C38284",
}

# mi_database_citation → MI ontology URI
_MI_CITATION_ID: dict[str, str] = {
    "PubMed":           "http://purl.obolibrary.org/obo/MI_0446",
    "ChEMBL Database":  "http://purl.obolibrary.org/obo/MI_0446",
}


def _load_nlp_models() -> dict:
    """Load trained NLP models from disk (once, then cached)."""
    global _NLP_MODELS
    if _NLP_MODELS is not None:
        return _NLP_MODELS
    models = {}
    model_files = {
        "assay_kit": "assay_kit.joblib",
    }
    for col, fname in model_files.items():
        path = _NLP_MODELS_DIR / fname
        if path.exists():
            models[col] = joblib.load(path)
    _NLP_MODELS = models
    return models


def predict_assay_columns(desc: str, existing: dict) -> dict:
    """Predict assay columns that are not yet populated.

    Combines:
    - ML classifier (assay_kit)
    - Regex pattern matching (bao_reference_compound, dose_reference_compound)
    - Static lookup dicts (ncit_route_of_administration_id, mi_database_citation_id)

    Only fills a column if it is currently None/empty.
    """
    if not desc:
        desc = ""

    result: dict = {}

    # ── assay_kit: ML classifier ──────────────────────────────────────────
    models = _load_nlp_models()
    kit_model = models.get("assay_kit")
    if kit_model is not None and existing.get("assay_kit") is None:
        try:
            predicted_kit = kit_model.predict([desc])[0]
            result["assay_kit"] = predicted_kit
        except Exception:
            pass

    # ── bao_reference_compound: keyword search in assay description ───────
    if existing.get("bao_reference_compound") is None and desc:
        for compound in _REFERENCE_COMPOUNDS:
            # Flexible match: allow hyphens/spaces around digits, case-insensitive
            pattern = re.escape(compound).replace(r"\ ", r"[ -]?").replace(r"\,", r",?")
            if re.search(pattern, desc, re.I):
                result["bao_reference_compound"] = compound
                break

    # ── dose_reference_compound: extract concentration from description ───
    if existing.get("dose_reference_compound") is None and desc:
        # Look for patterns like "500 nM", "0.5 uM", "10 mg/kg" near "reference" context
        dose_pat = re.search(
            r'(\d+(?:\.\d+)?)\s*(nM|[uµ]M|mM|mg/kg|nmol|μmol)',
            desc, re.I
        )
        if dose_pat:
            val  = dose_pat.group(1)
            unit = dose_pat.group(2).replace("u", "μ")
            result["dose_reference_compound"] = f"{val} {unit}"

    # ── ncit_route_of_administration_id: lookup from route name ──────────
    route = existing.get("ncit_route_of_administration")
    if route and existing.get("ncit_route_of_administration_id") is None:
        ncit_id = _ROUTE_TO_NCIT_ID.get(str(route).lower().strip())
        if ncit_id:
            result["ncit_route_of_administration_id"] = ncit_id

    # ── mi_database_citation_id: lookup from citation name ───────────────
    citation = existing.get("mi_database_citation")
    if citation and existing.get("mi_database_citation_id") is None:
        mi_id = _MI_CITATION_ID.get(str(citation).strip())
        if mi_id:
            result["mi_database_citation_id"] = mi_id

    return result


def extract_activity_fields(act):
    """Extract activity-level fields including pharmacological role from description."""
    action_type = act.get("action_type")
    desc        = act.get("assay_description") or ""

    # Prefer action_type if present; otherwise NLP from description
    if action_type:
        role  = action_type.lower().strip()
        chebi = CHEBI_ROLE_MAP.get(role)
    else:
        role, chebi = extract_role_from_text(desc)

    uo = act.get("uo_units")

    return {
        "chembl_assay_id":                    act.get("assay_chembl_id"),
        "assay_description":                  desc,
        "compound_pharmacological_role":      role,
        "chebi_compound_pharmacological_role_id": chebi,
        "endpoint":                           act.get("standard_type"),
        "endpoint_qualifier":                 act.get("standard_relation"),
        "endpoint_value":                     act.get("standard_value"),
        "unit_of_measurement":                act.get("standard_units"),
        "pchembl_value":                      act.get("pchembl_value"),
        "uo_id":                              uo,
        "chembl_target_id":                   act.get("target_chembl_id"),
        "chembl_document_id":                 act.get("document_chembl_id"),
        "document_journal":                   act.get("document_journal"),
        "document_year":                      act.get("document_year"),
        "chembl_source_id":                   act.get("src_id"),
        "_assay_description":                 desc,   # internal, for NLP
    }


def extract_assay_fields(assay, description=""):
    """Extract assay-level fields with improved BAO format and NLP."""
    if assay is None:
        return {}

    desc = description or assay.get("description") or ""

    # --- Determine assay format ---
    fmt_label = classify_assay_format(assay, desc)
    fmt_id    = BAO_FORMAT_LABELS.get(fmt_label)

    # --- In vivo flag: description starts with "In vivo" OR ChEMBL says organism-based ---
    bao_label_raw = (assay.get("bao_label") or "").lower()
    is_in_vivo = (
        bool(re.match(r'^\s*In\s+vivo\b', desc, re.I))
        or "organism" in bao_label_raw
    )

    # --- Experimental setting ---
    if is_in_vivo:
        setting_info = BAO_SETTING_MAP["In vivo"]
    else:
        setting_key  = FORMAT_TO_SETTING.get(fmt_label, "")
        setting_info = BAO_SETTING_MAP.get(setting_key, (None, None))

    # --- Bioassay type ---
    assay_type_desc = (assay.get("assay_type_description") or "").lower()
    type_entry  = BAO_TYPE_MAP.get(assay_type_desc)
    bao_type    = type_entry[0] if type_entry else (assay_type_desc if assay_type_desc else fmt_label)
    bao_type_id = type_entry[1] if type_entry else None

    # --- Tissue ---
    tissue = assay.get("assay_tissue") or extract_tissue_from_text(desc)
    bto_id = get_bto_id(tissue) if tissue else None

    # --- NCIT model system ---
    if is_in_vivo:
        model_info = ("whole organism", "C77665")
    else:
        model_info = NCIT_MODEL_SYSTEM_MAP.get(fmt_label, (None, None))

    # --- NLP-based method classification ---
    method_fields = classify_assay_methods(desc)

    # --- In vivo: override assay method to "in vivo assay method" ---
    if is_in_vivo:
        method_fields["bao_assay_method"]    = "in vivo assay method"
        method_fields["bao_assay_method_id"] = "http://www.bioassayontology.org/bao#BAO_0000406"

    # --- ODO assay ---
    odo_assay = classify_odo_assay(desc)

    # --- Radiolabeled reference compound ---
    radiolabeled = extract_radiolabeled_entity(desc)

    # --- Assay parameters ---
    params_list = assay.get("assay_parameters") or []
    params_str  = ("; ".join(
        f"{p.get('type', '')}: {p.get('value', '')}"
        for p in params_list) if params_list else None)

    # --- Subcellular format ---
    subcellular_format_raw = assay.get("assay_subcellular_fraction")
    subcellular_format = None
    bao_subcellular_id = None

    if subcellular_format_raw:
        # Normalize ChEMBL's raw subcellular_fraction to proper BAO label
        sf_lower = subcellular_format_raw.lower()
        if "membrane" in sf_lower:
            if fmt_label == "tissue-based format":
                subcellular_format = "tissue-membrane format"
                bao_subcellular_id = "http://www.bioassayontology.org/bao#BAO_0020012"
            else:
                subcellular_format = "cell-membrane format"
                bao_subcellular_id = "http://www.bioassayontology.org/bao#BAO_0000249"
        else:
            subcellular_format = subcellular_format_raw  # keep raw if not recognized

    if not subcellular_format:
        # NLP / context-based detection
        if fmt_label == "cell-free format" and re.search(r'\bmembrane[s]?\b', desc, re.I):
            subcellular_format = "cell-membrane format"
            bao_subcellular_id = "http://www.bioassayontology.org/bao#BAO_0000249"
        elif fmt_label == "tissue-based format":
            if re.search(r'\bmembrane[s]?\b', desc, re.I):
                subcellular_format = "tissue-membrane format"
                bao_subcellular_id = "http://www.bioassayontology.org/bao#BAO_0020012"
            elif is_in_vivo and radiolabeled:
                # In vivo radioligand binding in tissue → membrane preparation assumed
                subcellular_format = "tissue-membrane format"
                bao_subcellular_id = "http://www.bioassayontology.org/bao#BAO_0020012"

    # --- In vivo: animal model and route of administration ---
    assay_organism = assay.get("assay_organism")
    animal_model   = ORGANISM_COMMON_NAME.get(assay_organism) if assay_organism else None
    route_of_admin = extract_route_of_administration(desc) if is_in_vivo else None

    result = {
        "bao_experimental_setting":    setting_info[0],
        "bao_experimental_setting_id": setting_info[1],
        "bao_assay_format":            fmt_label,
        "bao_assay_format_id":         fmt_id,
        "bao_bioassay_type":           bao_type,
        "bao_bioassay_type_id":        bao_type_id,
        "subcellular_format":          subcellular_format,
        "bao_subcellular_format_id":   bao_subcellular_id,
        "subcellular_format_unmapped": 0 if subcellular_format else None,
        "cell_line_name":              assay.get("assay_cell_type"),
        "_cell_chembl_id":             assay.get("cell_chembl_id"),
        "tissue_name":                 tissue,
        "bto_tissue_id":               bto_id,
        "_tissue_chembl_id":           assay.get("tissue_chembl_id"),
        "ncbi_tissue_taxonomy":        assay_organism,
        "ncbi_tissue_taxonomy_id":     assay.get("assay_tax_id"),
        "ncit_vertebrate_taxonomy":    assay_organism,
        "ncit_vertebrate_taxonomy_id": assay.get("assay_tax_id"),
        "ncit_model_system":           model_info[0],
        "ncit_model_system_id":        model_info[1],
        "ncit_animal_model":           animal_model,
        "ncit_animal_animal_model_strain": assay.get("assay_strain"),
        "ncit_route_of_administration":    route_of_admin,
        "chembl_assay_parameters":              params_str,
        "odo_assay":                            odo_assay,
        "reference_radiolabeled_molecular_entity": radiolabeled,
    }

    result.update(method_fields)
    return result


def extract_cell_line_fields(cell):
    """Extract cell line fields."""
    if cell is None:
        return {}
    raw_clo = cell.get("clo_id")
    # ChEMBL returns "CLO_XXXXXXX"; DB expects "CLO:XXXXXXX"
    clo_id = raw_clo.replace("CLO_", "CLO:", 1) if raw_clo else None
    return {
        "cell_line_name":          cell.get("cell_name"),
        "cellosaurus_cell_line_id": cell.get("cellosaurus_id"),
        "clo_cell_line_id":        clo_id,
    }


def extract_target_fields(target):
    """
    Extract target-level fields.
    Fixes:
    - InterPro family = IPR000276 (G protein-coupled receptor)
    - InterPro category name/ID are SWAPPED in the DB schema
    - PRO ID via UniProt API
    - DTO GPCR category via PANTHER → DTO mapping
    - target_name normalization
    - GPCR signaling pathway: prefer GO:0007193
    """
    if target is None:
        return {}

    components = target.get("target_components") or []
    uniprot_id       = None
    protein_name     = None
    pro_id           = None
    gpcr_signaling   = None
    gpcr_signaling_id = None
    dto_gpcr_cat     = None
    dto_gpcr_cat_id  = None
    ncit_subfamily   = None
    ncit_subfamily_id = None

    # InterPro fields
    interpro_family_name    = None
    interpro_family_name_id = None
    interpro_category       = None   # NOTE: DB stores the ID string here
    interpro_category_id    = None   # NOTE: DB stores the name string here

    if components:
        comp = components[0]
        uniprot_id   = comp.get("accession")
        protein_name = comp.get("component_description")

        # --- UniProt lookup (PRO ID via OLS4 protein-name search) ---
        if uniprot_id:
            get_uniprot_data(uniprot_id)   # warm cache
            pro_id = _get_pro_id_from_protein_name(protein_name)

        xrefs = comp.get("target_component_xrefs") or []

        # --- InterPro classification ---
        interpro_refs = [x for x in xrefs if x.get("xref_src_db") == "InterPro"]

        # Family = IPR000276 (GPCR Rhodopsin superfamily)
        for xref in interpro_refs:
            if xref.get("xref_id") == "IPR000276":
                interpro_family_name    = "G protein-coupled receptor"
                interpro_family_name_id = "IPR000276"
                break

        # If IPR000276 not present, use first available non-generic entry
        if interpro_family_name_id is None:
            for xref in interpro_refs:
                xid = xref.get("xref_id", "")
                if xid not in ("IPR017452",):  # skip 7TM-generic
                    interpro_family_name_id = xid
                    interpro_family_name    = xref.get("xref_name", "").lower()
                    break

        # Category: receptor superfamily level (e.g. IPR001418 Opioid receptor)
        # DB schema quirk: 'category' column = IPR ID; 'category_id' column = name
        # Strategy for category selection:
        #   1. Prefer entries explicitly listed in PREFERRED_CATEGORY_IPR
        #   2. Fall back to any entry in INTERPRO_TO_GPROTEIN
        #   3. Last resort: first non-generic entry
        PREFERRED_CATEGORY_IPR = ["IPR001418"]  # Opioid receptor – preferred class level
        candidates = [
            x for x in interpro_refs
            if x.get("xref_id") not in ("IPR000276", "IPR017452",
                                         interpro_family_name_id)
        ]
        category_xref = None
        # Pass 1: preferred explicit list
        for xref in candidates:
            if xref.get("xref_id") in PREFERRED_CATEGORY_IPR:
                category_xref = xref
                break
        # Pass 2: any entry with G-protein mapping
        if category_xref is None:
            for xref in candidates:
                if xref.get("xref_id") in INTERPRO_TO_GPROTEIN:
                    category_xref = xref
                    break
        # Pass 3: first remaining candidate
        if category_xref is None and candidates:
            category_xref = candidates[0]

        if category_xref:
            category_ipr_id   = category_xref.get("xref_id", "")
            category_ipr_name = get_interpro_name(category_ipr_id)
            interpro_category    = category_ipr_id    # stored in "name" column per DB
            interpro_category_id = category_ipr_name  # stored in "id" column per DB
            # NCIT G-protein coupling: search all candidates (not just the selected one)
            for xref in candidates:
                gp = INTERPRO_TO_GPROTEIN.get(xref.get("xref_id", ""))
                if gp:
                    ncit_subfamily, ncit_subfamily_id = gp
                    break

        # --- GO process (signaling pathway) ---
        go_refs = [x for x in xrefs if x.get("xref_src_db") == "GoProcess"]
        # Prefer GO:0007193 (general adenylate cyclase-inhibiting GPCR)
        _go_priority = [
            "GO:0007193",   # adenylate cyclase-inhibiting GPCR signaling
            "GO:0007186",   # GPCR signaling pathway
            "GO:0007187",   # GPCR signaling, coupled via Gi
        ]
        # First pass: look for priority GO terms
        for go_id in _go_priority:
            for xref in go_refs:
                if xref.get("xref_id") == go_id:
                    gpcr_signaling    = xref.get("xref_name")
                    gpcr_signaling_id = go_id
                    break
            if gpcr_signaling:
                break
        # Second pass: any adenylate-cyclase-inhibiting GPCR GO term →
        # normalize all sub-terms to GO:0007193 (the canonical parent)
        _AC_INHIBIT = "adenylate cyclase-inhibiting g protein-coupled"
        if not gpcr_signaling:
            for xref in go_refs:
                name = xref.get("xref_name", "")
                if _AC_INHIBIT in name.lower():
                    # Normalize: always use the general parent GO:0007193
                    gpcr_signaling    = ("adenylate cyclase-inhibiting G protein-coupled "
                                         "receptor signaling pathway")
                    gpcr_signaling_id = "GO:0007193"
                    break
        # Third pass: any opioid/GPCR signaling GO term
        if not gpcr_signaling:
            for xref in go_refs:
                name = xref.get("xref_name", "")
                if ("opioid" in name.lower()
                        or "g protein-coupled receptor signaling" in name.lower()):
                    gpcr_signaling    = name
                    gpcr_signaling_id = xref.get("xref_id")
                    break
        # Fourth pass: if G(i) Alpha coupling known from InterPro, infer GO:0007193
        # (All opioid receptors couple via Gi/adenylate cyclase-inhibiting pathway)
        if not gpcr_signaling and ncit_subfamily == "G(i) Alpha" and go_refs:
            gpcr_signaling    = ("adenylate cyclase-inhibiting G protein-coupled "
                                 "receptor signaling pathway")
            gpcr_signaling_id = "GO:0007193"
        # Final override: G(i) Alpha confirmed → always prefer GO:0007193 over generic GO:0007186
        if ncit_subfamily == "G(i) Alpha":
            gpcr_signaling    = ("adenylate cyclase-inhibiting G protein-coupled "
                                 "receptor signaling pathway")
            gpcr_signaling_id = "GO:0007193"

        # --- PANTHER → DTO GPCR category ---
        panther_refs = [x for x in xrefs if x.get("xref_src_db") == "PANTHER"]
        for xref in panther_refs:
            xid = xref.get("xref_id", "")
            # Use top-level family (no ":" suffix like PTHR24229:SF2)
            if ":" not in xid:
                dto_info = PANTHER_TO_DTO.get(xid)
                if dto_info:
                    dto_gpcr_cat, dto_gpcr_cat_id = dto_info
                break

    # --- Target type normalization ---
    target_type_raw = target.get("target_type")
    target_type = None
    if target_type_raw:
        target_type = target_type_raw.replace("_", " ").title()
        if target_type == "Single Protein":
            target_type = "Single protein"

    # --- Target name normalization ---
    target_name = normalize_target_name(target.get("pref_name"))

    return {
        "uniprot_protein_id":              uniprot_id,
        "interpro_protein_family_name":    interpro_family_name,
        "interpro_protein_family_name_id": interpro_family_name_id,
        "interpro_protein_category":       interpro_category,    # DB schema: ID stored here
        "interpro_protein_category_id":    interpro_category_id, # DB schema: name stored here
        "ncit_protein_subfamily_name":     ncit_subfamily,
        "ncit_protein_subfamily_name_id":  ncit_subfamily_id,
        "dto_gpcr_category":               dto_gpcr_cat,
        "dto_gpcr_category_id":            dto_gpcr_cat_id,
        "protein_name":                    protein_name,
        "pro_protein_name_id":             pro_id,
        "target_name":                     target_name,
        "chembl_target_id":                target.get("target_chembl_id"),
        "target_type":                     target_type,
        "ncbi_target_taxonomy":            target.get("organism"),
        "ncbi_target_taxonomy_id":         target.get("tax_id"),
        "embl_ebi_gpcr_signaling_pathway":    gpcr_signaling,
        "embl_ebi_gpcr_signaling_pathway_id": gpcr_signaling_id,
    }


def extract_document_fields(doc):
    """Extract document-level fields."""
    if doc is None:
        return {}
    src_id = doc.get("src_id")
    return {
        "pubmed_id":            doc.get("pubmed_id"),
        "document_doi":         doc.get("doi"),
        "document_journal":     normalize_journal_abbrev(doc.get("journal")),
        "document_year":        doc.get("year"),
        "source_description":   SOURCE_MAP.get(src_id),
        "chembl_source_id":     src_id,
        "patent_id":            doc.get("patent_id"),
        "mi_database_citation": "ChEMBL Database" if src_id == 1 else None,
    }


# ---------------------------------------------------------------------------
# Build all rows for one molecule
# ---------------------------------------------------------------------------

def build_rows_for_molecule(chembl_id, target_filter=None):
    """Build all data rows for a single ChEMBL molecule."""
    print(f"  Fetching molecule {chembl_id}...")
    mol = fetch_molecule(chembl_id)
    if mol is None:
        print(f"    Molecule {chembl_id} not found!")
        return []

    mol_fields = extract_molecule_fields(mol)

    # PubChem CID + IUPAC name – use InChIKey of the (stereo) library compound
    inchikey_for_pc = (mol_fields.get("rdkit_library_standard_inchi_key")
                       or mol_fields.get("rdkit_parent_structure_inchi_key"))
    pubchem_cid, pubchem_iupac = get_pubchem_data(inchikey_for_pc)
    mol_fields["pubchem_cid"]        = pubchem_cid
    mol_fields["pubchem_iupac_name"] = pubchem_iupac

    time.sleep(RATE_LIMIT_DELAY)

    print(f"  Fetching activities for {chembl_id}...")
    activities = fetch_activities(chembl_id)
    if not activities:
        print(f"    No activities found for {chembl_id}")
        return []

    print(f"    Found {len(activities)} activities")

    rows = []
    for act in activities:
        act_fields = extract_activity_fields(act)

        if target_filter and act_fields.get("chembl_target_id") != target_filter:
            continue

        desc = act_fields.pop("_assay_description", "") or ""

        # Assay
        assay_id     = act_fields.get("chembl_assay_id")
        assay_fields = {}
        cell_fields  = {}
        if assay_id:
            assay        = get_assay(assay_id)
            assay_fields = extract_assay_fields(assay, description=desc)
            cell_chembl_id = assay_fields.pop("_cell_chembl_id", None)
            assay_fields.pop("_tissue_chembl_id", None)
            if cell_chembl_id:
                cell        = get_cell_line(cell_chembl_id)
                cell_fields = extract_cell_line_fields(cell)

        # Target
        target_id    = act_fields.get("chembl_target_id")
        target_fields = {}
        if target_id:
            target        = get_target(target_id)
            target_fields = extract_target_fields(target)

        # Document
        doc_id     = act_fields.get("chembl_document_id")
        doc_fields = {}
        if doc_id:
            doc        = get_document(doc_id)
            doc_fields = extract_document_fields(doc)

        # Merge all fields
        row = {}
        row.update(mol_fields)
        row.update(act_fields)
        row.update(assay_fields)
        row.update(cell_fields)
        row.update(target_fields)
        row.update(doc_fields)

        # Remove internal helpers
        for k in list(row.keys()):
            if k.startswith("_"):
                del row[k]

        # NLP / pattern-based prediction for previously-empty columns
        nlp_predicted = predict_assay_columns(desc, row)
        for col, val in nlp_predicted.items():
            if row.get(col) is None and val is not None:
                row[col] = val

        rows.append(row)

    if target_filter and not rows:
        print(f"    No activities found for target {target_filter}")

    return rows


# ---------------------------------------------------------------------------
# Full 131-column ordering (unchanged from original dataset)
# ---------------------------------------------------------------------------

COLUMN_ORDER = [
    "chembl_compound_id",
    "pubchem_cid",
    "rdkit_library_standard_inchi",
    "rdkit_library_standard_inchi_key",
    "rdkit_canonical_smiles",
    "pubchem_iupac_name",
    "rdkit_molecular_foruLa",
    "rdkit_molecular_weight",
    "rdkit_parent_structure_smiles",
    "rdkit_parent_structure_inchi",
    "rdkit_parent_structure_inchi_key",
    "rdkit_parent_structure_molecular_formula",
    "rdkit_parent_structure_molecular_weight",
    "reference_radiolabeled_molecular_entity",
    "chembl_molecule_max_phase",
    "chembl_#ro5_violations",
    "chembl_alogp",
    "bao_reference_compound",
    "dose_reference_compound",
    "qikprop_dipole",
    "qikprop_sasa",
    "qikprop_fisa",
    "qikprop_donor_hb",
    "qikprop_accpt_hb",
    "qikprop_qplog_pw",
    "qikprop_qplog_po/w",
    "qikprop_qplogs",
    "qikprop_qplog_khsa",
    "qikprop_percent_human_oral_absorption",
    "chembl_chemical_entity_name",
    "chembl_chemical_entity_key",
    "chembl_assay_id",
    "bao_experimental_setting",
    "bao_experimental_setting_id",
    "compound_pharmacological_role",
    "chebi_compound_pharmacological_role_id",
    "compound_pharmacological_role_unmapped",
    "chembl_binding_site_description",
    "bao_bioassay_1",
    "bao_bioasssay_1_id",
    "bao_bioassay_2",
    "bao_bioasssay_2_id",
    "bao_bioassay_type",
    "bao_bioassay_type_id",
    "subcellular_format",
    "bao_subcellular_format_id",
    "subcellular_format_unmapped",
    "bao_assay_format",
    "bao_assay_format_id",
    "bao_assay_format_l2",
    "bao_assay _format_l2_id",
    "bao_assay_format_l3",
    "bao_assay_format_l3_id",
    "bao_assay_method",
    "bao_assay_method_id",
    "physical_detection_method",
    "physical_detection_method_id",
    "assay_kit",
    "bao_assay_kit_id",
    "assay_kit_unmapped",
    "assay_description",
    "odo_assay",
    "chembl_assay_property",
    "odo_functional_bias_assay_1",
    "odo_functional_bias_assay_2",
    "chembl_assay_parameters",
    "odo_assay_property",
    "single_concentration_screen",
    "cell_line_name",
    "cellosaurus_cell_line_id",
    "clo_cell_line_id",
    "cell_line_unmapped",
    "bias_assay_2_cell_line",
    "mba_tissue_id",
    "pato_tissue_id",
    "tissue_name",
    "bto_tissue_id",
    "tissue_unmapped",
    "ncbi_tissue_taxonomy",
    "ncbi_tissue_taxonomy_id",
    "ncit_animal_model",
    "ncit_animal_animal_model_strain",
    "ncit_animal_model_strain_id",
    "mgi_animal_strain_id",
    "ncit_vertebrate_taxonomy",
    "ncit_vertebrate_taxonomy_id",
    "ncit_model_system",
    "ncit_model_system_id",
    "uniprot_protein_id",
    "interpro_protein_family_name",
    "interpro_protein_family_name_id",
    "ncit_protein_subfamily_name",
    "ncit_protein_subfamily_name_id",
    "interpro_protein_category_id",
    "interpro_protein_category",
    "dto_gpcr_category",
    "dto_gpcr_category_id",
    "protein_name",
    "pro_protein_name_id",
    "target_name",
    "chembl_target_id",
    "target_type",
    "ncbi_target_taxonomy",
    "ncbi_target_taxonomy_id",
    "endpoint",
    "endpoint_qualifier",
    "endpoint_value",
    "sem_endpoint_qualifier",
    "sem_value",
    "cl_lower_95%",
    "cl_upper_95%",
    "unit_of_measurement",
    "pchembl_value",
    "uo_id",
    "odo_assay_endpoint_description",
    "chembl_dose_administered",
    "ncit_route_of_administration",
    "ncit_route_of_administration_id",
    "embl_ebi_gpcr_signaling_pathway",
    "embl_ebi_gpcr_signaling_pathway_id",
    "mi_database_citation",
    "mi_database_citation_id",
    "pubmed_id",
    "document_doi",
    "chembl_document_id",
    "source_description",
    "chembl_source_id",
    "document_journal",
    "document_year",
    "patent_id",
    "odo_supplementary_reference",
]


# ---------------------------------------------------------------------------
# Excel formatting
# ---------------------------------------------------------------------------

def apply_excel_formatting(df: pd.DataFrame, out_path) -> None:
    """Post-process the saved Excel file:
    - Highlight qikprop_* columns in amber (estimated values).
    - Auto-fit every column width to its longest cell value.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill
    from openpyxl.comments import Comment

    wb = load_workbook(out_path)
    ws = wb.active

    QIKPROP_FILL = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
    qikprop_col_indices = {
        i + 1 for i, c in enumerate(df.columns) if c.startswith("qikprop_")
    }

    # ── Color + comment for qikprop columns ──────────────────────────────
    # Build per-column comment text (includes model R²)
    col_names = list(df.columns)
    for col_idx in qikprop_col_indices:
        col_name = col_names[col_idx - 1]
        r2 = _MODEL_R2.get(col_name)
        if r2 is not None:
            r2_str = f"  R² (test) = {r2:.3f}"
        else:
            r2_str = "  (לא חושב — דורש חישוב קוונטי)"
        comment_text = (
            "⚠ ערך מוערך — לא נלקח ממאמר!\n"
            "מודל: Random Forest (sklearn)\n"
            "אימון: 12,905 מולקולות מה-DB\n"
            f"תכונות: 217 דסקריפטורים של RDKit\n"
            f"{r2_str}"
        )
        header_cell = ws.cell(row=1, column=col_idx)
        header_cell.fill = QIKPROP_FILL
        header_cell.comment = Comment(comment_text, "System")
        for row_idx in range(2, ws.max_row + 1):
            ws.cell(row=row_idx, column=col_idx).fill = QIKPROP_FILL

    # ── Color + comment for NLP-predicted columns ────────────────────────
    NLP_FILL = PatternFill(start_color="CCE5FF", end_color="CCE5FF", fill_type="solid")
    nlp_col_indices = {
        i + 1 for i, c in enumerate(df.columns) if c in NLP_PREDICTED_COLS
    }
    for col_idx in nlp_col_indices:
        col_name = col_names[col_idx - 1]
        comment_text = (
            "⚠ ערך מוערך — לא נלקח ישירות ממאמר!\n"
            "מקור: ML (text classifier / pattern matching / lookup)\n"
            f"עמודה: {col_name}"
        )
        header_cell = ws.cell(row=1, column=col_idx)
        header_cell.fill = NLP_FILL
        header_cell.comment = Comment(comment_text, "System")
        for row_idx in range(2, ws.max_row + 1):
            ws.cell(row=row_idx, column=col_idx).fill = NLP_FILL

    # ── Auto-fit column widths ────────────────────────────────────────────
    for col_cells in ws.columns:
        max_len = 0
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        # Add small padding; cap at 80 to avoid extremely wide columns
        ws.column_dimensions[col_letter].width = min(max_len + 2, 80)

    wb.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch molecule data from ChEMBL + PubChem + UniProt + PubMed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ids",    nargs="+", help="ChEMBL compound IDs")
    parser.add_argument("--smiles", nargs="+", help="SMILES strings")
    parser.add_argument("--file",   "-f",      help="File with IDs or SMILES")
    parser.add_argument("--target", "-t",      help="Filter by target ChEMBL ID")
    parser.add_argument("--output", "-o",      default="chembl_output.xlsx",
                        help="Output Excel file")
    args = parser.parse_args()

    chembl_ids = []
    if args.ids:
        chembl_ids.extend(args.ids)
    if args.smiles:
        for smi in args.smiles:
            cid = resolve_smiles_to_chembl(smi)
            if cid:
                chembl_ids.append(cid)
    if args.file:
        with open(args.file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.upper().startswith("CHEMBL"):
                    chembl_ids.append(line.upper())
                else:
                    cid = resolve_smiles_to_chembl(line)
                    if cid:
                        chembl_ids.append(cid)

    if not chembl_ids:
        print("No molecules specified.")
        parser.print_help()
        sys.exit(1)

    seen = set()
    unique_ids = [c for c in chembl_ids if c not in seen and not seen.add(c)]

    all_rows = []
    for i, chembl_id in enumerate(unique_ids, 1):
        print(f"\n[{i}/{len(unique_ids)}] Processing {chembl_id}")
        rows = build_rows_for_molecule(chembl_id, target_filter=args.target)
        all_rows.extend(rows)
        print(f"    -> {len(rows)} rows added")

    if not all_rows:
        print("No data found!")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = None
    extra = [c for c in df.columns if c not in COLUMN_ORDER]
    df = df[COLUMN_ORDER + extra]

    out = Path(args.output)
    df.to_excel(out, index=False, engine="openpyxl")
    apply_excel_formatting(df, out)

    populated = sum(1 for c in COLUMN_ORDER if df[c].notna().any())
    print(f"\nSaved {len(df)} rows → {out}")
    print(f"Columns: {len(COLUMN_ORDER)} total | {populated} populated")


if __name__ == "__main__":
    main()
