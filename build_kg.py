"""
build_kg.py – ODO Knowledge Graph Builder
Reads Final_updated_Dataset_v2025_11-12.xlsx and produces RDF Turtle files
in the output/ directory, ready to be loaded into GraphDB.
"""

import os
import re
import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD

# ── Namespaces ──────────────────────────────────────────────────────────────
ODO   = Namespace("http://odo-project.org/ontology#")
ODOD  = Namespace("http://odo-project.org/data#")
BAO   = Namespace("http://www.bioassayontology.org/bao#")
CHEBI = Namespace("http://purl.obolibrary.org/obo/CHEBI_")
GO    = Namespace("http://purl.obolibrary.org/obo/GO_")
NCIT  = Namespace("http://purl.obolibrary.org/obo/NCIT_")
PRO   = Namespace("http://purl.obolibrary.org/obo/PR_")
UO    = Namespace("http://purl.obolibrary.org/obo/UO_")
BTO   = Namespace("http://purl.obolibrary.org/obo/BTO_")
CLO   = Namespace("http://purl.obolibrary.org/obo/CLO_")
UP    = Namespace("https://www.uniprot.org/uniprot/")
INTERPRO = Namespace("https://www.ebi.ac.uk/interpro/entry/")
CHEMBL_C = Namespace("https://www.ebi.ac.uk/chembl/compound_report_card/")
PUBCHEM  = Namespace("https://pubchem.ncbi.nlm.nih.gov/compound/")
PUBMED   = Namespace("https://pubmed.ncbi.nlm.nih.gov/")
DOI_NS   = Namespace("https://doi.org/")
CELLOS   = Namespace("https://www.cellosaurus.org/")
DTO      = Namespace("http://www.drugtargetontology.org/dto/")

EXCEL_PATH = os.path.join(os.path.dirname(__file__),
                          "Final_updated_Dataset_v2025_11-12.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def safe(val):
    """Return the value if it is not NaN/None/empty, else None."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def slugify(text):
    """Turn a string into a URI-safe slug (keep alphanumeric, replace rest with _)."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(text).strip())


def extract_bao_id(uri_str):
    """Extract 'BAO_XXXXXXX' from a BAO URI; fall back to slugify."""
    if not uri_str:
        return None
    m = re.search(r'BAO[_#](\d+)', str(uri_str))
    return f"BAO_{m.group(1)}" if m else slugify(uri_str)


_VALID_URI = re.compile(r'^https?://[^\s<>"{}|\\^`\[\]]+$')

def add_uri_sameAs(g, subject, uri_str):
    """Add owl:sameAs link only if uri_str looks like a valid HTTP URI."""
    if uri_str and _VALID_URI.match(uri_str):
        g.add((subject, OWL.sameAs, URIRef(uri_str)))


def bind_prefixes(g):
    g.bind("odo",  ODO)
    g.bind("odod", ODOD)
    g.bind("bao",  BAO)
    g.bind("chebi", CHEBI)
    g.bind("go",   GO)
    g.bind("uo",   UO)
    g.bind("bto",  BTO)
    g.bind("clo",  CLO)
    g.bind("up",   UP)
    g.bind("owl",  OWL)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)


# ────────────────────────────────────────────────────────────────────────────
# ENTITY BUILDERS (each called once per unique entity)
# ────────────────────────────────────────────────────────────────────────────

def build_compound(g, row, seen):
    cid = safe(row["chembl_compound_id"])
    if not cid:
        return None
    if cid in seen["compound"]:
        return ODOD[f"compound_{slugify(cid)}"]
    seen["compound"].add(cid)

    uri = ODOD[f"compound_{slugify(cid)}"]
    g.add((uri, RDF.type, ODO.Compound))
    g.add((uri, ODO.chemblId, Literal(cid)))

    for col, prop, dtype in [
        ("pubchem_cid",                     ODO.pubchemCid,           XSD.string),
        ("rdkit_library_standard_inchi",    ODO.inchi,                XSD.string),
        ("rdkit_library_standard_inchi_key",ODO.inchiKey,             XSD.string),
        ("rdkit_canonical_smiles",          ODO.smiles,               XSD.string),
        ("pubchem_iupac_name",              ODO.iupacName,            XSD.string),
        ("rdkit_molecular_foruLa",          ODO.molecularFormula,     XSD.string),
        ("rdkit_molecular_weight",          ODO.molecularWeight,      XSD.float),
        ("chembl_molecule_max_phase",       ODO.maxPhase,             XSD.integer),
        ("chembl_#ro5_violations",          ODO.ro5Violations,        XSD.integer),
        ("chembl_alogp",                    ODO.alogP,                XSD.float),
        ("chembl_chemical_entity_name",     ODO.chemicalEntityName,   XSD.string),
        ("qikprop_dipole",                  ODO.qpDipole,             XSD.float),
        ("qikprop_sasa",                    ODO.qpSASA,               XSD.float),
        ("qikprop_fisa",                    ODO.qpFISA,               XSD.float),
        ("qikprop_donor_hb",                ODO.qpDonorHB,            XSD.float),
        ("qikprop_accpt_hb",                ODO.qpAcceptorHB,         XSD.float),
        ("qikprop_qplog_pw",                ODO.qpLogPw,              XSD.float),
        ("qikprop_qplog_po/w",              ODO.qpLogPow,             XSD.float),
        ("qikprop_qplogs",                  ODO.qpLogS,               XSD.float),
        ("qikprop_qplog_khsa",              ODO.qpLogKhsa,            XSD.float),
        ("qikprop_percent_human_oral_absorption", ODO.humanOralAbsorption, XSD.float),
    ]:
        v = safe(row.get(col))
        if v:
            try:
                coerced = int(float(v)) if dtype == XSD.integer else float(v) if dtype == XSD.float else v
                g.add((uri, prop, Literal(coerced, datatype=dtype)))
            except Exception:
                g.add((uri, prop, Literal(str(v))))

    # radiolabeled flag
    rl = safe(row.get("reference_radiolabeled_molecular_entity"))
    if rl:
        g.add((uri, ODO.isRadiolabeled, Literal(rl.lower() in ("true","1","yes","t"), datatype=XSD.boolean)))

    # Linked Data – external URIs (only for clean numeric/alphanumeric IDs)
    add_uri_sameAs(g, uri, f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}")
    pcid = safe(row.get("pubchem_cid"))
    if pcid and re.match(r'^\d+$', str(pcid).strip()):
        add_uri_sameAs(g, uri, f"https://pubchem.ncbi.nlm.nih.gov/compound/{pcid}")

    # Parent compound (sub-node)
    parent_inchi = safe(row.get("rdkit_parent_structure_inchi"))
    parent_key   = safe(row.get("rdkit_parent_structure_inchi_key"))
    if parent_key and parent_key != safe(row.get("rdkit_library_standard_inchi_key")):
        parent_uri = ODOD[f"parent_{slugify(parent_key)}"]
        if parent_key not in seen["parent"]:
            seen["parent"].add(parent_key)
            g.add((parent_uri, RDF.type, ODO.ParentCompound))
            if parent_inchi:
                g.add((parent_uri, ODO.inchi, Literal(parent_inchi)))
            g.add((parent_uri, ODO.inchiKey, Literal(parent_key)))
            for col2, prop2 in [
                ("rdkit_parent_structure_smiles",              ODO.smiles),
                ("rdkit_parent_structure_molecular_formula",   ODO.molecularFormula),
                ("rdkit_parent_structure_molecular_weight",    ODO.molecularWeight),
            ]:
                v2 = safe(row.get(col2))
                if v2:
                    g.add((parent_uri, prop2, Literal(v2)))
        g.add((uri, ODO.hasParentCompound, parent_uri))

    return uri


def build_assay(g, row, seen):
    aid = safe(row.get("chembl_assay_id"))
    if not aid or aid in seen["assay"]:
        return ODOD[f"assay_{slugify(aid)}"] if aid else None
    seen["assay"].add(aid)

    uri = ODOD[f"assay_{slugify(aid)}"]
    g.add((uri, RDF.type, ODO.Assay))
    g.add((uri, ODO.chemblAssayId, Literal(aid)))

    for col, prop in [
        ("bao_experimental_setting",       ODO.experimentalSetting),
        ("odo_assay",                      ODO.assayName),
        ("assay_description",              ODO.assayDescription),
        ("chembl_binding_site_description",ODO.bindingSiteDescription),
        ("bao_assay_method",               ODO.assayMethod),
        ("physical_detection_method",      ODO.detectionMethod),
        ("assay_kit",                      ODO.assayKit),
        ("chembl_assay_property",          ODO.assayProperty),
        ("odo_functional_bias_assay_1",    ODO.functionalBiasAssay1),
        ("odo_functional_bias_assay_2",    ODO.functionalBiasAssay2),
    ]:
        v = safe(row.get(col))
        if v:
            g.add((uri, prop, Literal(v)))

    # BAO experimental setting URI
    bao_exp_id = safe(row.get("bao_experimental_setting_id"))
    if bao_exp_id and _VALID_URI.match(bao_exp_id):
        g.add((uri, ODO.hasExperimentalSettingConcept, URIRef(bao_exp_id)))

    return uri


def build_assay_format(g, row, seen):
    fmt = safe(row.get("bao_assay_format"))
    fmt_id = safe(row.get("bao_assay_format_id"))
    key = extract_bao_id(fmt_id) or (slugify(fmt) if fmt else None)
    if not key or key in seen["assay_format"]:
        return ODOD[f"assayformat_{key}"] if key else None
    seen["assay_format"].add(key)

    uri = ODOD[f"assayformat_{key}"]
    g.add((uri, RDF.type, ODO.AssayFormat))
    if fmt:
        g.add((uri, RDFS.label, Literal(fmt)))
    if fmt_id:
        g.add((uri, OWL.sameAs, URIRef(fmt_id)))

    # Hierarchical format levels
    for col, prop in [
        ("bao_assay_format_l2",     ODO.assayFormatL2),
        ("bao_assay_format_l3",     ODO.assayFormatL3),
        ("subcellular_format",      ODO.subcellularFormat),
    ]:
        v = safe(row.get(col))
        if v:
            g.add((uri, prop, Literal(v)))

    return uri


def build_bioassay_type(g, row, seen):
    bt = safe(row.get("bao_bioassay_type"))
    bt_id = safe(row.get("bao_bioassay_type_id"))
    key = extract_bao_id(bt_id) or (slugify(bt) if bt else None)
    if not key or key in seen["bioassay_type"]:
        return ODOD[f"bioassaytype_{key}"] if key else None
    seen["bioassay_type"].add(key)

    uri = ODOD[f"bioassaytype_{key}"]
    g.add((uri, RDF.type, ODO.BioassayType))
    if bt:
        g.add((uri, RDFS.label, Literal(bt)))
    if bt_id:
        g.add((uri, OWL.sameAs, URIRef(bt_id)))
    scs = safe(row.get("single_concentration_screen"))
    if scs:
        g.add((uri, ODO.singleConcentrationScreen, Literal(scs)))

    return uri


def build_target(g, row, seen):
    tname = safe(row.get("target_name"))
    tcid  = safe(row.get("chembl_target_id"))
    key = tcid or slugify(tname) if tname else None
    if not key:
        return None
    if key in seen["target"]:
        return ODOD[f"target_{slugify(key)}"]
    seen["target"].add(key)

    uri = ODOD[f"target_{slugify(key)}"]
    g.add((uri, RDF.type, ODO.Target))
    if tname:
        g.add((uri, ODO.targetName, Literal(tname)))
        g.add((uri, RDFS.label, Literal(tname)))
    if tcid:
        g.add((uri, ODO.chemblTargetId, Literal(tcid)))

    for col, prop in [
        ("target_type",           ODO.targetType),
        ("ncbi_target_taxonomy",  ODO.targetSpecies),
    ]:
        v = safe(row.get(col))
        if v:
            g.add((uri, prop, Literal(v)))

    ncbi_id = safe(row.get("ncbi_target_taxonomy_id"))
    if ncbi_id:
        g.add((uri, ODO.ncbiTaxonomyId, Literal(ncbi_id)))

    if tcid:
        add_uri_sameAs(g, uri, f"https://www.ebi.ac.uk/chembl/target_report_card/{tcid}")

    return uri


def build_protein(g, row, seen):
    upid = safe(row.get("uniprot_protein_id"))
    pname = safe(row.get("protein_name"))
    key = upid or (slugify(pname) if pname else None)
    if not key:
        return None
    if key in seen["protein"]:
        return ODOD[f"protein_{slugify(key)}"]
    seen["protein"].add(key)

    uri = ODOD[f"protein_{slugify(key)}"]
    g.add((uri, RDF.type, ODO.Protein))
    if pname:
        g.add((uri, ODO.proteinName, Literal(pname)))
        g.add((uri, RDFS.label, Literal(pname)))
    if upid:
        g.add((uri, ODO.uniprotId, Literal(upid)))

    for col, prop in [
        ("interpro_protein_family_name",  ODO.proteinFamilyName),
        ("ncit_protein_subfamily_name",   ODO.proteinSubfamilyName),
        ("interpro_protein_category",     ODO.proteinCategory),
        ("dto_gpcr_category",             ODO.gpcrCategory),
    ]:
        v = safe(row.get(col))
        if v:
            g.add((uri, prop, Literal(v)))

    # PRO link
    pro_id = safe(row.get("pro_protein_name_id"))
    if pro_id:
        pro_clean = pro_id.replace("PR:", "")
        g.add((uri, OWL.sameAs, URIRef(f"http://purl.obolibrary.org/obo/PR_{pro_clean}")))

    # UniProt external link
    if upid and "/" not in upid:
        add_uri_sameAs(g, uri, f"https://www.uniprot.org/uniprot/{upid}")

    # InterPro link
    interpro_id = safe(row.get("interpro_protein_family_name_id"))
    if interpro_id:
        add_uri_sameAs(g, uri, f"https://www.ebi.ac.uk/interpro/entry/{interpro_id}")

    # DTO link
    dto_id = safe(row.get("dto_gpcr_category_id"))
    if dto_id:
        g.add((uri, OWL.sameAs, URIRef(dto_id)))

    return uri


def build_cell_line(g, row, seen):
    cln  = safe(row.get("cell_line_name"))
    clid = safe(row.get("cellosaurus_cell_line_id"))
    cloid = safe(row.get("clo_cell_line_id"))
    key = clid or cloid or (slugify(cln) if cln else None)
    if not key or key in seen["cell_line"]:
        return ODOD[f"cellline_{slugify(key)}"] if key else None
    seen["cell_line"].add(key)

    uri = ODOD[f"cellline_{slugify(key)}"]
    g.add((uri, RDF.type, ODO.CellLine))
    if cln:
        g.add((uri, ODO.cellLineName, Literal(cln)))
        g.add((uri, RDFS.label, Literal(cln)))
    if clid:
        g.add((uri, ODO.cellosaurusId, Literal(clid)))
        add_uri_sameAs(g, uri, f"https://www.cellosaurus.org/{clid}")
    if cloid:
        g.add((uri, ODO.cloId, Literal(cloid)))
        clo_clean = cloid.replace("CLO:", "")
        g.add((uri, OWL.sameAs, URIRef(f"http://purl.obolibrary.org/obo/CLO_{clo_clean}")))

    return uri


def build_tissue(g, row, seen):
    tname = safe(row.get("tissue_name"))
    btoid = safe(row.get("bto_tissue_id"))
    key = btoid or (slugify(tname) if tname else None)
    if not key or key in seen["tissue"]:
        return ODOD[f"tissue_{slugify(key)}"] if key else None
    seen["tissue"].add(key)

    uri = ODOD[f"tissue_{slugify(key)}"]
    g.add((uri, RDF.type, ODO.Tissue))
    if tname:
        g.add((uri, ODO.tissueName, Literal(tname)))
        g.add((uri, RDFS.label, Literal(tname)))
    if btoid:
        g.add((uri, ODO.btoId, Literal(btoid)))
        # Use only the first BTO ID if multiple are listed (e.g. "BTO:0000620; 0001427")
        bto_first = btoid.split(";")[0].strip()
        bto_clean = bto_first.replace("BTO:", "").replace("BTO_", "").strip()
        if re.match(r'^\d+$', bto_clean):
            g.add((uri, OWL.sameAs, URIRef(f"http://purl.obolibrary.org/obo/BTO_{bto_clean}")))

    ncbi = safe(row.get("ncbi_tissue_taxonomy"))
    ncbi_id = safe(row.get("ncbi_tissue_taxonomy_id"))
    if ncbi:
        g.add((uri, ODO.ncbiTissueTaxonomy, Literal(ncbi)))
    if ncbi_id:
        g.add((uri, ODO.ncbiTaxonomyId, Literal(ncbi_id)))

    return uri


def build_organism(g, row, seen):
    taxon = safe(row.get("ncbi_target_taxonomy"))
    # Normalize species name to standard binomial casing (e.g. "Homo Sapiens" → "Homo sapiens")
    if taxon:
        parts = taxon.strip().split()
        if len(parts) >= 2:
            taxon = parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])
        else:
            taxon = taxon.strip()
    taxon_id = safe(row.get("ncbi_target_taxonomy_id"))
    key = taxon_id or (slugify(taxon) if taxon else None)
    if not key or key in seen["organism"]:
        return ODOD[f"organism_{slugify(key)}"] if key else None
    seen["organism"].add(key)

    uri = ODOD[f"organism_{slugify(key)}"]
    g.add((uri, RDF.type, ODO.Organism))
    if taxon:
        g.add((uri, ODO.vertebrateTaxonomy, Literal(taxon)))
        g.add((uri, RDFS.label, Literal(taxon)))
    if taxon_id:
        g.add((uri, ODO.ncbiTaxonomyId, Literal(str(taxon_id))))

    for col, prop in [
        ("ncit_animal_model",               ODO.animalModel),
        ("ncit_animal_animal_model_strain", ODO.animalModelStrain),
    ]:
        v = safe(row.get(col))
        if v:
            g.add((uri, prop, Literal(v)))

    return uri


def build_model_system(g, row, cell_uri, tissue_uri, organism_uri, seen):
    ms_val = safe(row.get("ncit_model_system"))
    ms_id  = safe(row.get("ncit_model_system_id"))
    base = ms_id or (slugify(ms_val) if ms_val else None)
    if not base:
        return None

    # Unique key per combination of model-system type + specific biological material
    cell_key  = safe(row.get("cellosaurus_cell_line_id")) or safe(row.get("clo_cell_line_id")) or safe(row.get("cell_line_name")) or ""
    tissue_key = safe(row.get("bto_tissue_id")) or safe(row.get("tissue_name")) or ""
    org_key   = safe(row.get("ncbi_target_taxonomy_id")) or safe(row.get("ncbi_target_taxonomy")) or ""
    key = f"{base}_{slugify(cell_key)}_{slugify(tissue_key)}_{slugify(org_key)}"

    uri = ODOD[f"modelsystem_{slugify(key)}"]
    if key not in seen["model_system"]:
        seen["model_system"].add(key)
        g.add((uri, RDF.type, ODO.ModelSystem))
        if ms_val:
            g.add((uri, RDFS.label, Literal(ms_val)))
        if ms_id:
            g.add((uri, OWL.sameAs, URIRef(f"http://purl.obolibrary.org/obo/NCIT_{ms_id}")))
        if cell_uri:
            g.add((uri, ODO.usesCellLine, cell_uri))
        if tissue_uri:
            g.add((uri, ODO.usesTissue, tissue_uri))
        if organism_uri:
            g.add((uri, ODO.usesOrganism, organism_uri))

    return uri


def build_signaling_pathway(g, row, seen):
    sp_name = safe(row.get("embl_ebi_gpcr_signaling_pathway"))
    sp_id   = safe(row.get("embl_ebi_gpcr_signaling_pathway_id"))
    # Take only the first pathway if multiple are comma-separated
    if sp_name:
        sp_name = sp_name.split(",")[0].strip()
    key = sp_id or (slugify(sp_name) if sp_name else None)
    if not key or key in seen["signaling"]:
        return ODOD[f"signaling_{slugify(key)}"] if key else None
    seen["signaling"].add(key)

    uri = ODOD[f"signaling_{slugify(key)}"]
    g.add((uri, RDF.type, ODO.SignalingPathway))
    if sp_name:
        g.add((uri, ODO.pathwayName, Literal(sp_name)))
        g.add((uri, RDFS.label, Literal(sp_name)))
    if sp_id:
        go_clean = sp_id.replace("GO:", "")
        g.add((uri, ODO.goId, Literal(sp_id)))
        g.add((uri, OWL.sameAs, URIRef(f"http://purl.obolibrary.org/obo/GO_{go_clean}")))

    return uri


def build_document(g, row, seen):
    pmid  = safe(row.get("pubmed_id"))
    # Excel stores numeric IDs as floats (e.g. 24107104.0) – normalize to integer string
    if pmid:
        try:
            pmid = str(int(float(pmid)))
        except (ValueError, OverflowError):
            pass
    doi   = safe(row.get("document_doi"))
    chdoc = safe(row.get("chembl_document_id"))
    key = pmid or doi or chdoc
    if not key or key in seen["document"]:
        return ODOD[f"doc_{slugify(key)}"] if key else None
    seen["document"].add(key)

    uri = ODOD[f"doc_{slugify(key)}"]
    g.add((uri, RDF.type, ODO.Document))

    # pubmedId uses the already-normalized integer string, not the raw float from row
    if pmid:
        g.add((uri, ODO.pubmedId, Literal(pmid, datatype=XSD.string)))

    for col, prop, dtype in [
        ("document_doi",         ODO.documentDoi,      XSD.string),
        ("chembl_document_id",   ODO.chemblDocumentId, XSD.string),
        ("source_description",   ODO.sourceDescription,XSD.string),
        ("document_journal",     ODO.documentJournal,  XSD.string),
        ("document_year",        ODO.documentYear,     XSD.integer),
        ("patent_id",            ODO.patentId,         XSD.string),
        ("mi_database_citation", ODO.databaseCitation, XSD.string),
    ]:
        v = safe(row.get(col))
        if v:
            try:
                coerced = int(float(v)) if dtype == XSD.integer else v
                g.add((uri, prop, Literal(coerced, datatype=dtype)))
            except Exception:
                g.add((uri, prop, Literal(str(v))))

    # Label for the document
    label = doi or pmid or chdoc
    if label:
        g.add((uri, RDFS.label, Literal(label)))

    # External links
    if pmid:
        add_uri_sameAs(g, uri, f"https://pubmed.ncbi.nlm.nih.gov/{pmid}")
    if doi:
        add_uri_sameAs(g, uri, f"https://doi.org/{doi}")

    return uri


def build_activity(g, row_idx, row, compound_uri, assay_uri, target_uri,
                   doc_uri, signaling_uri):
    uri = ODOD[f"activity_{row_idx}"]
    g.add((uri, RDF.type, ODO.Activity))

    if compound_uri:
        g.add((uri, ODO.hasCompound, compound_uri))
    if assay_uri:
        g.add((uri, ODO.hasAssay, assay_uri))
    if target_uri:
        g.add((uri, ODO.hasTarget, target_uri))
    if doc_uri:
        g.add((uri, ODO.publishedIn, doc_uri))
    if signaling_uri:
        g.add((uri, ODO.hasSignalingPathway, signaling_uri))

    for col, prop, dtype in [
        ("endpoint",                  ODO.endpointType,           XSD.string),
        ("endpoint_qualifier",        ODO.endpointQualifier,      XSD.string),
        ("endpoint_value",            ODO.endpointValue,          XSD.float),
        ("sem_endpoint_qualifier",    ODO.semQualifier,           XSD.string),
        ("sem_value",                 ODO.semValue,               XSD.float),
        ("cl_lower_95%",              ODO.clLower95,              XSD.string),
        ("cl_upper_95%",              ODO.clUpper95,              XSD.string),
        ("unit_of_measurement",       ODO.unitLabel,              XSD.string),
        ("pchembl_value",             ODO.pchemblValue,           XSD.float),
        ("compound_pharmacological_role", ODO.pharmacologicalRoleLabel, XSD.string),
        ("odo_assay_endpoint_description", ODO.endpointDescription, XSD.string),
        ("bao_reference_compound",    ODO.isBaoReferenceCompound, XSD.string),
        ("dose_reference_compound",   ODO.doseReferenceCompound,  XSD.string),
    ]:
        v = safe(row.get(col))
        if v:
            try:
                coerced = float(v) if dtype == XSD.float else v
                g.add((uri, prop, Literal(coerced, datatype=dtype)))
            except Exception:
                g.add((uri, prop, Literal(str(v))))

    # Pharmacological role → ChEBI URI
    chebi_id = safe(row.get("chebi_compound_pharmacological_role_id"))
    if chebi_id:
        chebi_clean = chebi_id.replace("http://purl.obolibrary.org/obo/CHEBI_", "").replace("CHEBI:", "")
        g.add((uri, ODO.hasPharmacologicalRole,
               URIRef(f"http://purl.obolibrary.org/obo/CHEBI_{chebi_clean}")))

    # Unit → UO URI
    uo_id = safe(row.get("uo_id"))
    if uo_id:
        uo_clean = uo_id.replace("UO:", "").replace("UO_", "")
        g.add((uri, ODO.hasUnit, URIRef(f"http://purl.obolibrary.org/obo/UO_{uo_clean}")))

    # In-vivo parameters
    rov = safe(row.get("ncit_route_of_administration"))
    dose = safe(row.get("chembl_dose_administered"))
    if rov or dose:
        invivo_uri = ODOD[f"invivo_{row_idx}"]
        g.add((invivo_uri, RDF.type, ODO.InVivoParameters))
        g.add((uri, ODO.hasInVivoParameters, invivo_uri))
        if rov:
            g.add((invivo_uri, ODO.routeOfAdministration, Literal(rov)))
            rov_id = safe(row.get("ncit_route_of_administration_id"))
            if rov_id:
                ncit_clean = rov_id.replace("NCIT:", "")
                g.add((invivo_uri, OWL.sameAs,
                       URIRef(f"http://purl.obolibrary.org/obo/NCIT_{ncit_clean}")))
        if dose:
            g.add((invivo_uri, ODO.doseAdministered, Literal(dose)))

    return uri


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading Excel file…")
    df = pd.read_excel(EXCEL_PATH, sheet_name="2025_ODO_Database")
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    # Separate graphs per entity type for easier partial imports
    graphs = {
        "compounds":       Graph(),
        "assays":          Graph(),
        "activities":      Graph(),
        "model_systems":   Graph(),
        "proteins_targets":Graph(),
        "documents":       Graph(),
        "signaling":       Graph(),
    }
    for g in graphs.values():
        bind_prefixes(g)

    seen = {
        "compound": set(), "parent": set(), "assay": set(),
        "assay_format": set(), "bioassay_type": set(),
        "target": set(), "protein": set(),
        "cell_line": set(), "tissue": set(), "organism": set(),
        "model_system": set(), "signaling": set(), "document": set(),
    }

    gc = graphs["compounds"]
    ga = graphs["assays"]
    gact = graphs["activities"]
    gms = graphs["model_systems"]
    gpt = graphs["proteins_targets"]
    gd = graphs["documents"]
    gsp = graphs["signaling"]

    total = len(df)
    for i, (row_idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 5000 == 0:
            print(f"  Processing row {i+1:,}/{total:,}…")

        # Track whether target/assay are newly created before building them
        _tcid  = safe(row.get("chembl_target_id"))
        _tname = safe(row.get("target_name"))
        _tkey  = _tcid or (slugify(_tname) if _tname else None)
        target_is_new = bool(_tkey and _tkey not in seen["target"])

        # Build entity nodes
        compound_uri  = build_compound(gc, row, seen)
        assay_uri     = build_assay(ga, row, seen)
        fmt_uri       = build_assay_format(ga, row, seen)
        bt_uri        = build_bioassay_type(ga, row, seen)
        target_uri    = build_target(gpt, row, seen)
        protein_uri   = build_protein(gpt, row, seen)
        cell_uri      = build_cell_line(gms, row, seen)
        tissue_uri    = build_tissue(gms, row, seen)
        organism_uri  = build_organism(gms, row, seen)
        ms_uri        = build_model_system(gms, row, cell_uri, tissue_uri, organism_uri, seen)
        signal_uri    = build_signaling_pathway(gsp, row, seen)
        doc_uri       = build_document(gd, row, seen)

        # Wire assay → format / bioassay_type / model_system / protein / target
        if assay_uri:
            if fmt_uri:
                ga.add((assay_uri, ODO.hasAssayFormat, fmt_uri))
            if bt_uri:
                ga.add((assay_uri, ODO.hasBioassayType, bt_uri))
            if ms_uri:
                ga.add((assay_uri, ODO.hasModelSystem, ms_uri))
            if target_uri:
                ga.add((assay_uri, ODO.targetsReceptor, target_uri))
            if protein_uri:
                ga.add((assay_uri, ODO.hasProtein, protein_uri))
            if doc_uri:
                ga.add((assay_uri, ODO.publishedIn, doc_uri))

        # Wire target → protein only when the target node is first created,
        # preventing spurious multi-protein links on single-protein targets
        if target_uri and protein_uri and target_is_new:
            gpt.add((target_uri, ODO.encodedBy, protein_uri))

        # Wire compound → assay
        if compound_uri and assay_uri:
            gc.add((compound_uri, ODO.testedIn, assay_uri))

        # Build activity node (one per row)
        build_activity(gact, row_idx, row, compound_uri, assay_uri, target_uri,
                       doc_uri, signal_uri)

    # Serialize
    print("\nSerializing TTL files…")
    for name, g in graphs.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.ttl")
        g.serialize(destination=path, format="turtle")
        triple_count = len(g)
        print(f"  {name}.ttl  →  {triple_count:,} triples")

    total_triples = sum(len(g) for g in graphs.values())
    print(f"\nDone. Total triples: {total_triples:,}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
