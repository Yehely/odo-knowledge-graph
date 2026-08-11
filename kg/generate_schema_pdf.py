#!/usr/bin/env python3
"""Generate a static PDF schema visualization for the ODO Knowledge Graph."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as pe
import numpy as np

# ── Palette ────────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
HEADER_BG = "#16213e"
GROUPS = {
    "core": {"fill": "#c0392b", "stroke": "#e74c3c", "label": "Core (Measurement)"},
    "bio":  {"fill": "#1a5276", "stroke": "#2980b9", "label": "Biological Context"},
    "meta": {"fill": "#1e8449", "stroke": "#27ae60", "label": "Metadata / Classification"},
}
TYPE_COLORS = {
    "string":  {"bg": "#0d2b1e", "fg": "#a5d6a7"},
    "float":   {"bg": "#0d1b3e", "fg": "#90caf9"},
    "integer": {"bg": "#1a0d2e", "fg": "#ce93d8"},
    "boolean": {"bg": "#2e0d0d", "fg": "#ffcdd2"},
    "uri":     {"bg": "#1c2124", "fg": "#90a4ae"},
}

# ── Descriptions (English) ─────────────────────────────────────────────────
DESC = {
    "chemblId":               "ChEMBL compound identifier",
    "pubchemCid":             "PubChem compound identifier",
    "inchi":                  "InChI string representing the chemical structure",
    "inchiKey":               "InChI Key — short hash as unique identifier",
    "smiles":                 "Molecular structure in SMILES format",
    "iupacName":              "Systematic IUPAC name",
    "chemicalEntityName":     "Chemical entity name per ChEMBL",
    "molecularFormula":       "Molecular formula (e.g. C17H19NO3)",
    "molecularWeight":        "Molecular weight (Daltons)",
    "maxPhase":               "Maximum clinical phase reached (0–4)",
    "ro5Violations":          "Number of Lipinski Rule-of-5 violations",
    "alogP":                  "Octanol/water partition coefficient (lipophilicity)",
    "isRadiolabeled":         "Whether the compound is radiolabeled",
    "qpDipole":               "Dipole moment (QikProp) — molecular polarity",
    "qpSASA":                 "Solvent accessible surface area (QikProp)",
    "qpFISA":                 "Hydrophilic surface area (QikProp)",
    "qpDonorHB":              "Hydrogen bond donors (QikProp)",
    "qpAcceptorHB":           "Hydrogen bond acceptors (QikProp)",
    "qpLogPw":                "log water/gas partition (QikProp)",
    "qpLogPow":               "log octanol/water partition (QikProp)",
    "qpLogS":                 "log water solubility (QikProp)",
    "qpLogKhsa":              "log human serum albumin binding (QikProp)",
    "humanOralAbsorption":    "% estimated human oral absorption (QikProp)",
    "endpointType":           "Endpoint type: Ki, IC50, EC50, Kd, Imax…",
    "endpointQualifier":      "Value qualifier: =, <, >, ≤, ≥",
    "endpointValue":          "Biological activity measurement value (numeric)",
    "endpointDescription":    "Textual description of the endpoint",
    "semQualifier":           "Standard error qualifier (SEM)",
    "semValue":               "Standard error of the mean (SEM)",
    "clLower95":              "Lower bound of 95% confidence interval",
    "clUpper95":              "Upper bound of 95% confidence interval",
    "pchemblValue":           "pChEMBL = -log10(molar value) — unified activity measure",
    "unitLabel":              "Unit of measurement: nM, uM, %, mg/kg…",
    "hasUnit":                "URI to Unit Ontology (UO) for standardization",
    "pharmacologicalRoleLabel":"Pharmacological role: agonist, antagonist, inhibitor…",
    "hasPharmacologicalRole": "URI to ChEBI for pharmacological role",
    "isBaoReferenceCompound": "Whether compound is a BAO reference compound",
    "doseReferenceCompound":  "Reference compound dose",
    "singleConcentrationScreen":"Whether experiment is a single-concentration screen",
    "chemblAssayId":          "ChEMBL assay identifier",
    "experimentalSetting":    "Experimental setting: in vitro / in vivo",
    "assayName":              "Assay name",
    "assayDescription":       "Detailed assay description",
    "bindingSiteDescription": "Binding site description",
    "assayMethod":            "Assay method (BAO)",
    "detectionMethod":        "Detection method: radioactive, HTRF, BRET…",
    "assayKit":               "Commercial/custom assay kit used",
    "assayProperty":          "Additional assay property from ChEMBL",
    "functionalBiasAssay1":   "Primary functional bias assay (beta-arrestin, cAMP…)",
    "functionalBiasAssay2":   "Secondary functional bias assay",
    "hasExperimentalSettingConcept":"URI to BAO for experimental setting concept",
    "targetName":             "Biological target name (opioid receptor…)",
    "chemblTargetId":         "Target identifier in ChEMBL",
    "targetType":             "Target type: single protein / protein family…",
    "targetSpecies":          "Model species: Homo sapiens, Rattus norvegicus…",
    "ncbiTaxonomyId":         "NCBI taxonomy identifier",
    "uniprotId":              "Protein identifier in UniProt",
    "proteinName":            "Protein name",
    "proteinFamilyName":      "Protein family (InterPro)",
    "proteinSubfamilyName":   "Protein subfamily (NCIT)",
    "proteinCategory":        "Category (InterPro)",
    "gpcrCategory":           "GPCR category (Drug Target Ontology)",
    "sameAs (UniProt)":       "Linked-data link to UniProt",
    "sameAs (InterPro)":      "Linked-data link to InterPro",
    "sameAs (DTO)":           "Linked-data link to Drug Target Ontology",
    "sameAs (BAO)":           "Linked-data link to BioAssay Ontology",
    "sameAs (NCIT)":          "Linked-data link to NCI Thesaurus",
    "cellLineName":           "Cell line name: CHO, HEK293, AtT-20…",
    "cellosaurusId":          "Identifier in Cellosaurus (cell line database)",
    "cloId":                  "Identifier in Cell Line Ontology (CLO)",
    "tissueName":             "Tissue name: brain, vas deferens, intestine…",
    "btoId":                  "BRENDA Tissue Ontology identifier (BTO)",
    "ncbiTissueTaxonomy":     "Tissue taxonomy (NCBI)",
    "vertebrateTaxonomy":     "Vertebrate animal taxonomy",
    "animalModel":            "Animal model (NCIT)",
    "animalModelStrain":      "Animal strain",
    "rdfs:label":             "Model system name (NCIT)",
    "rdfs:label (name)":      "Model system name (NCIT)",
    "assayFormatL2":          "Assay format level 2 (BAO): cell-based, cell-free…",
    "assayFormatL3":          "Assay format level 3 (BAO): further detail",
    "subcellularFormat":      "Subcellular location of assay",
    "pathwayName":            "Signaling pathway name (GPCR signaling pathway)",
    "goId":                   "Gene Ontology ID for signaling pathway",
    "pubmedId":               "PubMed identifier of scientific publication",
    "documentDoi":            "DOI (Digital Object Identifier) of document",
    "chemblDocumentId":       "Document identifier in ChEMBL",
    "documentJournal":        "Journal name",
    "documentYear":           "Publication year",
    "patentId":               "Patent number (if data originates from a patent)",
    "sourceDescription":      "Source description",
    "databaseCitation":       "Database citation (MI ontology)",
    "routeOfAdministration":  "Drug administration route: IV, PO, SC, IP…",
    "doseAdministered":       "Dose administered in in vivo experiment",
}

# ── Schema data (mirroring schema_visualization.html) ─────────────────────
NODES = [
    {"id": "Compound",     "count": 12906, "group": "core", "pos": (0.30, 0.40), "fields": [
        {"p": "chemblId",            "e": "chembl_compound_id",                      "t": "string"},
        {"p": "pubchemCid",          "e": "pubchem_cid",                             "t": "string"},
        {"p": "inchi",               "e": "rdkit_library_standard_inchi",            "t": "string"},
        {"p": "inchiKey",            "e": "rdkit_library_standard_inchi_key",        "t": "string"},
        {"p": "smiles",              "e": "rdkit_canonical_smiles",                  "t": "string"},
        {"p": "iupacName",           "e": "pubchem_iupac_name",                      "t": "string"},
        {"p": "chemicalEntityName",  "e": "chembl_chemical_entity_name",             "t": "string"},
        {"p": "molecularFormula",    "e": "rdkit_molecular_formula",                 "t": "string"},
        {"p": "molecularWeight",     "e": "rdkit_molecular_weight",                  "t": "float"},
        {"p": "maxPhase",            "e": "chembl_molecule_max_phase",               "t": "integer"},
        {"p": "ro5Violations",       "e": "chembl_#ro5_violations",                  "t": "integer"},
        {"p": "alogP",               "e": "chembl_alogp",                            "t": "float"},
        {"p": "isRadiolabeled",      "e": "reference_radiolabeled_molecular_entity", "t": "boolean"},
        {"p": "qpDipole",            "e": "qikprop_dipole",                          "t": "float"},
        {"p": "qpSASA",              "e": "qikprop_sasa",                            "t": "float"},
        {"p": "qpFISA",              "e": "qikprop_fisa",                            "t": "float"},
        {"p": "qpDonorHB",           "e": "qikprop_donor_hb",                        "t": "float"},
        {"p": "qpAcceptorHB",        "e": "qikprop_accpt_hb",                        "t": "float"},
        {"p": "qpLogPw",             "e": "qikprop_qplog_pw",                        "t": "float"},
        {"p": "qpLogPow",            "e": "qikprop_qplog_po/w",                      "t": "float"},
        {"p": "qpLogS",              "e": "qikprop_qplogs",                          "t": "float"},
        {"p": "qpLogKhsa",           "e": "qikprop_qplog_khsa",                      "t": "float"},
        {"p": "humanOralAbsorption", "e": "qikprop_percent_human_oral_absorption",   "t": "float"},
    ]},
    {"id": "Activity",     "count": 36523, "group": "core", "pos": (0.30, 0.60), "fields": [
        {"p": "endpointType",              "e": "endpoint",                              "t": "string"},
        {"p": "endpointQualifier",         "e": "endpoint_qualifier",                   "t": "string"},
        {"p": "endpointValue",             "e": "endpoint_value",                       "t": "float"},
        {"p": "endpointDescription",       "e": "odo_assay_endpoint_description",       "t": "string"},
        {"p": "semQualifier",              "e": "sem_endpoint_qualifier",               "t": "string"},
        {"p": "semValue",                  "e": "sem_value",                            "t": "float"},
        {"p": "clLower95",                 "e": "cl_lower_95%",                         "t": "string"},
        {"p": "clUpper95",                 "e": "cl_upper_95%",                         "t": "string"},
        {"p": "pchemblValue",              "e": "pchembl_value",                        "t": "float"},
        {"p": "unitLabel",                 "e": "unit_of_measurement",                  "t": "string"},
        {"p": "hasUnit",                   "e": "uo_id",                                "t": "uri"},
        {"p": "pharmacologicalRoleLabel",  "e": "compound_pharmacological_role",        "t": "string"},
        {"p": "hasPharmacologicalRole",    "e": "chebi_compound_pharmacological_role_id","t": "uri"},
        {"p": "isBaoReferenceCompound",    "e": "bao_reference_compound",              "t": "string"},
        {"p": "doseReferenceCompound",     "e": "dose_reference_compound",             "t": "string"},
        {"p": "singleConcentrationScreen", "e": "single_concentration_screen",         "t": "string"},
    ]},
    {"id": "Assay",        "count": 4515,  "group": "core", "pos": (0.48, 0.22), "fields": [
        {"p": "chemblAssayId",                "e": "chembl_assay_id",                "t": "string"},
        {"p": "experimentalSetting",          "e": "bao_experimental_setting",       "t": "string"},
        {"p": "assayName",                    "e": "odo_assay",                      "t": "string"},
        {"p": "assayDescription",             "e": "assay_description",              "t": "string"},
        {"p": "bindingSiteDescription",       "e": "chembl_binding_site_description","t": "string"},
        {"p": "assayMethod",                  "e": "bao_assay_method",               "t": "string"},
        {"p": "detectionMethod",              "e": "physical_detection_method",      "t": "string"},
        {"p": "assayKit",                     "e": "assay_kit",                      "t": "string"},
        {"p": "assayProperty",                "e": "chembl_assay_property",          "t": "string"},
        {"p": "functionalBiasAssay1",         "e": "odo_functional_bias_assay_1",    "t": "string"},
        {"p": "functionalBiasAssay2",         "e": "odo_functional_bias_assay_2",    "t": "string"},
        {"p": "hasExperimentalSettingConcept","e": "bao_experimental_setting_id",    "t": "uri"},
    ]},
    {"id": "Target",       "count": 43,    "group": "core", "pos": (0.55, 0.40), "fields": [
        {"p": "targetName",     "e": "target_name",            "t": "string"},
        {"p": "chemblTargetId", "e": "chembl_target_id",        "t": "string"},
        {"p": "targetType",     "e": "target_type",             "t": "string"},
        {"p": "targetSpecies",  "e": "ncbi_target_taxonomy",    "t": "string"},
        {"p": "ncbiTaxonomyId", "e": "ncbi_target_taxonomy_id", "t": "string"},
    ]},
    {"id": "Protein",      "count": 77,    "group": "bio",  "pos": (0.68, 0.18), "fields": [
        {"p": "uniprotId",            "e": "uniprot_protein_id",              "t": "string"},
        {"p": "proteinName",          "e": "protein_name",                    "t": "string"},
        {"p": "proteinFamilyName",    "e": "interpro_protein_family_name",    "t": "string"},
        {"p": "proteinSubfamilyName", "e": "ncit_protein_subfamily_name",     "t": "string"},
        {"p": "proteinCategory",      "e": "interpro_protein_category",       "t": "string"},
        {"p": "gpcrCategory",         "e": "dto_gpcr_category",               "t": "string"},
        {"p": "sameAs (UniProt)",     "e": "uniprot_protein_id -> URI",       "t": "uri"},
        {"p": "sameAs (InterPro)",    "e": "interpro_protein_family_name_id", "t": "uri"},
        {"p": "sameAs (DTO)",         "e": "dto_gpcr_category_id",            "t": "uri"},
    ]},
    {"id": "ModelSystem",  "count": 137,   "group": "bio",  "pos": (0.70, 0.55), "fields": [
        {"p": "rdfs:label",   "e": "ncit_model_system",    "t": "string"},
        {"p": "sameAs (NCIT)","e": "ncit_model_system_id", "t": "uri"},
    ]},
    {"id": "CellLine",     "count": 63,    "group": "bio",  "pos": (0.85, 0.45), "fields": [
        {"p": "cellLineName",  "e": "cell_line_name",           "t": "string"},
        {"p": "cellosaurusId", "e": "cellosaurus_cell_line_id", "t": "string"},
        {"p": "cloId",         "e": "clo_cell_line_id",         "t": "string"},
    ]},
    {"id": "Tissue",       "count": 20,    "group": "bio",  "pos": (0.82, 0.62), "fields": [
        {"p": "tissueName",         "e": "tissue_name",            "t": "string"},
        {"p": "btoId",              "e": "bto_tissue_id",           "t": "string"},
        {"p": "ncbiTissueTaxonomy", "e": "ncbi_tissue_taxonomy",    "t": "string"},
        {"p": "ncbiTaxonomyId",     "e": "ncbi_tissue_taxonomy_id", "t": "string"},
    ]},
    {"id": "Organism",     "count": 13,    "group": "bio",  "pos": (0.72, 0.72), "fields": [
        {"p": "vertebrateTaxonomy", "e": "ncbi_target_taxonomy",            "t": "string"},
        {"p": "ncbiTaxonomyId",     "e": "ncbi_target_taxonomy_id",         "t": "string"},
        {"p": "animalModel",        "e": "ncit_animal_model",               "t": "string"},
        {"p": "animalModelStrain",  "e": "ncit_animal_animal_model_strain", "t": "string"},
    ]},
    {"id": "ParentCompound","count": 4424, "group": "meta", "pos": (0.12, 0.25), "fields": [
        {"p": "inchi",            "e": "rdkit_parent_structure_inchi",             "t": "string"},
        {"p": "inchiKey",         "e": "rdkit_parent_structure_inchi_key",         "t": "string"},
        {"p": "smiles",           "e": "rdkit_parent_structure_smiles",            "t": "string"},
        {"p": "molecularFormula", "e": "rdkit_parent_structure_molecular_formula", "t": "string"},
        {"p": "molecularWeight",  "e": "rdkit_parent_structure_molecular_weight",  "t": "float"},
    ]},
    {"id": "Document",     "count": 1165,  "group": "meta", "pos": (0.48, 0.72), "fields": [
        {"p": "pubmedId",         "e": "pubmed_id",           "t": "string"},
        {"p": "documentDoi",      "e": "document_doi",         "t": "string"},
        {"p": "chemblDocumentId", "e": "chembl_document_id",   "t": "string"},
        {"p": "documentJournal",  "e": "document_journal",     "t": "string"},
        {"p": "documentYear",     "e": "document_year",        "t": "integer"},
        {"p": "patentId",         "e": "patent_id",            "t": "string"},
        {"p": "sourceDescription","e": "source_description",   "t": "string"},
        {"p": "databaseCitation", "e": "mi_database_citation", "t": "string"},
    ]},
    {"id": "SignalingPathway","count": 2,  "group": "meta", "pos": (0.10, 0.55), "fields": [
        {"p": "pathwayName","e": "embl_ebi_gpcr_signaling_pathway",    "t": "string"},
        {"p": "goId",       "e": "embl_ebi_gpcr_signaling_pathway_id", "t": "string"},
    ]},
    {"id": "AssayFormat",  "count": 5,     "group": "meta", "pos": (0.70, 0.38), "fields": [
        {"p": "rdfs:label",      "e": "bao_assay_format",    "t": "string"},
        {"p": "assayFormatL2",   "e": "bao_assay_format_l2", "t": "string"},
        {"p": "assayFormatL3",   "e": "bao_assay_format_l3", "t": "string"},
        {"p": "subcellularFormat","e": "subcellular_format",  "t": "string"},
        {"p": "sameAs (BAO)",    "e": "bao_assay_format_id", "t": "uri"},
    ]},
    {"id": "BioassayType", "count": 4,     "group": "meta", "pos": (0.82, 0.28), "fields": [
        {"p": "rdfs:label",  "e": "bao_bioassay_type",    "t": "string"},
        {"p": "sameAs (BAO)","e": "bao_bioassay_type_id", "t": "uri"},
    ]},
    {"id": "InVivoParameters","count": 370,"group": "meta", "pos": (0.10, 0.75), "fields": [
        {"p": "routeOfAdministration","e": "ncit_route_of_administration",   "t": "string"},
        {"p": "doseAdministered",     "e": "chembl_dose_administered",        "t": "string"},
        {"p": "sameAs (NCIT)",        "e": "ncit_route_of_administration_id", "t": "uri"},
    ]},
]

LINKS = [
    {"source": "Compound",    "target": "Assay",            "label": "testedIn"},
    {"source": "Compound",    "target": "Activity",         "label": "hasActivity"},
    {"source": "Compound",    "target": "ParentCompound",   "label": "hasParentCompound"},
    {"source": "Activity",    "target": "Compound",         "label": "hasCompound"},
    {"source": "Activity",    "target": "Assay",            "label": "hasAssay"},
    {"source": "Activity",    "target": "Target",           "label": "hasTarget"},
    {"source": "Activity",    "target": "SignalingPathway",  "label": "hasSignalingPathway"},
    {"source": "Activity",    "target": "Document",         "label": "publishedIn"},
    {"source": "Activity",    "target": "InVivoParameters", "label": "hasInVivoParameters"},
    {"source": "Assay",       "target": "Target",           "label": "targetsReceptor"},
    {"source": "Assay",       "target": "Protein",          "label": "hasProtein"},
    {"source": "Assay",       "target": "ModelSystem",      "label": "hasModelSystem"},
    {"source": "Assay",       "target": "AssayFormat",      "label": "hasAssayFormat"},
    {"source": "Assay",       "target": "BioassayType",     "label": "hasBioassayType"},
    {"source": "Assay",       "target": "Document",         "label": "publishedIn"},
    {"source": "Target",      "target": "Protein",          "label": "encodedBy"},
    {"source": "ModelSystem", "target": "CellLine",         "label": "usesCellLine"},
    {"source": "ModelSystem", "target": "Tissue",           "label": "usesTissue"},
    {"source": "ModelSystem", "target": "Organism",         "label": "usesOrganism"},
]

# ── helpers ────────────────────────────────────────────────────────────────
def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

def draw_rounded_rect(ax, x, y, w, h, r, fc, ec, lw=1.0, alpha=1.0, zorder=2):
    fancy = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(fancy)

def truncate(s, n):
    return s if len(s) <= n else s[:n-1] + "…"

# ── PAGE 1: title ──────────────────────────────────────────────────────────
def page_title(pdf):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ── thin accent bar across the top ──
    ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color="#16213e", zorder=1, transform=ax.transData))
    ax.axhline(0.93, color="#e94560", lw=2, zorder=3)

    # ── main title ──
    ax.text(0.5, 0.80, "ODO Knowledge Graph",
            ha="center", va="center",
            fontsize=40, fontweight="bold", color="#e94560", zorder=5)
    ax.text(0.5, 0.70, "Schema Visualization",
            ha="center", va="center",
            fontsize=26, color="#cccccc", zorder=5)

    # ── thin divider ──
    ax.axhline(0.62, xmin=0.15, xmax=0.85, color="#0f3460", lw=1.2, zorder=3)

    # ── stats cards (4, evenly spaced) ──
    # 4 cards each 0.17 wide, gap 0.04  → total span = 4*0.17 + 3*0.04 = 0.80
    # left edge = (1 - 0.80) / 2 = 0.10
    stats = [
        ("15",       "Classes"),
        ("19",       "Object Properties"),
        ("84",       "Datatype Fields"),
        ("915,951",  "Triples"),
    ]
    card_w, card_h = 0.175, 0.145
    gap = 0.04
    total_w = len(stats) * card_w + (len(stats) - 1) * gap
    start_x = (1 - total_w) / 2
    card_y = 0.44

    for i, (val, lbl) in enumerate(stats):
        cx = start_x + i * (card_w + gap)
        cx_center = cx + card_w / 2
        # card with border gradient feel
        draw_rounded_rect(ax, cx, card_y, card_w, card_h, 0.018,
                          fc="#16213e", ec="#2a4a7f", lw=1.8, zorder=3)
        # big number
        ax.text(cx_center, card_y + card_h * 0.65, val,
                ha="center", va="center",
                fontsize=24, fontweight="bold", color="#e94560", zorder=5)
        # label
        ax.text(cx_center, card_y + card_h * 0.22, lbl,
                ha="center", va="center",
                fontsize=9, color="#9aabb8", zorder=5)

    # ── legend (color groups) ──
    # centered block of 3 items
    leg_items = list(GROUPS.values())
    leg_sq = 0.018
    leg_gap = 0.005
    # estimate total width: sq + gap + text (~0.17 each item) + spacing between items (0.06)
    item_w = 0.20
    leg_total = len(leg_items) * item_w
    leg_start = (1 - leg_total) / 2
    leg_y = 0.295

    ax.text(0.5, leg_y + 0.055, "Node Groups",
            ha="center", va="center", fontsize=8, color="#556677", zorder=5)

    for i, g in enumerate(leg_items):
        lx = leg_start + i * item_w
        draw_rounded_rect(ax, lx, leg_y - leg_sq / 2, leg_sq, leg_sq, 0.004,
                          fc=g["fill"], ec=g["stroke"], lw=1.2, zorder=5)
        ax.text(lx + leg_sq + leg_gap + 0.004, leg_y, g["label"],
                ha="left", va="center",
                fontsize=9.5, color="#d0d0d0", zorder=5)

    # ── thin bottom divider + footer ──
    ax.axhline(0.13, xmin=0.15, xmax=0.85, color="#0f3460", lw=1.0, zorder=3)
    ax.text(0.5, 0.08, "Generated 2026-06-02   ·   ODO Project",
            ha="center", va="center", fontsize=8.5, color="#445566", zorder=5)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)

# ── PAGE 2: schema graph ───────────────────────────────────────────────────
def page_graph(pdf):
    fig, ax = plt.subplots(figsize=(16.54, 11.69))  # A3 landscape
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    W, H = 16.54, 11.69
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    # header bar
    hbar = plt.Rectangle((0, H-0.55), W, 0.55, color=HEADER_BG, zorder=1)
    ax.add_patch(hbar)
    ax.text(0.15, H-0.27, "ODO Knowledge Graph — Schema Overview",
            ha="left", va="center", fontsize=13, fontweight="bold",
            color="#e94560", zorder=5)
    ax.text(W-0.15, H-0.27, "15 classes · 19 object properties · 84 datatype fields · 915,951 triples",
            ha="right", va="center", fontsize=9, color="#888888", zorder=5)

    pos_map = {n["id"]: (n["pos"][0]*W*0.92 + W*0.04,
                          (1-n["pos"][1])*(H-0.9)*0.92 + 0.3)
               for n in NODES}
    nw, nh = 1.55, 0.48

    # ── edges ──
    # Curvature (arc3 rad) per edge — prevents visual overlap on the static layout.
    # Compound↔Activity share the same y, so they need the largest separation (±0.28).
    # Edges going to the same destination (Target, Protein, Document) get opposite bends.
    LINK_RADS = {
        ("Compound",    "Assay"):             0.0,
        ("Compound",    "Activity"):         -0.28,  # bidirectional pair: bend down
        ("Compound",    "ParentCompound"):    0.0,
        ("Activity",    "Compound"):          0.28,  # bidirectional pair: bend up
        ("Activity",    "Assay"):             0.18,  # separate from Compound→Assay
        ("Activity",    "Target"):           -0.10,
        ("Activity",    "SignalingPathway"):  0.0,
        ("Activity",    "Document"):         -0.14,  # opposite to Assay→Document
        ("Activity",    "InVivoParameters"): 0.0,
        ("Assay",       "Target"):            0.10,  # opposite to Activity→Target
        ("Assay",       "Protein"):           0.0,
        ("Assay",       "ModelSystem"):      -0.20,
        ("Assay",       "AssayFormat"):      -0.10,
        ("Assay",       "BioassayType"):      0.18,
        ("Assay",       "Document"):          0.14,  # opposite to Activity→Document
        ("Target",      "Protein"):           0.0,
        ("ModelSystem", "CellLine"):         -0.14,
        ("ModelSystem", "Tissue"):            0.0,
        ("ModelSystem", "Organism"):          0.14,
    }

    for lk in LINKS:
        s, t = lk["source"], lk["target"]
        sx, sy = pos_map[s]
        tx, ty = pos_map[t]
        grp = GROUPS[next(n for n in NODES if n["id"] == s)["group"]]
        rad = LINK_RADS.get((s, t), 0.0)

        # Clip arrow endpoints to node boundary
        dx, dy = tx - sx, ty - sy
        ln = np.sqrt(dx*dx + dy*dy) or 1
        ca, sa = dx/ln, dy/ln
        clip = min(nw/2 / (abs(ca) + 1e-9), nh/2 / (abs(sa) + 1e-9))
        x1, y1 = sx + ca*clip, sy + sa*clip
        x2, y2 = tx - ca*clip, ty - sa*clip

        # FancyArrowPatch: always has a proper arrowhead, even on curved paths
        arrow = FancyArrowPatch(
            posA=(x1, y1), posB=(x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.1,
            edgecolor=grp["stroke"],
            facecolor=grp["stroke"],
            alpha=0.72,
            zorder=2,
            transform=ax.transData,
            clip_on=False,
        )
        ax.add_patch(arrow)

        # Edge label at the arc midpoint
        # arc3 midpoint formula: chord_mid + 0.5*rad*(perpendicular_chord_vector)
        mid_x = (x1+x2)/2 + 0.5 * rad * (y2-y1)
        mid_y = (y1+y2)/2 - 0.5 * rad * (x2-x1)
        ang = np.degrees(np.arctan2(ty - sy, tx - sx))
        if ang > 90 or ang < -90:
            ang += 180
        ax.text(mid_x, mid_y, lk["label"],
                ha="center", va="center",
                fontsize=5.5, color="#c8c8c8", alpha=0.78,
                rotation=ang, rotation_mode="anchor",
                bbox=dict(facecolor=BG, edgecolor="none", pad=0.6, alpha=0.75),
                zorder=4)

    # ── nodes ──
    for nd in NODES:
        x, y = pos_map[nd["id"]]
        grp = GROUPS[nd["group"]]
        # shadow
        draw_rounded_rect(ax, x-nw/2+0.04, y-nh/2-0.04, nw, nh, 0.07,
                          fc="#000000", ec="none", lw=0, alpha=0.35, zorder=3)
        # body
        draw_rounded_rect(ax, x-nw/2, y-nh/2, nw, nh, 0.07,
                          fc=grp["fill"], ec=grp["stroke"], lw=1.5, zorder=4)
        # class name
        ax.text(x, y+0.09, nd["id"], ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=6)
        # count + fields
        ax.text(x, y-0.04,
                f"{nd['count']:,}  ·  {len(nd['fields'])} fields",
                ha="center", va="center",
                fontsize=7, color="#cccccc", alpha=0.65, zorder=6)
        # click hint — mirrors the HTML "▾ click for fields"
        ax.text(x, y-0.17, "▾ click for details", ha="center", va="center",
                fontsize=6, color="white", alpha=0.32, zorder=6)

    # ── legend ──
    lx, ly = 0.12, 0.52
    draw_rounded_rect(ax, lx-0.05, ly-0.08, 1.65, 0.56, 0.07,
                      fc="#16213e", ec="#0f3460", lw=1, alpha=0.92, zorder=7)
    ax.text(lx + 0.70, ly + 0.38, "Groups", ha="center", va="center",
            fontsize=8, color="#888888", zorder=8)
    for i, (key, g) in enumerate(GROUPS.items()):
        ry = ly + 0.20 - i*0.14
        draw_rounded_rect(ax, lx, ry-0.04, 0.09, 0.09, 0.02,
                          fc=g["fill"], ec=g["stroke"], lw=0.8, zorder=8)
        ax.text(lx + 0.13, ry, g["label"], ha="left", va="center",
                fontsize=8, color="#dddddd", zorder=8)

    # No tight_layout — keep exact data-coords → PDF-point mapping (72 pts/inch)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)

# ── PAGE per class: fields ──────────────────────────────────────────────────
def page_class(pdf, nd):
    # Fixed A4 landscape for ALL class pages — no variable heights that confuse viewers
    FW, FH = 11.69, 8.27
    COLS = 3
    HEADER_H = 0.55
    BADGE_H  = 0.44
    BADGE_Y  = FH - HEADER_H - 0.18 - BADGE_H   # top of badge
    SEP_Y    = BADGE_Y - 0.20                    # separator line y
    CONTENT_TOP = SEP_Y - 0.14                   # top of card area
    CARD_H   = 0.48
    ROW_GAP  = 0.09
    COL_GAP  = 0.12
    MARGIN   = 0.28
    CARD_W   = (FW - 2*MARGIN - (COLS-1)*COL_GAP) / COLS  # ≈3.60"

    n_fields = len(nd["fields"])
    grp = GROUPS[nd["group"]]

    fig, ax = plt.subplots(figsize=(FW, FH))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.axis("off")

    # ── field cards (drawn first so header always wins on top) ──
    for i, f in enumerate(nd["fields"]):
        col = i % COLS
        row = i // COLS
        cx = MARGIN + col * (CARD_W + COL_GAP)
        cy = CONTENT_TOP - (row + 1) * CARD_H - row * ROW_GAP

        tc = TYPE_COLORS.get(f["t"], TYPE_COLORS["string"])
        draw_rounded_rect(ax, cx, cy, CARD_W, CARD_H, 0.05,
                          fc="#0e1e30", ec=grp["stroke"], lw=0.7, alpha=0.9, zorder=3)

        # type badge
        bw = 0.52
        draw_rounded_rect(ax, cx + CARD_W - bw - 0.05, cy + CARD_H - 0.21,
                          bw, 0.16, 0.03, fc=tc["bg"], ec="none", lw=0, zorder=4)
        ax.text(cx + CARD_W - bw/2 - 0.05, cy + CARD_H - 0.13,
                f["t"].upper(), ha="center", va="center",
                fontsize=6.5, color=tc["fg"], fontfamily="monospace", zorder=5)

        # property name
        ax.text(cx + 0.09, cy + CARD_H - 0.13, truncate(f["p"], 26),
                ha="left", va="center", fontsize=9,
                color="#81d4fa", fontfamily="monospace", zorder=5)
        # excel column
        ax.text(cx + 0.09, cy + CARD_H - 0.29, truncate(f["e"], 32),
                ha="left", va="center", fontsize=7.5,
                color="#a5d6a7", alpha=0.72, fontfamily="monospace", zorder=5)
        # description
        ax.text(cx + 0.09, cy + 0.10, truncate(DESC.get(f["p"], ""), 54),
                ha="left", va="center", fontsize=6.5, color="#aaaaaa", zorder=5)

    # ── HEADER — drawn LAST with highest zorder so it always shows ──
    # Header bar background
    ax.add_patch(mpatches.Rectangle(
        (0, FH - HEADER_H), FW, HEADER_H,
        color=HEADER_BG, zorder=20, clip_on=False))
    # Header bottom border
    ax.axhline(FH - HEADER_H, color="#0f3460", lw=1.2, zorder=21)
    # Header text
    ax.text(0.25, FH - HEADER_H/2,
            f"ODO Knowledge Graph  —  {nd['id']}",
            ha="left", va="center", fontsize=13, fontweight="bold",
            color="#e94560", zorder=22)
    ax.text(FW - 0.25, FH - HEADER_H/2,
            f"{nd['group'].upper()}   ·   {nd['count']:,} instances   ·   {n_fields} fields",
            ha="right", va="center", fontsize=9, color="#888888", zorder=22)

    # Class badge
    bw2, bh2 = 2.6, BADGE_H
    draw_rounded_rect(ax, FW/2 - bw2/2, BADGE_Y, bw2, bh2, 0.08,
                      fc=grp["fill"], ec=grp["stroke"], lw=2.0, zorder=20)
    ax.text(FW/2, BADGE_Y + bh2/2, nd["id"],
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="white", zorder=22)

    # Section label + separator
    ax.text(FW/2, SEP_Y + 0.08,
            f"Datatype Properties  ({n_fields})",
            ha="center", va="center", fontsize=8, color="#556677", zorder=10)
    ax.axhline(SEP_Y - 0.01, xmin=0.03, xmax=0.97,
               color=grp["stroke"], lw=0.5, alpha=0.35, zorder=10)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)

# ── PAGE: object properties reference ─────────────────────────────────────
def page_links(pdf):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    FW, FH_ax = 11.69, 8.27
    ax.set_xlim(0, FW); ax.set_ylim(0, FH_ax)
    ax.axis("off")

    hbar = plt.Rectangle((0, FH_ax-0.55), FW, 0.55, color=HEADER_BG, zorder=1)
    ax.add_patch(hbar)
    ax.text(0.2, FH_ax-0.27, "ODO Knowledge Graph — Object Properties",
            ha="left", va="center", fontsize=13, fontweight="bold",
            color="#e94560", zorder=5)
    ax.text(FW-0.2, FH_ax-0.27, "19 object properties",
            ha="right", va="center", fontsize=9, color="#888888", zorder=5)

    # table header
    hy = FH_ax - 0.80
    ax.text(0.25,  hy, "Property",  ha="left",  fontsize=9, fontweight="bold", color="#e0e0e0", zorder=5)
    ax.text(3.8,   hy, "Domain",    ha="left",  fontsize=9, fontweight="bold", color="#e0e0e0", zorder=5)
    ax.text(6.8,   hy, "Range",     ha="left",  fontsize=9, fontweight="bold", color="#e0e0e0", zorder=5)
    ax.axhline(hy - 0.05, xmin=0.02, xmax=0.98, color="#0f3460", lw=1, zorder=3)

    node_group = {n["id"]: n["group"] for n in NODES}
    row_h = (FH_ax - 1.15) / (len(LINKS) + 1)

    for i, lk in enumerate(LINKS):
        ry = hy - 0.18 - i * row_h
        bg_col = "#111827" if i % 2 == 0 else "#0d1520"
        draw_rounded_rect(ax, 0.1, ry - row_h*0.38, FW-0.2, row_h*0.78, 0.03,
                          fc=bg_col, ec="none", lw=0, zorder=2)

        sg = GROUPS[node_group[lk["source"]]]["stroke"]
        tg = GROUPS[node_group[lk["target"]]]["stroke"]

        ax.text(0.25,  ry, lk["label"], ha="left", fontsize=8.5,
                color="#81d4fa", fontfamily="monospace", zorder=5)
        ax.text(3.8,   ry, lk["source"], ha="left", fontsize=8.5,
                color=sg, zorder=5)
        ax.text(6.8,   ry, lk["target"], ha="left", fontsize=8.5,
                color=tg, zorder=5)

        # arrow glyph between domain and range
        ax.annotate("", xy=(6.6, ry), xytext=(5.8, ry),
            arrowprops=dict(arrowstyle="-|>", color="#555577", lw=0.8,
                            mutation_scale=8),
            zorder=4)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)

# ── PDF link annotations (pikepdf post-processing) ─────────────────────────
def add_pdf_links(pdf_path):
    """Add internal /Link annotations on the graph page so every node
    rectangle is clickable and jumps to the corresponding class detail page."""
    import pikepdf
    from pikepdf import Dictionary, Array, Name
    import tempfile, shutil

    # Graph page coordinate system:
    #   figure = 16.54 × 11.69 in (A3 landscape)
    #   subplots_adjust(0,0,1,1) → axes fills figure exactly
    #   data coords in inches  →  PDF points = data × 72  (y=0 at bottom)
    FIG_W, FIG_H = 16.54, 11.69
    NW_IN, NH_IN = 1.55, 0.48      # node width/height in inches
    PAD = 0.06                      # extra click-area padding (inches)

    # Page layout:  0=title  1=graph  2=object-props  3..17=class pages (NODES order)
    node_to_page_idx = {nd["id"]: i + 3 for i, nd in enumerate(NODES)}

    def node_pdf_rect(nd):
        px, py = nd["pos"]
        x = px * FIG_W * 0.92 + FIG_W * 0.04
        y = (1 - py) * (FIG_H - 0.9) * 0.92 + 0.3
        x1 = (x - NW_IN/2 - PAD) * 72
        y1 = (y - NH_IN/2 - PAD) * 72
        x2 = (x + NW_IN/2 + PAD) * 72
        y2 = (y + NH_IN/2 + PAD) * 72
        return [x1, y1, x2, y2]

    with pikepdf.open(pdf_path) as pdf:
        graph_page = pdf.pages[1]
        annots = []

        for nd in NODES:
            rect = node_pdf_rect(nd)
            target = pdf.pages[node_to_page_idx[nd["id"]]]

            annot = pdf.make_indirect(Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array(rect),
                Border=Array([0, 0, 0]),   # invisible border (node already styled)
                H=Name.I,                  # invert on click
                A=Dictionary(
                    Type=Name.Action,
                    S=Name.GoTo,
                    D=Array([target.obj, Name.Fit]),
                ),
            ))
            annots.append(annot)

        graph_page["/Annots"] = Array(annots)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        pdf.save(tmp_path)

    shutil.move(tmp_path, pdf_path)
    print(f"  ✓ {len(NODES)} clickable node links added to graph page")

# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = "ODO_Knowledge_Graph_Schema.pdf"
    with PdfPages(out) as pdf:
        print("  Page 1: title …")
        page_title(pdf)

        print("  Page 2: schema graph …")
        page_graph(pdf)

        print("  Page 3: object properties …")
        page_links(pdf)

        for nd in NODES:
            print(f"  Page for class: {nd['id']} …")
            page_class(pdf, nd)

        meta = pdf.infodict()
        meta["Title"]   = "ODO Knowledge Graph — Schema"
        meta["Author"]  = "ODO Project"
        meta["Subject"] = "Schema visualization: 15 classes, 19 object properties, 84 datatype fields"

    print("\n  Post-processing: adding clickable links …")
    add_pdf_links(out)
    print(f"\nDone -> {out}")
