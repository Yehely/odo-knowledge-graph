# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All scripts run inside the `odo` conda environment.

```bash
# Step 1 – ETL: Excel → RDF Turtle files in output/
conda run -n odo python3 build_kg.py

# Step 2 – Load into GraphDB (must be running on localhost:7200)
conda run -n odo python3 setup_graphdb.py

# Step 3 – Run 16 SPARQL validation queries
conda run -n odo python3 validate_kg.py
```

GraphDB UI and SPARQL endpoint: **http://localhost:7200**  
Repository ID: **`odo-kg`**

## Architecture

The pipeline has three stages:

### 1. `build_kg.py` – ETL (Excel → Turtle)

Reads `Final ODO Dataset_v2026-06-10.xlsx` (sheet `Full Dataset`, ~37k rows) and writes 7 thematic Turtle files to `output/`:

| File | Contents |
|---|---|
| `compounds.ttl` | Chemical compounds + parent compounds |
| `proteins_targets.ttl` | Opioid receptors (Target) and GPCRs (Protein) |
| `model_systems.ttl` | CellLine, Tissue, Organism, ModelSystem |
| `signaling.ttl` | SignalingPathway nodes |
| `documents.ttl` | Publications (PubMed / DOI / ChEMBL) |
| `assays.ttl` | Assay, AssayFormat, BioassayType nodes |
| `activities.ttl` | One Activity node per row (the core measurement) |

**Two namespaces:**
- `odo:` → `http://odo-project.org/ontology#` — classes and properties (defined in `odo_ontology.ttl`)
- `odod:` → `http://odo-project.org/data#` — all data instances

**Entity deduplication:** A `seen` dict of sets tracks which entities have already been written. Each `build_*()` function returns the entity's URI (creating it only on first encounter) so downstream code can link to it.

**Activity URI:** Content-addressed — SHA-1 of compound+assay+target+endpoint+value+qualifier+pubmed_id, truncated to 16 hex chars. This keeps URIs stable across re-runs as long as source data is unchanged.

**`safe(val)`:** The universal NaN/None/empty-string guard. Every column read goes through it. Never access raw row values without it.

**`add_uri_sameAs(g, subject, uri_str)`:** Validates the URI with a regex before adding `owl:sameAs`. Use this whenever linking to external databases (ChEMBL, PubChem, UniProt, etc.).

### 2. `setup_graphdb.py` – Repository creation and import

Creates the `odo-kg` repository via GraphDB's REST API (Turtle config, `rdfsplus-optimized` ruleset), clears stale data, then imports files in dependency order: compounds → proteins_targets → model_systems → signaling → documents → assays → activities. Runs 4 summary SPARQL queries after import.

### 3. `validate_kg.py` – 16 SPARQL validation checks

Runs checks including: total triple count, entity counts per class, high-affinity binders (Ki < 1 nM), pharmacological roles per target, cell line coverage, linked-data `owl:sameAs` sampling, and integrity checks (e.g. activities with no compound).

### Ontology (`odo_ontology.ttl`)

Defines all OWL classes and properties. Key class hierarchy:
```
odo:Compound → odo:ParentCompound (subclass)
odo:Activity  links to Compound, Assay, Target, Document, SignalingPathway
odo:Assay     links to AssayFormat, BioassayType, ModelSystem, Target, Protein
odo:Target    → odo:encodedBy → odo:Protein
odo:ModelSystem links to CellLine, Tissue, Organism
```

External ontologies aligned via `owl:sameAs`: BAO, ChEBI, GO, BTO, CLO, NCIT, PRO, UO, UniProt, InterPro, ChEMBL, PubChem, Cellosaurus, DTO.
