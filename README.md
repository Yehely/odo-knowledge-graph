# ODO Knowledge Graph – Opioid Drug-Receptor Interactions

A knowledge graph built from the ODO 2025 database, containing **37,362 biological activity measurements** across **~13,350 chemical compounds** tested against opioid receptors (mu, kappa, delta, NOP).

Built as a final project for a B.Sc. in Bioinformatics.

---

## What the Graph Contains

| Entity | Count |
|---|---|
| Chemical compounds | ~13,350 |
| Activity measurements (Ki / IC50 / EC50 / ...) | 37,362 |
| Scientific publications | 1,165 |
| Biological targets (receptors) | 55 |
| Proteins (GPCRs) | 12 |
| Cell lines | 63 |
| Tissues | 20 |
| Biological species | 13 |
| **Total RDF triples** | **~960,000** |

The graph is linked to external databases via `owl:sameAs`: **ChEMBL**, **PubChem**, **UniProt**, **Cellosaurus**, **BTO**.

---

## Requirements

- Python 3.8+
- Conda (recommended) with an `odo` environment:
  ```bash
  conda create -n odo python=3.10
  conda activate odo
  pip install pandas openpyxl rdflib requests
  ```
- [GraphDB Free](https://www.ontotext.com/products/graphdb/download/) installed and running on `localhost:7200`
- Source data file: `Final_updated_Dataset_v2025_11-12.xlsx` (not included in this repo)

---

## Quick Start

```bash
# Step 1 – Generate RDF files from the Excel dataset
conda run -n odo python3 build_kg.py

# Step 2 – Create the GraphDB repository and import data
conda run -n odo python3 setup_graphdb.py

# Step 3 – Validate the data (optional)
conda run -n odo python3 validate_kg.py
```

Then open: **http://localhost:7200**

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md) (Hebrew).

---

## Repository Structure

```
odo-knowledge-graph/
├── build_kg.py           # ETL pipeline: Excel → RDF Turtle (~960K triples)
├── setup_graphdb.py      # Creates GraphDB repository and imports data
├── validate_kg.py        # 14 SPARQL validation queries
├── odo_ontology.ttl      # OWL ontology: classes and properties
└── output/               # Generated RDF files (produced by build_kg.py)
    ├── compounds.ttl
    ├── assays.ttl
    ├── activities.ttl
    ├── model_systems.ttl
    ├── proteins_targets.ttl
    ├── documents.ttl
    └── signaling.ttl
```

---

## Ontology – Main Classes

```
odo:Compound         – Chemical compound
odo:Target           – Biological target (receptor)
odo:Protein          – Specific protein (GPCR)
odo:Assay            – Biological assay
odo:Activity         – Measurement result (Ki / IC50 / EC50 ...)
odo:CellLine         – Cell line (CHO, HEK293 ...)
odo:Tissue           – Tissue (brain, vas deferens ...)
odo:Organism         – Species (Homo sapiens, Rattus norvegicus ...)
odo:Document         – Scientific publication
odo:SignalingPathway – Intracellular signaling pathway
```

Namespace: `http://odo-project.org/ontology#`

---

## Example SPARQL Query

```sparql
-- Compounds with Ki < 1 nM at the mu opioid receptor
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?chemblId ?ki WHERE {
  ?act odo:endpointType "Ki" ;
       odo:endpointValue ?ki ;
       odo:hasCompound ?c ;
       odo:hasTarget [ odo:targetName "Mu opioid receptor" ] .
  ?c odo:chemblId ?chemblId .
  FILTER (?ki < 1.0)
} ORDER BY ?ki LIMIT 20
```

Run queries at: **http://localhost:7200/sparql**

---

## License

For academic use only. Data sourced from the ODO 2025 database.
