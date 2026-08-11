# Setup Guide

Covers every part of the project: the knowledge-graph pipeline, both GNN
packages, and the `literature/` tools (paper acquisition, ChEMBL
auto-extraction, Llama-based extraction).

---

## 1. Base environment

```bash
conda create -n odo python=3.11
conda activate odo
```

Get the code and the dataset:

```bash
git clone https://github.com/Yehely/odo-project.git
cd odo-project
```

> Place `Final ODO Dataset_v2026-06-10.xlsx` **in the repo root** — it's
> not distributed via GitHub (see `.gitignore`).

---

## 2. Knowledge Graph pipeline

### 2.1 Install GraphDB

1. Go to https://www.ontotext.com/products/graphdb/download/
2. Download **GraphDB Free** (v11.x)
3. Install, then click **Start** — GraphDB comes up on port **7200**
4. Open `http://localhost:7200` in a browser to confirm it's running

### 2.2 Install Python packages

```bash
pip install pandas openpyxl rdflib requests
```

### 2.3 Run the pipeline

```bash
# Step 1 — Excel -> RDF Turtle files in kg/output/
conda run -n odo python3 kg/build_kg.py

# Step 2 — load into GraphDB (must be running on localhost:7200)
conda run -n odo python3 kg/setup_graphdb.py

# Step 3 — 16 SPARQL validation checks
conda run -n odo python3 kg/validate_kg.py
```

`build_kg.py` takes ~2-4 minutes and writes 7 Turtle files (~870K triples
total) to `kg/output/`. `setup_graphdb.py` creates the `odo-kg` repository
and imports them (~3-5 minutes).

### 2.4 Browse the graph

- **Web UI**: `http://localhost:7200`
- **Visual graph**: Explore → Visual Graph, search a full URI, e.g.
  `http://odo-project.org/data#compound_CHEMBL70`
- **SPARQL**: SPARQL tab → repository `odo-kg` → write a query → Run

```sparql
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?targetName (COUNT(DISTINCT ?c) AS ?n) WHERE {
  ?act odo:hasCompound ?c ; odo:hasTarget ?t .
  ?t odo:targetName ?targetName .
} GROUP BY ?targetName ORDER BY DESC(?n)
```

Namespace prefixes (add under GraphDB's **Setup → Namespaces**):

| Prefix | URI |
|---|---|
| `odo:` | `http://odo-project.org/ontology#` |
| `odod:` | `http://odo-project.org/data#` |

---

## 3. GNN models

Both packages read the same dataset file directly (no GraphDB required).

### 3.1 Bipartite GNN (`gnn/`)

```bash
pip install -r requirements_gnn.txt
# one-command setup + default training run:
chmod +x run_full_pipeline.sh && ./run_full_pipeline.sh
# or train a specific variant, e.g.:
conda run -n odo python3 train_gnn.py --split compound_random --fp-bits 2048
```

See the model list and results table in the main [README](../README.md#gnn-models--how-to-run-each).

### 3.2 Heterogeneous GNN (`hetero_gnn/`)

```bash
conda run -n odo python3 hetero_gnn/run_preprocess.py
conda run -n odo python3 hetero_gnn/run_train.py
```

See [`hetero_gnn/README.md`](../hetero_gnn/README.md) for the full option set (hyperparameter search, inference).

---

## 4. Literature tools (`literature/`)

### 4.1 Paper acquisition (`GetFiles.py` + `CleanFiles.py`)

```bash
pip install pandas biopython beautifulsoup4 lxml openpyxl
export ENTREZ_EMAIL="you@example.com"   # required by NCBI's usage policy

conda run -n odo python3 literature/GetFiles.py     # -> literature/Full_Text_Articles/
conda run -n odo python3 literature/CleanFiles.py   # -> literature/Cleaned_Text_Articles/

# optional: verified re-download if you suspect a mismatched article
conda run -n odo python3 literature/Fix_GetFiles.py   # -> literature/Verified_Full_Text/
# optional: filter out abstract-only stubs before running an extractor
conda run -n odo python3 literature/sortFullText.py   # -> literature/Sort_Full_Text_And_Not/
```

### 4.2 ChEMBL auto-extractor (`literature/chembl_extractor/`)

```bash
cd literature/chembl_extractor
pip install -r requirements.txt

python3 train_qikprop_models.py     # train supporting ML models (once)
python3 train_nlp_models.py

python3 chembl_fetcher.py --ids CHEMBL101454 --output outputs/result.xlsx   # CLI
python3 app.py                                                              # or web UI, http://localhost:5050
```

See [`literature/README.md`](../literature/README.md) for accuracy-evaluation commands.

### 4.3 Llama extractor (`literature/llama_extractor/`)

```bash
cd literature/llama_extractor
pip install -r requirements.txt
export GROQ_API_KEY="your-groq-key"   # https://console.groq.com

# from cleaned full-text (after 4.1):
python3 LlamaExtractor.py

# from manually downloaded PDFs — place them in manual_pdfs/ first (see manual_pdfs/README.md):
python3 LlamaExtractorDownloaded.py

# validate either output against the real dataset:
python3 Comparison.py --input labeled_data_from_llama.json
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` running `setup_graphdb.py` | Make sure GraphDB is running (click Start) |
| `Repository already exists` | Not an error — the script continues importing |
| `ImportError: No module named rdflib` | `conda activate odo && pip install rdflib` |
| Import fails partway through | Re-run — the script picks up where it left off |
| `KeyError: 'GROQ_API_KEY'` | `export GROQ_API_KEY=...` before running a `llama_extractor` script |
