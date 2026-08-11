# ODO Knowledge Graph — Opioid Drug-Receptor Interactions

A knowledge graph and GNN prediction pipeline built from the ODO 2026 database,
containing **~34,000 biological activity measurements** across **~12,906 chemical
compounds** tested against opioid receptors (MOR, DOR, KOR, NOP).

Built as a final project for a B.Sc. in Bioinformatics.

For deeper walkthroughs and background beyond this README, see the
**[Wiki](../../wiki)**.

---

## Requirements

- **Python 3.11** via Conda (`odo` environment)
- **Source data**: `Final ODO Dataset_v2026-06-10.xlsx` — place in the repo root (not in git)
- **GraphDB Free** (only for the Knowledge Graph pipeline, not required for GNN)

---

## Quick Start — Full Pipeline (one command)

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

This creates the conda environment, installs packages, builds the bipartite graph, and trains the default model (temporal split, 2048-bit FP).

---

## GNN Models — How to Run Each

There are two independent GNN packages in this repo — `gnn/` (the earlier,
2-node-type bipartite model) and `hetero_gnn/` (a newer, 5-node-type
heterogeneous model). They use different graphs, different training loops,
and are not interchangeable.

### Bipartite GNN (`gnn/`)

All commands use the `odo` conda environment. The preprocessing step runs automatically if the graph file does not exist yet.

#### Model 1 — Paper Model (Temporal Split, 2048-bit FP)

Train ≤ 2015, Test 2016–2020. Matches the architecture described in Progress Report 5.

```bash
conda run -n odo python3 train_gnn.py --split temporal --fp-bits 2048
```

**Expected results:** RMSE=1.26 | MAE=0.98 | Pearson r=0.53 | R²=0.19

---

#### Model 2 — Compound-Level Random Split (2048-bit FP)

20% of **compounds** held out for test — no compound appears in both train and test.
The most honest evaluation of generalisation to novel chemistry.

```bash
conda run -n odo python3 train_gnn.py --split compound_random --fp-bits 2048
```

**Expected results:** RMSE=0.93 | MAE=0.69 | Pearson r=0.77 | R²=0.55

---

#### Model 3 — Fine-Tuned (Temporal, 512-bit FP, stronger regularisation)

Reduced fingerprint + higher dropout (0.4) + higher weight decay (1e-4) + temporal sample weighting.

```bash
conda run -n odo python3 train_gnn.py --split temporal --fp-bits 512
```

**Expected results:** RMSE=1.26 | MAE=0.99 | Pearson r=0.51 | R²=0.19

---

#### Model 4 — Ensemble (5 models, temporal split)

Trains 5 independent models with different random seeds and averages their predictions.

```bash
conda run -n odo python3 train_ensemble.py --split temporal --fp-bits 2048
```

**Expected results:** RMSE=1.23 | MAE=0.98 | Pearson r=0.53 | R²=0.22

---

#### Model 5 — Diagnostic: Edge-Level Random Split

Randomly splits individual **experiments** (not compounds). Inflated results due to data leakage — same compound can appear in both train and test. For diagnostic/comparison only.

```bash
conda run -n odo python3 train_gnn.py --split random --fp-bits 2048
```

**Expected results:** RMSE=0.83 | MAE=0.60 | Pearson r=0.82 | R²=0.65 ⚠️ inflated

---

#### Results Summary

| Model | Split | FP bits | RMSE | R² | Notes |
|---|---|---|---|---|---|
| Paper model | temporal | 2048 | 1.26 | 0.19 | As per report |
| Compound-random | compound_random | 2048 | 0.93 | **0.55** | Most honest |
| Fine-tuned | temporal | 512 | 1.26 | 0.19 | No improvement |
| Ensemble (5×) | temporal | 2048 | 1.23 | **0.22** | Best temporal |
| Edge-random | random | 2048 | 0.83 | 0.65 | ⚠️ leakage |

---

### Heterogeneous GNN (`hetero_gnn/`)

A 5-node-type graph (**Compound, Target, Assay, Model System, Document**)
with 2-hop, attention-weighted hub-and-spoke message passing through Assay
nodes. Full spec in [`docs/HETEROGENEOUS_GNN_ARCHITECTURE.md`](docs/HETEROGENEOUS_GNN_ARCHITECTURE.md);
implementation notes and rationale in [`hetero_gnn/README.md`](hetero_gnn/README.md).

```bash
# Step 1 — Excel → hetero_gnn/processed_hetero_graph.pt
conda run -n odo python3 hetero_gnn/run_preprocess.py

# Step 2 — train
conda run -n odo python3 hetero_gnn/run_train.py

# Optional — hyperparameter search (Optuna), then trains the winning config
conda run -n odo python3 hetero_gnn/search_hparams.py --n-trials 20

# Inference on a new compound (after training)
conda run -n odo python3 hetero_gnn/predict.py
```

---

## Knowledge Graph Pipeline (GraphDB)

Requires GraphDB running on `localhost:7200`.

```bash
# Step 1 — Excel → RDF Turtle files
conda run -n odo python3 build_kg.py

# Step 2 — Load into GraphDB
conda run -n odo python3 setup_graphdb.py

# Step 3 — Validate (16 SPARQL checks)
conda run -n odo python3 validate_kg.py
```

---

## Literature Extraction (optional)

`literature/` fetches and cleans the full-text source papers behind the
dataset's `pubmed_id` column (via NCBI Entrez/PMC) — raw material for
LLM-assisted extraction or validation, not part of the KG/GNN pipeline
itself. See [`literature/README.md`](literature/README.md).

---

## Repository Structure

```
odo-knowledge-graph/
├── build_kg.py                   # ETL: Excel → RDF Turtle
├── setup_graphdb.py              # Load RDF into GraphDB
├── validate_kg.py                # 16 SPARQL validation queries
├── odo_ontology.ttl              # OWL ontology
├── run_full_pipeline.sh          # One-command setup and training
├── preprocess_bipartite_graph.py # Excel → processed_<split>_<fp>fp.pt
├── train_gnn.py                  # Train a single bipartite GNN model
├── train_ensemble.py             # Train 5-model bipartite ensemble
├── generate_bipartite_viz.py     # Bipartite graph HTML visualization
├── generate_report.py            # Insights report generator
├── generate_schema_pdf.py        # Ontology schema diagram (PDF)
├── gnn/                           # Bipartite (2-node-type) GraphSAGE model
│   ├── model.py                  # OpioidGNN architecture
│   ├── train.py                  # Training loop + asymmetric loss
│   ├── dataset.py                # Load .pt graph for training
│   ├── config.py                 # Hyperparameters
│   ├── features.py               # Feature engineering utilities
│   └── predict.py                # Inference on new SMILES
├── hetero_gnn/                    # Heterogeneous (5-node-type) GNN model
│   ├── model.py / config.py / dataset.py / preprocess.py
│   ├── train.py / run_train.py / run_preprocess.py / search_hparams.py
│   └── predict.py
├── literature/                    # PMC full-text acquisition + cleaning
│   ├── GetFiles.py / CleanFiles.py
│   └── README.md
└── docs/                          # Technical specs and setup guide
    ├── ARCHITECTURE_SKILLS.md          # Bipartite GNN architecture spec
    ├── HETEROGENEOUS_GNN_ARCHITECTURE.md  # Heterogeneous GNN architecture spec
    └── SETUP_GUIDE.md                  # Hebrew install/setup walkthrough
```

---

## Architecture

**OpioidGNN** (`gnn/`) — Bipartite Heterogeneous GraphSAGE

- Compound encoder: Linear(FP_bits+12 → 256) + BatchNorm + ReLU + Dropout(0.4)
- Target encoder: Linear(13+32 → 256) + BatchNorm + ReLU + Dropout(0.4)
- 2-hop bipartite message passing (HeteroConv, SAGEConv)
- Edge MLP head: 528 → 256 → 128 → 1 (pChEMBL regression)
- Loss: Asymmetric MSE (α=3 for high-affinity under-predictions, pChEMBL ≥ 7)

Full spec: [`docs/ARCHITECTURE_SKILLS.md`](docs/ARCHITECTURE_SKILLS.md).

**HeteroOpioidGNN** (`hetero_gnn/`) — 5-node-type Heterogeneous GNN

- Per-node-type encoders into a shared 256-dim latent space (Dropout 0.4)
- 2-hop hub-and-spoke message passing through Assay, using **learned
  attention-weighted pooling** (not a plain mean) at both hops
- Edge MLP head: ~530 → 256 → 128 → 1 (pKi regression)
- Loss: combined exact-MSE + censored-hinge (handles `<`/`>` qualified measurements)

Full spec: [`docs/HETEROGENEOUS_GNN_ARCHITECTURE.md`](docs/HETEROGENEOUS_GNN_ARCHITECTURE.md).

---

## Documentation

- [`docs/ARCHITECTURE_SKILLS.md`](docs/ARCHITECTURE_SKILLS.md) — bipartite GNN architecture spec
- [`docs/HETEROGENEOUS_GNN_ARCHITECTURE.md`](docs/HETEROGENEOUS_GNN_ARCHITECTURE.md) — heterogeneous GNN architecture spec
- [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) — Hebrew install/setup walkthrough
- [`hetero_gnn/README.md`](hetero_gnn/README.md) — implementation notes and design rationale for the heterogeneous model
- [`literature/README.md`](literature/README.md) — literature extraction usage
- **[Wiki](../../wiki)** — project overview, pipeline walkthrough, and model comparison narrative

---

## License

For academic use only. Data sourced from the ODO 2026 database.
