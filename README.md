# ODO Knowledge Graph — Opioid Drug-Receptor Interactions

A knowledge graph and GNN prediction pipeline built from the ODO 2026 database,
containing **37,362 biological activity measurements** across **13,396 chemical
compounds** tested against opioid receptors (MOR, DOR, KOR, NOP) — the counts
produced by the ChEMBL-ID identity rule (`kg/build_kg.py` / `gnn/preprocess_bipartite_graph.py`),
the pipeline that builds the actual RDF knowledge graph. The heterogeneous
GNN pipeline (`hetero_gnn/preprocess.py`) uses a different identity rule
(PubChem CID / UniProt accession) and gets 37,353 activities / 13,297
compounds from the same source file — see
[`analysis/target_inventory.md`](analysis/target_inventory.md) for the full
reconciliation.

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
chmod +x gnn/run_full_pipeline.sh
./gnn/run_full_pipeline.sh
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
Note: this is not a forward-in-time split — per `gnn/preprocess_bipartite_graph.py`'s
`make_temporal_splits` (`TEMPORAL_CUTOFF=2016`, `TEST_MAX_YEAR=2020`), only
`doc_year` in `[2016, 2020]` goes to test; activities published after 2020
(and any with an unknown year) fall back into the train/val pool rather than
being excluded or held out.

```bash
conda run -n odo python3 gnn/train_gnn.py --split temporal --fp-bits 2048
```

**Expected results:** RMSE=1.26 | MAE=0.98 | Pearson r=0.53 | R²=0.19

---

#### Model 2 — Compound-Level Random Split (2048-bit FP)

20% of **compounds** held out for test — no compound appears in both train and test.
The most honest evaluation of generalisation to novel chemistry.

```bash
conda run -n odo python3 gnn/train_gnn.py --split compound_random --fp-bits 2048
```

**Expected results:** RMSE=0.93 | MAE=0.69 | Pearson r=0.77 | R²=0.55

---

#### Model 3 — Fine-Tuned (Temporal, 512-bit FP, stronger regularisation)

Reduced fingerprint + higher dropout (0.4) + higher weight decay (1e-4) + temporal sample weighting.

```bash
conda run -n odo python3 gnn/train_gnn.py --split temporal --fp-bits 512
```

**Expected results:** RMSE=1.26 | MAE=0.99 | Pearson r=0.51 | R²=0.19

---

#### Model 4 — Ensemble (5 models, temporal split)

Trains 5 independent models with different random seeds and averages their predictions.

```bash
conda run -n odo python3 gnn/train_ensemble.py --split temporal --fp-bits 2048
```

**Expected results:** RMSE=1.23 | MAE=0.98 | Pearson r=0.53 | R²=0.22

---

#### Model 5 — Diagnostic: Edge-Level Random Split

Randomly splits individual **experiments** (not compounds). Inflated results due to data leakage — same compound can appear in both train and test. For diagnostic/comparison only.

```bash
conda run -n odo python3 gnn/train_gnn.py --split random --fp-bits 2048
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
with 2-hop hub-and-spoke message passing through Assay nodes, using a
selectable pooling mode (attention-weighted, the default, or plain mean —
see Architecture below). Full spec in [`docs/HETEROGENEOUS_GNN_ARCHITECTURE.md`](docs/HETEROGENEOUS_GNN_ARCHITECTURE.md);
implementation notes and rationale in [`hetero_gnn/README.md`](hetero_gnn/README.md).

```bash
# Step 1 — Excel → hetero_gnn/processed_hetero_graph.pt
conda run -n odo python3 hetero_gnn/run_preprocess.py

# Step 2 — train (--seed and --out default to a seed/pooling-mode-tagged
# results file, e.g. hetero_gnn/test_results_seed42_mean.txt, so different
# runs no longer overwrite each other). --pooling-mode mean matches the
# reported result below; the current config.py default is "attention",
# which has no verified surviving result (see the note at the top of
# reports/generate_final_report.py) — omitting --pooling-mode here would
# train a different, unvalidated configuration.
conda run -n odo python3 hetero_gnn/run_train.py --seed 42 --pooling-mode mean

# Optional — hyperparameter search (Optuna), then trains the winning config
conda run -n odo python3 hetero_gnn/search_hparams.py --n-trials 20

# Inference on a new compound (after training)
conda run -n odo python3 hetero_gnn/predict.py
```

**Expected results (temporal split, train < 2015 / test ≥ 2015, mean-aggregation pooling):** RMSE=1.23 | MAE=0.98 | Pearson r=0.54 | R²=0.26

This number was originally produced by a run with no recorded seed (see
`hetero_gnn/test_results.txt`, still on disk, not reproducible bit-for-bit) —
the pooling mode above is inferred from how the report itself labeled this
result ("mean-aggregation baseline"), not from a recorded config value, since
no run before this change recorded its config at all. The command above now
records its own seed, pooling mode, encoder dropout, and commit hash in the
results file for every future run, so this won't happen again — it does not
guarantee the historical 0.26 itself replays exactly.

---

### Graph-free baselines (`baselines/`)

Global-mean, per-target-mean, and fingerprint-only (ECFP4 + target one-hot,
no graph structure) RandomForest baselines, scored on the same saved
temporal-split graphs the two models above use — the point of comparison
for "is the graph actually helping."

```bash
conda run -n odo python3 baselines/run_baselines.py
```

One run produces results for **both** splits (labelled separately in
`baselines/summary.md` — they use different temporal partitions and are
not comparable to each other):

| Split | Baseline | R² |
|---|---|---|
| hetero (train<2015/test≥2015) | Global mean | ≈-0.00 |
| hetero | Per-target mean | 0.01 |
| hetero | Fingerprint-only RF | **0.230 ± 0.005** (5 seeds) |
| bipartite (train≤2015/test 2016-2020) | Global mean | ≈-0.01 |
| bipartite | Per-target mean | 0.06 |
| bipartite | Fingerprint-only RF | 0.180 ± 0.002 (5 seeds) |

The hetero-split fingerprint-only RF (0.230) is within a few hundredths of
R² of the heterogeneous GNN itself (0.26) on the same test set — see
`analysis/target_inventory.md` and `baselines/summary.md` for the full
read.

---

## Knowledge Graph Pipeline (GraphDB)

Requires GraphDB running on `localhost:7200`.

```bash
# Step 1 — Excel → RDF Turtle files
conda run -n odo python3 kg/build_kg.py

# Step 2 — Load into GraphDB
conda run -n odo python3 kg/setup_graphdb.py

# Step 3 — Validate (16 SPARQL checks)
conda run -n odo python3 kg/validate_kg.py
```

---

## Smart Retrieval (optional)

`smart_retrieval/` fetches and cleans the full-text source papers behind the
dataset's `pubmed_id` column (via NCBI Entrez/PMC), not part of the
KG/GNN pipeline itself. Two extraction tools build on top of it:
`smart_retrieval/chembl_extractor/`, which auto-generates new ODO-schema
database rows for a given ChEMBL compound from 6 external data sources
plus trained ML models (CLI + local web UI), and
`smart_retrieval/llama_extractor/`, which runs an LLM directly over paper text
to extract drug-target experiments. See
[`smart_retrieval/README.md`](smart_retrieval/README.md).

---

## Repository Structure

```
odo-project/
├── kg/                             # Knowledge-graph ETL pipeline
│   ├── build_kg.py                #   ETL: Excel → RDF Turtle
│   ├── setup_graphdb.py           #   Load RDF into GraphDB
│   └── validate_kg.py             #   16 SPARQL validation queries
│   odo_ontology.ttl               #   OWL ontology
├── gnn/                            # Bipartite (2-node-type) GraphSAGE model
│   ├── model.py                   #   OpioidGNN architecture
│   ├── train.py                   #   Training loop + asymmetric loss
│   ├── dataset.py                 #   Load .pt graph for training
│   ├── config.py                  #   Hyperparameters
│   ├── features.py                #   Feature engineering utilities
│   ├── predict.py                 #   Inference on new SMILES
│   ├── preprocess_bipartite_graph.py  # Excel → processed_<split>_<fp>fp.pt
│   ├── train_gnn.py               #   Train a single bipartite GNN model
│   ├── train_ensemble.py          #   Train 5-model bipartite ensemble
│   ├── run_full_pipeline.sh       #   One-command setup and training
│   └── requirements.txt
├── hetero_gnn/                    # Heterogeneous (5-node-type) GNN model
│   ├── model.py / config.py / dataset.py / preprocess.py
│   ├── train.py / run_train.py / run_preprocess.py / search_hparams.py
│   └── predict.py
├── smart_retrieval/                    # PMC full-text acquisition + cleaning
│   ├── GetFiles.py / CleanFiles.py
│   ├── chembl_extractor/          # ChEMBL API + ML based row generator
│   ├── llama_extractor/           # LLM-based experiment extraction
│   └── README.md
└── docs/                          # Technical specs and setup guide
    ├── ARCHITECTURE_SKILLS.md          # Bipartite GNN architecture spec
    ├── HETEROGENEOUS_GNN_ARCHITECTURE.md  # Heterogeneous GNN architecture spec
    └── SETUP_GUIDE.md                  # Full project install/setup walkthrough
```

All commands below assume you're running from the repo root — every
script resolves the dataset and its own outputs relative to that.

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
- 2-hop hub-and-spoke message passing through Assay. Pooling mode is a
  selectable flag (`POOLING_MODE` in `hetero_gnn/config.py`, or
  `--pooling-mode` on the CLI): **"attention" (the default)** — a learned
  Linear→Tanh→Linear per-hop scoring head — or **"mean"**, a plain average
  over each hop's neighbours. The mean variant is the one that reproduces
  the earlier reported result (R²=0.26 — see the Heterogeneous GNN section
  above). The two modes have not been compared under equal
  hyperparameter-search budgets: the attention variant's best observed
  figure (R²=0.31) came from `hetero_gnn/search_hparams.py`'s Optuna search
  rather than a single `run_train.py` run under the same conditions as the
  0.26 figure, so it is not a like-for-like comparison and is not reported
  as a result here. Example:
  `conda run -n odo python3 hetero_gnn/run_train.py --pooling-mode attention --seed 42`
- Edge MLP head: ~530 → 256 → 128 → 1 (pKi regression)
- Loss: combined exact-MSE + censored-hinge (handles `<`/`>` qualified measurements)

Full spec: [`docs/HETEROGENEOUS_GNN_ARCHITECTURE.md`](docs/HETEROGENEOUS_GNN_ARCHITECTURE.md).

---

## Documentation

- [`docs/ARCHITECTURE_SKILLS.md`](docs/ARCHITECTURE_SKILLS.md) — bipartite GNN architecture spec
- [`docs/HETEROGENEOUS_GNN_ARCHITECTURE.md`](docs/HETEROGENEOUS_GNN_ARCHITECTURE.md) — heterogeneous GNN architecture spec
- [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) — full project install/setup walkthrough
- [`hetero_gnn/README.md`](hetero_gnn/README.md) — implementation notes and design rationale for the heterogeneous model
- [`smart_retrieval/README.md`](smart_retrieval/README.md) — literature extraction usage
- **[Wiki](../../wiki)** — project overview, pipeline walkthrough, and model comparison narrative

---

## License

For academic use only. Data sourced from the ODO 2026 database.
