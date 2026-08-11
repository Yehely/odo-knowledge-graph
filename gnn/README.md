# Bipartite GNN — `gnn/`

**OpioidGNN**: a 2-node-type bipartite GraphSAGE model predicting pChEMBL
binding affinity from `Compound` ↔ `Target` edges (one edge per bioassay
experiment). The earlier of the two GNN packages in this repo — see
[`hetero_gnn/`](../hetero_gnn/README.md) for the newer 5-node-type model.

Full architecture spec: [`docs/ARCHITECTURE_SKILLS.md`](../docs/ARCHITECTURE_SKILLS.md).

## Modules

| File | Purpose |
|---|---|
| `model.py` | `OpioidGNN` architecture — encoders, 2-hop `HeteroConv`/`SAGEConv` message passing, edge MLP head |
| `train.py` | Training loop + asymmetric MSE loss (penalizes under-predicting high-affinity binders) |
| `dataset.py` | Loads the preprocessed `.pt` graph for training |
| `config.py` | Hyperparameters |
| `features.py` | Compound/target/edge feature engineering |
| `predict.py` | Inference on new SMILES |
| `validate_model.py` | Post-training validation checks |
| `generate_model_report.py` | HTML training-run report |

## Usage

See the main [README](../README.md#gnn-models--how-to-run-each) for the
full list of runnable model variants (temporal split, compound-random
split, fine-tuned, ensemble, diagnostic) with commands and expected
results. Graph preprocessing (`../preprocess_bipartite_graph.py`) runs
automatically the first time a training command needs a graph file that
doesn't exist yet.

## Output

`checkpoints/`, `test_results.txt`, `ensemble_test_results.txt`, and the
processed `.pt` graph files are all gitignored — regenerate by re-running
preprocessing/training.
