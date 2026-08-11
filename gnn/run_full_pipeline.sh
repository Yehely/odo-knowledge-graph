#!/usr/bin/env bash
# ============================================================
# ODO Bipartite GNN — Full Pipeline Setup & Training
#
# Usage (run from the repo root):
#   chmod +x gnn/run_full_pipeline.sh
#   ./gnn/run_full_pipeline.sh
#
# What this script does:
#   1. Creates the 'odo' conda environment (if not exists)
#   2. Installs all required Python packages
#   3. Builds the bipartite graph from the Excel dataset
#   4. Trains the GNN model
#   5. Prints the final test-set results
#
# Prerequisite:
#   Place "Final ODO Dataset_v2026-06-10.xlsx" in the repo root.
# ============================================================

set -e  # exit immediately on error

CONDA_ENV="odo"
EXCEL_FILE="Final ODO Dataset_v2026-06-10.xlsx"
GRAPH_FILE="gnn/processed_bipartite_graph.pt"

# ── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}[STEP $1]${NC} $2"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 0. Sanity checks ────────────────────────────────────────
step 0 "Checking prerequisites"

command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda first."

if [ ! -f "$EXCEL_FILE" ]; then
    die "Dataset not found: '$EXCEL_FILE'\nPlease copy it into the repo root and re-run this script from there."
fi

echo "  conda  : $(conda --version)"
echo "  dataset: $EXCEL_FILE ✓"

# ── 1. Create conda environment ──────────────────────────────
step 1 "Setting up conda environment '$CONDA_ENV'"

if conda env list | grep -q "^$CONDA_ENV "; then
    warn "Environment '$CONDA_ENV' already exists — skipping creation."
else
    conda create -y -n "$CONDA_ENV" python=3.11
    echo "  Created environment '$CONDA_ENV'."
fi

# ── 2. Install packages ──────────────────────────────────────
step 2 "Installing Python packages"

conda run -n "$CONDA_ENV" pip install --quiet \
    torch \
    torch-geometric \
    rdkit \
    pandas \
    openpyxl \
    scikit-learn \
    scipy \
    matplotlib \
    numpy

echo "  All packages installed."

# ── 3. Build bipartite graph ─────────────────────────────────
step 3 "Building bipartite graph (Excel → $GRAPH_FILE)"

if [ -f "$GRAPH_FILE" ]; then
    warn "$GRAPH_FILE already exists — skipping preprocessing."
    warn "Delete it and re-run if you want to rebuild from scratch."
else
    conda run -n "$CONDA_ENV" python3 gnn/preprocess_bipartite_graph.py
fi

# ── 4. Train GNN ─────────────────────────────────────────────
step 4 "Training the GNN model"

conda run -n "$CONDA_ENV" python3 gnn/train_gnn.py

# ── 5. Print results ─────────────────────────────────────────
step 5 "Results"

if [ -f "gnn/test_results.txt" ]; then
    cat gnn/test_results.txt
else
    warn "gnn/test_results.txt not found."
fi

echo -e "\n${GREEN}Done.${NC} Model checkpoint: gnn/checkpoints/best_model.pt"
