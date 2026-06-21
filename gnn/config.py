"""Central hyperparameter and path configuration for the ODO GNN."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")

TTL_COMPOUNDS   = os.path.join(OUTPUT_DIR, "compounds.ttl")
TTL_ACTIVITIES  = os.path.join(OUTPUT_DIR, "activities.ttl")
TTL_TARGETS     = os.path.join(OUTPUT_DIR, "proteins_targets.ttl")
TTL_ASSAYS      = os.path.join(OUTPUT_DIR, "assays.ttl")
TTL_DOCUMENTS   = os.path.join(OUTPUT_DIR, "documents.ttl")

CHECKPOINT_DIR  = os.path.join(ROOT, "gnn", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
MORGAN_BITS   = 2048   # Morgan fingerprint size (ECFP4, radius=2)
MORGAN_RADIUS = 2

ADMET_COLS = [
    "mw", "alogp", "ro5_violations",
    "qp_logpow", "qp_logs", "qp_donor_hb", "qp_acceptor_hb",
    "qp_logkhsa", "human_oral_absorption", "qp_sasa",
]

ENDPOINT_TYPES  = ["Ki", "IC50", "EC50", "Inhibition", "Activity", "Binding", "other"]
QUALIFIERS      = ["=", "<", ">", "<=", ">=", "other"]
EXP_SETTINGS    = ["in vitro", "in vivo", "other"]

# Minimum pChEMBL and activity filter
MIN_PCHEMBL = 2.0
MAX_PCHEMBL = 15.0
EXACT_QUALIFIER = "="   # only "=" measurements used as training labels

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
COMPOUND_HIDDEN = 256   # projection dimension for compound features
TARGET_HIDDEN   = 256   # projection dimension for target features
GNN_LAYERS      = 2     # 2-hop bipartite SAGEConv (as per report)
DROPOUT         = 0.3
EDGE_HEAD_DIMS  = [512, 256, 128, 1]  # MLP dims (input=256+256+edge_dim, output=1)

# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
# Papers published in 1977-2014 account for ~79.4% of the corpus.
# Papers from 2015 onward form the held-out test set (~20.6%).
# Validation is a random 10% of the training (pre-2015) activities.
TEMPORAL_CUTOFF_YEAR = 2015   # first year of test set (train: year < 2015)
VAL_FRAC_OF_TRAIN    = 0.10   # fraction of pre-cutoff activities used for val

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
SEED          = 42
TRAIN_FRAC    = 0.80   # kept for backward compat; temporal split is used instead
VAL_FRAC      = 0.10
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
MAX_EPOCHS    = 200
PATIENCE      = 20     # early stopping patience
LR_PATIENCE   = 10     # ReduceLROnPlateau patience
BATCH_SIZE    = 512    # edges per mini-batch (used in NeighborLoader)
