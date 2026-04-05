"""
AML GNN Pipeline — Central Config
All hyperparameters and constants in one place.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "aml_dataset")
GRAPH_DIR = os.path.join(BASE_DIR, "graph_cache")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for _d in [GRAPH_DIR, MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Pattern → Class mapping ──────────────────────────────────────────────────
PATTERN_TO_CLASS = {
    # Benign → class 0
    "benign_salary":          0,
    "benign_merchant":        0,
    "benign_family":          0,
    "benign_atm":             0,
    "benign_bills":           0,
    # Suspicious → classes 1-9
    "mule_ring":              1,
    "chain_layering":         2,
    "circular_loop":          3,
    "structuring_fanout":     4,
    "funnel_fanin":           5,
    "diamond_fragmentation":  6,
    "cross_channel_velocity": 7,
    "pan_nesting":            8,
    "burst_velocity_ring":    9,
    # Hybrid → class 10
    "hybrid_fanout_chain_atm":       10,
    "hybrid_ring_loop_funnel":       10,
    "hybrid_nesting_chain_wallet":   10,
    "hybrid_struct_fanin_cross":     10,
    "hybrid_diamond_ring_burst":     10,
    "hybrid_loop_nesting_chain":     10,
}

CLASS_NAMES = [
    "benign",
    "mule_ring",
    "chain_layering",
    "circular_loop",
    "structuring_fanout",
    "funnel_fanin",
    "diamond_fragmentation",
    "cross_channel_velocity",
    "pan_nesting",
    "burst_velocity_ring",
    "hybrid",
]

NUM_CLASSES = len(CLASS_NAMES)   # 11

# ─── Graph / Model ────────────────────────────────────────────────────────────
HIDDEN_DIM       = 128
DROPOUT          = 0.3
NUM_LAYERS       = 2

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE            = 1024
NUM_NEIGHBORS         = [25, 10]   # [1st-hop, 2nd-hop]
EPOCHS                = 50
EARLY_STOP_PATIENCE   = 7
LEARNING_RATE         = 1e-3
WEIGHT_DECAY          = 5e-4

# ─── OOD (novel pattern) detection ───────────────────────────────────────────
OOD_PERCENTILE      = 95   # 95th-pct of training distances = threshold
OOD_MIN_RISK_SCORE  = 0.50 # only flag OOD if node is also suspicious

# ─── Cluster output ───────────────────────────────────────────────────────────
CLUSTER_RISK_THRESHOLD = 0.50   # flag cluster if mean node risk > this

# ─── Valid transaction types (must match generate_aml_dataset.py) ─────────────
TRANSACTION_TYPES = ["UPI", "ATM", "App", "Wallet", "Web"]

# ─── Production pipeline ──────────────────────────────────────────────────────
PRODUCTION_BATCH_WINDOW_MS = 500      # accumulate suspicious txns over 500ms
MIN_RING_SIZE              = 3        # skip rings with fewer accounts
MAX_INFER_NODES            = 5_000    # safety cap on nodes per inference batch
TXN_STORE_MAX_TXN          = 2_000_000  # max transactions in memory store

# ─── Node feature columns (must match build_graph.py + inference.py) ──────────
FEATURE_COLS = [
    "sent_count", "recv_count", "total_degree",
    "out_ratio",  "in_ratio",
    "sent_avg_amt", "sent_max_amt", "sent_std_amt",
    "recv_avg_amt", "recv_max_amt", "recv_std_amt",
    "cross_channel_score",
    "atm_ratio", "wallet_ratio",
    "unique_pincodes",
    "burst_velocity",
    "amount_range",
    "community_id_norm",   # added after Louvain
]
