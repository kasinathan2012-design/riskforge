# ============================================================
# RISKFORGE — Central Configuration
# ============================================================

# --- File Paths ---
CREDIT_CARD_PATH = "data/raw/creditcard.csv"
PAYSIM_PATH      = "data/raw/paysim.csv"

# --- Risk Aggregation Weights ---
# Must sum to 1.0
W_STATISTICAL  = 0.40   # behavioral (XGBoost) signal weight
W_STRUCTURAL   = 0.30   # graph structure signal weight
W_PROPAGATION  = 0.30   # diffusion propagation signal weight

# --- Risk Propagation ---
ALPHA = 0.6   # self-risk retention factor
BETA  = 0.4   # neighbor-risk absorption factor
PROPAGATION_ITERATIONS = 10

# --- XGBoost ---
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 10,   # handles class imbalance
    "eval_metric": "auc",
    "random_state": 42,
}

# --- Output ---
TOP_N_ACCOUNTS = 50   # how many high-risk accounts to display
