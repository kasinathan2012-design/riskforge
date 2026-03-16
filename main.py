# ============================================================
# RISKFORGE — Main Pipeline Entry Point
# ============================================================
import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess    import preprocess
from graph_builder import build_graph, get_adjacency_matrix
from stat_engine   import train_model, score_transactions
from struct_engine import extract_graph_features, score_structural_risk, get_account_labels
from propagation   import build_initial_risk, propagate_risk
from aggregator    import aggregate_scores, display_top_accounts, save_results
import config


def run_pipeline(dataset: str, path: str):
    print(f"\n{'='*65}")
    print(f"  RISKFORGE Pipeline — Dataset: {dataset.upper()}")
    print(f"{'='*65}\n")

    # ── Step 1: Preprocess ──────────────────────────────────────
    print(">>> STEP 1: Preprocessing")
    df, feature_cols = preprocess(dataset, path)

    # ── Step 2: Build Transaction Graph ────────────────────────
    print("\n>>> STEP 2: Building Transaction Graph")
    G = build_graph(df)
    A_norm, node_list = get_adjacency_matrix(G)

    # ── Step 3: Statistical Risk (XGBoost) ─────────────────────
    print("\n>>> STEP 3: Statistical Risk Engine")
    model       = train_model(df, feature_cols)
    stat_scores = score_transactions(model, df, feature_cols)

    # ── Step 4: Structural Risk (Graph Features) ───────────────
    print("\n>>> STEP 4: Structural Risk Engine")
    graph_features  = extract_graph_features(G)
    account_labels  = get_account_labels(df)
    struct_scores   = score_structural_risk(graph_features, account_labels)

    # ── Step 5: Risk Propagation ───────────────────────────────
    print("\n>>> STEP 5: Risk Propagation Engine")
    init_risk   = build_initial_risk(stat_scores, struct_scores)
    prop_scores = propagate_risk(init_risk, A_norm, node_list)

    # ── Step 6: Aggregate & Rank ───────────────────────────────
    print("\n>>> STEP 6: Aggregating Scores & Ranking")
    final_scores = aggregate_scores(stat_scores, struct_scores, prop_scores)
    display_top_accounts(final_scores)
    save_results(final_scores)

    return final_scores


if __name__ == "__main__":
    # Default: run on PaySim (richer graph structure)
    # Switch to 'creditcard' and CREDIT_CARD_PATH as needed
    run_pipeline(
        dataset="paysim",
        path=config.PAYSIM_PATH,
    )
