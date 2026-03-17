# ============================================================
# RISKFORGE — Main Pipeline Entry Point
# ============================================================
import sys
import os
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

    print(">>> STEP 1: Preprocessing")
    df, feature_cols = preprocess(dataset, path)

    print("\n>>> STEP 2: Building Transaction Graph")
    G = build_graph(df)
    A_norm, node_list = get_adjacency_matrix(G)

    print("\n>>> STEP 3: Statistical Risk Engine")
    model, threshold = train_model(df, feature_cols)
    stat_scores = score_transactions(model, df, feature_cols, threshold)

    print("\n>>> STEP 4: Structural Risk Engine")
    graph_features = extract_graph_features(G, A_norm=A_norm, node_list=node_list)
    account_labels = get_account_labels(df)
    struct_scores  = score_structural_risk(graph_features, account_labels)

    print("\n>>> STEP 5: Risk Propagation Engine")
    init_risk   = build_initial_risk(stat_scores, struct_scores)
    prop_scores = propagate_risk(init_risk, A_norm, node_list)

    print("\n>>> STEP 6: Aggregating Scores & Ranking")
    final_scores = aggregate_scores(stat_scores, struct_scores, prop_scores)

    # Debug distribution
    print("\n[Debug] Final risk score distribution:")
    print(final_scores["final_risk"].describe())
    print(f"\nHigh   (>=0.9): {(final_scores['final_risk'] >= 0.9).sum():,}")
    print(f"Medium (0.6-0.9): {((final_scores['final_risk'] >= 0.6) & (final_scores['final_risk'] < 0.9)).sum():,}")
    print(f"Low    (<0.6):  {(final_scores['final_risk'] < 0.6).sum():,}")

    display_top_accounts(final_scores)
    save_results(final_scores)

    return final_scores

    # Step 7: Evaluation
    print("\n>>> STEP 7: Ablation Study")
    from src.evaluator import run_ablation
    run_ablation("data/results.csv", df)


if __name__ == "__main__":
    run_pipeline(
        dataset="paysim",
        path=config.PAYSIM_PATH,
    )