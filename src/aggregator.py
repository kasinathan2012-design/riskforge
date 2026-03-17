# ============================================================
# Module 6: Risk Aggregation & Ranking
# ============================================================
import os
import csv
import pandas as pd
import numpy as np
from config import W_STATISTICAL, W_STRUCTURAL, W_PROPAGATION, TOP_N_ACCOUNTS


def aggregate_scores(stat_scores: pd.DataFrame,
                     struct_scores: pd.DataFrame,
                     prop_scores: pd.DataFrame) -> pd.DataFrame:
    df = (
        stat_scores[["account", "stat_risk"]]
        .merge(struct_scores[["account", "struct_risk"]], on="account", how="outer")
        .merge(prop_scores[["account", "prop_risk"]],    on="account", how="outer")
        .fillna(0)
    )

    df["raw_score"] = (
        W_STATISTICAL  * df["stat_risk"]   +
        W_STRUCTURAL   * df["struct_risk"] +
        W_PROPAGATION  * df["prop_risk"]
    )

    # Blend 70% raw score + 30% rank percentile
    # Raw score preserves the actual distribution so most accounts
    # stay low risk — rank percentile just breaks ties
    rank_pct = df["raw_score"].rank(method="average", pct=True)
    df["final_risk"] = 0.7 * df["raw_score"] + 0.3 * rank_pct

    # Normalize to [0, 1]
    mn, mx = df["final_risk"].min(), df["final_risk"].max()
    df["final_risk"] = (df["final_risk"] - mn) / (mx - mn + 1e-9)

    # Individual signals — rank for display differentiation
    df["stat_risk"]   = df["stat_risk"].rank(method="average",   pct=True)
    df["struct_risk"] = df["struct_risk"].rank(method="average", pct=True)
    df["prop_risk"]   = df["prop_risk"].rank(method="average",   pct=True)

    df = df.sort_values("final_risk", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df

def display_top_accounts(df: pd.DataFrame, n: int = TOP_N_ACCOUNTS):
    print(f"\n{'='*65}")
    print(f"  RISKFORGE — Top {n} High-Risk Accounts")
    print(f"{'='*65}")
    print(f"{'Rank':<6} {'Account':<20} {'Final':>7} {'Stat':>7} "
          f"{'Struct':>8} {'Prop':>7}")
    print(f"{'-'*65}")
    for _, row in df.head(n).iterrows():
        print(f"{int(row['rank']):<6} {str(row['account']):<20} "
              f"{row['final_risk']:>7.4f} {row['stat_risk']:>7.4f} "
              f"{row['struct_risk']:>8.4f} {row['prop_risk']:>7.4f}")
    print(f"{'='*65}\n")


def save_results(df: pd.DataFrame, path: str = "data/results.csv"):
    """Write row by row using csv module — near-zero memory overhead."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(df.columns.tolist())
        for row in df.itertuples(index=False):
            writer.writerow(row)
    print(f"[Aggregator] Results saved to {path}")