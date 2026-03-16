# ============================================================
# Module 6: Risk Aggregation & Ranking
# ============================================================
import pandas as pd
from config import W_STATISTICAL, W_STRUCTURAL, W_PROPAGATION, TOP_N_ACCOUNTS


def aggregate_scores(stat_scores: pd.DataFrame,
                     struct_scores: pd.DataFrame,
                     prop_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the three risk signals using weighted aggregation:

        R_final = w1 * R_stat + w2 * R_struct + w3 * R_prop
    """
    df = (
        stat_scores[["account", "stat_risk"]]
        .merge(struct_scores[["account", "struct_risk"]], on="account", how="outer")
        .merge(prop_scores[["account", "prop_risk"]],    on="account", how="outer")
        .fillna(0)
    )

    df["final_risk"] = (
        W_STATISTICAL  * df["stat_risk"]   +
        W_STRUCTURAL   * df["struct_risk"] +
        W_PROPAGATION  * df["prop_risk"]
    )

    # Normalise final score
    mn, mx = df["final_risk"].min(), df["final_risk"].max()
    df["final_risk"] = (df["final_risk"] - mn) / (mx - mn + 1e-9)

    df = df.sort_values("final_risk", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def display_top_accounts(df: pd.DataFrame, n: int = TOP_N_ACCOUNTS):
    """Print the top N highest-risk accounts with a score breakdown."""
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
    df.to_csv(path, index=False)
    print(f"[Aggregator] Results saved to {path}")
