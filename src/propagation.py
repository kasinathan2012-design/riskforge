# ============================================================
# Module 5: Risk Propagation Engine
# ============================================================
import numpy as np
import pandas as pd
from config import ALPHA, BETA, PROPAGATION_ITERATIONS


def propagate_risk(initial_scores: pd.DataFrame,
                   A_norm,
                   node_list: list) -> pd.DataFrame:
    """
    R(k+1) = α * R(k) + β * A * R(k)
    .dot() works for both scipy sparse and dense numpy arrays.
    """
    score_map = dict(zip(initial_scores["account"], initial_scores["init_risk"]))
    R = np.array([score_map.get(node, 0.0) for node in node_list], dtype=np.float32)

    print(f"[Propagation] Running {PROPAGATION_ITERATIONS} iterations (α={ALPHA}, β={BETA})...")

    for i in range(PROPAGATION_ITERATIONS):
        R_new = ALPHA * R + BETA * A_norm.dot(R)
        R_new = np.clip(R_new, 0, 1)

        delta = np.linalg.norm(R_new - R)
        R = R_new
        if delta < 1e-6:
            print(f"[Propagation] Converged at iteration {i+1}")
            break

    result = pd.DataFrame({"account": node_list, "prop_risk": R})

    mn, mx = result["prop_risk"].min(), result["prop_risk"].max()
    result["prop_risk"] = (result["prop_risk"] - mn) / (mx - mn + 1e-9)

    print(f"[Propagation] Done. Mean propagated risk: {result['prop_risk'].mean():.4f}")
    return result


def build_initial_risk(stat_scores: pd.DataFrame,
                        struct_scores: pd.DataFrame) -> pd.DataFrame:
    merged = stat_scores[["account", "stat_risk"]].merge(
        struct_scores[["account", "struct_risk"]], on="account", how="outer"
    ).fillna(0)

    merged["init_risk"] = 0.5 * merged["stat_risk"] + 0.5 * merged["struct_risk"]
    return merged[["account", "init_risk"]]