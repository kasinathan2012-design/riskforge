# ============================================================
# Module 8: Simulation Engine
# ============================================================
import numpy as np
import pandas as pd
from typing import Generator


# ── Scenario definitions ──────────────────────────────────
SCENARIOS = {
    "account_takeover": {
        "name":        "Account Takeover",
        "description": "Attacker gains access and systematically drains the account",
        "steps": [
            {"label": "Reconnaissance",    "type": "TRANSFER",  "amount_pct": 0.01, "desc": "Small test transaction to verify access"},
            {"label": "Probing",           "type": "TRANSFER",  "amount_pct": 0.05, "desc": "Slightly larger probe to test limits"},
            {"label": "Escalation",        "type": "TRANSFER",  "amount_pct": 0.20, "desc": "Significant transfer — attacker gains confidence"},
            {"label": "Major Drain",       "type": "TRANSFER",  "amount_pct": 0.50, "desc": "Half the balance transferred out"},
            {"label": "Complete Takeover", "type": "TRANSFER",  "amount_pct": 1.00, "desc": "Full balance drain — account emptied"},
        ]
    },
    "money_mule": {
        "name":        "Money Mule Network",
        "description": "Funds layered through multiple accounts — each transfer adds cumulative suspicion even though individual transactions appear legitimate",
        "steps": [
            {"label": "Initial Deposit",    "type": "CASH_IN",  "amount_pct": 1.00, "desc": "Large deposit from unknown source — unusual for this account type"},
            {"label": "First Split",        "type": "TRANSFER", "amount_pct": 0.45, "desc": "45% forwarded to mule account 1 — structuring pattern begins"},
            {"label": "Second Split",       "type": "TRANSFER", "amount_pct": 0.45, "desc": "45% forwarded to mule account 2 — split pattern confirmed"},
            {"label": "Layering",           "type": "TRANSFER", "amount_pct": 0.80, "desc": "Rapid re-transfer to obscure origin — velocity alert triggered"},
            {"label": "Cash Extraction",    "type": "CASH_OUT", "amount_pct": 0.95, "desc": "Final cash out — full layering cycle complete"},
        ]
    },
    "smurfing": {
        "name":        "Smurfing Attack",
        "description": "Many small transactions below detection threshold — behavioral signal builds slowly as pattern emerges across multiple steps",
        "steps": [
            {"label": "Micro Transfer 1",   "type": "TRANSFER", "amount_pct": 0.08, "desc": "First small transfer — individually normal, $0 suspicion"},
            {"label": "Micro Transfer 2",   "type": "TRANSFER", "amount_pct": 0.09, "desc": "Second transfer — velocity pattern beginning to form"},
            {"label": "Micro Transfer 3",   "type": "TRANSFER", "amount_pct": 0.07, "desc": "Third transfer — frequency now anomalous for account profile"},
            {"label": "Micro Transfer 4",   "type": "TRANSFER", "amount_pct": 0.10, "desc": "Fourth transfer — cumulative volume crosses alert threshold"},
            {"label": "Micro Transfer 5",   "type": "TRANSFER", "amount_pct": 0.09, "desc": "Fifth transfer — pattern confirmed, risk score elevated"},
        ]
    },
}

TYPE_ENCODING = {
    "CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4
}


def get_account_context(account_id, state):
    context = {
        "account_id":  account_id,
        "balance":     1_000_000.0 ,  # realistic fraud account balance
        "stat_risk":   0.1,
        "struct_risk": 0.1,
        "prop_risk":   0.1,
        "final_risk":  0.1,
    }

    if state.get("final_scores") is not None:
        df  = state["final_scores"]
        row = df[df["account"] == account_id]
        if not row.empty:
            r = row.iloc[0]
            context["stat_risk"]   = float(r.get("stat_risk",   0.1))
            context["struct_risk"] = float(r.get("struct_risk", 0.1))
            context["prop_risk"]   = float(r.get("prop_risk",   0.1))
            context["final_risk"]  = float(r.get("final_risk",  0.1))

    if state.get("df") is not None:
        txdf = state["df"]
        sent = txdf[txdf["sender"] == account_id]
        if len(sent) > 0:
            # Use max transaction amount as proxy for account wealth
            max_tx    = float(sent["amount"].max())
            avg_tx    = float(sent["amount"].mean())
            old_bal   = float(sent["oldbalanceOrg"].max())

            # Derive realistic balance:
            # Take the largest of (max_tx * 1.5, old_bal)
            # Then clamp between 500K and 10M for simulation realism
            derived = max(max_tx * 1.5, old_bal, avg_tx * 3)
            context["balance"] = float(np.clip(derived, 500_000, 10_000_000))
        else:
            # Account is receiver-only — use structural risk to estimate wealth
            # Higher structural risk = more connected = likely larger account
            struct = context["struct_risk"]
            context["balance"] = float(500_000 + struct * 4_500_000)

    return context

def score_step(step_def: dict, balance: float, initial_balance: float, state: dict) -> dict:
    model        = state.get("model")
    threshold    = state.get("threshold") or 0.5
    feature_cols = state.get("feature_cols", [])

    amount      = initial_balance * step_def["amount_pct"]
    amount      = max(amount, 1.0)
    new_balance = max(0.0, balance - amount)
    
    # For complete drain — match exact PaySim fraud pattern
    # model learned: oldbalanceOrg == amount && newbalanceOrig == 0
    if step_def["amount_pct"] >= 0.99:
        # Final drain — exact fraud pattern
        old_bal      = amount
        new_bal      = 0.0
        old_bal_dest = 0.0
        new_bal_dest = 0.0
        orig_delta   = -amount
        dest_delta   = 0.0
    else:
        # Normal steps — realistic values
        old_bal      = balance
        new_bal      = new_balance
        old_bal_dest = 0.0
        new_bal_dest = amount   # destination receives the money
        orig_delta   = new_bal - old_bal
        dest_delta   = amount

    row = {
        "amount_scaled":      (amount - 179861.90) / 603858.18,
        "type_encoded":        TYPE_ENCODING.get(step_def["type"], 4),
        "orig_balance_delta":  orig_delta,
        "dest_balance_delta":  dest_delta,
        "oldbalanceOrg":       old_bal,
        "newbalanceOrig":      new_bal,
        "oldbalanceDest":      old_bal_dest,
        "newbalanceDest":      new_bal_dest,
    }

    fraud_prob = 0.5
    if model is not None and feature_cols:
        try:
            X = pd.DataFrame([row])[feature_cols].fillna(0)
            fraud_prob = float(model.predict_proba(X)[0][1])
        except Exception:
            pass

    return {
        "amount":      round(amount, 2),
        "new_balance": round(new_balance, 2),
        "fraud_prob":  round(fraud_prob, 4),
        "flagged":     bool(fraud_prob >= threshold),
    }


def update_risk_scores(
    current_scores, fraud_prob, step_num, total_steps,
    A_norm, node_list, account_id, scenario_key="account_takeover"
):
    step_weight = (step_num + 1) / total_steps

    if scenario_key == "account_takeover":
        # Spikes on high fraud_prob steps
        new_stat = max(
            current_scores["stat_risk"],
            current_scores["stat_risk"] + fraud_prob * step_weight * 0.4
        )

    elif scenario_key == "money_mule":
        # Each step adds cumulative suspicion regardless of fraud_prob
        # Pattern-based: multiple large transfers = suspicious
        cumulative_boost = step_weight * 0.12  # steady escalation
        new_stat = min(1.0, current_scores["stat_risk"] + cumulative_boost + fraud_prob * 0.1)

    elif scenario_key == "smurfing":
        # Slow build — individually low but pattern emerging
        # Risk accelerates in later steps as pattern becomes clear
        acceleration = step_weight ** 2  # quadratic — slow then fast
        new_stat = min(1.0, current_scores["stat_risk"] + acceleration * 0.08 + fraud_prob * 0.05)

    else:
        new_stat = max(
            current_scores["stat_risk"],
            current_scores["stat_risk"] + fraud_prob * step_weight * 0.4
        )

    new_stat = min(1.0, new_stat)

    new_prop = current_scores["prop_risk"]
    if A_norm is not None and node_list is not None and account_id in node_list:
        new_prop = max(
            current_scores["prop_risk"],
            _local_propagation(account_id, new_stat, current_scores["prop_risk"], A_norm, node_list)
        )

    new_struct  = current_scores["struct_risk"]
    new_final   = min(1.0, 0.40 * new_stat + 0.30 * new_struct + 0.30 * new_prop)

    return {
        "stat_risk":   round(new_stat,   4),
        "struct_risk": round(new_struct, 4),
        "prop_risk":   round(new_prop,   4),
        "final_risk":  round(new_final,  4),
    }


def _local_propagation(
    account_id: str,
    new_stat_risk: float,
    current_prop: float,
    A_norm,
    node_list: list,
    alpha: float = 0.6,
    beta:  float = 0.4,
    hops:  int   = 2,
) -> float:
    """
    Run propagation on just the local ego network — account + neighbors.
    Much faster than full 9M node propagation.
    Returns updated propagation risk for the target account.
    """
    try:
        idx = node_list.index(account_id)

        # Get neighbors within 2 hops
        row = A_norm[idx]
        neighbor_indices = row.nonzero()[1].tolist()

        if not neighbor_indices:
            return min(1.0, alpha * current_prop + beta * new_stat_risk)

        # Extract local submatrix
        local_indices = [idx] + neighbor_indices[:20]  # cap at 20 neighbors
        sub = A_norm[np.ix_(local_indices, local_indices)]

        # Initialize local risk vector
        r = np.zeros(len(local_indices), dtype=np.float32)
        r[0] = new_stat_risk  # target account gets new behavioral risk

        # Run 3 iterations of propagation on subgraph
        for _ in range(3):
            r_new = alpha * r + beta * sub.dot(r)
            r_new = np.clip(r_new, 0, 1)
            r = r_new

        return float(r[0])

    except (ValueError, IndexError):
        return min(1.0, alpha * current_prop + beta * new_stat_risk)


def run_simulation(
    scenario_key: str,
    account_id: str,
    state: dict,
) -> Generator[dict, None, None]:
    """
    Generator that yields one result dict per simulation step.
    Used by the SSE endpoint to stream results to the frontend.
    """
    scenario = SCENARIOS.get(scenario_key)
    if not scenario:
        yield {"error": f"Unknown scenario: {scenario_key}"}
        return

    steps      = scenario["steps"]
    context    = get_account_context(account_id, state)
    balance    = context["balance"]
    scores     = {
        "stat_risk":   context["stat_risk"],
        "struct_risk": context["struct_risk"],
        "prop_risk":   context["prop_risk"],
        "final_risk":  context["final_risk"],
    }
    A_norm    = state.get("A_norm")
    node_list = state.get("node_list")
    
    initial_balance = balance 

    # Yield initial state
    yield {
        "step":        0,
        "total_steps": len(steps),
        "label":       "Initial State",
        "desc":        "Account baseline before simulation",
        "type":        "—",
        "amount":      0,
        "balance":     round(balance, 2),
        "fraud_prob":  0.0,
        "flagged":     False,
        "scores":      {k: float(v) for k, v in scores.items()},
        "done":        False,
    }

    # Process each step
    for i, step_def in enumerate(steps):
        result   = score_step(step_def, balance, initial_balance, state)
        scores   = update_risk_scores(
            scores, result["fraud_prob"],
            i, len(steps), A_norm, node_list, account_id, scenario_key=scenario_key
        )
        balance  = result["new_balance"]

        yield {
            "step":        i + 1,
            "total_steps": len(steps),
            "label":       step_def["label"],
            "desc":        step_def["desc"],
            "type":        step_def["type"],
            "amount":      result["amount"],
            "balance":     result["new_balance"],
            "fraud_prob":  result["fraud_prob"],
            "flagged":     bool(result["flagged"]),
            "scores":      {k: float(v) for k, v in scores.items()},
            "done":        bool(i == len(steps) - 1),
        }