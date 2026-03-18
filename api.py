# ============================================================
# RISKFORGE — FastAPI Backend
# ============================================================
import sys, os, gc, traceback, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sklearn.preprocessing import StandardScaler
    
import numpy as np
import pandas as pd
import numpy as np
import threading
import joblib 

from preprocess    import preprocess
from graph_builder import build_graph, get_adjacency_matrix
from stat_engine   import train_model, score_transactions
from struct_engine import extract_graph_features, score_structural_risk, get_account_labels
from propagation   import build_initial_risk, propagate_risk
from aggregator    import aggregate_scores
import config

app = FastAPI(title="RISKFORGE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────
state = {
    "status":        "idle",
    "error":         None,
    "df":            None,
    "A_norm":        None,
    "node_list":     None,
    "model":         None,
    "threshold":     None,
    "feature_cols":  None,
    "stat_scores":   None,
    "struct_scores": None,
    "prop_scores":   None,
    "final_scores":  None,
}

# ── Auto-load saved results on startup ───────────────────────
def load_results_from_disk():
    path = "data/results.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            state["final_scores"] = df
            state["status"] = "ready"
            print(f"[API] Auto-loaded {len(df):,} accounts from {path}")
        except Exception as e:
            print(f"[API] Could not load saved results: {e}")
    else:
        print("[API] No saved results found — run main.py first")

    # Load raw PaySim for account detail lookups
    raw_path = config.PAYSIM_PATH
    if os.path.exists(raw_path):
        try:
            print("[API] Loading raw PaySim data for account lookups...")
            df_raw = pd.read_csv(raw_path)
            df_raw = df_raw.rename(columns={
                "nameOrig": "sender",
                "nameDest": "receiver",
                "isFraud":  "label",
            })
            df_raw["type_encoded"] = pd.Categorical(df_raw["type"]).codes
            state["df"] = df_raw
            print(f"[API] Loaded {len(df_raw):,} transactions for account lookups")
        except Exception as e:
            print(f"[API] Could not load raw data: {e}")

load_results_from_disk()

# Train injection model in background so startup isn't blocked
def load_model_on_startup():
    model_path     = "data/injection_model.pkl"
    threshold_path = "data/injection_threshold.txt"
    feature_path   = "data/injection_features.txt"

    # Try loading from disk first
    if os.path.exists(model_path):
        try:
            state["model"]        = joblib.load(model_path)
            state["threshold"]    = float(open(threshold_path).read())
            state["feature_cols"] = open(feature_path).read().split(",")
            print("[API] Injection model loaded from disk instantly.")
            return  # ← exit here, no retraining needed
        except Exception as e:
            print(f"[API] Could not load saved model: {e} — retraining...")

    # Train and save
    try:
        print("[API] Training injection model on startup...")
        from stat_engine import train_model
        from preprocess  import preprocess

        if os.path.exists(config.PAYSIM_PATH):
            df, feature_cols = preprocess("paysim", config.PAYSIM_PATH)
            model, threshold = train_model(df, feature_cols)
            state["model"]        = model
            state["threshold"]    = threshold
            state["feature_cols"] = feature_cols

            # Save to disk for next restart
            joblib.dump(model, model_path)
            open(threshold_path, "w").write(str(threshold))
            open(feature_path,   "w").write(",".join(feature_cols))
            print("[API] Injection model trained and saved to disk.")
        else:
            print("[API] No dataset found — injection scoring unavailable")
    except Exception as e:
        print(f"[API] Could not load injection model: {traceback.format_exc()}")

threading.Thread(target=load_model_on_startup, daemon=True).start()

# ── Request models ────────────────────────────────────────────
class PipelineRequest(BaseModel):
    dataset: str = "paysim"

class InjectRequest(BaseModel):
    sender:   str
    receiver: str
    amount:   float
    tx_type:  str = "TRANSFER"


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    return {
        "status":          state["status"],
        "error":           state["error"],
        "accounts_scored": len(state["final_scores"]) if state["final_scores"] is not None else 0,
        "model_loaded":    state["model"] is not None,
    }


@app.post("/api/reload")
def reload_results():
    path = "data/results.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No results.csv found — run: python main.py first")
    try:
        df = pd.read_csv(path)
        state["final_scores"] = df
        state["status"] = "ready"
        return {"message": f"Reloaded {len(df):,} accounts from disk", "accounts": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results")
def get_results(limit: int = 500, min_risk: float = 0.0):
    if state["final_scores"] is None:
        raise HTTPException(status_code=404, detail="No results yet — run main.py first")
    df = state["final_scores"]
    high   = df[df["final_risk"] >= 0.9].head(200)
    medium = df[(df["final_risk"] >= 0.6) & (df["final_risk"] < 0.9)].head(150)
    low    = df[df["final_risk"] < 0.6].head(150)
    result = pd.concat([high, medium, low]).sort_values("final_risk", ascending=False)
    return {
        "total_accounts": len(df),
        "returned":       len(result),
        "accounts":       result.to_dict(orient="records"),
    }

@app.get("/api/benchmarks")
def get_benchmarks():
    path = "data/benchmarks.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

@app.post("/api/benchmarks")
def save_benchmarks(papers: list[Any]):
    path = "data/benchmarks.json"
    with open(path, "w") as f:
        json.dump(papers, f, indent=2)
    return {"saved": len(papers)}

@app.get("/api/account/{account_id}")
def get_account(account_id: str):
    if state["final_scores"] is None:
        raise HTTPException(status_code=404, detail="No results yet")

    df  = state["final_scores"]
    row = df[df["account"] == account_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    record      = row.iloc[0].to_dict()
    tx_history  = []
    account_stats = {}

    if state["df"] is not None:
        txdf     = state["df"]
        sent     = txdf[txdf["sender"]   == account_id]
        received = txdf[txdf["receiver"] == account_id]
        all_txs  = pd.concat([sent, received]).sort_values("step").head(20)

        tx_history = all_txs[[
            "sender", "receiver", "amount", "label",
            "type", "step", "oldbalanceOrg", "newbalanceOrig"
        ]].to_dict(orient="records")

        account_stats = {
            "total_sent":        int(len(sent)),
            "is_flagged_by_system": int(sent["isFlaggedFraud"].sum()) if "isFlaggedFraud" in sent.columns and len(sent) > 0 else 0,
            "total_volume":      float(sent["amount"].sum() + received["amount"].sum()),
            "avg_tx_amount":     float(sent["amount"].mean()) if len(sent) > 0 else 0.0,
            "max_tx_amount":     float(sent["amount"].max())  if len(sent) > 0 else 0.0,
            "fraud_tx_count":    int(sent["label"].sum()),
            "first_active_step": int(all_txs["step"].min()) if len(all_txs) > 0 else 0,
            "last_active_step":  int(all_txs["step"].max()) if len(all_txs) > 0 else 0,
            "tx_types":          sent["type"].value_counts().to_dict() if len(sent) > 0 else {},
            "balance_before":    float(sent["oldbalanceOrg"].iloc[0])   if len(sent) > 0 else 0.0,
            "balance_after":     float(sent["newbalanceOrig"].iloc[-1]) if len(sent) > 0 else 0.0,
        }

    return {
        "account":      record,
        "transactions": tx_history,
        "stats":        account_stats,
    }

@app.get("/api/simulate")
def simulate(scenario: str = "account_takeover", account_id: str = "C1548769886"):
    """
    Stream simulation results step by step using Server-Sent Events.
    Frontend receives each step as it's computed and animates live.
    """
    import json
    import time
    from src.simulator import run_simulation, SCENARIOS

    if scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")

    def event_stream():
        for step_result in run_simulation(scenario, account_id, state):
            data = json.dumps(step_result)
            yield f"data: {data}\n\n"
            time.sleep(1.2)  # pause between steps for animation

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/api/simulate/scenarios")
def get_scenarios():
    """Return available scenarios for the frontend dropdown."""
    from src.simulator import SCENARIOS
    return [
        {
            "key":         k,
            "name":        v["name"],
            "description": v["description"],
            "steps":       len(v["steps"]),
        }
        for k, v in SCENARIOS.items()
    ]
    
@app.get("/api/stats")
def get_stats():
    if state["final_scores"] is None:
        raise HTTPException(status_code=404, detail="No results yet")
    df   = state["final_scores"]
    txdf = state["df"]
    return {
        "total_accounts":     int(len(df)),
        "high_risk":          int((df["final_risk"] >= 0.9).sum()),
        "medium_risk":        int(((df["final_risk"] >= 0.6) & (df["final_risk"] < 0.9)).sum()),
        "low_risk":           int((df["final_risk"] < 0.6).sum()),
        "total_transactions": int(len(txdf)) if txdf is not None else 0,
        "fraud_rate":         float(txdf["label"].mean()) if txdf is not None else 0,
    }


@app.get("/api/ablation")
def get_ablation():
    path = "data/ablation.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No ablation results — run main.py first")
    with open(path) as f:
        return json.load(f)

@app.get("/api/random-account/{risk_level}")
def get_random_account(risk_level: str):
    if state["final_scores"] is None:
        raise HTTPException(status_code=404, detail="No results yet")
    df = state["final_scores"]

    if risk_level == "high":
        pool = df[
            (df["stat_risk"]  > 0.7) &
            (df["prop_risk"]  > 0.7)
        ]
    elif risk_level == "medium":
        pool = df[
            (df["stat_risk"].between(0.3, 0.7)) &
            (df["prop_risk"].between(0.3, 0.7))
        ]
    elif risk_level == "low":
        pool = df[
            (df["stat_risk"]   < 0.3) &
            (df["prop_risk"]   < 0.3)
        ]
    else:
        raise HTTPException(status_code=400, detail="risk_level must be high, medium or low")

    if len(pool) == 0:
        # fallback to final_risk thresholds
        if risk_level == "high":
            pool = df[df["final_risk"] >= 0.9]
        elif risk_level == "medium":
            pool = df[df["final_risk"].between(0.6, 0.9)]
        else:
            pool = df[df["final_risk"] < 0.6]

    if len(pool) == 0:
        raise HTTPException(status_code=404, detail=f"No {risk_level} risk accounts found")

    sample = pool.sample(1).iloc[0]
    return {
        "account":    sample["account"],
        "final_risk": float(sample["final_risk"]),
        "stat_risk":  float(sample["stat_risk"]),
        "struct_risk":float(sample["struct_risk"]),
        "prop_risk":  float(sample["prop_risk"]),
    }
    
@app.post("/api/inject-transaction")
def inject_transaction(req: InjectRequest):
    if state["model"] is None:
        raise HTTPException(status_code=404, detail="Model not loaded yet — please wait.")

    model        = state["model"]
    threshold    = state["threshold"]
    feature_cols = state["feature_cols"]

    amount = req.amount

    if amount > 500_000:
        # Large transfer — drain pattern matches PaySim fraud
        row = {
            "amount_scaled":      (amount - 179861.90) / 603858.18,
            "type_encoded":        4,
            "orig_balance_delta":  -amount,
            "dest_balance_delta":  0.0,
            "oldbalanceOrg":       amount,
            "newbalanceOrig":      0.0,
            "oldbalanceDest":      0.0,
            "newbalanceDest":      0.0,
        }
    else:
        # Small transfer — normal pattern
        row = {
            "amount_scaled":      (amount - 179861.90) / 603858.18,
            "type_encoded":        4,
            "orig_balance_delta":  -amount,
            "dest_balance_delta":  amount,
            "oldbalanceOrg":       amount * 2,
            "newbalanceOrig":      amount,
            "oldbalanceDest":      0.0,
            "newbalanceDest":      amount,
        }

    new_row    = pd.DataFrame([row])
    X          = new_row[feature_cols].fillna(0)
    fraud_prob = float(model.predict_proba(X)[0][1])
    fraud_flag = int(fraud_prob >= threshold)

    return {
        "sender":     req.sender,
        "receiver":   req.receiver,
        "amount":     req.amount,
        "fraud_prob": round(fraud_prob, 4),
        "fraud_flag": fraud_flag,
        "flagged":    bool(fraud_flag == 1),
        "model_used": "PaySim XGBoost",
        "note":       "Large amounts scored as drain pattern, small as normal transfer"
    }
# ── Serve dashboard ───────────────────────────────────────────
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")