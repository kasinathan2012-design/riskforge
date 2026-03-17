# ============================================================
# RISKFORGE — FastAPI Backend
# ============================================================
import sys, os, gc, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import threading

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

load_results_from_disk()


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
        raise HTTPException(
            status_code=404,
            detail="No results.csv found — run: python main.py first"
        )
    try:
        df = pd.read_csv(path)
        state["final_scores"] = df
        state["status"] = "ready"
        return {
            "message":  f"Reloaded {len(df):,} accounts from disk",
            "accounts": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results")
def get_results(limit: int = 500, min_risk: float = 0.0):
    if state["final_scores"] is None:
        raise HTTPException(status_code=404, detail="No results yet — run main.py first")

    df = state["final_scores"]

    # Sample across all risk tiers so dashboard shows a meaningful mix
    high   = df[df["final_risk"] >= 0.9].head(200)
    medium = df[(df["final_risk"] >= 0.6) & (df["final_risk"] < 0.9)].head(150)
    low    = df[df["final_risk"] < 0.6].head(150)

    result = pd.concat([high, medium, low]).sort_values("final_risk", ascending=False)

    return {
        "total_accounts": len(df),
        "returned":       len(result),
        "accounts":       result.to_dict(orient="records"),
    }


@app.get("/api/account/{account_id}")
def get_account(account_id: str):
    if state["final_scores"] is None:
        raise HTTPException(status_code=404, detail="No results yet")
    df  = state["final_scores"]
    row = df[df["account"] == account_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    record = row.iloc[0].to_dict()
    tx_history = []
    if state["df"] is not None:
        txs = state["df"][
            (state["df"]["sender"]   == account_id) |
            (state["df"]["receiver"] == account_id)
        ].head(20)
        tx_history = txs[["sender", "receiver", "amount", "label"]].to_dict(orient="records")
    return {"account": record, "transactions": tx_history}


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
    import json
    with open(path) as f:
        return json.load(f)
    
@app.post("/api/inject-transaction")
def inject_transaction(req: InjectRequest):
    if state["model"] is None:
        raise HTTPException(
            status_code=404,
            detail="Model not loaded — run main.py first to enable transaction injection."
        )
    model        = state["model"]
    threshold    = state["threshold"]
    feature_cols = state["feature_cols"]

    new_row = pd.DataFrame([{
        "sender":             req.sender,
        "receiver":           req.receiver,
        "amount":             req.amount,
        "label":              0,
        "type":               req.tx_type,
        "type_encoded":       0,
        "orig_balance_delta": 0,
        "dest_balance_delta": 0,
        "oldbalanceOrg":      0,
        "newbalanceOrig":     0,
        "oldbalanceDest":     0,
        "newbalanceDest":     0,
        "amount_scaled":      req.amount / 1e6,
    }])

    X          = new_row[feature_cols].fillna(0)
    fraud_prob = float(model.predict_proba(X)[0][1])
    fraud_flag = int(fraud_prob >= threshold)

    return {
        "sender":     req.sender,
        "receiver":   req.receiver,
        "amount":     req.amount,
        "fraud_prob": round(fraud_prob, 4),
        "fraud_flag": fraud_flag,
        "flagged":    fraud_flag == 1,
        "note":       "Score based on current loaded model"
    }


# ── Serve dashboard ───────────────────────────────────────────
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")