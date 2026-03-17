# ============================================================
# RISKFORGE — FastAPI Backend
# ============================================================
import sys, os, gc, traceback, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from typing import Any
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