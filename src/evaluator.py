# ============================================================
# Module 7: Evaluation — Ablation Study
# ============================================================
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score
)


def build_account_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive account-level ground truth from transaction labels.
    An account is labelled fraud if it sent at least one fraudulent transaction.
    """
    return (
        df.groupby("sender")["label"]
        .max()
        .reset_index()
        .rename(columns={"sender": "account", "label": "true_label"})
    )


def find_best_threshold(y_true, y_score) -> float:
    """Find threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best = np.argmax(f1s)
    return float(thresholds[best]) if best < len(thresholds) else 0.5


def evaluate_signal(y_true, y_score, name: str) -> dict:
    """Evaluate a single risk signal against ground truth."""
    # Skip if no positive examples
    if y_true.sum() == 0:
        return {"name": name, "precision": 0, "recall": 0, "f1": 0, "auc": 0}

    threshold = find_best_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "name":      name,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc":       round(float(roc_auc_score(y_true, y_score)), 4),
        "threshold": round(threshold, 4),
        "positives": int(y_pred.sum()),
        "true_positives": int((y_pred & y_true).sum()),
    }


def run_ablation(results_path: str, transactions_df: pd.DataFrame) -> list:
    """
    Run full ablation study comparing all signal combinations.
    Returns list of metric dicts for each model configuration.
    """
    print("[Evaluator] Running ablation study...")

    # Load scores
    scores = pd.read_csv(results_path)

    # Build ground truth
    labels = build_account_labels(transactions_df)

    # Merge
    df = scores.merge(labels, on="account", how="inner")
    print(f"[Evaluator] Matched {len(df):,} accounts with ground truth labels")
    print(f"[Evaluator] Fraud accounts: {df['true_label'].sum():,} ({df['true_label'].mean()*100:.2f}%)")

    y_true = df["true_label"].values

    # Evaluate each signal
    results = [
        evaluate_signal(y_true, df["stat_risk"].values,   "Behavioral Only"),
        evaluate_signal(y_true, df["struct_risk"].values, "Structural Only"),
        evaluate_signal(y_true, df["prop_risk"].values,   "Propagation Only"),
        evaluate_signal(y_true, df["final_risk"].values,  "Multi-Signal"),
    ]

    # Print summary
    print(f"\n{'='*65}")
    print(f"  Ablation Study Results")
    print(f"{'='*65}")
    print(f"{'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"{'-'*65}")
    for r in results:
        print(f"{r['name']:<22} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f}")
    print(f"{'='*65}\n")

    # Save to disk so API can serve it
    out_path = "data/ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Evaluator] Results saved to {out_path}")

    return results