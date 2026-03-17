import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from config import XGBOOST_PARAMS


def find_best_threshold(y_true, y_proba) -> float:
    """
    Find the probability threshold that maximises F1 score on the fraud class.
    Default 0.5 is wrong for imbalanced data — this finds the sweet spot
    between precision and recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"[StatEngine] Best threshold: {best_threshold:.4f} "
          f"(precision={precisions[best_idx]:.2f}, "
          f"recall={recalls[best_idx]:.2f}, "
          f"F1={f1_scores[best_idx]:.2f})")
    return best_threshold


def train_model(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Train XGBoost. Returns (model, best_threshold)."""
    X = df[feature_cols].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    threshold = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    print("[StatEngine] Classification Report (tuned threshold):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"[StatEngine] ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")

    return model, threshold


def score_transactions(model: XGBClassifier,
                        df: pd.DataFrame,
                        feature_cols: list,
                        threshold: float) -> pd.DataFrame:
    X = df[feature_cols].fillna(0)
    df = df.copy()
    df["fraud_prob"] = model.predict_proba(X)[:, 1]
    df["fraud_flag"]  = (df["fraud_prob"] >= threshold).astype(int)

    account_risk = (
        df.groupby("sender")["fraud_prob"]
        .agg(
            stat_risk_mean="mean",
            stat_risk_max="max",
            stat_risk_std="std",
            tx_count="count",
        )
        .reset_index()
        .rename(columns={"sender": "account"})
    )

    account_risk["stat_risk_std"] = account_risk["stat_risk_std"].fillna(0)

    account_risk["stat_risk"] = (
        0.5 * account_risk["stat_risk_mean"] +
        0.3 * account_risk["stat_risk_max"] +
        0.2 * account_risk["stat_risk_std"]
    )

    r = account_risk["stat_risk"]
    account_risk["stat_risk"] = (r - r.min()) / (r.max() - r.min() + 1e-9)

    print(f"[StatEngine] Scored {len(account_risk):,} accounts")
    return account_risk[["account", "stat_risk", "tx_count"]]