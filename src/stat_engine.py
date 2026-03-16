# ============================================================
# Module 3: Statistical Risk Engine (XGBoost)
# ============================================================
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from config import XGBOOST_PARAMS


def train_model(df: pd.DataFrame, feature_cols: list) -> XGBClassifier:
    """Train XGBoost on transaction-level labels."""
    X = df[feature_cols].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluation
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("[StatEngine] Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"[StatEngine] ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    return model


def score_transactions(model: XGBClassifier,
                        df: pd.DataFrame,
                        feature_cols: list) -> pd.DataFrame:
    """
    Assign fraud probability to each transaction,
    then aggregate to account-level behavioral risk score.
    """
    X = df[feature_cols].fillna(0)
    df = df.copy()
    df["fraud_prob"] = model.predict_proba(X)[:, 1]

    # Aggregate per sender account
    account_risk = (
        df.groupby("sender")["fraud_prob"]
        .agg(
            stat_risk_mean="mean",
            stat_risk_max="max",
            tx_count="count",
        )
        .reset_index()
        .rename(columns={"sender": "account"})
    )

    # Final behavioral score = weighted blend of mean and max
    account_risk["stat_risk"] = (
        0.6 * account_risk["stat_risk_mean"] +
        0.4 * account_risk["stat_risk_max"]
    )

    # Normalise to [0, 1]
    r = account_risk["stat_risk"]
    account_risk["stat_risk"] = (r - r.min()) / (r.max() - r.min() + 1e-9)

    print(f"[StatEngine] Scored {len(account_risk):,} accounts")
    return account_risk[["account", "stat_risk", "tx_count"]]
