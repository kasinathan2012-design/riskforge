# ============================================================
# Module 1: Data Preprocessing & Feature Engineering
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_creditcard(path: str) -> pd.DataFrame:
    """
    Load the Kaggle Credit Card Fraud dataset.
    Columns: Time, V1–V28 (PCA features), Amount, Class (0=legit, 1=fraud)
    Since sender/receiver aren't present, we assign synthetic account IDs.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Class": "label"})

    # Synthetic account IDs — each row treated as a unique sender
    # In a real dataset these would be real account numbers
    df["sender"]   = ["ACC_" + str(i) for i in range(len(df))]
    df["receiver"] = ["ACC_" + str(np.random.randint(0, len(df) // 10)) for _ in range(len(df))]

    df["amount"] = df["Amount"]
    return df


def load_paysim(path: str) -> pd.DataFrame:
    """
    Load the PaySim dataset.
    Columns: step, type, amount, nameOrig, nameDest, oldbalanceOrg,
             newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud
    """
    df = pd.read_csv(path)
    df = df.rename(columns={
        "nameOrig":  "sender",
        "nameDest":  "receiver",
        "isFraud":   "label",
        "amount":    "amount",
    })

    # Encode transaction type
    df["type_encoded"] = pd.Categorical(df["type"]).codes
    return df


def engineer_features_creditcard(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Feature engineering for credit card data.
    Returns the dataframe and the list of feature column names.
    """
    scaler = StandardScaler()
    df["amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["amount_scaled"]
    return df, feature_cols


def engineer_features_paysim(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Feature engineering for PaySim data.
    """
    scaler = StandardScaler()
    df["amount_scaled"] = scaler.fit_transform(df[["amount"]])

    # Balance delta features — useful fraud signals
    df["orig_balance_delta"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]

    feature_cols = [
        "amount_scaled",
        "type_encoded",
        "orig_balance_delta",
        "dest_balance_delta",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    return df, feature_cols


def preprocess(dataset: str, path: str) -> tuple[pd.DataFrame, list]:
    """
    Master entry point. Returns (dataframe, feature_columns).
    dataset: 'creditcard' or 'paysim'
    """
    if dataset == "creditcard":
        df = load_creditcard(path)
        df, features = engineer_features_creditcard(df)
    elif dataset == "paysim":
        df = load_paysim(path)
        df, features = engineer_features_paysim(df)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"[Preprocess] Loaded {len(df):,} transactions | Fraud rate: "
          f"{df['label'].mean()*100:.2f}%")
    return df, features
