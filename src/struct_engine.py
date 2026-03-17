# ============================================================
# Module 4: Structural Risk Engine
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def extract_graph_features(G, A_norm, node_list) -> pd.DataFrame:
    """
    Compute all features directly from the sparse matrix.
    No NetworkX iteration — avoids MemoryError on 9M nodes.
    """
    print("[StructEngine] Computing graph features...")

    # Degree features — pure numpy, no Python loops
    in_degree  = np.asarray(A_norm.astype(bool).sum(axis=0)).flatten().astype(np.float32)
    out_degree = np.asarray(A_norm.astype(bool).sum(axis=1)).flatten().astype(np.float32)
    in_weight  = np.asarray(A_norm.sum(axis=0)).flatten().astype(np.float32)
    out_weight = np.asarray(A_norm.sum(axis=1)).flatten().astype(np.float32)
    fan_out    = out_degree / np.maximum(in_degree, 1.0)

    print("[StructEngine] Computing PageRank...")
    pagerank = _pagerank_sparse(A_norm)

    features = pd.DataFrame({
        "account":    node_list,
        "in_degree":  in_degree,
        "out_degree": out_degree,
        "in_weight":  in_weight,
        "out_weight": out_weight,
        "pagerank":   pagerank,
        "fan_out":    fan_out,
    })

    print(f"[StructEngine] Extracted features for {len(features):,} accounts")
    return features


def _pagerank_sparse(A_norm, alpha=0.85, max_iter=100, tol=1e-6):
    """Power iteration PageRank — returns numpy array."""
    n = A_norm.shape[0]
    r = np.full(n, 1.0 / n, dtype=np.float32)
    for _ in range(max_iter):
        r_new = alpha * A_norm.T.dot(r) + (1 - alpha) / n
        if np.linalg.norm(r_new - r) < tol:
            break
        r = r_new
    return r


def score_structural_risk(features: pd.DataFrame,
                           account_labels: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    df = features.merge(account_labels, on="account", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    feature_cols = ["in_degree", "out_degree", "in_weight",
                    "out_weight", "pagerank", "fan_out"]

    X = df[feature_cols].fillna(0)
    y = df["label"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if y.sum() > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = LogisticRegression(class_weight="balanced", max_iter=500)
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)[:, 1]
        print(f"[StructEngine] ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        df["struct_risk"] = clf.predict_proba(X_scaled)[:, 1]
    else:
        print("[StructEngine] Not enough fraud labels — using PageRank as proxy")
        df["struct_risk"] = MinMaxScaler().fit_transform(df[["pagerank"]])

    s = df["struct_risk"]
    df["struct_risk"] = (s - s.min()) / (s.max() - s.min() + 1e-9)

    return df[["account", "struct_risk"]]


def get_account_labels(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("sender")["label"]
        .max()
        .reset_index()
        .rename(columns={"sender": "account"})
    )