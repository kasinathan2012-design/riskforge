# ============================================================
# Module 4: Structural Risk Engine
# ============================================================
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def extract_graph_features(G: nx.DiGraph) -> pd.DataFrame:
    print("[StructEngine] Computing graph features...")

    in_degree  = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    in_weight  = dict(G.in_degree(weight="weight"))
    out_weight = dict(G.out_degree(weight="weight"))

    # PageRank — fast even on 9M nodes
    print("[StructEngine] Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")

    # Betweenness removed (O(n²), hours on 9M nodes).
    # Replaced with two fast fraud-relevant features:

    # Fan-out ratio: high out/in degree ratio = potential money mule or distributor
    fan_out = {
        n: out_degree.get(n, 0) / max(in_degree.get(n, 1), 1)
        for n in G.nodes()
    }

    # Self-loop: account transacts with itself
    self_loops = {n: 1 if G.has_edge(n, n) else 0 for n in G.nodes()}

    nodes = list(G.nodes())
    features = pd.DataFrame({
        "account":    nodes,
        "in_degree":  [in_degree.get(n, 0)  for n in nodes],
        "out_degree": [out_degree.get(n, 0) for n in nodes],
        "in_weight":  [in_weight.get(n, 0)  for n in nodes],
        "out_weight": [out_weight.get(n, 0) for n in nodes],
        "pagerank":   [pagerank.get(n, 0)   for n in nodes],
        "fan_out":    [fan_out.get(n, 0)    for n in nodes],
        "self_loop":  [self_loops.get(n, 0) for n in nodes],
    })

    print(f"[StructEngine] Extracted features for {len(features):,} accounts")
    return features


def score_structural_risk(features: pd.DataFrame,
                           account_labels: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    df = features.merge(account_labels, on="account", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    feature_cols = ["in_degree", "out_degree", "in_weight",
                    "out_weight", "pagerank", "fan_out", "self_loop"]

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