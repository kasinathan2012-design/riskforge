# ============================================================
# Module 2: Graph Construction
# ============================================================
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags


def build_graph(df: pd.DataFrame):
    """
    Returns aggregated edge dataframe instead of NetworkX graph.
    Avoids holding 9M nodes in memory as Python objects.
    """
    print("[Graph] Aggregating edges...")
    edges = (
        df.groupby(["sender", "receiver"])
        .agg(weight=("amount", "sum"), count=("amount", "count"))
        .reset_index()
    )
    print(f"[Graph] Unique edges: {len(edges):,}")
    return edges


def get_adjacency_matrix(edges: pd.DataFrame) -> tuple:
    """
    Build sparse CSR matrix directly from edge dataframe.
    No NetworkX, no Python dicts — pure pandas + scipy.
    """
    print("[Graph] Building sparse adjacency matrix...")

    # Encode sender and receiver as integer codes using pandas Categorical
    # This is done in C — no Python dict comprehension
    all_accounts = pd.Categorical(
        pd.concat([edges["sender"], edges["receiver"]], ignore_index=True)
    )
    node_list = all_accounts.categories.tolist()
    n = len(node_list)
    print(f"[Graph] Nodes: {n:,}")

    src_codes = pd.Categorical(
        edges["sender"], categories=all_accounts.categories
    ).codes.astype(np.int32)

    dst_codes = pd.Categorical(
        edges["receiver"], categories=all_accounts.categories
    ).codes.astype(np.int32)

    weights = edges["weight"].values.astype(np.float32)

    # Free intermediate objects
    del all_accounts
    import gc; gc.collect()

    A = csr_matrix(
        (weights, (src_codes, dst_codes)),
        shape=(n, n),
        dtype=np.float32
    )

    del src_codes, dst_codes, weights
    gc.collect()

    # Row-normalise
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = diags(1.0 / row_sums)
    A_norm = D_inv @ A
    del A, row_sums, D_inv
    gc.collect()

    print(f"[Graph] Sparse matrix built — {A_norm.nnz:,} non-zero entries")
    return A_norm, node_list