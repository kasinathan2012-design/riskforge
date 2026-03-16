# ============================================================
# Module 2: Graph Construction
# ============================================================
import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted transaction graph.
    Uses pandas groupby — much faster than iterating rows.
    """
    print("[Graph] Aggregating edges...")
    edges = (
        df.groupby(["sender", "receiver"])
        .agg(weight=("amount", "sum"), count=("amount", "count"))
        .reset_index()
    )

    G = nx.from_pandas_edgelist(
        edges,
        source="sender",
        target="receiver",
        edge_attr=["weight", "count"],
        create_using=nx.DiGraph(),
    )

    print(f"[Graph] Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")
    return G


def get_adjacency_matrix(G: nx.DiGraph) -> tuple:
    """
    Returns (row-normalised sparse CSR matrix, ordered node list).
    Uses scipy sparse — handles millions of nodes without memory issues.
    """
    import numpy as np
    from scipy.sparse import csr_matrix, diags

    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    print(f"[Graph] Building sparse adjacency matrix ({n:,} x {n:,})...")

    rows, cols, data = [], [], []
    for src, dst, attr in G.edges(data=True):
        rows.append(node_index[src])
        cols.append(node_index[dst])
        data.append(attr.get("weight", 1.0))

    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = diags(1.0 / row_sums)
    A_norm = D_inv @ A

    print(f"[Graph] Sparse matrix built — {A_norm.nnz:,} non-zero entries")
    return A_norm, nodes