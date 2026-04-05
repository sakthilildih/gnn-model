"""
build_graph.py
Loads transactions.csv → builds PyG Data object with:
  - 18 aggregated node features
  - Louvain community_id feature
  - Multi-class node labels (pattern_family → class index)
  - Train / Val / Test masks  (cluster-level split, no leakage)
  - Per-account cluster mapping + amount for blocked-amount reporting
"""

import os, pickle, time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import community as community_louvain
from tqdm import tqdm

from config import (
    DATA_DIR, GRAPH_DIR, PATTERN_TO_CLASS, NUM_CLASSES, FEATURE_COLS
)


# FEATURE_COLS imported from config.py


def build_graph(force_rebuild: bool = False):
    graph_path = os.path.join(GRAPH_DIR, "graph_data.pt")
    meta_path  = os.path.join(GRAPH_DIR, "graph_meta.pkl")

    if not force_rebuild and os.path.exists(graph_path) and os.path.exists(meta_path):
        print("  [cache hit] Loading pre-built graph...")
        data = torch.load(graph_path, weights_only=False)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return data, meta

    t0 = time.time()
    print("Loading transaction CSV...")
    df = pd.read_csv(
        os.path.join(DATA_DIR, "transactions.csv"),
        parse_dates=["timestamp"],
    )
    dc = pd.read_csv(os.path.join(DATA_DIR, "cluster_summary.csv"))
    print(f"  {len(df):,} rows | {len(dc):,} clusters")

    # ── Account index ─────────────────────────────────────────────────────────
    all_accounts = pd.unique(
        np.concatenate([
            df["sender_account_id"].values,
            df["receiver_account_id"].values,
        ])
    )
    acc_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
    n_nodes    = len(all_accounts)
    print(f"  Unique accounts (nodes): {n_nodes:,}")

    # ── Per-account amount (for total_amount_blocked) ─────────────────────────
    # Sum of all amounts where account was sender OR receiver
    sent_amt = df.groupby("sender_account_id")["amount"].sum().rename("sent_amt_total")
    recv_amt = df.groupby("receiver_account_id")["amount"].sum().rename("recv_amt_total")

    # ── Node feature computation ──────────────────────────────────────────────
    print("Computing node features...")
    df["ts_unix"] = df["timestamp"].astype(np.int64) // 10 ** 9

    sent = (
        df.groupby("sender_account_id")
        .agg(
            sent_count        = ("transaction_id", "count"),
            sent_avg_amt      = ("amount",          "mean"),
            sent_max_amt      = ("amount",           "max"),
            sent_std_amt      = ("amount",            "std"),
            sent_unique_types = ("transaction_type", "nunique"),
            sent_atm_cnt      = ("transaction_type", lambda x: (x == "ATM").sum()),
            sent_wallet_cnt   = ("transaction_type", lambda x: (x == "Wallet").sum()),
            sent_unique_pins  = ("receiver_pincode", "nunique"),
            ts_min            = ("ts_unix",           "min"),
            ts_max            = ("ts_unix",            "max"),
        )
        .reset_index()
        .rename(columns={"sender_account_id": "account_id"})
    )

    recv = (
        df.groupby("receiver_account_id")
        .agg(
            recv_count        = ("transaction_id", "count"),
            recv_avg_amt      = ("amount",          "mean"),
            recv_max_amt      = ("amount",           "max"),
            recv_std_amt      = ("amount",            "std"),
            recv_unique_types = ("transaction_type", "nunique"),
            recv_unique_pins  = ("sender_pincode",  "nunique"),
        )
        .reset_index()
        .rename(columns={"receiver_account_id": "account_id"})
    )

    node_df = pd.DataFrame({"account_id": all_accounts})
    node_df  = node_df.merge(sent, on="account_id", how="left")
    node_df  = node_df.merge(recv, on="account_id", how="left")
    node_df  = node_df.merge(sent_amt, left_on="account_id",
                              right_index=True, how="left")
    node_df  = node_df.merge(recv_amt, left_on="account_id",
                              right_index=True, how="left")
    node_df.fillna(0, inplace=True)

    # Derived features
    node_df["total_degree"]       = node_df["sent_count"] + node_df["recv_count"]
    node_df["out_ratio"]          = node_df["sent_count"] / (node_df["total_degree"] + 1e-8)
    node_df["in_ratio"]           = node_df["recv_count"] / (node_df["total_degree"] + 1e-8)
    node_df["cross_channel_score"]= (node_df["sent_unique_types"] +
                                      node_df["recv_unique_types"]) / 2
    node_df["atm_ratio"]          = node_df["sent_atm_cnt"] / (node_df["sent_count"] + 1e-8)
    node_df["wallet_ratio"]       = node_df["sent_wallet_cnt"] / (node_df["sent_count"] + 1e-8)
    node_df["unique_pincodes"]    = (node_df["sent_unique_pins"] +
                                      node_df["recv_unique_pins"])
    time_span_min = (node_df["ts_max"] - node_df["ts_min"]) / 60.0 + 1
    node_df["burst_velocity"]     = node_df["total_degree"] / time_span_min
    node_df["amount_range"]       = node_df["sent_max_amt"] - node_df["sent_avg_amt"]
    # Total amount linked to account (both sides) → used in blocked-amount report
    node_df["total_amount_linked"]= node_df["sent_amt_total"] + node_df["recv_amt_total"]

    # ── Edge index ────────────────────────────────────────────────────────────
    print("Building edge index...")
    src_idx = df["sender_account_id"].map(acc_to_idx).values.astype(np.int64)
    dst_idx = df["receiver_account_id"].map(acc_to_idx).values.astype(np.int64)
    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

    # Edge amounts — needed for total_amount_blocked downstream
    edge_amounts = torch.tensor(df["amount"].values, dtype=torch.float)

    # ── Louvain community detection ───────────────────────────────────────────
    print("Building undirected NetworkX graph for Louvain...")
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edges_nx = list(zip(src_idx.tolist(), dst_idx.tolist()))
    G.add_edges_from(edges_nx)

    print("Running Louvain (this may take a few minutes)...")
    t_louv = time.time()
    partition = community_louvain.best_partition(G, random_state=42)
    print(f"  Louvain done in {time.time()-t_louv:.1f}s | "
          f"{max(partition.values())+1} communities found")

    # Map back: node index → community id
    comm_arr = np.array([partition.get(i, 0) for i in range(n_nodes)], dtype=np.float32)
    node_df["louvain_community"] = comm_arr
    max_comm = comm_arr.max() + 1
    node_df["community_id_norm"] = comm_arr / max_comm

    # ── Node labels ───────────────────────────────────────────────────────────
    print("Assigning node labels...")
    cluster_pattern = dc.set_index("cluster_id")["pattern_family"].to_dict()
    cluster_split   = dc.set_index("cluster_id")["split"].to_dict()

    # account → cluster_id  (each account belongs to exactly one cluster)
    acc_cluster = (
        pd.concat([
            df[["sender_account_id",   "cluster_id"]].rename(columns={"sender_account_id":   "account_id"}),
            df[["receiver_account_id", "cluster_id"]].rename(columns={"receiver_account_id": "account_id"}),
        ])
        .drop_duplicates(subset="account_id", keep="first")
        .set_index("account_id")["cluster_id"]
        .to_dict()
    )

    node_df["cluster_id"]     = node_df["account_id"].map(acc_cluster)
    node_df["pattern_family"] = node_df["cluster_id"].map(cluster_pattern).fillna("benign_unknown")
    node_df["split"]          = node_df["cluster_id"].map(cluster_split).fillna("train")
    node_df["y"]              = (
        node_df["pattern_family"]
        .map(PATTERN_TO_CLASS)
        .fillna(0)
        .astype(int)
    )

    # ── Standardise features ──────────────────────────────────────────────────
    X     = node_df[FEATURE_COLS].values.astype(np.float32)
    f_mean = X.mean(axis=0)
    f_std  = X.std(axis=0) + 1e-8
    X_norm = (X - f_mean) / f_std

    # ── PyG Data object ───────────────────────────────────────────────────────
    data = Data(
        x          = torch.tensor(X_norm, dtype=torch.float),
        edge_index = edge_index,
        edge_attr  = edge_amounts,          # transaction amount on each edge
        y          = torch.tensor(node_df["y"].values, dtype=torch.long),
        train_mask = torch.tensor(node_df["split"].values == "train", dtype=torch.bool),
        val_mask   = torch.tensor(node_df["split"].values == "val",   dtype=torch.bool),
        test_mask  = torch.tensor(node_df["split"].values == "test",  dtype=torch.bool),
    )

    print(f"  Graph: {data.num_nodes:,} nodes | "
          f"{data.num_edges:,} edges | "
          f"{data.num_node_features} features | "
          f"{NUM_CLASSES} classes")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(data, graph_path)

    meta = {
        "acc_to_idx":              acc_to_idx,
        "idx_to_acc":              {v: k for k, v in acc_to_idx.items()},
        "feature_cols":            FEATURE_COLS,
        "feature_mean":            f_mean.tolist(),
        "feature_std":             f_std.tolist(),
        "n_nodes":                 n_nodes,
        "cluster_pattern":         cluster_pattern,
        "cluster_split":           cluster_split,
        "acc_cluster":             acc_cluster,
        "louvain_max_communities": int(max_comm),  # used by production inference
        # amount per node (for blocked amount report)
        "node_total_amount":  node_df.set_index("account_id")["total_amount_linked"].to_dict(),
        # cluster → sum of all transaction amounts (ground-truth for report)
        "cluster_total_amount": (
            df.groupby("cluster_id")["amount"].sum().to_dict()
        ),
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    # also save node table for debugging
    node_df.to_csv(os.path.join(GRAPH_DIR, "node_features.csv"), index=False)

    print(f"  Graph saved to {GRAPH_DIR}/  ({time.time()-t0:.1f}s total)")
    return data, meta


if __name__ == "__main__":
    build_graph(force_rebuild=True)
