"""
production_inference.py — AML GNN Production Inference Engine

Called by the integration layer after XGBoost flags suspicious account IDs.

Pipeline
--------
XGBoost → suspicious_account_ids
       → TransactionStore.get_related_transactions(account_ids, hops=2)
       → AMLProductionEngine.process_batch(account_ids, txn_df)
       → List[RingDict]

Ring output format
------------------
{
    "ring_id":              "RING_20260405_000001",
    "accounts":             ["ACC001", "ACC002", ...],
    "account_count":        7,
    "pattern":              "mule_ring",
    "risk_score":           0.94,           # mean mule-prob across ring
    "is_novel":             False,
    "confidence":           0.88,           # fraction voting for top class
    "total_amount_blocked": 2_345_678.50,
    "transaction_ids":      ["TXN001", ...],
    "flagged_accounts":     ["ACC001"],     # original XGBoost seed accounts
    "detected_at":          "2026-04-05T15:33:35"
}

Usage
-----
    engine = AMLProductionEngine()
    engine.load()                           # once at server start-up

    rings = engine.process_batch(
        suspicious_account_ids = ["ACC001", "ACC002"],
        txn_df                 = related_transactions_df,   # from TransactionStore
    )
"""

import os, pickle, time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import networkx as nx
import community as community_louvain

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_DIR, GRAPH_DIR, FEATURE_COLS, CLASS_NAMES, NUM_CLASSES,
    HIDDEN_DIM, DROPOUT, OOD_MIN_RISK_SCORE, CLUSTER_RISK_THRESHOLD,
    MIN_RING_SIZE, MAX_INFER_NODES,
)
from model import AMLGraphSAGE


# ── Ring ID counter ───────────────────────────────────────────────────────────
_ring_counter = 0

def _next_ring_id() -> str:
    global _ring_counter
    _ring_counter += 1
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"RING_{date_str}_{_ring_counter:06d}"


# ── Node feature extraction (mirrors build_graph.py exactly) ─────────────────

def _extract_node_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 17 raw node features (community_id_norm added separately).
    Mirrors build_graph.py so features are identical to training.
    """
    df = txn_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ts_unix"]   = df["timestamp"].astype(np.int64) // 10 ** 9

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

    all_accounts = pd.unique(
        np.concatenate([df["sender_account_id"].values,
                        df["receiver_account_id"].values])
    )
    node_df = pd.DataFrame({"account_id": all_accounts})
    node_df = node_df.merge(sent, on="account_id", how="left")
    node_df = node_df.merge(recv, on="account_id", how="left")
    node_df.fillna(0, inplace=True)

    node_df["total_degree"]        = node_df["sent_count"] + node_df["recv_count"]
    node_df["out_ratio"]           = (node_df["sent_count"]
                                      / (node_df["total_degree"] + 1e-8))
    node_df["in_ratio"]            = (node_df["recv_count"]
                                      / (node_df["total_degree"] + 1e-8))
    node_df["cross_channel_score"] = ((node_df["sent_unique_types"]
                                       + node_df["recv_unique_types"]) / 2)
    node_df["atm_ratio"]           = (node_df["sent_atm_cnt"]
                                      / (node_df["sent_count"] + 1e-8))
    node_df["wallet_ratio"]        = (node_df["sent_wallet_cnt"]
                                      / (node_df["sent_count"] + 1e-8))
    node_df["unique_pincodes"]     = (node_df["sent_unique_pins"]
                                      + node_df["recv_unique_pins"])
    time_span_min = (node_df["ts_max"] - node_df["ts_min"]) / 60.0 + 1
    node_df["burst_velocity"]      = node_df["total_degree"] / time_span_min
    node_df["amount_range"]        = node_df["sent_max_amt"] - node_df["sent_avg_amt"]
    return node_df


# ── Main engine ───────────────────────────────────────────────────────────────

class AMLProductionEngine:
    """
    Production GNN inference engine — load once, call per batch.

    Attributes loaded from disk
    ---------------------------
    model          : AMLGraphSAGE (weights from best_model.pth)
    centroids      : numpy array [NUM_CLASSES, HIDDEN_DIM]
    ood_threshold  : float (95th-pct centroid distance from training)
    feature_mean   : numpy array [18]  — training normalisation mean
    feature_std    : numpy array [18]  — training normalisation std
    louvain_max    : int — number of Louvain communities in training graph
                     used to scale community_id_norm at inference time
    """

    def __init__(self):
        self.model          = None
        self.centroids      = None
        self.ood_threshold  = None
        self.feature_mean   = None
        self.feature_std    = None
        self.louvain_max    = None
        self._loaded        = False

    # ── Load artifacts ────────────────────────────────────────────────────────

    def load(self,
             meta_path: str  = None,
             model_dir: str  = None) -> "AMLProductionEngine":
        """
        Load model weights and all inference artefacts from disk.
        Call once at application start-up.
        """
        meta_path = meta_path or os.path.join(GRAPH_DIR, "graph_meta.pkl")
        model_dir = model_dir or MODEL_DIR

        print("  [AMLProductionEngine] Loading artefacts...")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        in_channels = len(meta["feature_cols"])
        self.model  = AMLGraphSAGE(
            in_channels      = in_channels,
            hidden_channels  = HIDDEN_DIM,
            out_channels     = NUM_CLASSES,
            dropout          = DROPOUT,
        )
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "best_model.pth"),
                       map_location="cpu", weights_only=True)
        )
        self.model.eval()

        self.centroids     = np.load(os.path.join(model_dir, "centroids.npy"))
        self.ood_threshold = float(
            np.load(os.path.join(model_dir, "ood_threshold.npy"))[0]
        )
        self.feature_mean  = np.array(meta["feature_mean"], dtype=np.float32)
        self.feature_std   = np.array(meta["feature_std"],  dtype=np.float32)
        # Saved by build_graph.py — needed so community_id_norm is in the same
        # distribution as during training (critical for correct classification)
        self.louvain_max   = float(meta.get("louvain_max_communities", 1000))

        self._loaded = True
        print(f"  [AMLProductionEngine] Ready  "
              f"(louvain_max={int(self.louvain_max)}, "
              f"ood_threshold={self.ood_threshold:.4f})")
        return self

    # ── Main inference entry-point ────────────────────────────────────────────

    def process_batch(
        self,
        suspicious_account_ids: list,
        txn_df: pd.DataFrame,
    ) -> list:
        """
        Run GNN inference on a batch of suspicious accounts.

        Parameters
        ----------
        suspicious_account_ids : account IDs flagged by XGBoost (the "seeds").
        txn_df                 : related transactions pulled from TransactionStore
                                 (2-hop expanded around the seed accounts).

        Returns
        -------
        List of ring dicts, sorted by risk_score descending.
        Only rings with >= MIN_RING_SIZE accounts are returned.
        """
        if not self._loaded:
            raise RuntimeError("Call engine.load() before process_batch().")
        if txn_df.empty or len(suspicious_account_ids) == 0:
            return []

        # Safety cap
        all_accs = pd.unique(
            np.concatenate([txn_df["sender_account_id"].values,
                            txn_df["receiver_account_id"].values])
        )
        if len(all_accs) > MAX_INFER_NODES:
            # Keep only accounts connected to seeds (1-hop)
            seed_set = set(suspicious_account_ids)
            direct   = set()
            for _, row in txn_df.iterrows():
                if row["sender_account_id"] in seed_set:
                    direct.add(row["receiver_account_id"])
                if row["receiver_account_id"] in seed_set:
                    direct.add(row["sender_account_id"])
            keep = seed_set | direct
            txn_df = txn_df[
                txn_df["sender_account_id"].isin(keep) |
                txn_df["receiver_account_id"].isin(keep)
            ].copy()

        # ── 1. Node features ──────────────────────────────────────────────────
        node_df     = _extract_node_features(txn_df)
        accounts    = node_df["account_id"].tolist()
        acc_to_idx  = {a: i for i, a in enumerate(accounts)}
        n_nodes     = len(accounts)

        # ── 2. Edge index ─────────────────────────────────────────────────────
        src_idx = txn_df["sender_account_id"].map(acc_to_idx).dropna().astype(int).values
        dst_idx = txn_df["receiver_account_id"].map(acc_to_idx).dropna().astype(int).values

        if len(src_idx) == 0:
            return []

        # ── 3. Louvain community detection ────────────────────────────────────
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(zip(src_idx.tolist(), dst_idx.tolist()))
        partition = community_louvain.best_partition(G, random_state=42)
        comm_arr  = np.array([partition.get(i, 0) for i in range(n_nodes)],
                             dtype=np.float32)

        # Normalise using training scale — keeps this feature in-distribution
        node_df["community_id_norm"] = comm_arr / self.louvain_max

        # ── 4. Feature standardisation ────────────────────────────────────────
        X        = node_df[FEATURE_COLS].values.astype(np.float32)
        X_norm   = (X - self.feature_mean) / self.feature_std
        x_tensor = torch.tensor(X_norm, dtype=torch.float)
        ei_tensor= torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

        # ── 5. GNN forward pass ───────────────────────────────────────────────
        with torch.no_grad():
            mule_probs, pred_classes, embs = self.model.mule_score(
                x_tensor, ei_tensor
            )
        mule_probs_np   = mule_probs.numpy()
        pred_classes_np = pred_classes.numpy()
        embs_np         = embs.numpy()

        # ── 6. OOD detection per node ─────────────────────────────────────────
        is_novel = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            if mule_probs_np[i] >= OOD_MIN_RISK_SCORE:
                dists = np.linalg.norm(self.centroids - embs_np[i], axis=1)
                if dists.min() > self.ood_threshold:
                    is_novel[i] = True

        node_df["louvain_community"] = comm_arr.astype(int)
        node_df["mule_score"]        = mule_probs_np
        node_df["pred_class"]        = pred_classes_np
        node_df["is_novel"]          = is_novel

        # ── 7. Aggregate per Louvain community → rings ────────────────────────
        txn_ids_by_account = (
            txn_df.groupby("sender_account_id")["transaction_id"].apply(list)
            .add(txn_df.groupby("receiver_account_id")["transaction_id"].apply(list),
                 fill_value=[])
        )

        acc_amounts = (
            txn_df.groupby("sender_account_id")["amount"].sum()
            .add(txn_df.groupby("receiver_account_id")["amount"].sum(), fill_value=0)
        )

        seed_set  = set(suspicious_account_ids)
        rings_out = []
        detected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        for comm_id, grp in node_df.groupby("louvain_community"):
            cluster_risk = float(grp["mule_score"].mean())
            if cluster_risk < CLUSTER_RISK_THRESHOLD:
                continue   # benign community — skip

            member_accounts = grp["account_id"].tolist()
            if len(member_accounts) < MIN_RING_SIZE:
                continue   # too small to be a meaningful ring

            is_novel_cluster = bool(grp["is_novel"].any())

            # Pattern: majority vote among suspicious nodes
            susp_preds = grp.loc[grp["mule_score"] >= 0.5, "pred_class"].values
            if len(susp_preds) == 0:
                susp_preds = grp["pred_class"].values

            if is_novel_cluster:
                pattern    = "UNKNOWN_SUSPICIOUS"
                confidence = None
            else:
                counter    = Counter(susp_preds.tolist())
                top_cls, top_cnt = counter.most_common(1)[0]
                pattern    = CLASS_NAMES[top_cls]
                if pattern == "benign":
                    pattern    = "UNKNOWN_SUSPICIOUS"
                    confidence = None
                else:
                    confidence = round(top_cnt / len(susp_preds), 4)

            # Total amount blocked
            total_blocked = float(
                acc_amounts.reindex(member_accounts).fillna(0).sum()
            )

            # Transaction IDs involved
            txn_ids = []
            seen_txn = set()
            for acc in member_accounts:
                for tid in txn_ids_by_account.get(acc, []):
                    if isinstance(tid, list):
                        for t in tid:
                            if t not in seen_txn:
                                seen_txn.add(t); txn_ids.append(t)
                    elif tid not in seen_txn:
                        seen_txn.add(tid); txn_ids.append(tid)

            rings_out.append({
                "ring_id":              _next_ring_id(),
                "accounts":             member_accounts,
                "account_count":        len(member_accounts),
                "pattern":              pattern,
                "risk_score":           round(cluster_risk, 4),
                "is_novel":             is_novel_cluster,
                "confidence":           confidence,
                "total_amount_blocked": round(total_blocked, 2),
                "transaction_ids":      txn_ids,
                "flagged_accounts":     [a for a in member_accounts
                                         if a in seed_set],
                "detected_at":          detected_at,
            })

        rings_out.sort(key=lambda x: x["risk_score"], reverse=True)
        return rings_out
