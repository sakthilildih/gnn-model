"""
inference.py — Production inference engine
Input  : DataFrame of transactions for a set of flagged accounts (from Redis)
Output : List of cluster dicts with:
    cluster_id, accounts, account_count, pattern,
    risk_score, is_novel_pattern, confidence,
    total_amount_blocked
"""
import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
import community as community_louvain
from collections import Counter, defaultdict

from config import (
    MODEL_DIR, FEATURE_COLS, CLASS_NAMES, NUM_CLASSES,
    HIDDEN_DIM, DROPOUT, OOD_MIN_RISK_SCORE, CLUSTER_RISK_THRESHOLD,
)
from model import AMLGraphSAGE


# ── Load model artefacts once (call at startup) ───────────────────────────────

def load_model_artifacts(meta: dict):
    """
    Returns (model, centroids, ood_threshold, feature_mean, feature_std)
    meta: the graph_meta.pkl dict saved during build_graph
    """
    # Model
    in_channels = len(meta["feature_cols"])
    model = AMLGraphSAGE(
        in_channels=in_channels,
        hidden_channels=HIDDEN_DIM,
        out_channels=NUM_CLASSES,
        dropout=DROPOUT,
    )
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model.pth"),
                   map_location="cpu", weights_only=True)
    )
    model.eval()

    centroids     = np.load(os.path.join(MODEL_DIR, "centroids.npy"))
    ood_threshold = float(np.load(os.path.join(MODEL_DIR, "ood_threshold.npy"))[0])
    feature_mean  = np.array(meta["feature_mean"], dtype=np.float32)
    feature_std   = np.array(meta["feature_std"],  dtype=np.float32)

    return model, centroids, ood_threshold, feature_mean, feature_std


# ── Node feature extraction (mirrors build_graph.py) ─────────────────────────

def extract_node_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a transactions DataFrame for a suspicious subgraph,
    return a node_df with all FEATURE_COLS (un-normalised).
    """
    df = df.copy()
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
        .reset_index().rename(columns={"sender_account_id": "account_id"})
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
        .reset_index().rename(columns={"receiver_account_id": "account_id"})
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
    node_df["out_ratio"]           = node_df["sent_count"] / (node_df["total_degree"] + 1e-8)
    node_df["in_ratio"]            = node_df["recv_count"] / (node_df["total_degree"] + 1e-8)
    node_df["cross_channel_score"] = (node_df["sent_unique_types"] +
                                       node_df["recv_unique_types"]) / 2
    node_df["atm_ratio"]           = node_df["sent_atm_cnt"] / (node_df["sent_count"] + 1e-8)
    node_df["wallet_ratio"]        = node_df["sent_wallet_cnt"] / (node_df["sent_count"] + 1e-8)
    node_df["unique_pincodes"]     = (node_df["sent_unique_pins"] +
                                       node_df["recv_unique_pins"])
    time_span_min = (node_df["ts_max"] - node_df["ts_min"]) / 60.0 + 1
    node_df["burst_velocity"]      = node_df["total_degree"] / time_span_min
    node_df["amount_range"]        = node_df["sent_max_amt"] - node_df["sent_avg_amt"]

    # Store total amount for blocked-amount computation
    sent_totals = df.groupby("sender_account_id")["amount"].sum()
    recv_totals = df.groupby("receiver_account_id")["amount"].sum()
    node_df["total_amount_linked"] = (
        node_df["account_id"].map(sent_totals).fillna(0) +
        node_df["account_id"].map(recv_totals).fillna(0)
    )

    return node_df


# ── OOD detection ─────────────────────────────────────────────────────────────

def ood_score(emb: np.ndarray, centroids: np.ndarray) -> tuple:
    """
    Returns (min_distance, nearest_class_idx).
    If min_distance > ood_threshold → UNKNOWN_SUSPICIOUS.
    """
    dists     = np.linalg.norm(centroids - emb, axis=1)  # [NUM_CLASSES]
    nearest_c = int(dists.argmin())
    return float(dists[nearest_c]), nearest_c


# ── Core inference function ───────────────────────────────────────────────────

def run_inference(
    transactions_df: pd.DataFrame,
    model: AMLGraphSAGE,
    centroids: np.ndarray,
    ood_threshold: float,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> list[dict]:
    """
    Parameters
    ----------
    transactions_df : DataFrame with columns matching transactions.csv schema
                      (only contains transactions of flagged accounts)
    Returns
    -------
    List of cluster dicts — one per discovered community.
    """
    if transactions_df.empty:
        return []

    # ── 1. Node features ──────────────────────────────────────────────────────
    node_df = extract_node_features(transactions_df)

    # Feature col "community_id_norm" is added after Louvain below
    pre_louvain_cols = [c for c in FEATURE_COLS if c != "community_id_norm"]

    # ── 2. Build graph & run Louvain ──────────────────────────────────────────
    accounts      = node_df["account_id"].tolist()
    acc_to_idx    = {acc: i for i, acc in enumerate(accounts)}
    n_nodes       = len(accounts)

    src_idx = transactions_df["sender_account_id"].map(acc_to_idx).dropna().astype(int).values
    dst_idx = transactions_df["receiver_account_id"].map(acc_to_idx).dropna().astype(int).values

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(src_idx.tolist(), dst_idx.tolist()))

    partition = community_louvain.best_partition(G, random_state=42)
    comm_arr  = np.array([partition.get(i, 0) for i in range(n_nodes)], dtype=np.float32)
    max_comm  = comm_arr.max() + 1
    node_df["community_id_norm"] = comm_arr / max_comm

    # ── 3. Normalise features & build edge_index tensor ───────────────────────
    X         = node_df[FEATURE_COLS].values.astype(np.float32)
    X_norm    = (X - feature_mean) / feature_std
    x_tensor  = torch.tensor(X_norm, dtype=torch.float)
    ei_tensor = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

    # ── 4. Forward pass ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        mule_probs, pred_classes, embs = model.mule_score(x_tensor, ei_tensor)

    mule_probs_np  = mule_probs.numpy()
    pred_classes_np = pred_classes.numpy()
    embs_np         = embs.numpy()

    # ── 5. OOD detection per node ─────────────────────────────────────────────
    is_novel = np.zeros(n_nodes, dtype=bool)
    for i in range(n_nodes):
        if mule_probs_np[i] >= OOD_MIN_RISK_SCORE:
            dist, _ = ood_score(embs_np[i], centroids)
            if dist > ood_threshold:
                is_novel[i] = True

    # ── 6. Aggregate per Louvain community ────────────────────────────────────
    # Map node index → community
    node_df["louvain_community"] = comm_arr.astype(int)
    node_df["mule_score"]        = mule_probs_np
    node_df["pred_class"]        = pred_classes_np
    node_df["is_novel"]          = is_novel

    # Transaction amounts per account (for blocked amount)
    acc_amounts = (
        transactions_df
        .groupby("sender_account_id")["amount"].sum()
        .add(transactions_df.groupby("receiver_account_id")["amount"].sum(), fill_value=0)
    )

    clusters_out = []
    for comm_id, grp in node_df.groupby("louvain_community"):
        cluster_risk = float(grp["mule_score"].mean())
        if cluster_risk < CLUSTER_RISK_THRESHOLD:
            continue   # skip benign cluster

        # Pattern: majority vote among predicted classes (excluding benign=0)
        susp_preds = grp.loc[grp["mule_score"] >= 0.5, "pred_class"].values
        if len(susp_preds) == 0:
            susp_preds = grp["pred_class"].values

        is_novel_cluster = bool(grp["is_novel"].any())

        if is_novel_cluster:
            pattern    = "UNKNOWN_SUSPICIOUS"
            confidence = None
        else:
            counter       = Counter(susp_preds.tolist())
            top_class     = counter.most_common(1)[0]
            pattern       = CLASS_NAMES[top_class[0]]
            if pattern == "benign":
                pattern = "UNKNOWN_SUSPICIOUS"
                confidence = None
            else:
                confidence = round(top_class[1] / len(susp_preds), 4)

        # Total amount blocked = sum of all transaction amounts in this cluster
        member_accounts      = grp["account_id"].tolist()
        cluster_txns         = transactions_df[
            transactions_df["sender_account_id"].isin(member_accounts) |
            transactions_df["receiver_account_id"].isin(member_accounts)
        ]
        total_amount_blocked = float(cluster_txns["amount"].sum())

        clusters_out.append({
            "cluster_id":           f"INFER_{comm_id:05d}",
            "accounts":             member_accounts,
            "account_count":        len(member_accounts),
            "pattern":              pattern,
            "risk_score":           round(cluster_risk, 4),
            "is_novel_pattern":     is_novel_cluster,
            "confidence":           confidence,
            "total_amount_blocked": round(total_amount_blocked, 2),
        })

    clusters_out.sort(key=lambda x: x["risk_score"], reverse=True)
    return clusters_out


# ── Convenience: run on test set from saved graph ─────────────────────────────

def run_on_test_split():
    """
    Loads transactions.csv test-split rows, runs inference,
    saves test_cluster_report.csv.
    """
    import json

    # Load graph meta
    meta_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "graph_cache", "graph_meta.pkl"
    )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "aml_dataset"
    )
    print("Loading test transactions...")
    df  = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    test_df = df[df["split"] == "test"].copy()
    print(f"  Test rows: {len(test_df):,}")

    print("Loading model artefacts...")
    model, centroids, ood_threshold, f_mean, f_std = load_model_artifacts(meta)

    print("Running inference on test split...")
    results = run_inference(test_df, model, centroids, ood_threshold, f_mean, f_std)

    print(f"\n{'─'*70}")
    print(f"  Suspicious clusters found : {len(results)}")
    if results:
        total_blocked = sum(r["total_amount_blocked"] for r in results)
        novel_count   = sum(1 for r in results if r["is_novel_pattern"])
        print(f"  Total amount blocked      : ₹{total_blocked:,.2f}")
        print(f"  Novel (unknown) patterns  : {novel_count}")
        print(f"\n  Top 5 clusters by risk:")
        for r in results[:5]:
            print(f"    {r['cluster_id']} | accounts={r['account_count']:>3} "
                  f"| pattern={r['pattern']:<30} "
                  f"| risk={r['risk_score']:.3f} "
                  f"| blocked=₹{r['total_amount_blocked']:>14,.2f}")
    print(f"{'─'*70}")

    # Save report CSVs
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models"
    )
    report_df = pd.DataFrame([{
        "cluster_id":           r["cluster_id"],
        "account_count":        r["account_count"],
        "accounts":             "|".join(r["accounts"]),
        "pattern":              r["pattern"],
        "risk_score":           r["risk_score"],
        "is_novel_pattern":     r["is_novel_pattern"],
        "confidence":           r["confidence"],
        "total_amount_blocked": r["total_amount_blocked"],
    } for r in results])
    csv_path = os.path.join(model_dir, "test_cluster_report.csv")
    report_df.to_csv(csv_path, index=False)
    print(f"\n  Cluster report saved → {csv_path}")

    # Also save pretty JSON
    json_path = os.path.join(model_dir, "test_cluster_report.json")
    with open(json_path, "w") as f:
        json.dump(results[:200], f, indent=2)   # first 200 for readability
    print(f"  JSON report saved     → {json_path}")

    return results


if __name__ == "__main__":
    run_on_test_split()
