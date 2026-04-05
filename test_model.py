"""
test_model.py — End-to-end model test on freshly generated mini-dataset
  1. Generates 1 cluster per each of the 10 AML patterns + 2 benign clusters
  2. Runs the trained model inference on it
  3. Prints a side-by-side comparison: GROUND TRUTH vs MODEL PREDICTION
  4. Saves individual cluster images to test_cluster_images/
"""
import os, sys, random, json, pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import community as community_louvain

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aml_gnn"))
from config import (FEATURE_COLS, MODEL_DIR, NUM_CLASSES, CLASS_NAMES,
                    HIDDEN_DIM, DROPOUT, OOD_MIN_RISK_SCORE, CLUSTER_RISK_THRESHOLD)
from model import AMLGraphSAGE

# ── channels ──────────────────────────────────────────────────────────────────
CHANNELS = ["UPI", "ATM", "App", "Wallet", "Web"]  # must match training data
PINS = ["400001", "560001", "500001", "700001", "110001", "600001"]
random.seed(0)
np.random.seed(0)

def rnd_acc():
    return f"TST{random.randint(10000, 99999)}"

def rnd_ts(base=1714000000, spread=300):
    return pd.Timestamp(base + random.randint(0, spread), unit="s")

def rnd_amt(lo, hi):
    return round(random.uniform(lo, hi), 2)

# ── Motif generators ───────────────────────────────────────────────────────────
def gen_mule_ring(cluster_id, n=7):
    accs = [rnd_acc() for _ in range(n)]
    rows = []
    for i in range(n):
        rows.append({"sender_account_id": accs[i],
                     "receiver_account_id": accs[(i+1) % n],
                     "amount": rnd_amt(45000, 49500),
                     "transaction_type": random.choice(["UPI", "App", "Wallet"]),
                     "sender_pincode": random.choice(PINS),
                     "receiver_pincode": random.choice(PINS),
                     "timestamp": rnd_ts()})
    return accs, rows, "mule_ring"

def gen_chain_layering(cluster_id, n=8):
    accs = [rnd_acc() for _ in range(n)]
    rows = []
    for i in range(n - 1):
        rows.append({"sender_account_id": accs[i],
                     "receiver_account_id": accs[i+1],
                      "amount": rnd_amt(300000, 800000),
                      "transaction_type": random.choice(["App", "Web"]),
                      "sender_pincode": PINS[i % len(PINS)],
                      "receiver_pincode": PINS[(i+2) % len(PINS)],
                      "timestamp": rnd_ts(spread=600)})
    return accs, rows, "chain_layering"

def gen_circular_loop(cluster_id, n=5):
    accs = [rnd_acc() for _ in range(n)]
    rows = []
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append({"sender_account_id": accs[i],
                             "receiver_account_id": accs[j],
                             "amount": rnd_amt(10000, 30000),
                             "transaction_type": "UPI",
                             "sender_pincode": PINS[0],
                             "receiver_pincode": PINS[0],
                             "timestamp": rnd_ts(spread=120)})
    return accs, rows, "circular_loop"

def gen_structuring_fanout(cluster_id, n_recv=10):
    source = rnd_acc()
    receivers = [rnd_acc() for _ in range(n_recv)]
    rows = [{"sender_account_id": source,
              "receiver_account_id": r,
              "amount": rnd_amt(45000, 49000),
              "transaction_type": "UPI",
              "sender_pincode": PINS[0],
              "receiver_pincode": random.choice(PINS),
              "timestamp": rnd_ts(spread=30)} for r in receivers]
    return [source] + receivers, rows, "structuring_fanout"

def gen_funnel_fanin(cluster_id, n_send=10):
    dest = rnd_acc()
    senders = [rnd_acc() for _ in range(n_send)]
    rows = [{"sender_account_id": s,
              "receiver_account_id": dest,
               "amount": rnd_amt(20000, 45000),
               "transaction_type": random.choice(["App", "UPI"]),
               "sender_pincode": random.choice(PINS),
               "receiver_pincode": PINS[1],
               "timestamp": rnd_ts(spread=60)} for s in senders]
    return [dest] + senders, rows, "funnel_fanin"

def gen_diamond_fragmentation(cluster_id):
    source = rnd_acc()
    mid1, mid2 = rnd_acc(), rnd_acc()
    dest = rnd_acc()
    rows = [
        {"sender_account_id": source, "receiver_account_id": mid1, "amount": rnd_amt(100000, 200000),
         "transaction_type": "App", "sender_pincode": PINS[0], "receiver_pincode": PINS[1], "timestamp": rnd_ts()},
        {"sender_account_id": source, "receiver_account_id": mid2, "amount": rnd_amt(100000, 200000),
         "transaction_type": "Web", "sender_pincode": PINS[0], "receiver_pincode": PINS[2], "timestamp": rnd_ts()},
        {"sender_account_id": mid1, "receiver_account_id": dest, "amount": rnd_amt(80000, 150000),
         "transaction_type": "UPI", "sender_pincode": PINS[1], "receiver_pincode": PINS[3], "timestamp": rnd_ts()},
        {"sender_account_id": mid2, "receiver_account_id": dest, "amount": rnd_amt(80000, 150000),
         "transaction_type": "Wallet", "sender_pincode": PINS[2], "receiver_pincode": PINS[3], "timestamp": rnd_ts()},
    ]
    return [source, mid1, mid2, dest], rows, "diamond_fragmentation"

def gen_cross_channel_velocity(cluster_id, n=6):
    accs = [rnd_acc() for _ in range(n)]
    channels_seq = ["App", "Wallet", "ATM"]  # must match training channel types
    rows = []
    base_t = 1714000000
    for i in range(n-1):
        rows.append({"sender_account_id": accs[i],
                     "receiver_account_id": accs[i+1],
                     "amount": rnd_amt(10000, 50000),
                     "transaction_type": channels_seq[i % len(channels_seq)],
                     "sender_pincode": PINS[i % len(PINS)],
                     "receiver_pincode": PINS[(i+3) % len(PINS)],
                     "timestamp": pd.Timestamp(base_t + (i * 90), unit="s")})  # 90s apart
    return accs, rows, "cross_channel_velocity"

def gen_pan_nesting(cluster_id, n=8):
    parent = rnd_acc()
    children = [rnd_acc() for _ in range(n)]
    rows = []
    for c in children:
        rows.append({"sender_account_id": parent, "receiver_account_id": c,
                     "amount": rnd_amt(500000, 2000000),
                     "transaction_type": "App",
                     "sender_pincode": PINS[0], "receiver_pincode": random.choice(PINS),
                     "timestamp": rnd_ts(spread=3600)})
        rows.append({"sender_account_id": c, "receiver_account_id": parent,
                     "amount": rnd_amt(100000, 400000),
                     "transaction_type": "UPI",
                     "sender_pincode": random.choice(PINS), "receiver_pincode": PINS[0],
                     "timestamp": rnd_ts(spread=7200)})
    return [parent] + children, rows, "pan_nesting"

def gen_burst_velocity_ring(cluster_id, n=8):
    accs = [rnd_acc() for _ in range(n)]
    rows = []
    base_t = 1714000000
    for _ in range(30):   # many transactions in short time
        i, j = random.sample(range(n), 2)
        rows.append({"sender_account_id": accs[i],
                     "receiver_account_id": accs[j],
                     "amount": rnd_amt(5000, 30000),
                     "transaction_type": random.choice(CHANNELS),
                     "sender_pincode": PINS[i % len(PINS)],
                     "receiver_pincode": PINS[j % len(PINS)],
                     "timestamp": pd.Timestamp(base_t + random.randint(0, 120), unit="s")})
    return accs, rows, "burst_velocity_ring"

def gen_benign(cluster_id, n=6):
    employer = rnd_acc()
    employees = [rnd_acc() for _ in range(n)]
    rows = [{"sender_account_id": employer,
               "receiver_account_id": e,
               "amount": rnd_amt(20000, 60000),
               "transaction_type": "App",
               "sender_pincode": PINS[0],
               "receiver_pincode": random.choice(PINS),
               "timestamp": rnd_ts(spread=86400)} for e in employees]
    return [employer] + employees, rows, "benign"

# ── Build mini test dataset ────────────────────────────────────────────────────
def build_test_data():
    # 4 copies of each pattern gives Louvain richer context and
    # reduces the chance of clusters being merged or misidentified
    pattern_generators = [
        (gen_mule_ring,             "mule_ring"),
        (gen_chain_layering,        "chain_layering"),
        (gen_circular_loop,         "circular_loop"),
        (gen_structuring_fanout,    "structuring_fanout"),
        (gen_funnel_fanin,          "funnel_fanin"),
        (gen_diamond_fragmentation, "diamond_fragmentation"),
        (gen_cross_channel_velocity,"cross_channel_velocity"),
        (gen_pan_nesting,           "pan_nesting"),
        (gen_burst_velocity_ring,   "burst_velocity_ring"),
        (gen_benign,                "benign"),
    ]
    COPIES_PER_PATTERN = 4   # more copies = richer graph for Louvain

    all_txns = []
    ground_truth = []
    txn_id = 1
    cluster_idx = 0

    for gen_fn, pattern_name in pattern_generators:
        for _ in range(COPIES_PER_PATTERN):
            cluster_id = f"TEST_{cluster_idx:03d}"
            accs, rows, pattern = gen_fn(cluster_id)
            for r in rows:
                r["transaction_id"] = f"TSTXN{txn_id:07d}"
                r["cluster_id"] = cluster_id
                txn_id += 1
            all_txns.extend(rows)
            ground_truth.append({"cluster_id": cluster_id, "pattern": pattern, "accounts": accs})
            cluster_idx += 1

    return pd.DataFrame(all_txns), ground_truth

# ── Node features (same as inference.py) ──────────────────────────────────────
def extract_node_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ts_unix"] = df["timestamp"].astype(np.int64) // 10**9

    sent = (df.groupby("sender_account_id").agg(
        sent_count=("transaction_id","count"),
        sent_avg_amt=("amount","mean"), sent_max_amt=("amount","max"),
        sent_std_amt=("amount","std"), sent_unique_types=("transaction_type","nunique"),
        sent_atm_cnt=("transaction_type", lambda x:(x=="ATM").sum()),
        sent_wallet_cnt=("transaction_type", lambda x:(x=="Wallet").sum()),
        sent_unique_pins=("receiver_pincode","nunique"),
        ts_min=("ts_unix","min"), ts_max=("ts_unix","max"),
    ).reset_index().rename(columns={"sender_account_id":"account_id"}))

    recv = (df.groupby("receiver_account_id").agg(
        recv_count=("transaction_id","count"),
        recv_avg_amt=("amount","mean"), recv_max_amt=("amount","max"),
        recv_std_amt=("amount","std"), recv_unique_types=("transaction_type","nunique"),
        recv_unique_pins=("sender_pincode","nunique"),
    ).reset_index().rename(columns={"receiver_account_id":"account_id"}))

    all_accounts = pd.unique(np.concatenate([df["sender_account_id"].values,
                                              df["receiver_account_id"].values]))
    node_df = pd.DataFrame({"account_id": all_accounts})
    node_df = node_df.merge(sent, on="account_id", how="left")
    node_df = node_df.merge(recv, on="account_id", how="left")
    node_df.fillna(0, inplace=True)

    node_df["total_degree"] = node_df["sent_count"] + node_df["recv_count"]
    node_df["out_ratio"] = node_df["sent_count"] / (node_df["total_degree"] + 1e-8)
    node_df["in_ratio"] = node_df["recv_count"] / (node_df["total_degree"] + 1e-8)
    node_df["cross_channel_score"] = (node_df["sent_unique_types"] + node_df["recv_unique_types"]) / 2
    node_df["atm_ratio"] = node_df["sent_atm_cnt"] / (node_df["sent_count"] + 1e-8)
    node_df["wallet_ratio"] = node_df["sent_wallet_cnt"] / (node_df["sent_count"] + 1e-8)
    node_df["unique_pincodes"] = node_df["sent_unique_pins"] + node_df["recv_unique_pins"]
    time_span_min = (node_df["ts_max"] - node_df["ts_min"]) / 60.0 + 1
    node_df["burst_velocity"] = node_df["total_degree"] / time_span_min
    node_df["amount_range"] = node_df["sent_max_amt"] - node_df["sent_avg_amt"]
    return node_df

# ── Image export for one cluster ───────────────────────────────────────────────
def save_cluster_image(sub_df, c_meta, output_path):
    G = nx.from_pandas_edgelist(
        sub_df,
        source="sender_account_id",
        target="receiver_account_id",
        edge_attr=["amount", "transaction_type"],
        create_using=nx.DiGraph()
    )
    n_nodes = len(G.nodes)
    fig_size = max(8, n_nodes * 0.8)
    plt.figure(figsize=(fig_size, fig_size))

    if c_meta["is_novel"]:
        node_color = "#FF8C00"
        border_color = "#cc5500"
    elif c_meta["predicted_pattern"] == "benign":
        node_color = "#2ECC71"
        border_color = "#1a7a43"
    else:
        node_color = "#E74C3C"
        border_color = "#8B0000"

    pos = nx.kamada_kawai_layout(G) if n_nodes <= 20 else nx.spring_layout(G, k=1.2, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_color,
                            edgecolors=border_color, linewidths=2.5)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color="#555555", arrows=True,
                            arrowsize=20, width=1.5,
                            connectionstyle="arc3,rad=0.1")

    edge_labels = {(u, v): f"₹{d['amount']:,.0f}\n{d['transaction_type']}"
                   for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=6, label_pos=0.5,
                                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                            ec="#aaaaaa", alpha=0.85))

    pred = c_meta["predicted_pattern"]
    truth = c_meta["ground_truth"]
    match = "MATCH: YES" if pred == truth or (pred == "UNKNOWN_SUSPICIOUS" and truth != "benign") else "MATCH: NO"
    novel_tag = " [NOVEL OOD]" if c_meta["is_novel"] else ""

    plt.title(
        f"{c_meta['cluster_id']} | Ground Truth: {truth}\n"
        f"Predicted: {pred}{novel_tag} | Risk: {c_meta['risk_score']:.2f} | {match}\n"
        f"Accounts: {c_meta['account_count']} | Blocked: ₹{c_meta['total_amount_blocked']:,.2f}",
        fontsize=10, fontweight="bold"
    )

    # Legend
    patches = [mpatches.Patch(color="#E74C3C", label="Known AML Pattern"),
               mpatches.Patch(color="#FF8C00", label="Novel/OOD Suspicious"),
               mpatches.Patch(color="#2ECC71", label="Benign")]
    plt.legend(handles=patches, loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ── Main test runner ───────────────────────────────────────────────────────────
def run_test():
    OUTPUT_DIR = "test_cluster_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "═"*65)
    print("  AML GNN — Live Model Test on Freshly Generated Data")
    print("═"*65 + "\n")

    # 1. Generate fresh test data
    print("🧪  Generating fresh test clusters (1 per pattern)...")
    df, ground_truth = build_test_data()
    print(f"    {len(df):,} transactions | {len(ground_truth)} clusters\n")

    # 2. Load model + meta
    print("📦  Loading model artefacts...")
    meta_path = os.path.join("graph_cache", "graph_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    feature_mean = np.array(meta["feature_mean"], dtype=np.float32)
    feature_std  = np.array(meta["feature_std"],  dtype=np.float32)
    centroids    = np.load(os.path.join(MODEL_DIR, "centroids.npy"))
    ood_thresh   = float(np.load(os.path.join(MODEL_DIR, "ood_threshold.npy"))[0])

    in_channels = len(meta["feature_cols"])
    model = AMLGraphSAGE(in_channels=in_channels, hidden_channels=HIDDEN_DIM,
                          out_channels=NUM_CLASSES, dropout=DROPOUT)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "best_model.pth"), map_location="cpu", weights_only=True))
    model.eval()

    # 3. Build graph from test data
    print("🔗  Building graph + running Louvain...\n")
    node_df = extract_node_features(df)
    all_accounts = node_df["account_id"].tolist()
    acc_to_idx   = {a: i for i, a in enumerate(all_accounts)}
    n_nodes = len(all_accounts)

    src_idx = df["sender_account_id"].map(acc_to_idx).dropna().astype(int).values
    dst_idx = df["receiver_account_id"].map(acc_to_idx).dropna().astype(int).values

    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n_nodes))
    G_nx.add_edges_from(zip(src_idx.tolist(), dst_idx.tolist()))
    partition = community_louvain.best_partition(G_nx, random_state=42)
    comm_arr = np.array([partition.get(i, 0) for i in range(n_nodes)], dtype=np.float32)
    # Normalize using training-scale community count so the feature
    # distribution at inference matches what the model saw during training
    louvain_max = float(meta.get("louvain_max_communities", comm_arr.max() + 1))
    node_df["community_id_norm"] = comm_arr / louvain_max

    X = node_df[FEATURE_COLS].values.astype(np.float32)
    X_norm = (X - feature_mean) / feature_std
    x_tensor = torch.tensor(X_norm, dtype=torch.float)
    ei_tensor = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

    # 4. Inference
    with torch.no_grad():
        mule_probs, pred_classes, embs = model.mule_score(x_tensor, ei_tensor)

    mule_probs_np   = mule_probs.numpy()
    pred_classes_np = pred_classes.numpy()
    embs_np         = embs.numpy()

    # OOD per node
    is_novel = np.zeros(n_nodes, dtype=bool)
    for i in range(n_nodes):
        if mule_probs_np[i] >= OOD_MIN_RISK_SCORE:
            dists = np.linalg.norm(centroids - embs_np[i], axis=1)
            if dists.min() > ood_thresh:
                is_novel[i] = True

    node_df["mule_score"]  = mule_probs_np
    node_df["pred_class"]  = pred_classes_np
    node_df["is_novel"]    = is_novel

    # 5. Build acc→ground truth map
    gt_map = {acc: (g["pattern"], g["cluster_id"])
               for g in ground_truth for acc in g["accounts"]}

    # 6. Aggregate results per Louvain community
    comm_ids = comm_arr.astype(int)
    node_df["louvain_community"] = comm_ids

    print(f"{'─'*65}")
    print(f"{'Cluster':<14} {'Ground Truth':<30} {'Predicted':<30} {'Risk':>6} {'Result'}")
    print(f"{'─'*65}")

    results = []
    from collections import Counter

    acc_amounts = (df.groupby("sender_account_id")["amount"].sum()
                   .add(df.groupby("receiver_account_id")["amount"].sum(), fill_value=0))

    for comm_id, grp in node_df.groupby("louvain_community"):
        cluster_risk = float(grp["mule_score"].mean())
        susp = grp.loc[grp["mule_score"] >= 0.5, "pred_class"].values
        if len(susp) == 0:
            susp = grp["pred_class"].values

        is_novel_cluster = bool(grp["is_novel"].any())

        if is_novel_cluster:
            predicted = "UNKNOWN_SUSPICIOUS"
        else:
            counter = Counter(susp.tolist())
            top = counter.most_common(1)[0][0]
            predicted = CLASS_NAMES[top]

        members = grp["account_id"].tolist()
        # Ground truth from generated labels
        gt_patterns = [gt_map.get(a, ("unknown", "?"))[0] for a in members]
        truth = Counter(gt_patterns).most_common(1)[0][0]
        g_cluster_id = gt_map.get(members[0], ("?", "COMM_?"))[1]

        total_blocked = float(acc_amounts.reindex(members).fillna(0).sum())
        
        # Evaluation
        correct = (predicted == truth or
                   (is_novel_cluster and truth not in ["benign", "unknown"]))
        result = "✅" if correct else "❌"

        print(f"  comm_{comm_id:<8} {truth:<30} {predicted:<30} {cluster_risk:>5.2f}  {result}")

        # Save image
        sub_df = df[df["sender_account_id"].isin(members) &
                     df["receiver_account_id"].isin(members)]
        c_meta = {
            "cluster_id":           f"COMM_{comm_id}",
            "predicted_pattern":    predicted,
            "ground_truth":         truth,
            "risk_score":           round(cluster_risk, 4),
            "is_novel":             is_novel_cluster,
            "account_count":        len(members),
            "total_amount_blocked": round(total_blocked, 2),
        }
        safe = predicted.replace(" ", "_")
        img_path = os.path.join(OUTPUT_DIR, f"COMM_{comm_id:03d}_{safe}_GT_{truth}.png")
        save_cluster_image(sub_df, c_meta, img_path)
        results.append(c_meta)

    print(f"{'─'*65}")
    correct_count = sum(
        1 for r in results
        if r["predicted_pattern"] == r["ground_truth"] or
           (r["is_novel"] and r["ground_truth"] not in ["benign", "unknown"])
    )
    print(f"\n  Detection accuracy:  {correct_count}/{len(results)} clusters correctly identified")
    print(f"  Images saved to:     .\\{OUTPUT_DIR}\\")
    print(f"\n{'═'*65}\n")


if __name__ == "__main__":
    run_test()
