"""
run_production_demo.py — End-to-end AML Production Pipeline Demo

Simulates the full pipeline:
  1. Transactions stream in at 7,000/sec (simulated from aml_dataset)
  2. XGBoost flags suspicious accounts (using suspicion-model.pkl)
  3. TransactionStore expands to 2-hop chains
  4. AMLProductionEngine runs GNN → outputs rings

Usage
-----
    cd c:\Users\SAKTHIVEL R\Desktop\set
    python run_production_demo.py

    # Options
    python run_production_demo.py --batches 10   # run 10 batches
    python run_production_demo.py --no-xgb       # use label column instead of XGBoost
"""

import os, sys, time, json, argparse, pickle
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.abspath(__file__))
GNN    = os.path.join(ROOT, "aml_gnn")
XGBDIR = os.path.join(ROOT, "XGBOOST")
sys.path.insert(0, GNN)

from transaction_store     import TransactionStore
from production_inference  import AMLProductionEngine

# ── XGBoost feature columns (from Model Card) ─────────────────────────────────
XGB_FEATURE_COLS = [
    "in_txn_5m", "out_txn_5m", "in_amt_5m", "out_amt_5m",
    "avg_tx_gap", "recv_send_gap", "pct_forwarded_60s", "in_out_ratio",
    "uniq_senders_10m", "uniq_receivers_10m", "new_counterparty_pct",
    "region_count", "pass_through_ratio", "fanin_burst", "fanout_burst",
    "sender_region_risk", "receiver_region_risk", "corridor_multiplier",
]
XGB_THRESHOLD = 0.30   # from Model Card


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Feature Engineering
# Computes 18 behavioral features per account from a raw transaction window.
# ─────────────────────────────────────────────────────────────────────────────

def compute_xgboost_features(txn_window: pd.DataFrame) -> pd.DataFrame:
    """
    Compute XGBoost's 18 behavioral features for each account in the window.
    Returns DataFrame with columns: account_id + XGB_FEATURE_COLS.
    """
    if txn_window.empty:
        return pd.DataFrame()

    df = txn_window.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ts_unix"]   = df["timestamp"].astype(np.int64) // 10 ** 9

    all_accounts = pd.unique(
        np.concatenate([df["sender_account_id"].values,
                        df["receiver_account_id"].values])
    )

    rows = []
    for acc in all_accounts:
        sent = df[df["sender_account_id"] == acc]
        recv = df[df["receiver_account_id"] == acc]

        in_txn_5m  = len(recv)
        out_txn_5m = len(sent)
        in_amt_5m  = float(recv["amount"].sum())
        out_amt_5m = float(sent["amount"].sum())

        # avg time gap between consecutive transactions
        all_ts = sorted(
            df[df["sender_account_id"].eq(acc) |
               df["receiver_account_id"].eq(acc)]["ts_unix"].tolist()
        )
        if len(all_ts) >= 2:
            gaps = [all_ts[i+1] - all_ts[i] for i in range(len(all_ts)-1)]
            avg_tx_gap = float(np.mean(gaps))
        else:
            avg_tx_gap = 0.0

        # recv_send_gap: time between receiving and re-sending
        if len(recv) > 0 and len(sent) > 0:
            last_recv_ts   = float(recv["ts_unix"].max())
            sent_after     = sent[sent["ts_unix"] >= last_recv_ts]["ts_unix"]
            recv_send_gap  = float(sent_after.min() - last_recv_ts) if len(sent_after) else 9999.0
        else:
            recv_send_gap  = 9999.0

        # pct_forwarded_60s
        fwd_count = 0
        for recv_ts in recv["ts_unix"].values:
            if any((sent["ts_unix"] >= recv_ts) &
                   (sent["ts_unix"] <= recv_ts + 60)):
                fwd_count += 1
        pct_forwarded_60s = fwd_count / max(1, in_txn_5m)

        in_out_ratio        = in_txn_5m / max(1, out_txn_5m)
        uniq_senders_10m    = int(recv["sender_account_id"].nunique())
        uniq_receivers_10m  = int(sent["receiver_account_id"].nunique())
        new_counterparty_pct= 1.0   # all are "new" without history
        region_count        = int(pd.unique(
            np.concatenate([
                sent["receiver_pincode"].astype(str).values if len(sent) else [],
                recv["sender_pincode"].astype(str).values if len(recv) else [],
            ])
        ).shape[0])
        pass_through_ratio  = out_amt_5m / max(1.0, in_amt_5m)
        fanin_burst         = int(in_txn_5m > 5)
        fanout_burst        = int(out_txn_5m > 5)
        sender_region_risk  = 0.15   # India domestic default
        receiver_region_risk= 0.15
        corridor_multiplier = 1.0

        rows.append({
            "account_id":           acc,
            "in_txn_5m":            in_txn_5m,
            "out_txn_5m":           out_txn_5m,
            "in_amt_5m":            in_amt_5m,
            "out_amt_5m":           out_amt_5m,
            "avg_tx_gap":           avg_tx_gap,
            "recv_send_gap":        recv_send_gap,
            "pct_forwarded_60s":    pct_forwarded_60s,
            "in_out_ratio":         in_out_ratio,
            "uniq_senders_10m":     uniq_senders_10m,
            "uniq_receivers_10m":   uniq_receivers_10m,
            "new_counterparty_pct": new_counterparty_pct,
            "region_count":         region_count,
            "pass_through_ratio":   pass_through_ratio,
            "fanin_burst":          fanin_burst,
            "fanout_burst":         fanout_burst,
            "sender_region_risk":   sender_region_risk,
            "receiver_region_risk": receiver_region_risk,
            "corridor_multiplier":  corridor_multiplier,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main Demo
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(n_batches: int = 5, use_xgboost: bool = True,
             batch_size_txns: int = 3_500):
    """
    Simulate n_batches × 500ms windows of streaming transactions.

    Parameters
    ----------
    n_batches        : number of 500ms windows to simulate
    use_xgboost      : if True, run XGBoost for flagging; else use risk_label
    batch_size_txns  : transactions per 500ms window (7k/sec → 3.5k per 500ms)
    """
    print("\n" + "═" * 70)
    print("  AML Production Pipeline Demo")
    print("  XGBoost (Layer 1) → Transaction Expansion → GNN (Layer 2)")
    print("═" * 70)

    # ── Load XGBoost model ────────────────────────────────────────────────────
    xgb_model = None
    xgb_path  = os.path.join(XGBDIR, "suspicion-model.pkl")
    if use_xgboost and os.path.exists(xgb_path):
        print("\n  Loading XGBoost model...")
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        print(f"  XGBoost ready  (threshold={XGB_THRESHOLD})")
    else:
        print("\n  [INFO] Using risk_label column as XGBoost proxy")
        use_xgboost = False

    # ── Load GNN engine ───────────────────────────────────────────────────────
    print()
    engine = AMLProductionEngine().load()

    # ── Load transaction dataset ──────────────────────────────────────────────
    print("\n  Loading AML transaction dataset...")
    txn_path = os.path.join(ROOT, "aml_dataset", "transactions.csv")
    full_df  = pd.read_csv(txn_path, parse_dates=["timestamp"])
    # Use test split for demo
    demo_df  = full_df[full_df["split"] == "test"].reset_index(drop=True)
    print(f"  Test-split transactions: {len(demo_df):,}")

    # ── Prime the TransactionStore ────────────────────────────────────────────
    store = TransactionStore()
    print("\n  Priming transaction store (all test transactions)...")
    store.ingest(demo_df)
    print(f"  {store}")

    # ── Simulation loop ───────────────────────────────────────────────────────
    print(f"\n  Simulating {n_batches} × 500ms batches "
          f"(~{batch_size_txns:,} txns each)...\n")
    print("─" * 70)

    all_rings      = []
    total_start    = time.time()
    cursor         = 0

    for batch_num in range(1, n_batches + 1):
        t0 = time.time()

        # Grab the next batch of transactions
        batch_end  = min(cursor + batch_size_txns, len(demo_df))
        batch_df   = demo_df.iloc[cursor:batch_end].copy()
        cursor     = batch_end
        if batch_df.empty:
            print("  [INFO] No more transactions — rewinding")
            cursor = 0
            continue

        # ── Layer 1: XGBoost ──────────────────────────────────────────────
        if use_xgboost:
            feat_df    = compute_xgboost_features(batch_df)
            if feat_df.empty:
                continue
            X_xgb      = feat_df[XGB_FEATURE_COLS].values
            probs      = xgb_model.predict_proba(X_xgb)[:, 1]
            susp_mask  = probs >= XGB_THRESHOLD
            susp_accts = feat_df.loc[susp_mask, "account_id"].tolist()
        else:
            # Use ground truth as XGBoost proxy
            susp_clusters = batch_df[batch_df["risk_label"] == 1]["cluster_id"].unique()
            susp_accts    = pd.unique(
                np.concatenate([
                    batch_df[batch_df["cluster_id"].isin(susp_clusters)]["sender_account_id"].values,
                    batch_df[batch_df["cluster_id"].isin(susp_clusters)]["receiver_account_id"].values,
                ])
            ).tolist()

        if not susp_accts:
            print(f"  Batch {batch_num:02d} │ No suspicious accounts flagged")
            continue

        # ── Expand: pull 2-hop chain from store ──────────────────────────
        related_df = store.get_related_transactions(susp_accts, hops=2)

        # ── Layer 2: GNN ──────────────────────────────────────────────────
        rings = engine.process_batch(susp_accts, related_df)
        elapsed = time.time() - t0

        # ── Print results ─────────────────────────────────────────────────
        print(f"  Batch {batch_num:02d} │ txns={len(batch_df):,}  "
              f"flagged={len(susp_accts):,}  "
              f"related={len(related_df):,}  "
              f"rings={len(rings)}  "
              f"latency={elapsed*1000:.0f}ms")

        for ring in rings[:5]:   # print top 5 per batch
            novel_tag  = " [NOVEL]" if ring["is_novel"] else ""
            conf_tag   = f" conf={ring['confidence']:.2f}" if ring["confidence"] else ""
            print(f"    ▶ {ring['ring_id']}  "
                  f"pattern={ring['pattern']}{novel_tag}{conf_tag}  "
                  f"accounts={ring['account_count']}  "
                  f"risk={ring['risk_score']:.3f}  "
                  f"blocked=₹{ring['total_amount_blocked']:>14,.2f}")
        if len(rings) > 5:
            print(f"    … {len(rings)-5} more rings")

        all_rings.extend(rings)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("\n" + "═" * 70)
    print("  PIPELINE SUMMARY")
    print("═" * 70)

    if all_rings:
        total_blocked = sum(r["total_amount_blocked"] for r in all_rings)
        novel_count   = sum(1 for r in all_rings if r["is_novel"])
        pattern_dist  = {}
        for r in all_rings:
            pattern_dist[r["pattern"]] = pattern_dist.get(r["pattern"], 0) + 1

        print(f"  Total rings detected      : {len(all_rings)}")
        print(f"  Total amount blocked      : ₹{total_blocked:,.2f}")
        print(f"  Novel (unknown) patterns  : {novel_count}")
        print(f"  Total elapsed             : {total_elapsed:.2f}s")
        print(f"\n  Pattern breakdown:")
        for pat, cnt in sorted(pattern_dist.items(), key=lambda x: -x[1]):
            print(f"    {pat:<30}: {cnt}")

        # Save output
        out_path = os.path.join(ROOT, "models", "production_rings.json")
        with open(out_path, "w") as f:
            json.dump(all_rings[:500], f, indent=2, default=str)
        print(f"\n  Ring report saved → {out_path}")
    else:
        print("  No rings detected in demo batches.")
    print("═" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Production Pipeline Demo")
    parser.add_argument("--batches",  type=int,  default=5,
                        help="Number of 500ms simulation batches (default=5)")
    parser.add_argument("--no-xgb",  action="store_true",
                        help="Use risk_label instead of running XGBoost")
    parser.add_argument("--batch-size", type=int, default=3_500,
                        help="Transactions per batch (default=3500)")
    args = parser.parse_args()

    run_demo(
        n_batches      = args.batches,
        use_xgboost    = not args.no_xgb,
        batch_size_txns= args.batch_size,
    )
