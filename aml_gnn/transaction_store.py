"""
transaction_store.py — Rolling-window in-memory transaction store.

In the production pipeline:
  1. Every raw transaction (7k/sec) is ingested here via ingest().
  2. When XGBoost flags suspicious account_ids, call get_related_transactions()
     to pull the full 2-hop chain of connected transactions for GNN analysis.

Thread-safe. Prunes old transactions when capacity is exceeded.
"""

import threading
from collections import defaultdict, deque
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TXN_STORE_MAX_TXN

# Required columns in every ingested transaction row
REQUIRED_COLS = [
    "transaction_id", "timestamp", "amount", "transaction_type",
    "sender_account_id", "receiver_account_id",
    "sender_pincode", "receiver_pincode",
]


class TransactionStore:
    """
    In-memory, thread-safe rolling window store for raw transactions.

    Usage
    -----
    store = TransactionStore()

    # Feed the raw stream (call every batch, e.g. every 100ms)
    store.ingest(batch_df)

    # After XGBoost flags accounts:
    related_txns = store.get_related_transactions(
        seed_account_ids=["ACC001", "ACC002"],
        hops=2
    )
    """

    def __init__(self, max_transactions: int = TXN_STORE_MAX_TXN):
        self._lock          = threading.RLock()
        self._max           = max_transactions
        # account_id → deque of transaction dicts (ordered by ingestion time)
        self._by_account    = defaultdict(list)
        # ordered log of all transactions for pruning
        self._all_txns      = deque()
        self._total         = 0

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, txn_df: pd.DataFrame) -> None:
        """
        Add a batch of transactions to the store.

        Parameters
        ----------
        txn_df : DataFrame with at least the REQUIRED_COLS columns.
        """
        if txn_df.empty:
            return

        missing = [c for c in REQUIRED_COLS if c not in txn_df.columns]
        if missing:
            raise ValueError(f"TransactionStore.ingest: missing columns {missing}")

        with self._lock:
            for rec in txn_df[REQUIRED_COLS + [c for c in txn_df.columns
                                                if c not in REQUIRED_COLS]
                               ].to_dict("records"):
                self._all_txns.append(rec)
                self._by_account[rec["sender_account_id"]].append(rec)
                self._by_account[rec["receiver_account_id"]].append(rec)
                self._total += 1

            if self._total > self._max:
                self._prune()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_related_transactions(
        self,
        seed_account_ids: list,
        hops: int = 2,
    ) -> pd.DataFrame:
        """
        BFS from seed accounts to find the full N-hop transaction chain.

        Parameters
        ----------
        seed_account_ids : list of account IDs flagged by XGBoost.
        hops             : how many hops to expand (2 = 2nd-degree connections).

        Returns
        -------
        DataFrame of all related transactions (deduplicated).
        """
        if not seed_account_ids:
            return pd.DataFrame()

        with self._lock:
            visited  = set(seed_account_ids)
            frontier = set(seed_account_ids)

            for _ in range(hops):
                new_frontier = set()
                for acc in frontier:
                    for txn in self._by_account.get(acc, []):
                        for nbr in (txn["sender_account_id"],
                                    txn["receiver_account_id"]):
                            if nbr not in visited:
                                visited.add(nbr)
                                new_frontier.add(nbr)
                frontier = new_frontier
                if not frontier:
                    break

            # Collect all transactions involving any visited account
            seen_ids = set()
            rows     = []
            for acc in visited:
                for txn in self._by_account.get(acc, []):
                    tid = txn.get("transaction_id")
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        rows.append(txn)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Bulk load (for demo/testing) ──────────────────────────────────────────

    def load_from_csv(self, csv_path: str, chunksize: int = 50_000) -> None:
        """Load historical transactions from a CSV file (for demo purposes)."""
        print(f"  Loading transactions from {csv_path}...")
        for chunk in pd.read_csv(csv_path, chunksize=chunksize,
                                  parse_dates=["timestamp"]):
            self.ingest(chunk)
        print(f"  Store loaded: {self._total:,} transactions, "
              f"{len(self._by_account):,} unique accounts")

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune(self) -> None:
        """Remove oldest 10% of transactions when capacity is exceeded."""
        n_remove = max(1, self._max // 10)
        removed  = 0
        while removed < n_remove and self._all_txns:
            old = self._all_txns.popleft()
            removed += 1
            self._total -= 1
            # Remove from account index (slow but correct for demo)
            for acc_key in ("sender_account_id", "receiver_account_id"):
                acc = old.get(acc_key)
                if acc and acc in self._by_account:
                    try:
                        self._by_account[acc].remove(old)
                    except ValueError:
                        pass

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._total

    @property
    def n_accounts(self) -> int:
        return len(self._by_account)

    def __repr__(self) -> str:
        return (f"TransactionStore(transactions={self._total:,}, "
                f"accounts={len(self._by_account):,})")
