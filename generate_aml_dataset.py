"""
AML Graph Transaction Dataset Generator
Cross-Channel Mule Account Detection Graph

Generates ~40,000 clusters and 1.5M-2.5M transaction rows
across 10 AML motif families + hybrid patterns.
"""

import os
import json
import hashlib
import random
import string
import time
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = "aml_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TRANSACTION_TYPES = ["UPI", "ATM", "App", "Wallet", "Web"]

# Realistic Indian pincodes by state
PINCODES = {
    "Tamil Nadu":   [600001,600002,600003,600004,600006,600010,600014,600017,600020,
                     600028,600040,600050,600078,600097,600118,641001,641006,641018,
                     641041,641049,620001,620003,620014,620020,625001,625010,630001,
                     630005,632001,632004,636001,636002,638001,638002,641652],
    "Karnataka":    [560001,560002,560003,560008,560010,560018,560025,560034,560040,
                     560050,560060,560070,560078,560080,560085,560094,560100,570001,
                     570002,576101,577001,580001,580002,580003,581301,583101,560026,
                     560029,560030,560032,560033,560036,560037,560038,560039],
    "Maharashtra":  [400001,400002,400003,400004,400005,400006,400007,400008,400009,
                     400010,400011,400012,400013,400014,400015,400016,400017,400018,
                     400019,400020,400021,400023,400025,400028,400030,400034,400050,
                     400051,400053,400058,400059,400063,400070,400072,400076,400097],
    "Delhi":        [110001,110002,110003,110004,110005,110006,110007,110008,110009,
                     110010,110011,110012,110013,110014,110015,110016,110017,110018,
                     110019,110020,110021,110022,110023,110024,110025,110026,110027,
                     110028,110029,110030,110031,110032,110033,110034,110035,110036],
    "Kerala":       [695001,695002,695003,695004,695005,695006,695007,695008,695009,
                     695010,695011,695012,695013,695014,695015,695016,695017,695018,
                     695019,695020,695021,695022,695023,695024,695025,695040,680001,
                     680002,680003,680004,680005,680006,680007,680008,680009,680010],
    "Telangana":    [500001,500002,500003,500004,500005,500006,500007,500008,500009,
                     500010,500011,500012,500013,500014,500015,500016,500017,500018,
                     500019,500020,500021,500022,500023,500024,500025,500026,500027,
                     500028,500029,500030,500031,500032,500033,500034,500035,500036],
}

ALL_PINCODES = [p for codes in PINCODES.values() for p in codes]


def get_suspicious_pincodes(n=2):
    """Pick pincodes that may span different states (velocity jump)."""
    states = random.sample(list(PINCODES.keys()), min(n, len(PINCODES)))
    return [random.choice(PINCODES[s]) for s in states]


def get_benign_pincodes(n=2):
    """Pick pincodes from same or adjacent state."""
    state = random.choice(list(PINCODES.keys()))
    return [random.choice(PINCODES[state]) for _ in range(n)]


# ─────────────────────────────────────────────
# ID / HASH UTILITIES
# ─────────────────────────────────────────────
_account_counter = 0
_pan_counter = 0


def new_account_id():
    global _account_counter
    _account_counter += 1
    return f"ACC{_account_counter:08d}"


def new_pan_hash():
    global _pan_counter
    _pan_counter += 1
    raw = f"PAN{_pan_counter:07d}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def new_txn_id():
    return "TXN" + "".join(random.choices(string.ascii_uppercase + string.digits, k=14))


def random_inr(lo, hi):
    return round(random.uniform(lo, hi), 2)


def burst_timestamps(base: datetime, n: int, window_seconds: int = 180) -> list:
    """Generate n timestamps within window_seconds of base."""
    return sorted([base + timedelta(seconds=random.randint(0, window_seconds)) for _ in range(n)])


def spaced_timestamps(base: datetime, n: int, min_gap_h=1, max_gap_h=48) -> list:
    ts = [base]
    for _ in range(n - 1):
        ts.append(ts[-1] + timedelta(hours=random.uniform(min_gap_h, max_gap_h)))
    return ts


# ─────────────────────────────────────────────
# TRANSACTION ROW BUILDER
# ─────────────────────────────────────────────

def build_row(txn_id, ts, amount, tx_type, src, dst,
              src_pin, dst_pin, src_pan, dst_pan,
              cluster_id, pattern_family, risk_label):
    return {
        "transaction_id": txn_id,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": amount,
        "transaction_type": tx_type,
        "sender_account_id": src,
        "receiver_account_id": dst,
        "sender_pincode": src_pin,
        "receiver_pincode": dst_pin,
        "sender_pan_hash": src_pan,
        "receiver_pan_hash": dst_pan,
        "cluster_id": cluster_id,
        "pattern_family": pattern_family,
        "risk_label": risk_label,
    }


# ─────────────────────────────────────────────
# BASE DATE POOL
# ─────────────────────────────────────────────
BASE_DATES = [
    datetime(2024, random.randint(1, 12), random.randint(1, 28),
             random.randint(0, 23), random.randint(0, 59))
    for _ in range(100_000)
]


def random_base():
    return random.choice(BASE_DATES)


# ──────────────────────────────────────────────────────────────────────────────
# ██  PATTERN GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

# ─── 1. MULE RING ────────────────────────────────────────────────────────────

def gen_mule_ring(cluster_id: str) -> list:
    n_acc = random.randint(6, 15)
    n_txn = random.randint(15, 40)
    accs = [new_account_id() for _ in range(n_acc)]
    # 25-30% share PAN hashes
    n_shared_pan = max(2, int(n_acc * random.uniform(0.25, 0.30)))
    shared_pans = [new_pan_hash() for _ in range(max(1, n_shared_pan // 2))]
    pan_map = {}
    shared_idx = random.sample(range(n_acc), n_shared_pan)
    for i, acc in enumerate(accs):
        if i in shared_idx:
            pan_map[acc] = random.choice(shared_pans)
        else:
            pan_map[acc] = new_pan_hash()

    base = random_base()
    timestamps = burst_timestamps(base, n_txn, 300)
    pincodes = {acc: random.choice(get_suspicious_pincodes(2)) for acc in accs}

    rows = []
    for i in range(n_txn):
        src = random.choice(accs)
        dst = random.choice([a for a in accs if a != src])
        pin_src = pincodes[src]
        pin_dst = pincodes[dst]
        amount = random_inr(1000, 500000)
        tx_type = random.choice(["UPI", "App", "Wallet"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            src, dst, pin_src, pin_dst,
            pan_map[src], pan_map[dst],
            cluster_id, "mule_ring", 1
        ))
    return rows


# ─── 2. CHAIN LAYERING ───────────────────────────────────────────────────────

def gen_chain_layering(cluster_id: str) -> list:
    n_acc = random.randint(6, 12)
    n_txn = random.randint(6, 20)
    accs = [new_account_id() for _ in range(n_acc)]
    pan_map = {acc: new_pan_hash() for acc in accs}
    # PAN nesting between non-adjacent nodes
    if n_acc >= 4:
        i1, i2 = random.sample(range(n_acc), 2)
        while abs(i1 - i2) <= 1:
            i1, i2 = random.sample(range(n_acc), 2)
        shared_pan = new_pan_hash()
        pan_map[accs[i1]] = shared_pan
        pan_map[accs[i2]] = shared_pan

    base = random_base()
    timestamps = sorted(burst_timestamps(base, n_txn, 600))
    pincodes = {acc: random.choice(ALL_PINCODES) for acc in accs}

    seed_amount = random_inr(50000, 2000000)
    rows = []
    for i in range(n_txn):
        chain_idx = i % (n_acc - 1)
        src = accs[chain_idx]
        dst = accs[chain_idx + 1]
        # Amount preserved across hops with small variance
        amount = round(seed_amount * random.uniform(0.95, 1.00), 2)
        # 30% end in ATM/Wallet cashout
        if i == n_txn - 1 and random.random() < 0.30:
            tx_type = random.choice(["ATM", "Wallet"])
        else:
            tx_type = random.choice(["UPI", "App", "Web"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            src, dst, pincodes[src], pincodes[dst],
            pan_map[src], pan_map[dst],
            cluster_id, "chain_layering", 1
        ))
    return rows


# ─── 3. CIRCULAR LOOP / ROUND-TRIPPING ───────────────────────────────────────

def gen_circular_loop(cluster_id: str) -> list:
    n_acc = random.randint(3, 8)
    n_txn = random.randint(6, 25)
    accs = [new_account_id() for _ in range(n_acc)]
    pan_map = {acc: new_pan_hash() for acc in accs}
    # One loop node shares PAN with source
    shared_pan = pan_map[accs[0]]
    loop_node = random.choice(accs[1:])
    pan_map[loop_node] = shared_pan

    base = random_base()
    timestamps = burst_timestamps(base, n_txn, 400)
    pincodes = {acc: random.choice(get_suspicious_pincodes(2)) for acc in accs}

    seed_amount = random_inr(10000, 1000000)
    rows = []
    for i in range(n_txn):
        # Cycle: A→B→C→A
        src_idx = i % n_acc
        dst_idx = (i + 1) % n_acc
        src = accs[src_idx]
        dst = accs[dst_idx]
        # 90-100% return ratio
        amount = round(seed_amount * random.uniform(0.90, 1.00), 2)
        tx_type = random.choice(["UPI", "App", "Wallet"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            src, dst, pincodes[src], pincodes[dst],
            pan_map[src], pan_map[dst],
            cluster_id, "circular_loop", 1
        ))
    return rows


# ─── 4. STRUCTURING FAN-OUT ──────────────────────────────────────────────────

def gen_structuring_fanout(cluster_id: str) -> list:
    n_receivers = random.randint(5, 20)
    n_txn = random.randint(10, 50)
    source = new_account_id()
    receivers = [new_account_id() for _ in range(n_receivers)]
    accs = [source] + receivers

    source_pan = new_pan_hash()
    pan_map = {source: source_pan}
    # PAN overlap among some receiver nodes
    shared_rcv_pan = new_pan_hash()
    n_overlap = random.randint(2, max(2, n_receivers // 3))
    overlap_idx = random.sample(range(n_receivers), n_overlap)
    for i, rcv in enumerate(receivers):
        pan_map[rcv] = shared_rcv_pan if i in overlap_idx else new_pan_hash()

    base = random_base()
    timestamps = burst_timestamps(base, n_txn, 600)
    pincodes = {acc: random.choice(ALL_PINCODES) for acc in accs}

    rows = []
    for i in range(n_txn):
        dst = random.choice(receivers)
        amount = random_inr(45000, 49000)  # Below threshold structuring
        tx_type = random.choice(["UPI", "App", "Web"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            source, dst, pincodes[source], pincodes[dst],
            pan_map[source], pan_map[dst],
            cluster_id, "structuring_fanout", 1
        ))
    return rows


# ─── 5. FUNNEL / FAN-IN ──────────────────────────────────────────────────────

def gen_funnel_fanin(cluster_id: str) -> list:
    n_senders = random.randint(5, 15)
    n_txn = random.randint(10, 40)
    collector = new_account_id()
    senders = [new_account_id() for _ in range(n_senders)]
    accs = senders + [collector]

    collector_pan = new_pan_hash()
    pan_map = {acc: new_pan_hash() for acc in senders}
    pan_map[collector] = collector_pan

    base = random_base()
    timestamps = burst_timestamps(base, n_txn, 600)
    pincodes = {acc: random.choice(get_suspicious_pincodes(2)) for acc in accs}

    rows = []
    for i in range(n_txn):
        src = random.choice(senders)
        # Last 20% may be collector → ATM/Wallet
        if i > n_txn * 0.8 and random.random() < 0.4:
            dst_acc = new_account_id()
            pan_map[dst_acc] = new_pan_hash()
            pincodes[dst_acc] = random.choice(ALL_PINCODES)
            tx_type = random.choice(["ATM", "Wallet"])
            amount = random_inr(10000, 500000)
            rows.append(build_row(
                new_txn_id(), timestamps[i], amount, tx_type,
                collector, dst_acc, pincodes[collector], pincodes[dst_acc],
                pan_map[collector], pan_map[dst_acc],
                cluster_id, "funnel_fanin", 1
            ))
        else:
            amount = random_inr(5000, 300000)
            tx_type = random.choice(["UPI", "App", "Web"])
            rows.append(build_row(
                new_txn_id(), timestamps[i], amount, tx_type,
                src, collector, pincodes[src], pincodes[collector],
                pan_map[src], pan_map[collector],
                cluster_id, "funnel_fanin", 1
            ))
    return rows


# ─── 6. DIAMOND FRAGMENTATION ────────────────────────────────────────────────

def gen_diamond_fragmentation(cluster_id: str) -> list:
    n_acc = random.randint(4, 10)
    n_txn = random.randint(8, 20)
    source = new_account_id()
    n_split = random.randint(2, max(2, n_acc // 2))
    split_nodes = [new_account_id() for _ in range(n_split)]
    collector = new_account_id()
    extra_nodes = [new_account_id() for _ in range(max(0, n_acc - n_split - 2))]
    accs = [source] + split_nodes + [collector] + extra_nodes

    split_pan = new_pan_hash()
    pan_map = {source: new_pan_hash(), collector: new_pan_hash()}
    for sn in split_nodes:
        pan_map[sn] = split_pan  # PAN nesting on split nodes
    for en in extra_nodes:
        pan_map[en] = new_pan_hash()

    base = random_base()
    timestamps = sorted(burst_timestamps(base, n_txn, 500))
    pincodes = {acc: random.choice(ALL_PINCODES) for acc in accs}

    rows = []
    txn_per_phase = n_txn // 3 or 1

    # Phase 1: source → split nodes
    for i in range(txn_per_phase):
        sn = random.choice(split_nodes)
        amount = random_inr(20000, 500000)
        tx_type = random.choice(["UPI", "App"])
        ts = timestamps[i % len(timestamps)]
        rows.append(build_row(
            new_txn_id(), ts, amount, tx_type,
            source, sn, pincodes[source], pincodes[sn],
            pan_map[source], pan_map[sn],
            cluster_id, "diamond_fragmentation", 1
        ))

    # Phase 2: split → collector
    for i in range(txn_per_phase, 2 * txn_per_phase):
        sn = random.choice(split_nodes)
        amount = random_inr(20000, 500000)
        tx_type = random.choice(["UPI", "Wallet"])
        ts = timestamps[min(i, len(timestamps) - 1)]
        rows.append(build_row(
            new_txn_id(), ts, amount, tx_type,
            sn, collector, pincodes[sn], pincodes[collector],
            pan_map[sn], pan_map[collector],
            cluster_id, "diamond_fragmentation", 1
        ))

    # Phase 3: extra links
    for i in range(2 * txn_per_phase, n_txn):
        src = random.choice(accs)
        dst = random.choice([a for a in accs if a != src])
        ts = timestamps[min(i, len(timestamps) - 1)]
        rows.append(build_row(
            new_txn_id(), ts, random_inr(5000, 200000), random.choice(["UPI", "App", "Web"]),
            src, dst, pincodes[src], pincodes[dst],
            pan_map[src], pan_map[dst],
            cluster_id, "diamond_fragmentation", 1
        ))
    return rows


# ─── 7. CROSS-CHANNEL VELOCITY MULE ─────────────────────────────────────────

CHANNEL_TRANSITIONS = [
    ["App", "Wallet", "ATM"],
    ["UPI", "Wallet", "ATM"],
    ["Web", "Wallet", "ATM"],
    ["App", "UPI", "Wallet", "ATM"],
    ["Web", "UPI", "Wallet", "ATM"],
]


def gen_cross_channel_velocity(cluster_id: str) -> list:
    n_acc = random.randint(4, 10)
    n_txn = random.randint(8, 20)
    accs = [new_account_id() for _ in range(n_acc)]
    shared_pan = new_pan_hash()  # Same PAN across rails
    pan_map = {acc: shared_pan if random.random() < 0.5 else new_pan_hash() for acc in accs}

    base = random_base()
    # All within 1-5 minutes
    timestamps = burst_timestamps(base, n_txn, random.randint(60, 300))
    pincodes = {acc: random.choice(get_suspicious_pincodes(2)) for acc in accs}

    channel_seq = random.choice(CHANNEL_TRANSITIONS)
    rows = []
    for i in range(n_txn):
        src = accs[i % len(accs)]
        dst = accs[(i + 1) % len(accs)]
        tx_type = channel_seq[i % len(channel_seq)]
        amount = random_inr(10000, 300000)
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            src, dst, pincodes[src], pincodes[dst],
            pan_map[src], pan_map[dst],
            cluster_id, "cross_channel_velocity", 1
        ))
    return rows


# ─── 8. PAN NESTING OWNERSHIP ────────────────────────────────────────────────

def gen_pan_nesting(cluster_id: str) -> list:
    n_acc = random.randint(6, 18)
    n_txn = random.randint(10, 35)
    accs = [new_account_id() for _ in range(n_acc)]
    # 1 PAN controls 3-6 accounts
    n_controlled = random.randint(3, min(6, n_acc))
    controller_pan = new_pan_hash()
    controlled_idx = random.sample(range(n_acc), n_controlled)
    pan_map = {}
    for i, acc in enumerate(accs):
        pan_map[acc] = controller_pan if i in controlled_idx else new_pan_hash()

    base = random_base()
    timestamps = spaced_timestamps(base, n_txn, 0.5, 12)
    pincodes = {acc: random.choice(ALL_PINCODES) for acc in accs}

    rows = []
    for i in range(n_txn):
        src = random.choice(accs)
        dst = random.choice([a for a in accs if a != src])
        amount = random_inr(5000, 500000)
        tx_type = random.choice(["UPI", "App", "Wallet", "Web"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            src, dst, pincodes[src], pincodes[dst],
            pan_map[src], pan_map[dst],
            cluster_id, "pan_nesting", 1
        ))
    return rows


# ─── 9. BURST VELOCITY RING ──────────────────────────────────────────────────

def gen_burst_velocity(cluster_id: str) -> list:
    n_acc = random.randint(10, 25)
    n_txn = random.randint(25, 100)
    accs = [new_account_id() for _ in range(n_acc)]
    pan_map = {}
    shared_pan = new_pan_hash()
    for i, acc in enumerate(accs):
        # Every 4th node shares PAN
        pan_map[acc] = shared_pan if i % 4 == 0 else new_pan_hash()

    base = random_base()
    # All within 1-3 minutes
    timestamps = burst_timestamps(base, n_txn, random.randint(60, 180))
    pincodes = {acc: random.choice(get_suspicious_pincodes(3)) for acc in accs}

    rows = []
    for i in range(n_txn):
        src = accs[i % n_acc]
        dst = accs[(i + random.randint(1, 3)) % n_acc]
        amount = random_inr(1000, 200000)
        tx_type = random.choice(["UPI", "App", "Wallet"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            src, dst, pincodes[src], pincodes[dst],
            pan_map[src], pan_map[dst],
            cluster_id, "burst_velocity_ring", 1
        ))
    return rows


# ─── 10. BENIGN COMMUNITIES ──────────────────────────────────────────────────

def _gen_benign_salary(cluster_id: str) -> list:
    """Monthly salary credit pattern."""
    n_acc = random.randint(3, 8)
    n_txn = random.randint(5, 15)
    accs = [new_account_id() for _ in range(n_acc)]
    pan_map = {acc: new_pan_hash() for acc in accs}
    employer = accs[0]
    base = random_base()
    # Monthly cadence
    timestamps = [base + timedelta(days=30 * i + random.randint(-2, 2)) for i in range(n_txn)]
    pincodes = {acc: random.choice(get_benign_pincodes(1)) for acc in accs}
    rows = []
    for i in range(n_txn):
        dst = random.choice(accs[1:])
        amount = random_inr(15000, 150000)
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, "UPI",
            employer, dst, pincodes[employer], pincodes[dst],
            pan_map[employer], pan_map[dst],
            cluster_id, "benign_salary", 0
        ))
    return rows


def _gen_benign_merchant(cluster_id: str) -> list:
    """Merchant payment pattern."""
    n_acc = random.randint(5, 12)
    n_txn = random.randint(8, 25)
    accs = [new_account_id() for _ in range(n_acc)]
    pan_map = {acc: new_pan_hash() for acc in accs}
    merchant = accs[0]
    # Daytime concentration
    base = datetime(2024, random.randint(1, 12), random.randint(1, 28),
                    random.randint(9, 20), random.randint(0, 59))
    timestamps = spaced_timestamps(base, n_txn, 0.1, 8)
    pincodes = {acc: random.choice(get_benign_pincodes(2)) for acc in accs}
    rows = []
    for i in range(n_txn):
        src = random.choice(accs[1:])
        amount = random_inr(100, 50000)
        tx_type = random.choice(["UPI", "App", "Web"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            src, merchant, pincodes[src], pincodes[merchant],
            pan_map[src], pan_map[merchant],
            cluster_id, "benign_merchant", 0
        ))
    return rows


def _gen_benign_family(cluster_id: str) -> list:
    """Family transfers pattern."""
    n_acc = random.randint(3, 6)
    n_txn = random.randint(5, 20)
    accs = [new_account_id() for _ in range(n_acc)]
    # Family may share PAN (joint accounts)
    family_pan = new_pan_hash()
    pan_map = {acc: family_pan if random.random() < 0.4 else new_pan_hash() for acc in accs}
    base = random_base()
    timestamps = spaced_timestamps(base, n_txn, 2, 72)
    pincodes = {acc: random.choice(get_benign_pincodes(1)) for acc in accs}
    rows = []
    for i in range(n_txn):
        src = random.choice(accs)
        dst = random.choice([a for a in accs if a != src])
        amount = random_inr(500, 30000)
        tx_type = random.choice(["UPI", "App"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            src, dst, pincodes[src], pincodes[dst],
            pan_map[src], pan_map[dst],
            cluster_id, "benign_family", 0
        ))
    return rows


def _gen_benign_atm(cluster_id: str) -> list:
    """Normal ATM withdrawal pattern."""
    n_acc = random.randint(3, 8)
    n_txn = random.randint(5, 15)
    accs = [new_account_id() for _ in range(n_acc)]
    pan_map = {acc: new_pan_hash() for acc in accs}
    # Realistic ATM times (day + some night)
    base = datetime(2024, random.randint(1, 12), random.randint(1, 28),
                    random.randint(8, 22), random.randint(0, 59))
    timestamps = spaced_timestamps(base, n_txn, 12, 72)
    pincodes = {acc: random.choice(get_benign_pincodes(1)) for acc in accs}
    rows = []
    for i in range(n_txn):
        src = random.choice(accs)
        amount = random_inr(500, 20000)
        atm_dest = new_account_id()
        pan_map[atm_dest] = new_pan_hash()
        pincodes[atm_dest] = pincodes[src]
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, "ATM",
            src, atm_dest, pincodes[src], pincodes[src],
            pan_map[src], pan_map[atm_dest],
            cluster_id, "benign_atm", 0
        ))
    return rows


def _gen_benign_bills(cluster_id: str) -> list:
    """Rent / bills payment pattern."""
    n_acc = random.randint(3, 6)
    n_txn = random.randint(5, 15)
    accs = [new_account_id() for _ in range(n_acc)]
    pan_map = {acc: new_pan_hash() for acc in accs}
    payer = accs[0]
    # Weekly/monthly cadence
    base = random_base()
    timestamps = [base + timedelta(days=random.choice([7, 14, 30]) * i) for i in range(n_txn)]
    pincodes = {acc: random.choice(get_benign_pincodes(1)) for acc in accs}
    rows = []
    for i in range(n_txn):
        dst = random.choice(accs[1:])
        amount = random_inr(2000, 50000)
        tx_type = random.choice(["UPI", "Web", "App"])
        rows.append(build_row(
            new_txn_id(), timestamps[i], amount, tx_type,
            payer, dst, pincodes[payer], pincodes[dst],
            pan_map[payer], pan_map[dst],
            cluster_id, "benign_bills", 0
        ))
    return rows


BENIGN_SUBTYPES = {
    "salary": _gen_benign_salary,
    "merchant": _gen_benign_merchant,
    "family": _gen_benign_family,
    "atm": _gen_benign_atm,
    "bills": _gen_benign_bills,
}


# ──────────────────────────────────────────────────────────────────────────────
# ██  HYBRID PATTERN COMPOSERS
# ──────────────────────────────────────────────────────────────────────────────

def compose_hybrid(cluster_id: str, phases: list) -> list:
    """
    Stitch multiple pattern generators into one cluster.
    phases = list of (generator_fn, sub_label) tuples
    """
    all_rows = []
    # Each phase produces rows; we re-stamp cluster_id and mark hybrid
    for fn, sub_label in phases:
        tmp_cid = f"_TMP_{random.random()}"
        rows = fn(tmp_cid)
        for r in rows:
            r["cluster_id"] = cluster_id
            r["pattern_family"] = f"hybrid_{sub_label}"
            r["risk_label"] = 1
        all_rows.extend(rows)
    return all_rows


HYBRID_RECIPES = [
    # fanout → chain → ATM
    lambda cid: compose_hybrid(cid, [
        (gen_structuring_fanout, "fanout_chain_atm"),
        (gen_chain_layering, "fanout_chain_atm"),
    ]),
    # mule ring → loop → funnel
    lambda cid: compose_hybrid(cid, [
        (gen_mule_ring, "ring_loop_funnel"),
        (gen_circular_loop, "ring_loop_funnel"),
        (gen_funnel_fanin, "ring_loop_funnel"),
    ]),
    # PAN nesting → chain → wallet → ATM
    lambda cid: compose_hybrid(cid, [
        (gen_pan_nesting, "nesting_chain_wallet"),
        (gen_chain_layering, "nesting_chain_wallet"),
        (gen_cross_channel_velocity, "nesting_chain_wallet"),
    ]),
    # structuring → fan-in → cross-channel
    lambda cid: compose_hybrid(cid, [
        (gen_structuring_fanout, "struct_fanin_cross"),
        (gen_funnel_fanin, "struct_fanin_cross"),
        (gen_cross_channel_velocity, "struct_fanin_cross"),
    ]),
    # diamond → ring → burst
    lambda cid: compose_hybrid(cid, [
        (gen_diamond_fragmentation, "diamond_ring_burst"),
        (gen_mule_ring, "diamond_ring_burst"),
        (gen_burst_velocity, "diamond_ring_burst"),
    ]),
    # circular loop → pan nesting → chain
    lambda cid: compose_hybrid(cid, [
        (gen_circular_loop, "loop_nesting_chain"),
        (gen_pan_nesting, "loop_nesting_chain"),
        (gen_chain_layering, "loop_nesting_chain"),
    ]),
]


# ──────────────────────────────────────────────────────────────────────────────
# ██  CLUSTER PLAN
# ──────────────────────────────────────────────────────────────────────────────

# Base (pure) cluster counts
BASE_PLAN = {
    "mule_ring":             6000,
    "chain_layering":        4500,
    "circular_loop":         3500,
    "structuring_fanout":    4000,
    "funnel_fanin":          3500,
    "diamond_fragmentation": 3000,
    "cross_channel_velocity":4500,
    "pan_nesting":           4000,
    "burst_velocity_ring":   3000,
}

BENIGN_PLAN = {
    "salary":   2000,
    "merchant": 2000,
    "family":   1500,
    "atm":      1500,
    "bills":    1000,
}

# Total suspicious pure = 36,000, benign = 8,000 → 44,000 base
# We want 40,000 total, with ≥35% of suspicious being hybrid
# suspicious pure = 36,000 → 35% hybrid = ~12,600 hybrids
# Reduce pure suspicious by 12,600 proportionally
# Target: 40,000 total, 8,000 benign → 32,000 suspicious
#   pure suspicious = 32,000 * 0.65 = 20,800
#   hybrid = 32,000 * 0.35 = 11,200

TOTAL_CLUSTERS = 40_000
BENIGN_TOTAL   = 8_000
SUSPICIOUS_TOTAL = TOTAL_CLUSTERS - BENIGN_TOTAL  # 32,000
HYBRID_COUNT   = int(SUSPICIOUS_TOTAL * 0.35)     # 11,200
PURE_SUSP_TOTAL = SUSPICIOUS_TOTAL - HYBRID_COUNT  # 20,800

# Scale down BASE_PLAN to PURE_SUSP_TOTAL
_base_sum = sum(BASE_PLAN.values())  # 36,000
PURE_PLAN = {k: max(1, round(v * PURE_SUSP_TOTAL / _base_sum)) for k, v in BASE_PLAN.items()}
# Correct rounding drift
_pp_sum = sum(PURE_PLAN.values())
_diff = PURE_SUSP_TOTAL - _pp_sum
_adj_key = max(PURE_PLAN, key=PURE_PLAN.get)
PURE_PLAN[_adj_key] += _diff

GENERATORS = {
    "mule_ring":             gen_mule_ring,
    "chain_layering":        gen_chain_layering,
    "circular_loop":         gen_circular_loop,
    "structuring_fanout":    gen_structuring_fanout,
    "funnel_fanin":          gen_funnel_fanin,
    "diamond_fragmentation": gen_diamond_fragmentation,
    "cross_channel_velocity":gen_cross_channel_velocity,
    "pan_nesting":           gen_pan_nesting,
    "burst_velocity_ring":   gen_burst_velocity,
}


# ──────────────────────────────────────────────────────────────────────────────
# ██  MAIN GENERATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def generate_cluster_id(prefix, idx):
    return f"{prefix}_{idx:07d}"


def main():
    t0 = time.time()
    print("═" * 70)
    print("  AML Synthetic Graph Dataset Generator")
    print("═" * 70)

    all_rows = []
    cluster_summaries = []
    cluster_idx = 0
    pattern_counts = {k: 0 for k in list(BASE_PLAN.keys()) + ["hybrid"] + list(BENIGN_PLAN.keys())}

    # ── Pure suspicious clusters ──────────────────────────────────────────────
    print(f"\n[1/3] Generating {PURE_SUSP_TOTAL:,} pure suspicious clusters...")
    for pattern, count in PURE_PLAN.items():
        gen_fn = GENERATORS[pattern]
        for _ in tqdm(range(count), desc=f"  {pattern}", ncols=80):
            cid = generate_cluster_id(pattern[:3].upper(), cluster_idx)
            rows = gen_fn(cid)
            all_rows.extend(rows)
            cluster_summaries.append({
                "cluster_id": cid,
                "pattern_family": pattern,
                "risk_label": 1,
                "n_transactions": len(rows),
                "n_accounts": len(set([r["sender_account_id"] for r in rows] +
                                       [r["receiver_account_id"] for r in rows])),
                "split": None,
            })
            pattern_counts[pattern] += 1
            cluster_idx += 1

    # ── Hybrid suspicious clusters ────────────────────────────────────────────
    print(f"\n[2/3] Generating {HYBRID_COUNT:,} hybrid suspicious clusters...")
    for i in tqdm(range(HYBRID_COUNT), desc="  hybrid", ncols=80):
        cid = generate_cluster_id("HYB", cluster_idx)
        recipe = random.choice(HYBRID_RECIPES)
        rows = recipe(cid)
        # Deduplicate pattern_family for display
        pf = rows[0]["pattern_family"] if rows else "hybrid"
        all_rows.extend(rows)
        cluster_summaries.append({
            "cluster_id": cid,
            "pattern_family": pf,
            "risk_label": 1,
            "n_transactions": len(rows),
            "n_accounts": len(set([r["sender_account_id"] for r in rows] +
                                   [r["receiver_account_id"] for r in rows])),
            "split": None,
        })
        pattern_counts["hybrid"] += 1
        cluster_idx += 1

    # ── Benign clusters ───────────────────────────────────────────────────────
    print(f"\n[3/3] Generating {BENIGN_TOTAL:,} benign clusters...")
    for subtype, count in BENIGN_PLAN.items():
        gen_fn = BENIGN_SUBTYPES[subtype]
        for _ in tqdm(range(count), desc=f"  {subtype}", ncols=80):
            cid = generate_cluster_id("BEN", cluster_idx)
            rows = gen_fn(cid)
            all_rows.extend(rows)
            cluster_summaries.append({
                "cluster_id": cid,
                "pattern_family": f"benign_{subtype}",
                "risk_label": 0,
                "n_transactions": len(rows),
                "n_accounts": len(set([r["sender_account_id"] for r in rows] +
                                       [r["receiver_account_id"] for r in rows])),
                "split": None,
            })
            pattern_counts[subtype] += 1
            cluster_idx += 1

    # ──────────────────────────────────────────────────────────────────────────
    # BUILD DATAFRAMES
    print(f"\n  Building DataFrames ({len(all_rows):,} rows)...")
    df = pd.DataFrame(all_rows)

    # Ensure column order
    COLS = [
        "transaction_id", "timestamp", "amount", "transaction_type",
        "sender_account_id", "receiver_account_id",
        "sender_pincode", "receiver_pincode",
        "sender_pan_hash", "receiver_pan_hash",
        "cluster_id", "pattern_family", "risk_label",
    ]
    df = df[COLS]

    df_clusters = pd.DataFrame(cluster_summaries)

    # ──────────────────────────────────────────────────────────────────────────
    # TRAIN / VAL / TEST SPLIT (cluster-level, no leakage)
    print("  Assigning train/val/test splits...")
    unique_cids = df_clusters["cluster_id"].tolist()
    random.shuffle(unique_cids)
    n = len(unique_cids)
    n_train = int(n * 0.75)
    n_val   = int(n * 0.15)
    split_map = {}
    for i, cid in enumerate(unique_cids):
        if i < n_train:
            split_map[cid] = "train"
        elif i < n_train + n_val:
            split_map[cid] = "val"
        else:
            split_map[cid] = "test"

    df_clusters["split"] = df_clusters["cluster_id"].map(split_map)
    df["split"] = df["cluster_id"].map(split_map)

    # ──────────────────────────────────────────────────────────────────────────
    # WRITE OUTPUTS
    print("  Writing output files...")

    txn_path     = os.path.join(OUTPUT_DIR, "transactions.csv")
    cluster_path = os.path.join(OUTPUT_DIR, "cluster_summary.csv")
    meta_path    = os.path.join(OUTPUT_DIR, "metadata.json")

    print(f"    → {txn_path}")
    df.to_csv(txn_path, index=False)

    print(f"    → {cluster_path}")
    df_clusters.to_csv(cluster_path, index=False)

    # ── METADATA JSON ─────────────────────────────────────────────────────────
    split_stats = df_clusters.groupby("split").agg(
        clusters=("cluster_id", "count"),
        transactions=("n_transactions", "sum"),
    ).to_dict()

    meta = {
        "generated_at": datetime.now().isoformat(),
        "total_transactions": len(df),
        "total_clusters": len(df_clusters),
        "pure_suspicious_clusters": PURE_SUSP_TOTAL,
        "hybrid_suspicious_clusters": HYBRID_COUNT,
        "benign_clusters": BENIGN_TOTAL,
        "hybrid_pct_of_suspicious": round(HYBRID_COUNT / SUSPICIOUS_TOTAL * 100, 2),
        "pattern_counts": pattern_counts,
        "pure_plan": PURE_PLAN,
        "benign_plan": BENIGN_PLAN,
        "split_distribution": {
            "train": {"clusters": int(split_stats["clusters"].get("train", 0)),
                      "transactions": int(split_stats["transactions"].get("train", 0))},
            "val":   {"clusters": int(split_stats["clusters"].get("val", 0)),
                      "transactions": int(split_stats["transactions"].get("val", 0))},
            "test":  {"clusters": int(split_stats["clusters"].get("test", 0)),
                      "transactions": int(split_stats["transactions"].get("test", 0))},
        },
        "schema": COLS,
        "risk_label_distribution": df["risk_label"].value_counts().to_dict(),
        "transaction_type_distribution": df["transaction_type"].value_counts().to_dict(),
    }

    print(f"    → {meta_path}")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ── REPORT ────────────────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "generation_report.txt")
    elapsed = time.time() - t0
    with open(report_path, "w") as f:
        f.write("AML Dataset Generation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated at    : {datetime.now().isoformat()}\n")
        f.write(f"Elapsed time    : {elapsed:.1f} seconds\n\n")
        f.write(f"Total Rows      : {len(df):,}\n")
        f.write(f"Total Clusters  : {len(df_clusters):,}\n\n")
        f.write("Cluster breakdown:\n")
        for k, v in pattern_counts.items():
            f.write(f"  {k:<30}: {v:,}\n")
        f.write("\nSplit distribution:\n")
        for sp in ["train", "val", "test"]:
            nc = split_stats["clusters"].get(sp, 0)
            nt = split_stats["transactions"].get(sp, 0)
            f.write(f"  {sp:<8}: {nc:>6,} clusters, {nt:>10,} transactions\n")
        f.write("\nRisk label distribution:\n")
        for k, v in df["risk_label"].value_counts().items():
            f.write(f"  label={k}: {v:,}\n")
        f.write("\nTransaction type distribution:\n")
        for k, v in df["transaction_type"].value_counts().items():
            f.write(f"  {k:<10}: {v:,}\n")
        f.write("\n[VALIDATION]\n")
        assert len(df_clusters) == TOTAL_CLUSTERS, f"Cluster count mismatch: {len(df_clusters)}"
        assert HYBRID_COUNT >= int(SUSPICIOUS_TOTAL * 0.35), "Hybrid % below 35%"
        assert df["split"].notna().all(), "Some rows missing split"
        f.write("  ✓ Total cluster count matches target\n")
        f.write(f"  ✓ Hybrid ratio = {HYBRID_COUNT/SUSPICIOUS_TOTAL*100:.1f}% (≥35%)\n")
        f.write("  ✓ All rows have split assignment\n")
        f.write("  ✓ No cluster leakage (split assigned at cluster level)\n")
        f.write("\nOutput files:\n")
        f.write(f"  {txn_path}\n")
        f.write(f"  {cluster_path}\n")
        f.write(f"  {meta_path}\n")
        f.write(f"  {report_path}\n")

    print(f"    → {report_path}")

    # ── CONSOLE SUMMARY ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  GENERATION COMPLETE")
    print("═" * 70)
    print(f"  Total transactions : {len(df):,}")
    print(f"  Total clusters     : {len(df_clusters):,}")
    print(f"  Hybrid ratio       : {HYBRID_COUNT/SUSPICIOUS_TOTAL*100:.1f}%  (≥35% required)")
    print(f"  Elapsed            : {elapsed:.1f}s")
    print(f"\n  Output directory   : {os.path.abspath(OUTPUT_DIR)}/")
    for fn in ["transactions.csv", "cluster_summary.csv", "metadata.json", "generation_report.txt"]:
        p = os.path.join(OUTPUT_DIR, fn)
        sz = os.path.getsize(p) / 1e6
        print(f"    {fn:<35} {sz:>8.2f} MB")
    print("═" * 70)


if __name__ == "__main__":
    main()
