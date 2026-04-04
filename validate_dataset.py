"""Validates generated AML dataset and writes a readable generation report."""
import pandas as pd
import json
import os

OUTPUT_DIR = "aml_dataset"

df  = pd.read_csv(os.path.join(OUTPUT_DIR, "transactions.csv"))
dc  = pd.read_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"))
with open(os.path.join(OUTPUT_DIR, "metadata.json")) as f:
    meta = json.load(f)

lines = []
def p(s=""):
    print(s)
    lines.append(s)

p("=" * 60)
p("  AML DATASET GENERATION REPORT & VALIDATION")
p("=" * 60)
p()
p(f"Total transaction rows  : {len(df):,}")
p(f"Total clusters          : {len(dc):,}")
p(f"Unique cluster IDs      : {df['cluster_id'].nunique():,}")
p()

p("--- Schema columns ---")
p(", ".join(df.columns.tolist()))
p()

p("--- Pattern family distribution (clusters) ---")
pf_counts = dc["pattern_family"].value_counts()
for pf, cnt in pf_counts.items():
    p(f"  {pf:<35}: {cnt:,}")
p()

p("--- Risk label distribution (transactions) ---")
rl = df["risk_label"].value_counts()
for k, v in rl.items():
    label = "suspicious" if k == 1 else "benign"
    p(f"  label={k} ({label}): {v:,}")
p()

p("--- Transaction type distribution ---")
for k, v in df["transaction_type"].value_counts().items():
    p(f"  {k:<10}: {v:,}")
p()

p("--- Split distribution (clusters) ---")
for split, cnt in dc["split"].value_counts().items():
    txns = df[df["split"] == split].shape[0]
    p(f"  {split:<8}: {cnt:>6,} clusters  |  {txns:>10,} transactions")
p()

# No-leakage check
split_check = df.groupby("cluster_id")["split"].nunique()
leaky = int((split_check > 1).sum())
p(f"Clusters with split leakage : {leaky}  (should be 0)")
p()

# Hybrid check
susp    = dc[dc["risk_label"] == 1]
hybrids = dc[dc["pattern_family"].str.startswith("hybrid")]
hybrid_pct = len(hybrids) / len(susp) * 100
p(f"Suspicious clusters : {len(susp):,}")
p(f"Hybrid clusters     : {len(hybrids):,}")
p(f"Hybrid ratio        : {hybrid_pct:.1f}%  (minimum 35% required)")
p()

p("[VALIDATION RESULTS]")
ok = True
if len(dc) != 40_000:
    p(f"  FAIL: cluster count = {len(dc):,}, expected 40,000")
    ok = False
else:
    p(f"  PASS: cluster count = {len(dc):,} == 40,000")

if hybrid_pct < 35.0:
    p(f"  FAIL: hybrid ratio {hybrid_pct:.1f}% < 35%")
    ok = False
else:
    p(f"  PASS: hybrid ratio {hybrid_pct:.1f}% >= 35%")

if leaky > 0:
    p(f"  FAIL: {leaky} clusters have split leakage")
    ok = False
else:
    p("  PASS: No cluster split leakage")

required_cols = [
    "transaction_id", "timestamp", "amount", "transaction_type",
    "sender_account_id", "receiver_account_id",
    "sender_pincode", "receiver_pincode",
    "sender_pan_hash", "receiver_pan_hash",
    "cluster_id", "pattern_family", "risk_label",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    p(f"  FAIL: missing columns {missing}")
    ok = False
else:
    p("  PASS: All required columns present")

p()
if ok:
    p("  ALL VALIDATIONS PASSED")
else:
    p("  ONE OR MORE VALIDATIONS FAILED")
p("=" * 60)
p()

# Write report
report_path = os.path.join(OUTPUT_DIR, "generation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nReport written to: {report_path}")

# File sizes
print("\n--- Output file sizes ---")
for fn in ["transactions.csv", "cluster_summary.csv", "metadata.json", "generation_report.txt"]:
    p2 = os.path.join(OUTPUT_DIR, fn)
    if os.path.exists(p2):
        sz = os.path.getsize(p2) / 1e6
        print(f"  {fn:<35}: {sz:>8.2f} MB")
