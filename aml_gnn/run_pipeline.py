"""
run_pipeline.py — Single entry point for the full AML GNN pipeline.

Steps:
  1. build_graph  → graph_cache/graph_data.pt
  2. train        → models/best_model.pth + centroids + history
  3. inference    → models/test_cluster_report.csv + .json

Usage:
  python run_pipeline.py              # full pipeline
  python run_pipeline.py --skip-build # skip graph build if cache exists
  python run_pipeline.py --infer-only # only run inference (model must exist)
"""
import sys, os

# Make sure sibling imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

def main():
    parser = argparse.ArgumentParser(description="AML GNN Pipeline")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip graph building if cache exists")
    parser.add_argument("--infer-only", action="store_true",
                        help="Only run inference (model must already be trained)")
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  AML GNN Pipeline — Cross-Channel Mule Detection")
    print("═" * 65 + "\n")

    # ── Step 1: Build graph ────────────────────────────────────────────────────
    if not args.infer_only:
        print("━━━ STEP 1: Graph Construction ━━━")
        from build_graph import build_graph
        force = not args.skip_build
        data, meta = build_graph(force_rebuild=force)
        print()

        # ── Step 2: Train ─────────────────────────────────────────────────────
        print("━━━ STEP 2: Model Training ━━━")
        from train import train
        train()
        print()

    # ── Step 3: Inference on test split ───────────────────────────────────────
    print("━━━ STEP 3: Inference & Cluster Report ━━━")
    from inference import run_on_test_split
    results = run_on_test_split()

    print("\n" + "═" * 65)
    print("  PIPELINE COMPLETE")
    print("═" * 65)
    print("\n  Output files:")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "models")
    model_dir = os.path.normpath(model_dir)
    for fn in ["best_model.pth", "centroids.npy", "ood_threshold.npy",
               "training_history.csv",
               "test_cluster_report.csv", "test_cluster_report.json"]:
        p = os.path.join(model_dir, fn)
        if os.path.exists(p):
            sz = os.path.getsize(p) / 1024
            print(f"    {fn:<35} {sz:>8.1f} KB")
    print()


if __name__ == "__main__":
    main()
