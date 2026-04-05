"""
run_pipeline.py — Single entry point for the full AML GNN pipeline.

Steps:
  1. build_graph  → graph_cache/graph_data.pt  (+ louvain_max saved in meta)
  2. train        → models/best_model.pth + centroids + history
  3. inference    → models/test_cluster_report.csv + .json

Usage:
  python run_pipeline.py                # full rebuild + retrain + infer
  python run_pipeline.py --skip-build   # skip graph build if cache exists
  python run_pipeline.py --infer-only   # only inference (model must exist)
  python run_pipeline.py --test         # run test_model.py after training
  python run_pipeline.py --demo         # run production pipeline demo

Production pipeline entry point:
  cd .. && python run_production_demo.py
"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="AML GNN Pipeline")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip graph build if cache exists")
    parser.add_argument("--infer-only", action="store_true",
                        help="Only run inference (model must already be trained)")
    parser.add_argument("--test",  action="store_true",
                        help="Run test_model.py after training to verify pattern detection")
    parser.add_argument("--demo",  action="store_true",
                        help="Run production pipeline demo after training")
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
        print(f"  louvain_max_communities = {meta.get('louvain_max_communities', 'N/A')}\n")

        # ── Step 2: Train ─────────────────────────────────────────────────
        print("━━━ STEP 2: Model Training ━━━")
        from train import train
        train()
        print()

    # ── Step 3: Inference on test split ───────────────────────────────────────
    print("━━━ STEP 3: Inference & Cluster Report ━━━")
    from inference import run_on_test_split
    results = run_on_test_split()

    # ── Step 4 (optional): Pattern detection test ─────────────────────────────
    if args.test:
        print("\n━━━ STEP 4: Pattern Detection Test ━━━")
        test_py = os.path.join(ROOT, "test_model.py")
        subprocess.run([sys.executable, test_py], cwd=ROOT, check=True)

    # ── Step 5 (optional): Production demo ───────────────────────────────────
    if args.demo:
        print("\n━━━ STEP 5: Production Pipeline Demo ━━━")
        demo_py = os.path.join(ROOT, "run_production_demo.py")
        subprocess.run([sys.executable, demo_py, "--no-xgb"], cwd=ROOT, check=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  PIPELINE COMPLETE")
    print("═" * 65)
    print("\n  Output files:")
    model_dir = os.path.normpath(os.path.join(ROOT, "models"))
    for fn in ["best_model.pth", "centroids.npy", "ood_threshold.npy",
               "training_history.csv",
               "test_cluster_report.csv", "test_cluster_report.json",
               "production_rings.json"]:
        p = os.path.join(model_dir, fn)
        if os.path.exists(p):
            sz = os.path.getsize(p) / 1024
            print(f"    {fn:<35} {sz:>8.1f} KB")
    print()


if __name__ == "__main__":
    main()
