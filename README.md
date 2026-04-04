# AML Graph Transaction Dataset and GNN Pipeline

This repository contains the complete pipeline for the **Cross-Channel Mule Account Detection** Graph.

The dataset, which simulates 40,000 real-world AML clusters (including hybrid routing, nesting, structuring, etc.), is included.

## Getting Started

1. **Unzip the Dataset:**
   The generated dataset is zipped for storage. Unzip `aml_dataset.zip` into the project root directory. This will create the `aml_dataset/` directory with `transactions.csv` (~220 MB), `cluster_summary.csv`, and other metadata files.
   
2. **Setup virtual environment & requirements:**
   Ensure you have installed:
   - `torch`, `torch-geometric`
   - `pandas`, `numpy`, `networkx`, `python-louvain`

3. **Train the GNN model:**
   Since the dataset is already generated, you can skip dataset generation and directly run the training pipeline:
   ```bash
   python aml_gnn/run_pipeline.py --skip-build
   ```
   *Note: If the `graph_cache/` doesn't exist yet, it'll take ~2 minutes to run Louvain clustering and build the graph before training starts.*

## Architecture

- **Data Generator:** Highly configurable Python engine outputting pure and hybrid AML topologies along realistic payment rails. 
- **Graph Builder (`build_graph.py`):** Loads edges, runs Louvain communities, generates derived features (velocity, cross-channel hops), and outputs a PyG `Data` block with temporal split mapping.
- **Model Training (`train.py`):** A dual-head GraphSAGE classifying patterns and isolating unknown out-of-distribution (OOD) novel anomalies.
- **Inference (`inference.py`):** Outputs production-grade clusters with flagged anomaly states and accumulated amounts.
