# AML Fraud Ring Detection — Graph Neural Network (GNN)

This repository contains the Layer-2 Anti-Money Laundering (AML) system, built using **GraphSAGE**. 

It is designed to receive suspicious transactions flagged by an XGBoost model (Layer-1), instantly build a transaction network graph, and output fully mapped fraud rings (e.g. Mule Rings, Structuring, Fan-in) with total blocked amounts.

---

## 🚀 Moving to an NVIDIA GPU System

If you have just cloned this repo to a new machine with an NVIDIA GPU, follow these exact steps to load the dataset, retrain the model on the GPU, and run the pipeline.

### Step 1: Unzip the Databases
Because GitHub limits file sizes, the dataset and graph cache are zipped.
Open a terminal in the root of the project and extract them:

**Windows (PowerShell):**
```powershell
Expand-Archive -Path aml_dataset.zip -DestinationPath . -Force
Expand-Archive -Path graph_cache.zip -DestinationPath . -Force
```

**Linux/Mac:**
```bash
unzip aml_dataset.zip -d .
unzip graph_cache.zip -d .
```

### Step 2: Install PyTorch Geometric + CUDA
Since you are on a GPU system, you must install the CUDA versions of PyTorch and PyTorch Geometric so `train.py` will use your GPU perfectly.

```bash
# 1. Install regular requirements
pip install -r requirements.txt

# 2. Install PyTorch with CUDA (Check https://pytorch.org/get-started/locally/ for the right command for your CUDA version)
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric and its dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Step 3: Run the Full Pipeline (Train + Infer)
Now you can re-train the model. It will automatically detect your GPU (`cuda`) and zip through the epochs.

```bash
cd aml_gnn
python run_pipeline.py
```
*This command will:*
1. Build the graph features.
2. Train the GraphSAGE model on the GPU.
3. Save `best_model.pth` and centroids into the `models/` directory.
4. Run inference to generate a cluster report.

### Step 4: Verify Pattern Detection
To verify that the model correctly identifies the 10 different ML patterns, run the test script:

```bash
cd ..
# The `--test` flag runs the test dataset against the trained weights
python aml_gnn/run_pipeline.py --test
```
*You will find output graph images in `test_cluster_images/`.*

### Step 5: Run Production E2E Pipeline Demo
To test how this engine runs in real-time when chained with your friend's XGBoost system:

```bash
cd ..
python run_production_demo.py
```
This sets up the `TransactionStore`, simulates 7,000 transactions/sec, calls `AMLProductionEngine`, and finds the rings!

---

## Architecture of the Pipeline
```text
[XGBoost Layer 1] ──(suspicious accounts)──> [Transaction Store] ──(2-hop expansion)──> [AMLProductionEngine]

AMLProductionEngine output looks like:
{
  "ring_id": "RING_20260405_000001",
  "accounts": ["ACC001", "ACC002", "ACC105"],
  "pattern": "mule_ring",
  "total_amount_blocked": 1250000.50
}
```
