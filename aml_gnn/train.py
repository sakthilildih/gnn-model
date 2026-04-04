"""
train.py — GraphSAGE training loop
  - CrossEntropyLoss with class weights (handles imbalance)
  - NeighborLoader mini-batch [25, 10]
  - Early stopping on Val PR-AUC
  - Saves: best_model.pth, centroids.npy, ood_threshold.npy, training_history.csv
"""
import os, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

from config import (
    GRAPH_DIR, MODEL_DIR, CLASS_NAMES, NUM_CLASSES,
    HIDDEN_DIM, DROPOUT, BATCH_SIZE, NUM_NEIGHBORS,
    EPOCHS, EARLY_STOP_PATIENCE, LEARNING_RATE, WEIGHT_DECAY,
    OOD_PERCENTILE,
)
from model import AMLGraphSAGE


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_class_weights(y_tensor: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = torch.bincount(y_tensor, minlength=NUM_CLASSES).float()
    counts = torch.clamp(counts, min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES   # normalise
    return weights


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged PR-AUC across all classes."""
    aucs = []
    for c in range(NUM_CLASSES):
        binary_true = (y_true == c).astype(int)
        if binary_true.sum() == 0:
            continue
        p, r, _ = precision_recall_curve(binary_true, y_score[:, c])
        aucs.append(auc(r, p))
    return float(np.mean(aucs))


# ── Evaluation pass (no gradient) ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE * 2,
        input_nodes=mask,
        shuffle=False,
    )
    all_logits, all_embs, all_y = [], [], []
    for batch in loader:
        batch = batch.to(device)
        n     = batch.batch_size          # target nodes (first n in batch)
        logits, emb = model(batch.x, batch.edge_index)
        all_logits.append(logits[:n].cpu())
        all_embs.append(emb[:n].cpu())
        all_y.append(batch.y[:n].cpu())

    logits = torch.cat(all_logits)
    embs   = torch.cat(all_embs)
    y_true = torch.cat(all_y).numpy()

    probs     = torch.softmax(logits, dim=-1).numpy()
    y_pred    = probs.argmax(axis=-1)

    f1      = f1_score(y_true, y_pred, average="macro", zero_division=0)
    pr_auc_ = pr_auc_score(y_true, probs)
    return pr_auc_, f1, probs, y_true, embs.numpy()


# ── Main training ─────────────────────────────────────────────────────────────

def train():
    # ── Load graph ────────────────────────────────────────────────────────────
    graph_path = os.path.join(GRAPH_DIR, "graph_data.pt")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            "graph_data.pt not found — run build_graph.py first."
        )
    print("Loading graph...")
    data = torch.load(graph_path, weights_only=False)
    print(f"  {data.num_nodes:,} nodes | {data.num_edges:,} edges | "
          f"{data.num_node_features} features")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Class weights ─────────────────────────────────────────────────────────
    train_y       = data.y[data.train_mask]
    class_weights = compute_class_weights(train_y).to(device)
    print("  Class distribution (train):")
    counts = torch.bincount(train_y, minlength=NUM_CLASSES)
    for i, (name, cnt) in enumerate(zip(CLASS_NAMES, counts)):
        print(f"    {name:<30}: {cnt:>8,}  weight={class_weights[i]:.4f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AMLGraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)
    print(f"\n  Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=0,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_prauc = 0.0
    patience_cnt   = 0
    history        = []

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPR-AUC':>10} "
          f"{'ValF1':>8} {'Time':>7}")
    print("─" * 50)

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        model.train()
        total_loss, total_nodes = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            n     = batch.batch_size          # only target nodes
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index)
            loss = criterion(logits[:n], batch.y[:n])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss  += loss.item() * n
            total_nodes += n

        train_loss = total_loss / total_nodes

        # Validation
        val_prauc, val_f1, _, _, _ = evaluate(
            model, data, data.val_mask, device
        )
        scheduler.step(val_prauc)

        elapsed = time.time() - t_ep
        print(f"{epoch:>6} {train_loss:>10.4f} {val_prauc:>10.4f} "
              f"{val_f1:>8.4f} {elapsed:>6.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_prauc": val_prauc, "val_f1": val_f1,
        })

        # Save best
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            patience_cnt   = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"         ✓ saved best model (val_prauc={val_prauc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    # ── Save training history ─────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(
        os.path.join(MODEL_DIR, "training_history.csv"), index=False
    )

    # ── Load best model for centroid computation ───────────────────────────────
    print("\nComputing OOD centroids from best model...")
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model.pth"), weights_only=True)
    )
    model.eval()

    _, _, _, y_train_np, train_embs = evaluate(
        model, data, data.train_mask, device
    )

    # Per-class centroid in embedding space
    centroids = np.zeros((NUM_CLASSES, HIDDEN_DIM), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask_c = (y_train_np == c)
        if mask_c.sum() > 0:
            centroids[c] = train_embs[mask_c].mean(axis=0)

    # OOD threshold: 95th-pct of min-distance-to-nearest-centroid over training
    dists = []
    for emb, true_c in zip(train_embs, y_train_np):
        d = np.linalg.norm(emb - centroids[true_c])
        dists.append(d)
    ood_threshold = np.percentile(dists, OOD_PERCENTILE)
    print(f"  OOD threshold (p{OOD_PERCENTILE}): {ood_threshold:.4f}")

    np.save(os.path.join(MODEL_DIR, "centroids.npy"),    centroids)
    np.save(os.path.join(MODEL_DIR, "ood_threshold.npy"),
            np.array([ood_threshold]))

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\nTest set evaluation:")
    test_prauc, test_f1, test_probs, test_y, _ = evaluate(
        model, data, data.test_mask, device
    )
    test_pred = test_probs.argmax(axis=-1)
    print(f"  PR-AUC : {test_prauc:.4f}")
    print(f"  F1     : {test_f1:.4f}")
    print("\nClassification Report:")
    # Only show classes present in test set
    present = sorted(set(test_y.tolist()))
    print(classification_report(
        test_y, test_pred,
        labels=present,
        target_names=[CLASS_NAMES[i] for i in present],
        zero_division=0,
    ))

    print(f"\nAll model files saved to: {MODEL_DIR}/")
    print("  best_model.pth")
    print("  centroids.npy")
    print("  ood_threshold.npy")
    print("  training_history.csv")


if __name__ == "__main__":
    train()
