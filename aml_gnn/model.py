"""
model.py — 2-layer GraphSAGE with dual heads:
  HEAD A : multi-class  → pattern_family prediction (CrossEntropyLoss)
  HEAD B : 128-dim embedding extraction → OOD novel-pattern detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from config import HIDDEN_DIM, DROPOUT, NUM_CLASSES


class AMLGraphSAGE(nn.Module):
    """
    2-layer GraphSAGE.

    forward()  → (logits [N, NUM_CLASSES], embeddings [N, HIDDEN_DIM])
    encode()   → embeddings only  (used at inference for OOD)
    """

    def __init__(
        self,
        in_channels:  int = None,          # auto-set from data
        hidden_channels: int = HIDDEN_DIM,
        out_channels: int = NUM_CLASSES,
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: captures local mule-neighbour relationships
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.bn1   = nn.BatchNorm1d(hidden_channels)

        # Layer 2: captures ring / chain context (2-hop)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.bn2   = nn.BatchNorm1d(hidden_channels)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    # ── encoding (shared trunk) ───────────────────────────────────────────────
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        return x   # 128-dim node embeddings

    # ── full forward ──────────────────────────────────────────────────────────
    def forward(self, x, edge_index):
        emb    = self.encode(x, edge_index)
        emb_d  = F.dropout(emb, p=self.dropout, training=self.training)
        logits = self.classifier(emb_d)
        return logits, emb

    # ── mule probability (P(class != benign)) ─────────────────────────────────
    @torch.no_grad()
    def mule_score(self, x, edge_index):
        """Returns probability that each node is suspicious (1 - P(benign))."""
        logits, emb = self.forward(x, edge_index)
        probs       = torch.softmax(logits, dim=-1)
        # class 0 = benign
        mule_prob   = 1.0 - probs[:, 0]
        pred_class  = probs.argmax(dim=-1)
        return mule_prob, pred_class, emb
