"""
export_cluster_images.py
Generates a separate graph image for EVERY cluster found during inference.
Optimized for speed: processes all clusters in parallel using pandas groupby.
"""
import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

def export_all_clusters():
    OUTPUT_DIR = "cluster_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading test report...")
    with open("models/test_cluster_report.json", "r") as f:
        report = json.load(f)
        
    print("Loading transactions... (this takes a few seconds)")
    df = pd.read_csv("aml_dataset/transactions.csv")
    
    # ── 1. Map Accounts to their Inferred Cluster ──
    print("Mapping accounts to communities...")
    acc_to_cluster = {}
    cluster_metadata = {}
    
    for c_meta in report:
        c_id = c_meta["cluster_id"]
        cluster_metadata[c_id] = c_meta
        for acc in c_meta["accounts"]:
            acc_to_cluster[acc] = c_id

    # Add the inferred cluster column to the edges
    df["sender_infer"]   = df["sender_account_id"].map(acc_to_cluster)
    df["receiver_infer"] = df["receiver_account_id"].map(acc_to_cluster)
    
    # Only keep intra-cluster edges
    suspicious_edges = df[(df["sender_infer"] == df["receiver_infer"]) & (df["sender_infer"].notna())]
    
    print(f"Exporting independent images for {len(report)} clusters into '{OUTPUT_DIR}/'...")
    
    # ── 2. Export Loop ──
    grouped = suspicious_edges.groupby("sender_infer")
    
    for c_id in tqdm(cluster_metadata.keys(), desc="Rendering Graphs"):
        if c_id not in grouped.groups:
            continue 
            
        sub_df = grouped.get_group(c_id)
        c_meta = cluster_metadata[c_id]
        pattern = c_meta["pattern"]
        
        # Build graph WITH EDGE ATTRIBUTES extracted from the dataframe
        G = nx.from_pandas_edgelist(
            sub_df, 
            source="sender_account_id", 
            target="receiver_account_id", 
            edge_attr=["amount", "transaction_type"],
            create_using=nx.DiGraph()
        )
        
        # Make the canvas much larger to make room for reading labels
        plt.figure(figsize=(10, 10))
        node_color = "#FFA500" if c_meta["is_novel_pattern"] else "#DC143C"
        
        # Space the nodes out specifically to prevent overlapping labels
        if len(G.nodes) > 15:
            pos = nx.spring_layout(G, k=1.0, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)
            
        # Draw Nodes and Labels
        nx.draw(
            G, pos, 
            with_labels=True, 
            node_size=2000, 
            node_color=node_color,
            font_size=8,
            font_color="white",
            font_weight="bold",
            edge_color="gray",
            linewidths=2,
            edgecolors='black',
            arrows=True,
            arrowsize=20,
            alpha=0.9
        )
        
        # ── EXTRACT AND DRAW EDGE LABELS ──
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            amt = data.get("amount", 0)
            txn_type = data.get("transaction_type", "Unknown")
            edge_labels[(u, v)] = f"₹{amt:,.0f}\n{txn_type}"
            
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color="black",
            font_size=8,
            label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9)
        )
        
        # Add Title
        plt.title(
            f"Cluster: {c_id} | Pattern: {pattern}\n"
            f"Accounts: {c_meta['account_count']} | Blocked: ₹{c_meta['total_amount_blocked']:,.0f}",
            fontsize=12, fontweight="bold", color="#333333"
        )
        
        safe_pattern = pattern.replace(' ', '_').replace('/', '')
        filepath = os.path.join(OUTPUT_DIR, f"{c_id}_{safe_pattern}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close() # Free memory

    print(f"\n✅ All {len(report)} images successfully saved to '{OUTPUT_DIR}/' with rich labels!")

if __name__ == "__main__":
    export_all_clusters()
