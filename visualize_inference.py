"""
visualize_inference.py
Extracts graph structure of newly inferred clusters from the model's test report,
matches them against raw transactions, and plots them!
"""
import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def plot_inferred_clusters():
    OUTPUT_IMG = "infer_motifs.png"
    
    with open("models/test_cluster_report.json", "r") as f:
        report = json.load(f)
        
    print("Loading test transactions...")
    df = pd.read_csv("aml_dataset/transactions.csv")
    
    # Let's pick the top 3 specific clusters we showed in the chat
    target_cluster_ids = ["INFER_00000", "INFER_00001", "INFER_00004", "INFER_00003"]
    clusters_to_plot = [c for c in report if c["cluster_id"] in target_cluster_ids]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, c_meta in enumerate(clusters_to_plot):
        ax = axes[idx]
        accounts = set(c_meta["accounts"])
        
        # Filter transactions that involve these specific accounts
        # To get the closed topology, both sender and receiver should be in the cluster
        sub_df = df[df["sender_account_id"].isin(accounts) & df["receiver_account_id"].isin(accounts)]
        
        G = nx.from_pandas_edgelist(
            sub_df, 
            source="sender_account_id", 
            target="receiver_account_id", 
            create_using=nx.DiGraph()
        )
        
        pattern = c_meta["pattern"]
        if c_meta["is_novel_pattern"]:
            node_color = "darkorange"   # Orange for Unknown/Zero-Day
        else:
            node_color = "crimson"      # Red for identified AML
            
        pos = nx.kamada_kawai_layout(G)
        
        nx.draw(
            G, pos, ax=ax, 
            with_labels=False, 
            node_size=200, 
            node_color=node_color,
            edge_color="tab:gray",
            linewidths=1.5,
            edgecolors='black',
            arrows=True,
            arrowsize=15,
            alpha=0.9
        )
        
        title = f"{c_meta['cluster_id']} | {pattern}\n"
        title += f"Accounts: {len(accounts)} | Blocked: ₹{c_meta['total_amount_blocked']:,.0f}"
        ax.set_title(title, fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"✅ Successfully saved visualizations to {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_inferred_clusters()
