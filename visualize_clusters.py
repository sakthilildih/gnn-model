"""
Visualizes representative AML graph motifs (clusters) 
from the raw transaction dataset to demonstrate the dataset patterns.
"""
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def plot_graph_samples():
    OUTPUT_IMG = "graph_motifs.png"
    print("Loading data...")
    # Load raw transactions and cluster info
    df = pd.read_csv('aml_dataset/transactions.csv')
    dc = pd.read_csv('aml_dataset/cluster_summary.csv')

    # Patterns we want to visualize
    target_patterns = [
        "mule_ring", 
        "chain_layering", 
        "structuring_fanout", 
        "funnel_fanin", 
        "diamond_fragmentation", 
        "benign_salary"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, pattern in enumerate(target_patterns):
        ax = axes[idx]
        
        # Get the first cluster_id of this pattern
        cluster_id = dc[dc['pattern_family'] == pattern]['cluster_id'].iloc[0]
        
        # Get all edges (transactions) for this cluster
        sub_df = df[df['cluster_id'] == cluster_id]
        
        # Build directed graph
        G = nx.from_pandas_edgelist(
            sub_df, 
            source='sender_account_id', 
            target='receiver_account_id', 
            create_using=nx.DiGraph()
        )
        
        is_suspicious = "benign" not in pattern
        node_color = "crimson" if is_suspicious else "mediumseagreen"
        
        # Use simple kamada_kawai layout for clarity
        pos = nx.kamada_kawai_layout(G)
        
        nx.draw(
            G, pos, ax=ax, 
            with_labels=False, 
            node_size=150, 
            node_color=node_color,
            edge_color="tab:gray",
            linewidths=1,
            edgecolors='black',
            arrows=True,
            arrowsize=12,
            alpha=0.9
        )
        
        ax.set_title(f"Pattern: {pattern}\n({len(G.nodes)} nodes, {len(G.edges)} egdes)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"✅ Successfully saved visualizations to {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_graph_samples()
