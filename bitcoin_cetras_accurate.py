#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CETraS (Connectivity-Enhanced Transaction Graph Sampling) - Accurate Implementation
基于论文: Large Language Models for Cryptocurrency Transaction Analysis (arXiv:2501.18158)

Algorithm: 
1. Compute node importance: I_node = [log(a_in + a_out + 1) + β·log(d_in + d_out + 1)] / (L_s + 1)
   - a_in/out: input/output token amount
   - d_in/out: in/out degree
   - L_s: shortest distance from node to reference node n0
   - β: parameter (default=2)
   
2. Sample nodes by importance probability
3. Connect sampled nodes via shortest paths to maintain connectivity
"""

import os
import json
import math
import random
import zipfile
import argparse
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter


def load_edges_from_zip(zip_path: str, csv_name: str = None, edge_cap: int = 800000) -> Tuple[List[Tuple[str, str]], Dict]:
    """Load edges from CSV in ZIP file, with optional transaction amounts."""
    edges = []
    amounts = {}  # (src, dst) -> amount
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV found in {zip_path}")
        target_csv = csv_name if csv_name else csv_files[0]
        
        with z.open(target_csv) as f:
            df = pd.read_csv(f, nrows=edge_cap)
            
            for _, row in df.iterrows():
                src = str(row.get('source_address', ''))
                dst = str(row.get('destination_address', ''))
                
                if src and dst and src != 'nan' and dst != 'nan':
                    edges.append((src, dst))
                    
                    # Record amount if available
                    if 'satoshi' in row:
                        amount = float(row['satoshi']) if pd.notna(row['satoshi']) else 0.0
                        if (src, dst) in amounts:
                            amounts[(src, dst)] += amount
                        else:
                            amounts[(src, dst)] = amount
    
    print(f"Loaded {len(edges)} edges from {target_csv}")
    return edges, amounts


def build_graph_with_amounts(edges: List[Tuple[str, str]], amounts: Dict) -> nx.DiGraph:
    """Build directed graph with edge weights (transaction amounts)."""
    G = nx.DiGraph()
    
    # Add edges with amounts
    for src, dst in edges:
        amount = amounts.get((src, dst), 0.0)
        if G.has_edge(src, dst):
            # Aggregate amounts for duplicate edges
            G[src][dst]['amount'] += amount
        else:
            G.add_edge(src, dst, amount=amount)
    
    return G


def cetras_sampling(G: nx.DiGraph,
                    target_nodes: int = 10000,
                    beta: float = 2.0,
                    seed: int = 7) -> nx.DiGraph:
    """
    CETraS: Connectivity-Enhanced Transaction Graph Sampling
    
    Paper: Lei et al., "Large Language Models for Cryptocurrency Transaction Analysis" (2025)
    arXiv:2501.18158
    
    Algorithm:
    1. Select reference node n0 (highest degree)
    2. Compute node importance I_node for each node
    3. Sample nodes by importance probability
    4. Connect sampled nodes via shortest paths
    """
    
    random.seed(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n <= target_nodes:
        print(f"Graph has {n} nodes, returning full graph")
        return G.copy()
    
    print(f"Running CETraS on graph with {n} nodes...")
    
    # Step 1: Select reference node n0 (highest total degree)
    degrees = {node: G.in_degree(node) + G.out_degree(node) for node in nodes}
    n0 = max(degrees.items(), key=lambda x: x[1])[0]
    print(f"Reference node n0: {n0} (degree={degrees[n0]})")
    
    # Step 2: Compute shortest distances from n0
    print("Computing shortest paths from reference node...")
    try:
        # Use BFS for shortest path lengths
        lengths = nx.single_source_shortest_path_length(G.to_undirected(), n0)
    except:
        # If graph is disconnected, use large distance for unreachable nodes
        lengths = {}
        for node in nodes:
            try:
                lengths[node] = nx.shortest_path_length(G.to_undirected(), n0, node)
            except:
                lengths[node] = n  # Large distance for disconnected nodes
    
    # Step 3: Compute token amounts (a_in, a_out) for each node
    print("Computing token amounts...")
    a_in = defaultdict(float)
    a_out = defaultdict(float)
    
    for src, dst, data in G.edges(data=True):
        amount = data.get('amount', 0.0)
        a_out[src] += amount
        a_in[dst] += amount
    
    # Step 4: Compute node importance I_node
    print("Computing node importance scores...")
    importance = {}
    
    for node in nodes:
        d_in = G.in_degree(node)
        d_out = G.out_degree(node)
        amount_in = a_in.get(node, 0.0)
        amount_out = a_out.get(node, 0.0)
        L_s = lengths.get(node, n)  # Distance from n0
        
        # I_node = [log(a_in + a_out + 1) + β·log(d_in + d_out + 1)] / (L_s + 1)
        importance[node] = (
            math.log(amount_in + amount_out + 1) + 
            beta * math.log(d_in + d_out + 1)
        ) / (L_s + 1)
    
    # Step 5: Convert importance to sampling probability
    total_importance = sum(importance.values())
    if total_importance == 0:
        # Fallback to uniform if all importance is 0
        prob = {node: 1.0/n for node in nodes}
    else:
        prob = {node: importance[node] / total_importance for node in nodes}
    
    # Step 6: Sample nodes according to probability (optimized with numpy)
    print(f"Sampling {target_nodes} nodes by importance probability...")
    import numpy as np
    
    nodes_list = list(prob.keys())
    probs_list = np.array([prob[node] for node in nodes_list])
    
    # Normalize probabilities
    probs_list = probs_list / probs_list.sum()
    
    # Sample without replacement using numpy (much faster!)
    sample_size = min(target_nodes, n)
    sampled_indices = np.random.choice(len(nodes_list), size=sample_size, replace=False, p=probs_list)
    sampled_subset = set([nodes_list[i] for i in sampled_indices])
    
    print(f"Sampled {len(sampled_subset)} nodes (subset)")
    
    # Step 7: Connect sampled nodes via shortest paths (connectivity enhancement)
    print("Enhancing connectivity by adding shortest paths...")
    G_sampled = nx.DiGraph()
    
    # Add sampled nodes
    for node in sampled_subset:
        G_sampled.add_node(node)
    
    # For each sampled node, find shortest path from n0 and add all nodes/edges on path
    G_und = G.to_undirected()
    
    path_nodes = set(sampled_subset)
    path_edges = set()
    
    for node in sampled_subset:
        try:
            # Find shortest path from n0 to this node
            path = nx.shortest_path(G_und, n0, node)
            
            # Add all nodes and edges on this path
            for i in range(len(path)):
                path_nodes.add(path[i])
                if i > 0:
                    # Add edge (could be either direction in original directed graph)
                    if G.has_edge(path[i-1], path[i]):
                        path_edges.add((path[i-1], path[i]))
                    if G.has_edge(path[i], path[i-1]):
                        path_edges.add((path[i], path[i-1]))
        except nx.NetworkXNoPath:
            # Node not reachable from n0, just add the node
            pass
    
    # Build the final sampled graph
    G_sampled.add_nodes_from(path_nodes)
    for src, dst in path_edges:
        if G.has_edge(src, dst):
            G_sampled.add_edge(src, dst, **G[src][dst])
    
    print(f"Final sampled graph: {G_sampled.number_of_nodes()} nodes, {G_sampled.number_of_edges()} edges")
    
    return G_sampled


def export_top_nodes_and_edges(Gs: nx.DiGraph, 
                               nodes_output: str = "llm4tg_nodes_top1000_cetras.jsonl",
                               edges_output: str = "llm4tg_edges_top1000_cetras.csv",
                               top_k: int = 1000):
    """Export top-k nodes with profiles and induced edges for LLM analysis."""
    
    # Compute node metrics
    nodes_data = []
    in_deg = dict(Gs.in_degree())
    out_deg = dict(Gs.out_degree())
    
    # Betweenness for top nodes
    if Gs.number_of_nodes() > 500:
        betweenness = nx.betweenness_centrality(Gs, k=min(100, Gs.number_of_nodes()//5), normalized=True)
    else:
        betweenness = nx.betweenness_centrality(Gs, normalized=True)
    
    for node in Gs.nodes():
        profile = {
            "id": node,
            "in_degree": in_deg.get(node, 0),
            "out_degree": out_deg.get(node, 0),
            "total_degree": in_deg.get(node, 0) + out_deg.get(node, 0),
            "betweenness": betweenness.get(node, 0.0),
        }
        nodes_data.append(profile)
    
    # Sort by total degree and take top-k
    nodes_data.sort(key=lambda x: x["total_degree"], reverse=True)
    top_nodes = nodes_data[:top_k]
    top_node_ids = set([n["id"] for n in top_nodes])
    
    # Write nodes JSONL
    with open(nodes_output, 'w', encoding='utf-8') as f:
        for node_profile in top_nodes:
            f.write(json.dumps(node_profile, ensure_ascii=False) + '\n')
    print(f"Wrote {len(top_nodes)} nodes to {nodes_output}")
    
    # Get induced edges for top nodes
    induced_edges = []
    for u, v in Gs.edges():
        if u in top_node_ids and v in top_node_ids:
            induced_edges.append((u, v))
    
    # Write edges CSV  
    with open(edges_output, 'w', encoding='utf-8') as f:
        f.write("source,target\n")
        for u, v in induced_edges:
            f.write(f"{u},{v}\n")
    print(f"Wrote {len(induced_edges)} edges to {edges_output}")
    
    return top_nodes, induced_edges


def main():
    parser = argparse.ArgumentParser(description="CETraS: Connectivity-Enhanced Transaction Graph Sampling (Accurate)")
    parser.add_argument("--zip", required=True, help="Path to ZIP with CSV")
    parser.add_argument("--csv_name", default=None, help="Specific CSV name in ZIP")
    parser.add_argument("--edge_cap", type=int, default=800000, help="Max edges to load")
    parser.add_argument("--sample_nodes", type=int, default=10000, help="Target sample size")
    parser.add_argument("--beta", type=float, default=2.0, help="Weight for degree importance (paper default=2)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--out_json", default="cetras_summary.json", help="Output summary JSON")
    parser.add_argument("--export_llm", action="store_true", help="Export top-K nodes/edges for LLM")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of top nodes to export for LLM (default: 1000)")
    
    args = parser.parse_args()
    
    # Load edges and amounts
    print(f"Loading edges from {args.zip}...")
    edges, amounts = load_edges_from_zip(args.zip, args.csv_name, args.edge_cap)
    
    print(f"Building graph with transaction amounts...")
    G = build_graph_with_amounts(edges, amounts)
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print(f"\nRunning CETraS sampling (β={args.beta})...")
    Gs = cetras_sampling(G, 
                        target_nodes=args.sample_nodes,
                        beta=args.beta,
                        seed=args.seed)
    
    # Create summary
    summary = {
        "method": "CETraS (Connectivity-Enhanced Transaction Graph Sampling)",
        "paper": "Lei et al., arXiv:2501.18158, 2025",
        "parameters": {
            "target_nodes": args.sample_nodes,
            "beta": args.beta,
            "seed": args.seed
        },
        "source": os.path.basename(args.zip),
        "graph_stats": {
            "nodes": Gs.number_of_nodes(),
            "edges": Gs.number_of_edges(),
            "density": nx.density(Gs)
        }
    }
    
    # Save summary
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {args.out_json}")
    
    # Export for LLM if requested
    if args.export_llm:
        print(f"\nExporting top-{args.top_k} nodes and edges for LLM analysis...")
        export_top_nodes_and_edges(Gs,
                                  nodes_output=f"llm4tg_nodes_top{args.top_k}_cetras.jsonl",
                                  edges_output=f"llm4tg_edges_top{args.top_k}_cetras.csv",
                                  top_k=args.top_k)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
