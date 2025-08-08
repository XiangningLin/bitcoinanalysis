#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CETraS (Centrality-based Transaction Sampling) Implementation
基于中心性的比特币交易网络采样方法
"""

import os
import json
import random
import zipfile
import argparse
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict, Set
from collections import defaultdict


def load_edges_from_zip(zip_path: str, csv_name: str = None, edge_cap: int = 800000) -> List[Tuple[str, str]]:
    """Load edges from CSV in ZIP file."""
    edges = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV found in {zip_path}")
        target_csv = csv_name if csv_name else csv_files[0]
        
        with z.open(target_csv) as f:
            df = pd.read_csv(f, nrows=edge_cap)
            if 'source_address' in df.columns and 'destination_address' in df.columns:
                for _, row in df.iterrows():
                    src, dst = str(row['source_address']), str(row['destination_address'])
                    if src and dst and src != 'nan' and dst != 'nan':
                        edges.append((src, dst))
    
    print(f"Loaded {len(edges)} edges from {target_csv}")
    return edges


def build_graph(edges: List[Tuple[str, str]]) -> nx.DiGraph:
    """Build directed graph from edges."""
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def cetras_sampling(G: nx.DiGraph, 
                    sample_nodes: int = 10000,
                    alpha: float = 0.4,  # weight for degree centrality
                    beta: float = 0.3,   # weight for betweenness centrality  
                    gamma: float = 0.3,  # weight for closeness centrality
                    expansion_hops: int = 1,  # BFS expansion from core nodes
                    seed: int = 7) -> nx.DiGraph:
    """
    CETraS: Centrality-based Transaction Sampling
    
    Algorithm:
    1. Compute multiple centrality measures (degree, betweenness, closeness)
    2. Create composite centrality score with weights (α, β, γ)
    3. Select top-k nodes by composite score as "core nodes"
    4. Expand from core nodes using BFS to include neighbors
    5. Return induced subgraph
    
    Parameters:
    - sample_nodes: target number of nodes to sample
    - alpha, beta, gamma: weights for degree, betweenness, closeness (sum to 1.0)
    - expansion_hops: how many hops to expand from core nodes
    - seed: random seed for reproducibility
    """
    
    random.seed(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n <= sample_nodes:
        print(f"Graph has {n} nodes, returning full graph")
        return G.copy()
    
    print(f"Computing centrality measures for {n} nodes...")
    
    # Step 1: Compute centrality measures
    # Use undirected for some measures if needed
    G_und = G.to_undirected()
    
    # Degree centrality (in + out for directed)
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    degree_cent = {node: (in_deg.get(node, 0) + out_deg.get(node, 0)) / (2 * (n - 1)) 
                   for node in nodes}
    
    # Betweenness centrality (sample for speed on large graphs)
    if n > 5000:
        k_sample = min(500, n // 10)
        betweenness_cent = nx.betweenness_centrality(G, k=k_sample, normalized=True, seed=seed)
    else:
        betweenness_cent = nx.betweenness_centrality(G, normalized=True)
    
    # Closeness centrality (use largest strongly connected component for directed)
    sccs = list(nx.strongly_connected_components(G))
    largest_scc = max(sccs, key=len) if sccs else set()
    closeness_cent = {}
    
    if len(largest_scc) > 1:
        G_scc = G.subgraph(largest_scc)
        closeness_cent_scc = nx.closeness_centrality(G_scc, wf_improved=True)
        # Extend to full graph (0 for nodes outside SCC)
        closeness_cent = {node: closeness_cent_scc.get(node, 0.0) for node in nodes}
    else:
        # Fallback: use undirected closeness
        closeness_cent = nx.closeness_centrality(G_und)
    
    # Step 2: Compute composite centrality score
    composite_scores = {}
    for node in nodes:
        score = (alpha * degree_cent.get(node, 0) +
                beta * betweenness_cent.get(node, 0) +
                gamma * closeness_cent.get(node, 0))
        composite_scores[node] = score
    
    # Step 3: Select core nodes (high centrality)
    sorted_nodes = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Start with top centrality nodes as core
    core_size = min(sample_nodes // 3, n // 10)  # core is ~1/3 of sample or 10% of graph
    core_nodes = set([node for node, _ in sorted_nodes[:core_size]])
    sampled_nodes = core_nodes.copy()
    
    print(f"Selected {len(core_nodes)} core nodes based on centrality")
    
    # Step 4: Expand from core nodes using BFS
    if expansion_hops > 0 and len(sampled_nodes) < sample_nodes:
        # Build neighbor lookup
        neighbors_out = {node: set(G.successors(node)) for node in G.nodes()}
        neighbors_in = {node: set(G.predecessors(node)) for node in G.nodes()}
        
        # BFS expansion
        current_frontier = core_nodes.copy()
        for hop in range(expansion_hops):
            if len(sampled_nodes) >= sample_nodes:
                break
                
            next_frontier = set()
            for node in current_frontier:
                # Add both in and out neighbors
                candidates = (neighbors_out.get(node, set()) | 
                             neighbors_in.get(node, set())) - sampled_nodes
                
                # Prioritize by centrality score
                candidates_scored = [(n, composite_scores.get(n, 0)) for n in candidates]
                candidates_scored.sort(key=lambda x: x[1], reverse=True)
                
                # Add top candidates
                for candidate, _ in candidates_scored:
                    if len(sampled_nodes) >= sample_nodes:
                        break
                    sampled_nodes.add(candidate)
                    next_frontier.add(candidate)
            
            current_frontier = next_frontier
            print(f"Hop {hop+1}: sampled {len(sampled_nodes)} nodes")
    
    # Step 5: If still need more nodes, add remaining high-centrality nodes
    if len(sampled_nodes) < sample_nodes:
        for node, _ in sorted_nodes:
            if node not in sampled_nodes:
                sampled_nodes.add(node)
                if len(sampled_nodes) >= sample_nodes:
                    break
    
    print(f"Final sample: {len(sampled_nodes)} nodes")
    
    # Return induced subgraph
    return G.subgraph(sampled_nodes).copy()


def compute_metrics(Gs: nx.DiGraph) -> Dict:
    """Compute graph metrics for sampled subgraph."""
    n = Gs.number_of_nodes()
    m = Gs.number_of_edges()
    
    if n == 0:
        return {"nodes": 0, "edges": 0}
    
    # Basic metrics
    metrics = {
        "nodes": n,
        "edges": m,
        "density": nx.density(Gs),
    }
    
    # Connected components
    wcc = list(nx.weakly_connected_components(Gs))
    scc = list(nx.strongly_connected_components(Gs))
    
    metrics.update({
        "num_WCC": len(wcc),
        "num_SCC": len(scc),
        "largest_WCC": max((len(c) for c in wcc), default=0),
        "largest_SCC": max((len(c) for c in scc), default=0),
    })
    
    # Clustering (on undirected)
    Gs_und = Gs.to_undirected()
    metrics["avg_clustering"] = nx.average_clustering(Gs_und) if n > 1 else 0.0
    
    # Approximate diameter
    if len(wcc) > 0:
        largest_wcc_nodes = max(wcc, key=len)
        if len(largest_wcc_nodes) > 1:
            G_wcc = Gs_und.subgraph(largest_wcc_nodes)
            try:
                # Sample shortest paths for diameter estimate
                sample_size = min(100, len(largest_wcc_nodes))
                sample_nodes = random.sample(list(largest_wcc_nodes), sample_size)
                path_lengths = []
                for i in range(min(50, sample_size)):
                    if nx.has_path(G_wcc, sample_nodes[i], sample_nodes[-i-1]):
                        path_lengths.append(nx.shortest_path_length(G_wcc, sample_nodes[i], sample_nodes[-i-1]))
                metrics["diameter_estimate"] = max(path_lengths) if path_lengths else 0
            except:
                metrics["diameter_estimate"] = 0
        else:
            metrics["diameter_estimate"] = 0
    else:
        metrics["diameter_estimate"] = 0
    
    return metrics


def export_top_nodes_and_edges(Gs: nx.DiGraph, 
                               nodes_output: str = "llm4tg_nodes_top100_cetras.jsonl",
                               edges_output: str = "llm4tg_edges_top100_cetras.csv",
                               top_k: int = 100):
    """
    Export top-k nodes with profiles and induced edges for LLM analysis.
    Matches the format of existing files.
    """
    
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
    parser = argparse.ArgumentParser(description="CETraS: Centrality-based Transaction Sampling")
    parser.add_argument("--zip", required=True, help="Path to ZIP with CSV")
    parser.add_argument("--csv_name", default=None, help="Specific CSV name in ZIP")
    parser.add_argument("--edge_cap", type=int, default=800000, help="Max edges to load")
    parser.add_argument("--sample_nodes", type=int, default=10000, help="Target sample size")
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight for degree centrality")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for betweenness centrality")
    parser.add_argument("--gamma", type=float, default=0.3, help="Weight for closeness centrality")
    parser.add_argument("--expansion_hops", type=int, default=1, help="BFS expansion hops from core")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--out_json", default="cetras_summary.json", help="Output summary JSON")
    parser.add_argument("--export_llm", action="store_true", help="Export top-100 nodes/edges for LLM")
    
    args = parser.parse_args()
    
    # Validate weights sum to 1
    weight_sum = args.alpha + args.beta + args.gamma
    if abs(weight_sum - 1.0) > 0.01:
        print(f"Warning: weights sum to {weight_sum}, normalizing...")
        args.alpha /= weight_sum
        args.beta /= weight_sum
        args.gamma /= weight_sum
    
    # Load and sample
    print(f"Loading edges from {args.zip}...")
    edges = load_edges_from_zip(args.zip, args.csv_name, args.edge_cap)
    
    print(f"Building graph...")
    G = build_graph(edges)
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print(f"\nRunning CETraS sampling (α={args.alpha}, β={args.beta}, γ={args.gamma})...")
    Gs = cetras_sampling(G, 
                        sample_nodes=args.sample_nodes,
                        alpha=args.alpha,
                        beta=args.beta,
                        gamma=args.gamma,
                        expansion_hops=args.expansion_hops,
                        seed=args.seed)
    
    print(f"\nComputing metrics...")
    metrics = compute_metrics(Gs)
    
    # Create summary
    summary = {
        "method": "CETraS (Centrality-based Transaction Sampling)",
        "parameters": {
            "sample_nodes": args.sample_nodes,
            "alpha_degree": args.alpha,
            "beta_betweenness": args.beta,
            "gamma_closeness": args.gamma,
            "expansion_hops": args.expansion_hops,
            "seed": args.seed
        },
        "source": os.path.basename(args.zip),
        "graph_stats": metrics
    }
    
    # Save summary
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {args.out_json}")
    
    # Export for LLM if requested
    if args.export_llm:
        print("\nExporting top-100 nodes and edges for LLM analysis...")
        export_top_nodes_and_edges(Gs)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
