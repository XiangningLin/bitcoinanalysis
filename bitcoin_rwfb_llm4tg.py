
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, zipfile, json, math, random
from collections import deque
import pandas as pd
import numpy as np
import networkx as nx

def load_edges_from_zip(zip_path, csv_name=None, edge_cap=None, chunksize=200_000):
    zf = zipfile.ZipFile(zip_path)
    if csv_name is None:
        # pick first .csv in archive
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("No CSV found in ZIP")
        csv_name = names[0]
    edges = []
    with zf.open(csv_name) as f:
        for chunk in pd.read_csv(f, chunksize=chunksize, dtype={"timestamp":"string","source_address":"string","destination_address":"string","satoshi":"float64"}, low_memory=False):
            chunk = chunk.dropna(subset=["source_address","destination_address"])
            chunk = chunk[(chunk["source_address"].str.len()>0)&(chunk["destination_address"].str.len()>0)]
            edges.extend(zip(chunk["source_address"].tolist(), chunk["destination_address"].tolist()))
            if edge_cap is not None and len(edges) >= edge_cap:
                edges = edges[:edge_cap]; break
    return edges

def build_graph(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

def rwfb_sample_fast(G, sample_nodes=10000, p_flyback=0.3, teleport_p=0.01, seed=7):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if not nodes:
        raise ValueError("Empty graph")
    current = rng.choice(nodes)
    visited = set([current])
    node_neighbors = {u: list(G.successors(u)) for u in nodes}
    steps = 0
    limit = sample_nodes * 50
    while len(visited) < sample_nodes and steps < limit:
        steps += 1
        if rng.random() < teleport_p:
            current = rng.choice(nodes); visited.add(current); continue
        if rng.random() < p_flyback:
            pass
        else:
            nbrs = node_neighbors.get(current, [])
            if nbrs:
                current = rng.choice(nbrs); visited.add(current)
            else:
                current = rng.choice(nodes); visited.add(current)
    return G.subgraph(visited).copy()

def approx_diameter(G_und):
    if G_und.number_of_nodes() == 0:
        return 0
    start = next(iter(G_und.nodes()))
    lengths = nx.single_source_shortest_path_length(G_und, start)
    far = max(lengths, key=lengths.get)
    lengths2 = nx.single_source_shortest_path_length(G_und, far)
    return max(lengths2.values())

def compute_metrics(Gs):
    G_und = Gs.to_undirected()
    metrics = {
        "nodes": Gs.number_of_nodes(),
        "edges": Gs.number_of_edges(),
        "avg_clustering": nx.average_clustering(G_und),
        "num_SCC": nx.number_strongly_connected_components(Gs),
        "num_WCC": nx.number_connected_components(G_und),
        "largest_SCC": max((len(c) for c in nx.strongly_connected_components(Gs)), default=0),
        "largest_WCC": max((len(c) for c in nx.connected_components(G_und)), default=0),
        "diameter_estimate": approx_diameter(G_und),
    }
    return metrics

def centrality_lcc(Gs, k_sample=500, seed=7):
    G_und = Gs.to_undirected()
    if G_und.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node","degree","closeness","betweenness"])
    lcc_nodes = max(nx.connected_components(G_und), key=len)
    Glcc = Gs.subgraph(lcc_nodes).copy()
    clo = nx.closeness_centrality(Glcc.to_undirected())
    k = min(k_sample, Glcc.number_of_nodes())
    btw = nx.betweenness_centrality(Glcc.to_undirected(), k=k, seed=seed)
    df = pd.DataFrame({
        "node": list(Glcc.nodes()),
        "degree": [Glcc.degree(n) for n in Glcc.nodes()],
        "closeness": [clo[n] for n in Glcc.nodes()],
        "betweenness": [btw[n] for n in Glcc.nodes()],
    }).sort_values("degree", ascending=False)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to ZIP with CSV (timestamp, source_address, destination_address, satoshi)")
    ap.add_argument("--csv_name", default=None, help="CSV name inside the ZIP (optional)")
    ap.add_argument("--edge_cap", type=int, default=800000, help="Max edges to read for memory control")
    ap.add_argument("--sample_nodes", type=int, default=10000, help="RWFB sample node count")
    ap.add_argument("--p_flyback", type=float, default=0.3, help="RWFB flying-back probability")
    ap.add_argument("--teleport_p", type=float, default=0.01, help="Random teleport probability")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--out_json", default="llm4tg_summary.json", help="Output JSON with metrics and top nodes")
    args = ap.parse_args()

    edges = load_edges_from_zip(args.zip, args.csv_name, args.edge_cap)
    G = build_graph(edges)
    Gs = rwfb_sample_fast(G, args.sample_nodes, args.p_flyback, args.teleport_p, args.seed)
    metrics = compute_metrics(Gs)
    cent = centrality_lcc(Gs)

    top_by_deg = cent.head(30).to_dict(orient="records")
    top_by_btw = cent.sort_values("betweenness", ascending=False).head(30).to_dict(orient="records")

    summary = {
        "snapshot": f"RWFB sample ({args.sample_nodes} nodes) from {os.path.basename(args.zip)}",
        "graph_stats": metrics,
        "top_hubs_by_degree": top_by_deg,
        "top_bridges_by_betweenness": top_by_btw,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    import os
    main()
