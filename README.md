# Bitcoin Network Analysis with RWFB Sampling

This project provides tools for analyzing Bitcoin transaction networks using Random Walk with Fly-Back (RWFB) sampling and network metrics computation.

## Overview

The main script `bitcoin_rwfb_llm4tg.py` processes Bitcoin transaction data from CSV files to:
- Load transaction edges from compressed CSV files
- Build directed transaction graphs
- Apply RWFB sampling to extract representative subgraphs
- Compute network metrics and centrality measures
- Generate summary reports with top hubs and bridges

## Files

- `bitcoin_rwfb_llm4tg.py` - Main analysis script
- `crypto_network_llm_pipeline.ipynb` - Jupyter notebook with extended analysis pipeline
- `2020-10_01.csv.zip` - Sample Bitcoin transaction data (compressed)
- `llm4tg_summary.json` - Output summary file

## Usage

### Basic Command

```bash
python bitcoin_rwfb_llm4tg.py --zip 2020-10_01.csv.zip --edge_cap 1000000 --sample_nodes 10000 --p_flyback 0.3 --teleport_p 0.01 --out_json llm4tg_summary.json
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--zip` | string | **required** | Path to ZIP file containing CSV transaction data |
| `--csv_name` | string | None | Specific CSV filename inside ZIP (auto-detected if not specified) |
| `--edge_cap` | int | 800000 | Maximum number of edges to read (memory control) |
| `--sample_nodes` | int | 10000 | Number of nodes to sample using RWFB |
| `--p_flyback` | float | 0.3 | RWFB flying-back probability |
| `--teleport_p` | float | 0.01 | Random teleport probability |
| `--seed` | int | 7 | Random seed for reproducibility |
| `--out_json` | string | llm4tg_summary.json | Output JSON file path |

### Input Data Format

The CSV file should contain the following columns:
- `timestamp`: Transaction timestamp
- `source_address`: Source Bitcoin address
- `destination_address`: Destination Bitcoin address  
- `satoshi`: Transaction amount in satoshis

## Algorithm Details

### RWFB (Random Walk with Fly-Back) Sampling

The script implements RWFB sampling to extract representative subgraphs:

1. **Random Walk**: Traverses the graph following edges
2. **Fly-Back**: With probability `p_flyback`, stays at current node
3. **Teleportation**: With probability `teleport_p`, jumps to random node
4. **Sampling**: Continues until `sample_nodes` are visited

### Network Metrics Computed

- **Basic Statistics**: Node count, edge count
- **Clustering**: Average clustering coefficient
- **Connectivity**: Number and size of strongly/weakly connected components
- **Diameter**: Estimated graph diameter
- **Centrality**: Degree, closeness, and betweenness centrality

## Output

The script generates a JSON file containing:

```json
{
  "snapshot": "RWFB sample description",
  "graph_stats": {
    "nodes": 10000,
    "edges": 50000,
    "avg_clustering": 0.15,
    "num_SCC": 5,
    "num_WCC": 3,
    "largest_SCC": 8000,
    "largest_WCC": 9500,
    "diameter_estimate": 12
  },
  "top_hubs_by_degree": [...],
  "top_bridges_by_betweenness": [...]
}
```

## Dependencies

- Python 3.6+
- pandas
- numpy
- networkx
- argparse (built-in)
- zipfile (built-in)
- json (built-in)
- collections (built-in)
- random (built-in)

## Installation

```bash
pip install pandas numpy networkx
```

## Example Analysis

The provided command analyzes a Bitcoin transaction dataset from October 1, 2020:

- Limits processing to 1 million edges for memory efficiency
- Samples 10,000 nodes using RWFB with 30% fly-back probability
- Uses 1% teleportation probability for exploration
- Outputs results to `llm4tg_summary.json`

## Research Applications

This tool is designed for:
- Bitcoin network topology analysis
- Identifying key transaction hubs and bridges
- Network robustness studies
- Temporal network evolution analysis
- Academic research on cryptocurrency networks

## License

This project is provided for research and educational purposes.

## Citation

If you use this tool in your research, please cite the relevant papers on RWFB sampling and Bitcoin network analysis. 