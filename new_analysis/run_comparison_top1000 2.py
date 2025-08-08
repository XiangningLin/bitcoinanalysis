#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-1000 Comparison Analysis
对比 RWFB vs CETraS 的 Top-1000 结果
"""

import os
import re
import csv
import json
import math
import random
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(CURRENT_DIR, "outputs", "compare_top1000")
os.makedirs(OUT_DIR, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_roles(items: List[dict]) -> List[str]:
    roles = []
    for item in items:
        if isinstance(item, dict) and "role" in item and "_error" not in item:
            roles.append(str(item["role"]).lower())  # Lowercase for consistency
    return roles


STOPWORDS = set("""
a an the and or of to in on for with from by is are was were be being been this that these those
as at it its their his her they them we you your our ours can could should may might will would
not no into over under above below more less most least very really just also etc based using use than
then thus hence such via per about across among between within without during after before near around
one two three four five six seven eight nine ten
""".split())


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def top_keywords(text: str, k: int = 20) -> List[Tuple[str, int]]:
    return Counter(tokenize(text)).most_common(k)


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    eps = 1e-12
    def get(d, k):
        return max(eps, d.get(k, 0.0))
    m = {k: 0.5*(get(p,k)+get(q,k)) for k in keys}
    def kl(a, b):
        return sum(get(a,k)*math.log2(get(a,k)/get(b,k)) for k in keys)
    return 0.5*kl(p,m) + 0.5*kl(q,m)


def dist_from_counts(cnt: Counter) -> Dict[str, float]:
    total = sum(cnt.values()) or 1
    return {k: v/total for k, v in cnt.items()}


def bar_compare(labels: List[str], v1: List[float], v2: List[float], title: str, out_png: str, l1: str, l2: str):
    import numpy as np
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(x - w/2, v1, width=w, label=l1, alpha=0.8)
    ax.bar(x + w/2, v2, width=w, label=l2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(title, fontsize=14)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_table_md(path: str, headers: List[str], rows: List[List[str]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join([" --- " for _ in headers]) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")


def save_table_png(path: str, headers: List[str], rows: List[List[str]], title: str = None):
    n_rows = max(1, len(rows))
    fig_height = 0.6 + 0.3 * n_rows
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
    fig.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()


def load_graph_edges(path: str) -> List[Tuple[str,str]]:
    edges = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                edges.append((row[0], row[1]))
    return edges


def degree_gini(edges: List[Tuple[str,str]]) -> float:
    deg = Counter()
    for u,v in edges:
        deg[u]+=1; deg[v]+=1
    vals = sorted(deg.values())
    if not vals:
        return 0.0
    n = len(vals)
    cum = sum((i+1) * x for i, x in enumerate(vals))
    total = sum(vals)
    return (2*cum)/(n*total) - (n+1)/n


def main():
    print("="*70)
    print("Top-1000 对比分析")
    print("="*70)
    
    # Load results
    paths = {
        "rwfb_roles": "outputs/rwfb_top1000/rwfb_role_predictions.json",
        "rwfb_anom": "outputs/rwfb_top1000/rwfb_anomaly_explained.txt",
        "rwfb_sum": "outputs/rwfb_top1000/rwfb_decentralization_summary.txt",
        "cetras_roles": "outputs/cetras_top1000/cetras_role_predictions.json",
        "cetras_anom": "outputs/cetras_top1000/cetras_anomaly_explained.txt",
        "cetras_sum": "outputs/cetras_top1000/cetras_decentralization_summary.txt",
    }
    
    # Check files
    for key, path in paths.items():
        full_path = os.path.join(CURRENT_DIR, path)
        if os.path.exists(full_path):
            print(f"✅ {key}")
        else:
            print(f"❌ 缺失: {key}")
            return
    
    # Roles
    print("\n📊 分析角色分布...")
    rwfb_roles = extract_roles(load_json(os.path.join(CURRENT_DIR, paths["rwfb_roles"])))
    cetras_roles = extract_roles(load_json(os.path.join(CURRENT_DIR, paths["cetras_roles"])))
    
    print(f"RWFB 有效角色: {len(rwfb_roles)}")
    print(f"CETraS 有效角色: {len(cetras_roles)}")
    
    cnt_rwfb = Counter(rwfb_roles)
    cnt_cetras = Counter(cetras_roles)
    all_roles = sorted(set(cnt_rwfb) | set(cnt_cetras))
    
    dist_rwfb = dist_from_counts(cnt_rwfb)
    dist_cetras = dist_from_counts(cnt_cetras)
    jsd_roles = js_divergence(dist_rwfb, dist_cetras)
    
    print(f"Role JSD: {jsd_roles:.4f} bits")
    print(f"RWFB top roles: {cnt_rwfb.most_common(5)}")
    print(f"CETraS top roles: {cnt_cetras.most_common(5)}")
    
    # Role comparison figure
    v1 = [cnt_rwfb.get(r, 0) for r in all_roles]
    v2 = [cnt_cetras.get(r, 0) for r in all_roles]
    bar_compare(all_roles, v1, v2, "Role Distribution: RWFB vs CETraS (Top-1000)", 
                os.path.join(OUT_DIR, "compare_roles.png"), "RWFB", "CETraS")
    
    # Keywords
    print("\n📝 分析关键词...")
    anom_rwfb = read_text(os.path.join(CURRENT_DIR, paths["rwfb_anom"]))
    anom_cetras = read_text(os.path.join(CURRENT_DIR, paths["cetras_anom"]))
    sum_rwfb = read_text(os.path.join(CURRENT_DIR, paths["rwfb_sum"]))
    sum_cetras = read_text(os.path.join(CURRENT_DIR, paths["cetras_sum"]))
    
    kw_anom_rwfb = top_keywords(anom_rwfb, 15)
    kw_anom_cetras = top_keywords(anom_cetras, 15)
    kw_sum_rwfb = top_keywords(sum_rwfb, 15)
    kw_sum_cetras = top_keywords(sum_cetras, 15)
    
    # Anomaly keywords figure
    labels_a = [w for w,_ in kw_anom_rwfb]
    vals_a1 = [c for _,c in kw_anom_rwfb]
    map_c = dict(kw_anom_cetras)
    vals_a2 = [map_c.get(w, 0) for w in labels_a]
    bar_compare(labels_a, vals_a1, vals_a2, "Anomaly Keywords: RWFB vs CETraS (Top-1000)", 
                os.path.join(OUT_DIR, "compare_anomaly_keywords.png"), "RWFB", "CETraS")
    
    # Summary keywords figure
    labels_s = [w for w,_ in kw_sum_rwfb]
    vals_s1 = [c for _,c in kw_sum_rwfb]
    map_sc = dict(kw_sum_cetras)
    vals_s2 = [map_sc.get(w, 0) for w in labels_s]
    bar_compare(labels_s, vals_s1, vals_s2, "Summary Keywords: RWFB vs CETraS (Top-1000)", 
                os.path.join(OUT_DIR, "compare_summary_keywords.png"), "RWFB", "CETraS")
    
    # Calculate metrics
    def dist_from_kw(kws):
        total = sum(c for _,c in kws) or 1
        return {w:c/total for w,c in kws}
    
    jsd_anom = js_divergence(dist_from_kw(kw_anom_rwfb), dist_from_kw(kw_anom_cetras))
    jsd_sum = js_divergence(dist_from_kw(kw_sum_rwfb), dist_from_kw(kw_sum_cetras))
    
    # SDM
    SDM = 0.5*jsd_roles + 0.3*jsd_anom + 0.2*jsd_sum
    
    print(f"Keywords JSD (anomaly): {jsd_anom:.4f}")
    print(f"Keywords JSD (summary): {jsd_sum:.4f}")
    print(f"SDM: {SDM:.4f}")
    
    # Load edges for CI
    rwfb_edges = load_graph_edges(os.path.join(CURRENT_DIR, "data", "llm4tg_edges_top1000_rwfb.csv"))
    cetras_edges = load_graph_edges(os.path.join(CURRENT_DIR, "data", "llm4tg_edges_top1000_cetras.csv"))
    
    gini_rwfb = degree_gini(rwfb_edges)
    gini_cetras = degree_gini(cetras_edges)
    
    # CI
    def summarize_label(text: str) -> int:
        t = text.lower()
        pos = ["decentral", "multiple hubs", "small-world", "resilien", "distributed"]
        neg = ["centraliz", "single hub", "dominant hub", "fragil", "concentrat"]
        score = sum(1 for p in pos if p in t) - sum(1 for n in neg if n in t)
        return 1 if score >= 0 else 0
    
    llm_label_rwfb = summarize_label(sum_rwfb)
    llm_label_cetras = summarize_label(sum_cetras)
    
    CI_rwfb = 1.0 - abs((1.0 - gini_rwfb) - llm_label_rwfb)
    CI_cetras = 1.0 - abs((1.0 - gini_cetras) - llm_label_cetras)
    
    print(f"Gini (RWFB/CETraS): {gini_rwfb:.4f} / {gini_cetras:.4f}")
    print(f"CI (RWFB/CETraS): {CI_rwfb:.4f} / {CI_cetras:.4f}")
    
    # Permutation test
    def perm_test_jsd(labels_a, labels_b, iters=500):
        pool = labels_a + labels_b
        n = len(labels_a)
        obs_jsd = js_divergence(dist_from_counts(Counter(labels_a)), dist_from_counts(Counter(labels_b)))
        count = sum(1 for _ in range(iters) 
                   if js_divergence(dist_from_counts(Counter((random.shuffle(pool), pool[:n])[1])), 
                                   dist_from_counts(Counter(pool[n:]))) >= obs_jsd - 1e-12)
        return count / iters
    
    p_jsd = perm_test_jsd(rwfb_roles, cetras_roles, 500) if rwfb_roles and cetras_roles else 1.0
    print(f"Permutation p-value: {p_jsd:.4f}")
    
    # Save tables
    smd_rows = [
        ["roles_jsd_bits", f"{jsd_roles:.4f}"],
        ["keywords_jsd_anomaly", f"{jsd_anom:.4f}"],
        ["keywords_jsd_summary", f"{jsd_sum:.4f}"],
        ["SDM_weighted", f"{SDM:.4f}"],
        ["gini_rwfb", f"{gini_rwfb:.4f}"],
        ["gini_cetras", f"{gini_cetras:.4f}"],
        ["CI_rwfb", f"{CI_rwfb:.4f}"],
        ["CI_cetras", f"{CI_cetras:.4f}"],
        ["perm_pvalue_jsd", f"{p_jsd:.4f}"],
    ]
    
    save_table_md(os.path.join(OUT_DIR, "smd_ci_table.md"), ["metric","value"], smd_rows)
    save_table_png(os.path.join(OUT_DIR, "smd_ci_table.png"), ["metric","value"], smd_rows, 
                   title="SDM / CI Metrics (Top-1000, Accurate CETraS)")
    
    # Report
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("# RWFB vs CETraS Comparison (Top-1000, Accurate CETraS)\n\n")
        f.write("## Data Source\n")
        f.write("- **Dataset**: 2020-10_00.csv (23M+ transactions)\n")
        f.write("- **Sampling**: Both 30K nodes → Top-1000\n")
        f.write(f"- **RWFB Edges**: {len(rwfb_edges)} (Top-1000)\n")
        f.write(f"- **CETraS Edges**: {len(cetras_edges)} (Top-1000)\n\n")
        f.write("## Key Metrics\n")
        f.write(f"- **Role JSD**: {jsd_roles:.4f} bits (p={p_jsd:.4f})\n")
        f.write(f"- **Keywords JSD**: anomaly={jsd_anom:.4f}, summary={jsd_sum:.4f}\n")
        f.write(f"- **SDM**: {SDM:.4f}\n")
        f.write(f"- **Gini**: RWFB={gini_rwfb:.3f}, CETraS={gini_cetras:.3f}\n")
        f.write(f"- **CI**: RWFB={CI_rwfb:.3f}, CETraS={CI_cetras:.3f}\n\n")
        f.write(f"## Role Counts\n")
        f.write(f"- **RWFB**: {len(rwfb_roles)} valid roles\n")
        f.write(f"- **CETraS**: {len(cetras_roles)} valid roles\n")
    
    print(f"\n✅ 对比分析完成！")
    print(f"📁 结果保存到: {OUT_DIR}")
    print(f"📊 生成文件: {len(os.listdir(OUT_DIR))}")


if __name__ == "__main__":
    main()
