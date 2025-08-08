#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RWFB vs CETraS comparison script
- Compares role distributions (with Jensen–Shannon divergence)
- Compares anomaly/summary keywords
- Writes figures and a brief markdown report under ma/outputs/compare/
"""

import os
import re
import io
import csv
import json
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
OUT_DIR = os.path.join(CURRENT_DIR, "outputs", "compare")
os.makedirs(OUT_DIR, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def find_input_paths() -> Dict[str, str]:
    """Resolve input files, preferring ma/outputs/<tag>/ first, fallback to BASE_DIR.
    Returns dict with keys: rwfb_roles, rwfb_anom, rwfb_sum, cetras_roles, cetras_anom, cetras_sum
    """
    paths = {}
    # Preferred locations under ma/outputs
    pref = {
        "rwfb_roles": os.path.join(CURRENT_DIR, "outputs", "rwfb", "rwfb_role_predictions.json"),
        "rwfb_anom": os.path.join(CURRENT_DIR, "outputs", "rwfb", "rwfb_anomaly_explained.txt"),
        "rwfb_sum":  os.path.join(CURRENT_DIR, "outputs", "rwfb", "rwfb_decentralization_summary.txt"),
        "cetras_roles": os.path.join(CURRENT_DIR, "outputs", "cetras", "cetras_role_predictions.json"),
        "cetras_anom":  os.path.join(CURRENT_DIR, "outputs", "cetras", "cetras_anomaly_explained.txt"),
        "cetras_sum":   os.path.join(CURRENT_DIR, "outputs", "cetras", "cetras_decentralization_summary.txt"),
    }
    # Alternative preferred paths (enhanced pipeline outputs)
    alt = {
        "cetras_roles": os.path.join(CURRENT_DIR, "outputs", "cetras_plus", "cetras_role_predictions.json"),
        "cetras_anom":  os.path.join(CURRENT_DIR, "outputs", "cetras_plus", "cetras_anomaly_explained.txt"),
        "cetras_sum":   os.path.join(CURRENT_DIR, "outputs", "cetras_plus", "cetras_decentralization_summary.txt"),
    }
    fb = {
        "rwfb_roles": os.path.join(BASE_DIR, "rwfb_role_predictions.json"),
        "rwfb_anom": os.path.join(BASE_DIR, "rwfb_anomaly_explained.txt"),
        "rwfb_sum":  os.path.join(BASE_DIR, "rwfb_decentralization_summary.txt"),
        "cetras_roles": os.path.join(BASE_DIR, "cetras_role_predictions.json"),
        "cetras_anom":  os.path.join(BASE_DIR, "cetras_anomaly_explained.txt"),
        "cetras_sum":   os.path.join(BASE_DIR, "cetras_decentralization_summary.txt"),
    }
    for k, p in pref.items():
        if os.path.exists(p):
            paths[k] = p
        elif k in alt and os.path.exists(alt[k]):
            paths[k] = alt[k]
        else:
            paths[k] = fb[k]
    return paths


def extract_roles(items: List[dict]) -> List[str]:
    roles = []
    for item in items:
        if isinstance(item, dict) and "role" in item:
            roles.append(str(item["role"]))
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
    cnt = Counter(tokenize(text))
    return cnt.most_common(k)


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen–Shannon divergence in bits. Inputs are probability dicts.
    """
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
    w = 0.4
    plt.figure(figsize=(10,4))
    plt.bar(x - w/2, v1, width=w, label=l1)
    plt.bar(x + w/2, v2, width=w, label=l2)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_table_md(path: str, headers: List[str], rows: List[List[str]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join([" --- " for _ in headers]) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")


def save_table_png(path: str, headers: List[str], rows: List[List[str]], title: str = None):
    """Render a simple table to PNG using matplotlib."""
    n_rows = max(1, len(rows))
    fig_height = 0.6 + 0.35 * n_rows
    fig, ax = plt.subplots(figsize=(6.5, fig_height))
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=12)
    table = ax.table(cellText=rows, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    fig.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    paths = find_input_paths()

    # Roles
    roles_rwfb = extract_roles(load_json(paths["rwfb_roles"]))
    roles_cet  = extract_roles(load_json(paths["cetras_roles"]))
    cnt_rwfb = Counter(roles_rwfb)
    cnt_cet  = Counter(roles_cet)
    all_roles = sorted(set(cnt_rwfb) | set(cnt_cet))
    dist_rwfb = dist_from_counts(cnt_rwfb)
    dist_cet  = dist_from_counts(cnt_cet)
    jsd = js_divergence(dist_rwfb, dist_cet)

    # Role compare figure
    v1 = [cnt_rwfb.get(r, 0) for r in all_roles]
    v2 = [cnt_cet.get(r, 0) for r in all_roles]
    bar_compare(all_roles, v1, v2, "Role Distribution: RWFB vs CETraS", os.path.join(OUT_DIR, "compare_roles.png"), "RWFB", "CETraS")

    # Keywords
    anom_rwfb = read_text(paths["rwfb_anom"]) if os.path.exists(paths["rwfb_anom"]) else ""
    anom_cet  = read_text(paths["cetras_anom"]) if os.path.exists(paths["cetras_anom"]) else ""
    sum_rwfb  = read_text(paths["rwfb_sum"]) if os.path.exists(paths["rwfb_sum"]) else ""
    sum_cet   = read_text(paths["cetras_sum"]) if os.path.exists(paths["cetras_sum"]) else ""

    kw_anom_rwfb = top_keywords(anom_rwfb, 15)
    kw_anom_cet  = top_keywords(anom_cet, 15)
    kw_sum_rwfb  = top_keywords(sum_rwfb, 15)
    kw_sum_cet   = top_keywords(sum_cet, 15)

    # Anomaly keywords figure
    labels_a = [w for w,_ in kw_anom_rwfb]
    vals_a1 = [c for _,c in kw_anom_rwfb]
    # align by RWFB labels for left plot
    map_c = dict(kw_anom_cet)
    vals_a2 = [map_c.get(w, 0) for w in labels_a]
    bar_compare(labels_a, vals_a1, vals_a2, "Anomaly Keywords: RWFB vs CETraS", os.path.join(OUT_DIR, "compare_anomaly_keywords.png"), "RWFB", "CETraS")

    # Summary keywords figure
    labels_s = [w for w,_ in kw_sum_rwfb]
    vals_s1 = [c for _,c in kw_sum_rwfb]
    map_sc = dict(kw_sum_cet)
    vals_s2 = [map_sc.get(w, 0) for w in labels_s]
    bar_compare(labels_s, vals_s1, vals_s2, "Summary Keywords: RWFB vs CETraS", os.path.join(OUT_DIR, "compare_summary_keywords.png"), "RWFB", "CETraS")

    # Save simple metrics table
    rows = []
    for r in all_roles:
        rows.append([r, cnt_rwfb.get(r,0), cnt_cet.get(r,0), round(dist_rwfb.get(r,0.0),4), round(dist_cet.get(r,0.0),4)])
    save_table_md(os.path.join(OUT_DIR, "roles_table.md"), ["role","rwfb_count","cetras_count","rwfb_frac","cetras_frac"], rows)

    # ------- Semantic Drift Metric (SMD): combine role JSD + keyword JSDs -------
    def dist_from_top_kw(kws: List[Tuple[str,int]]) -> Dict[str,float]:
        total = sum(c for _,c in kws) or 1
        return {w:c/total for w,c in kws}
    dist_anom_rwfb = dist_from_top_kw(kw_anom_rwfb)
    dist_anom_cet  = dist_from_top_kw(kw_anom_cet)
    dist_sum_rwfb  = dist_from_top_kw(kw_sum_rwfb)
    dist_sum_cet   = dist_from_top_kw(kw_sum_cet)
    jsd_anom = js_divergence(dist_anom_rwfb, dist_anom_cet)
    jsd_sum  = js_divergence(dist_sum_rwfb, dist_sum_cet)
    # Weighted combination (roles heavier)
    SMD = 0.5*jsd + 0.3*jsd_anom + 0.2*jsd_sum

    # ------- Consistency Index (CI): LLM summary vs graph decentralization score -------
    def find_edges_paths() -> Tuple[str,str]:
        rwfb_edges = os.path.join(BASE_DIR, "llm4tg_edges_top100_rwfb.csv")
        cetras_edges = os.path.join(BASE_DIR, "llm4tg_edges_top100_cetras_fast.csv")
        return rwfb_edges, cetras_edges

    def load_graph_edges(path: str) -> List[Tuple[str,str]]:
        if not os.path.exists(path):
            return []
        edges = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    u, v = row[0], row[1]
                    if u and v:
                        edges.append((u,v))
        return edges

    def degree_gini(edges: List[Tuple[str,str]]) -> float:
        deg = Counter()
        for u,v in edges:
            deg[u]+=1; deg[v]+=1
        vals = sorted(deg.values())
        if not vals:
            return 0.0
        n = len(vals)
        cum = 0
        for i, x in enumerate(vals, 1):
            cum += i * x
        total = sum(vals)
        return (2*cum)/(n*total) - (n+1)/n  # Gini in [0,1]

    def summarize_label(text: str) -> int:
        t = text.lower()
        pos = ["decentral", "multiple hubs", "small-world", "resilien", "distributed"]
        neg = ["centraliz", "single hub", "dominant hub", "fragil", "concentrat"]
        score = sum(1 for p in pos if p in t) - sum(1 for n in neg if n in t)
        return 1 if score >= 0 else 0

    rwfb_edges_path, cetras_edges_path = find_edges_paths()
    rwfb_edges = load_graph_edges(rwfb_edges_path)
    cetras_edges = load_graph_edges(cetras_edges_path)
    gini_rwfb = degree_gini(rwfb_edges)
    gini_cet  = degree_gini(cetras_edges)
    # Map to decentralization score: higher decentralization -> lower gini
    dec_score_rwfb = 1.0 - gini_rwfb
    dec_score_cet  = 1.0 - gini_cet
    llm_label_rwfb = summarize_label(sum_rwfb)
    llm_label_cet  = summarize_label(sum_cet)
    # CI = 1 - abs(score - label)
    CI_rwfb = 1.0 - abs(dec_score_rwfb - llm_label_rwfb)
    CI_cet  = 1.0 - abs(dec_score_cet  - llm_label_cet)

    # ------- Weak-label anomaly evaluation & significance -------
    # Baseline detectors on edges: fan-in/out & batching heuristics
    def baseline_anomalies(edges: List[Tuple[str,str]]) -> Dict[str,int]:
        in_deg = Counter(); out_deg = Counter()
        for u,v in edges:
            out_deg[u]+=1; in_deg[v]+=1
        fin = sum(1 for n in set(in_deg)|set(out_deg) if in_deg[n] >= 5 and out_deg[n] <= 1)
        fout = sum(1 for n in set(in_deg)|set(out_deg) if out_deg[n] >= 5 and in_deg[n] <= 1)
        batch = sum(1 for n in set(in_deg)|set(out_deg) if out_deg[n] >= 10)
        return {"fan_in": fin, "fan_out": fout, "batch": batch}

    base_rwfb = baseline_anomalies(rwfb_edges)
    base_cet  = baseline_anomalies(cetras_edges)
    # Keyword coverage in LLM anomalies
    def coverage(text: str, keys: List[str]) -> float:
        t = text.lower()
        hits = sum(1 for k in keys if k in t)
        return hits / max(1,len(keys))
    keys = ["peeling", "coinjoin", "mixer", "fan-in", "fanin", "fan-out", "fanout", "batch", "consolidation"]
    cov_rwfb = coverage(anom_rwfb, keys)
    cov_cet  = coverage(anom_cet, keys)

    # Permutation test for role JSD significance
    def perm_test_jsd(labels_a: List[str], labels_b: List[str], iters: int = 1000) -> float:
        pool = labels_a + labels_b
        n = len(labels_a)
        obs_jsd = js_divergence(dist_from_counts(Counter(labels_a)), dist_from_counts(Counter(labels_b)))
        count = 0
        for _ in range(iters):
            random.shuffle(pool)
            A = pool[:n]; B = pool[n:]
            j = js_divergence(dist_from_counts(Counter(A)), dist_from_counts(Counter(B)))
            if j >= obs_jsd - 1e-12:
                count += 1
        return count / iters
    p_jsd = perm_test_jsd(roles_rwfb, roles_cet, iters=500)

    # ------- Save extended tables -------
    smd_rows = [
        ["roles_jsd_bits", f"{jsd:.4f}"],
        ["keywords_jsd_anomaly", f"{jsd_anom:.4f}"],
        ["keywords_jsd_summary", f"{jsd_sum:.4f}"],
        ["SMD_weighted", f"{SMD:.4f}"],
        ["gini_rwfb", f"{gini_rwfb:.4f}"],
        ["gini_cetras", f"{gini_cet:.4f}"],
        ["CI_rwfb", f"{CI_rwfb:.4f}"],
        ["CI_cetras", f"{CI_cet:.4f}"],
        ["perm_pvalue_jsd", f"{p_jsd:.4f}"],
    ]
    save_table_md(os.path.join(OUT_DIR, "smd_ci_table.md"), ["metric","value"], smd_rows)
    save_table_png(os.path.join(OUT_DIR, "smd_ci_table.png"), ["metric","value"], smd_rows, title="SMD / CI Metrics")

    save_table_md(
        os.path.join(OUT_DIR, "anomalies_table.md"),
        ["dataset","fan_in","fan_out","batch","keyword_coverage"],
        [
            ["RWFB", base_rwfb["fan_in"], base_rwfb["fan_out"], base_rwfb["batch"], f"{cov_rwfb:.2f}"],
            ["CETraS", base_cet["fan_in"], base_cet["fan_out"], base_cet["batch"], f"{cov_cet:.2f}"],
        ]
    )

    # ------- Optional: Address label alignment (placeholder) -------
    label_path = os.path.join(BASE_DIR, "labels", "address_labels.json")
    label_eval_rows = []
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            addr_labels = json.load(f)  # {address: label}
        # try evaluate RWFB predictions
        def eval_labels(pred_path: str, name: str):
            preds = load_json(pred_path)
            hit = tot = 0
            for item in preds:
                if not isinstance(item, dict):
                    continue
                addr = item.get("id") or item.get("address")
                role = item.get("role")
                if addr in addr_labels and role:
                    tot += 1
                    if str(addr_labels[addr]).lower() in str(role).lower():
                        hit += 1
            acc = hit / tot if tot else 0.0
            label_eval_rows.append([name, tot, f"{acc:.2f}"])

        # Try both ma outputs and base fallbacks
        rwfb_pred = paths["rwfb_roles"]
        cetras_pred = paths["cetras_roles"]
        eval_labels(rwfb_pred, "RWFB")
        eval_labels(cetras_pred, "CETraS")

        save_table_md(os.path.join(OUT_DIR, "label_eval_table.md"), ["dataset","labeled_overlap","match_acc"], label_eval_rows)

    # Report (extended)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("# RWFB vs CETraS Comparison (Extended)\n\n")
        f.write(f"- Roles JSD (bits): {jsd:.4f} (perm p={p_jsd:.4f})\n")
        f.write(f"- Keywords JSD (anomaly/summary): {jsd_anom:.4f} / {jsd_sum:.4f}\n")
        f.write(f"- SMD (weighted): {SMD:.4f}\n")
        f.write(f"- Gini (RWFB/CETraS): {gini_rwfb:.3f} / {gini_cet:.3f} → CI: {CI_rwfb:.3f} / {CI_cet:.3f}\n")
        if label_eval_rows:
            f.write("- Label alignment: see label_eval_table.md\n")
        f.write("\nFigures added:\n")
        f.write("- compare_roles.png\n- compare_anomaly_keywords.png\n- compare_summary_keywords.png\n")
        f.write("\nTables added:\n")
        f.write("- roles_table.md\n- smd_ci_table.md\n- anomalies_table.md\n")
        if label_eval_rows:
            f.write("- label_eval_table.md\n")


if __name__ == "__main__":
    main()


