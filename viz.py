#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz.py — 读取 6 份 LLM 输出文件，生成 6 张图：
1) rwfb_role_distribution.png
2) cetras_role_distribution.png
3) rwfb_anomaly_keywords.png
4) cetras_anomaly_keywords.png
5) rwfb_summary_keywords.png
6) cetras_summary_keywords.png
"""

import os, json, re, sys
from collections import Counter
import matplotlib.pyplot as plt

# ---------- 配置：六个输入文件（放在脚本同目录） ----------
FILES = {
    "rwfb_roles": "rwfb_role_predictions.json",
    "cetras_roles": "cetras_role_predictions.json",
    "rwfb_anom": "rwfb_anomaly_explained.txt",
    "cetras_anom": "cetras_anomaly_explained.txt",
    "rwfb_sum": "rwfb_decentralization_summary.txt",
    "cetras_sum": "cetras_decentralization_summary.txt",
}

# ---------- 工具函数 ----------
def check_files():
    missing = [p for p in FILES.values() if not os.path.exists(p)]
    if missing:
        print("❌ 缺少文件：", ", ".join(missing))
        print("请把 6 个结果文件放在和 viz.py 同一目录。")
        sys.exit(1)

def load_roles(path):
    # 期望 JSON 数组，每项包含 role；如果模型有时返回原文，做兜底解析
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    roles = []
    for item in data:
        if isinstance(item, dict) and "role" in item:
            roles.append(item["role"])
        elif isinstance(item, dict) and "_raw" in item:
            # 若模型把 JSON 放在字符串里，尝试提取
            m = re.search(r"\[[\s\S]*\]", item["_raw"])
            if m:
                try:
                    arr = json.loads(m.group(0))
                    for x in arr:
                        if isinstance(x, dict) and "role" in x:
                            roles.append(x["role"])
                except Exception:
                    pass
    return roles

STOPWORDS = set("""
a an the and or of to in on for with from by is are was were be being been this that these those
as at it its their his her they them we you your our ours can could should may might will would
not no into over under above below more less most least very really just also etc based using use than
then thus hence such via per about across among between within without during after before near around
one two three four five six seven eight nine ten
""".split())

def tokenize(text: str):
    words = re.findall(r"[A-Za-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def top_keywords(text: str, k=15):
    tokens = tokenize(text)
    cnt = Counter(tokens)
    return cnt.most_common(k)

def bar_plot(labels, values, title, outfile, output_dir="figures"):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, outfile)
    
    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"✅ Saved: {output_path}")

# ---------- 主流程 ----------
def main():
    check_files()

    # 角色分布
    roles_rwfb = load_roles(FILES["rwfb_roles"])
    roles_cet  = load_roles(FILES["cetras_roles"])
    cnt_rwfb = Counter(roles_rwfb)
    cnt_cet  = Counter(roles_cet)
    all_roles = sorted(set(cnt_rwfb) | set(cnt_cet))
    vals_rwfb = [cnt_rwfb.get(r, 0) for r in all_roles]
    vals_cet  = [cnt_cet.get(r, 0) for r in all_roles]

    bar_plot(all_roles, vals_rwfb, "RWFB Top100 — Role Distribution", "rwfb_role_distribution.png")
    bar_plot(all_roles, vals_cet,  "CETraS-like Top100 — Role Distribution", "cetras_role_distribution.png")

    # 异常关键词
    anom_rwfb = open(FILES["rwfb_anom"], "r", encoding="utf-8").read()
    anom_cet  = open(FILES["cetras_anom"], "r", encoding="utf-8").read()
    kw_anom_rwfb = top_keywords(anom_rwfb, 15)
    kw_anom_cet  = top_keywords(anom_cet, 15)
    bar_plot([w for w,_ in kw_anom_rwfb], [c for _,c in kw_anom_rwfb],
             "RWFB — Anomaly Keywords", "rwfb_anomaly_keywords.png")
    bar_plot([w for w,_ in kw_anom_cet],  [c for _,c in kw_anom_cet],
             "CETraS-like — Anomaly Keywords", "cetras_anomaly_keywords.png")

    # 摘要关键词
    sum_rwfb = open(FILES["rwfb_sum"], "r", encoding="utf-8").read()
    sum_cet  = open(FILES["cetras_sum"], "r", encoding="utf-8").read()
    kw_sum_rwfb = top_keywords(sum_rwfb, 15)
    kw_sum_cet  = top_keywords(sum_cet, 15)
    bar_plot([w for w,_ in kw_sum_rwfb], [c for _,c in kw_sum_rwfb],
             "RWFB — Summary Keywords", "rwfb_summary_keywords.png")
    bar_plot([w for w,_ in kw_sum_cet],  [c for _,c in kw_sum_cet],
             "CETraS-like — Summary Keywords", "cetras_summary_keywords.png")

    print("\n🎉 Done. 6 张图片已生成到 figures/ 文件夹，直接用 LaTeX 模板 \\includegraphics{figures/xxx.png} 引用即可。")

if __name__ == "__main__":
    main()
