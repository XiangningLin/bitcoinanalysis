#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Top-1000 Batched Analysis (直接调用 OpenAI，不用 Agent 框架)
"""

import os
import sys
import re
import json
import time
from typing import Dict, List
from openai import OpenAI

CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def split_batches(items: List, size: int) -> List[List]:
    return [items[i:i+size] for i in range(0, len(items), size)]


def filter_edges(edges_csv: str, node_ids: set, max_lines: int = 1000) -> str:
    lines = ["source,target\n"]
    count = 0
    for line in edges_csv.split('\n')[1:]:
        if not line.strip() or count >= max_lines:
            break
        parts = line.split(',')
        if len(parts) >= 2:
            src, dst = parts[0].strip(), parts[1].strip()
            if src in node_ids and dst in node_ids:
                lines.append(line + '\n')
                count += 1
    return ''.join(lines)


def extract_prompts(txt: str) -> Dict[str, str]:
    def grab(title: str) -> str:
        m = re.search(rf"### {re.escape(title)}([\s\S]*?)(?=###|\Z)", txt)
        return (m.group(1).strip() if m else "")
    return {
        "p1": grab("Prompt 1 — Node Role Classification") or grab("Prompt 1"),
        "p2": grab("Prompt 2 — Anomaly Pattern Explanation") or grab("Prompt 2"),
        "p3": grab("Prompt 3 — Decentralization Snapshot Summary") or grab("Prompt 3"),
    }


def call_openai(client: OpenAI, prompt: str, max_retries: int = 3) -> str:
    """调用 OpenAI API 并处理重试"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "你是区块链交易网络分析助理，请严格按要求输出 JSON 或文本。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  尝试 {attempt+1}/{max_retries} 失败: {e}")
            if "rate_limit" in str(e).lower():
                wait_time = 30
                print(f"  ⏳ 速率限制，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                time.sleep(5)
    
    return f"[ERROR after {max_retries} retries]"


def parse_json_response(response: str) -> List[Dict]:
    """解析 JSON 响应"""
    # Strategy 1: markdown code block
    match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Strategy 2: find array
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
        else:
            return [{"_raw": response[:300], "_error": "No JSON found"}]
    
    try:
        data = json.loads(json_str)
        return data if isinstance(data, list) else data.get("items", [{"_raw": json_str}])
    except Exception as e:
        return [{"_raw": json_str[:300], "_error": str(e)}]


def analyze_roles_batched(client: OpenAI, tag: str, nodes_file: str, edges_file: str, base_prompt: str):
    """分批角色分类"""
    print(f"\n{'='*60}")
    print(f"🔵 {tag} 角色分类（分批）")
    print(f"{'='*60}")
    
    # Load data
    nodes = read_jsonl(os.path.join(CURRENT_DIR, "data", nodes_file))
    edges_csv = read_text(os.path.join(CURRENT_DIR, "data", edges_file))
    
    batches = split_batches(nodes, 200)
    print(f"共 {len(nodes)} 节点，分成 {len(batches)} 批")
    
    all_results = []
    
    for idx, batch in enumerate(batches, 1):
        print(f"\n🔄 批次 {idx}/{len(batches)} ({len(batch)} 节点)...")
        
        # Build batch data
        batch_jsonl = '\n'.join(json.dumps(n, ensure_ascii=False) for n in batch)
        batch_ids = set(n['id'] for n in batch)
        batch_edges = filter_edges(edges_csv, batch_ids, max_lines=500)
        
        # Build prompt
        prompt = (
            f"[Task] {tag} Batch {idx}/{len(batches)} - Node Role Classification\n"
            f"节点画像（JSONL，{len(batch)}个节点）：\n```\n{batch_jsonl}\n```\n"
            f"诱导边（CSV，部分）：\n```\n{batch_edges}\n```\n"
            f"{base_prompt}"
        )
        
        # Call OpenAI
        response = call_openai(client, prompt)
        
        # Parse
        batch_results = parse_json_response(response)
        all_results.extend(batch_results)
        
        valid_count = sum(1 for r in batch_results if "_error" not in r and "_raw" not in r)
        print(f"  ✅ 完成，有效结果: {valid_count}/{len(batch_results)}")
        
        # Delay
        if idx < len(batches):
            print(f"  ⏳ 等待 20 秒...")
            time.sleep(20)
    
    print(f"\n✅ {tag} 角色分类完成，共 {len(all_results)} 条结果")
    return all_results


def analyze_text(client: OpenAI, tag: str, nodes_file: str, edges_file: str, base_prompt: str, task_name: str):
    """分析文本任务（异常/摘要）"""
    print(f"\n📝 {tag} {task_name}...")
    
    # Sample nodes for context
    nodes = read_jsonl(os.path.join(CURRENT_DIR, "data", nodes_file))
    edges_csv = read_text(os.path.join(CURRENT_DIR, "data", edges_file))
    
    import random
    random.seed(7)
    sampled = random.sample(nodes, min(200, len(nodes)))
    sampled_ids = set(n['id'] for n in sampled)
    
    nodes_blob = '\n'.join(json.dumps(n, ensure_ascii=False) for n in sampled)
    edges_blob = filter_edges(edges_csv, sampled_ids, max_lines=500)
    
    prompt = (
        f"[Task] {tag} - {task_name}\n"
        f"节点画像（JSONL，采样200个）：\n```\n{nodes_blob}\n```\n"
        f"诱导边（CSV）：\n```\n{edges_blob}\n```\n"
        f"{base_prompt}"
    )
    
    response = call_openai(client, prompt)
    print(f"  ✅ 完成")
    return response


def main():
    print("="*70)
    print("Top-1000 简化分批分析 (直接调用 OpenAI)")
    print("="*70)
    
    # Setup
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 请设置 OPENAI_API_KEY")
        return
    
    client = OpenAI(api_key=api_key)
    prompts = extract_prompts(read_text(os.path.join(BASE_DIR, "llm4tg_prompts.txt")))
    
    # RWFB Analysis
    print("\n" + "🔵"*35)
    rwfb_roles = analyze_roles_batched(client, "RWFB", 
                                       "llm4tg_nodes_top1000_rwfb.jsonl",
                                       "llm4tg_edges_top1000_rwfb.csv",
                                       prompts["p1"])
    
    rwfb_anom = analyze_text(client, "RWFB",
                             "llm4tg_nodes_top1000_rwfb.jsonl",
                             "llm4tg_edges_top1000_rwfb.csv",
                             prompts["p2"], "异常分析")
    
    rwfb_sum = analyze_text(client, "RWFB",
                            "llm4tg_nodes_top1000_rwfb.jsonl",
                            "llm4tg_edges_top1000_rwfb.csv",
                            prompts["p3"], "去中心化摘要")
    
    # Save RWFB
    out_dir = os.path.join(CURRENT_DIR, "outputs", "rwfb_top1000")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "rwfb_role_predictions.json"), 'w') as f:
        json.dump(rwfb_roles, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "rwfb_anomaly_explained.txt"), 'w') as f:
        f.write(rwfb_anom)
    with open(os.path.join(out_dir, "rwfb_decentralization_summary.txt"), 'w') as f:
        f.write(rwfb_sum)
    
    print(f"\n✅ RWFB 结果保存到: {out_dir}")
    
    # CETraS Analysis
    print("\n" + "🟢"*35)
    cetras_roles = analyze_roles_batched(client, "CETraS",
                                         "llm4tg_nodes_top1000_cetras.jsonl",
                                         "llm4tg_edges_top1000_cetras.csv",
                                         prompts["p1"])
    
    cetras_anom = analyze_text(client, "CETraS",
                               "llm4tg_nodes_top1000_cetras.jsonl",
                               "llm4tg_edges_top1000_cetras.csv",
                               prompts["p2"], "异常分析")
    
    cetras_sum = analyze_text(client, "CETraS",
                              "llm4tg_nodes_top1000_cetras.jsonl",
                              "llm4tg_edges_top1000_cetras.csv",
                              prompts["p3"], "去中心化摘要")
    
    # Save CETraS
    out_dir = os.path.join(CURRENT_DIR, "outputs", "cetras_top1000")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "cetras_role_predictions.json"), 'w') as f:
        json.dump(cetras_roles, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "cetras_anomaly_explained.txt"), 'w') as f:
        f.write(cetras_anom)
    with open(os.path.join(out_dir, "cetras_decentralization_summary.txt"), 'w') as f:
        f.write(cetras_sum)
    
    print(f"\n✅ CETraS 结果保存到: {out_dir}")
    
    print("\n" + "="*70)
    print("🎉 所有分析完成！现在可以运行对比分析。")
    print("="*70)


if __name__ == "__main__":
    main()
