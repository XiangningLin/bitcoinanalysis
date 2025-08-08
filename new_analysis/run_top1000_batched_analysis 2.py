#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batched Multi-Agent Analysis for Top-1000 Nodes
将 Top-1000 分批处理，避免 token 超限
"""

import os
import sys
import re
import json
import asyncio
from typing import Dict, List

CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_framework.core import AgentCoordinator, Task, TaskPriority, Message, MessageType
from agent_framework.agents import RoleClassifierAgent, AnomalyAnalystAgent, DecentralizationSummarizerAgent
from agent_framework.agents.llm_agent import LLMProvider


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def split_nodes_into_batches(nodes: List[Dict], batch_size: int = 200) -> List[List[Dict]]:
    """Split nodes into batches"""
    return [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]


def filter_edges_for_nodes(edges_csv: str, node_ids: set) -> str:
    """Filter edges to only include those between given nodes"""
    filtered_lines = ["source,target\n"]
    for line in edges_csv.split('\n')[1:]:  # Skip header
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            src, dst = parts[0].strip(), parts[1].strip()
            if src in node_ids and dst in node_ids:
                filtered_lines.append(line + '\n')
    return ''.join(filtered_lines)


def extract_prompts(txt: str) -> Dict[str, str]:
    def grab(title: str) -> str:
        m = re.search(rf"### {re.escape(title)}([\s\S]*?)(?=###|\Z)", txt)
        return (m.group(1).strip() if m else "")
    return {
        "p1": grab("Prompt 1 — Node Role Classification") or grab("Prompt 1"),
        "p2": grab("Prompt 2 — Anomaly Pattern Explanation") or grab("Prompt 2"),
        "p3": grab("Prompt 3 — Decentralization Snapshot Summary") or grab("Prompt 3"),
    }


class OpenAICompatProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = None, temperature: float = 0.2):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._temperature = temperature

    async def generate_response(self, messages, **kwargs) -> str:
        import asyncio
        def _call():
            resp = self._client.chat.completions.create(
                model=kwargs.get("model", self._model), 
                temperature=kwargs.get("temperature", self._temperature),
                messages=messages, 
                max_tokens=kwargs.get("max_tokens", 2000),  # Increased for batch
            )
            return resp.choices[0].message.content
        return await asyncio.to_thread(_call)

    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "OpenAI", "model": self._model}


async def analyze_role_batched(tag: str, nodes_file: str, edges_file: str, base_prompt: str, batch_size: int = 200):
    """Analyze roles in batches"""
    
    # Load data
    nodes_path = os.path.join(CURRENT_DIR, "data", nodes_file)
    edges_path = os.path.join(CURRENT_DIR, "data", edges_file)
    
    all_nodes = read_jsonl(nodes_path)
    edges_csv = read_text(edges_path)
    
    print(f"\n{'='*60}")
    print(f"分批角色分析: {tag}")
    print(f"总节点数: {len(all_nodes)}, 批次大小: {batch_size}")
    print(f"{'='*60}")
    
    # Split into batches
    batches = split_nodes_into_batches(all_nodes, batch_size)
    print(f"分成 {len(batches)} 批")
    
    # Setup LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 未设置 OPENAI_API_KEY")
        return []
    
    provider = OpenAICompatProvider(api_key, model="gpt-4o-mini")
    role_agent = RoleClassifierAgent(name=f"{tag}RoleBatch", llm_provider=provider)
    role_agent.llm_config["use_history"] = False
    role_agent.llm_config["max_tokens"] = 2000
    
    await role_agent.start()
    
    # Process each batch
    all_results = []
    
    for batch_idx, batch_nodes in enumerate(batches, 1):
        print(f"\n🔄 处理批次 {batch_idx}/{len(batches)} ({len(batch_nodes)} 节点)...")
        
        # Create batch JSONL
        batch_jsonl = '\n'.join(json.dumps(node, ensure_ascii=False) for node in batch_nodes)
        
        # Filter edges for this batch
        batch_node_ids = set(node['id'] for node in batch_nodes)
        batch_edges = filter_edges_for_nodes(edges_csv, batch_node_ids)
        
        # Build prompt
        prompt = (
            f"[Task] {tag} / Prompt 1 (Batch {batch_idx}/{len(batches)}) - Node Role Classification\n"
            f"节点画像（JSONL，Batch {batch_idx}，共{len(batch_nodes)}个节点）：\n```\n{batch_jsonl}\n```\n"
            f"这批节点的诱导边（CSV）：\n```\n{batch_edges[:5000]}\n```\n"  # Limit edge preview
            f"{base_prompt}"
        )
        
        # Generate response
        response = await role_agent.generate_llm_response(prompt, use_history=False)
        
        # Parse JSON with better error handling
        try:
            import re
            
            # Save raw response for debugging
            raw_path = os.path.join(CURRENT_DIR, "outputs", f"temp_{tag.lower()}_batch{batch_idx}.raw.txt")
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            with open(raw_path, 'w') as f:
                f.write(response)
            
            # Try multiple parsing strategies
            json_str = None
            
            # Strategy 1: markdown code block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            
            # Strategy 2: any code block
            if not json_str:
                json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
            
            # Strategy 3: find JSON array directly
            if not json_str:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
            
            if json_str:
                batch_results = json.loads(json_str)
                if isinstance(batch_results, list):
                    all_results.extend(batch_results)
                    print(f"✅ 批次 {batch_idx} 完成: 解析得到 {len(batch_results)} 条角色预测")
                else:
                    print(f"⚠️  批次 {batch_idx}: 返回格式不是数组，尝试提取 items")
                    if isinstance(batch_results, dict) and "items" in batch_results:
                        all_results.extend(batch_results["items"])
                    else:
                        all_results.append({"_raw": response[:500], "_batch": batch_idx})
            else:
                print(f"❌ 批次 {batch_idx}: 未找到 JSON")
                all_results.append({"_raw": response[:500], "_batch": batch_idx})
        
        except Exception as e:
            print(f"❌ 批次 {batch_idx} 解析失败: {e}")
            print(f"   响应预览: {response[:200]}...")
            all_results.append({"_raw": response[:500], "_batch": batch_idx, "_error": str(e)})
        
        # Delay between batches to avoid rate limiting
        if batch_idx < len(batches):
            await asyncio.sleep(15)
    
    await role_agent.stop()
    
    print(f"\n✅ 所有批次完成，共 {len(all_results)} 条结果")
    return all_results


async def analyze_text_task(tag: str, nodes_file: str, edges_file: str, base_prompt: str, 
                            task_type: str, agent_class, capability: str):
    """Analyze text tasks (anomaly/summary) - uses sampled nodes/edges"""
    
    nodes_path = os.path.join(CURRENT_DIR, "data", nodes_file)
    edges_path = os.path.join(CURRENT_DIR, "data", edges_file)
    
    # For text tasks, sample nodes/edges to reduce tokens
    all_nodes = read_jsonl(nodes_path)
    edges_csv = read_text(edges_path)
    
    # Sample 200 representative nodes for context
    import random
    random.seed(7)
    sampled_nodes = random.sample(all_nodes, min(200, len(all_nodes)))
    sampled_ids = set(n['id'] for n in sampled_nodes)
    
    nodes_blob = '\n'.join(json.dumps(n, ensure_ascii=False) for n in sampled_nodes)
    edges_blob = filter_edges_for_nodes(edges_csv, sampled_ids)
    
    # Setup LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    provider = OpenAICompatProvider(api_key, model="gpt-4o-mini")
    
    agent = agent_class(name=f"{tag}{task_type}Agent", llm_provider=provider)
    agent.llm_config["use_history"] = False
    agent.llm_config["max_tokens"] = 1000
    
    coordinator = AgentCoordinator()
    
    await coordinator.start()
    await agent.start()
    
    # Register
    caps = [c.__dict__ for c in agent.get_capabilities()]
    await coordinator.register_agent_capabilities(agent.agent_id, agent.get_capabilities())
    
    # Submit task
    task = Task(
        task_id=f"{tag.lower()}-{task_type}",
        name=f"{tag} {task_type}",
        description=f"{task_type} analysis",
        required_capability=capability,
        payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, 
                "base_prompt": base_prompt, "tag": tag},
        priority=TaskPriority.MEDIUM
    )
    
    await coordinator.submit_task(task)
    
    # Wait
    async def wait_done(task_id: str, timeout: float = 180.0):
        import time
        start = time.time()
        while time.time() - start < timeout:
            status = coordinator.get_task_status(task_id)
            if status and status.get("status") == "completed":
                return status
            await asyncio.sleep(2.0)
        return coordinator.get_task_status(task_id)
    
    result = await wait_done(f"{tag.lower()}-{task_type}")
    
    await agent.stop()
    await coordinator.stop()
    
    return result


async def main():
    print("=== Top-1000 Batched Analysis (RWFB vs CETraS) ===")
    
    # Load prompts
    prompts_path = os.path.join(BASE_DIR, "llm4tg_prompts.txt")
    prompts = extract_prompts(read_text(prompts_path))
    
    # Analyze RWFB
    print("\n" + "="*70)
    print("🔵 RWFB Top-1000 分析")
    print("="*70)
    
    rwfb_roles = await analyze_role_batched("RWFB", 
                                            "llm4tg_nodes_top1000_rwfb.jsonl",
                                            "llm4tg_edges_top1000_rwfb.csv",
                                            prompts["p1"],
                                            batch_size=200)
    
    print("\n📝 RWFB 异常分析...")
    rwfb_anom = await analyze_text_task("RWFB", 
                                        "llm4tg_nodes_top1000_rwfb.jsonl",
                                        "llm4tg_edges_top1000_rwfb.csv",
                                        prompts["p2"],
                                        "anomaly", AnomalyAnalystAgent, "anomaly_analysis")
    
    print("\n📝 RWFB 去中心化摘要...")
    rwfb_sum = await analyze_text_task("RWFB",
                                       "llm4tg_nodes_top1000_rwfb.jsonl",
                                       "llm4tg_edges_top1000_rwfb.csv",
                                       prompts["p3"],
                                       "summary", DecentralizationSummarizerAgent, "decentralization_summary")
    
    # Save RWFB results
    out_rwfb = os.path.join(CURRENT_DIR, "outputs", "rwfb_top1000")
    os.makedirs(out_rwfb, exist_ok=True)
    
    with open(os.path.join(out_rwfb, "rwfb_role_predictions.json"), 'w') as f:
        json.dump(rwfb_roles, f, ensure_ascii=False, indent=2)
    
    if rwfb_anom and rwfb_anom.get("result"):
        with open(os.path.join(out_rwfb, "rwfb_anomaly_explained.txt"), 'w') as f:
            f.write(rwfb_anom["result"].get("text", ""))
    
    if rwfb_sum and rwfb_sum.get("result"):
        with open(os.path.join(out_rwfb, "rwfb_decentralization_summary.txt"), 'w') as f:
            f.write(rwfb_sum["result"].get("text", ""))
    
    print(f"✅ RWFB 结果已保存到: {out_rwfb}")
    
    # Analyze CETraS
    print("\n" + "="*70)
    print("🟢 CETraS Top-1000 分析")
    print("="*70)
    
    cetras_roles = await analyze_role_batched("CETraS",
                                              "llm4tg_nodes_top1000_cetras.jsonl",
                                              "llm4tg_edges_top1000_cetras.csv",
                                              prompts["p1"],
                                              batch_size=200)
    
    print("\n📝 CETraS 异常分析...")
    cetras_anom = await analyze_text_task("CETraS",
                                          "llm4tg_nodes_top1000_cetras.jsonl",
                                          "llm4tg_edges_top1000_cetras.csv",
                                          prompts["p2"],
                                          "anomaly", AnomalyAnalystAgent, "anomaly_analysis")
    
    print("\n📝 CETraS 去中心化摘要...")
    cetras_sum = await analyze_text_task("CETraS",
                                         "llm4tg_nodes_top1000_cetras.jsonl",
                                         "llm4tg_edges_top1000_cetras.csv",
                                         prompts["p3"],
                                         "summary", DecentralizationSummarizerAgent, "decentralization_summary")
    
    # Save CETraS results
    out_cetras = os.path.join(CURRENT_DIR, "outputs", "cetras_top1000")
    os.makedirs(out_cetras, exist_ok=True)
    
    with open(os.path.join(out_cetras, "cetras_role_predictions.json"), 'w') as f:
        json.dump(cetras_roles, f, ensure_ascii=False, indent=2)
    
    if cetras_anom and cetras_anom.get("result"):
        with open(os.path.join(out_cetras, "cetras_anomaly_explained.txt"), 'w') as f:
            f.write(cetras_anom["result"].get("text", ""))
    
    if cetras_sum and cetras_sum.get("result"):
        with open(os.path.join(out_cetras, "cetras_decentralization_summary.txt"), 'w') as f:
            f.write(cetras_sum["result"].get("text", ""))
    
    print(f"✅ CETraS 结果已保存到: {out_cetras}")
    
    print("\n" + "="*70)
    print("🎉 所有分析完成！")
    print("下一步: 运行 run_comparison_top1000.py 生成对比图表")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
