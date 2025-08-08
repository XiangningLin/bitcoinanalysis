#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Multi-Agent Analysis Pipeline for Fresh 2020-10 Bitcoin Data
使用新采样数据运行完整的多智能体分析流程
"""

import os
import sys
import re
import json
import asyncio
from typing import Dict

# Add project root to path
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


def read_text_limited(path: str, max_chars: int, note: str = "\n\n[Truncated]\n") -> str:
    data = read_text(path)
    return data if len(data) <= max_chars else data[:max_chars] + note


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
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._temperature = temperature

    async def generate_response(self, messages, **kwargs) -> str:
        import asyncio
        def _call():
            resp = self._client.chat.completions.create(
                model=kwargs.get("model", self._model), 
                temperature=kwargs.get("temperature", self._temperature),
                messages=messages, 
                max_tokens=kwargs.get("max_tokens", 800),
            )
            return resp.choices[0].message.content
        return await asyncio.to_thread(_call)

    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "OpenAI", "model": self._model}


class MockProvider(LLMProvider):
    async def generate_response(self, messages, **kwargs) -> str:
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
        return f"[MOCKED OUTPUT - New Analysis]\n{last_user['content'][:200]}..."
    
    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "mock", "model": "mock-1"}


async def run_analysis_for_dataset(tag: str, nodes_file: str, edges_file: str, prompts: Dict[str, str]):
    """Run multi-agent analysis for one dataset (RWFB or CETraS)"""
    
    # Read data
    nodes_path = os.path.join(CURRENT_DIR, "data", nodes_file)
    edges_path = os.path.join(CURRENT_DIR, "data", edges_file)
    nodes_blob = read_text_limited(nodes_path, 30000)
    edges_blob = read_text_limited(edges_path, 20000)
    
    # Setup LLM provider
    api_key = os.environ.get("OPENAI_API_KEY")
    provider: LLMProvider = OpenAICompatProvider(api_key) if api_key else MockProvider()
    
    # Create agents
    role_agent = RoleClassifierAgent(name=f"{tag}RoleAgent", llm_provider=provider)
    anom_agent = AnomalyAnalystAgent(name=f"{tag}AnomAgent", llm_provider=provider)
    decen_agent = DecentralizationSummarizerAgent(name=f"{tag}DecentAgent", llm_provider=provider)
    coordinator = AgentCoordinator()
    
    # Configure agents
    for a in (role_agent, anom_agent, decen_agent):
        a.llm_config["use_history"] = False
        a.llm_config["max_tokens"] = 800
    
    # Start agents
    await coordinator.start()
    await role_agent.start()
    await anom_agent.start() 
    await decen_agent.start()
    
    # Register capabilities
    for agent in (role_agent, anom_agent, decen_agent):
        caps = [c.__dict__ for c in agent.get_capabilities()]
        msg = Message(receiver_id=coordinator.agent_id, msg_type=MessageType.REQUEST,
                      content={"type": "register_capabilities", "capabilities": caps})
        await agent.send_message(msg)
        await coordinator.register_agent_capabilities(agent.agent_id, agent.get_capabilities())
    
    # Submit tasks
    t1 = Task(task_id=f"{tag.lower()}-roles", name=f"{tag} Role Classification", 
              description="Classify roles", required_capability="role_classification",
              payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, 
                      "base_prompt": prompts["p1"], "tag": tag},
              priority=TaskPriority.MEDIUM)
    
    t2 = Task(task_id=f"{tag.lower()}-anomaly", name=f"{tag} Anomaly Analysis", 
              description="Explain patterns", required_capability="anomaly_analysis",
              payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, 
                      "base_prompt": prompts["p2"], "tag": tag},
              priority=TaskPriority.MEDIUM)
    
    t3 = Task(task_id=f"{tag.lower()}-decent", name=f"{tag} Decentralization Summary", 
              description="Summarize", required_capability="decentralization_summary",
              payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, 
                      "base_prompt": prompts["p3"], "tag": tag},
              priority=TaskPriority.MEDIUM)
    
    await coordinator.submit_task(t1)
    await asyncio.sleep(2.0)
    await coordinator.submit_task(t2)
    await asyncio.sleep(2.0)
    await coordinator.submit_task(t3)
    
    # Wait for completion
    async def wait_done(task_id: str, timeout: float = 120.0):
        import time
        start = time.time()
        while time.time() - start < timeout:
            status = coordinator.get_task_status(task_id)
            if status and status.get("status") == "completed":
                return status
            await asyncio.sleep(1.0)
        return coordinator.get_task_status(task_id)
    
    print(f"\n=== Waiting for {tag} analysis completion ===")
    s1 = await wait_done(f"{tag.lower()}-roles")
    s2 = await wait_done(f"{tag.lower()}-anomaly")
    s3 = await wait_done(f"{tag.lower()}-decent")
    
    # Save results
    out_dir = os.path.join(CURRENT_DIR, "outputs", tag.lower())
    os.makedirs(out_dir, exist_ok=True)
    
    if s1 and s1.get("result"):
        r = s1["result"]
        items = r.get("items", [])
        raw = r.get("raw", "")
        with open(os.path.join(out_dir, f"{tag.lower()}_role_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, f"{tag.lower()}_role_predictions.part1.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
    
    if s2 and s2.get("result"):
        with open(os.path.join(out_dir, f"{tag.lower()}_anomaly_explained.txt"), "w", encoding="utf-8") as f:
            f.write(s2["result"].get("text", ""))
    
    if s3 and s3.get("result"):
        with open(os.path.join(out_dir, f"{tag.lower()}_decentralization_summary.txt"), "w", encoding="utf-8") as f:
            f.write(s3["result"].get("text", ""))
    
    print(f"=== {tag} analysis completed, results saved to {out_dir} ===")
    
    # Stop agents
    await role_agent.stop()
    await anom_agent.stop()
    await decen_agent.stop()
    await coordinator.stop()
    
    return s1, s2, s3


async def main():
    print("=== Starting New Multi-Agent Bitcoin Analysis (2020-10 Data) ===")
    
    # Load prompts
    prompts_path = os.path.join(BASE_DIR, "llm4tg_prompts.txt")
    prompts_txt = read_text(prompts_path)
    prompts = extract_prompts(prompts_txt)
    
    print("Prompts loaded:", list(prompts.keys()))
    
    # Run RWFB analysis
    print("\n🔄 Running RWFB analysis...")
    await run_analysis_for_dataset("RWFB", 
                                  "llm4tg_nodes_top100_rwfb.jsonl", 
                                  "llm4tg_edges_top100_rwfb.csv", 
                                  prompts)
    
    # Run CETraS analysis  
    print("\n🔄 Running CETraS analysis...")
    await run_analysis_for_dataset("CETraS", 
                                  "llm4tg_nodes_top100_cetras.jsonl", 
                                  "llm4tg_edges_top100_cetras.csv", 
                                  prompts)
    
    print("\n✅ All analysis completed! Check new_analysis/outputs/ for results.")
    print("Next: Run comparison analysis with new_analysis/run_comparison.py")


if __name__ == "__main__":
    asyncio.run(main())

