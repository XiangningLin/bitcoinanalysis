#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry CETraS analysis with gpt-4o-mini
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
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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


async def main():
    print("=== Retrying CETraS Analysis with gpt-4o-mini ===")
    
    # Load prompts and data
    prompts_path = os.path.join(BASE_DIR, "llm4tg_prompts.txt")
    prompts_txt = read_text(prompts_path)
    prompts = extract_prompts(prompts_txt)
    
    nodes_path = os.path.join(CURRENT_DIR, "data", "llm4tg_nodes_top100_cetras.jsonl")
    edges_path = os.path.join(CURRENT_DIR, "data", "llm4tg_edges_top100_cetras.csv")
    nodes_blob = read_text_limited(nodes_path, 30000)
    edges_blob = read_text_limited(edges_path, 20000)
    
    # Setup LLM provider with mini model
    api_key = os.environ.get("OPENAI_API_KEY")
    provider = OpenAICompatProvider(api_key, model="gpt-4o-mini")
    
    # Create agents
    role_agent = RoleClassifierAgent(name="CETraSRoleAgent", llm_provider=provider)
    anom_agent = AnomalyAnalystAgent(name="CETraSAnomAgent", llm_provider=provider)
    decen_agent = DecentralizationSummarizerAgent(name="CETraSDecentAgent", llm_provider=provider)
    coordinator = AgentCoordinator()
    
    # Configure agents
    for a in (role_agent, anom_agent, decen_agent):
        a.llm_config["use_history"] = False
        a.llm_config["max_tokens"] = 600  # Reduced to avoid rate limits
    
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
    
    # Submit tasks with delays
    print("🔄 Submitting CETraS tasks with delays...")
    
    t1 = Task(task_id="cetras-roles-retry", name="CETraS Role Classification", 
              description="Classify roles", required_capability="role_classification",
              payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, 
                      "base_prompt": prompts["p1"], "tag": "CETraS"},
              priority=TaskPriority.MEDIUM)
    
    await coordinator.submit_task(t1)
    await asyncio.sleep(20)  # Longer delay
    
    t2 = Task(task_id="cetras-anomaly-retry", name="CETraS Anomaly Analysis", 
              description="Explain patterns", required_capability="anomaly_analysis",
              payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, 
                      "base_prompt": prompts["p2"], "tag": "CETraS"},
              priority=TaskPriority.MEDIUM)
    
    await coordinator.submit_task(t2)
    await asyncio.sleep(20)  # Longer delay
    
    t3 = Task(task_id="cetras-decent-retry", name="CETraS Decentralization Summary", 
              description="Summarize", required_capability="decentralization_summary",
              payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, 
                      "base_prompt": prompts["p3"], "tag": "CETraS"},
              priority=TaskPriority.MEDIUM)
    
    await coordinator.submit_task(t3)
    
    # Wait for completion
    async def wait_done(task_id: str, timeout: float = 180.0):
        import time
        start = time.time()
        while time.time() - start < timeout:
            status = coordinator.get_task_status(task_id)
            if status and status.get("status") == "completed":
                return status
            await asyncio.sleep(2.0)
        return coordinator.get_task_status(task_id)
    
    print("⏳ Waiting for CETraS analysis completion...")
    s1 = await wait_done("cetras-roles-retry")
    s2 = await wait_done("cetras-anomaly-retry")
    s3 = await wait_done("cetras-decent-retry")
    
    # Save results
    out_dir = os.path.join(CURRENT_DIR, "outputs", "cetras")
    os.makedirs(out_dir, exist_ok=True)
    
    if s1 and s1.get("result"):
        r = s1["result"]
        items = r.get("items", [])
        raw = r.get("raw", "")
        with open(os.path.join(out_dir, "cetras_role_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "cetras_role_predictions.part1.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
        print("✅ CETraS roles saved")
    
    if s2 and s2.get("result"):
        with open(os.path.join(out_dir, "cetras_anomaly_explained.txt"), "w", encoding="utf-8") as f:
            f.write(s2["result"].get("text", ""))
        print("✅ CETraS anomaly analysis saved")
    
    if s3 and s3.get("result"):
        with open(os.path.join(out_dir, "cetras_decentralization_summary.txt"), "w", encoding="utf-8") as f:
            f.write(s3["result"].get("text", ""))
        print("✅ CETraS decentralization summary saved")
    
    # Stop agents
    await role_agent.stop()
    await anom_agent.stop()
    await decen_agent.stop()
    await coordinator.stop()
    
    print("\n🎉 CETraS analysis completed with gpt-4o-mini!")
    print("Now run: python run_comparison.py")


if __name__ == "__main__":
    asyncio.run(main())

