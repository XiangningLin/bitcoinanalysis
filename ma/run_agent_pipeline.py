#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent-integrated pipeline for bitcoinanalysis (organized under ma/)

Reads/writes data in parent folder 'bitcoinanalysis/'.
"""

import os
import re
import json
import asyncio
import sys
from typing import Dict, List

# Paths
CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))           # bitcoinanalysis/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))          # workspace root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_framework.agents.llm_agent import LLMAgent, LLMProvider


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_text_limited(path: str, max_chars: int, trailer_note: str = "\n\n[Truncated for token limits]\n") -> str:
    data = read_text(path)
    if len(data) <= max_chars:
        return data
    return data[:max_chars] + trailer_note


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
    """Compat provider using sync OpenAI client under the hood (run in thread)."""

    def __init__(self, api_key: str, model: str = None, temperature: float = 0.2):
        from openai import OpenAI  # type: ignore
        self._client = OpenAI(api_key=api_key)
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._temperature = temperature

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import asyncio
        model = kwargs.get("model", self._model)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", 2000)

        def _call():
            resp = self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        return await asyncio.to_thread(_call)

    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "OpenAI", "model": self._model}


class MockProvider(LLMProvider):
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
        return f"[MOCKED OUTPUT]\n{last_user['content'][:200]}..."

    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "mock", "model": "mock-1"}


def build_prompt_1(nodes_blob: str, edges_blob: str, base_prompt: str, task_name: str) -> str:
    return (
        f"[Task] {task_name} / Prompt 1 - Node Role Classification\n"
        f"以下为本批节点画像（JSONL）:\n```\n{nodes_blob}\n```\n"
        f"以下为 Top100 诱导边（CSV，部分截断，仅作辅助）:\n```\n{edges_blob}\n```\n"
        f"{base_prompt}"
    )


def build_prompt_2(nodes_blob: str, edges_blob: str, base_prompt: str, task_name: str) -> str:
    return (
        f"[Task] {task_name} / Prompt 2 - Anomaly Pattern Explanation\n"
        f"节点画像（JSONL，Top100）如下：\n```\n{nodes_blob}\n```\n"
        f"Top100 诱导边（CSV，部分截断）如下：\n```\n{edges_blob}\n```\n"
        f"{base_prompt}"
    )


def build_prompt_3(nodes_blob: str, edges_blob: str, base_prompt: str, task_name: str) -> str:
    return (
        f"[Task] {task_name} / Prompt 3 - Decentralization Snapshot Summary\n"
        f"节点画像（JSONL，Top100）如下：\n```\n{nodes_blob}\n```\n"
        f"Top100 诱导边（CSV，部分截断）如下：\n```\n{edges_blob}\n```\n"
        f"{base_prompt}"
    )


def _clean_json_like(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
    return s.strip()


async def run_for_dataset(llm: LLMAgent, nodes_name: str, edges_name: str, prompts: Dict[str, str], tag: str):
    nodes_path = os.path.join(BASE_DIR, nodes_name)
    edges_path = os.path.join(BASE_DIR, edges_name)
    nodes_blob = read_text_limited(nodes_path, max_chars=30000)
    edges_blob = read_text_limited(edges_path, max_chars=20000)

    # Prepare output directory under ma/: outputs/<tag>
    out_dir = os.path.join(CURRENT_DIR, "outputs", tag.lower())
    os.makedirs(out_dir, exist_ok=True)

    # Prompt 1: roles -> JSON
    p1 = build_prompt_1(nodes_blob, edges_blob, prompts["p1"], tag)
    await asyncio.sleep(2.5)
    resp1 = await llm.generate_llm_response(p1)
    raw1 = resp1.strip()
    try:
        parsed = json.loads(_clean_json_like(raw1))
    except Exception:
        parsed = [{"_raw": raw1}]
    with open(os.path.join(out_dir, f"{tag.lower()}_role_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"{tag.lower()}_role_predictions.part1.raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw1)

    # Prompt 2
    p2 = build_prompt_2(nodes_blob, edges_blob, prompts["p2"], tag)
    await asyncio.sleep(2.5)
    resp2 = await llm.generate_llm_response(p2)
    with open(os.path.join(out_dir, f"{tag.lower()}_anomaly_explained.txt"), "w", encoding="utf-8") as f:
        f.write(resp2)

    # Prompt 3
    p3 = build_prompt_3(nodes_blob, edges_blob, prompts["p3"], tag)
    await asyncio.sleep(2.5)
    resp3 = await llm.generate_llm_response(p3)
    with open(os.path.join(out_dir, f"{tag.lower()}_decentralization_summary.txt"), "w", encoding="utf-8") as f:
        f.write(resp3)


async def main():
    prompts_txt = read_text(os.path.join(BASE_DIR, "llm4tg_prompts.txt"))
    P = extract_prompts(prompts_txt)

    api_key = os.environ.get("OPENAI_API_KEY")
    provider: LLMProvider = (
        OpenAICompatProvider(api_key=api_key, model=os.environ.get("OPENAI_MODEL", "gpt-4o"))
        if api_key else MockProvider()
    )

    llm = LLMAgent(name="BitcoinLLM", llm_provider=provider)
    llm.llm_config["max_tokens"] = 800
    llm.llm_config["use_history"] = False
    await llm.start()

    try:
        await run_for_dataset(llm, "llm4tg_nodes_top100_rwfb.jsonl", "llm4tg_edges_top100_rwfb.csv", P, "RWFB")
        await run_for_dataset(llm, "llm4tg_nodes_top100_cetras_fast.jsonl", "llm4tg_edges_top100_cetras_fast.csv", P, "CETraS")
    finally:
        await llm.stop()


if __name__ == "__main__":
    asyncio.run(main())


