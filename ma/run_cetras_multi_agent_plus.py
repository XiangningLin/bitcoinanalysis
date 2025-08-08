# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import os, sys, re, json, asyncio
# from typing import Dict

# CURRENT_DIR = os.path.dirname(__file__)
# BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
# PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from agent_framework.core import AgentCoordinator, Task, TaskPriority, Message, MessageType
# from agent_framework.core.blackboard import Blackboard
# from agent_framework.agents import (
#     RoleClassifierAgent, AnomalyAnalystAgent, DecentralizationSummarizerAgent,
#     JudgeAgent, MergeAgent, GraphMetricsAgent,
# )
# from agent_framework.agents.llm_agent import LLMProvider


# def read_text(path: str) -> str:
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read()


# def read_text_limited(path: str, max_chars: int, note: str = "\n\n[Truncated]\n") -> str:
#     d = read_text(path)
#     return d if len(d) <= max_chars else d[:max_chars] + note


# def extract_prompts(txt: str) -> Dict[str, str]:
#     def grab(title: str) -> str:
#         m = re.search(rf"### {re.escape(title)}([\s\S]*?)(?=###|\Z)", txt)
#         return (m.group(1).strip() if m else "")
#     return {
#         "p1": grab("Prompt 1 — Node Role Classification") or grab("Prompt 1"),
#         "p2": grab("Prompt 2 — Anomaly Pattern Explanation") or grab("Prompt 2"),
#         "p3": grab("Prompt 3 — Decentralization Snapshot Summary") or grab("Prompt 3"),
#     }


# class OpenAICompatProvider(LLMProvider):
#     def __init__(self, api_key: str, model: str = None, temperature: float = 0.2):
#         from openai import OpenAI
#         self._client = OpenAI(api_key=api_key)
#         self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
#         self._temperature = temperature

#     async def generate_response(self, messages, **kwargs) -> str:
#         import asyncio
#         def _call():
#             resp = self._client.chat.completions.create(
#                 model=kwargs.get("model", self._model), temperature=kwargs.get("temperature", self._temperature),
#                 messages=messages, max_tokens=kwargs.get("max_tokens", 800),
#             )
#             return resp.choices[0].message.content
#         return await asyncio.to_thread(_call)

#     def get_model_info(self) -> Dict[str, str]:
#         return {"provider": "OpenAI", "model": self._model}


# class MockProvider(LLMProvider):
#     async def generate_response(self, messages, **kwargs) -> str:
#         last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
#         return f"[MOCKED OUTPUT]\n{last_user['content'][:200]}..."
#     def get_model_info(self) -> Dict[str, str]:
#         return {"provider": "mock", "model": "mock-1"}


# async def main():
#     prompts = extract_prompts(read_text(os.path.join(BASE_DIR, "llm4tg_prompts.txt")))
#     nodes_blob = read_text_limited(os.path.join(BASE_DIR, "llm4tg_nodes_top100_cetras_fast.jsonl"), 30000)
#     edges_blob = read_text_limited(os.path.join(BASE_DIR, "llm4tg_edges_top100_cetras_fast.csv"), 20000)

#     api_key = os.environ.get("OPENAI_API_KEY")
#     provider: LLMProvider = OpenAICompatProvider(api_key) if api_key else MockProvider()

#     role_agent  = RoleClassifierAgent(name="RoleAgent", llm_provider=provider)
#     anom_agent  = AnomalyAnalystAgent(name="AnomalyAgent", llm_provider=provider)
#     decen_agent = DecentralizationSummarizerAgent(name="DecentAgent", llm_provider=provider)
#     judge_agent = JudgeAgent(name="JudgeAgent", llm_provider=provider)
#     merge_agent = MergeAgent(name="MergeAgent", llm_provider=provider)
#     graph_agent = GraphMetricsAgent(name="GraphAgent", llm_provider=provider)
#     coordinator = AgentCoordinator()

#     for a in (role_agent, anom_agent, decen_agent, judge_agent, merge_agent, graph_agent):
#         a.llm_config["use_history"] = False
#         a.llm_config["max_tokens"] = 800

#     await coordinator.start()
#     for a in (role_agent, anom_agent, decen_agent, judge_agent, merge_agent, graph_agent):
#         await a.start()

#     for agent in (role_agent, anom_agent, decen_agent, judge_agent, merge_agent, graph_agent):
#         caps = [c.__dict__ for c in agent.get_capabilities()]
#         msg = Message(receiver_id=coordinator.agent_id, msg_type=MessageType.REQUEST,
#                       content={"type": "register_capabilities", "capabilities": caps})
#         await agent.send_message(msg)
#         await coordinator.register_agent_capabilities(agent.agent_id, agent.get_capabilities())

#     bb = await Blackboard.get_instance()
#     await bb.put("cetras", "nodes_blob", nodes_blob)
#     await bb.put("cetras", "edges_blob", edges_blob)

#     t_metrics = Task(task_id="cetras-metrics", name="Compute metrics", description="Compute graph metrics",
#                      required_capability="compute_graph_metrics",
#                      payload={"edges_csv_blob": edges_blob}, priority=TaskPriority.MEDIUM)
#     await coordinator.submit_task(t_metrics)

#     async def submit_role():
#         t = Task(task_id="cetras-roles+", name="Roles", description="Classify roles",
#                  required_capability="role_classification",
#                  payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, "base_prompt": prompts["p1"], "tag": "CETraS"},
#                  priority=TaskPriority.MEDIUM)
#         await coordinator.submit_task(t)

#     async def submit_anom():
#         t = Task(task_id="cetras-anom+", name="Anomaly", description="Analyze anomalies",
#                  required_capability="anomaly_analysis",
#                  payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, "base_prompt": prompts["p2"], "tag": "CETraS"},
#                  priority=TaskPriority.MEDIUM)
#         await coordinator.submit_task(t)

#     async def submit_decen():
#         t = Task(task_id="cetras-decen+", name="Decentralization", description="Summarize decentralization",
#                  required_capability="decentralization_summary",
#                  payload={"nodes_blob": nodes_blob, "edges_blob": edges_blob, "base_prompt": prompts["p3"], "tag": "CETraS"},
#                  priority=TaskPriority.MEDIUM)
#         await coordinator.submit_task(t)

#     await submit_role(); await asyncio.sleep(2.0)
#     await submit_anom(); await asyncio.sleep(2.0)
#     await submit_decen()

#     async def wait_done(task_id: str, timeout: float = 120.0):
#         import time
#         start = time.time()
#         while time.time() - start < timeout:
#             status = coordinator.get_task_status(task_id)
#             if status and status.get("status") == "completed":
#                 return status
#             await asyncio.sleep(1.0)
#         return coordinator.get_task_status(task_id)

#     s_metrics = await wait_done("cetras-metrics")
#     s_roles   = await wait_done("cetras-roles+")
#     s_anom    = await wait_done("cetras-anom+")
#     s_decen   = await wait_done("cetras-decen+")

#     # Prepare output directory under ma/: outputs/cetras_plus
#     out_dir = os.path.join(CURRENT_DIR, "outputs", "cetras_plus")
#     os.makedirs(out_dir, exist_ok=True)

#     if s_roles and s_roles.get("result"):
#         r = s_roles["result"]
#         json.dump(r.get("items", []), open(os.path.join(out_dir, "cetras_role_predictions.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
#         open(os.path.join(out_dir, "cetras_role_predictions.part1.raw.txt"),"w",encoding="utf-8").write(r.get("raw",""))
#     if s_anom and s_anom.get("result"):
#         open(os.path.join(out_dir, "cetras_anomaly_explained.txt"),"w",encoding="utf-8").write(s_anom["result"].get("text",""))
#     if s_decen and s_decen.get("result"):
#         open(os.path.join(out_dir, "cetras_decentralization_summary.txt"),"w",encoding="utf-8").write(s_decen["result"].get("text",""))

#     drafts = []
#     if s_anom and s_anom.get("result"): drafts.append(s_anom["result"].get("text",""))
#     if s_decen and s_decen.get("result"): drafts.append(s_decen["result"].get("text",""))

#     if drafts:
#         # Judge drafts
#         for i, d in enumerate(drafts):
#             t = Task(task_id=f"judge-{i}", name="Judge", description="Review draft",
#                      required_capability="judge_quality",
#                      payload={"draft": d, "criteria": "accuracy, clarity, non-redundancy"})
#             await coordinator.submit_task(t)
#             await wait_done(f"judge-{i}")

#         # Merge
#         t = Task(task_id="merge-final", name="Merge", description="Merge drafts",
#                  required_capability="merge_drafts",
#                  payload={"drafts": drafts, "instructions": "Merge into a coherent, non-duplicative 150-word executive summary."})
#         await coordinator.submit_task(t)
#         s_merge = await wait_done("merge-final")
#         if s_merge and s_merge.get("result"):
#             open(os.path.join(out_dir, "cetras_executive_summary.txt"),"w",encoding="utf-8").write(s_merge["result"].get("text",""))

#     for a in (role_agent, anom_agent, decen_agent, judge_agent, merge_agent, graph_agent):
#         await a.stop()
#     await coordinator.stop()


# if __name__ == "__main__":
#     asyncio.run(main())


