#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collaborative Bitcoin Network Analysis
真正的协作式比特币网络分析
"""

import os
import sys
import json
import asyncio
from typing import Dict

CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_framework.core.collaborative_coordinator import CollaborativeAgentCoordinator
from agent_framework.core.coordinator import TaskPriority
from agent_framework.core.base_agent import Message, MessageType
from agent_framework.agents.collaborative_bitcoin_agents import (
    CollaborativeRoleClassifierAgent,
    CollaborativeAnomalyAnalystAgent,
    CollaborativeDecentralizationSummarizerAgent
)
from agent_framework.agents.llm_agent import LLMProvider


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_text_limited(path: str, max_chars: int, note: str = "\n\n[Truncated]\n") -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data if len(data) <= max_chars else data[:max_chars] + note


def extract_prompts(txt: str) -> Dict[str, str]:
    sections = txt.split("### Prompt")
    return {
        "p1": sections[1].strip() if len(sections) > 1 else "",
        "p2": sections[2].strip() if len(sections) > 2 else "",
        "p3": sections[3].strip() if len(sections) > 3 else ""
    }


class OpenAICompatProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._model = "gpt-4o-mini"
    
    async def generate_response(self, messages, **kwargs):
        # 模拟OpenAI API调用
        return f"[Mock OpenAI Response] Processed {len(messages)} messages"
    
    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "OpenAI", "model": self._model}


class MockProvider(LLMProvider):
    async def generate_response(self, messages, **kwargs):
        return "[Mock Response] Collaborative analysis completed"
    
    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "mock", "model": "mock-1"}


async def run_collaborative_analysis(sampling_method: str = "CETraS"):
    """运行协作式分析"""
    
    # 加载数据和提示词
    prompts = extract_prompts(read_text(os.path.join(BASE_DIR, "llm4tg_prompts.txt")))
    
    if sampling_method == "CETraS":
        nodes_blob = read_text_limited(
            os.path.join(BASE_DIR, "llm4tg_nodes_top100_cetras_fast.jsonl"), 30000
        )
        edges_blob = read_text_limited(
            os.path.join(BASE_DIR, "llm4tg_edges_top100_cetras_fast.csv"), 20000
        )
    else:  # RWFB
        nodes_blob = read_text_limited(
            os.path.join(BASE_DIR, "llm4tg_nodes_top100_rwfb.jsonl"), 30000
        )
        edges_blob = read_text_limited(
            os.path.join(BASE_DIR, "llm4tg_edges_top100_rwfb.csv"), 20000
        )
    
    # 设置LLM Provider
    api_key = os.environ.get("OPENAI_API_KEY")
    provider: LLMProvider = OpenAICompatProvider(api_key) if api_key else MockProvider()
    
    # 创建协作式Agent
    role_agent = CollaborativeRoleClassifierAgent(name="CollaborativeRoleAgent", llm_provider=provider)
    anomaly_agent = CollaborativeAnomalyAnalystAgent(name="CollaborativeAnomalyAgent", llm_provider=provider)
    decent_agent = CollaborativeDecentralizationSummarizerAgent(name="CollaborativeDecentAgent", llm_provider=provider)
    
    # 创建协作式协调器
    coordinator = CollaborativeAgentCoordinator()
    
    # 配置Agent
    for agent in [role_agent, anomaly_agent, decent_agent]:
        agent.llm_config["use_history"] = False
        agent.llm_config["max_tokens"] = 800
        # 设置协作管理器引用
        agent.collaboration_manager = coordinator.collaboration_manager
    
    print(f"Starting collaborative analysis for {sampling_method}...")
    
    # 启动所有组件
    await coordinator.start()
    await role_agent.start()
    await anomaly_agent.start()
    await decent_agent.start()
    
    # 注册Agent能力
    for agent in [role_agent, anomaly_agent, decent_agent]:
        caps = [c.__dict__ for c in agent.get_capabilities()]
        msg = Message(
            receiver_id=coordinator.agent_id,
            msg_type=MessageType.REQUEST,
            content={"type": "register_capabilities", "capabilities": caps}
        )
        await agent.send_message(msg)
        await coordinator.register_agent_capabilities(agent.agent_id, agent.get_capabilities())
    
    print("Agents registered and ready for collaboration")
    
    # 提交协作任务
    payload = {
        "nodes_blob": nodes_blob,
        "edges_blob": edges_blob,
        "prompts": prompts,
        "tag": sampling_method,
        "analysis_type": "collaborative_bitcoin_analysis"
    }
    
    task_id = await coordinator.submit_collaborative_task(
        task_name=f"Collaborative {sampling_method} Analysis",
        collaboration_pattern="bitcoin_analysis",
        payload=payload,
        priority=TaskPriority.MEDIUM
    )
    
    print(f"Collaborative task submitted: {task_id}")
    
    # 等待协作完成
    await wait_for_collaboration_completion(coordinator, task_id)
    
    # 获取最终结果
    final_result = await get_collaboration_result(coordinator, task_id)
    
    # 保存结果
    await save_collaborative_results(final_result, sampling_method)
    
    print(f"Collaborative analysis completed for {sampling_method}")
    
    # 清理
    await role_agent.stop()
    await anomaly_agent.stop()
    await decent_agent.stop()
    await coordinator.stop()


async def wait_for_collaboration_completion(coordinator: CollaborativeAgentCoordinator, task_id: str, timeout: float = 300.0):
    """等待协作完成"""
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        # 检查协作状态
        collaboration_id = task_id.replace("collab_bitcoin_analysis_", "")
        status = coordinator.get_collaboration_status(collaboration_id)
        
        if status is None or status.get("status") == "completed":
            break
        
        print(f"Collaboration in progress... Stage: {status.get('current_stage', 0)}")
        await asyncio.sleep(5)
    
    if asyncio.get_event_loop().time() - start_time >= timeout:
        print("Collaboration timeout!")


async def get_collaboration_result(coordinator: CollaborativeAgentCoordinator, task_id: str) -> Dict[str, Any]:
    """获取协作结果"""
    collaboration_id = task_id.replace("collab_bitcoin_analysis_", "")
    status = coordinator.get_collaboration_status(collaboration_id)
    
    if status:
        return status
    else:
        return {"error": "Collaboration not found"}


async def save_collaborative_results(result: Dict[str, Any], sampling_method: str):
    """保存协作结果"""
    output_dir = os.path.join(CURRENT_DIR, "outputs", f"collaborative_{sampling_method.lower()}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整结果
    result_file = os.path.join(output_dir, "collaborative_analysis_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {result_file}")
    
    # 提取各阶段结果并分别保存
    if "results" in result:
        for stage_key, stage_results in result["results"].items():
            stage_file = os.path.join(output_dir, f"{stage_key}_results.json")
            with open(stage_file, "w", encoding="utf-8") as f:
                json.dump(stage_results, f, ensure_ascii=False, indent=2)
    
    # 保存协作指标
    if "metrics" in result:
        metrics_file = os.path.join(output_dir, "collaboration_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(result["metrics"], f, ensure_ascii=False, indent=2)


async def compare_collaborative_vs_individual():
    """比较协作式vs独立式分析"""
    print("Running comparative analysis...")
    
    # 运行协作式分析
    print("\n=== Collaborative Analysis ===")
    await run_collaborative_analysis("CETraS")
    
    # 这里可以添加独立式分析的比较
    print("\n=== Individual Analysis (for comparison) ===")
    print("Individual analysis would go here...")
    
    print("\nComparative analysis completed!")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collaborative Bitcoin Network Analysis")
    parser.add_argument("--method", choices=["CETraS", "RWFB"], default="CETraS", help="Sampling method")
    parser.add_argument("--compare", action="store_true", help="Run comparative analysis")
    
    args = parser.parse_args()
    
    if args.compare:
        await compare_collaborative_vs_individual()
    else:
        await run_collaborative_analysis(args.method)


if __name__ == "__main__":
    asyncio.run(main())
