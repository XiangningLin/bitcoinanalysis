#!/usr/bin/env python3
"""
Multi-LLM Performance Benchmark
对比不同LLM在Bitcoin交易分析任务上的性能

测试维度：
1. 准确性 (Accuracy)
2. 一致性 (Consistency)  
3. 响应时间 (Response Time)
4. 成本效率 (Cost Efficiency)
5. 语义质量 (Semantic Quality)
"""

import asyncio
import json
import time
import os
import sys
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_framework.agents.llm_agent import (
    create_openai_agent, 
    create_anthropic_agent,
    create_gemini_agent
)

# 测试任务集合
BENCHMARK_TASKS = [
    {
        "id": 1,
        "category": "role_classification",
        "description": "Classify Bitcoin address role",
        "prompt": """Given the following Bitcoin transaction pattern:
        
Address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
- Total inputs: 0
- Total outputs: 50 BTC
- Unique receivers: 1
- Transaction frequency: Once

Classify this address role as one of: Exchange, Miner, Merchant, Personal Wallet, Mixer
Provide your answer and reasoning.""",
        "ground_truth": "Miner",
        "reasoning_required": True
    },
    {
        "id": 2,
        "category": "pattern_analysis",
        "description": "Analyze transaction pattern",
        "prompt": """Analyze this Bitcoin transaction pattern:

Address Activity:
- Receives from 100+ different addresses daily
- Sends to 50+ different addresses daily  
- Average transaction size: 0.5 BTC
- Active 24/7

What type of entity is this most likely? Explain your reasoning.""",
        "ground_truth": "Exchange",
        "reasoning_required": True
    },
    {
        "id": 3,
        "category": "anomaly_detection",
        "description": "Detect anomalous behavior",
        "prompt": """Evaluate if this transaction pattern is anomalous:

Address X:
- Suddenly receives 1000 small transactions (0.001 BTC each) in 1 hour
- Previously averaged 5 transactions per day
- Immediately consolidates all funds to a new address

Is this anomalous? What might it indicate?""",
        "ground_truth": "Yes",
        "reasoning_required": True
    },
    {
        "id": 4,
        "category": "centralization_assessment",
        "description": "Assess network centralization",
        "prompt": """Given these metrics for a Bitcoin transaction network:

- Top 10 addresses control 45% of transaction volume
- Gini coefficient: 0.78
- Network density: 0.23
- Average path length: 3.4

How centralized is this network? Provide a score from 1-10 (10 = highly centralized).""",
        "ground_truth": "7-8",
        "reasoning_required": True
    },
    {
        "id": 5,
        "category": "semantic_description",
        "description": "Describe transaction semantics",
        "prompt": """Describe the semantic role of this Bitcoin address in 2-3 sentences:

Characteristics:
- Receives funds from miners
- Holds funds for extended periods (weeks/months)
- Sends small amounts to various addresses occasionally
- Total balance always increasing

What is this address's likely purpose?""",
        "ground_truth": "Long-term holder / Investment wallet",
        "reasoning_required": True
    }
]

class LLMBenchmark:
    """LLM性能基准测试"""
    
    def __init__(self):
        self.results = []
        self.models = {}
        
        # 尝试加载.env文件
        self._load_env_files()
        
        # 尝试从环境变量加载API keys
        self.api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY')
        }
        
        print("="*70)
        print("Multi-LLM Benchmark Initialization")
        print("="*70)
        self._check_api_keys()
    
    def _load_env_files(self):
        """尝试从多个位置加载.env文件"""
        try:
            from dotenv import load_dotenv
            
            # 尝试多个位置
            possible_paths = [
                Path(__file__).parent.parent.parent / 'bitcoinanalysis' / '.env',
                Path(__file__).parent.parent.parent / '.env',
                Path.cwd() / '.env',
                Path.cwd() / 'bitcoinanalysis' / '.env'
            ]
            
            for env_path in possible_paths:
                if env_path.exists():
                    load_dotenv(env_path)
                    print(f"✓ Loaded environment from: {env_path}")
                    break
        except ImportError:
            pass  # dotenv not installed
    
    def _check_api_keys(self):
        """检查API keys可用性"""
        print("\nAPI Key Status:")
        for provider, key in self.api_keys.items():
            status = "✓ Available" if key else "✗ Not found"
            print(f"  {provider.capitalize()}: {status}")
        
        available_providers = [p for p, k in self.api_keys.items() if k]
        
        if not available_providers:
            print("\n⚠️  WARNING: No API keys found!")
            print("   Set environment variables:")
            print("     export OPENAI_API_KEY='your-key'")
            print("     export ANTHROPIC_API_KEY='your-key'")
            print("     export GOOGLE_API_KEY='your-key'")
            print("\n   Cannot run real benchmarks without API keys.")
            return False
        
        print(f"\n✓ {len(available_providers)} provider(s) available for testing")
        return True
    
    async def initialize_models(self):
        """初始化所有可用的LLM模型"""
        print("\n" + "="*70)
        print("Initializing LLM Models")
        print("="*70)
        
        # GPT-4
        if self.api_keys['openai']:
            try:
                self.models['gpt-4'] = create_openai_agent(
                    api_key=self.api_keys['openai'],
                    model="gpt-4",
                    name="GPT-4 Analyzer"
                )
                print("✓ GPT-4 initialized")
            except Exception as e:
                print(f"✗ GPT-4 failed: {e}")
        
        # GPT-4o-mini (更便宜的版本)
        if self.api_keys['openai']:
            try:
                self.models['gpt-4o-mini'] = create_openai_agent(
                    api_key=self.api_keys['openai'],
                    model="gpt-4o-mini",
                    name="GPT-4o-mini Analyzer"
                )
                print("✓ GPT-4o-mini initialized")
            except Exception as e:
                print(f"✗ GPT-4o-mini failed: {e}")
        
        # Claude-3.5-Sonnet
        if self.api_keys['anthropic']:
            try:
                self.models['claude-3.5-sonnet'] = create_anthropic_agent(
                    api_key=self.api_keys['anthropic'],
                    model="claude-3-5-sonnet-20241022",
                    name="Claude Analyzer"
                )
                print("✓ Claude-3.5-Sonnet initialized")
            except Exception as e:
                print(f"✗ Claude failed: {e}")
        
        # Gemini-1.5-Pro
        if self.api_keys['google']:
            try:
                self.models['gemini-1.5-pro'] = create_gemini_agent(
                    api_key=self.api_keys['google'],
                    model="gemini-1.5-pro",
                    name="Gemini Analyzer"
                )
                print("✓ Gemini-1.5-Pro initialized")
            except Exception as e:
                print(f"✗ Gemini failed: {e}")
        
        if not self.models:
            print("\n✗ No models successfully initialized")
            return False
        
        print(f"\n✓ {len(self.models)} model(s) ready for benchmarking")
        return True
    
    async def run_single_test(self, model_name: str, agent, task: Dict) -> Dict[str, Any]:
        """运行单个测试任务"""
        print(f"  Testing {model_name} on task {task['id']}...", end=" ", flush=True)
        
        start_time = time.time()
        
        try:
            # 生成响应
            response = await agent.generate_llm_response(task['prompt'])
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 评估响应质量
            quality_score = self._evaluate_response_quality(
                response, 
                task['ground_truth'],
                task.get('reasoning_required', False)
            )
            
            print(f"✓ ({response_time:.2f}s, score: {quality_score}/10)")
            
            return {
                'model': model_name,
                'task_id': task['id'],
                'category': task['category'],
                'response': response,
                'response_time': response_time,
                'quality_score': quality_score,
                'ground_truth': task['ground_truth'],
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return {
                'model': model_name,
                'task_id': task['id'],
                'category': task['category'],
                'error': str(e),
                'success': False
            }
    
    def _evaluate_response_quality(self, response: str, ground_truth: str, 
                                   reasoning_required: bool) -> float:
        """评估响应质量 (简单启发式评分)"""
        score = 5.0  # 基础分
        
        # 检查是否提到ground truth
        if ground_truth.lower() in response.lower():
            score += 3.0
        
        # 检查是否包含推理过程
        if reasoning_required:
            reasoning_indicators = ['because', 'due to', 'since', 'reason', 
                                   'indicates', 'suggests', 'analysis']
            if any(word in response.lower() for word in reasoning_indicators):
                score += 1.5
        
        # 检查响应长度(太短可能不够详细)
        if len(response) > 100:
            score += 0.5
        
        return min(score, 10.0)
    
    async def run_benchmark(self):
        """运行完整基准测试"""
        if not self.models:
            print("\n✗ No models available for testing")
            return
        
        print("\n" + "="*70)
        print(f"Running Benchmark: {len(self.models)} models × {len(BENCHMARK_TASKS)} tasks")
        print("="*70)
        
        for model_name, agent in self.models.items():
            print(f"\nModel: {model_name}")
            
            for task in BENCHMARK_TASKS:
                result = await self.run_single_test(model_name, agent, task)
                self.results.append(result)
                
                # Rate limiting
                await asyncio.sleep(1)
        
        print("\n" + "="*70)
        print("✓ Benchmark Complete!")
        print("="*70)
    
    def analyze_results(self) -> pd.DataFrame:
        """分析测试结果"""
        if not self.results:
            print("No results to analyze")
            return None
        
        # 转换为DataFrame
        df = pd.DataFrame([r for r in self.results if r['success']])
        
        if df.empty:
            print("No successful results to analyze")
            return None
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS ANALYSIS")
        print("="*70)
        
        # 按模型聚合统计
        model_stats = df.groupby('model').agg({
            'quality_score': ['mean', 'std'],
            'response_time': ['mean', 'std'],
            'task_id': 'count'
        }).round(3)
        
        model_stats.columns = ['Avg Quality', 'Std Quality', 
                              'Avg Time(s)', 'Std Time(s)', 'Tasks Completed']
        
        print("\nModel Performance Summary:")
        print(model_stats.to_string())
        
        # 按类别统计
        category_stats = df.groupby(['model', 'category'])['quality_score'].mean().round(2)
        
        print("\n\nPerformance by Task Category:")
        print(category_stats.to_string())
        
        # 保存详细结果
        df.to_csv('multi_llm_benchmark_results.csv', index=False)
        print("\n✓ Detailed results saved to: multi_llm_benchmark_results.csv")
        
        return df
    
    def visualize_results(self, df: pd.DataFrame, output_dir: str = '../../fig'):
        """可视化测试结果"""
        if df is None or df.empty:
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Quality Score Comparison
        ax1 = axes[0, 0]
        df.groupby('model')['quality_score'].mean().sort_values().plot(
            kind='barh', ax=ax1, color='skyblue'
        )
        ax1.set_xlabel('Average Quality Score (0-10)')
        ax1.set_title('Model Quality Comparison')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Response Time Comparison
        ax2 = axes[0, 1]
        df.groupby('model')['response_time'].mean().sort_values().plot(
            kind='barh', ax=ax2, color='lightcoral'
        )
        ax2.set_xlabel('Average Response Time (seconds)')
        ax2.set_title('Model Speed Comparison')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Quality by Category (heatmap)
        ax3 = axes[1, 0]
        pivot_quality = df.pivot_table(
            values='quality_score', 
            index='category', 
            columns='model',
            aggfunc='mean'
        )
        sns.heatmap(pivot_quality, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax3)
        ax3.set_title('Quality Score by Task Category')
        
        # 4. Scatter: Quality vs Speed
        ax4 = axes[1, 1]
        model_summary = df.groupby('model').agg({
            'quality_score': 'mean',
            'response_time': 'mean'
        }).reset_index()
        
        for _, row in model_summary.iterrows():
            ax4.scatter(row['response_time'], row['quality_score'], s=200, alpha=0.6)
            ax4.annotate(row['model'], 
                        (row['response_time'], row['quality_score']),
                        fontsize=9, ha='center')
        
        ax4.set_xlabel('Average Response Time (s)')
        ax4.set_ylabel('Average Quality Score')
        ax4.set_title('Quality vs Speed Trade-off')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'multi_llm_performance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        plt.close()

async def main():
    """主函数"""
    print("\n" + "="*70)
    print("MULTI-LLM BENCHMARK FOR BITCOIN TRANSACTION ANALYSIS")
    print("="*70)
    
    benchmark = LLMBenchmark()
    
    # 初始化模型
    models_ready = await benchmark.initialize_models()
    
    if not models_ready:
        print("\n✗ Cannot proceed without models")
        print("\nPlease set API keys and try again:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  export GOOGLE_API_KEY='AI...'")
        return
    
    # 运行基准测试
    await benchmark.run_benchmark()
    
    # 分析结果
    df = benchmark.analyze_results()
    
    # 可视化
    if df is not None:
        benchmark.visualize_results(df)
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 multi_llm_benchmark_results.csv")
    print("  📈 ../../fig/multi_llm_performance_comparison.png")
    print("\nKey findings:")
    if df is not None:
        best_quality = df.groupby('model')['quality_score'].mean().idxmax()
        best_speed = df.groupby('model')['response_time'].mean().idxmin()
        print(f"  🏆 Best quality: {best_quality}")
        print(f"  ⚡ Fastest: {best_speed}")

if __name__ == "__main__":
    asyncio.run(main())

