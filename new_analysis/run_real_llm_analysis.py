#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实的LLM分析脚本 - 使用OpenAI API
"""

import os
import sys
import json
import time
from typing import Dict, List
from openai import OpenAI

def test_single_node_analysis():
    """测试单个节点的LLM分析"""
    
    # 检查API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ 请先设置 OPENAI_API_KEY")
        return
    
    client = OpenAI()
    
    # 加载一个测试节点
    with open('data/llm4tg_nodes_top1000_cetras.jsonl', 'r') as f:
        first_line = f.readline().strip()
        test_node = json.loads(first_line)
    
    print(f"🔍 测试节点: {test_node['id']}")
    print(f"   度数: {test_node['total_degree']}")
    print(f"   中介中心性: {test_node['betweenness']:.6f}")
    
    # 构建prompt
    prompt = f"""
作为Bitcoin交易网络分析专家，请分析这个节点的角色：

节点信息：
- ID: {test_node['id']}
- 入度: {test_node['in_degree']}
- 出度: {test_node['out_degree']}
- 总度数: {test_node['total_degree']}
- 中介中心性: {test_node['betweenness']:.6f}

请基于这些指标，将此节点分类为以下角色之一：
- exchange_hot_wallet (交易所热钱包)
- exchange_cold_wallet (交易所冷钱包)
- mixer_tumbler (混币服务)
- mining_pool_payout (矿池支付)
- merchant_gateway (商户网关)
- service_aggregator (服务聚合器)
- retail_user (零售用户)

请用JSON格式返回分析结果：
{{"role": "角色名称", "confidence": 0.85, "rationale": "分类理由"}}
"""

    print("\n🤖 调用OpenAI API...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个Bitcoin交易网络分析专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        result = response.choices[0].message.content
        print(f"\n✅ LLM响应:")
        print(result)
        
        # 尝试解析JSON
        try:
            parsed_result = json.loads(result.strip('```json').strip('```').strip())
            print(f"\n📊 解析结果:")
            print(f"   角色: {parsed_result.get('role', 'unknown')}")
            print(f"   置信度: {parsed_result.get('confidence', 0)}")
            print(f"   理由: {parsed_result.get('rationale', 'N/A')}")
            return parsed_result
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            print(f"原始响应: {result}")
            return None
            
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return None

def run_small_batch_analysis():
    """运行小批量真实分析（前10个节点）"""
    
    print("🚀 运行小批量真实LLM分析 (前10个节点)")
    print("="*60)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ 请先设置 OPENAI_API_KEY")
        return
    
    client = OpenAI()
    
    # 加载CETraS数据的前10个节点
    cetras_nodes = []
    with open('data/llm4tg_nodes_top1000_cetras.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # 只处理前10个
                break
            cetras_nodes.append(json.loads(line.strip()))
    
    # 加载RWFB数据的前10个节点
    rwfb_nodes = []
    with open('data/llm4tg_nodes_top1000_rwfb.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # 只处理前10个
                break
            rwfb_nodes.append(json.loads(line.strip()))
    
    print(f"📊 加载了 {len(cetras_nodes)} CETraS节点, {len(rwfb_nodes)} RWFB节点")
    
    # 分析CETraS节点
    print("\n🔍 分析CETraS节点...")
    cetras_results = []
    for i, node in enumerate(cetras_nodes):
        print(f"   处理节点 {i+1}/10: {node['id']}")
        result = analyze_node_with_llm(client, node, f"CETraS-{i+1}")
        if result:
            cetras_results.append(result)
        time.sleep(1)  # 避免rate limit
    
    # 分析RWFB节点
    print("\n🔍 分析RWFB节点...")
    rwfb_results = []
    for i, node in enumerate(rwfb_nodes):
        print(f"   处理节点 {i+1}/10: {node['id']}")
        result = analyze_node_with_llm(client, node, f"RWFB-{i+1}")
        if result:
            rwfb_results.append(result)
        time.sleep(1)  # 避免rate limit
    
    # 分析结果
    print(f"\n📊 分析结果:")
    print(f"   CETraS成功: {len(cetras_results)}/10")
    print(f"   RWFB成功: {len(rwfb_results)}/10")
    
    # 统计角色分布
    cetras_roles = {}
    for result in cetras_results:
        role = result.get('role', 'unknown')
        cetras_roles[role] = cetras_roles.get(role, 0) + 1
    
    rwfb_roles = {}
    for result in rwfb_results:
        role = result.get('role', 'unknown')
        rwfb_roles[role] = rwfb_roles.get(role, 0) + 1
    
    print(f"\n🎯 角色分布对比:")
    print("CETraS角色:")
    for role, count in cetras_roles.items():
        print(f"   {role}: {count}")
    
    print("RWFB角色:")
    for role, count in rwfb_roles.items():
        print(f"   {role}: {count}")
    
    # 保存结果
    results = {
        "method": "small_batch_real_llm",
        "sample_size": 10,
        "cetras_results": cetras_results,
        "rwfb_results": rwfb_results,
        "cetras_role_distribution": cetras_roles,
        "rwfb_role_distribution": rwfb_roles,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs("outputs/real_llm", exist_ok=True)
    with open("outputs/real_llm/small_batch_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到 outputs/real_llm/small_batch_results.json")
    
    return results

def analyze_node_with_llm(client, node, node_label):
    """使用LLM分析单个节点"""
    
    prompt = f"""
作为Bitcoin交易网络分析专家，请分析这个节点的角色：

节点信息：
- ID: {node['id']}
- 入度: {node['in_degree']}
- 出度: {node['out_degree']}
- 总度数: {node['total_degree']}
- 中介中心性: {node['betweenness']:.6f}

分析要点：
1. 高入度+低出度通常表示冷钱包或接收地址
2. 高出度+低入度通常表示热钱包或支付地址
3. 高中介中心性表示重要的中转节点
4. 平衡的入出度可能是交易服务或混币器

请将此节点分类为以下角色之一：
- exchange_hot_wallet (交易所热钱包)
- exchange_cold_wallet (交易所冷钱包)  
- mixer_tumbler (混币服务)
- mining_pool_payout (矿池支付)
- merchant_gateway (商户网关)
- service_aggregator (服务聚合器)
- retail_user (零售用户)

请用JSON格式返回：
{{"role": "角色名称", "confidence": 0.85, "rationale": "基于度数分布和中介中心性的分类理由"}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个Bitcoin交易网络分析专家，擅长根据节点的度数特征和中心性指标判断其在网络中的角色。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        result = response.choices[0].message.content.strip()
        
        # 尝试提取JSON
        if result.startswith('```json'):
            result = result[7:]
        if result.endswith('```'):
            result = result[:-3]
        result = result.strip()
        
        # 解析JSON
        parsed_result = json.loads(result)
        parsed_result['node_id'] = node['id']
        parsed_result['node_label'] = node_label
        
        return parsed_result
        
    except json.JSONDecodeError as e:
        print(f"    ⚠️ JSON解析失败: {e}")
        # 尝试从原始文本中提取信息
        try:
            # 简单的后备解析
            if 'exchange_hot_wallet' in result:
                role = 'exchange_hot_wallet'
            elif 'exchange_cold_wallet' in result:
                role = 'exchange_cold_wallet'
            elif 'mixer' in result or 'tumbler' in result:
                role = 'mixer_tumbler'
            elif 'mining' in result:
                role = 'mining_pool_payout'
            elif 'merchant' in result:
                role = 'merchant_gateway'
            elif 'service' in result:
                role = 'service_aggregator'
            else:
                role = 'retail_user'
            
            return {
                'role': role,
                'confidence': 0.7,
                'rationale': 'Extracted from text due to JSON parse error',
                'node_id': node['id'],
                'node_label': node_label,
                'raw_response': result
            }
        except:
            print(f"    ❌ 完全解析失败")
            return None
            
    except Exception as e:
        print(f"    ❌ API调用失败: {e}")
        return None

def main():
    print("🧪 真实LLM分析测试")
    print("="*50)
    
    # 检查API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ 请先设置 OPENAI_API_KEY")
        print("export OPENAI_API_KEY='your-key-here'")
        return
    
    # 选择测试类型
    choice = input("\n选择测试类型:\n1. 单节点测试\n2. 小批量测试(10个节点)\n请输入选择 (1或2): ")
    
    if choice == '1':
        test_single_node_analysis()
    elif choice == '2':
        results = run_small_batch_analysis()
        print(f"\n🎉 小批量分析完成!")
        print(f"📁 结果保存在 outputs/real_llm/")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
