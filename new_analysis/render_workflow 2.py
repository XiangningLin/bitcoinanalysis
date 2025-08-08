#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render professional workflow diagram for the paper
使用现代化设计风格
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

CURRENT_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(CURRENT_DIR, "outputs", "workflow.png")


def add_box(ax, xy, width, height, text, color="#E8F4F8", edge_color="#2E5090", 
            fontsize=10, shadow=True):
    """添加圆角矩形框（带阴影）"""
    x, y = xy
    
    # 阴影效果
    if shadow:
        shadow_box = FancyBboxPatch((x+0.03, y-0.03), width, height, 
                                   boxstyle="round,pad=0.08",
                                   facecolor='#00000015', edgecolor='none')
        ax.add_patch(shadow_box)
    
    # 主体框
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=edge_color, linewidth=2.0)
    ax.add_patch(box)
    
    # 文字
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, 
            weight='bold', color='#1F2937')
    return box


def add_arrow(ax, start, end, label="", style='-|>', color='#64748B', lw=2.5):
    """添加现代化箭头"""
    arrow = FancyArrowPatch(start, end,
                           arrowstyle=style, 
                           color=color, lw=lw,
                           connectionstyle="arc3,rad=0.1",
                           alpha=0.8)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2
        ax.text(mid_x, mid_y + 0.15, label, 
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.9),
                color='#475569', weight='600')


def add_icon(ax, xy, icon_type='agent', size=0.3, color='#4F46E5'):
    """添加精美图标（使用 Unicode emoji 或几何图形）"""
    x, y = xy
    
    if icon_type == 'agent':
        # 使用 emoji 机器人图标
        ax.text(x, y, '🤖', fontsize=size*80, ha='center', va='center', zorder=15)
    
    elif icon_type == 'coordinator':
        # 使用齿轮图标表示协调器
        ax.text(x, y, '⚙️', fontsize=size*80, ha='center', va='center', zorder=15)
    
    elif icon_type == 'data':
        # 使用数据库图标
        ax.text(x, y, '💾', fontsize=size*80, ha='center', va='center', zorder=15)
    
    elif icon_type == 'llm':
        # 使用闪电图标表示 AI
        ax.text(x, y, '⚡', fontsize=size*80, ha='center', va='center', zorder=15)


def render_workflow():
    # 使用更大画布和专业配色
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis('off')
    
    # 背景渐变效果
    from matplotlib.patches import Rectangle
    bg = Rectangle((0, 0), 10, 7, facecolor='#F8FAFC', zorder=-1)
    ax.add_patch(bg)
    
    # Title with underline
    ax.text(5, 7.1, "Multi-Agent LLM Framework Workflow",
            ha='center', fontsize=16, weight='bold', color='#0F172A')
    ax.plot([2, 8], [6.95, 6.95], 'k-', lw=1.5, alpha=0.3)
    
    # === Layer 1: Data Input ===
    add_box(ax, (3.8, 6.1), 2.4, 0.6, "Transaction Graph", 
            "#EFF6FF", "#1E40AF", 11)
    
    # === Layer 2: Dual Sampling Paths ===
    add_box(ax, (1.2, 4.9), 1.7, 0.7, "RWFB\nSampling", 
            "#DBEAFE", "#2563EB", 10)
    add_box(ax, (7.1, 4.9), 1.7, 0.7, "CETraS\nSampling", 
            "#D1FAE5", "#059669", 10)
    
    add_arrow(ax, (4.5, 6.1), (2.0, 5.6), "", color='#3B82F6', lw=2.5)
    add_arrow(ax, (5.5, 6.1), (7.9, 5.6), "", color='#10B981', lw=2.5)
    
    # === Layer 3: Framework Core ===
    # Agent Coordinator (中心位置)
    add_box(ax, (3.5, 3.7), 3.0, 0.8, "Agent Coordinator\n(Task Routing)", 
            "#FEF3C7", "#D97706", 10)
    
    add_arrow(ax, (2.0, 4.9), (4.5, 4.5), "", color='#64748B', lw=2.0)
    add_arrow(ax, (7.9, 4.9), (5.5, 4.5), "", color='#64748B', lw=2.0)
    
    # === Layer 4: Three Agents (横排，精美设计) ===
    # 添加分组框
    group_box = Rectangle((1.0, 1.5), 8.0, 1.4, 
                          facecolor='#F1F5F9', edgecolor='#CBD5E1', 
                          linewidth=1.5, linestyle='--', alpha=0.5, zorder=0)
    ax.add_patch(group_box)
    
    ax.text(5, 2.8, "Specialized Agents", ha='center', fontsize=10, 
            weight='bold', color='#334155', style='italic')
    
    add_box(ax, (1.3, 1.8), 2.2, 0.75, "Role\nClassifier", 
            "#EEF2FF", "#4F46E5", 9)
    add_box(ax, (3.9, 1.8), 2.2, 0.75, "Anomaly\nAnalyst", 
            "#FDF2F8", "#DB2777", 9)
    add_box(ax, (6.5, 1.8), 2.2, 0.75, "Decentralization\nSummarizer", 
            "#FEF3C7", "#EA580C", 9)
    
    # Coordinator to Agents (美化箭头)
    add_arrow(ax, (5.0, 3.7), (2.4, 2.55), "", color='#6366F1', lw=2.0)
    add_arrow(ax, (5.0, 3.7), (5.0, 2.55), "", color='#DB2777', lw=2.0)
    add_arrow(ax, (5.0, 3.7), (7.6, 2.55), "", color='#EA580C', lw=2.0)
    
    # === Layer 5: LLM API ===
    add_box(ax, (3.9, 0.9), 2.2, 0.6, "LLM API", 
            "#FEF2F2", "#DC2626", 10)
    
    # Agents to LLM (细箭头)
    for x_agent in [2.4, 5.0, 7.6]:
        add_arrow(ax, (x_agent, 1.8), (5.0, 1.5), "", 
                 color='#94A3B8', lw=1.5, style='->')
    
    # === Layer 6: Output ===
    add_box(ax, (3.6, 0.1), 2.8, 0.5, "Analysis Results", 
            "#F0FDF4", "#15803D", 10, shadow=False)
    
    add_arrow(ax, (5.0, 0.9), (5.0, 0.6), "", color='#15803D', lw=2.5)
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='#EFF6FF', edgecolor='#1E40AF', 
                      label='Data Layer', linewidth=2),
        mpatches.Patch(facecolor='#FEF3C7', edgecolor='#D97706', 
                      label='Framework Core', linewidth=2),
        mpatches.Patch(facecolor='#EEF2FF', edgecolor='#4F46E5', 
                      label='LLM Agents', linewidth=2),
        mpatches.Patch(facecolor='#FEF2F2', edgecolor='#DC2626', 
                      label='External API', linewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=9, framealpha=0.98, edgecolor='#CBD5E1', 
             fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=350, bbox_inches='tight', facecolor='white')
    print(f"✅ Professional workflow diagram saved to: {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    render_workflow()
