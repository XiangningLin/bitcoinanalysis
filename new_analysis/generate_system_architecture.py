"""
Generate system architecture diagram inspired by the reference images
Shows layered architecture with clear control flow and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Colors
COLORS = {
    'coordinator': '#FFB6C1',
    'messagebus': '#FF6B6B',
    'agents': '#87CEEB',
    'collaboration': '#FFD700',
    'database': '#90EE90',
    'control': '#FF4444',
    'data': '#4169E1',
}

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# ================== Top Layer: Input/Output ==================
# LLM Inference Requests (top)
requests_y = 11
for i in range(15):
    x = 2 + i * 0.8
    color = ['#FF6B6B', '#4169E1', '#FFD700', '#87CEEB'][i % 4]
    arrow = FancyArrowPatch(
        (x, requests_y + 0.3),
        (x, requests_y),
        arrowstyle='->,head_width=0.15,head_length=0.2',
        linewidth=1.5,
        color=color,
        alpha=0.6
    )
    ax.add_patch(arrow)

ax.text(8, requests_y + 0.7, 'LLM Analysis Requests (RWFB & CETraS samples)',
        ha='center', fontsize=11, fontweight='bold', style='italic',
        color='#333333')

# ================== Layer 1: Frontend/Interface ==================
frontend_y = 9.8
frontend_box = Rectangle(
    (1.5, frontend_y - 0.4),
    13, 0.8,
    facecolor='#F0F0F0',
    edgecolor='black',
    linewidth=2.5
)
ax.add_patch(frontend_box)
ax.text(8, frontend_y, 'Frontend / API Gateway',
        ha='center', va='center', fontsize=12, fontweight='bold')

# ================== Layer 2: Coordinator & CollaborationManager ==================
layer2_y = 8.2

# CollaborativeCoordinator (left)
coord_box = FancyBboxPatch(
    (2, layer2_y - 0.6),
    4.5, 1.4,
    boxstyle="round,pad=0.1",
    edgecolor='black',
    facecolor=COLORS['coordinator'],
    linewidth=2.5
)
ax.add_patch(coord_box)
ax.text(4.25, layer2_y + 0.45, 'CollaborativeCoordinator',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Add numbered circles for stages
stage_items = ['①', '②', '③', '④', '⑤']
for i, stage_num in enumerate(stage_items):
    circle = Circle((2.5 + i * 0.8, layer2_y - 0.2), 0.18,
                   facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(2.5 + i * 0.8, layer2_y - 0.2, stage_num,
            ha='center', va='center', fontsize=9, fontweight='bold')

ax.text(4.25, layer2_y - 0.55, 'Orchestrate 5-Stage Workflow',
        ha='center', va='center', fontsize=8, style='italic')

# CollaborationManager (right)
collab_box = FancyBboxPatch(
    (9.5, layer2_y - 0.6),
    4.5, 1.4,
    boxstyle="round,pad=0.1",
    edgecolor='black',
    facecolor=COLORS['collaboration'],
    linewidth=2.5
)
ax.add_patch(collab_box)
ax.text(11.75, layer2_y + 0.45, 'CollaborationManager',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Stage control indicators
stage_names = ['Info\nShare', 'Valid', 'Consensus']
for i, name in enumerate(stage_names):
    small_box = Rectangle(
        (10 + i * 1.3, layer2_y - 0.3),
        1.1, 0.5,
        facecolor='white',
        edgecolor='#FF6B35',
        linewidth=1.5
    )
    ax.add_patch(small_box)
    ax.text(10.55 + i * 1.3, layer2_y - 0.05, name,
            ha='center', va='center', fontsize=7, fontweight='bold',
            color='#FF6B35')

# Arrow between Coordinator and CollaborationManager
arrow_coord_collab = FancyArrowPatch(
    (6.5, layer2_y),
    (9.5, layer2_y),
    arrowstyle='<->,head_width=0.4,head_length=0.5',
    linewidth=2.5,
    color='#FF6B35',
    linestyle='--'
)
ax.add_patch(arrow_coord_collab)
ax.text(8, layer2_y + 0.4, 'Stage Control',
        ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#FF6B35'))

# Arrows from Frontend to Layer 2
arrow_front_coord = FancyArrowPatch(
    (4.25, frontend_y - 0.4),
    (4.25, layer2_y + 0.8),
    arrowstyle='->,head_width=0.5,head_length=0.6',
    linewidth=2.5,
    color=COLORS['control'],
    alpha=0.7
)
ax.add_patch(arrow_front_coord)
ax.text(4.8, 9, '❹', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# ================== Layer 3: MessageBus ==================
messagebus_y = 6
messagebus_box = FancyBboxPatch(
    (2.5, messagebus_y - 0.5),
    11, 1.2,
    boxstyle="round,pad=0.1",
    edgecolor='black',
    facecolor=COLORS['messagebus'],
    linewidth=3
)
ax.add_patch(messagebus_box)
ax.text(8, messagebus_y + 0.3, 'MessageBus',
        ha='center', va='center', fontsize=13, fontweight='bold', color='white')
ax.text(8, messagebus_y - 0.15, 'Broadcast | Route | Track Message History',
        ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Arrows from Coordinator to MessageBus
arrow_coord_bus = FancyArrowPatch(
    (4.25, layer2_y - 0.6),
    (5, messagebus_y + 0.5),
    arrowstyle='->,head_width=0.5,head_length=0.6',
    linewidth=2.5,
    color=COLORS['control'],
    alpha=0.8
)
ax.add_patch(arrow_coord_bus)
ax.text(4, 7, '❺', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# ================== Layer 4: Function & Database ==================
database_y = 4.5
db_box = Rectangle(
    (1.5, database_y - 0.4),
    13, 0.8,
    facecolor=COLORS['database'],
    edgecolor='black',
    linewidth=2
)
ax.add_patch(db_box)

# Text only
ax.text(8, database_y, 'Metrics & Results Database',
        ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow from MessageBus to Database
arrow_bus_db = FancyArrowPatch(
    (8, messagebus_y - 0.5),
    (8, database_y + 0.4),
    arrowstyle='<->,head_width=0.4,head_length=0.5',
    linewidth=2,
    color='#666666',
    linestyle='--'
)
ax.add_patch(arrow_bus_db)
ax.text(8.5, 5.2, '❶', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# ================== Layer 5: Agents (in Warmed Container Pool) ==================
agents_y = 2.5
container_box = FancyBboxPatch(
    (1, agents_y - 1.7),
    14, 2.8,
    boxstyle="round,pad=0.15",
    edgecolor='#FF6B35',
    facecolor='#FFF5F5',
    linewidth=3,
    linestyle='--'
)
ax.add_patch(container_box)
ax.text(8, agents_y + 1, 'Warmed Agent Pool',
        ha='center', va='center', fontsize=10, fontweight='bold',
        color='#FF6B35')

# Three agents
agents_info = [
    {'name': 'RoleClassifier\nAgent', 'x': 3, 'color': '#4169E1', 
     'functions': ['Classify', 'Adjust', 'Vote']},
    {'name': 'AnomalyAnalyst\nAgent', 'x': 8, 'color': '#FF6B35',
     'functions': ['Detect', 'Assess', 'Share']},
    {'name': 'Decentralization\nAgent', 'x': 13, 'color': '#00B945',
     'functions': ['Synthesize', 'Score', 'Refine']}
]

for agent in agents_info:
    # Main agent box
    agent_box = FancyBboxPatch(
        (agent['x'] - 1.5, agents_y - 1.3),
        3, 2,
        boxstyle="round,pad=0.12",
        edgecolor='black',
        facecolor=agent['color'],
        alpha=0.3,
        linewidth=2.5
    )
    ax.add_patch(agent_box)
    
    # Agent name
    ax.text(agent['x'], agents_y + 0.4, agent['name'],
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Pre-loading indicator
    preload_box = Rectangle(
        (agent['x'] - 1.3, agents_y - 0.1),
        2.6, 0.4,
        facecolor='white',
        edgecolor=agent['color'],
        linewidth=2
    )
    ax.add_patch(preload_box)
    ax.text(agent['x'], agents_y + 0.1, 'Pre-Loading Agent',
            fontsize=8, ha='center', va='center', fontweight='bold',
            color=agent['color'], style='italic')
    
    # Function boxes
    for i, func in enumerate(agent['functions']):
        func_box = Rectangle(
            (agent['x'] - 1.2 + i * 0.85, agents_y - 0.75),
            0.75, 0.4,
            facecolor='white',
            edgecolor=agent['color'],
            linewidth=1.5
        )
        ax.add_patch(func_box)
        ax.text(agent['x'] - 0.825 + i * 0.85, agents_y - 0.55,
                func, ha='center', va='center', fontsize=7, fontweight='bold')

# Add numbered circles for agent workflow
ax.text(3, agents_y - 1.05, '❷', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
ax.text(8, agents_y - 1.05, '❸', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
ax.text(13, agents_y - 1.05, '❸', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# Arrows from MessageBus to Agents
for agent in agents_info:
    # Down arrow (MessageBus -> Agent)
    arrow_bus_agent = FancyArrowPatch(
        (agent['x'], messagebus_y - 0.5),
        (agent['x'], agents_y + 0.7),
        arrowstyle='->,head_width=0.5,head_length=0.6',
        linewidth=2.5,
        color=COLORS['data'],
        alpha=0.7
    )
    ax.add_patch(arrow_bus_agent)
    
    # Up arrow (Agent -> MessageBus)
    arrow_agent_bus = FancyArrowPatch(
        (agent['x'] + 0.3, agents_y + 0.7),
        (agent['x'] + 0.3, messagebus_y - 0.5),
        arrowstyle='->,head_width=0.5,head_length=0.6',
        linewidth=2.5,
        color=agent['color'],
        alpha=0.7
    )
    ax.add_patch(arrow_agent_bus)

# Peer-to-peer arrows between agents (curved)
from matplotlib.patches import ConnectionPatch

# Agent 1 <-> Agent 2
con1 = ConnectionPatch((4.5, agents_y), (6.5, agents_y), "data", "data",
                       arrowstyle="<->", shrinkA=5, shrinkB=5,
                       mutation_scale=20, fc='#666666', 
                       linewidth=2, linestyle=':', alpha=0.6)
ax.add_artist(con1)

# Agent 2 <-> Agent 3
con2 = ConnectionPatch((9.5, agents_y), (11.5, agents_y), "data", "data",
                       arrowstyle="<->", shrinkA=5, shrinkB=5,
                       mutation_scale=20, fc='#666666', 
                       linewidth=2, linestyle=':', alpha=0.6)
ax.add_artist(con2)

ax.text(8, agents_y - 1.6, 'Peer-to-Peer Feedback (via MessageBus)',
        fontsize=9, ha='center', style='italic', fontweight='bold',
        color='#666666')

# ================== Bottom: CPU/GPU Offloader ==================
offloader_y = 0.3
offloader_box = Rectangle(
    (4, offloader_y - 0.2),
    8, 0.5,
    facecolor='#FFE4B5',
    edgecolor='#FF6B35',
    linewidth=2.5
)
ax.add_patch(offloader_box)
ax.text(8, offloader_y, 'CPU ⇄ GPU Dynamic Offloader',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color='#FF6B35', style='italic')

# Arrow from agents to offloader
arrow_agent_offload = FancyArrowPatch(
    (8, agents_y - 1.3),
    (8, offloader_y + 0.3),
    arrowstyle='<->,head_width=0.3,head_length=0.4',
    linewidth=2,
    color='#666666',
    linestyle='--'
)
ax.add_patch(arrow_agent_offload)
ax.text(8.5, 1, '❼', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# ================== Legend and Annotations ==================

# Control Flow vs Data Flow legend
ax.text(0.5, 10, 'Legend:', fontsize=10, fontweight='bold', ha='left')

# Control flow arrow
arrow_control = FancyArrowPatch(
    (0.5, 9.5),
    (1.5, 9.5),
    arrowstyle='->,head_width=0.3,head_length=0.4',
    linewidth=2.5,
    color=COLORS['control']
)
ax.add_patch(arrow_control)
ax.text(2, 9.5, 'Control Flow', fontsize=8, va='center', fontweight='bold')

# Data flow arrow
arrow_data = FancyArrowPatch(
    (0.5, 9),
    (1.5, 9),
    arrowstyle='->,head_width=0.3,head_length=0.4',
    linewidth=2.5,
    color=COLORS['data']
)
ax.add_patch(arrow_data)
ax.text(2, 9, 'Data Flow', fontsize=8, va='center', fontweight='bold')

# Peer feedback
ax.plot([0.5, 1.5], [8.5, 8.5], ':', linewidth=2, color='#666666')
ax.text(2, 8.5, 'Peer Feedback', fontsize=8, va='center', fontweight='bold')

# Key metrics box
metrics_text = (
    'Collaboration Metrics:\n'
    '• 9 info exchanges\n'
    '• 195/2000 adjusted (9.75%)\n'
    '• Efficiency: 85%\n'
    '• SDM: 0.397→0.257 (↓35%)\n'
    '• Disagreement: 18.8%→9.2% (↓51%)'
)
ax.text(14.5, 4, metrics_text,
        fontsize=7.5, ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD', 
                 edgecolor='black', linewidth=2))

# Title
ax.text(8, 12.2, 'Collaborative Multi-Agent System Architecture',
        fontsize=16, ha='center', fontweight='bold', color='#333333')

ax.text(8, 11.8, 'Bias Decomposition via Result Reconciliation',
        fontsize=12, ha='center', style='italic', color='#666666')

# Stage descriptions (right side)
stage_x = 15.2
stage_descriptions = [
    '① Initial Analysis',
    '② Information Sharing', 
    '③ Collaborative Validation',
    '④ Consensus Building',
    '⑤ Result Integration'
]

ax.text(stage_x, 8, '5-Stage Workflow:', fontsize=9, fontweight='bold', ha='left')
for i, desc in enumerate(stage_descriptions):
    y = 7.5 - i * 0.4
    ax.text(stage_x, y, desc, fontsize=7.5, ha='left',
            color=['#4169E1', '#FF6B35', '#00B945', '#845B97', '#FFD700'][i])

ax.set_xlim(0, 17)
ax.set_ylim(0, 12.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/linxiangning/Desktop/000000ieeeprojects/fig/workflow_collaborative.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Generated system architecture diagram: /Users/linxiangning/Desktop/000000ieeeprojects/fig/workflow_collaborative.png")
plt.close()

