"""
Generate collaborative multi-agent architecture diagram showing agents, MessageBus, and Coordinator
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Nature journal style colors
NATURE_COLORS = {
    'primary': '#0C5DA5',
    'secondary': '#FF6B35',
    'accent1': '#00B945',
    'accent2': '#845B97',
    'coordinator': '#FFD700',
    'messagebus': '#DC143C',
    'neutral': '#474747',
}

fig, ax = plt.subplots(1, 1, figsize=(16, 11))

# ============ Central Components ============

# MessageBus (center)
messagebus_x, messagebus_y = 8, 5.5
messagebus = FancyBboxPatch(
    (messagebus_x - 1.5, messagebus_y - 0.6),
    3, 1.2,
    boxstyle="round,pad=0.15",
    edgecolor='black',
    facecolor=NATURE_COLORS['messagebus'],
    alpha=0.3,
    linewidth=3
)
ax.add_patch(messagebus)
ax.text(messagebus_x, messagebus_y + 0.25, 'MessageBus', 
        ha='center', va='center', fontsize=13, fontweight='bold', color='black')
ax.text(messagebus_x, messagebus_y - 0.15, 'Broadcast & Route\nInformation Exchange',
        ha='center', va='center', fontsize=9, color=NATURE_COLORS['neutral'])

# Coordinator (top)
coord_x, coord_y = 8, 9
coordinator = FancyBboxPatch(
    (coord_x - 1.8, coord_y - 0.7),
    3.6, 1.4,
    boxstyle="round,pad=0.15",
    edgecolor='black',
    facecolor=NATURE_COLORS['coordinator'],
    alpha=0.3,
    linewidth=3
)
ax.add_patch(coordinator)
ax.text(coord_x, coord_y + 0.35, 'CollaborativeCoordinator', 
        ha='center', va='center', fontsize=13, fontweight='bold', color='black')
ax.text(coord_x, coord_y - 0.2, 'Orchestrate 5-Stage Workflow\nTrack Metrics & Consensus',
        ha='center', va='center', fontsize=9, color=NATURE_COLORS['neutral'])

# CollaborationManager (right of coordinator)
collab_mgr_x, collab_mgr_y = 12.5, 9
collab_mgr = FancyBboxPatch(
    (collab_mgr_x - 1.2, collab_mgr_y - 0.6),
    2.4, 1.2,
    boxstyle="round,pad=0.1",
    edgecolor='black',
    facecolor='#9370DB',
    alpha=0.25,
    linewidth=2.5
)
ax.add_patch(collab_mgr)
ax.text(collab_mgr_x, collab_mgr_y + 0.25, 'Collaboration\nManager', 
        ha='center', va='center', fontsize=10, fontweight='bold', color='black')
ax.text(collab_mgr_x, collab_mgr_y - 0.25, 'Stage Control',
        ha='center', va='center', fontsize=8, color=NATURE_COLORS['neutral'])

# ============ Three Agents (bottom) ============

agents_y = 2
agents_info = [
    {
        'name': 'Role\nClassifier',
        'x': 3,
        'color': NATURE_COLORS['primary'],
        'tasks': 'Classify nodes\nAdjust confidence\nIntegrate feedback'
    },
    {
        'name': 'Anomaly\nAnalyst',
        'x': 8,
        'color': NATURE_COLORS['secondary'],
        'tasks': 'Detect patterns\nContext-aware risk\nShare findings'
    },
    {
        'name': 'Decentralization\nSummarizer',
        'x': 13,
        'color': NATURE_COLORS['accent1'],
        'tasks': 'Assess centralization\nSynthesize evidence\nRefine scores'
    }
]

agent_boxes = []
for agent in agents_info:
    box = FancyBboxPatch(
        (agent['x'] - 1.3, agents_y - 0.8),
        2.6, 1.6,
        boxstyle="round,pad=0.15",
        edgecolor='black',
        facecolor=agent['color'],
        alpha=0.3,
        linewidth=2.5
    )
    ax.add_patch(box)
    
    ax.text(agent['x'], agents_y + 0.5, agent['name'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    ax.text(agent['x'], agents_y - 0.15, agent['tasks'],
            ha='center', va='center', fontsize=8, color=NATURE_COLORS['neutral'])
    
    agent_boxes.append((agent['x'], agents_y, agent['color']))

# ============ Arrows: Coordinator -> MessageBus ============
arrow1 = FancyArrowPatch(
    (coord_x, coord_y - 0.7),
    (messagebus_x, messagebus_y + 0.6),
    arrowstyle='->,head_width=0.5,head_length=0.6',
    linewidth=2.5,
    color=NATURE_COLORS['coordinator'],
    alpha=0.8
)
ax.add_patch(arrow1)
ax.text(coord_x + 0.5, 7.3, 'Task\nAssignment',
        fontsize=8, ha='center', style='italic', color=NATURE_COLORS['neutral'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# ============ Arrows: CollaborationManager -> Coordinator ============
arrow_cm = FancyArrowPatch(
    (collab_mgr_x - 1.2, collab_mgr_y),
    (coord_x + 1.8, coord_y),
    arrowstyle='<->,head_width=0.4,head_length=0.5',
    linewidth=2,
    color='purple',
    alpha=0.7,
    linestyle='--'
)
ax.add_patch(arrow_cm)
ax.text(10.2, 9.7, 'Stage\nControl',
        fontsize=7, ha='center', style='italic', color='purple')

# ============ Arrows: MessageBus <-> Agents (bidirectional) ============
for i, (agent_x, agent_y, agent_color) in enumerate(agent_boxes):
    # Agent -> MessageBus
    arrow_up = FancyArrowPatch(
        (agent_x, agent_y + 0.8),
        (messagebus_x + (agent_x - messagebus_x) * 0.3, messagebus_y - 0.5),
        arrowstyle='->,head_width=0.4,head_length=0.5',
        linewidth=2,
        color=agent_color,
        alpha=0.7
    )
    ax.add_patch(arrow_up)
    
    # MessageBus -> Agent
    arrow_down = FancyArrowPatch(
        (messagebus_x + (agent_x - messagebus_x) * 0.35, messagebus_y - 0.6),
        (agent_x, agent_y + 0.9),
        arrowstyle='->,head_width=0.4,head_length=0.5',
        linewidth=2,
        color=NATURE_COLORS['messagebus'],
        alpha=0.6,
        linestyle='--'
    )
    ax.add_patch(arrow_down)

# Add labels for agent communication
ax.text(3.5, 4, 'Share\nResults',
        fontsize=7, ha='center', style='italic', color=NATURE_COLORS['primary'])
ax.text(8, 4, 'Broadcast\nInfo',
        fontsize=7, ha='center', style='italic', color=NATURE_COLORS['secondary'])
ax.text(12.5, 4, 'Send\nMetrics',
        fontsize=7, ha='center', style='italic', color=NATURE_COLORS['accent1'])

ax.text(3.5, 4.8, 'Receive\nFeedback',
        fontsize=7, ha='center', style='italic', color=NATURE_COLORS['messagebus'])
ax.text(8, 4.8, 'Get\nContext',
        fontsize=7, ha='center', style='italic', color=NATURE_COLORS['messagebus'])
ax.text(12.5, 4.8, 'Receive\nInputs',
        fontsize=7, ha='center', style='italic', color=NATURE_COLORS['messagebus'])

# ============ Inter-agent collaboration arrows (curved) ============
from matplotlib.patches import ConnectionPatch

# Agent 1 -> Agent 2
con1 = ConnectionPatch((4.5, agents_y), (6.7, agents_y), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=15, fc=NATURE_COLORS['primary'], 
                       linewidth=1.5, linestyle=':', alpha=0.5)
ax.add_artist(con1)

# Agent 2 -> Agent 3
con2 = ConnectionPatch((9.3, agents_y), (11.7, agents_y), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=15, fc=NATURE_COLORS['secondary'], 
                       linewidth=1.5, linestyle=':', alpha=0.5)
ax.add_artist(con2)

# Agent 3 -> Agent 1 (curved back)
con3 = ConnectionPatch((13, agents_y - 0.7), (3, agents_y - 0.7), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=15, fc=NATURE_COLORS['accent1'], 
                       connectionstyle="arc3,rad=-.3",
                       linewidth=1.5, linestyle=':', alpha=0.5)
ax.add_artist(con3)

ax.text(8, 0.8, 'Peer-to-Peer Feedback (via MessageBus)',
        fontsize=8, ha='center', style='italic', color=NATURE_COLORS['neutral'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# ============ Data Flow annotation ============
ax.text(1, 9, 'Input:\nTop-1000\nNodes',
        fontsize=9, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4F8', 
                 edgecolor=NATURE_COLORS['primary'], linewidth=2))

ax.text(15, 9, 'Output:\nSDM=0.257\nDisagree=9.2%',
        fontsize=9, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF4E8', 
                 edgecolor=NATURE_COLORS['accent1'], linewidth=2))

# ============ 5-Stage Workflow (right side) ============
stage_x = 15
stage_y_start = 6.5
stage_spacing = 0.9

stages_short = [
    '① Initial Analysis',
    '② Info Sharing',
    '③ Validation',
    '④ Consensus',
    '⑤ Integration'
]

for i, stage in enumerate(stages_short):
    y = stage_y_start - i * stage_spacing
    stage_color = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                   NATURE_COLORS['accent1'], NATURE_COLORS['accent2'], 
                   NATURE_COLORS['coordinator']][i]
    
    stage_box = Rectangle(
        (stage_x - 0.05, y - 0.25),
        1.6, 0.5,
        facecolor=stage_color,
        alpha=0.25,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(stage_box)
    ax.text(stage_x + 0.75, y, stage, fontsize=8, ha='center', va='center',
            fontweight='bold')

ax.text(stage_x + 0.75, stage_y_start + 0.6, 'Workflow',
        fontsize=10, ha='center', fontweight='bold', color=NATURE_COLORS['neutral'])

# Draw arrow connecting stages
for i in range(len(stages_short) - 1):
    y_start = stage_y_start - i * stage_spacing - 0.25
    y_end = stage_y_start - (i + 1) * stage_spacing + 0.25
    arrow_stage = FancyArrowPatch(
        (stage_x + 0.75, y_start),
        (stage_x + 0.75, y_end),
        arrowstyle='->,head_width=0.3,head_length=0.4',
        linewidth=1.5,
        color=NATURE_COLORS['neutral'],
        alpha=0.6
    )
    ax.add_patch(arrow_stage)

# ============ Key Metrics Box (bottom right) ============
metrics_text = (
    'Collaboration Metrics:\n'
    '• 9 info exchanges\n'
    '• 195/2000 adjustments (9.75%)\n'
    '• 85% efficiency\n'
    '• SDM: 0.397→0.257 (↓35%)'
)
ax.text(12.5, 0.3, metrics_text,
        fontsize=8, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', 
                 edgecolor='black', linewidth=2))

# ============ Title ============
ax.text(8, 10.5, 'Collaborative Multi-Agent Architecture',
        fontsize=16, ha='center', va='center', fontweight='bold',
        color=NATURE_COLORS['neutral'])

ax.text(8, 10.1, 'Result Reconciliation via Information Sharing & Consensus Building',
        fontsize=11, ha='center', va='center', style='italic',
        color=NATURE_COLORS['neutral'])

# ============ Legend ============
legend_elements = [
    mpatches.Patch(facecolor=NATURE_COLORS['primary'], alpha=0.3, 
                  edgecolor='black', label='Role Classifier Agent', linewidth=2),
    mpatches.Patch(facecolor=NATURE_COLORS['secondary'], alpha=0.3, 
                  edgecolor='black', label='Anomaly Analyst Agent', linewidth=2),
    mpatches.Patch(facecolor=NATURE_COLORS['accent1'], alpha=0.3, 
                  edgecolor='black', label='Decentralization Agent', linewidth=2),
    mpatches.Patch(facecolor=NATURE_COLORS['coordinator'], alpha=0.3, 
                  edgecolor='black', label='Coordinator', linewidth=2),
    mpatches.Patch(facecolor=NATURE_COLORS['messagebus'], alpha=0.3, 
                  edgecolor='black', label='MessageBus', linewidth=2),
]

ax.legend(handles=legend_elements, loc='upper left', fontsize=8, 
         frameon=True, fancybox=False, edgecolor='black', ncol=2)

# Set axis
ax.set_xlim(0, 17)
ax.set_ylim(0, 11)
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/linxiangning/Desktop/000000ieeeprojects/fig/workflow_collaborative.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Generated architecture diagram: /Users/linxiangning/Desktop/000000ieeeprojects/fig/workflow_collaborative.png")
plt.close()

