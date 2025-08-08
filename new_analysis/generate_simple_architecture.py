"""
Generate a simple, clean architecture diagram focused on core components
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 11))

# Colors - simple and clear
COLOR_COORDINATOR = '#FFB6C1'
COLOR_MESSAGEBUS = '#FF6B6B'
COLOR_AGENT = '#87CEEB'
COLOR_ARROW = '#333333'
COLOR_USER = '#90EE90'

# ================ Layer 0: User/Researcher (Top) ================
user_y = 9.5
user_box = FancyBboxPatch(
    (3, user_y - 0.5),
    8, 1,
    boxstyle="round,pad=0.12",
    edgecolor='black',
    facecolor=COLOR_USER,
    linewidth=3
)
ax.add_patch(user_box)
ax.text(7, user_y + 0.2, 'Researcher / Analysis Request',
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(7, user_y - 0.15, 'Submit: Compare RWFB vs CETraS (1000 nodes each)',
        ha='center', va='center', fontsize=9, style='italic')

# ================ Layer 1: Coordinator + CollaborationManager ================
coord_y = 7.5

# Coordinator (left)
coord_box = FancyBboxPatch(
    (3, coord_y - 0.6),
    3.5, 1.3,
    boxstyle="round,pad=0.12",
    edgecolor='black',
    facecolor=COLOR_COORDINATOR,
    linewidth=3
)
ax.add_patch(coord_box)
ax.text(4.75, coord_y + 0.3, 'Coordinator',
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(4.75, coord_y - 0.15, 'Task Scheduling',
        ha='center', va='center', fontsize=9, style='italic')

# CollaborationManager (right)
collab_box = FancyBboxPatch(
    (7.5, coord_y - 0.6),
    3.5, 1.3,
    boxstyle="round,pad=0.12",
    edgecolor='black',
    facecolor='#FFD700',
    linewidth=3
)
ax.add_patch(collab_box)
ax.text(9.25, coord_y + 0.3, 'Collaboration\nManager',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(9.25, coord_y - 0.15, 'Stage Control',
        ha='center', va='center', fontsize=9, style='italic')

# Arrow from User to Coordinator
arrow_user_coord = FancyArrowPatch(
    (7, user_y - 0.5),
    (4.75, coord_y + 0.65),
    arrowstyle='->,head_width=0.6,head_length=0.7',
    linewidth=3,
    color='#00AA00',
    alpha=0.8
)
ax.add_patch(arrow_user_coord)
ax.text(5.5, 8.7, 'Submit\nTask',
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                 edgecolor='#00AA00', linewidth=2))

# Arrow between Coordinator and CollaborationManager
arrow_coord_collab = FancyArrowPatch(
    (6.5, coord_y),
    (7.5, coord_y),
    arrowstyle='<->,head_width=0.5,head_length=0.6',
    linewidth=2.5,
    color='#FF6B35',
    linestyle='--'
)
ax.add_patch(arrow_coord_collab)
ax.text(7, coord_y + 0.5, 'Stage\nTransition',
        ha='center', va='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#FF6B35'))

# Arrow from Coordinator back to User (results)
arrow_coord_user = FancyArrowPatch(
    (6.5, coord_y + 0.65),
    (8.5, user_y - 0.5),
    arrowstyle='->,head_width=0.6,head_length=0.7',
    linewidth=3,
    color='#0066CC',
    alpha=0.8
)
ax.add_patch(arrow_coord_user)
ax.text(8.5, 8.7, 'Return\nResults',
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                 edgecolor='#0066CC', linewidth=2))

# ================ Layer 2: MessageBus (Middle) ================
bus_y = 5
bus_box = FancyBboxPatch(
    (3.5, bus_y - 0.6),
    7, 1.3,
    boxstyle="round,pad=0.12",
    edgecolor='black',
    facecolor=COLOR_MESSAGEBUS,
    linewidth=3
)
ax.add_patch(bus_box)
ax.text(7, bus_y + 0.3, 'MessageBus',
        ha='center', va='center', fontsize=13, fontweight='bold', color='white')
ax.text(7, bus_y - 0.15, 'Information Sharing & Routing',
        ha='center', va='center', fontsize=10, color='white', style='italic')

# ================ Layer 3: Three Agents (Bottom) ================
agent_y = 2
agent_spacing = 4

agents = [
    {'name': 'Role Classifier\nAgent', 'x': 2.5, 'label': 'A'},
    {'name': 'Anomaly Analyst\nAgent', 'x': 7, 'label': 'B'},
    {'name': 'Decentralization\nAgent', 'x': 11.5, 'label': 'C'}
]

for agent in agents:
    # Agent box
    agent_box = FancyBboxPatch(
        (agent['x'] - 1.5, agent_y - 0.8),
        3, 1.7,
        boxstyle="round,pad=0.12",
        edgecolor='black',
        facecolor=COLOR_AGENT,
        linewidth=3
    )
    ax.add_patch(agent_box)
    
    # Agent name
    ax.text(agent['x'], agent_y + 0.4, agent['name'],
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Agent label
    ax.text(agent['x'], agent_y - 0.35, f'Agent {agent["label"]}',
            ha='center', va='center', fontsize=9, style='italic')

# ================ Arrows: Coordinator <-> MessageBus ================
# Coordinator -> MessageBus (send tasks)
arrow_coord_bus_down = FancyArrowPatch(
    (4.75, coord_y - 0.6),
    (5.5, bus_y + 0.65),
    arrowstyle='->,head_width=0.6,head_length=0.7',
    linewidth=3,
    color=COLOR_ARROW,
    alpha=0.8
)
ax.add_patch(arrow_coord_bus_down)
ax.text(4.5, 6.7, 'Send Tasks',
        ha='center', va='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black'))

# MessageBus -> Coordinator (return results)
arrow_coord_bus_up = FancyArrowPatch(
    (6, bus_y + 0.65),
    (5, coord_y - 0.6),
    arrowstyle='->,head_width=0.6,head_length=0.7',
    linewidth=3,
    color=COLOR_ARROW,
    alpha=0.8
)
ax.add_patch(arrow_coord_bus_up)
ax.text(6.5, 6.7, 'Collect Results',
        ha='center', va='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black'))

# ================ Arrows: MessageBus <-> Agents ================
for agent in agents:
    # MessageBus -> Agent
    arrow_bus_agent = FancyArrowPatch(
        (agent['x'], bus_y - 0.6),
        (agent['x'], agent_y + 0.9),
        arrowstyle='->,head_width=0.6,head_length=0.7',
        linewidth=3,
        color=COLOR_ARROW,
        alpha=0.8
    )
    ax.add_patch(arrow_bus_agent)
    
    # Agent -> MessageBus
    offset = 0.4
    arrow_agent_bus = FancyArrowPatch(
        (agent['x'] + offset, agent_y + 0.9),
        (agent['x'] + offset, bus_y - 0.6),
        arrowstyle='->,head_width=0.6,head_length=0.7',
        linewidth=3,
        color=COLOR_ARROW,
        alpha=0.8
    )
    ax.add_patch(arrow_agent_bus)

# Labels for agent communication
ax.text(2.5, 3.7, 'Receive\nTask', ha='center', fontsize=8, fontweight='bold')
ax.text(3.2, 3.7, 'Share\nResults', ha='center', fontsize=8, fontweight='bold')

ax.text(7, 3.7, 'Get\nContext', ha='center', fontsize=8, fontweight='bold')
ax.text(7.7, 3.7, 'Broadcast\nInfo', ha='center', fontsize=8, fontweight='bold')

ax.text(11.5, 3.7, 'Receive\nInputs', ha='center', fontsize=8, fontweight='bold')
ax.text(12.2, 3.7, 'Send\nMetrics', ha='center', fontsize=8, fontweight='bold')

# ================ Collaboration Details (Bottom Box) ================
collab_detail_y = 0.2
collab_box = FancyBboxPatch(
    (1, collab_detail_y - 0.3),
    12, 1.3,
    boxstyle="round,pad=0.1",
    edgecolor='#FF6B35',
    facecolor='#FFF8DC',
    linewidth=2.5,
    linestyle='--',
    alpha=0.8
)
ax.add_patch(collab_box)

ax.text(7, collab_detail_y + 0.65, 'How Agents Collaborate (via MessageBus):',
        ha='center', fontsize=10, fontweight='bold', color='#FF6B35')

# Collaboration steps
collab_steps = [
    '① Agent A shares role distributions',
    '② Agent B uses roles for context-aware anomaly detection',
    '③ Agent C synthesizes both for centralization score',
    '④ Agents adjust low-confidence predictions based on feedback',
    '⑤ Weighted voting resolves conflicts'
]

step_x_start = 1.5
for i, step in enumerate(collab_steps):
    if i < 3:
        x = step_x_start + i * 3.8
        y = collab_detail_y + 0.15
    else:
        x = step_x_start + (i - 3) * 5.5
        y = collab_detail_y - 0.3
    
    ax.text(x, y, step, ha='left', fontsize=7.5, 
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                     edgecolor='#FF6B35', linewidth=1))

# ================ Peer-to-Peer Communication (Curved Arrows) ================
from matplotlib.patches import ConnectionPatch

# A -> B (Agent A shares to Agent B)
arrow_ab = FancyArrowPatch(
    (4, agent_y + 0.2),
    (5.5, agent_y + 0.2),
    arrowstyle='->,head_width=0.5,head_length=0.6',
    linewidth=2.5,
    color='#4169E1',
    linestyle=':',
    alpha=0.7
)
ax.add_patch(arrow_ab)
ax.text(4.75, agent_y + 0.6, '①', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#4169E1'))

# B -> C (Agent B shares to Agent C)
arrow_bc = FancyArrowPatch(
    (8.5, agent_y + 0.2),
    (10, agent_y + 0.2),
    arrowstyle='->,head_width=0.5,head_length=0.6',
    linewidth=2.5,
    color='#FF6B35',
    linestyle=':',
    alpha=0.7
)
ax.add_patch(arrow_bc)
ax.text(9.25, agent_y + 0.6, '②', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#FF6B35'))

# Feedback arrows (dashed, opposite direction)
arrow_ba = FancyArrowPatch(
    (5.5, agent_y - 0.2),
    (4, agent_y - 0.2),
    arrowstyle='->,head_width=0.4,head_length=0.5',
    linewidth=2,
    color='#666666',
    linestyle='--',
    alpha=0.6
)
ax.add_patch(arrow_ba)

arrow_cb = FancyArrowPatch(
    (10, agent_y - 0.2),
    (8.5, agent_y - 0.2),
    arrowstyle='->,head_width=0.4,head_length=0.5',
    linewidth=2,
    color='#666666',
    linestyle='--',
    alpha=0.6
)
ax.add_patch(arrow_cb)

ax.text(4.75, agent_y - 0.6, 'Feedback', fontsize=7, ha='center',
        style='italic', color='#666666')
ax.text(9.25, agent_y - 0.6, 'Feedback', fontsize=7, ha='center',
        style='italic', color='#666666')

# ================ Stage Indicators (Right Side) ================
stage_x = 12
stage_y_start = 7.5
stages = ['Stage 1: Initial Analysis',
          'Stage 2: Info Sharing',
          'Stage 3: Validation',
          'Stage 4: Consensus',
          'Stage 5: Integration']

ax.text(stage_x + 1.2, stage_y_start + 0.5, 'Workflow:',
        ha='center', fontsize=10, fontweight='bold')

for i, stage in enumerate(stages):
    y = stage_y_start - i * 0.65
    stage_num, stage_name = stage.split(': ')
    ax.text(stage_x + 1.2, y, f'{stage_num}:', 
            ha='right', fontsize=9, fontweight='bold')
    ax.text(stage_x + 1.3, y, stage_name,
            ha='left', fontsize=9)
    
    # Arrow between stages
    if i < len(stages) - 1:
        arrow_stage = FancyArrowPatch(
            (stage_x + 1.2, y - 0.25),
            (stage_x + 1.2, y - 0.4),
            arrowstyle='->,head_width=0.25,head_length=0.3',
            linewidth=1.5,
            color='#666666'
        )
        ax.add_patch(arrow_stage)

# ================ Key Results (Bottom Right) ================
results_text = (
    'Results:\n'
    'Disagreement: 18.8%→9.2% (-51%)\n'
    'SDM: 0.397→0.257 (-35%)\n'
    'Adjustments: 195/2000 (9.75%)'
)
ax.text(12.7, 1.5, results_text,
        ha='center', va='center', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFACD',
                 edgecolor='black', linewidth=2))

# ================ Title ================
ax.text(7, 10.8, 'Collaborative Multi-Agent Architecture',
        ha='center', fontsize=15, fontweight='bold')

# ================ Clean up ================
ax.set_xlim(0, 15)
ax.set_ylim(0.5, 10.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/linxiangning/Desktop/000000ieeeprojects/fig/workflow_collaborative.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Generated simplified architecture diagram")
plt.close()

