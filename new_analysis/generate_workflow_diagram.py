"""
Generate a clean workflow diagram with boxes and arrows for the 5-stage collaboration
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Nature journal style colors
NATURE_COLORS = {
    'primary': '#0C5DA5',
    'secondary': '#FF6B35',
    'accent1': '#00B945',
    'accent2': '#845B97',
    'accent3': '#FFA500',
    'neutral': '#474747',
}

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Define box positions (x, y, width, height)
box_width = 2.5
box_height = 1.2
spacing_y = 1.8

# Vertical positions for 5 stages
y_positions = [8, 6.2, 4.4, 2.6, 0.8]
x_center = 5

stages = [
    {
        'title': 'Stage 1: Initial Analysis',
        'content': 'Independent analysis by each agent\n• RoleClassifier: Classify nodes\n• Generate confidence scores\n• No information sharing',
        'color': NATURE_COLORS['primary']
    },
    {
        'title': 'Stage 2: Information Sharing',
        'content': 'Agents broadcast intermediate results\n• Share role distributions\n• Share anomaly patterns\n• Share network metrics',
        'color': NATURE_COLORS['secondary']
    },
    {
        'title': 'Stage 3: Collaborative Validation',
        'content': 'Cross-agent validation & feedback\n• Receive insights from other agents\n• Adjust low-confidence predictions (<0.75)\n• Context-aware risk assessment',
        'color': NATURE_COLORS['accent1']
    },
    {
        'title': 'Stage 4: Consensus Building',
        'content': 'Resolve conflicts through voting\n• Weighted voting (confidence × accuracy)\n• Mutual adjustment of classifications\n• Build consensus for disagreements',
        'color': NATURE_COLORS['accent2']
    },
    {
        'title': 'Stage 5: Result Integration',
        'content': 'Aggregate final results\n• Compute collaboration metrics\n• Calculate adjustment count (195/2000)\n• Report efficiency (85%)',
        'color': NATURE_COLORS['accent3']
    }
]

# Draw boxes and text
boxes = []
for i, (y_pos, stage) in enumerate(zip(y_positions, stages)):
    # Draw box
    box = FancyBboxPatch(
        (x_center - box_width/2, y_pos - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=stage['color'],
        alpha=0.3,
        linewidth=2.5
    )
    ax.add_patch(box)
    boxes.append((x_center, y_pos))
    
    # Title
    ax.text(x_center, y_pos + 0.4, stage['title'],
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='black')
    
    # Content (multi-line)
    ax.text(x_center, y_pos - 0.15, stage['content'],
            ha='center', va='center', fontsize=9,
            color=NATURE_COLORS['neutral'],
            multialignment='left')

# Draw arrows between stages
arrow_props = dict(
    arrowstyle='->,head_width=0.6,head_length=0.8',
    linewidth=3,
    color=NATURE_COLORS['neutral'],
    alpha=0.7
)

for i in range(len(boxes) - 1):
    x_start, y_start = boxes[i]
    x_end, y_end = boxes[i + 1]
    
    # Vertical arrow
    arrow = FancyArrowPatch(
        (x_start, y_start - box_height/2 - 0.05),
        (x_end, y_end + box_height/2 + 0.05),
        **arrow_props
    )
    ax.add_patch(arrow)
    
    # Arrow label
    if i == 0:
        label = "Share results"
    elif i == 1:
        label = "Apply feedback"
    elif i == 2:
        label = "Vote & adjust"
    else:
        label = "Finalize"
    
    mid_y = (y_start + y_end) / 2
    ax.text(x_center + 1.5, mid_y, label,
            fontsize=10, style='italic', color=NATURE_COLORS['neutral'],
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=NATURE_COLORS['neutral'], alpha=0.8))

# Add side annotations for key metrics
# Left side: Input/Output info
ax.text(1.5, 8, 'Input:\n1000 nodes\nper method',
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4F8', 
                 edgecolor=NATURE_COLORS['primary'], linewidth=2))

ax.text(8.5, 8, 'Baseline:\nSDM = 0.397\nDisagree = 18.8%',
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE8E8', 
                 edgecolor=NATURE_COLORS['secondary'], linewidth=2))

# Right side: Collaboration effects
ax.text(8.5, 4.4, 'Collaboration:\n9 exchanges\n3 agents × 3 iter',
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F8E8', 
                 edgecolor=NATURE_COLORS['accent1'], linewidth=2))

ax.text(8.5, 0.8, 'Output:\nSDM = 0.257 (↓35%)\nDisagree = 9.2% (↓51%)',
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF4E8', 
                 edgecolor=NATURE_COLORS['accent3'], linewidth=2))

# Add bias decomposition box at bottom
decomp_text = 'Bias Decomposition: 0.397 = 0.257 (Sampling 65%) + 0.140 (Analytical 35%)'
ax.text(x_center, -0.5, decomp_text,
        fontsize=11, ha='center', va='center', fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=NATURE_COLORS['neutral'], 
                 edgecolor='black', linewidth=2.5))

# Add agent icons on the left
agents = ['Role\nClassifier', 'Anomaly\nAnalyst', 'Decentral.\nSummarizer']
agent_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], NATURE_COLORS['accent1']]

for i, (agent, color) in enumerate(zip(agents, agent_colors)):
    y_agent = 8 - i * 1.2
    # Small circle for agent
    circle = plt.Circle((1.5, y_agent - 2.5), 0.25, color=color, alpha=0.6, 
                       edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(1.5, y_agent - 2.5, agent, fontsize=7, ha='center', va='center',
            fontweight='bold', color='white')

ax.text(1.5, 3.5, 'Participating\nAgents', fontsize=9, ha='center', va='top',
        fontweight='bold', color=NATURE_COLORS['neutral'])

# Title
ax.text(x_center, 9.5, 'Five-Stage Collaborative Multi-Agent Workflow',
        fontsize=16, ha='center', va='center', fontweight='bold',
        color=NATURE_COLORS['neutral'])

ax.text(x_center, 9.1, 'Result Reconciliation for Bias Decomposition',
        fontsize=11, ha='center', va='center', style='italic',
        color=NATURE_COLORS['neutral'])

# Set axis properties
ax.set_xlim(0, 10)
ax.set_ylim(-1.2, 10)
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/linxiangning/Desktop/000000ieeeprojects/fig/workflow_collaborative.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Generated new workflow diagram: /Users/linxiangning/Desktop/000000ieeeprojects/fig/workflow_collaborative.png")
plt.close()

