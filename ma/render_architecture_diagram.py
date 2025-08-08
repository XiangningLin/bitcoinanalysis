#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a high-level architecture diagram of the multi-agent framework and bitcoinanalysis pipeline.
Output: outputs/architecture.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow


CURRENT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "architecture.png")


def add_box(ax, xy, width, height, label, fc="#F5F7FA", ec="#2E3A59", fontsize=10, lw=1.2):
    x, y = xy
    rect = Rectangle((x, y), width, height, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha="center", va="center", fontsize=fontsize)
    return rect


def add_arrow(ax, xy_from, xy_to, text=None):
    ax.add_patch(FancyArrow(xy_from[0], xy_from[1], xy_to[0]-xy_from[0], xy_to[1]-xy_from[1],
                            width=0.03, length_includes_head=True, head_width=0.15, head_length=0.25,
                            color="#4B6AFF", alpha=0.8))
    if text:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx, my + 0.15, text, ha="center", va="bottom", fontsize=9, color="#2E3A59")


def render():
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Top: Message Bus and Coordinator
    bus = add_box(ax, (1.0, 5.6), 3.2, 1.0, "MessageBus\n(request/response/broadcast)")
    coord = add_box(ax, (5.8, 5.6), 3.2, 1.0, "AgentCoordinator\n(scheduling, retries, perf stats)")
    add_arrow(ax, (4.2, 6.1), (5.8, 6.1), text="capability register / status")

    # Middle: Agents
    role = add_box(ax, (0.6, 3.6), 2.6, 1.0, "RoleClassifierAgent\n(role_classification)")
    anom = add_box(ax, (3.7, 3.6), 2.6, 1.0, "AnomalyAnalystAgent\n(anomaly_analysis)")
    decn = add_box(ax, (6.8, 3.6), 2.6, 1.0, "DecentralizationSummarizer\n(decentralization_summary)")

    # Bus <-> Agents
    for x in (1.9, 5.0, 8.1):
        add_arrow(ax, (2.6, 5.6), (x, 4.6))  # bus -> agents (broadcast/requests)
        add_arrow(ax, (x, 4.6), (2.6, 5.6))  # agents -> bus (responses)

    # Coordinator -> Agents (via Bus abstracted): draw dashed arrows directly for clarity
    add_arrow(ax, (7.4, 5.6), (1.9, 4.6), text="assign task: role_classification")
    add_arrow(ax, (7.4, 5.6), (5.0, 4.6), text="assign task: anomaly_analysis")
    add_arrow(ax, (7.4, 5.6), (8.1, 4.6), text="assign task: decentralization_summary")

    # Bottom: Data I/O
    data_in = add_box(ax, (0.6, 1.2), 4.5, 1.0, "Inputs\nTop-100 nodes (JSONL), induced edges (CSV)")
    data_out = add_box(ax, (5.3, 1.2), 4.1, 1.0, "Outputs\nroles.json, anomaly.txt, summary.txt\n+ compare figures/tables")

    # Data flow to agents and outputs back
    add_arrow(ax, (2.85, 2.2), (1.9, 3.6), text="nodes/edges context")
    add_arrow(ax, (2.85, 2.2), (5.0, 3.6))
    add_arrow(ax, (2.85, 2.2), (8.1, 3.6))

    add_arrow(ax, (1.9, 3.6), (7.35, 2.2), text="role predictions")
    add_arrow(ax, (5.0, 3.6), (7.35, 2.2), text="anomaly explanation")
    add_arrow(ax, (8.1, 3.6), (7.35, 2.2), text="decentralization summary")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved diagram to: {OUT_PATH}")


if __name__ == "__main__":
    render()



