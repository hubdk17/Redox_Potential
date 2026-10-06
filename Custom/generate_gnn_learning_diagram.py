"""
Educational Diagram: How Graph Neural Networks (GNNs) Work
===========================================================
This script creates a clear, step-by-step educational diagram explaining
the core mechanics of Graph Neural Networks (Nodes, Edges, Message Passing,
Aggregation, Node Update, Multi-Hop Layers, and Readout / Prediction).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_gnn_learning_diagram(output_path):
    fig = plt.figure(figsize=(18, 9.5), dpi=300, facecolor='#0F172A')
    
    # Theme colors
    c_bg = '#0F172A'          # Dark slate background
    c_card = '#1E293B'        # Slate container
    c_card_border = '#334155' # Subtle border
    c_node_target = '#10B981' # Emerald for target node
    c_node_neighbor = '#38BDF8' # Sky blue for neighbor nodes
    c_node_distant = '#64748B' # Muted slate for distant nodes
    c_edge = '#94A3B8'        # Edge line color
    c_msg_arrow = '#F59E0B'   # Amber for message arrows
    c_text_title = '#FFFFFF'  # Primary white text
    c_text_muted = '#94A3B8'  # Subtitle gray
    c_accent_purple = '#A855F7' # Readout / Pooling accent

    # -------------------------------------------------------------------------
    # Super Title Banner
    # -------------------------------------------------------------------------
    fig.text(0.04, 0.95, "HOW GRAPH NEURAL NETWORKS (GNNs) WORK | INTUITIVE LEARNING GUIDE",
             color='#38BDF8', fontsize=13, fontweight='bold')
    fig.text(0.04, 0.915, "The 4 Core Steps: Graph Representation \u2192 Message Passing \u2192 Node Update \u2192 Readout",
             color=c_text_title, fontsize=19, fontweight='bold')
    fig.text(0.04, 0.885, "Unlike standard neural networks that process flat vectors, GNNs learn directly on relational networks & 3D molecules",
             color=c_text_muted, fontsize=11)

    # 4 Columns / Panels Layout
    # Panel 1: Input Graph (x: 0.04 - 0.26)
    # Panel 2: Message Passing & Aggregation (x: 0.28 - 0.52)
    # Panel 3: Node Update & Multi-Hop (x: 0.54 - 0.76)
    # Panel 4: Graph Readout & Output (x: 0.78 - 0.96)
    
    def draw_card(ax, title, step_num, step_color):
        card = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                     boxstyle='round,pad=0.02,rounding_size=0.03',
                                     facecolor=c_card, edgecolor=c_card_border, linewidth=1.5,
                                     transform=ax.transAxes, zorder=1)
        ax.add_patch(card)
        
        # Step Badge
        badge = patches.FancyBboxPatch((0.05, 0.91), 0.22, 0.065,
                                       boxstyle='round,pad=0.01,rounding_size=0.02',
                                       facecolor=step_color, edgecolor='none',
                                       transform=ax.transAxes, zorder=2)
        ax.add_patch(badge)
        ax.text(0.16, 0.942, f"STEP {step_num}", color='#FFFFFF', fontsize=9, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes, zorder=3)
        
        # Title
        ax.text(0.30, 0.942, title, color=c_text_title, fontsize=12, fontweight='bold',
                va='center', transform=ax.transAxes, zorder=3)

    # =========================================================================
    # PANEL 1: Input Graph (Nodes & Edges)
    # =========================================================================
    ax1 = fig.add_axes([0.03, 0.06, 0.225, 0.80])
    ax1.set_facecolor(c_bg)
    ax1.axis('off')
    draw_card(ax1, "Input Graph", 1, '#38BDF8')

    ax1.text(0.06, 0.86, "Entities as Nodes & Relations as Edges", color=c_text_muted, fontsize=9.5, transform=ax1.transAxes)

    # Node coordinates in panel 1
    p1_nodes = {
        'A': (0.30, 0.68),
        'B': (0.70, 0.72),
        'C': (0.50, 0.48), # Center
        'D': (0.22, 0.30),
        'E': (0.75, 0.32),
        'F': (0.48, 0.16)
    }
    p1_edges = [('A', 'C'), ('B', 'C'), ('D', 'C'), ('E', 'C'), ('A', 'B'), ('D', 'F'), ('E', 'F')]

    # Draw edges
    for u, v in p1_edges:
        x_pts = [p1_nodes[u][0], p1_nodes[v][0]]
        y_pts = [p1_nodes[u][1], p1_nodes[v][1]]
        ax1.plot(x_pts, y_pts, color=c_edge, lw=2.0, ls='-', zorder=2, transform=ax1.transAxes)

    # Draw nodes
    for name, (nx, ny) in p1_nodes.items():
        is_center = (name == 'C')
        circle = patches.Circle((nx, ny), 0.065,
                                facecolor=c_node_target if is_center else c_node_neighbor,
                                edgecolor='#FFFFFF', lw=1.5,
                                transform=ax1.transAxes, zorder=4)
        ax1.add_patch(circle)
        ax1.text(nx, ny, name, color='#FFFFFF', fontsize=11, fontweight='bold',
                 ha='center', va='center', transform=ax1.transAxes, zorder=5)

    # Callout Feature Vector Box
    f_box = patches.FancyBboxPatch((0.06, 0.05), 0.88, 0.14,
                                   boxstyle='round,pad=0.01,rounding_size=0.02',
                                   facecolor='#0F172A', edgecolor='#38BDF8', lw=1.2,
                                   transform=ax1.transAxes, zorder=3)
    ax1.add_patch(f_box)
    ax1.text(0.50, 0.145, "Initial Node Features h_i(0):", color='#38BDF8', fontsize=9, fontweight='bold',
             ha='center', transform=ax1.transAxes, zorder=4)
    ax1.text(0.50, 0.085, "[ Charge, SASA, Atom Type, H-Bonds ]", color='#E2E8F0', fontsize=8.5,
             ha='center', transform=ax1.transAxes, zorder=4)

    # =========================================================================
    # PANEL 2: Message Passing & Aggregation
    # =========================================================================
    ax2 = fig.add_axes([0.27, 0.06, 0.23, 0.80])
    ax2.set_facecolor(c_bg)
    ax2.axis('off')
    draw_card(ax2, "Message Passing", 2, c_msg_arrow)

    ax2.text(0.06, 0.86, "Neighbors send information to Target Node", color=c_text_muted, fontsize=9.5, transform=ax2.transAxes)

    # Zoomed-in Target Node & Neighbors
    center_pos = (0.50, 0.50)
    neighbors = {
        'A': (0.22, 0.72),
        'B': (0.78, 0.72),
        'D': (0.20, 0.28),
        'E': (0.80, 0.28)
    }

    # Draw message passing arrows toward center
    for n_name, (nx, ny) in neighbors.items():
        # Circle for neighbor
        n_circ = patches.Circle((nx, ny), 0.065, facecolor=c_node_neighbor, edgecolor='#FFFFFF', lw=1.5,
                                transform=ax2.transAxes, zorder=3)
        ax2.add_patch(n_circ)
        ax2.text(nx, ny, n_name, color='#FFFFFF', fontsize=11, fontweight='bold',
                 ha='center', va='center', transform=ax2.transAxes, zorder=4)
        
        # Message Arrow
        ax2.annotate('', xy=center_pos, xytext=(nx, ny),
                     arrowprops=dict(arrowstyle="-|>", color=c_msg_arrow, lw=2.5,
                                     mutation_scale=18, shrinkA=18, shrinkB=20),
                     transform=ax2.transAxes, zorder=2)

    # Center Target Node C
    c_circ = patches.Circle(center_pos, 0.08, facecolor=c_node_target, edgecolor='#FFFFFF', lw=2.0,
                            transform=ax2.transAxes, zorder=5)
    ax2.add_patch(c_circ)
    ax2.text(center_pos[0], center_pos[1], "Node C\n(Target)", color='#FFFFFF', fontsize=9, fontweight='bold',
             ha='center', va='center', transform=ax2.transAxes, zorder=6)

    # Aggregation Symbol Box
    agg_box = patches.FancyBboxPatch((0.06, 0.05), 0.88, 0.16,
                                     boxstyle='round,pad=0.01,rounding_size=0.02',
                                     facecolor='#0F172A', edgecolor=c_msg_arrow, lw=1.2,
                                     transform=ax2.transAxes, zorder=3)
    ax2.add_patch(agg_box)
    ax2.text(0.50, 0.165, "1. Message: m_u = W \u22c5 h_u", color=c_msg_arrow, fontsize=9.5, fontweight='bold',
             ha='center', transform=ax2.transAxes, zorder=4)
    ax2.text(0.50, 0.115, "2. Aggregate: M_C = \u2211 m_u  or  MEAN(m_u)", color='#E2E8F0', fontsize=9, fontweight='bold',
             ha='center', transform=ax2.transAxes, zorder=4)
    ax2.text(0.50, 0.070, "Gathers context from immediate 1-hop neighbors", color=c_text_muted, fontsize=8,
             ha='center', transform=ax2.transAxes, zorder=4)

    # =========================================================================
    # PANEL 3: Node State Update & Multi-Hop Flow
    # =========================================================================
    ax3 = fig.add_axes([0.515, 0.06, 0.23, 0.80])
    ax3.set_facecolor(c_bg)
    ax3.axis('off')
    draw_card(ax3, "Node Update & Layers", 3, c_node_target)

    ax3.text(0.06, 0.86, "Combine Self + Messages & Stack Layers", color=c_text_muted, fontsize=9.5, transform=ax3.transAxes)

    # Visualizing Self Update
    # Old state + Aggregated Messages -> New State
    # Box for Update formula
    up_box = patches.FancyBboxPatch((0.06, 0.60), 0.88, 0.23,
                                    boxstyle='round,pad=0.01,rounding_size=0.02',
                                    facecolor='#0F172A', edgecolor=c_node_target, lw=1.5,
                                    transform=ax3.transAxes, zorder=3)
    ax3.add_patch(up_box)
    ax3.text(0.50, 0.78, "Node Update Equation:", color=c_node_target, fontsize=10, fontweight='bold',
             ha='center', transform=ax3.transAxes, zorder=4)
    ax3.text(0.50, 0.72, "h_v(l+1) = \u03c3( W_self \u22c5 h_v(l) + M_v )", color='#FFFFFF', fontsize=11, fontweight='bold',
             ha='center', transform=ax3.transAxes, zorder=4)
    ax3.text(0.50, 0.65, "(\u03c3 is non-linear activation: ReLU / LeakyReLU)", color=c_text_muted, fontsize=8.2,
             ha='center', transform=ax3.transAxes, zorder=4)

    # Multi-hop receptive field concept
    ax3.text(0.50, 0.52, "\u25bc STACKING GNN LAYERS (MULTI-HOP) \u25bc", color='#38BDF8', fontsize=9.5, fontweight='bold',
             ha='center', transform=ax3.transAxes)

    hops_data = [
        ("Layer 1 (1-Hop):", "Each node knows its direct neighbors."),
        ("Layer 2 (2-Hop):", "Each node knows neighbors of neighbors!"),
        ("Layer 3 (3-Hop):", "Information spans whole active site.")
    ]
    y_h = 0.44
    for title_h, desc_h in hops_data:
        h_box = patches.FancyBboxPatch((0.06, y_h - 0.07), 0.88, 0.08,
                                       boxstyle='round,pad=0.005,rounding_size=0.015',
                                       facecolor='#1E293B', edgecolor='#475569', lw=1.0,
                                       transform=ax3.transAxes, zorder=3)
        ax3.add_patch(h_box)
        ax3.text(0.10, y_h - 0.018, title_h, color='#38BDF8', fontsize=8.8, fontweight='bold',
                 transform=ax3.transAxes, zorder=4)
        ax3.text(0.10, y_h - 0.052, desc_h, color='#E2E8F0', fontsize=8.2,
                 transform=ax3.transAxes, zorder=4)
        y_h -= 0.11

    # Bottom summary
    bot_box = patches.FancyBboxPatch((0.06, 0.05), 0.88, 0.10,
                                     boxstyle='round,pad=0.01,rounding_size=0.02',
                                     facecolor='#0F172A', edgecolor='#334155', lw=1.0,
                                     transform=ax3.transAxes, zorder=3)
    ax3.add_patch(bot_box)
    ax3.text(0.50, 0.10, "Nodes now hold rich 3D structural context!", color='#10B981', fontsize=9, fontweight='bold',
             ha='center', transform=ax3.transAxes, zorder=4)

    # =========================================================================
    # PANEL 4: Readout & Graph-Level Prediction
    # =========================================================================
    ax4 = fig.add_axes([0.76, 0.06, 0.22, 0.80])
    ax4.set_facecolor(c_bg)
    ax4.axis('off')
    draw_card(ax4, "Readout & Prediction", 4, c_accent_purple)

    ax4.text(0.06, 0.86, "Pooling Nodes to Predict Graph Target", color=c_text_muted, fontsize=9.5, transform=ax4.transAxes)

    # Nodes gathered together
    node_positions = [(0.25, 0.72), (0.45, 0.75), (0.65, 0.72), (0.82, 0.75)]
    for i, (px, py) in enumerate(node_positions):
        c = patches.Circle((px, py), 0.055, facecolor='#10B981' if i==1 else '#38BDF8',
                           edgecolor='#FFFFFF', lw=1.2, transform=ax4.transAxes, zorder=3)
        ax4.add_patch(c)
        ax4.text(px, py, f"h_{i}", color='#FFFFFF', fontsize=8.5, fontweight='bold',
                 ha='center', va='center', transform=ax4.transAxes, zorder=4)

    # Pooling Funnel Arrow
    ax4.annotate('', xy=(0.50, 0.52), xytext=(0.50, 0.65),
                 arrowprops=dict(arrowstyle="-|>", color=c_accent_purple, lw=3.0,
                                 mutation_scale=20),
                 transform=ax4.transAxes, zorder=3)

    # Readout / Pooling Box
    r_box = patches.FancyBboxPatch((0.10, 0.40), 0.80, 0.11,
                                   boxstyle='round,pad=0.01,rounding_size=0.02',
                                   facecolor='#0F172A', edgecolor=c_accent_purple, lw=1.5,
                                   transform=ax4.transAxes, zorder=3)
    ax4.add_patch(r_box)
    ax4.text(0.50, 0.47, "Graph Readout / Pooling", color=c_accent_purple, fontsize=9.5, fontweight='bold',
             ha='center', transform=ax4.transAxes, zorder=4)
    ax4.text(0.50, 0.42, "h_graph = \u2211 w_i \u22c5 h_i(L)", color='#FFFFFF', fontsize=9.5, fontweight='bold',
             ha='center', transform=ax4.transAxes, zorder=4)

    # Downward Arrow to MLP
    ax4.annotate('', xy=(0.50, 0.28), xytext=(0.50, 0.39),
                 arrowprops=dict(arrowstyle="-|>", color='#38BDF8', lw=2.5,
                                 mutation_scale=18),
                 transform=ax4.transAxes, zorder=3)

    # MLP Predictor Box
    mlp_box = patches.FancyBboxPatch((0.15, 0.21), 0.70, 0.07,
                                     boxstyle='round,pad=0.01,rounding_size=0.02',
                                     facecolor='#1E293B', edgecolor='#38BDF8', lw=1.2,
                                     transform=ax4.transAxes, zorder=3)
    ax4.add_patch(mlp_box)
    ax4.text(0.50, 0.245, "Multi-Layer Perceptron (MLP)", color='#38BDF8', fontsize=8.5, fontweight='bold',
             ha='center', va='center', transform=ax4.transAxes, zorder=4)

    # Downward Arrow to Final Target
    ax4.annotate('', xy=(0.50, 0.12), xytext=(0.50, 0.20),
                 arrowprops=dict(arrowstyle="-|>", color='#10B981', lw=2.5,
                                 mutation_scale=18),
                 transform=ax4.transAxes, zorder=3)

    # Final Output Badge
    out_badge = patches.FancyBboxPatch((0.08, 0.04), 0.84, 0.075,
                                       boxstyle='round,pad=0.01,rounding_size=0.02',
                                       facecolor='#059669', edgecolor='#34D399', lw=1.5,
                                       transform=ax4.transAxes, zorder=3)
    ax4.add_patch(out_badge)
    ax4.text(0.50, 0.078, "Predicted Redox Potential (Em): -420 mV", color='#FFFFFF', fontsize=9, fontweight='bold',
             ha='center', va='center', transform=ax4.transAxes, zorder=4)

    # -------------------------------------------------------------------------
    # Connecting Arrows Between Panels (Step 1 -> 2 -> 3 -> 4)
    # -------------------------------------------------------------------------
    fig.patches.extend([
        patches.FancyArrowPatch((0.257, 0.46), (0.268, 0.46), arrowstyle='->', mutation_scale=20,
                                color='#38BDF8', lw=2.5, zorder=10),
        patches.FancyArrowPatch((0.502, 0.46), (0.513, 0.46), arrowstyle='->', mutation_scale=20,
                                color=c_msg_arrow, lw=2.5, zorder=10),
        patches.FancyArrowPatch((0.747, 0.46), (0.758, 0.46), arrowstyle='->', mutation_scale=20,
                                color=c_node_target, lw=2.5, zorder=10),
    ])

    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"GNN educational diagram successfully saved to:\n  {output_path}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "Custom", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "gnn_learning_diagram.png")
    create_gnn_learning_diagram(out_path)

if __name__ == "__main__":
    main()
