"""
Clean White Background GNN Diagram: Input Graph & Message Passing
================================================================
Simple, intuitive educational diagram with:
- Crisp white background (#FFFFFF)
- Exactly 2 sections: (1) Input Graph, (2) Message Passing
- NO step numbering / no 'steps' text
- Clean publication-grade aesthetics
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_simple_white_diagram(output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2), dpi=300, facecolor='#FFFFFF')
    
    # -------------------------------------------------------------------------
    # Styling Palette (Clean White Textbook Style)
    # -------------------------------------------------------------------------
    c_bg = '#FFFFFF'
    c_card = '#F8FAFC'
    c_border = '#CBD5E1'
    c_text_dark = '#0F172A'
    c_text_muted = '#475569'
    c_node_center = '#059669'   # Rich emerald green for target
    c_node_neighbor = '#0284C7' # Vibrant blue for neighbors
    c_node_other = '#64748B'    # Slate for other nodes
    c_edge = '#94A3B8'
    c_arrow = '#D97706'         # Amber for incoming message arrows
    c_box_bg = '#FFFFFF'

    fig.patch.set_facecolor(c_bg)

    # Supertitle
    fig.text(0.5, 0.94, "Graph Neural Network: Input Graph & Message Passing",
             color=c_text_dark, fontsize=18, fontweight='bold', ha='center')
    fig.text(0.5, 0.90, "How local chemical neighborhoods share information to update structural representations",
             color=c_text_muted, fontsize=11, ha='center')

    # =========================================================================
    # LEFT PANEL: Input Graph
    # =========================================================================
    ax1.set_facecolor(c_bg)
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Card Container
    card1 = patches.FancyBboxPatch((0.03, 0.04), 0.94, 0.82,
                                  boxstyle='round,pad=0.02,rounding_size=0.03',
                                  facecolor=c_card, edgecolor=c_border, linewidth=1.5)
    ax1.add_patch(card1)

    # Panel Title (No "Step" text)
    ax1.text(0.50, 0.81, "Input Graph", color=c_text_dark, fontsize=14, fontweight='bold', ha='center')
    ax1.text(0.50, 0.77, "Nodes represent atoms / residues; edges represent bonds",
             color=c_text_muted, fontsize=9.5, ha='center')

    # Graph nodes layout
    nodes_p1 = {
        'A': (0.32, 0.61),
        'B': (0.68, 0.64),
        'C': (0.50, 0.46),  # Central target
        'D': (0.25, 0.30),
        'E': (0.75, 0.32),
        'F': (0.50, 0.18)
    }
    edges_p1 = [('A', 'C'), ('B', 'C'), ('D', 'C'), ('E', 'C'), ('A', 'B'), ('D', 'F'), ('E', 'F')]

    # Draw edges
    for u, v in edges_p1:
        ax1.plot([nodes_p1[u][0], nodes_p1[v][0]], [nodes_p1[u][1], nodes_p1[v][1]],
                 color=c_edge, lw=2.2, zorder=2)

    # Draw nodes
    for name, (nx, ny) in nodes_p1.items():
        is_target = (name == 'C')
        color = c_node_center if is_target else (c_node_neighbor if name in ['A', 'B', 'D', 'E'] else c_node_other)
        circle = patches.Circle((nx, ny), 0.062, facecolor=color, edgecolor='#FFFFFF', lw=2.0, zorder=4)
        ax1.add_patch(circle)
        ax1.text(nx, ny, name, color='#FFFFFF', fontsize=11, fontweight='bold',
                 ha='center', va='center', zorder=5)

    # Feature vector callout box at bottom
    feat_box = patches.FancyBboxPatch((0.10, 0.06), 0.80, 0.085,
                                      boxstyle='round,pad=0.01,rounding_size=0.015',
                                      facecolor=c_box_bg, edgecolor='#0284C7', lw=1.2, zorder=3)
    ax1.add_patch(feat_box)
    ax1.text(0.50, 0.115, "Node Features h\u1d65:", color='#0284C7', fontsize=9.5, fontweight='bold', ha='center')
    ax1.text(0.50, 0.08, "[ Electrostatic Charge, SASA, Atom Type, H-Bonds ]",
             color=c_text_muted, fontsize=8.5, ha='center')

    # =========================================================================
    # RIGHT PANEL: Message Passing
    # =========================================================================
    ax2.set_facecolor(c_bg)
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Card Container
    card2 = patches.FancyBboxPatch((0.03, 0.04), 0.94, 0.82,
                                  boxstyle='round,pad=0.02,rounding_size=0.03',
                                  facecolor=c_card, edgecolor=c_border, linewidth=1.5)
    ax2.add_patch(card2)

    # Panel Title (No "Step" text)
    ax2.text(0.50, 0.81, "Message Passing & Aggregation", color=c_text_dark, fontsize=14, fontweight='bold', ha='center')
    ax2.text(0.50, 0.77, "Target Node C collects and aggregates messages from neighbors",
             color=c_text_muted, fontsize=9.5, ha='center')

    # Zoomed-in layout for central node and immediate neighbors
    target_pos = (0.50, 0.50)
    neighbors_p2 = {
        'A': (0.24, 0.67),
        'B': (0.76, 0.67),
        'D': (0.22, 0.33),
        'E': (0.78, 0.33)
    }

    # Draw neighbors and incoming message arrows
    for n_name, (nx, ny) in neighbors_p2.items():
        # Neighbor circle
        n_circ = patches.Circle((nx, ny), 0.065, facecolor=c_node_neighbor, edgecolor='#FFFFFF', lw=2.0, zorder=4)
        ax2.add_patch(n_circ)
        ax2.text(nx, ny, n_name, color='#FFFFFF', fontsize=11, fontweight='bold',
                 ha='center', va='center', zorder=5)

        # Message Arrow toward Target Node C
        ax2.annotate('', xy=target_pos, xytext=(nx, ny),
                     arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=2.8,
                                     mutation_scale=20, shrinkA=22, shrinkB=25),
                     zorder=3)

    # Center Target Node C
    c_circ = patches.Circle(target_pos, 0.082, facecolor=c_node_center, edgecolor='#FFFFFF', lw=2.5, zorder=6)
    ax2.add_patch(c_circ)
    ax2.text(target_pos[0], target_pos[1] + 0.012, "Node C", color='#FFFFFF', fontsize=11, fontweight='bold',
             ha='center', va='center', zorder=7)
    ax2.text(target_pos[0], target_pos[1] - 0.022, "(Target)", color='#E2E8F0', fontsize=8.5,
             ha='center', va='center', zorder=7)

    # Mathematical explanation callout at bottom
    math_box = patches.FancyBboxPatch((0.08, 0.05), 0.84, 0.16,
                                      boxstyle='round,pad=0.01,rounding_size=0.015',
                                      facecolor=c_box_bg, edgecolor=c_arrow, lw=1.4, zorder=3)
    ax2.add_patch(math_box)
    
    ax2.text(0.50, 0.17, "1. Compute Message:  m_u\u2192v = W \u22c5 h_u",
             color=c_arrow, fontsize=9.5, fontweight='bold', ha='center', zorder=4)
    ax2.text(0.50, 0.125, "2. Aggregate Messages:  M_v = \u2211 m_u\u2192v   (or Average)",
             color=c_text_dark, fontsize=9.5, fontweight='bold', ha='center', zorder=4)
    ax2.text(0.50, 0.08, "3. Update Node:  h_v(new) = \u03c3( W_self \u22c5 h_v + M_v )",
             color=c_node_center, fontsize=9.5, fontweight='bold', ha='center', zorder=4)

    # Big transition arrow between Panel 1 and Panel 2
    trans_arrow = patches.FancyArrowPatch((0.485, 0.50), (0.515, 0.50),
                                         arrowstyle='->', mutation_scale=22,
                                         color='#64748B', lw=2.5, zorder=10,
                                         transform=fig.transFigure)
    fig.patches.append(trans_arrow)

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, bottom=0.05, wspace=0.10)
    plt.savefig(output_path, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close()
    print(f"Clean white diagram successfully saved to:\n  {output_path}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "Custom", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "gnn_input_and_message_passing_white.png")
    create_simple_white_diagram(out_path)

if __name__ == "__main__":
    main()
