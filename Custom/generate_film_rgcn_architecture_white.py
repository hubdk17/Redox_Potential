"""
Generate FiLM-RGCN End-to-End Architecture Diagram in Clean White Background
===========================================================================
Recreates the 4-card architectural pipeline with:
- Crisp pure white background (#FFFFFF)
- Publication-quality high-contrast cards, typography, and connecting arrows
- Exact same content and structure as the slide diagram
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram_white(output_path):
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=300, facecolor='#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    # Card helper
    def draw_card(x, y, w, h, title, subtitle, items, border_color, fill_color='#F8FAFC'):
        # Card Background & Border
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08,rounding_size=0.18',
                                     facecolor=fill_color, edgecolor=border_color, linewidth=2.2, zorder=2)
        ax.add_patch(box)

        # Title
        ax.text(x + w/2, y + h - 0.38, title, color='#0F172A', fontsize=11.5, fontweight='bold',
                ha='center', va='center', zorder=3)
        
        # Subtitle
        if subtitle:
            ax.text(x + w/2, y + h - 0.68, subtitle, color=border_color, fontsize=9.5, fontweight='bold',
                    ha='center', va='center', zorder=3)

        # Dividing thin line under header
        ax.plot([x + 0.18, x + w - 0.18], [y + h - 0.88, y + h - 0.88],
                color='#E2E8F0', lw=1.0, zorder=3)

        # Bullets
        y_text = y + h - 1.22
        for item in items:
            ax.text(x + 0.18, y_text, f'•  {item}', color='#334155', fontsize=8.8,
                    va='center', fontweight='medium', zorder=3)
            y_text -= 0.38

    # 1. Card 1: 3D Protein Graph
    draw_card(0.2, 0.4, 2.05, 3.6,
              '1. 3D Protein Graph', 'Raw Coordinates',
              ['371 Fe-S Proteins',
               'Nodes: Clusters & AA',
               'Atom-level centers',
               'Experimental pH (s)',
               'No manual tables!'],
              border_color='#0284C7', fill_color='#F0F9FF')

    # Arrow 1 -> 2
    ax.annotate('', xy=(2.52, 2.2), xytext=(2.28, 2.2),
                arrowprops=dict(arrowstyle='-|>', color='#0284C7', lw=2.8, mutation_scale=18),
                zorder=4)

    # 2. Card 2: 5-Relational RGCN
    draw_card(2.55, 0.4, 2.15, 3.6,
              '2. 5-Relational RGCN', 'Spatial & Covalent',
              ['r1: Peptide Backbone',
               'r2: Fe-S Coordinate',
               'r3: Polar H-Bonds',
               'r4: Hydrophobic Pak',
               'r5: Inter-Cluster'],
              border_color='#6366F1', fill_color='#EEF2FF')

    # Arrow 2 -> 3
    ax.annotate('', xy=(5.00, 2.2), xytext=(4.75, 2.2),
                arrowprops=dict(arrowstyle='-|>', color='#6366F1', lw=2.8, mutation_scale=18),
                zorder=4)

    # 3. Card 3: pH-FiLM Modulation
    draw_card(5.05, 0.4, 2.15, 3.6,
              '3. pH-FiLM Modulation', 'Dynamic Conditioning',
              ['Solvent pH scalar (s)',
               'MLP: s \u2192 (\u03b3, \u03b2)',
               'Affine modulation:',
               'h\u2032 = \u03b3 \u2299 h + \u03b2',
               'Layer-wise injection'],
              border_color='#D97706', fill_color='#FFFBEB')

    # Arrow 3 -> 4
    ax.annotate('', xy=(7.48, 2.2), xytext=(7.25, 2.2),
                arrowprops=dict(arrowstyle='-|>', color='#D97706', lw=2.8, mutation_scale=18),
                zorder=4)

    # 4. Card 4: Dual-Scale Readout
    draw_card(7.52, 0.4, 2.25, 3.6,
              '4. Dual-Scale Readout', 'Physics-Preserving',
              ['Distance weight: e^(-d/d0)',
               'Inner Shell: Quantum FeS',
               'Outer Shell: Dielectric',
               'Dense MLP Predictor',
               'Output: Em (mV)'],
              border_color='#059669', fill_color='#ECFDF5')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"White background architecture diagram successfully saved to:\n  {output_path}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "Custom", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "film_rgcn_architecture_white.png")
    create_architecture_diagram_white(out_path)

if __name__ == "__main__":
    main()
