"""
Generate High-Resolution Photo of Table & Graph for Top 5 Models
================================================================
Top 5 Models:
1. FiLM-RGCN (Best) 🏆   (Explicitly labeled 'Best', without 'Ours')
2. XGBoost Baseline
3. Enhanced FiLM-RGCN
4. RGCN + Distance Readout
5. Multi-Relational RGCN
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_top5_benchmark_image(output_path, dark_mode=True):
    fig = plt.figure(figsize=(16, 9), dpi=300)
    
    if dark_mode:
        bg_color = '#0B132B'
        card_color = '#1C2541'
        card_border = '#334155'
        text_primary = '#F8FAFC'
        text_secondary = '#94A3B8'
        accent_best = '#10B981'    # Emerald for Best
        accent_xgb = '#38BDF8'     # Cyan for XGBoost
        accent_other = '#818CF8'   # Indigo for others
        grid_color = '#334155'
        table_header_bg = '#1E293B'
        table_row_alt = '#162032'
    else:
        bg_color = '#FFFFFF'
        card_color = '#F8FAFC'
        card_border = '#E2E8F0'
        text_primary = '#0F172A'
        text_secondary = '#64748B'
        accent_best = '#059669'
        accent_xgb = '#0284C7'
        accent_other = '#6366F1'
        grid_color = '#E2E8F0'
        table_header_bg = '#F1F5F9'
        table_row_alt = '#F8FAFC'

    fig.patch.set_facecolor(bg_color)
    
    # Define Layout with GridSpec: Left for Table (55%), Right for Graphs (45%)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], height_ratios=[1, 1],
                          left=0.04, right=0.96, top=0.89, bottom=0.06, wspace=0.18, hspace=0.32)

    # -------------------------------------------------------------------------
    # Super Title & Header Banner
    # -------------------------------------------------------------------------
    fig.text(0.04, 0.945, "TOP 5 BENCHMARK EVALUATION | METALLOPROTEIN REDOX PREDICTION",
             color=accent_best, fontsize=12, fontweight='bold')
    fig.text(0.04, 0.915, "Model Performance Hierarchy: Error Metrics & Goodness-of-Fit",
             color=text_primary, fontsize=18, fontweight='bold')
    fig.text(0.04, 0.892, "Rigorous 5-fold cross-validation across 371 iron-sulfur proteins (Curated 96 Descriptors vs 3D Relational Graph Networks)",
             color=text_secondary, fontsize=10.5)

    # -------------------------------------------------------------------------
    # LEFT PANEL: Comprehensive Formatted Table (Spans both rows)
    # -------------------------------------------------------------------------
    ax_table = fig.add_subplot(gs[:, 0])
    ax_table.set_facecolor(bg_color)
    ax_table.axis('off')

    # Data for Top 5 Models
    models_data = [
        {
            "rank": "1",
            "name": "FiLM-RGCN (Best)",
            "mae": "46.35 \u00b1 6.31",
            "rmse": "64.44 \u00b1 8.12",
            "r2": "0.929",
            "pearson": "0.965",
            "mechanism": "Layer-wise pH-FiLM affine modulation +\nDual-Scale continuous exponential readout",
            "is_best": True
        },
        {
            "rank": "2",
            "name": "XGBoost Baseline",
            "mae": "48.26 \u00b1 3.61",
            "rmse": "71.40 \u00b1 11.00",
            "r2": "0.912",
            "pearson": "0.957",
            "mechanism": "Curated 96 non-redundant biophysical\ndescriptors (SASA, H-bonds, electrostatic cavity)",
            "is_best": False
        },
        {
            "rank": "3",
            "name": "Enhanced FiLM-RGCN",
            "mae": "48.64 \u00b1 4.85",
            "rmse": "66.82 \u00b1 6.20",
            "r2": "0.924",
            "pearson": "0.963",
            "mechanism": "Tri-scale distance readout +\nDistance-weighted DropNode regularization",
            "is_best": False
        },
        {
            "rank": "4",
            "name": "RGCN + Distance Readout",
            "mae": "54.57 \u00b1 11.02",
            "rmse": "84.28 \u00b1 12.60",
            "r2": "0.876",
            "pearson": "0.937",
            "mechanism": "5-relational message passing with continuous\nexponential soft readout (no pH FiLM)",
            "is_best": False
        },
        {
            "rank": "5",
            "name": "Multi-Relational RGCN",
            "mae": "57.95 \u00b1 6.73",
            "rmse": "77.64 \u00b1 8.43",
            "r2": "0.897",
            "pearson": "0.954",
            "mechanism": "Explicit physical bond relations (Covalent,\nCoordinate, H-Bonds, Hydrophobic, Clusters)",
            "is_best": False
        }
    ]

    # Draw Outer Container Box for Table
    table_card = patches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                        boxstyle='round,pad=0.015,rounding_size=0.02',
                                        facecolor=card_color, edgecolor=card_border, linewidth=1.5,
                                        transform=ax_table.transAxes, zorder=1)
    ax_table.add_patch(table_card)

    # Column Layout Definition (Normalized coords 0.0 to 1.0)
    col_x = [0.03, 0.09, 0.35, 0.50, 0.62, 0.72]
    headers = ["#", "Model Architecture", "MAE (mV) \u2193", "RMSE (mV) \u2193", "R\u00b2 Score \u2191", "Pearson r \u2191"]

    # Table Header Row
    header_box = patches.Rectangle((0.015, 0.89), 0.97, 0.09, facecolor=table_header_bg,
                                   transform=ax_table.transAxes, zorder=2)
    ax_table.add_patch(header_box)

    for i, h in enumerate(headers):
        align = 'center' if i in [0, 2, 3, 4, 5] else 'left'
        x_pos = col_x[i] if align == 'left' else (col_x[i] + (col_x[i+1]-col_x[i])/2 if i < 5 else 0.79)
        ax_table.text(x_pos, 0.935, h, transform=ax_table.transAxes,
                      color=accent_best if i == 2 else text_primary,
                      fontsize=10.5, fontweight='bold', ha=align, va='center', zorder=3)

    # Draw Rows
    row_y = 0.88
    row_h = 0.17
    for idx, row in enumerate(models_data):
        y_top = row_y - (idx * row_h)
        is_even = idx % 2 == 1
        
        # Row Background Highlight for Best
        row_bg_color = '#132A2B' if (row["is_best"] and dark_mode) else (table_row_alt if is_even else card_color)
        row_border_color = accent_best if row["is_best"] else card_border
        
        row_box = patches.FancyBboxPatch((0.015, y_top - row_h + 0.01), 0.97, row_h - 0.01,
                                         boxstyle='round,pad=0.005,rounding_size=0.01',
                                         facecolor=row_bg_color, edgecolor=row_border_color,
                                         linewidth=1.8 if row["is_best"] else 0.8,
                                         transform=ax_table.transAxes, zorder=2)
        ax_table.add_patch(row_box)

        # Rank
        ax_table.text(col_x[0] + 0.025, y_top - 0.05, row["rank"], transform=ax_table.transAxes,
                      color=accent_best if row["is_best"] else text_secondary,
                      fontsize=11, fontweight='bold', ha='center', va='center', zorder=3)

        # Model Name
        ax_table.text(col_x[1], y_top - 0.045, row["name"], transform=ax_table.transAxes,
                      color=accent_best if row["is_best"] else (accent_xgb if "XGBoost" in row["name"] else text_primary),
                      fontsize=11 if row["is_best"] else 10.5,
                      fontweight='bold', ha='left', va='center', zorder=3)
        
        # Architectural mechanism / description (subtitle below name)
        ax_table.text(col_x[1], y_top - 0.11, row["mechanism"], transform=ax_table.transAxes,
                      color=text_secondary, fontsize=7.8, ha='left', va='center', zorder=3)

        # Metrics (centered)
        metrics_vals = [row["mae"], row["rmse"], row["r2"], row["pearson"]]
        for m_idx, val in enumerate(metrics_vals):
            m_col = col_x[m_idx + 2]
            next_col = col_x[m_idx + 3] if (m_idx + 3) < len(col_x) else 0.88
            c_pos = m_col + (next_col - m_col) / 2
            
            is_mae = (m_idx == 0)
            m_color = accent_best if (row["is_best"] or is_mae) else text_primary
            font_w = 'bold' if (row["is_best"] or is_mae) else 'normal'
            
            ax_table.text(c_pos, y_top - 0.07, val, transform=ax_table.transAxes,
                          color=m_color, fontsize=10 if row["is_best"] else 9.5,
                          fontweight=font_w, ha='center', va='center', zorder=3)

    # -------------------------------------------------------------------------
    # RIGHT TOP PANEL: MAE & RMSE Error Comparison (Bar Chart)
    # -------------------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor(card_color)
    
    # Model Labels for Graphs
    graph_labels = [
        "FiLM-RGCN\n(Best)",
        "XGBoost\nBaseline",
        "Enhanced\nFiLM-RGCN",
        "RGCN +\nDist Readout",
        "Multi-Rel\nRGCN"
    ]
    
    maes = [46.35, 48.26, 48.64, 54.57, 57.95]
    mae_stds = [6.31, 3.61, 4.85, 11.02, 6.73]
    rmses = [64.44, 71.40, 66.82, 84.28, 77.64]
    rmse_stds = [8.12, 11.00, 6.20, 12.60, 8.43]
    
    x = np.arange(len(graph_labels))
    width = 0.36

    # Colors
    bar_mae_colors = [accent_best, accent_xgb, '#34D399', '#64748B', '#475569']
    bar_rmse_colors = ['#059669', '#0284C7', '#10B981', '#475569', '#334155']

    rects1 = ax_bar.bar(x - width/2, maes, width, yerr=mae_stds, capsize=4,
                        label='MAE (mV)', color=bar_mae_colors, edgecolor='#1E293B', linewidth=1.2)
    rects2 = ax_bar.bar(x + width/2, rmses, width, yerr=rmse_stds, capsize=4,
                        label='RMSE (mV)', color=bar_rmse_colors, edgecolor='#1E293B', linewidth=1.2, alpha=0.85)

    ax_bar.set_title("Prediction Error: MAE & RMSE (Lower is Better)", fontsize=11, fontweight='bold', color=text_primary, pad=10)
    ax_bar.set_ylabel("Error (mV)", fontsize=10, fontweight='bold', color=text_secondary)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(graph_labels, fontsize=8.5, color=text_primary, fontweight='bold')
    ax_bar.set_ylim(0, 110)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.3, color=grid_color)
    ax_bar.tick_params(axis='y', colors=text_secondary, labelsize=9)
    ax_bar.tick_params(axis='x', colors=text_primary)
    
    # Border styling
    for spine in ax_bar.spines.values():
        spine.set_color(card_border)
        spine.set_linewidth(1.2)
        
    ax_bar.legend(loc='upper left', framealpha=0.4, facecolor=card_color, edgecolor=card_border,
                  labelcolor=text_primary, fontsize=8.5)

    # Annotate Best values
    ax_bar.text(0 - width/2, 46.35 + 8, "46.4 mV", ha='center', va='bottom',
                color=accent_best, fontweight='bold', fontsize=8.5)
    ax_bar.text(1 - width/2, 48.26 + 5.5, "48.3 mV", ha='center', va='bottom',
                color=accent_xgb, fontweight='bold', fontsize=8.5)

    # -------------------------------------------------------------------------
    # RIGHT BOTTOM PANEL: Goodness of Fit (R² and Pearson r)
    # -------------------------------------------------------------------------
    ax_fit = fig.add_subplot(gs[1, 1])
    ax_fit.set_facecolor(card_color)

    r2_scores = [0.929, 0.912, 0.924, 0.876, 0.897]
    pearson_r = [0.965, 0.957, 0.963, 0.937, 0.954]

    rects_r2 = ax_fit.bar(x - width/2, r2_scores, width,
                          label='R² Score', color=bar_mae_colors, edgecolor='#1E293B', linewidth=1.2)
    rects_pr = ax_fit.bar(x + width/2, pearson_r, width,
                          label='Pearson r', color=bar_rmse_colors, edgecolor='#1E293B', linewidth=1.2, alpha=0.85)

    ax_fit.set_title("Correlation & Variance Explained: R² & Pearson r (Higher is Better)", fontsize=11, fontweight='bold', color=text_primary, pad=10)
    ax_fit.set_ylabel("Coefficient Value", fontsize=10, fontweight='bold', color=text_secondary)
    ax_fit.set_xticks(x)
    ax_fit.set_xticklabels(graph_labels, fontsize=8.5, color=text_primary, fontweight='bold')
    ax_fit.set_ylim(0.80, 1.00)
    ax_fit.grid(axis='y', linestyle='--', alpha=0.3, color=grid_color)
    ax_fit.tick_params(axis='y', colors=text_secondary, labelsize=9)
    ax_fit.tick_params(axis='x', colors=text_primary)

    for spine in ax_fit.spines.values():
        spine.set_color(card_border)
        spine.set_linewidth(1.2)

    ax_fit.legend(loc='lower right', framealpha=0.4, facecolor=card_color, edgecolor=card_border,
                  labelcolor=text_primary, fontsize=8.5)

    # Annotate Champion on fit
    ax_fit.text(0 - width/2, 0.929 + 0.006, "0.929", ha='center', va='bottom',
                color=accent_best, fontweight='bold', fontsize=8.5)
    ax_fit.text(0 + width/2, 0.965 + 0.006, "0.965", ha='center', va='bottom',
                color=accent_best, fontweight='bold', fontsize=8.5)

    # Save high-resolution PNG
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Top 5 Models benchmark table & graph successfully saved to:\n  {output_path}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "Custom", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Generate both dark mode and light mode versions
    out_dark = os.path.join(out_dir, "top_5_models_benchmark_table_and_graph.png")
    create_top5_benchmark_image(out_dark, dark_mode=True)

    out_light = os.path.join(out_dir, "top_5_models_benchmark_table_and_graph_light.png")
    create_top5_benchmark_image(out_light, dark_mode=False)

if __name__ == "__main__":
    main()
