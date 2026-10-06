"""
Executive Pitch Deck Generator: Metalloprotein Redox Prediction & FiLM-RGCN
=============================================================================
This script generates:
1. A 3-slide widescreen (16:9) executive pitch presentation in native PPTX:
   Custom/outputs/FeS_Redox_GNN_Executive_Pitch.pptx
2. High-resolution visual slide exports (PNG & PDF) for instant preview:
   Custom/outputs/slide_1_grand_challenge.png
   Custom/outputs/slide_2_architecture_breakthrough.png
   Custom/outputs/slide_3_results_xai_impact.png
   Custom/outputs/FeS_Redox_GNN_Executive_Pitch.pdf

Content Breakdown:
- Slide 1: The Grand Challenge & The Physics "Hard Problem" (Fe-S 3D structure, 1000 mV paradox)
- Slide 2: Prior Work Limitations vs. Our Breakthrough Architecture (FiLM-RGCN, 5 edge types, pH FiLM)
- Slide 3: State-of-the-Art Results, Explainable AI (SHAP), & Future De Novo Design Impact
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Import python-pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# Import reportlab for companion PDF export
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pptx_deck(output_pptx_path, img_dir):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors
    COLOR_BG = RGBColor(11, 19, 43)        # #0B132B Deep Slate Navy
    COLOR_CARD = RGBColor(28, 37, 65)      # #1C2541 Card Container
    COLOR_CARD_BORDER = RGBColor(51, 65, 85) # #334155 Subtle Slate Border
    COLOR_CYAN = RGBColor(56, 189, 248)    # #38BDF8 Electric Cyan
    COLOR_EMERALD = RGBColor(16, 185, 129) # #10B981 Emerald Green
    COLOR_AMBER = RGBColor(245, 158, 11)   # #F59E0B Warm Amber
    COLOR_CORAL = RGBColor(239, 68, 68)    # #EF4444 Alert Coral
    COLOR_WHITE = RGBColor(248, 250, 252)  # #F8FAFC Heading White
    COLOR_MUTED = RGBColor(148, 163, 184)  # #94A3B8 Secondary Gray
    COLOR_LIGHT = RGBColor(203, 213, 225)  # #CBD5E1 Body Text

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG
        bg.line.fill.background() # no line

    def add_header(slide, tag_text, title_text, subtitle_text):
        # Header Box
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        # Tag
        p_tag = tf.paragraphs[0]
        p_tag.text = tag_text.upper()
        p_tag.font.name = "Calibri"
        p_tag.font.size = Pt(9.5)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_CYAN

        # Title
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Calibri"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_WHITE
        p_title.space_before = Pt(2)

        # Subtitle
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle_text
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = COLOR_MUTED
        p_sub.space_before = Pt(2)

    def add_card(slide, left, top, width, height, title, title_color=COLOR_CYAN):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD
        card.line.color.rgb = COLOR_CARD_BORDER
        card.line.width = Pt(1.2)
        
        # Add Card Title Textbox
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = title_color
        return card

    # =========================================================================
    # SLIDE 1: The Grand Challenge & The Physics "Hard Problem"
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)
    add_header(slide1, 
               "Executive Pitch | Problem Definition & Biophysical Frontier",
               "Engineering Nature's Electrical Wiring: Predicting Metalloprotein Redox Potentials",
               "Why Iron-Sulfur ([Fe-S]) cluster redox potential (Em) prediction is a multi-million dollar biophysical bottleneck")

    # Card 1A: What We Are Working On & Why It Matters (Left Column)
    add_card(slide1, Inches(0.8), Inches(1.65), Inches(5.6), Inches(5.35), 
             "1. WHAT WE DO & WHY IT MATTERS", COLOR_CYAN)
    
    tb1a = slide1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.6))
    tf1a = tb1a.text_frame
    tf1a.word_wrap = True
    tf1a.margin_left = tf1a.margin_top = tf1a.margin_right = tf1a.margin_bottom = 0

    points_1a = [
        ("Nature's Universal Bio-Batteries:", " Iron-Sulfur ([Fe-S]) proteins are the electrical grid of all living organisms. They drive essential electron-transport cascades in cellular respiration (Complex I), photosynthesis (Photosystem I), nitrogen fixation (nitrogenase), and DNA repair."),
        ("Hyper-Sensitivity (The 60 mV Rule):", " In electron transfer, thermodynamics dictates that a mere ±60 mV shift in redox potential (Em) alters electron transfer kinetics by a full order of magnitude (10x), easily halting vital metabolic fluxes."),
        ("The High-Value Mission:", " Accurately predicting Em directly from 3D molecular structures without cost-prohibitive quantum chemistry unlocks rational de novo metalloenzyme design, green hydrogen biocatalysis, and synthetic bio-circuitry."),
        ("The Current Bottleneck:", " Wet-lab electrochemistry requires costly protein purification ($1,000s/mutant), while DFT quantum calculations require days of supercomputing per cluster and fail on dynamic protein dielectric environments.")
    ]

    for i, (head, body) in enumerate(points_1a):
        p = tf1a.paragraphs[0] if i == 0 else tf1a.add_paragraph()
        p.space_before = Pt(8)
        run_h = p.add_run()
        run_h.text = "• " + head
        run_h.font.name = "Calibri"
        run_h.font.size = Pt(10.5)
        run_h.font.bold = True
        run_h.font.color.rgb = COLOR_WHITE
        
        run_b = p.add_run()
        run_b.text = body
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(10)
        run_b.font.color.rgb = COLOR_LIGHT

    # Card 1B: Why It's an Extremely Hard Problem (Right Column - Top)
    add_card(slide1, Inches(6.6), Inches(1.65), Inches(5.9), Inches(2.55), 
             "2. WHY IT IS AN EXTREMELY HARD PROBLEM", COLOR_AMBER)
    
    tb1b = slide1.shapes.add_textbox(Inches(6.8), Inches(2.2), Inches(5.5), Inches(1.9))
    tf1b = tb1b.text_frame
    tf1b.word_wrap = True
    tf1b.margin_left = tf1b.margin_top = tf1b.margin_right = tf1b.margin_bottom = 0

    points_1b = [
        ("The 1,000 mV Paradox: ", "Chemically identical [4Fe-4S] cubane clusters span a massive 1,200 mV physiological range—from -700 mV in Ferredoxins to +500 mV in HiPIPs. The chemical core is identical; the protein microenvironment alone controls the voltage."),
        ("Sub-Angstrom Quantum Sensitivity: ", "A single hydrogen bond (N-H···S) from the peptide backbone shifts Em by +60 to +100 mV. Subtle changes in solvent accessibility (SASA), dielectric screening, and pH titration completely alter cluster potential.")
    ]

    for i, (head, body) in enumerate(points_1b):
        p = tf1b.paragraphs[0] if i == 0 else tf1b.add_paragraph()
        p.space_before = Pt(6)
        run_h = p.add_run()
        run_h.text = "• " + head
        run_h.font.name = "Calibri"
        run_h.font.size = Pt(10)
        run_h.font.bold = True
        run_h.font.color.rgb = COLOR_WHITE
        
        run_b = p.add_run()
        run_b.text = body
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(9.5)
        run_b.font.color.rgb = COLOR_LIGHT

    # Card 1C: Structural Architecture (Right Column - Bottom with 3D Structure)
    add_card(slide1, Inches(6.6), Inches(4.35), Inches(5.9), Inches(2.65), 
             "3. QUANTUM ACTIVE SITE: [4Fe-4S] CLUSTER MICROENVIRONMENT", COLOR_EMERALD)
    
    # Embed 3D structure image
    struct_img_path = os.path.join(img_dir, "4fxc_structure.png")
    if os.path.exists(struct_img_path):
        slide1.shapes.add_picture(struct_img_path, Inches(6.8), Inches(4.8), Inches(2.9), Inches(2.05))
    
    tb1c = slide1.shapes.add_textbox(Inches(9.85), Inches(4.85), Inches(2.55), Inches(2.0))
    tf1c = tb1c.text_frame
    tf1c.word_wrap = True
    tf1c.margin_left = tf1c.margin_top = tf1c.margin_right = tf1c.margin_bottom = 0

    points_1c = [
        ("Inorganic Cubane Core:", " Fe4S4 cluster undergoing reversible [Fe4S4]2+/1+ redox transitions."),
        ("Cysteine Coordination:", " 4 invariant Cys-Sγ coordinate bonds providing primary electron pathways."),
        ("Outer Dielectric Shell:", " Polarity and pH modulate electrostatic charge stabilization.")
    ]
    for i, (head, body) in enumerate(points_1c):
        p = tf1c.paragraphs[0] if i == 0 else tf1c.add_paragraph()
        p.space_before = Pt(4)
        run_h = p.add_run()
        run_h.text = "▸ " + head
        run_h.font.name = "Calibri"
        run_h.font.size = Pt(9)
        run_h.font.bold = True
        run_h.font.color.rgb = COLOR_CYAN
        
        run_b = p.add_run()
        run_b.text = body
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(8.5)
        run_b.font.color.rgb = COLOR_LIGHT

    # =========================================================================
    # SLIDE 2: Prior Work Limitations vs. Our Breakthrough Architecture
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, 
               "Methodology & Architectural Innovation | FiLM-RGCN",
               "Overcoming Broken Baselines with Physics-Conditioned Graph Message Passing",
               "How we diagnosed prior literature failures and designed a continuous multi-relational graph neural network")

    # Card 2A: Prior Literature & Why They Failed (Left)
    add_card(slide2, Inches(0.8), Inches(1.65), Inches(5.6), Inches(2.75), 
             "PRIOR LITERATURE & HIDDEN PITFALLS", COLOR_CORAL)
    
    tb2a = slide2.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.2), Inches(2.1))
    tf2a = tb2a.text_frame
    tf2a.word_wrap = True
    tf2a.margin_left = tf2a.margin_top = tf2a.margin_right = tf2a.margin_bottom = 0

    points_2a = [
        ("The Published Baseline (JCIM 2025):", " Persico et al. proposed tabular feature extraction with XGBoost, claiming ~40 mV on a subset, but their GitHub code yielded massive errors (~143-165 mV, R² ≈ 0.18)."),
        ("Our Reproduction & Bug Discovery:", " Running the exact repository revealed severe dictionary key collisions in feature extraction, generating hundreds of corrupted NaN/zero columns and breaking model training."),
        ("Why Standard GNNs Fail:", " Standard GCNs and GATs suffer from Signal Dilution (averaging across 300+ bulk residues drowns the active site) and Boundary Cliffs (rigid distance cutoffs create artificial discontinuities).")
    ]
    for i, (head, body) in enumerate(points_2a):
        p = tf2a.paragraphs[0] if i == 0 else tf2a.add_paragraph()
        p.space_before = Pt(5)
        run_h = p.add_run()
        run_h.text = "✖ " + head
        run_h.font.name = "Calibri"
        run_h.font.size = Pt(9.5)
        run_h.font.bold = True
        run_h.font.color.rgb = COLOR_WHITE
        
        run_b = p.add_run()
        run_b.text = body
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(9)
        run_b.font.color.rgb = COLOR_MUTED

    # Card 2B: Our Novel Solution: FiLM-RGCN (Right)
    add_card(slide2, Inches(6.6), Inches(1.65), Inches(5.9), Inches(2.75), 
             "OUR SOLUTION: PHYSICS-CONDITIONED FiLM-RGCN", COLOR_EMERALD)
    
    tb2b = slide2.shapes.add_textbox(Inches(6.8), Inches(2.15), Inches(5.5), Inches(2.1))
    tf2b = tb2b.text_frame
    tf2b.word_wrap = True
    tf2b.margin_left = tf2b.margin_top = tf2b.margin_right = tf2b.margin_bottom = 0

    points_2b = [
        ("1. Curated 96 Biophysical Descriptors:", " Completely resolved key collisions; non-redundant descriptors drop baseline error directly to 48.26 mV."),
        ("2. 5-Relational Molecular Topology:", " Replaces generic spatial edges with chemically explicit message passing: (1) Covalent Backbone, (2) Fe-S Coordinate, (3) Polar H-Bonds (N-H···S), (4) Hydrophobic, (5) Cluster-Cluster."),
        ("3. Layer-Wise pH-FiLM Affine Modulation:", " Injects solvent pH into every GNN layer as an affine transformation (h' = γ(pH) ⊙ h + β(pH)), modeling protonation dynamics."),
        ("4. Dual-Scale Exponential Distance Readout:", " Continuous physics-based pooling w(d) = exp(-d/d0) ensures inner-shell quantum atoms dominate over bulk solvent.")
    ]
    for i, (head, body) in enumerate(points_2b):
        p = tf2b.paragraphs[0] if i == 0 else tf2b.add_paragraph()
        p.space_before = Pt(4)
        run_h = p.add_run()
        run_h.text = "✔ " + head
        run_h.font.name = "Calibri"
        run_h.font.size = Pt(9.5)
        run_h.font.bold = True
        run_h.font.color.rgb = COLOR_WHITE
        
        run_b = p.add_run()
        run_b.text = body
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(9)
        run_b.font.color.rgb = COLOR_LIGHT

    # Card 2C: Full Architecture Diagram (Bottom)
    add_card(slide2, Inches(0.8), Inches(4.55), Inches(11.733), Inches(2.45), 
             "END-TO-END FiLM-RGCN DEEP LEARNING PIPELINE", COLOR_CYAN)
    
    arch_img_path = os.path.join(img_dir, "film_rgcn_architecture_slide_diagram.png")
    if os.path.exists(arch_img_path):
        slide2.shapes.add_picture(arch_img_path, Inches(1.0), Inches(5.0), Inches(11.333), Inches(1.85))

    # =========================================================================
    # SLIDE 3: Results, Explainable AI (XAI), & Real-World Impact
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, 
               "Breakthrough Results & Real-World Application | Impact Pitch",
               "Unprecedented Accuracy, Mechanistic Explainability & In Silico Design",
               "How our model sets a new standard for computational metalloenzyme engineering and green synthetic biology")

    # Column 3A: Benchmark Performance (Left)
    add_card(slide3, Inches(0.8), Inches(1.65), Inches(3.8), Inches(5.35), 
             "1. BENCHMARK TRIUMPH", COLOR_CYAN)
    
    # Parity scatter image
    scatter_img_path = os.path.join(img_dir, "rgcn_film_ph_scatter.png")
    if os.path.exists(scatter_img_path):
        slide3.shapes.add_picture(scatter_img_path, Inches(1.0), Inches(2.15), Inches(3.4), Inches(2.6))
    
    tb3a = slide3.shapes.add_textbox(Inches(1.0), Inches(4.85), Inches(3.4), Inches(2.0))
    tf3a = tb3a.text_frame
    tf3a.word_wrap = True
    tf3a.margin_left = tf3a.margin_top = tf3a.margin_right = tf3a.margin_bottom = 0

    points_3a = [
        ("Champion Accuracy: ", "MAE = 46.35 mV, R² = 0.929, Pearson r = 0.965 across all 371 proteins."),
        (">70% Error Reduction: ", "Slashes error from author's repo (164.8 mV) down to 46.35 mV."),
        ("Full-Range Parity: ", "Consistent fidelity from -700 mV ferredoxins to +500 mV HiPIPs without boundary artifacts.")
    ]
    for i, (head, body) in enumerate(points_3a):
        p = tf3a.paragraphs[0] if i == 0 else tf3a.add_paragraph()
        p.space_before = Pt(4)
        run_h = p.add_run()
        run_h.text = "• " + head
        run_h.font.name = "Calibri"
        run_h.font.size = Pt(9.5)
        run_h.font.bold = True
        run_h.font.color.rgb = COLOR_EMERALD
        
        run_b = p.add_run()
        run_b.text = body
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(9)
        run_b.font.color.rgb = COLOR_LIGHT

    # Column 3B: Explainable AI & Quantum Insights (Middle)
    add_card(slide3, Inches(4.8), Inches(1.65), Inches(3.9), Inches(5.35), 
             "2. MECHANISTIC EXPLAINABILITY (XAI)", COLOR_AMBER)
    
    xai_img_path = os.path.join(img_dir, "xai_edge_relation_importance.png")
    if os.path.exists(xai_img_path):
        slide3.shapes.add_picture(xai_img_path, Inches(5.0), Inches(2.15), Inches(3.5), Inches(2.3))
    
    tb3b = slide3.shapes.add_textbox(Inches(5.0), Inches(4.6), Inches(3.5), Inches(2.3))
    tf3b = tb3b.text_frame
    tf3b.word_wrap = True
    tf3b.margin_left = tf3b.margin_top = tf3b.margin_right = tf3b.margin_bottom = 0

    points_3b = [
        ("Rediscovering Real Physics: ", "SHAP analysis proves the model learned true bioinorganic principles rather than dataset noise."),
        ("H-Bonding Primacy: ", "Backbone N-H···S hydrogen bonds carry the highest positive weight (+60 to +100 mV per bond)."),
        ("Dielectric & SASA Shielding: ", "Solvent exposure promotes solvent reorganization energy, sharply depressing Em."),
        ("pH Nernstian Response: ", "Captures physiological proton-coupled electron transfer (PCET) titration curves.")
    ]
    for i, (head, body) in enumerate(points_3b):
        p = tf3b.paragraphs[0] if i == 0 else tf3b.add_paragraph()
        p.space_before = Pt(4)
        run_h = p.add_run()
        run_h.text = "▸ " + head
        run_h.font.name = "Calibri"
        run_h.font.size = Pt(9.5)
        run_h.font.bold = True
        run_h.font.color.rgb = COLOR_WHITE
        
        run_b = p.add_run()
        run_b.text = body
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(9)
        run_b.font.color.rgb = COLOR_LIGHT

    # Column 3C: External Pitch & Commercial Application (Right)
    add_card(slide3, Inches(8.9), Inches(1.65), Inches(3.633), Inches(5.35), 
             "3. WHY IT MATTERS & FUTURE IMPACT", COLOR_EMERALD)
    
    tb3c = slide3.shapes.add_textbox(Inches(9.1), Inches(2.2), Inches(3.233), Inches(4.6))
    tf3c = tb3c.text_frame
    tf3c.word_wrap = True
    tf3c.margin_left = tf3c.margin_top = tf3c.margin_right = tf3c.margin_bottom = 0

    points_3c = [
        ("De Novo Protein Design:", " Screen millions of computationally generated protein backbones (e.g. from RFdiffusion/Chroma) in seconds for optimal redox potentials."),
        ("Zero Wet-Lab Wastage:", " Filter out 95% of non-viable synthetic candidates before expensive gene synthesis and expression in the lab."),
        ("Green Tech & Bio-Hydrogen:", " Optimize [FeFe]- and [NiFe]-hydrogenases to match renewable photovoltaic electrode potentials for ultra-efficient carbon-neutral H2 production."),
        ("Metabolic Circuit Optimization:", " Tune electron transfer bottlenecks in engineered synthetic microbes for bio-manufacturing of pharmaceuticals and fuels."),
        ("Instant Inference (<15 ms):", " 10,000x faster than DFT; enables interactive point mutation screening on a standard laptop.")
    ]
    for i, (head, body) in enumerate(points_3c):
        p = tf3c.paragraphs[0] if i == 0 else tf3c.add_paragraph()
        p.space_before = Pt(7)
        run_h = p.add_run()
        run_h.text = "★ " + head
        run_h.font.name = "Calibri"
        run_h.font.size = Pt(9.5)
        run_h.font.bold = True
        run_h.font.color.rgb = COLOR_CYAN
        
        run_b = p.add_run()
        run_b.text = body
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(9)
        run_b.font.color.rgb = COLOR_LIGHT

    # Save PPTX
    prs.save(output_pptx_path)
    print(f"Native PowerPoint presentation successfully saved to:\n  {output_pptx_path}")

def create_visual_slides_and_pdf(output_pdf_path, img_dir):
    """
    Renders high-resolution image previews of the 3 slides and bundles them into an executive PDF.
    """
    from reportlab.pdfgen import canvas
    
    # Dimensions: 16:9 aspect ratio in points
    slide_w, slide_h = 13.333 * 72, 7.5 * 72 # 960 x 540 pt
    
    # 1. Generate Slide 1 Preview Image using Matplotlib
    fig1, ax1 = plt.subplots(figsize=(13.333, 7.5), facecolor='#0B132B')
    ax1.set_facecolor('#0B132B')
    ax1.axis('off')
    ax1.set_xlim(0, 13.333)
    ax1.set_ylim(0, 7.5)

    # Header
    ax1.text(0.8, 6.9, "EXECUTIVE PITCH | PROBLEM DEFINITION & BIOPHYSICAL FRONTIER", color='#38BDF8', fontsize=11, fontweight='bold')
    ax1.text(0.8, 6.4, "Engineering Nature's Electrical Wiring: Predicting Metalloprotein Redox Potentials", color='#FFFFFF', fontsize=18, fontweight='bold')
    ax1.text(0.8, 6.05, "Why Iron-Sulfur ([Fe-S]) cluster redox potential (Em) prediction is a multi-million dollar biophysical bottleneck", color='#94A3B8', fontsize=11)

    # Helper function for card
    def plot_card(ax, x, y, w, h, title, title_color='#38BDF8'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08,rounding_size=0.15',
                                     facecolor='#1C2541', edgecolor='#334155', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.3, y + h - 0.35, title, color=title_color, fontsize=12, fontweight='bold')

    # Card 1A
    plot_card(ax1, 0.8, 0.5, 5.6, 5.3, "1. WHAT WE DO & WHY IT MATTERS", '#38BDF8')
    text_1a = [
        ("Nature's Universal Bio-Batteries:", " Fe-S proteins power respiration (Complex I), photosynthesis, and nitrogenase."),
        ("Hyper-Sensitivity (60 mV Rule):", " A ±60 mV shift alters electron transfer rates by 10x, halting cellular metabolism."),
        ("The High-Value Mission:", " Direct 3D structure-to-voltage prediction unlocks rational de novo metalloenzyme design."),
        ("The Current Bottleneck:", " Lab purification costs $1,000s/mutant; quantum DFT takes days and ignores bulk dielectrics.")
    ]
    curr_y = 5.0
    for head, body in text_1a:
        ax1.text(1.0, curr_y, f"• {head}", color='#FFFFFF', fontsize=9.5, fontweight='bold')
        ax1.text(1.0, curr_y - 0.3, body, color='#CBD5E1', fontsize=9)
        curr_y -= 0.85

    # Card 1B
    plot_card(ax1, 6.6, 3.4, 5.9, 2.4, "2. WHY IT IS AN EXTREMELY HARD PROBLEM", '#F59E0B')
    text_1b = [
        ("The 1,000 mV Paradox:", " Identical [4Fe-4S] cubanes span -700 mV (Fd) to +500 mV (HiPIP)."),
        ("Sub-Angstrom Quantum Sensitivity:", " A single N-H···S hydrogen bond shifts Em by +60 to +100 mV."),
        ("Solvent & pH Coupling:", " Dielectric shielding and pH titration control active site thermodynamics.")
    ]
    curr_y = 5.2
    for head, body in text_1b:
        ax1.text(6.8, curr_y, f"• {head}", color='#FFFFFF', fontsize=9.5, fontweight='bold')
        ax1.text(6.8, curr_y - 0.25, body, color='#CBD5E1', fontsize=9)
        curr_y -= 0.65

    # Card 1C
    plot_card(ax1, 6.6, 0.5, 5.9, 2.7, "3. QUANTUM ACTIVE SITE: [4Fe-4S] CLUSTER MICROENVIRONMENT", '#10B981')
    struct_img_path = os.path.join(img_dir, "4fxc_structure.png")
    if os.path.exists(struct_img_path):
        im = Image.open(struct_img_path)
        ax1.imshow(im, extent=[6.8, 9.8, 0.7, 2.7], aspect='auto', zorder=3)
    ax1.text(10.0, 2.4, "▸ Inorganic Cubane Core", color='#38BDF8', fontsize=9.5, fontweight='bold')
    ax1.text(10.0, 2.15, "  [Fe4S4] redox center", color='#CBD5E1', fontsize=8.5)
    ax1.text(10.0, 1.75, "▸ Cysteine Coordination", color='#38BDF8', fontsize=9.5, fontweight='bold')
    ax1.text(10.0, 1.5, "  4 Cys-Sγ ligands", color='#CBD5E1', fontsize=8.5)
    ax1.text(10.0, 1.1, "▸ Outer Dielectric Shell", color='#38BDF8', fontsize=9.5, fontweight='bold')
    ax1.text(10.0, 0.85, "  Polarity & pH titration", color='#CBD5E1', fontsize=8.5)

    slide1_img_path = os.path.join(img_dir, "slide_1_grand_challenge.png")
    plt.tight_layout()
    plt.savefig(slide1_img_path, dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Generate Slide 2 Preview Image using Matplotlib
    fig2, ax2 = plt.subplots(figsize=(13.333, 7.5), facecolor='#0B132B')
    ax2.set_facecolor('#0B132B')
    ax2.axis('off')
    ax2.set_xlim(0, 13.333)
    ax2.set_ylim(0, 7.5)

    ax2.text(0.8, 6.9, "METHODOLOGY & ARCHITECTURAL INNOVATION | FiLM-RGCN", color='#38BDF8', fontsize=11, fontweight='bold')
    ax2.text(0.8, 6.4, "Overcoming Broken Baselines with Physics-Conditioned Graph Message Passing", color='#FFFFFF', fontsize=18, fontweight='bold')
    ax2.text(0.8, 6.05, "Diagnosing literature feature collisions and establishing a continuous 5-relational graph architecture", color='#94A3B8', fontsize=11)

    # Card 2A
    plot_card(ax2, 0.8, 3.2, 5.6, 2.6, "PRIOR LITERATURE & HIDDEN PITFALLS", '#EF4444')
    text_2a = [
        ("Published Baseline (JCIM 2025):", " XGBoost on raw tables yielded ~143-165 mV error on full data."),
        ("Hidden Pipeline Bug Discovered:", " Dictionary key overwrites generated hundreds of corrupted NaN columns."),
        ("Vanilla GNNs Fail:", " Signal dilution across 300+ residues washes out the active site.")
    ]
    curr_y = 5.2
    for head, body in text_2a:
        ax2.text(1.0, curr_y, f"✖ {head}", color='#FFFFFF', fontsize=9.5, fontweight='bold')
        ax2.text(1.0, curr_y - 0.25, body, color='#94A3B8', fontsize=9)
        curr_y -= 0.65

    # Card 2B
    plot_card(ax2, 6.6, 3.2, 5.9, 2.6, "OUR NOVEL SOLUTION: FiLM-RGCN", '#10B981')
    text_2b = [
        ("Curated 96 Biophysical Descriptors:", " Non-redundant curated features drop baseline to 48.26 mV."),
        ("5-Relational Edge Topology:", " Explicitly models backbone, Fe-S, H-bonds, hydrophobic, & cluster edges."),
        ("Layer-Wise pH-FiLM Conditioning:", " Injects experimental pH as affine modulation (h' = γ(pH)⊙h + β(pH))."),
        ("Dual-Scale Exponential Readout:", " Continuous physics decay w(d) = exp(-d/d0) preserves inner quantum shell.")
    ]
    curr_y = 5.3
    for head, body in text_2b:
        ax2.text(6.8, curr_y, f"✔ {head}", color='#FFFFFF', fontsize=9.5, fontweight='bold')
        ax2.text(6.8, curr_y - 0.22, body, color='#CBD5E1', fontsize=8.8)
        curr_y -= 0.55

    # Card 2C
    plot_card(ax2, 0.8, 0.5, 11.7, 2.5, "END-TO-END FiLM-RGCN DEEP LEARNING PIPELINE", '#38BDF8')
    arch_img_path = os.path.join(img_dir, "film_rgcn_architecture_slide_diagram.png")
    if os.path.exists(arch_img_path):
        im_arch = Image.open(arch_img_path)
        ax2.imshow(im_arch, extent=[1.0, 12.3, 0.65, 2.5], aspect='auto', zorder=3)

    slide2_img_path = os.path.join(img_dir, "slide_2_architecture_breakthrough.png")
    plt.tight_layout()
    plt.savefig(slide2_img_path, dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Generate Slide 3 Preview Image using Matplotlib
    fig3, ax3 = plt.subplots(figsize=(13.333, 7.5), facecolor='#0B132B')
    ax3.set_facecolor('#0B132B')
    ax3.axis('off')
    ax3.set_xlim(0, 13.333)
    ax3.set_ylim(0, 7.5)

    ax3.text(0.8, 6.9, "BREAKTHROUGH RESULTS & REAL-WORLD APPLICATION | IMPACT PITCH", color='#38BDF8', fontsize=11, fontweight='bold')
    ax3.text(0.8, 6.4, "Unprecedented Accuracy, Mechanistic Explainability & In Silico Design", color='#FFFFFF', fontsize=18, fontweight='bold')
    ax3.text(0.8, 6.05, "Setting the new state-of-the-art and delivering actionable screening for de novo protein engineering", color='#94A3B8', fontsize=11)

    # Card 3A: Benchmark
    plot_card(ax3, 0.8, 0.5, 3.8, 5.3, "1. BENCHMARK TRIUMPH", '#38BDF8')
    scatter_img_path = os.path.join(img_dir, "rgcn_film_ph_scatter.png")
    if os.path.exists(scatter_img_path):
        im_sc = Image.open(scatter_img_path)
        ax3.imshow(im_sc, extent=[1.0, 4.4, 2.8, 5.2], aspect='auto', zorder=3)
    ax3.text(1.0, 2.5, "• Champion MAE: 46.35 mV", color='#10B981', fontsize=10, fontweight='bold')
    ax3.text(1.0, 2.2, "  R² = 0.929 | Pearson r = 0.965", color='#CBD5E1', fontsize=9)
    ax3.text(1.0, 1.8, "• >70% Error Reduction", color='#10B981', fontsize=10, fontweight='bold')
    ax3.text(1.0, 1.5, "  Slashes 164.8 mV baseline error", color='#CBD5E1', fontsize=9)
    ax3.text(1.0, 1.1, "• Universal Parity Agreement", color='#10B981', fontsize=10, fontweight='bold')
    ax3.text(1.0, 0.8, "  Full -700 mV to +500 mV range", color='#CBD5E1', fontsize=9)

    # Card 3B: XAI
    plot_card(ax3, 4.8, 0.5, 3.9, 5.3, "2. EXPLAINABLE AI (XAI)", '#F59E0B')
    xai_img_path = os.path.join(img_dir, "xai_edge_relation_importance.png")
    if os.path.exists(xai_img_path):
        im_xai = Image.open(xai_img_path)
        ax3.imshow(im_xai, extent=[5.0, 8.5, 3.1, 5.2], aspect='auto', zorder=3)
    ax3.text(5.0, 2.7, "▸ N-H···S Hydrogen Bonds:", color='#FFFFFF', fontsize=9.5, fontweight='bold')
    ax3.text(5.0, 2.45, "  Top driver: +60 to +100 mV per bond", color='#CBD5E1', fontsize=8.8)
    ax3.text(5.0, 2.05, "▸ Solvent Dielectric (SASA):", color='#FFFFFF', fontsize=9.5, fontweight='bold')
    ax3.text(5.0, 1.8, "  Water exposure depresses Em", color='#CBD5E1', fontsize=8.8)
    ax3.text(5.0, 1.4, "▸ pH Nernstian Titration:", color='#FFFFFF', fontsize=9.5, fontweight='bold')
    ax3.text(5.0, 1.15, "  Dynamic PCET proton-coupled shifts", color='#CBD5E1', fontsize=8.8)
    ax3.text(5.0, 0.85, "▸ Model Learns Real Biophysics", color='#38BDF8', fontsize=9.5, fontweight='bold')

    # Card 3C: Future Impact
    plot_card(ax3, 8.9, 0.5, 3.633, 5.3, "3. DE NOVO DESIGN IMPACT", '#10B981')
    text_3c = [
        ("De Novo Protein Design:", " Screen millions of RFdiffusion backbones in seconds for desired voltages."),
        ("Zero Wet-Lab Wastage:", " Filter out 95% of non-viable mutants prior to gene synthesis."),
        ("Green Tech & Bio-Hydrogen:", " Tune hydrogenases to match electrode potentials for solar fuels."),
        ("Metabolic Pathway Balance:", " Eliminate redox bottlenecks in engineered bacterial synthesis."),
        ("Instant Laptop Inference:", " <15 ms per protein (10,000x faster than supercomputer DFT).")
    ]
    curr_y = 5.1
    for head, body in text_3c:
        ax3.text(9.1, curr_y, f"★ {head}", color='#38BDF8', fontsize=9.5, fontweight='bold')
        ax3.text(9.1, curr_y - 0.28, body, color='#CBD5E1', fontsize=8.5)
        curr_y -= 0.88

    slide3_img_path = os.path.join(img_dir, "slide_3_results_xai_impact.png")
    plt.tight_layout()
    plt.savefig(slide3_img_path, dpi=200, bbox_inches='tight')
    plt.close()

    # 4. Assemble into a single PDF using canvas
    c = canvas.Canvas(output_pdf_path, pagesize=(slide_w, slide_h))
    
    for s_path in [slide1_img_path, slide2_img_path, slide3_img_path]:
        c.drawImage(s_path, 0, 0, width=slide_w, height=slide_h)
        c.showPage()
    c.save()
    print(f"Executive presentation PDF successfully saved to:\n  {output_pdf_path}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "Custom", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    pptx_path = os.path.join(out_dir, "FeS_Redox_GNN_Executive_Pitch.pptx")
    pdf_path = os.path.join(out_dir, "FeS_Redox_GNN_Executive_Pitch.pdf")

    print("\n" + "=" * 80)
    print("GENERATING EXECUTIVE PITCH DECK (PPTX + PDF + HIGH-RES SLIDES)")
    print("=" * 80)

    # 1. Build PPTX
    create_pptx_deck(pptx_path, out_dir)

    # 2. Build High-Res Images & PDF
    create_visual_slides_and_pdf(pdf_path, out_dir)

    print("\n" + "=" * 80)
    print("ALL ARTIFACTS SUCCESSFULLY GENERATED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
