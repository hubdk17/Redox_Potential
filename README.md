# FiLM-RGCN: Predicting Iron–Sulfur Protein Redox Potentials Using a pH-Conditioned Relational Graph Neural Network

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch--Geometric-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT%20with%20Attribution-yellow.svg)](LICENSE)
[![JCIM](https://img.shields.io/badge/Journal-JCIM%20(ACS)-005596.svg)](https://pubs.acs.org/journal/jcisd8)

---

> ## ⚠️ IMPORTANT NOTICE: INTELLECTUAL PROPERTY & CITATION REQUIREMENT
>
> **PLEASE READ BEFORE USING OR REFERENCING THIS REPOSITORY:**
>
> This repository contains novel scientific research, deep learning architectures (**FiLM-RGCN**), experimental benchmarks, and curated bioinorganic datasets prepared for publication in the *Journal of Chemical Information and Modeling (JCIM)*.
>
> **STRICT CITATION MANDATE:**
> - Any use, adaptation, re-implementation, or referencing of this code, methodology, model architecture, benchmark splits, figures, or findings in academic papers, preprints, theses, conference presentations, patents, or commercial applications **MUST explicitly cite this repository and the corresponding manuscript**:
>   ```
>   https://github.com/hubdk17/Redox_Potential
>   ```
> - **PLAGIARISM & UNCREDITED USE WARNING:** Copying, mirroring, repackaging, or claiming authorship of any part of this work without prominent attribution and citation is **strictly prohibited** and constitutes an actionable violation of scientific ethics and copyright law.
>
> See the [LICENSE](LICENSE) file for legal details.

---

## 📌 Overview

Iron–sulfur (Fe–S) metalloproteins are essential electron-transfer catalysts spanning an extraordinary $\sim 1000\text{ mV}$ physiological redox window despite sharing identical inorganic cores ($[2\text{Fe}--2\text{S}]$, $[4\text{Fe}--4\text{S}]$, $[3\text{Fe}--4\text{S}]$, $[1\text{Fe}]$). Their reduction potentials ($E_m$) are intricately tuned by subtle secondary-sphere electrostatics, hydrogen-bonding networks, dielectric shielding, and solvent pH.

**FiLM-RGCN** is an end-to-end geometric deep learning framework that:
1. Directly encodes atomic 3D protein structures into **multi-relational graphs** across **five distinct biophysical interaction types**, eliminating isotropic signal dilution.
2. Dynamically modulates intermediate structural embeddings by solvent pH via **Feature-wise Linear Modulation (FiLM)**.
3. Pools residue representations through a **dual-scale Fe–S-centered exponential radial distance readout** ($5.0\text{ \AA}$ and $12.0\text{ \AA}$ decay).
4. Demonstrates true structural generalization across non-homologous folds ($<30\%$ sequence identity), decisively outperforming tabular tree models (XGBoost).

---

## 🏗️ Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │          Atomic PDB Coordinates + pH          │
                  └───────────────────────┬───────────────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│     Multi-Relational Graph    │                       │     Solvent pH Conditioning   │
│   (5 Biophysical Edge Types)  │                       │   (z-score normalized pH)     │
└───────────────┬───────────────┘                       └───────────────┬───────────────┘
                │                                                       │
                ▼                                                       ▼
┌───────────────────────────────┐       FiLM (γ, β)     ┌───────────────────────────────┐
│   Relational Graph Conv (L=2) │ ◄─────────────────────│    Two-Layer MLP Generator    │
│       Node hidden dim = 7     │   h_film = (1+γ)⊙h+β  └───────────────────────────────┘
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                    Fe–S Centroid-Centered Dual-Scale Exponential Readout              │
│       w_inner = exp(-d / 5.0 Å)               w_outer = exp(-d / 12.0 Å)             │
│       z_inner = ∑ w_inner · h                 z_outer = ∑ w_outer · h                 │
│       z_mean  = Mean(h)                       z_max   = Max(h)                        │
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │   Global Representation:                      │
                    │   z_global = [z_inner ∥ z_outer ∥ z_mean ∥   │
                    │               z_max ∥ pH_norm]  (29-dim)      │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │   Two-Layer MLP Head  │
                                │    Predicted E_m (mV) │
                                └───────────────────────┘
```

### Five Relational Interaction Types:
- **Type 0 (Sequential Backbone):** Covalent peptide bonds ($\|\mathbf{r}_{i+1} - \mathbf{r}_i\|_2 \le 4.2\text{ \AA}$).
- **Type 1 (Spatial Contacts):** Non-bonded $\text{C}_\alpha$ proximities ($0.1\text{ \AA} < d \le 8.0\text{ \AA}$).
- **Type 2 (Polar / Hydrogen Bonds):** Amide-to-carbonyl $\text{N}\to\text{O}$ contacts ($d \le 3.5\text{ \AA}$).
- **Type 3 (Sulfur Coordination / Disulfides):** Cysteine side-chain $\text{S}_\gamma\cdots\text{S}_\gamma$ contacts ($d \le 2.5\text{ \AA}$).
- **Type 4 (Salt Bridges):** Electrostatic pairs between basic (Arg, Lys, His) and acidic (Asp, Glu) residues ($d \le 6.0\text{ \AA}$).

---

## 📊 Key Experimental Results

### 1. Generalization Across Structural and Homology Disconnect ($N = 364$)

| Cross-Validation Regime | Evaluated Model | MAE (mV) | RMSE (mV) | $R^2$ | Pearson $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random 5-Fold** *(Interpolation)* | Curated XGBoost (96 Feat.) | $\mathbf{48.39 \pm 3.98}$ | $\mathbf{70.84}$ | $\mathbf{0.917}$ | $\mathbf{0.958}$ |
| | FiLM-RGCN (Standardized) | $57.83 \pm 6.46$ | $82.75$ | $0.886$ | $0.944$ |
| | FiLM-RGCN (Optimized Grid) | $\mathbf{46.35 \pm 6.31}$ | $\mathbf{64.44}$ | $\mathbf{0.929}$ | $\mathbf{0.965}$ |
| **PDB-Grouped** *(Zero Scaffold Leakage)* | Curated XGBoost (96 Feat.) | $96.20 \pm 27.29$ | $131.66$ | $0.713$ | $0.860$ |
| | **FiLM-RGCN (Standardized)** | $\mathbf{83.82 \pm 16.38}$ | $\mathbf{113.30}$ | $\mathbf{0.787}$ | $\mathbf{0.890}$ |
| **Homology-Held-Out** *($<30\%$ Identity)* | Curated XGBoost (96 Feat.) | $151.59 \pm 25.53$ | $178.06$ | $0.474$ | $0.717$ |
| | **FiLM-RGCN (Standardized)** | $\mathbf{109.18 \pm 43.82}$ | $\mathbf{144.56}$ | $\mathbf{0.654}$ | $\mathbf{0.820}$ |

> **Key Finding:** On non-homologous folds ($<30\%$ sequence identity), XGBoost collapses to an MAE of $151.59\text{ mV}$, whereas FiLM-RGCN preserves an MAE of $109.18\text{ mV}$ (**outperforming tabular baselines by $42.4\text{ mV}$, $p = 0.0031$**).

### 2. Mutational Shift ($\Delta E_m$) & Scaffold Bootstrap ($N = 233$ Pairs, 28 Scaffolds)
- **Sample-Level Metrics:** $\text{MAE} = 60.8\text{ mV}$, $\text{RMSE} = 82.3\text{ mV}$, Pearson $r = 0.373$ ($p = 4.3 \times 10^{-9}$). Outperforms the zero-shift null baseline ($\text{RMSE} = 86.1\text{ mV}$).
- **Scaffold-Level Cluster Bootstrap (10,000 resamples):** Pearson $r = 0.282$ (95% CI: $[-0.198, 0.683]$), $\text{MAE} = 61.2\text{ mV}$ (95% CI: $[47.7, 74.9]\text{ mV}$).
- **Transparent Limitation:** Directional sign accuracy on subtle mutations ($|\Delta E_m| \le 30\text{ mV}$) is $42.5\%$, reflecting experimental protein film voltammetry measurement noise ($\pm 15\text{--}30\text{ mV}$).

### 3. Rigorous Physical Validations
- **Exact $SE(3)$ Invariance:** Discrepancy under 200 random 3D rotations/translations is $\le 6.1 \times 10^{-5}\text{ mV}$.
- **pH Titration Slopes:** Accurately reproduces non-ideal experimental slopes across multiple constructs (e.g., Rieske protein `1NYK`: exp $-56.9\text{ mV/pH}$ vs pred $-47.3\text{ mV/pH}$; `2NUK`: exp $-39.6\text{ mV/pH}$ vs pred $-38.5\text{ mV/pH}$).

---

## 📁 Repository Structure

```
├── Paper1_JCIM/                  # Full LaTeX manuscript & supporting information
│   ├── main.tex                  # Main paper source (JCIM ACS format)
│   ├── references.bib            # 42 verified peer-reviewed bibliography entries
│   ├── figures/                  # Publication-ready figures (Fig 1 to 5)
│   ├── tables/                   # LaTeX tables (Table 1, Table 2, Tables S1-S3)
│   └── supporting_information/   # Supporting information LaTeX document
├── Custom/                       # Core model development & validation scripts
│   ├── gnn_rgcn_film_enhanced.py # Complete FiLM-RGCN architecture & training engine
│   ├── phase2_homology_split_benchmark.py # Pairwise NW alignment & <30% homology splits
│   ├── phase4_to_8_causal_geometry_uncertainty.py # SE(3) invariance, ablations, MC Dropout
│   ├── run_pdb_grouped_and_mutagenesis.py # PDB-grouped CV & 233-mutant validation
│   └── generate_discussion_denovo_pdf.py # Manuscript visual suites & analysis
├── pdb-files/                    # High-resolution PDB coordinates for 76 crystal scaffolds
├── A/                            # Radii-specific tabular feature calculators
├── B/                            # Unified 96-descriptor extraction pipelines
├── Database Redox Pot Fe2S2 proteins.xlsx # Curated experimental Fe-S redox potentials
├── FINAL_EXPERIMENT_STATUS.xlsx  # Comprehensive forensic audit of all models
├── FINAL_RESEARCH_AUDIT.md       # Technical audit report & discrepancy resolutions
├── LICENSE                       # MIT License with Academic Attribution Requirement
└── README.md                     # Project documentation & citation guidelines
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/hubdk17/Redox_Potential.git
cd Redox_Potential

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric pandas numpy scikit-learn xgboost biopython openpyxl matplotlib seaborn
```

### 2. Running FiLM-RGCN 5-Fold Cross-Validation

```bash
# Run standardized 100-epoch FiLM-RGCN training
python Custom/gnn_rgcn_film_enhanced.py
```

### 3. Evaluating PDB-Grouped (Zero Scaffold Leakage) & Mutational Shifts

```bash
# Run PDB-grouped cross-validation and evaluate 233 wild-type/mutant pairs
python Custom/run_pdb_grouped_and_mutagenesis.py
```

### 4. Running the Non-Homologous ($<30\%$ Sequence Identity) Benchmark

```bash
# Compute pairwise Needleman-Wunsch alignments and run cluster cross-validation
python Custom/phase2_homology_split_benchmark.py
```

---

## 📖 Citation

If you use this codebase, models, benchmark data, or manuscript findings in your work, please cite:

```bibtex
@article{hubdk17_film_rgcn_2026,
  author    = {Author List},
  title     = {Predicting Iron--Sulfur Protein Redox Potentials Using a pH-Conditioned Relational Graph Neural Network},
  journal   = {Journal of Chemical Information and Modeling},
  year      = {2026},
  note      = {Code and Curated Data available at: \url{https://github.com/hubdk17/Redox_Potential}}
}
```

Or cite the repository directly:

```bibtex
@misc{hubdk17_redox_potential_repo,
  author       = {hubdk17},
  title        = {FiLM-RGCN: Fe--S Metalloprotein Redox Potential Prediction Framework},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/hubdk17/Redox_Potential}}
}
```

---

## ⚖️ License & Ethical Conduct

This project is licensed under the **MIT License with Mandatory Academic Attribution & Anti-Plagiarism Clause**. See [LICENSE](LICENSE) for full terms. Any unauthorized or uncredited distribution is strictly prohibited.
