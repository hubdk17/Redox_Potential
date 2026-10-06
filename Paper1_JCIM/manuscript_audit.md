# Manuscript Quality Control & Scientific Audit Checklist

**Target:** *Journal of Chemical Information and Modeling* (JCIM), ACS  
**Manuscript:** Fe–S Cluster Redox-Potential Prediction Using a pH-Conditioned Multi-Relational Graph Neural Network  
**Audit Date:** September 15, 2026  

---

## 1. Compliance Checklist

- [x] **Every numerical result traced to an output file:**
  - 46.35 ± 6.31 mV & 64.44 mV RMSE: `Custom/outputs/rgcn_film_ph_results.xlsx`
  - 57.83 ± 6.46 mV & 82.75 mV RMSE: `Custom/outputs/generalization_leakage_benchmark.xlsx`
  - 58.45 ± 6.66 mV: `Custom/outputs/targeted_ablations_summary.xlsx`
  - 83.82 ± 16.38 mV & 113.30 mV RMSE (PDB-grouped): `Custom/outputs/generalization_leakage_benchmark.xlsx`
  - 109.18 ± 43.82 mV & 144.56 mV RMSE (Homology <30%): `Custom/outputs/generalization_leakage_benchmark.xlsx`
  - Curated XGBoost 48.39 mV (Random), 96.20 mV (PDB-grouped), 151.59 mV (Homology): `Custom/outputs/generalization_leakage_benchmark.xlsx`
  - Mutational ΔEm: MAE = 54.38 mV, RMSE = 78.51 mV, r = 0.421, p = 2.4e-11, sign accuracy = 42.5%: `Custom/outputs/experiment_generalization_results.xlsx`
  - Null baseline (ΔEm = 0): RMSE = 86.13 mV: `Custom/outputs/experiment_generalization_results.xlsx`
  - Targeted ablations (No pH = 75.74 mV, Shuffled pH = 76.27 mV, No H-bond = 62.02 mV, No Disulfide = 63.21 mV, Shuffled nodes = 66.24 mV, Shuffled relations = 66.34 mV): `Custom/outputs/targeted_ablations_summary.xlsx`
  - Nuclearity breakdown ([2Fe-2S] N=293, MAE=56.18 mV; Rubredoxin N=52, MAE=47.91 mV; [4Fe-4S] N=11, MAE=131.51 mV; [3Fe-4S] N=8, MAE=110.77 mV): `Custom/outputs/subgroup_domain_shift_analysis.xlsx`
  - Geometric invariance: Max deviation <= 6.10e-5 mV across 50 rotations/translations per scaffold (4 scaffolds, 200 total transformations): `Custom/outputs/geometric_invariance_test_results.json`
  - Multi-pH series (1NYK: exp -56.9 vs pred -47.3 mV/pH; 2NUK: exp -39.6 vs pred -38.5 mV/pH): `Custom/outputs/ph_titration_mechanistic_validation.xlsx`
  - MC Dropout uncertainty: ~50% empirical coverage on nominal 95% interval across 50 iterations: `Custom/outputs/uncertainty_calibration_plot.png`
- [x] **Every figure traced to source data:**
  - Figure 1 (Workflow & Model Architecture): `Custom/outputs/film_rgcn_architecture_white.png`
  - Figure 2 (Multi-Regime Generalization Benchmark): `Custom/outputs/homology_generalization_comparison.png`
  - Figure 3 (Mechanistic Ablation Suite): `Custom/outputs/targeted_ablations_comparison.png`
  - Figure 4 (pH Titrations & Structural Displacement Counterfactuals): `Custom/outputs/ph_titration_mechanistic_validation.png` & `Custom/outputs/causal_perturbation_curves.png`
  - Figure 5 (Mutational ΔEm Validation): `Custom/outputs/mutational_delta_em_validation.png`
- [x] **Every table traced to source data:**
  - Main Table 1 (Random CV & Architecture Baseline): `Custom/outputs/generalization_leakage_benchmark.xlsx` & `Custom/outputs/rgcn_film_ph_results.xlsx`
  - Main Table 2 (Generalization Across Validation Regimes): `Custom/outputs/generalization_leakage_benchmark.xlsx`
  - Supporting Table S1 (Targeted Mechanistic Ablations): `Custom/outputs/targeted_ablations_summary.xlsx`
  - Supporting Table S2 (Nuclearity Subgroup Breakdown): `Custom/outputs/subgroup_domain_shift_analysis.xlsx`
  - Supporting Table S3 (In-Silico Mutational Shift Statistics): `Custom/outputs/experiment_generalization_results.xlsx`
- [x] **No invented data:** All 364 observations are from `Custom/features.xlsx` and crystal PDB coordinates.
- [x] **No invented citations:** Verified DOIs for Persico et al. (JCIM 2025), Schlichtkrull et al. (ESWC 2018), Perez et al. (FiLM 2018), Chen & Guestrin (XGBoost 2016), Beinert et al. (Science 1997), Stephens et al. (Chem Rev 1996).
- [x] **No unsupported mechanistic claim:** Described as "model counterfactual structural perturbations" and "mechanistically motivated structural features", never as wet-lab proof of electronic mechanism.
- [x] **No causal overclaim:** Model distance response curves are explicitly labeled as model sensitivity / counterfactual behavior, not physical causality.
- [x] **No leakage between train/test groups:**
  - PDB-grouped CV enforces strict base-PDB scaffold isolation (`GroupKFold`, 76 groups).
  - Homology-held-out CV enforces strict cross-fold sequence identity $<29.6\%$ (BioPython PairwiseAligner global alignment across all 76 scaffolds into 24 clusters).
  - Mutational validation uses out-of-fold predictions where WT and mutants were held out together.
- [x] **Random CV clearly distinguished from grouped/homology CV:** Explicitly separated into distinct subsections and distinct tables.
- [x] **46.35 and 57.83 clearly distinguished:**
  - 46.35 ± 6.31 mV: Optimized 200-epoch hyperparameter search result with per-epoch checkpointing.
  - 57.83 ± 6.46 mV: Standardized 100-epoch fixed multi-regime protocol used as the baseline anchor across regimes.
- [x] **57.83 and 58.45 clearly distinguished as stochastic/protocol replicates:**
  - Same model, same 100-epoch training loop, same 364 entries; differing by 0.62 mV due solely to random initialization and minibatch order.
- [x] **Mutation sign-accuracy limitation reported:** Documented honestly as ~42.5% (near chance) due to experimental resolution limits on near-neutral shifts ($\le 30$ mV).
- [x] **Nuclearity imbalance reported:** 80.5% [2Fe-2S] dominance clearly reported; [4Fe-4S] and [3Fe-4S] results framed as exploratory due to small sample sizes ($N=11$ and $N=8$).
- [x] **Uncertainty calibration limitation reported:** MC Dropout under-dispersion (~50% empirical coverage on nominal 95% intervals) transparently noted as a limitation.
- [x] **PDB-grouped and homology-held-out results clearly identified:** Explicitly presented with mean ± SD, $R^2$, and Pearson $r$.
- [x] **Hyperparameter-selection procedure documented:** Grid search in `Custom/gnn_rgcn_film_ph.py` documented; note on potential selection optimism under random CV discussed.
- [x] **Software versions documented:** Python 3.11, PyTorch 2.11.0, PyG (torch_geometric), BioPython 1.86, XGBoost 3.0.2, SciPy 1.15.2, scikit-learn 1.6.1, NVIDIA CUDA 12.8.
- [x] **Data availability statement prepared:** Describing curated datasets, raw PDB coordinates, trained model weights, and reproduction scripts.
- [x] **Supporting Information planned:** Complete ablation tables, fold-level metrics, nuclearity details, and full prediction catalog.
- [x] **References verified:** Verified in `Paper1_JCIM/references.bib`.
