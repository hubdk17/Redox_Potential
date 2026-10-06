"""
Master Reproducibility Script: Fe-S Metalloprotein Redox GNN Benchmark
========================================================================
This master script reproduces all core scientific audits, benchmark tables,
and publication figures for:
  "Physics-Conditioned Graph Representation Learning for Fe–S Metalloprotein
   Redox Potential Prediction"

Usage:
  python Custom/reproduce_all.py --all           # Run all analyses sequentially
  python Custom/reproduce_all.py --audit         # Run dataset forensic audit
  python Custom/reproduce_all.py --homology      # Run homology splitting benchmark
  python Custom/reproduce_all.py --mutagenesis   # Run mutational validation audit
  python Custom/reproduce_all.py --causal        # Run causal physical perturbations & pH validation
  python Custom/reproduce_all.py --figures       # Regenerate all publication figures
"""

import os
import sys
import argparse
import subprocess

def run_command(cmd, desc):
    print("\n" + "="*80)
    print(f"EXECUTING: {desc}")
    print(f"Command: {cmd}")
    print("="*80)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"WARNING: Command exited with return code {ret}")
    else:
        print(f"SUCCESS: {desc} completed successfully.")
    return ret

def main():
    parser = argparse.ArgumentParser(description="Master Reproducibility Runner for Fe-S GNN Benchmark")
    parser.add_argument("--all", action="store_true", help="Execute complete end-to-end benchmark suite")
    parser.add_argument("--audit", action="store_true", help="Run Phase 1 Dataset Forensic Audit")
    parser.add_argument("--homology", action="store_true", help="Run Phase 2 Homology-Held-Out Benchmark")
    parser.add_argument("--mutagenesis", action="store_true", help="Run Phase 3 Mutational Validation Audit")
    parser.add_argument("--causal", action="store_true", help="Run Phase 4-8 Causal, pH, & Geometric Invariance Audits")
    parser.add_argument("--figures", action="store_true", help="Regenerate all publication figures")
    args = parser.parse_args()

    python_bin = sys.executable
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not any([args.all, args.audit, args.homology, args.mutagenesis, args.causal, args.figures]):
        parser.print_help()
        print("\nNo flags specified. Running quick summary check...")
        args.audit = True

    if args.all or args.audit:
        script = os.path.join(root_dir, "Custom", "phase1_dataset_audit.py")
        run_command(f'"{python_bin}" "{script}"', "Phase 1: Dataset Forensic Audit")

    if args.all or args.homology:
        script = os.path.join(root_dir, "Custom", "phase2_homology_split_benchmark.py")
        run_command(f'"{python_bin}" "{script}"', "Phase 2: Homology-Held-Out Benchmark (<30% Id)")

    if args.all or args.mutagenesis:
        script = os.path.join(root_dir, "Custom", "audit_mutagenesis_experiment.py")
        run_command(f'"{python_bin}" "{script}"', "Phase 3: Mutational Validation Audit")

    if args.all or args.causal:
        script = os.path.join(root_dir, "Custom", "phase4_to_8_causal_geometry_uncertainty.py")
        run_command(f'"{python_bin}" "{script}"', "Phases 4-8: Causal Perturbations & Geometric Rigor")

    if args.all or args.figures:
        script = os.path.join(root_dir, "Custom", "generate_top5_models_photo.py")
        run_command(f'"{python_bin}" "{script}"', "Regenerate Benchmark Comparison Figures")

    print("\n" + "="*80)
    print("ALL REPRODUCIBILITY CHECKS FINISHED.")
    print("Master Audit Summary: FINAL_RESEARCH_AUDIT.md")
    print("="*80)

if __name__ == "__main__":
    main()
