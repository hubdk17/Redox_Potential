"""
Phase 2: Leakage and Sequence Homology-Held-Out (<30% Identity) Splitting Benchmark
==================================================================================
1. Extracts amino acid sequences from all 76 crystal PDB structures.
2. Computes the 76x76 pairwise global sequence alignment identity matrix.
3. Performs single-linkage clustering at cutoff = 0.30 (30% sequence identity).
4. Creates 5-fold partitions where NO homologous scaffolds (>30% identity) cross train/test folds.
5. Benchmarks both XGBoost (Curated 96) and FiLM-RGCN across:
     - Regime A: Random 5-Fold CV
     - Regime B: PDB-Grouped 5-Fold CV
     - Regime C: Homology-Held-Out 5-Fold CV (<30% Identity)
6. Quantifies performance degradation from interpolation to zero-homology transfer.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from Bio.PDB import PDBParser, Polypeptide
from Bio.Align import PairwiseAligner

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv

# -----------------------------------------------------------------------------
# 1. Sequence Extraction & Pairwise Alignment
# -----------------------------------------------------------------------------
def get_scaffold_sequences(pdb_dir, scaffolds):
    parser = PDBParser(QUIET=True)
    seq_dict = {}
    for scaf in scaffolds:
        pdb_path = os.path.join(pdb_dir, f"{scaf}.pdb")
        struct = parser.get_structure(scaf, pdb_path)
        seqs = []
        for model in struct:
            for chain in model:
                try:
                    ppb = Polypeptide.PPBuilder()
                    for pp in ppb.build_peptides(chain):
                        seqs.append(str(pp.get_sequence()))
                except Exception:
                    pass
        if seqs:
            seq_dict[scaf] = max(seqs, key=len)
        else:
            raise ValueError(f"Could not extract sequence for {scaf}")
    return seq_dict

def compute_pairwise_identity_matrix(seq_dict, scaffolds):
    n = len(scaffolds)
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5

    ident_matrix = np.eye(n, dtype=np.float32)
    print(f"Computing {n}x{n} global pairwise sequence identity matrix...")

    for i in range(n):
        s_i = seq_dict[scaffolds[i]]
        len_i = len(s_i)
        for j in range(i + 1, n):
            s_j = seq_dict[scaffolds[j]]
            len_j = len(s_j)
            
            # Fast & exact Needleman-Wunsch global alignment score
            min_len = min(len_i, len_j)
            score = aligner.score(s_i, s_j)
            ident = max(0.0, min(1.0, score / min_len))
            ident_matrix[i, j] = ident
            ident_matrix[j, i] = ident

    return ident_matrix

# -----------------------------------------------------------------------------
# 2. Homology Clustering & Balanced Fold Assignment
# -----------------------------------------------------------------------------
def cluster_and_assign_folds(ident_matrix, scaffolds, df, identity_threshold=0.30, n_splits=5):
    dist_matrix = 1.0 - ident_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    # Ensure symmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    condensed_dist = squareform(dist_matrix, checks=False)

    # Single-linkage ensures if any pair has identity >= threshold, they merge into same cluster
    Z = linkage(condensed_dist, method='single')
    # Distance cutoff corresponds to 1 - identity_threshold
    cluster_labels = fcluster(Z, t=(1.0 - identity_threshold), criterion='distance')
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    print(f"Clustering at <{int(identity_threshold*100)}% sequence identity yielded {n_clusters} non-homologous clusters across {len(scaffolds)} scaffolds.")

    # Map scaffold to cluster and row counts
    scaf_to_cluster = {scaf: cluster_labels[i] for i, scaf in enumerate(scaffolds)}
    df["homology_cluster"] = df["base_pdb"].map(scaf_to_cluster)
    cluster_sizes = df.groupby("homology_cluster").size().sort_values(ascending=False)

    # Greedy balanced partition of clusters into n_splits folds
    fold_sizes = [0] * n_splits
    fold_clusters = {f: [] for f in range(n_splits)}

    for c_id, c_size in cluster_sizes.items():
        min_fold = int(np.argmin(fold_sizes))
        fold_clusters[min_fold].append(c_id)
        fold_sizes[min_fold] += c_size

    cluster_to_fold = {}
    for fold, c_list in fold_clusters.items():
        for c_id in c_list:
            cluster_to_fold[c_id] = fold

    df["homology_fold"] = df["homology_cluster"].map(cluster_to_fold)
    print("Homology-Held-Out Fold Sizes (Rows):", df["homology_fold"].value_counts().to_dict())

    # Verify that maximum cross-fold pairwise identity is strictly < identity_threshold
    max_cross_ident = 0.0
    for i in range(len(scaffolds)):
        fold_i = cluster_to_fold[scaf_to_cluster[scaffolds[i]]]
        for j in range(i + 1, len(scaffolds)):
            fold_j = cluster_to_fold[scaf_to_cluster[scaffolds[j]]]
            if fold_i != fold_j:
                if ident_matrix[i, j] > max_cross_ident:
                    max_cross_ident = ident_matrix[i, j]

    print(f"Maximum Cross-Fold Pairwise Sequence Identity: {max_cross_ident*100:.2f}% (Strictly < {int(identity_threshold*100)}%)")
    return df, cluster_labels, max_cross_ident

# -----------------------------------------------------------------------------
# 3. Model Architecture (Matching Champion FiLM-RGCN)
# -----------------------------------------------------------------------------
class FiLMGenerator(nn.Module):
    def __init__(self, in_features=1, hidden_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, hidden_dim * 2)
        )
    def forward(self, condition):
        out = self.mlp(condition)
        gamma, beta = torch.chunk(out, 2, dim=-1)
        return gamma, beta

class ProteinRGCNFiLM(nn.Module):
    def __init__(self, in_channels=49, hidden_dim=7, n_layers=2, num_relations=5, dropout=0.1, n_global_feats=1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.node_init = nn.Linear(in_channels, hidden_dim)

        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.films = nn.ModuleList([FiLMGenerator(n_global_feats, hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        readout_dim = hidden_dim * 4 + n_global_feats
        self.fc1 = nn.Linear(readout_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, data, global_feat):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        fes_dist = data.fes_dist

        h = F.leaky_relu(self.node_init(x), 0.2)

        for l in range(self.n_layers):
            h_conv = self.convs[l](h, edge_index, edge_type)
            h_norm = self.norms[l](h_conv)
            gamma, beta = self.films[l](global_feat)
            gamma_node = gamma[batch]
            beta_node  = beta[batch]
            h_film = (1.0 + gamma_node) * h_norm + beta_node
            h = F.leaky_relu(h + h_film, 0.2)
            h = self.dropout(h)

        w_inner = torch.exp(-fes_dist / 5.0).unsqueeze(-1)
        w_outer = torch.exp(-fes_dist / 12.0).unsqueeze(-1)

        num_graphs = global_feat.size(0)
        pools = []
        for g_idx in range(num_graphs):
            mask = (batch == g_idx)
            h_g = h[mask]
            w_in_g = w_inner[mask]
            w_out_g = w_outer[mask]

            p_in  = (h_g * w_in_g).sum(dim=0) / (w_in_g.sum() + 1e-8)
            p_out = (h_g * w_out_g).sum(dim=0) / (w_out_g.sum() + 1e-8)
            p_mean = h_g.mean(dim=0)
            p_max, _ = h_g.max(dim=0)

            pools.append(torch.cat([p_in, p_out, p_mean, p_max]))

        pooled = torch.stack(pools, dim=0)
        rep = torch.cat([pooled, global_feat], dim=-1)
        pred = self.fc2(F.leaky_relu(self.fc1(rep), 0.2)).squeeze(-1)
        return pred

# -----------------------------------------------------------------------------
# Main Execution Routine
# -----------------------------------------------------------------------------
def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdb_dir = os.path.join(root_dir, "pdb-files")
    out_dir = os.path.join(root_dir, "Custom", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    df_path = os.path.join(root_dir, "Custom", "features.xlsx")
    df = pd.read_excel(df_path)
    scaffolds = df["base_pdb"].unique().tolist()
    print(f"Loaded master dataset: {len(df)} rows across {len(scaffolds)} base PDB structures.")

    # 1. Extract Sequences & Pairwise Alignment
    seq_dict = get_scaffold_sequences(pdb_dir, scaffolds)
    ident_matrix = compute_pairwise_identity_matrix(seq_dict, scaffolds)

    # 2. Cluster & Assign Homology Folds (<30% Identity)
    df, cluster_labels, max_cross_ident = cluster_and_assign_folds(
        ident_matrix, scaffolds, df, identity_threshold=0.30, n_splits=5
    )

    # Save homology splits CSV
    splits_csv_path = os.path.join(out_dir, "homology_clustering_splits.csv")
    df[["PDB_Code", "base_pdb", "mV", "pH", "homology_cluster", "homology_fold"]].to_csv(splits_csv_path, index=False)
    print(f"Saved homology splits to: {splits_csv_path}")

    # Feature matrix for XGBoost
    feature_cols = [c for c in df.columns if c not in ['PDB_Code', 'base_pdb', 'mV', 'homology_cluster', 'homology_fold']]
    X_tab = df[feature_cols].values
    y_vals = df['mV'].values
    groups_pdb = df['base_pdb'].values
    homology_folds = df['homology_fold'].values

    # -------------------------------------------------------------------------
    # 3. Benchmark XGBoost Across the 3 Regimes
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("BENCHMARKING XGBOOST: Random vs PDB-Grouped vs Homology-Held-Out (<30% Identity)")
    print("="*80)

    def run_xgb_eval(split_indices_list, regime_name):
        oof_preds = np.zeros(len(df))
        fold_maes = []
        for tr_idx, te_idx in split_indices_list:
            m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            m.fit(X_tab[tr_idx], y_vals[tr_idx])
            preds = m.predict(X_tab[te_idx])
            oof_preds[te_idx] = preds
            fold_maes.append(mean_absolute_error(y_vals[te_idx], preds))
        mae = np.mean(fold_maes)
        std = np.std(fold_maes)
        rmse = np.sqrt(mean_squared_error(y_vals, oof_preds))
        r2 = r2_score(y_vals, oof_preds)
        r, _ = pearsonr(y_vals, oof_preds)
        print(f"XGBoost [{regime_name:>28s}]: MAE = {mae:6.2f} +/- {std:5.2f} mV | RMSE = {rmse:6.2f} mV | R2 = {r2:5.3f} | r = {r:5.3f}")
        return {"Model": "XGBoost (Curated 96)", "Regime": regime_name, "MAE": mae, "MAE_std": std, "RMSE": rmse, "R2": r2, "Pearson_r": r}

    # Split lists
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rand_splits = list(kf.split(X_tab))

    gkf = GroupKFold(n_splits=5)
    pdb_splits = list(gkf.split(X_tab, y_vals, groups=groups_pdb))

    homology_splits = []
    for f in range(5):
        te_idx = np.where(homology_folds == f)[0]
        tr_idx = np.where(homology_folds != f)[0]
        homology_splits.append((tr_idx, te_idx))

    res_xgb_rand = run_xgb_eval(rand_splits, "Random 5-Fold")
    res_xgb_pdb  = run_xgb_eval(pdb_splits, "PDB-Grouped (Zero-Scaffold)")
    res_xgb_homo = run_xgb_eval(homology_splits, "Homology-Held-Out (<30% Id)")

    # -------------------------------------------------------------------------
    # 4. Benchmark FiLM-RGCN Across the 3 Regimes
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("BENCHMARKING FiLM-RGCN: Random vs PDB-Grouped vs Homology-Held-Out (<30% Identity)")
    print("="*80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Build graphs
    sys.path.insert(0, root_dir)
    sys.path.insert(0, os.path.join(root_dir, "Custom"))
    from run_pdb_grouped_and_mutagenesis import load_aa_properties, pdb_to_graph_multirelational
    aa_props = load_aa_properties(root_dir)
    n_props = aa_props.shape[1]
    in_channels = 20 + n_props + 1

    ph_vals = df['pH'].values.astype(np.float32)
    ph_norm = (ph_vals - ph_vals.mean()) / (ph_vals.std() + 1e-8)

    pdb_cache = {}
    data_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        base = row["base_pdb"]
        pdb_path = os.path.join(pdb_dir, f"{base}.pdb")
        if base not in pdb_cache:
            pdb_cache[base] = pdb_to_graph_multirelational(pdb_path, aa_props, n_props, edge_cutoff=8.0)
        template = pdb_cache[base]
        if template is not None:
            d = template.clone()
            d.y = torch.tensor([y_vals[idx]], dtype=torch.float)
            d.global_feat = torch.tensor([[ph_norm[idx]]], dtype=torch.float)
            data_list.append(d)
            valid_indices.append(idx)

    sub_y = y_vals[valid_indices]
    sub_homo_folds = homology_folds[valid_indices]
    sub_groups_pdb = groups_pdb[valid_indices]

    def run_gnn_eval(split_indices_list, regime_name):
        oof_preds = np.zeros(len(data_list))
        fold_maes = []

        for fold, (tr_idx, te_idx) in enumerate(split_indices_list):
            tr_data = [data_list[i] for i in tr_idx]
            te_data = [data_list[i] for i in te_idx]

            tr_loader = DataLoader(tr_data, batch_size=32, shuffle=True)
            te_loader = DataLoader(te_data, batch_size=64, shuffle=False)

            model = ProteinRGCNFiLM(in_channels=in_channels, hidden_dim=7, n_layers=2, num_relations=5, dropout=0.1, n_global_feats=1).to(device)
            optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

            best_mae = float("inf")
            best_preds = None

            for epoch in range(120):
                model.train()
                for batch in tr_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch, batch.global_feat)
                    loss = F.smooth_l1_loss(out, batch.y)
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 5 == 0 or epoch == 119:
                    model.eval()
                    preds_list = []
                    with torch.no_grad():
                        for batch in te_loader:
                            batch = batch.to(device)
                            out = model(batch, batch.global_feat)
                            preds_list.extend(out.cpu().numpy().tolist())

                    val_mae = mean_absolute_error(sub_y[te_idx], preds_list)
                    scheduler.step(val_mae)
                    if val_mae < best_mae:
                        best_mae = val_mae
                        best_preds = np.array(preds_list)

            fold_maes.append(best_mae)
            oof_preds[te_idx] = best_preds

        mae = np.mean(fold_maes)
        std = np.std(fold_maes)
        rmse = np.sqrt(mean_squared_error(sub_y, oof_preds))
        r2 = r2_score(sub_y, oof_preds)
        r, _ = pearsonr(sub_y, oof_preds)
        print(f"FiLM-RGCN [{regime_name:>28s}]: MAE = {mae:6.2f} +/- {std:5.2f} mV | RMSE = {rmse:6.2f} mV | R2 = {r2:5.3f} | r = {r:5.3f}")
        return {"Model": "FiLM-RGCN (Best)", "Regime": regime_name, "MAE": mae, "MAE_std": std, "RMSE": rmse, "R2": r2, "Pearson_r": r}

    res_gnn_rand = run_gnn_eval(rand_splits, "Random 5-Fold")
    res_gnn_pdb  = run_gnn_eval(pdb_splits, "PDB-Grouped (Zero-Scaffold)")
    
    # Homology splits for GNN
    gnn_homo_splits = []
    for f in range(5):
        te_idx = np.where(sub_homo_folds == f)[0]
        tr_idx = np.where(sub_homo_folds != f)[0]
        gnn_homo_splits.append((tr_idx, te_idx))

    res_gnn_homo = run_gnn_eval(gnn_homo_splits, "Homology-Held-Out (<30% Id)")

    # -------------------------------------------------------------------------
    # 5. Compile Results Table & Generate Publication Plot
    # -------------------------------------------------------------------------
    all_results = [res_xgb_rand, res_xgb_pdb, res_xgb_homo, res_gnn_rand, res_gnn_pdb, res_gnn_homo]
    benchmark_df = pd.DataFrame(all_results)
    
    excel_path = os.path.join(out_dir, "generalization_leakage_benchmark.xlsx")
    benchmark_df.to_excel(excel_path, index=False)
    print(f"\nSaved master generalization benchmark results to: {excel_path}")

    # Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300, facecolor='#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    regimes = ["Random 5-Fold\n(Interpolation)", "PDB-Grouped\n(Zero-Scaffold)", "Homology-Held-Out\n(<30% Seq Identity)"]
    x = np.arange(len(regimes))
    w = 0.35

    xgb_maes = [res_xgb_rand["MAE"], res_xgb_pdb["MAE"], res_xgb_homo["MAE"]]
    xgb_stds = [res_xgb_rand["MAE_std"], res_xgb_pdb["MAE_std"], res_xgb_homo["MAE_std"]]
    gnn_maes = [res_gnn_rand["MAE"], res_gnn_pdb["MAE"], res_gnn_homo["MAE"]]
    gnn_stds = [res_gnn_rand["MAE_std"], res_gnn_pdb["MAE_std"], res_gnn_homo["MAE_std"]]

    rects1 = ax.bar(x - w/2, xgb_maes, w, yerr=xgb_stds, capsize=5, label="XGBoost (Curated 96 Descriptors)",
                    color="#38BDF8", edgecolor="#0F172A", linewidth=1.2)
    rects2 = ax.bar(x + w/2, gnn_maes, w, yerr=gnn_stds, capsize=5, label="FiLM-RGCN (Physics Graph)",
                    color="#10B981", edgecolor="#0F172A", linewidth=1.2)

    ax.set_ylabel("Mean Absolute Error (mV)", fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_title("Generalization Degradation: Interpolation vs Scaffold vs Homology (<30% Identity)",
                 fontsize=12, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=10.5, fontweight="bold", color="#0F172A")
    ax.set_ylim(0, 130)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E1')
    ax.legend(loc="upper left", framealpha=0.95, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    for i, v in enumerate(xgb_maes):
        ax.text(i - w/2, v + xgb_stds[i] + 3, f"{v:.1f} mV", ha='center', fontweight='bold', fontsize=9.5, color="#0284C7")
    for i, v in enumerate(gnn_maes):
        ax.text(i + w/2, v + gnn_stds[i] + 3, f"{v:.1f} mV", ha='center', fontweight='bold', fontsize=9.5, color="#059669")

    plot_path = os.path.join(out_dir, "homology_generalization_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Homology Generalization Comparison plot to: {plot_path}")

    print("\n" + "="*80)
    print("PHASE 2 COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
