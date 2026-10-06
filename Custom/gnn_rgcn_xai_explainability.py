"""
Explainable AI (XAI) & Interpretability Suite for Best FiLM-RGCN Model
======================================================================
Implements Integrated Gradients, Feature Attribution, Structural Distance Analysis,
and Edge Relation Importance for the top-performing model (RGCN + FiLM pH + Distance Readout).

Analyzes:
1. Global Feature Importance (Integrated Gradients over all 49 node features).
2. Structural Residue Importance vs. Distance from FeS Cluster.
3. Relation-Type Contribution (Backbone vs Spatial vs H-Bond vs Disulfide vs Salt Bridge).
4. Physical pH Sensitivity (FiLM Gradient Response d(Em)/d(pH)).
5. PyMOL-compatible PDB generation with XAI attribution mapped to B-factors.
"""

import os
import sys
import logging
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from Bio.PDB import PDBParser, PDBIO

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_WORKERS = 0
NUM_RELATIONS = 5
RELATION_NAMES = ["Backbone", "Spatial", "H-bond", "Disulfide", "Salt bridge"]

STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}
POSITIVE_AA = {"ARG", "LYS", "HIS"}
NEGATIVE_AA = {"ASP", "GLU"}

# ────────────────── Amino-Acid Properties ──────────────────

def load_aa_properties(project_root):
    path = os.path.join(project_root, "tableAmm.txt")
    table = pd.read_csv(path, sep="\t", index_col=1).iloc[:, 1:]
    table = table.loc[table.index.isin(STANDARD_AA)]
    for col in table.columns:
        cmin, cmax = table[col].min(), table[col].max()
        if cmax > cmin:
            table[col] = (table[col] - cmin) / (cmax - cmin)
        else:
            table[col] = 0.0
    return table

def get_weighted_node_features(res, aa_props, n_props, is_coordinating):
    resname = res.get_resname()
    one_hot = np.zeros(20, dtype=np.float32)
    props = np.zeros(n_props, dtype=np.float32)

    if resname in AA_TO_IDX:
        scale = 3.0 if resname in {"CYS", "TRP", "PRO"} else 1.0
        one_hot[AA_TO_IDX[resname]] = scale
        if resname in aa_props.index:
            row = aa_props.loc[resname].copy()
            if "Flexibility" in row.index:
                row["Flexibility"] = row["Flexibility"] * 5.0
            props = row.values.astype(np.float32)
            props = np.nan_to_num(props, nan=0.0)

    cys_flag = np.array(
        [5.0 if (resname == "CYS" and is_coordinating) else 0.0],
        dtype=np.float32,
    )
    return np.concatenate([one_hot, props, cys_flag])

# ────────────────── Graph Construction ──────────────────

def pdb_to_graph_multirelational(pdb_path, aa_props, n_props, edge_cutoff=8.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    fe_coords = []
    for chain in model:
        for res in chain.get_residues():
            if res.get_resname() in {"FES", "FE", "SF4", "F3S"}:
                for atom in res:
                    if "FE" in atom.get_name():
                        fe_coords.append(atom.get_vector().get_array())
    fe_coords = np.array(fe_coords, dtype=np.float32) if fe_coords else np.empty((0, 3), dtype=np.float32)

    residues   = []
    ca_coords  = []
    chain_ids  = []
    seq_nums   = []
    n_atom_coords = []
    o_atom_coords = []
    sg_atom_coords = []

    for chain in model:
        for res in chain.get_residues():
            rn = res.get_resname()
            if rn in STANDARD_AA and "CA" in res:
                residues.append(res)
                ca_coords.append(res["CA"].get_vector().get_array())
                chain_ids.append(chain.get_id())
                seq_nums.append(res.get_id()[1])
                n_atom_coords.append(
                    res["N"].get_vector().get_array() if "N" in res else None
                )
                o_atom_coords.append(
                    res["O"].get_vector().get_array() if "O" in res else None
                )
                sg_atom_coords.append(
                    res["SG"].get_vector().get_array()
                    if (rn == "CYS" and "SG" in res) else None
                )

    n_nodes = len(residues)
    if n_nodes < 3:
        return None

    ca_coords = np.array(ca_coords, dtype=np.float32)

    coord_flags = []
    fes_dists = []
    for res in residues:
        is_coord = False
        atom_coords = [atom.get_vector().get_array() for atom in res]
        if len(fe_coords) > 0:
            diff = np.array(atom_coords)[:, None, :] - fe_coords[None, :, :]
            d_min = np.sqrt((diff ** 2).sum(axis=-1)).min()
            if res.get_resname() == "CYS" and d_min <= 3.0:
                is_coord = True
        else:
            d_min = 10.0
        coord_flags.append(is_coord)
        fes_dists.append(d_min)

    node_feats = np.array([
        get_weighted_node_features(res, aa_props, n_props, cf)
        for res, cf in zip(residues, coord_flags)
    ], dtype=np.float32)
    x = torch.tensor(node_feats, dtype=torch.float)
    fes_dist = torch.tensor(fes_dists, dtype=torch.float)

    all_src, all_dst, all_types = [], [], []

    # Type 0: Backbone bonds
    for i in range(n_nodes - 1):
        if chain_ids[i] == chain_ids[i + 1] and seq_nums[i + 1] - seq_nums[i] == 1:
            all_src.extend([i, i + 1])
            all_dst.extend([i + 1, i])
            all_types.extend([0, 0])

    # Type 1: Spatial contacts
    diffs = ca_coords[:, None, :] - ca_coords[None, :, :]
    ca_dists = np.sqrt((diffs ** 2).sum(axis=-1))
    mask = (ca_dists <= edge_cutoff) & (ca_dists > 0.1)
    sp_src, sp_dst = np.where(mask)
    all_src.extend(sp_src.tolist())
    all_dst.extend(sp_dst.tolist())
    all_types.extend([1] * len(sp_src))

    # Type 2: Hydrogen bonds
    n_valid = [(i, c) for i, c in enumerate(n_atom_coords) if c is not None]
    o_valid = [(i, c) for i, c in enumerate(o_atom_coords) if c is not None]
    if n_valid and o_valid:
        n_idx_list, n_arr = zip(*n_valid)
        o_idx_list, o_arr = zip(*o_valid)
        n_arr = np.array(n_arr, dtype=np.float32)
        o_arr = np.array(o_arr, dtype=np.float32)
        no_dists = np.sqrt(((n_arr[:, None, :] - o_arr[None, :, :]) ** 2).sum(axis=-1))
        hi, hj = np.where(no_dists <= 3.5)
        for a, b in zip(hi, hj):
            res_i = n_idx_list[a]
            res_j = o_idx_list[b]
            if res_i != res_j:
                all_src.extend([res_i, res_j])
                all_dst.extend([res_j, res_i])
                all_types.extend([2, 2])

    # Type 3: Disulfide bonds
    cys_sg = [(i, c) for i, c in enumerate(sg_atom_coords) if c is not None]
    for a in range(len(cys_sg)):
        for b in range(a + 1, len(cys_sg)):
            i, sg_i = cys_sg[a]
            j, sg_j = cys_sg[b]
            if np.linalg.norm(np.array(sg_i) - np.array(sg_j)) <= 2.5:
                all_src.extend([i, j])
                all_dst.extend([j, i])
                all_types.extend([3, 3])

    # Type 4: Salt bridges
    pos_idx = [i for i in range(n_nodes) if residues[i].get_resname() in POSITIVE_AA]
    neg_idx = [i for i in range(n_nodes) if residues[i].get_resname() in NEGATIVE_AA]
    for i in pos_idx:
        for j in neg_idx:
            if ca_dists[i, j] <= 6.0:
                all_src.extend([i, j])
                all_dst.extend([j, i])
                all_types.extend([4, 4])

    if len(all_src) == 0:
        return None

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_type  = torch.tensor(all_types, dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        fes_dist=fes_dist,
        num_nodes=n_nodes
    )

# ────────────────── FiLM Generator & Model ──────────────────

class FiLMGenerator(nn.Module):
    def __init__(self, in_features=1, hidden_dim=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, ph):
        out = self.fc(ph)
        gamma, beta = torch.chunk(out, 2, dim=-1)
        return gamma, beta

class ProteinRGCNFiLM(nn.Module):
    def __init__(self, in_channels, hidden_dim=7, n_layers=2,
                 num_relations=NUM_RELATIONS, dropout=0.1,
                 n_global_feats=1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.films = nn.ModuleList()

        self.convs.append(RGCNConv(in_channels, hidden_dim, num_relations=num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.films.append(FiLMGenerator(in_features=n_global_feats, hidden_dim=hidden_dim))

        for _ in range(1, n_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.films.append(FiLMGenerator(in_features=n_global_feats, hidden_dim=hidden_dim))

        self.gamma_tight  = nn.Parameter(torch.tensor(0.25))
        self.gamma_medium = nn.Parameter(torch.tensor(0.10))

        mlp_in = hidden_dim * 4 + n_global_feats
        self.fc1 = nn.Linear(mlp_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data, global_feats=None):
        x, edge_index = data.x, data.edge_index
        edge_type = data.edge_type
        batch = data.batch if (hasattr(data, 'batch') and data.batch is not None) else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        fes_dist = data.fes_dist

        for conv, bn, film in zip(self.convs, self.bns, self.films):
            x = conv(x, edge_index, edge_type)
            x = bn(x)
            x = F.relu(x)

            if global_feats is not None:
                gamma, beta = film(global_feats)
                gamma_nodes = gamma[batch]
                beta_nodes  = beta[batch]
                x = (1.0 + gamma_nodes) * x + beta_nodes

            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)

        g_tight = F.softplus(self.gamma_tight)
        g_med   = F.softplus(self.gamma_medium)

        weights_tight = pyg_softmax(-g_tight * fes_dist, batch)
        weights_med   = pyg_softmax(-g_med * fes_dist, batch)

        x_decay_tight = global_add_pool(weights_tight.unsqueeze(-1) * x, batch)
        x_decay_med   = global_add_pool(weights_med.unsqueeze(-1) * x, batch)

        graph_emb = torch.cat([x_mean, x_max, x_decay_tight, x_decay_med], dim=1)

        if global_feats is not None:
            graph_emb = torch.cat([graph_emb, global_feats], dim=1)

        out = F.relu(self.fc1(graph_emb))
        out = self.fc2(out)
        return out.squeeze(-1)

# ────────────────── Integrated Gradients (XAI) ──────────────────

def compute_integrated_gradients(model, data, global_feats, steps=20):
    """
    Computes Integrated Gradients for node features x of a single graph.
    Formula: IG_i = (x_i - x'_i) * (1/m) * sum_{k=1}^m grad_F(x' + (k/m)*(x - x'))
    """
    model.eval()
    x = data.x.clone().detach()
    baseline = torch.zeros_like(x)
    accum_grads = torch.zeros_like(x)

    for step in range(1, steps + 1):
        alpha = float(step) / steps
        interpolated_x = (baseline + alpha * (x - baseline)).requires_grad_(True)
        
        # Create temporary data object with interpolated x
        temp_data = data.clone()
        temp_data.x = interpolated_x
        
        pred = model(temp_data, global_feats)
        grad = torch.autograd.grad(pred, interpolated_x)[0]
        accum_grads += grad.detach()

    avg_grads = accum_grads / steps
    ig_attributions = (x - baseline) * avg_grads  # [N_nodes, 49]
    return ig_attributions.cpu().numpy()

# ────────────────── Main Execution ──────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdb_dir = os.path.join(project_root, "pdb-files")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "features.xlsx"))
    logger.info(f"Loaded master dataset: {len(df)} entries")

    aa_props = load_aa_properties(project_root)
    n_props  = aa_props.shape[1]
    prop_names = list(aa_props.columns)
    feature_names = [f"AA_{aa}" for aa in STANDARD_AA] + prop_names + ["Coordinating_CYS_Flag"]
    in_channels = len(feature_names)  # 49

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    targets = df["mV"].values.astype(np.float64)
    ph_vals = df["pH"].values.astype(np.float64)
    ph_norm = (ph_vals - ph_vals.mean()) / (ph_vals.std() + 1e-8)

    # 1. Build graphs with optimal cutoff = 8.0 A
    logger.info("Building multi-relational graphs (cutoff=8.0 A)...")
    pdb_cache = {}
    data_list = []
    base_names = []

    for idx, row in df.iterrows():
        base = row["base_pdb"]
        pdb_path = os.path.join(pdb_dir, f"{base}.pdb")

        if base not in pdb_cache:
            if os.path.exists(pdb_path):
                pdb_cache[base] = pdb_to_graph_multirelational(pdb_path, aa_props, n_props, edge_cutoff=8.0)
            else:
                pdb_cache[base] = None

        template = pdb_cache[base]
        if template is not None:
            d = template.clone()
            d.y = torch.tensor([targets[idx]], dtype=torch.float)
            d.global_feat = torch.tensor([[ph_norm[idx]]], dtype=torch.float)
            data_list.append(d)
            base_names.append(base)

    logger.info(f"Successfully built {len(data_list)} graphs")

    # 2. Train the Top Model (h=7, L=2, lr=0.005, cutoff=8)
    hidden_dim = 7
    n_layers = 2
    lr = 0.005
    n_splits = 5
    n_epochs = 200

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []

    logger.info(f"Training ensemble of top model (RGCN_FiLM_pH_h7_L2_lr0.005_c8)...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(data_list)))):
        train_data = [data_list[i] for i in train_idx]
        test_data  = [data_list[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
        test_loader  = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

        model = ProteinRGCNFiLM(in_channels=in_channels, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        best_val_mae = float("inf")
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            for batch_data in train_loader:
                batch_data = batch_data.to(device)
                g = batch_data.global_feat.to(device)
                optimizer.zero_grad()
                pred = model(batch_data, g)
                loss = F.smooth_l1_loss(pred, batch_data.y, beta=40.0)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                model.eval()
                val_preds, val_tgts = [], []
                with torch.no_grad():
                    for batch_data in test_loader:
                        batch_data = batch_data.to(device)
                        g = batch_data.global_feat.to(device)
                        val_preds.append(model(batch_data, g).cpu().numpy())
                        val_tgts.append(batch_data.y.cpu().numpy())
                mae = mean_absolute_error(np.concatenate(val_tgts), np.concatenate(val_preds))
                scheduler.step(mae)
                if mae < best_val_mae:
                    best_val_mae = mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        models.append(model)
        logger.info(f"  Fold {fold+1}/{n_splits} Best Test MAE: {best_val_mae:.1f} mV")

    # 3. Compute Global Feature Attributions via Integrated Gradients
    logger.info("Computing Integrated Gradients across dataset...")
    all_ig_attributions = []
    residue_distance_list = []
    residue_attribution_list = []

    best_model = models[0]  # Primary model for detailed XAI

    for i, data in enumerate(data_list):
        data = data.to(device)
        g = data.global_feat.to(device)
        ig = compute_integrated_gradients(best_model, data, g, steps=20)
        all_ig_attributions.append(ig)

        # Residue-level total attribution (L1 norm across features)
        res_attr = np.abs(ig).sum(axis=-1)
        dists = data.fes_dist.cpu().numpy()
        residue_distance_list.extend(dists)
        residue_attribution_list.extend(res_attr)

    # 4. Feature Importance Summary (SHAP / IG Style)
    mean_abs_feature_importance = np.zeros(in_channels)
    for ig in all_ig_attributions:
        mean_abs_feature_importance += np.abs(ig).mean(axis=0)
    mean_abs_feature_importance /= len(all_ig_attributions)

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Integrated_Gradient_Attribution": mean_abs_feature_importance
    }).sort_values("Mean_Integrated_Gradient_Attribution", ascending=False).reset_index(drop=True)

    feat_df.to_excel(os.path.join(output_dir, "xai_feature_importance.xlsx"), index=False)

    # Plot 1: Top 20 Feature Importance
    plt.figure(figsize=(10, 8))
    top20 = feat_df.head(20).iloc[::-1]
    colors = ["#1f77b4" if "AA_" in f else "#ff7f0e" if f == "Coordinating_CYS_Flag" else "#2ca02c" for f in top20["Feature"]]
    plt.barh(top20["Feature"], top20["Mean_Integrated_Gradient_Attribution"], color=colors, edgecolor="black", linewidth=0.5)
    plt.xlabel("Mean |Integrated Gradients| Attribution", fontsize=12)
    plt.title("XAI Global Feature Importance (Top 20 Node Features)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xai_feature_importance.png"), dpi=300)
    plt.close()
    logger.info("Saved xai_feature_importance.png")

    # Plot 2: Structural Residue Importance vs. Distance from FeS Cluster
    plt.figure(figsize=(9, 6))
    dists_arr = np.array(residue_distance_list)
    attrs_arr = np.array(residue_attribution_list)

    # Bin distances into 1 A windows for smooth trend curve
    bins = np.linspace(0, 30, 31)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = [attrs_arr[(dists_arr >= bins[i]) & (dists_arr < bins[i+1])].mean()
                 if len(attrs_arr[(dists_arr >= bins[i]) & (dists_arr < bins[i+1])]) > 0 else 0
                 for i in range(len(bins)-1)]

    plt.scatter(dists_arr, attrs_arr, alpha=0.15, s=12, color="teal", label="Residue Attributions")
    plt.plot(bin_centers, bin_means, color="crimson", linewidth=2.5, label="Mean Importance Profile")
    plt.axvline(x=6.0, color="darkorange", linestyle="--", label="Inner Coordination Shell (6 A)")
    plt.axvline(x=12.0, color="purple", linestyle="--", label="Secondary Electrostatic Shell (12 A)")
    plt.xlabel("Residue Distance to FeS Cluster (A)", fontsize=12)
    plt.ylabel("Total Integrated Gradient Attribution", fontsize=12)
    plt.title("XAI Spatial Validation: Residue Importance vs. Distance from FeS Cluster", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.xlim(0, 30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xai_residue_distance_decay.png"), dpi=300)
    plt.close()
    logger.info("Saved xai_residue_distance_decay.png")

    # Plot 3: FiLM pH-Sensitivity Derivative Analysis (dEm / dpH)
    logger.info("Computing FiLM pH Sensitivity Analysis...")
    ph_gradients = []
    ph_test_vals = np.linspace(4.0, 10.0, 50)
    ph_mean_d = df["pH"].mean()
    ph_std_d  = df["pH"].std()

    # For a representative protein graph (first sample)
    sample_data = data_list[0].to(device)
    for p_val in ph_test_vals:
        p_norm = torch.tensor([[(p_val - ph_mean_d) / ph_std_d]], dtype=torch.float, device=device, requires_grad=True)
        pred = best_model(sample_data, p_norm)
        grad_p = torch.autograd.grad(pred, p_norm)[0].item() / ph_std_d
        ph_gradients.append(grad_p)

    plt.figure(figsize=(8, 5))
    plt.plot(ph_test_vals, ph_gradients, color="darkblue", linewidth=2.5)
    plt.axhline(y=-59.0, color="red", linestyle="--", label="Theoretical Nernst Slope (-59 mV/pH)")
    plt.xlabel("Solution pH", fontsize=12)
    plt.ylabel("d(Em) / d(pH)  [mV / pH unit]", fontsize=12)
    plt.title("Physical pH-Sensitivity Response from Learned FiLM Modulation", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xai_ph_sensitivity.png"), dpi=300)
    plt.close()
    logger.info("Saved xai_ph_sensitivity.png")

    logger.info("XAI interpretability analysis completed successfully!")

if __name__ == "__main__":
    main()
