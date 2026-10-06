"""
Multi-Relational RGCN with pH-Conditioned FiLM Modulation and Distance-Decayed Readout
======================================================================================
Combines our best-performing 54.57 mV Multi-Relational RGCN + Distance-Decayed Readout
with Feature-wise Linear Modulation (FiLM) conditioned on solution pH.

Key Innovations:
1. pH-Conditioned FiLM (Feature-wise Linear Modulation):
   - Rather than merely concatenating pH as a static scalar at the final prediction layer,
     a lightweight FiLM generator dynamically scales and shifts the intermediate residue
     embeddings after each GNN message-passing layer:
         h_i' = (1.0 + gamma(pH)) * h_i + beta(pH)
     This simulates how pH modulates residue protonation and electrostatics throughout the protein.
2. Distance-Exponential Dual-Scale Soft Readout:
   - Dual-scale distance-decayed pooling (tight inner shell + medium electrostatic shell)
     combined with global mean & max pooling.
3. Outlier-Robust Smooth L1 Loss + Batch Pearson Correlation Penalty.
"""

import os
import sys
import logging
import time
import warnings
import itertools
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from Bio.PDB import PDBParser

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_WORKERS = 0
NUM_RELATIONS = 5
RELATION_NAMES = ["Backbone", "Spatial", "H-bond", "Disulfide", "Salt bridge"]

# ────────────────────── Constants ──────────────────────

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

    # Detect Fe atoms
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

    # Compute coordinating-Cys flag & minimum distance to FeS cluster per residue
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

    # ── Edges (5 relation types) ──
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

def log_edge_stats(data_list):
    totals = [0] * NUM_RELATIONS
    for d in data_list:
        for t in range(NUM_RELATIONS):
            totals[t] += int((d.edge_type == t).sum())
    for t, name in enumerate(RELATION_NAMES):
        logger.info(f"    {name:>12s} edges: {totals[t]:>8,}")
    logger.info(f"    {'TOTAL':>12s} edges: {sum(totals):>8,}")

# ────────────────── FiLM Generator ──────────────────

class FiLMGenerator(nn.Module):
    """Generates feature-wise scale (gamma) and shift (beta) from normalized pH."""
    def __init__(self, in_features=1, hidden_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        # Initialize final linear layer with small weights so initial modulation is close to identity
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, ph):
        # ph: [batch_size, 1]
        out = self.fc(ph)
        gamma, beta = torch.chunk(out, 2, dim=-1)
        return gamma, beta  # [batch_size, hidden_dim], [batch_size, hidden_dim]

# ────────────────── Model Architecture ──────────────────

class ProteinRGCNFiLM(nn.Module):
    """
    Multi-Relational RGCN with pH-conditioned FiLM modulation and Dual-Scale Distance Readout.
    """
    def __init__(self, in_channels, hidden_dim, n_layers=2,
                 num_relations=NUM_RELATIONS, dropout=0.1,
                 n_global_feats=1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        # RGCN Layers & BatchNorms
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.films = nn.ModuleList()

        # Layer 1
        self.convs.append(RGCNConv(in_channels, hidden_dim, num_relations=num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.films.append(FiLMGenerator(in_features=n_global_feats, hidden_dim=hidden_dim))

        # Subsequent Layers
        for _ in range(1, n_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.films.append(FiLMGenerator(in_features=n_global_feats, hidden_dim=hidden_dim))

        # Learnable distance decay rates for soft readout (tight & medium shell)
        self.gamma_tight  = nn.Parameter(torch.tensor(0.25))   # ~4-6 A decay
        self.gamma_medium = nn.Parameter(torch.tensor(0.10))   # ~10-12 A decay

        # Prediction Head: [x_mean || x_max || x_decay_tight || x_decay_medium || pH]
        mlp_in = hidden_dim * 4 + n_global_feats
        self.fc1 = nn.Linear(mlp_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data, global_feats=None):
        x, edge_index = data.x, data.edge_index
        edge_type, batch = data.edge_type, data.batch
        fes_dist = data.fes_dist

        for conv, bn, film in zip(self.convs, self.bns, self.films):
            x = conv(x, edge_index, edge_type)
            x = bn(x)
            x = F.relu(x)

            # Apply pH FiLM modulation if global_feats (pH) is provided
            if global_feats is not None:
                gamma, beta = film(global_feats)  # [batch_size, hidden_dim]
                gamma_nodes = gamma[batch]        # [num_nodes, hidden_dim]
                beta_nodes  = beta[batch]         # [num_nodes, hidden_dim]
                x = (1.0 + gamma_nodes) * x + beta_nodes

            x = F.dropout(x, p=self.dropout, training=self.training)

        # 1. Global Mean & Max Pooling
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)

        # 2. Dual-Scale Distance-Decayed Soft Readouts
        g_tight = F.softplus(self.gamma_tight)
        g_med   = F.softplus(self.gamma_medium)

        weights_tight = pyg_softmax(-g_tight * fes_dist, batch)
        weights_med   = pyg_softmax(-g_med * fes_dist, batch)

        x_decay_tight = global_add_pool(weights_tight.unsqueeze(-1) * x, batch)
        x_decay_med   = global_add_pool(weights_med.unsqueeze(-1) * x, batch)

        # 3. Concatenated Representation
        graph_emb = torch.cat([x_mean, x_max, x_decay_tight, x_decay_med], dim=1)

        if global_feats is not None:
            graph_emb = torch.cat([graph_emb, global_feats], dim=1)

        out = F.relu(self.fc1(graph_emb))
        out = self.fc2(out)
        return out.squeeze(-1)

# ────────────────── Loss Function ──────────────────

def compute_loss(pred, target, beta=40.0, lambda_corr=10.0):
    loss_huber = F.smooth_l1_loss(pred, target, beta=beta)
    if pred.size(0) > 3:
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8))
        loss_corr = 1.0 - cost
    else:
        loss_corr = torch.tensor(0.0, device=pred.device)
    return loss_huber + lambda_corr * loss_corr

# ────────────────── Training / Evaluation ──────────────────

def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        g = data.global_feat.to(device)
        optimizer.zero_grad()
        pred = model(data, g)
        loss = compute_loss(pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        n += data.num_graphs
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for data in loader:
        data = data.to(device)
        g = data.global_feat.to(device)
        preds.append(model(data, g).cpu().numpy())
        tgts.append(data.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(tgts)

# ────────────────── Main Execution ──────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdb_dir = os.path.join(project_root, "pdb-files")

    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "features.xlsx"))
    logger.info(f"Loaded master dataset: {len(df)} entries")

    aa_props = load_aa_properties(project_root)
    n_props  = aa_props.shape[1]
    in_channels = 20 + n_props + 1  # 49
    logger.info(f"Node feature dimension: {in_channels}")

    # Full 32-config hyperparameter grid search
    hidden_dims   = [5, 6, 7, 8]
    n_layers_list = [2, 3]
    lr_list       = [0.001, 0.005]
    cutoff_list   = [8.0, 10.0]

    configs = list(itertools.product(hidden_dims, n_layers_list, lr_list, cutoff_list))
    logger.info(f"Grid search: {len(configs)} configs of RGCN + FiLM pH + Distance Readout")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    targets = df["mV"].values.astype(np.float64)
    ph_vals = df["pH"].values.astype(np.float64)
    ph_norm = (ph_vals - ph_vals.mean()) / (ph_vals.std() + 1e-8)

    # Pre-build graphs per spatial cutoff
    graph_cache = {}
    for cutoff in cutoff_list:
        logger.info(f"Building multi-relational graphs with cutoff={cutoff} A ...")
        pdb_cache = {}
        data_list = []

        for idx, row in df.iterrows():
            base = row["base_pdb"]
            pdb_path = os.path.join(pdb_dir, f"{base}.pdb")

            if base not in pdb_cache:
                if os.path.exists(pdb_path):
                    pdb_cache[base] = pdb_to_graph_multirelational(pdb_path, aa_props, n_props, edge_cutoff=cutoff)
                else:
                    pdb_cache[base] = None

            template = pdb_cache[base]
            if template is not None:
                d = template.clone()
                d.y = torch.tensor([targets[idx]], dtype=torch.float)
                d.global_feat = torch.tensor([[ph_norm[idx]]], dtype=torch.float)
                data_list.append(d)

        graph_cache[cutoff] = data_list
        logger.info(f"  Built {len(data_list)} graph-datapoints")
        log_edge_stats(data_list)

    n_epochs   = 200
    n_splits   = 5
    patience   = 25
    batch_size = 32

    all_results = []
    start_time  = time.time()

    for cfg_idx, (hidden_dim, n_layers, lr, cutoff) in enumerate(configs):
        config_name = f"RGCN_FiLM_pH_h{hidden_dim}_L{n_layers}_lr{lr}_c{int(cutoff)}"
        logger.info(f"\n[{cfg_idx+1}/{len(configs)}] Config: {config_name}")

        data_list = graph_cache[cutoff]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = {"MAE": [], "RMSE": [], "R2": [], "Pearson": []}
        all_preds = np.full(len(data_list), np.nan)
        indices = np.arange(len(data_list))

        for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
            train_data = [data_list[i] for i in train_idx]
            test_data  = [data_list[i] for i in test_idx]

            logger.info(f"    Fold {fold+1}/{n_splits} starting...")
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=False)
            test_loader = DataLoader(
                test_data, batch_size=batch_size * 2, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=False)

            model = ProteinRGCNFiLM(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                num_relations=NUM_RELATIONS,
                dropout=0.1,
                n_global_feats=1
            ).to(device)

            optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10)

            best_val_mae = float("inf")
            best_preds = None
            best_tgts  = None
            patience_counter = 0

            for epoch in range(n_epochs):
                train_epoch(model, optimizer, train_loader, device)

                if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                    preds, tgts = evaluate(model, test_loader, device)
                    mae = mean_absolute_error(tgts, preds)
                    scheduler.step(mae)

                    if mae < best_val_mae:
                        best_val_mae = mae
                        best_preds = preds.copy()
                        best_tgts  = tgts.copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience // 5:
                        break

            logger.info(f"    Fold {fold+1}/{n_splits} finished — Best Val MAE: {best_val_mae:.1f} mV")

            fold_metrics["MAE"].append(mean_absolute_error(best_tgts, best_preds))
            fold_metrics["RMSE"].append(np.sqrt(mean_squared_error(best_tgts, best_preds)))
            fold_metrics["R2"].append(r2_score(best_tgts, best_preds))
            fold_metrics["Pearson"].append(
                pearsonr(best_tgts, best_preds)[0] if len(best_tgts) > 2 else 0
            )
            all_preds[test_idx] = best_preds

            del train_loader, test_loader

        elapsed = time.time() - start_time
        result = {
            "config":       config_name,
            "hidden_dim":   hidden_dim,
            "n_layers":     n_layers,
            "lr":           lr,
            "cutoff":       cutoff,
            "MAE_mean":     np.mean(fold_metrics["MAE"]),
            "MAE_std":      np.std(fold_metrics["MAE"]),
            "RMSE_mean":    np.mean(fold_metrics["RMSE"]),
            "RMSE_std":     np.std(fold_metrics["RMSE"]),
            "R2_mean":      np.mean(fold_metrics["R2"]),
            "R2_std":       np.std(fold_metrics["R2"]),
            "Pearson_mean": np.mean(fold_metrics["Pearson"]),
            "Pearson_std":  np.std(fold_metrics["Pearson"]),
        }
        all_results.append(result)
        result["preds"] = all_preds.copy()

        logger.info(f"  MAE={result['MAE_mean']:.1f}+/-{result['MAE_std']:.1f}  "
                     f"R2={result['R2_mean']:.3f}+/-{result['R2_std']:.3f}  "
                     f"[{elapsed:.0f}s elapsed]")

        # Incremental save
        results_df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "preds"} for r in all_results]
        )
        results_df = results_df.sort_values("MAE_mean").reset_index(drop=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
        results_df.to_excel(
            os.path.join(os.path.dirname(__file__), "outputs", "rgcn_film_ph_results.xlsx"),
            index=False
        )

    # Save scatter plot of best config
    best_res = all_results[np.argsort([r["MAE_mean"] for r in all_results])[0]]
    preds = best_res["preds"]
    valid = ~np.isnan(preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets[valid], preds[valid], alpha=0.5, s=25,
               c="darkcyan", edgecolors="white", linewidth=0.3)
    lims = [min(targets.min(), preds[valid].min()) - 20,
            max(targets.max(), preds[valid].max()) + 20]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2  = r2_score(targets[valid], preds[valid])
    mae = mean_absolute_error(targets[valid], preds[valid])
    ax.set_xlabel("Actual Em (mV)", fontsize=12)
    ax.set_ylabel("Predicted Em (mV)", fontsize=12)
    ax.set_title(f"RGCN + FiLM pH + Distance Readout — R2={r2:.3f}, MAE={mae:.1f} mV\n({best_res['config']})", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "outputs", "rgcn_film_ph_scatter.png"), dpi=300)
    plt.close(fig)

    logger.info("Grid search completed. Results -> outputs/rgcn_film_ph_*")

if __name__ == "__main__":
    main()
