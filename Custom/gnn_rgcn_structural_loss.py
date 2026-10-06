"""
Multi-Relational RGCN with Structure-Aware Auxiliary Losses
============================================================
Same 5-edge-type RGCN architecture that achieved MAE=57.9 mV, plus two
auxiliary losses that regularise the learned node representations to be
structurally aware:

  1. Contact-distance loss  — for sampled spatial-contact edges (type 1),
     predict the normalised Cα–Cα distance from the absolute difference
     of the two node embeddings.  Forces the embedding space to preserve
     distance relationships.

  2. Bond-angle loss — for interior backbone nodes, predict the
     normalised Cα–Cα–Cα angle from the node embedding.  Forces the
     model to encode local backbone geometry (α-helix ≈ 91°,
     β-sheet ≈ 120°).

Total loss per batch:
    L = L_redox  +  λ_dist · L_distance  +  λ_angle · L_angle

Only L_redox is used for early stopping and metric reporting.
"""

import os
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
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
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

DIST_NORM = 15.0        # normalise Cα–Cα distances to ≈ [0, 1]
EDGE_SAMPLE_CAP = 256   # max spatial edges sampled per batch for dist loss

# ────────────────────── Constants ──────────────────────

STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

POSITIVE_AA = {"ARG", "LYS", "HIS"}
NEGATIVE_AA = {"ASP", "GLU"}

# ────────────────── Amino-acid properties ──────────────────

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

# ────────────────── Node feature builder ──────────────────

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
        dtype=np.float32)
    return np.concatenate([one_hot, props, cys_flag])

# ────────────────── Graph builder (with structural targets) ────────────

def pdb_to_graph_structural(pdb_path, aa_props, n_props, edge_cutoff=8.0):
    """
    Same multi-relational graph as gnn_multirelational.py, plus:
      - edge_dist   : (E,)  raw Cα–Cα distance for every edge
      - node_angle  : (N,)  Cα–Cα–Cα angle / π  at each backbone interior
                            node (0 elsewhere)
      - has_angle   : (N,)  bool mask for nodes with a valid angle
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    # ── Fe atoms ──
    fe_coords = []
    for chain in model:
        for res in chain.get_residues():
            if res.get_resname() in {"FES", "FE", "SF4", "F3S"}:
                for atom in res:
                    if "FE" in atom.get_name():
                        fe_coords.append(atom.get_vector().get_array())
    fe_coords = np.array(fe_coords) if fe_coords else np.empty((0, 3))

    # ── Collect residues ──
    residues   = []
    ca_list    = []
    chain_ids  = []
    seq_nums   = []
    n_atoms_c  = []
    o_atoms_c  = []
    sg_atoms_c = []

    for chain in model:
        for res in chain.get_residues():
            rn = res.get_resname()
            if rn in STANDARD_AA and "CA" in res:
                residues.append(res)
                ca_list.append(res["CA"].get_vector().get_array())
                chain_ids.append(chain.get_id())
                seq_nums.append(res.get_id()[1])
                n_atoms_c.append(
                    res["N"].get_vector().get_array() if "N" in res else None)
                o_atoms_c.append(
                    res["O"].get_vector().get_array() if "O" in res else None)
                sg_atoms_c.append(
                    res["SG"].get_vector().get_array()
                    if (rn == "CYS" and "SG" in res) else None)

    n_nodes = len(residues)
    if n_nodes < 3:
        return None

    ca_coords = np.array(ca_list, dtype=np.float32)

    # ── Coordinating-Cys flags ──
    coord_flags = []
    for res in residues:
        is_coord = False
        if res.get_resname() == "CYS" and len(fe_coords) > 0:
            for atom in res:
                d = np.linalg.norm(
                    fe_coords - atom.get_vector().get_array(), axis=1)
                if d.min() <= 3.0:
                    is_coord = True
                    break
        coord_flags.append(is_coord)

    # ── Node features ──
    node_feats = np.array([
        get_weighted_node_features(res, aa_props, n_props, cf)
        for res, cf in zip(residues, coord_flags)
    ], dtype=np.float32)
    x = torch.tensor(node_feats, dtype=torch.float)

    # ═══════════════════ Edge detection ═══════════════════
    all_src, all_dst, all_types = [], [], []

    # Type 0: Backbone
    for i in range(n_nodes - 1):
        if (chain_ids[i] == chain_ids[i + 1]
                and seq_nums[i + 1] - seq_nums[i] == 1):
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
    nv = [(i, c) for i, c in enumerate(n_atoms_c) if c is not None]
    ov = [(i, c) for i, c in enumerate(o_atoms_c) if c is not None]
    if nv and ov:
        ni_list, n_arr = zip(*nv)
        oi_list, o_arr = zip(*ov)
        n_arr = np.array(n_arr, dtype=np.float32)
        o_arr = np.array(o_arr, dtype=np.float32)
        no_d = np.sqrt(((n_arr[:, None, :] - o_arr[None, :, :]) ** 2).sum(-1))
        hi, hj = np.where(no_d <= 3.5)
        for a, b in zip(hi, hj):
            ri, rj = ni_list[a], oi_list[b]
            if ri != rj:
                all_src.extend([ri, rj])
                all_dst.extend([rj, ri])
                all_types.extend([2, 2])

    # Type 3: Disulfide bonds
    cys_sg = [(i, c) for i, c in enumerate(sg_atoms_c) if c is not None]
    for a in range(len(cys_sg)):
        for b in range(a + 1, len(cys_sg)):
            i, sgi = cys_sg[a]
            j, sgj = cys_sg[b]
            if np.linalg.norm(np.array(sgi) - np.array(sgj)) <= 2.5:
                all_src.extend([i, j])
                all_dst.extend([j, i])
                all_types.extend([3, 3])

    # Type 4: Salt bridges
    pos = [i for i in range(n_nodes)
           if residues[i].get_resname() in POSITIVE_AA]
    neg = [i for i in range(n_nodes)
           if residues[i].get_resname() in NEGATIVE_AA]
    for i in pos:
        for j in neg:
            if ca_dists[i, j] <= 6.0:
                all_src.extend([i, j])
                all_dst.extend([j, i])
                all_types.extend([4, 4])

    if not all_src:
        return None

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_type  = torch.tensor(all_types, dtype=torch.long)

    # ── Edge distances (Cα–Cα for every edge) ──
    edge_dist = torch.tensor(
        [ca_dists[s, d] for s, d in zip(all_src, all_dst)],
        dtype=torch.float)

    # ── Backbone angles (Cα_i-1  –  Cα_i  –  Cα_i+1) ──
    node_angle = np.zeros(n_nodes, dtype=np.float32)
    has_angle  = np.zeros(n_nodes, dtype=np.bool_)

    for i in range(1, n_nodes - 1):
        # must be three consecutive residues in the same chain
        if (chain_ids[i - 1] == chain_ids[i] == chain_ids[i + 1]
                and seq_nums[i] - seq_nums[i - 1] == 1
                and seq_nums[i + 1] - seq_nums[i] == 1):
            v1 = ca_coords[i - 1] - ca_coords[i]
            v2 = ca_coords[i + 1] - ca_coords[i]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)
                                       + 1e-8)
            angle = np.arccos(np.clip(cos_a, -1.0, 1.0))
            node_angle[i] = angle / np.pi      # normalise to [0, 1]
            has_angle[i] = True

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_dist=edge_dist,
        node_angle=torch.tensor(node_angle, dtype=torch.float),
        has_angle=torch.tensor(has_angle, dtype=torch.bool),
        num_nodes=n_nodes,
    )


def log_edge_stats(data_list):
    totals = [0] * NUM_RELATIONS
    for d in data_list:
        for t in range(NUM_RELATIONS):
            totals[t] += int((d.edge_type == t).sum())
    for t, name in enumerate(RELATION_NAMES):
        logger.info(f"    {name:>12s} edges: {totals[t]:>8,}")
    logger.info(f"    {'TOTAL':>12s} edges: {sum(totals):>8,}")

    n_angles = sum(int(d.has_angle.sum()) for d in data_list)
    logger.info(f"    Backbone angle targets: {n_angles:>8,}")


# ────────────────── RGCN + Structural Heads ──────────────────

class ProteinRGCNStructural(nn.Module):
    """
    Same RGCN backbone as the 57.9 mV model, plus two small auxiliary
    heads for distance and angle prediction.
    """

    def __init__(self, in_channels, hidden_dim, n_layers=2,
                 num_relations=NUM_RELATIONS, dropout=0.1,
                 n_global_feats=1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        # ── RGCN backbone ──
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_dim, num_relations=num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(1, n_layers):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Main prediction head ──
        mlp_in = hidden_dim * 2 + n_global_feats
        self.fc1 = nn.Linear(mlp_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # ── Auxiliary: distance head ──
        #   Input: |h_i − h_j|   Output: normalised distance ∈ [0,1]
        mid = max(hidden_dim // 2, 2)
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
            nn.Sigmoid(),
        )

        # ── Auxiliary: angle head ──
        #   Input: h_i   Output: normalised angle ∈ [0,1]
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
            nn.Sigmoid(),
        )

    def forward(self, data, global_feats=None):
        x, edge_index = data.x, data.edge_index
        edge_type, batch = data.edge_type, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Graph-level readout ──
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        graph_emb = torch.cat([x_mean, x_max], dim=1)
        if global_feats is not None:
            graph_emb = torch.cat([graph_emb, global_feats], dim=1)

        pred = self.fc2(F.relu(self.fc1(graph_emb))).squeeze(-1)
        return pred, x   # also return node embeddings for aux losses

    def predict_distances(self, node_emb, edge_pairs):
        """Predict normalised Cα–Cα distance from embedding difference."""
        h_diff = torch.abs(node_emb[edge_pairs[0]] - node_emb[edge_pairs[1]])
        return self.dist_head(h_diff).squeeze(-1)

    def predict_angles(self, node_emb, mask):
        """Predict normalised backbone angle from single node embedding."""
        return self.angle_head(node_emb[mask]).squeeze(-1)


# ────────────────── Training / Evaluation ──────────────────

def train_epoch(model, optimizer, loader, device,
                lambda_dist=50.0, lambda_angle=50.0):
    """One training epoch with composite loss."""
    model.train()
    total_main = 0.0
    total_dist = 0.0
    total_angle = 0.0
    n = 0

    for data in loader:
        data = data.to(device)
        g = data.global_feat.to(device)

        pred, node_emb = model(data, g)

        # ── Main loss ──
        loss_main = F.mse_loss(pred, data.y)

        # ── Auxiliary: contact-distance loss ──
        loss_dist = torch.tensor(0.0, device=device)
        spatial_mask = (data.edge_type == 1)
        n_spatial = int(spatial_mask.sum())
        if n_spatial > 0:
            spatial_idx = torch.where(spatial_mask)[0]
            # Sample a subset to keep cost bounded
            if n_spatial > EDGE_SAMPLE_CAP:
                perm = torch.randperm(n_spatial, device=device)[:EDGE_SAMPLE_CAP]
                spatial_idx = spatial_idx[perm]

            edge_pairs = data.edge_index[:, spatial_idx]
            true_dists = data.edge_dist[spatial_idx] / DIST_NORM  # → ≈ [0, 1]
            pred_dists = model.predict_distances(node_emb, edge_pairs)
            loss_dist  = F.mse_loss(pred_dists, true_dists)

        # ── Auxiliary: bond-angle loss ──
        loss_angle = torch.tensor(0.0, device=device)
        angle_mask = data.has_angle
        if angle_mask.sum() > 0:
            true_angles = data.node_angle[angle_mask]
            pred_angles = model.predict_angles(node_emb, angle_mask)
            loss_angle  = F.mse_loss(pred_angles, true_angles)

        loss = loss_main + lambda_dist * loss_dist + lambda_angle * loss_angle

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = data.num_graphs
        total_main  += loss_main.item() * bs
        total_dist  += loss_dist.item() * bs
        total_angle += loss_angle.item() * bs
        n += bs

    return total_main / n, total_dist / n, total_angle / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for data in loader:
        data = data.to(device)
        g = data.global_feat.to(device)
        pred, _ = model(data, g)
        preds.append(pred.cpu().numpy())
        tgts.append(data.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(tgts)


# ────────────────── Main ──────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdb_dir = os.path.join(project_root, "pdb-files")

    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "features.xlsx"))
    logger.info(f"Loaded master dataset: {len(df)} entries")

    aa_props = load_aa_properties(project_root)
    n_props  = aa_props.shape[1]
    in_channels = 20 + n_props + 1  # 49
    logger.info(f"Node feature dimension: {in_channels}")

    # ── Grid (same 32 configs as the plain RGCN) ──
    hidden_dims   = [5, 6, 7, 8]
    n_layers_list = [2, 3]
    lr_list       = [0.001, 0.005]
    cutoff_list   = [8.0, 10.0]

    configs = list(itertools.product(hidden_dims, n_layers_list,
                                     lr_list, cutoff_list))
    logger.info(f"Grid search: {len(configs)} configs of RGCN + structural loss")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} "
                f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    targets = df["mV"].values.astype(np.float64)
    ph_vals = df["pH"].values.astype(np.float64)
    ph_norm = (ph_vals - ph_vals.mean()) / (ph_vals.std() + 1e-8)

    # ── Pre-build graphs ──
    graph_cache = {}
    for cutoff in cutoff_list:
        logger.info(f"Building structural graphs with cutoff={cutoff} Å ...")
        pdb_cache = {}
        data_list = []

        for idx, row in df.iterrows():
            base = row["base_pdb"]
            pdb_path = os.path.join(pdb_dir, f"{base}.pdb")

            if base not in pdb_cache:
                if os.path.exists(pdb_path):
                    pdb_cache[base] = pdb_to_graph_structural(
                        pdb_path, aa_props, n_props, edge_cutoff=cutoff)
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

    # ── Training ──
    n_epochs   = 200
    n_splits   = 5
    patience   = 25
    batch_size = 32
    lambda_dist  = 50.0
    lambda_angle = 50.0

    all_results = []
    start_time  = time.time()

    for cfg_idx, (hidden_dim, n_layers, lr, cutoff) in enumerate(configs):
        config_name = (f"RGCN_Struct_h{hidden_dim}_L{n_layers}"
                       f"_lr{lr}_c{int(cutoff)}")
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

            model = ProteinRGCNStructural(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                num_relations=NUM_RELATIONS,
                dropout=0.1,
                n_global_feats=1,
            ).to(device)

            optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10)

            best_val_mae = float("inf")
            best_preds = None
            best_tgts  = None
            patience_counter = 0

            for epoch in range(n_epochs):
                l_main, l_dist, l_ang = train_epoch(
                    model, optimizer, train_loader, device,
                    lambda_dist, lambda_angle)

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

            logger.info(f"    Fold {fold+1}/{n_splits} finished — "
                        f"Best Val MAE: {best_val_mae:.1f} mV")

            fold_metrics["MAE"].append(
                mean_absolute_error(best_tgts, best_preds))
            fold_metrics["RMSE"].append(
                np.sqrt(mean_squared_error(best_tgts, best_preds)))
            fold_metrics["R2"].append(
                r2_score(best_tgts, best_preds))
            fold_metrics["Pearson"].append(
                pearsonr(best_tgts, best_preds)[0]
                if len(best_tgts) > 2 else 0)
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

        logger.info(f"  MAE={result['MAE_mean']:.1f}±{result['MAE_std']:.1f}  "
                     f"R²={result['R2_mean']:.3f}±{result['R2_std']:.3f}  "
                     f"[{elapsed:.0f}s elapsed]")

        # Incremental save
        results_df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "preds"}
             for r in all_results])
        results_df = results_df.sort_values("MAE_mean").reset_index(drop=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"),
                    exist_ok=True)
        results_df.to_excel(
            os.path.join(os.path.dirname(__file__), "outputs",
                         "rgcn_structural_loss_results.xlsx"),
            index=False)

    # ── Scatter plot ──
    best_res = all_results[np.argsort([r["MAE_mean"]
                                        for r in all_results])[0]]
    preds = best_res["preds"]
    valid = ~np.isnan(preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets[valid], preds[valid], alpha=0.5, s=25,
               c="mediumvioletred", edgecolors="white", linewidth=0.3)
    lims = [min(targets.min(), preds[valid].min()) - 20,
            max(targets.max(), preds[valid].max()) + 20]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2  = r2_score(targets[valid], preds[valid])
    mae = mean_absolute_error(targets[valid], preds[valid])
    ax.set_xlabel("Actual Em (mV)", fontsize=12)
    ax.set_ylabel("Predicted Em (mV)", fontsize=12)
    ax.set_title(f"RGCN + Structural Loss — R²={r2:.3f}, MAE={mae:.1f} mV\n"
                 f"({best_res['config']})", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "outputs",
                             "rgcn_structural_loss_scatter.png"), dpi=300)
    plt.close(fig)

    logger.info("Grid search completed. Results → outputs/rgcn_structural_loss_*")


if __name__ == "__main__":
    main()
