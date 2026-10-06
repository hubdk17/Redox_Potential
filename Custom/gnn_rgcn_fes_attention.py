"""
GNN Architecture with Dedicated FeS-Centered Attention and Multi-Scale Shell Readout
===================================================================================
A lightweight multi-relational GNN designed to match the 57.9 mV RGCN baseline
while introducing distance-aware convolutions and cofactor-centered readout.

Features:
- Residue Graph with 9 distinct relation types (Backbone, 5 Radial bins, and 3 Chemical).
- Edge-conditioned convolution using RBF distance encodings.
- Dedicated FeS-centered attention mechanisms (no virtual nodes).
- Multi-scale radial shell pooling to capture local chemical environments.
- Multi-cluster support using a cluster-level GNN layer.
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
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from Bio.PDB import PDBParser

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_WORKERS = 0
NUM_RELATIONS = 9
RELATION_NAMES = [
    "Backbone",
    "Spatial 0-3A",
    "Spatial 3-5A",
    "Spatial 5-8A",
    "Spatial 8-12A",
    "Spatial 12-16A",
    "H-bond",
    "Disulfide",
    "Salt bridge"
]

STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

POSITIVE_AA = {"ARG", "LYS", "HIS"}
NEGATIVE_AA = {"ASP", "GLU"}

class ClusterInfo:
    """Wrapper container for per-graph cluster geometry & centers to prevent PyG DataLoader auto-concatenation errors."""
    def __init__(self, geom_tensor, centers_tensor):
        self.geom = geom_tensor
        self.centers = centers_tensor

# ────────────────── Physicochemical Properties ──────────────────

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

def build_multirelational_graph(pdb_path, aa_props, n_props, edge_cutoff=10.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    # Collect standard residues + coordinates
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
                    res["N"].get_vector().get_array() if "N" in res else None
                )
                o_atoms_c.append(
                    res["O"].get_vector().get_array() if "O" in res else None
                )
                sg_atoms_c.append(
                    res["SG"].get_vector().get_array()
                    if (rn == "CYS" and "SG" in res) else None
                )

    n_nodes = len(residues)
    if n_nodes < 3:
        return None

    ca_coords = np.array(ca_list, dtype=np.float32)
    diffs = ca_coords[:, None, :] - ca_coords[None, :, :]
    ca_dists = np.sqrt((diffs ** 2).sum(axis=-1))

    # Detect Fe atoms and group by cluster residue
    clusters = []
    for chain in model:
        for res in chain.get_residues():
            resname = res.get_resname()
            if resname in {"FES", "FE", "SF4", "F3S"}:
                fe_list = []
                for atom in res:
                    if "FE" in atom.get_name():
                        fe_list.append(atom.get_vector().get_array())
                if len(fe_list) > 0:
                    fe_arr = np.array(fe_list, dtype=np.float32)
                    clusters.append({
                        "resname": resname,
                        "fe_coords": fe_arr,
                        "center": fe_arr.mean(axis=0),
                        "res_id": f"{chain.get_id()}_{res.get_id()[1]}"
                    })

    if len(clusters) == 0:
        center = ca_coords.mean(axis=0)
        clusters.append({
            "resname": "DUMMY",
            "fe_coords": np.array([center], dtype=np.float32),
            "center": center,
            "res_id": "dummy_0"
        })

    # Coordinating flags
    coord_flags = []
    for res in residues:
        is_coord = False
        if res.get_resname() == "CYS":
            for c in clusters:
                for fe in c["fe_coords"]:
                    for atom in res:
                        d = np.linalg.norm(atom.get_vector().get_array() - fe)
                        if d <= 3.0:
                            is_coord = True
                            break
                    if is_coord:
                        break
                if is_coord:
                    break
        coord_flags.append(is_coord)

    # Node features
    node_feats = np.array([
        get_weighted_node_features(res, aa_props, n_props, cf)
        for res, cf in zip(residues, coord_flags)
    ], dtype=np.float32)
    x = torch.tensor(node_feats, dtype=torch.float)

    # Edge detection
    all_src, all_dst, all_types, all_dists = [], [], [], []

    # Type 0: Backbone
    for i in range(n_nodes - 1):
        if chain_ids[i] == chain_ids[i + 1] and seq_nums[i + 1] - seq_nums[i] == 1:
            all_src.extend([i, i + 1])
            all_dst.extend([i + 1, i])
            all_types.extend([0, 0])
            all_dists.extend([ca_dists[i, i+1], ca_dists[i+1, i]])

    # Types 1-5: Radial spatial contacts (up to edge_cutoff Å)
    b1, b2, b3, b4 = 0.3 * edge_cutoff, 0.5 * edge_cutoff, 0.7 * edge_cutoff, 0.85 * edge_cutoff
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = ca_dists[i, j]
            if d <= edge_cutoff and d > 0.1:
                if d <= b1:
                    t = 1
                elif d <= b2:
                    t = 2
                elif d <= b3:
                    t = 3
                elif d <= b4:
                    t = 4
                else:
                    t = 5
                all_src.extend([i, j])
                all_dst.extend([j, i])
                all_types.extend([t, t])
                all_dists.extend([d, d])

    # Type 6: Hydrogen bonds
    nv = [(idx, c) for idx, c in enumerate(n_atoms_c) if c is not None]
    ov = [(idx, c) for idx, c in enumerate(o_atoms_c) if c is not None]
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
                all_types.extend([6, 6])
                all_dists.extend([ca_dists[ri, rj], ca_dists[rj, ri]])

    # Type 7: Disulfide bonds
    cys_sg = [(idx, c) for idx, c in enumerate(sg_atoms_c) if c is not None]
    for a in range(len(cys_sg)):
        for b in range(a + 1, len(cys_sg)):
            i_idx, sgi = cys_sg[a]
            j_idx, sgj = cys_sg[b]
            if np.linalg.norm(np.array(sgi) - np.array(sgj)) <= 2.5:
                all_src.extend([i_idx, j_idx])
                all_dst.extend([j_idx, i_idx])
                all_types.extend([7, 7])
                all_dists.extend([ca_dists[i_idx, j_idx], ca_dists[j_idx, i_idx]])

    # Type 8: Salt bridges
    pos = [idx for idx in range(n_nodes) if residues[idx].get_resname() in POSITIVE_AA]
    neg = [idx for idx in range(n_nodes) if residues[idx].get_resname() in NEGATIVE_AA]
    for i_idx in pos:
        for j_idx in neg:
            if ca_dists[i_idx, j_idx] <= 6.0:
                all_src.extend([i_idx, j_idx])
                all_dst.extend([j_idx, i_idx])
                all_types.extend([8, 8])
                all_dists.extend([ca_dists[i_idx, j_idx], ca_dists[j_idx, i_idx]])

    if len(all_src) == 0:
        return None

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_type  = torch.tensor(all_types, dtype=torch.long)
    edge_dist  = torch.tensor(all_dists, dtype=torch.float)

    # Compute cluster-to-residue geometries
    geom_list = []
    for c in clusters:
        c_fe = c["fe_coords"]
        c_center = c["center"]
        c_geom = np.zeros((n_nodes, 9), dtype=np.float32)
        for i_node, res in enumerate(residues):
            atom_coords = [atom.get_vector().get_array() for atom in res]
            diff = np.array(atom_coords)[:, None, :] - c_fe[None, :, :]
            d_min = np.sqrt((diff ** 2).sum(axis=-1)).min()
            d_ca = np.linalg.norm(res["CA"].get_vector().get_array() - c_center)
            
            is_coord = 0.0
            if res.get_resname() == "CYS" and d_min <= 3.0:
                is_coord = 1.0

            c_geom[i_node, 0] = d_min
            c_geom[i_node, 1] = d_ca
            c_geom[i_node, 2] = is_coord
            
            if d_min <= 3.0:
                c_geom[i_node, 3] = 1.0
            elif d_min <= 5.0:
                c_geom[i_node, 4] = 1.0
            elif d_min <= 8.0:
                c_geom[i_node, 5] = 1.0
            elif d_min <= 12.0:
                c_geom[i_node, 6] = 1.0
            elif d_min <= 16.0:
                c_geom[i_node, 7] = 1.0
            else:
                c_geom[i_node, 8] = 1.0
        geom_list.append(c_geom)

    geom_tensor = torch.tensor(np.stack(geom_list, axis=0), dtype=torch.float)
    centers_tensor = torch.tensor(np.stack([c["center"] for c in clusters], axis=0), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_dist=edge_dist,
        cluster_info=ClusterInfo(geom_tensor, centers_tensor),
        num_nodes=n_nodes
    )

def log_edge_stats(data_list):
    """Print how many edges of each relation type exist across all graphs."""
    totals = [0] * NUM_RELATIONS
    for d in data_list:
        for t in range(NUM_RELATIONS):
            totals[t] += int((d.edge_type == t).sum())
    for t, name in enumerate(RELATION_NAMES):
        logger.info(f"    {name:>15s} edges: {totals[t]:>8,}")
    logger.info(f"    {'TOTAL':>15s} edges: {sum(totals):>8,}")

# ────────────────── GNN Modules ──────────────────

class DistanceAwareRGCNConv(MessagePassing):
    """Memory-efficient edge-conditioned relation-specific GNN layer."""
    def __init__(self, in_channels, out_channels, num_relations=9, k_dim=16):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        
        self.root_lin = nn.Linear(in_channels, out_channels)
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.dist_gates = nn.ModuleList([
            nn.Linear(k_dim, out_channels)
            for _ in range(num_relations)
        ])
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self.root_lin.reset_parameters()
        for gate in self.dist_gates:
            gate.reset_parameters()
        
    def forward(self, x, edge_index, edge_type, edge_attr):
        out = self.propagate(edge_index, x=x, edge_type=edge_type, edge_attr=edge_attr)
        out = out + self.root_lin(x)
        return out
        
    def message(self, x_j, edge_type, edge_attr):
        out_msg = torch.zeros(x_j.size(0), self.out_channels, device=x_j.device)
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            x_j_r = x_j[mask]
            attr_r = edge_attr[mask]
            
            # Relation transformation + distance modulation gate
            h_r = torch.matmul(x_j_r, self.weight[r])
            gate_r = torch.sigmoid(self.dist_gates[r](attr_r))
            
            out_msg[mask] = h_r * gate_r
        return out_msg

class DistanceAwareRGCN(nn.Module):
    """Lightweight 2-layer Distance-Aware RGCN."""
    def __init__(self, in_channels, hidden_dim, num_relations=9, k_dim=16, n_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(DistanceAwareRGCNConv(in_channels, hidden_dim, num_relations, k_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(1, n_layers):
            self.convs.append(DistanceAwareRGCNConv(hidden_dim, hidden_dim, num_relations, k_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
    def forward(self, x, edge_index, edge_type, edge_attr):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index, edge_type, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if h.size(1) == h_new.size(1):
                h = h + h_new
            else:
                h = h_new
        return h

# ────────────────── FeS-Centered Readout ──────────────────

class FeSCenteredAttention(nn.Module):
    """Attention score calculation relative to the FeS cluster."""
    def __init__(self, hidden_dim, geom_dim=9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + geom_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, h, geom):
        feat = torch.cat([h, geom], dim=1)
        scores = self.mlp(feat)
        alpha = torch.softmax(scores, dim=0)
        h_FeS = torch.sum(alpha * h, dim=0)
        return h_FeS, scores.squeeze(-1)

class MultiScaleFeSPooling(nn.Module):
    """Vectorized pooling of residue embeddings over multi-scale radial shells."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_local = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, geom, scores):
        # h: [N_g, hidden_dim]
        # geom: [N_g, 9] (indices 3..7 are the 5 radial shell one-hot columns)
        # scores: [N_g]
        shell_masks = geom[:, 3:8]  # [N_g, 5]
        
        # Mask out nodes not in shell (set unmasked positions to -1e9)
        masked_scores = scores.unsqueeze(1) - (1.0 - shell_masks) * 1e9  # [N_g, 5]
        alpha_matrix = torch.softmax(masked_scores, dim=0) * shell_masks  # [N_g, 5]
        
        # Matrix multiply: alpha_matrix.T [5, N_g] @ h [N_g, hidden_dim] -> [5, hidden_dim]
        h_shells = torch.matmul(alpha_matrix.t(), h)  # [5, hidden_dim]
        h_shell = h_shells.reshape(-1)  # [5 * hidden_dim]
        return self.mlp_local(h_shell)

class ClusterLevelGNN(nn.Module):
    """Cluster-level interaction layer for multi-cluster support."""
    def __init__(self, hidden_dim, k_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(k_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.pool_g = nn.Linear(hidden_dim, 1)
        
    def forward(self, h_FeS_list, centers, k_dim=16):
        C_g = len(h_FeS_list)
        if C_g == 1:
            return h_FeS_list[0], torch.ones(1, device=centers.device)
            
        h_FeS = torch.stack(h_FeS_list, dim=0)
        diffs = centers.unsqueeze(1) - centers.unsqueeze(0)
        dists = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-8)
        
        mu = torch.linspace(0.0, 50.0, k_dim, device=centers.device)
        step = 50.0 / (k_dim - 1)
        beta = 1.0 / (2.0 * (step ** 2))
        diff = dists.unsqueeze(-1) - mu.view(1, 1, k_dim)
        rbf_ab = torch.exp(-beta * (diff ** 2))
        
        h_new_list = []
        for a in range(C_g):
            msg_sum = torch.zeros(self.hidden_dim, device=centers.device)
            for b in range(C_g):
                if a == b:
                    continue
                weights = self.edge_mlp(rbf_ab[a, b]).view(self.hidden_dim, self.hidden_dim)
                msg_sum = msg_sum + torch.mv(weights, h_FeS[b])
            h_new_list.append(h_FeS[a] + msg_sum)
            
        h_updated = torch.stack(h_new_list, dim=0)
        beta_scores = self.pool_g(h_updated).squeeze(-1)
        beta_weights = torch.softmax(beta_scores, dim=0)
        
        h_FeS_global_graph = torch.sum(beta_weights.unsqueeze(-1) * h_updated, dim=0)
        return h_FeS_global_graph, beta_weights

# ────────────────── Unified GNN Model ──────────────────

class FeSRedoxGNN(nn.Module):
    """Unified network with multi-scale FeS attention & global pooling."""
    def __init__(self, in_channels, hidden_dim, num_relations=9, k_dim=16, n_layers=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        
        self.mu = nn.Parameter(torch.linspace(0.0, 16.0, k_dim), requires_grad=False)
        self.step = 16.0 / (k_dim - 1)
        self.beta = 1.0 / (2.0 * (self.step ** 2))
        
        self.rel_emb = nn.Embedding(num_relations, k_dim)
        self.gnn = DistanceAwareRGCN(in_channels, hidden_dim, num_relations, k_dim, n_layers, dropout)
        self.fes_attention = FeSCenteredAttention(hidden_dim, geom_dim=9)
        self.multiscale_pooling = MultiScaleFeSPooling(hidden_dim)
        self.cluster_gnn = ClusterLevelGNN(hidden_dim, k_dim)
        
        mlp_in = 4 * hidden_dim + 1  # [h_global (2h) || h_local_FeS (h) || h_global_FeS (h)] + pH
        self.fc1 = nn.Linear(mlp_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, data, global_feats=None):
        x, edge_index, edge_type, edge_dist = data.x, data.edge_index, data.edge_type, data.edge_dist
        E = edge_index.size(1)
        edge_attr = torch.zeros(E, self.k_dim, device=x.device)
        
        spatial_mask = (edge_type >= 1) & (edge_type <= 5)
        non_spatial_mask = ~spatial_mask
        
        if spatial_mask.sum() > 0:
            d = edge_dist[spatial_mask].unsqueeze(-1)
            diff = d - self.mu.unsqueeze(0)
            edge_attr[spatial_mask] = torch.exp(-self.beta * (diff ** 2))
            
        if non_spatial_mask.sum() > 0:
            edge_attr[non_spatial_mask] = self.rel_emb(edge_type[non_spatial_mask])
            
        h = self.gnn(x, edge_index, edge_type, edge_attr)
        
        batch = data.batch
        num_graphs = data.num_graphs
        
        # Vectorized global readout across all graphs in batch
        h_mean_batch = global_mean_pool(h, batch)
        h_max_batch  = global_max_pool(h, batch)
        h_global_batch = torch.cat([h_mean_batch, h_max_batch], dim=1)
        
        h_FeS_local_list = []
        h_FeS_global_list = []
        
        for g in range(num_graphs):
            node_mask = (batch == g)
            h_g = h[node_mask]
            
            cluster_info = data.cluster_info[g]
            geom_g = cluster_info.geom.to(x.device, non_blocking=True)
            centers_g = cluster_info.centers.to(x.device, non_blocking=True)
            C_g = geom_g.size(0)
            
            h_FeS_c_list = []
            h_FeS_local_c_list = []
            
            for c in range(C_g):
                geom_gc = geom_g[c]
                h_FeS_c, scores_gc = self.fes_attention(h_g, geom_gc)
                h_FeS_local_c = self.multiscale_pooling(h_g, geom_gc, scores_gc)
                
                h_FeS_c_list.append(h_FeS_c)
                h_FeS_local_c_list.append(h_FeS_local_c)
                
            h_FeS_global_graph, beta_weights = self.cluster_gnn(h_FeS_c_list, centers_g, self.k_dim)
            if C_g > 1:
                h_FeS_local_graph = torch.sum(beta_weights.unsqueeze(-1) * torch.stack(h_FeS_local_c_list, dim=0), dim=0)
            else:
                h_FeS_local_graph = h_FeS_local_c_list[0]
                
            h_FeS_local_list.append(h_FeS_local_graph)
            h_FeS_global_list.append(h_FeS_global_graph)
            
        h_FeS_local_batch = torch.stack(h_FeS_local_list, dim=0)
        h_FeS_global_batch = torch.stack(h_FeS_global_list, dim=0)
        
        h_graph = torch.cat([h_global_batch, h_FeS_local_batch, h_FeS_global_batch], dim=1)
        if global_feats is not None:
            h_final = torch.cat([h_graph, global_feats], dim=1)
        else:
            h_final = h_graph
            
        pred = self.fc2(F.relu(self.fc1(h_final))).squeeze(-1)
        return pred

# ────────────────── Training Loop ──────────────────

def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        g = data.global_feat.to(device)
        optimizer.zero_grad()
        pred = model(data, g)
        loss = F.mse_loss(pred, data.y)
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

    hidden_dims   = [5, 6, 7, 8]
    n_layers_list = [2, 3]
    lr_list       = [0.001, 0.005]
    cutoff_list   = [8.0, 10.0]

    configs = list(itertools.product(hidden_dims, n_layers_list, lr_list, cutoff_list))
    logger.info(f"Grid search: {len(configs)} configs of FeS Attention-pooling RGCN")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    targets = df["mV"].values.astype(np.float64)
    ph_vals = df["pH"].values.astype(np.float64)
    ph_norm = (ph_vals - ph_vals.mean()) / (ph_vals.std() + 1e-8)

    # Build graphs per cutoff
    graph_cache = {}
    for cutoff in cutoff_list:
        logger.info(f"Building multi-relational graphs with cutoff={cutoff} Å ...")
        pdb_cache = {}
        data_list = []

        for idx, row in df.iterrows():
            base = row["base_pdb"]
            pdb_path = os.path.join(pdb_dir, f"{base}.pdb")

            if base not in pdb_cache:
                if os.path.exists(pdb_path):
                    pdb_cache[base] = build_multirelational_graph(pdb_path, aa_props, n_props, edge_cutoff=cutoff)
                else:
                    pdb_cache[base] = None

            template = pdb_cache[base]
            if template is not None:
                d = template.clone()
                d.cluster_info = ClusterInfo(
                    template.cluster_info.geom.to(device),
                    template.cluster_info.centers.to(device)
                )
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
        config_name = f"FeS_Attn_RGCN_h{hidden_dim}_L{n_layers}_lr{lr}_c{int(cutoff)}"
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

            model = FeSRedoxGNN(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_relations=NUM_RELATIONS,
                k_dim=16,
                n_layers=n_layers,
                dropout=0.1
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

        logger.info(f"  MAE={result['MAE_mean']:.1f}±{result['MAE_std']:.1f}  "
                     f"R²={result['R2_mean']:.3f}±{result['R2_std']:.3f}  "
                     f"[{elapsed:.0f}s elapsed]")

        # Incremental save
        results_df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "preds"} for r in all_results]
        )
        results_df = results_df.sort_values("MAE_mean").reset_index(drop=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
        results_df.to_excel(
            os.path.join(os.path.dirname(__file__), "outputs", "rgcn_fes_attention_results.xlsx"),
            index=False
        )

    # Save best scatter plot
    best_res = all_results[np.argsort([r["MAE_mean"] for r in all_results])[0]]
    preds = best_res["preds"]
    valid = ~np.isnan(preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets[valid], preds[valid], alpha=0.5, s=25,
               c="teal", edgecolors="white", linewidth=0.3)
    lims = [min(targets.min(), preds[valid].min()) - 20,
            max(targets.max(), preds[valid].max()) + 20]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2  = r2_score(targets[valid], preds[valid])
    mae = mean_absolute_error(targets[valid], preds[valid])
    ax.set_xlabel("Actual Em (mV)", fontsize=12)
    ax.set_ylabel("Predicted Em (mV)", fontsize=12)
    ax.set_title(f"FeS Attention RGCN — R²={r2:.3f}, MAE={mae:.1f} mV\n({best_res['config']})", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "outputs", "rgcn_fes_attention_scatter.png"), dpi=300)
    plt.close(fig)

    logger.info("Grid search completed. Results -> outputs/rgcn_fes_attention_*")


if __name__ == "__main__":
    main()
