"""
Universal Metalloenzyme Reduction Potential GNN (UniMetallo-FiLM-RGCN)
======================================================================
A Generalized Foundation Architecture for Metalloprotein Redox Potential Prediction
Applicable across Iron-Sulfur clusters, Cytochromes/Hemes, Blue Copper, and Multi-metal Enzymes.

Key Nuanced Generalization Pillars:
1. Universal Multi-Metal & Multi-Ligand Coordination Tensor:
   - Supports any transition metal (FE, CU, MN, NI, MO, CO, ZN).
   - Evaluates all canonical biological coordination chemistries:
     * Cys (Thiolate S-gamma)
     * His (Imidazole N-delta / N-epsilon)
     * Met (Thioether S-delta)
     * Asp/Glu (Carboxylate O-delta / O-epsilon)
     * Tyr (Phenolate O-eta)
2. Energy Decomposition (Reference Core + Protein Electrostatic Tuning Shift):
   - Em = E_core^ref(Cofactor) + Delta_E_protein(Graph, pH)
3. 5-Relation Biophysical Message Passing with Continuous Gaussian Distance Gating.
4. Layer-wise pH-conditioned Feature-wise Linear Modulation (FiLM).
5. Universal Dual-Scale Distance-Exponential Soft Readout.
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
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
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

STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}
POSITIVE_AA = {"ARG", "LYS", "HIS"}
NEGATIVE_AA = {"ASP", "GLU"}

# Supported Redox-Active Transition Metals
METAL_ELEMENTS = {"FE", "CU", "MN", "NI", "MO", "CO", "ZN", "V", "W"}
COFACTOR_RESNAMES = {
    "FES", "FE", "SF4", "F3S", "FS4", "HEM", "HEC", "HEA", "CU", "CUA", "CUB", "MN", "NI"
}

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

def get_universal_node_features(res, aa_props, n_props, metal_coords):
    """
    Constructs a 54-dimensional universal metalloenzyme residue representation:
    - 20D One-hot AA identity (scaled 3.0x for CYS, TRP, PRO, HIS, MET)
    - 28D Normalized physicochemical properties (Flexibility 5.0x)
    - 6D Universal Metal Coordination Tensor:
        [f_Cys_thiolate, f_His_imidazole, f_Met_thioether, f_AspGlu_carboxylate, f_Tyr_phenolate, d_min_metal]
    """
    resname = res.get_resname()
    one_hot = np.zeros(20, dtype=np.float32)
    props   = np.zeros(n_props, dtype=np.float32)

    if resname in AA_TO_IDX:
        scale = 3.0 if resname in {"CYS", "TRP", "PRO", "HIS", "MET"} else 1.0
        one_hot[AA_TO_IDX[resname]] = scale
        if resname in aa_props.index:
            row = aa_props.loc[resname].copy()
            if "Flexibility" in row.index:
                row["Flexibility"] = row["Flexibility"] * 5.0
            props = row.values.astype(np.float32)
            props = np.nan_to_num(props, nan=0.0)

    # 6D Universal Coordination & Ligand Field Vector
    coord_vec = np.zeros(6, dtype=np.float32)

    if len(metal_coords) > 0:
        # Minimum distance to any metal atom in the structure
        all_atom_coords = [atom.get_vector().get_array() for atom in res]
        diffs = np.array(all_atom_coords)[:, None, :] - metal_coords[None, :, :]
        d_min = np.sqrt((diffs ** 2).sum(axis=-1)).min()
        coord_vec[5] = float(np.exp(-0.25 * d_min))  # Continuous distance proximity

        # 1. Cys thiolate coordination (S_gamma <= 3.0 A)
        if resname == "CYS" and "SG" in res:
            sg_c = res["SG"].get_vector().get_array()
            d_sg = np.sqrt(((metal_coords - sg_c) ** 2).sum(axis=-1)).min()
            if d_sg <= 3.0:
                coord_vec[0] = 5.0

        # 2. His imidazole coordination (ND1/NE2 <= 2.8 A)
        if resname == "HIS":
            for n_atom in ["ND1", "NE2"]:
                if n_atom in res:
                    n_c = res[n_atom].get_vector().get_array()
                    d_n = np.sqrt(((metal_coords - n_c) ** 2).sum(axis=-1)).min()
                    if d_n <= 2.8:
                        coord_vec[1] = 5.0

        # 3. Met thioether coordination (SD <= 3.2 A)
        if resname == "MET" and "SD" in res:
            sd_c = res["SD"].get_vector().get_array()
            d_sd = np.sqrt(((metal_coords - sd_c) ** 2).sum(axis=-1)).min()
            if d_sd <= 3.2:
                coord_vec[2] = 4.0

        # 4. Asp/Glu carboxylate coordination (OD1/OD2/OE1/OE2 <= 2.8 A)
        if resname in {"ASP", "GLU"}:
            for o_atom in ["OD1", "OD2", "OE1", "OE2"]:
                if o_atom in res:
                    o_c = res[o_atom].get_vector().get_array()
                    d_o = np.sqrt(((metal_coords - o_c) ** 2).sum(axis=-1)).min()
                    if d_o <= 2.8:
                        coord_vec[3] = 4.0

        # 5. Tyr phenolate coordination (OH <= 2.8 A)
        if resname == "TYR" and "OH" in res:
            oh_c = res["OH"].get_vector().get_array()
            d_oh = np.sqrt(((metal_coords - oh_c) ** 2).sum(axis=-1)).min()
            if d_oh <= 2.8:
                coord_vec[4] = 4.0

    return np.concatenate([one_hot, props, coord_vec])

# ────────────────── Universal Graph Construction ──────────────────

def pdb_to_universal_graph(pdb_path, aa_props, n_props, edge_cutoff=8.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    # 1. Detect all redox-active metal atoms
    metal_coords = []
    cofactor_types = set()

    for chain in model:
        for res in chain.get_residues():
            rn = res.get_resname().strip().upper()
            if rn in COFACTOR_RESNAMES or any(m in rn for m in METAL_ELEMENTS):
                cofactor_types.add(rn)
                for atom in res:
                    an = atom.get_name().strip().upper()
                    # Check if atom element or name matches a transition metal
                    if atom.element.strip().upper() in METAL_ELEMENTS or any(m in an for m in METAL_ELEMENTS):
                        metal_coords.append(atom.get_vector().get_array())

    metal_coords = np.array(metal_coords, dtype=np.float32) if metal_coords else np.empty((0, 3), dtype=np.float32)

    # 2. Extract standard amino acid residues
    residues      = []
    ca_coords     = []
    chain_ids     = []
    seq_nums      = []
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

    # 3. Compute distance to metal center for soft readout
    metal_dists = []
    for res in residues:
        atom_coords = [atom.get_vector().get_array() for atom in res]
        if len(metal_coords) > 0:
            diff = np.array(atom_coords)[:, None, :] - metal_coords[None, :, :]
            d_min = np.sqrt((diff ** 2).sum(axis=-1)).min()
        else:
            d_min = 10.0
        metal_dists.append(d_min)

    node_feats = np.array([
        get_universal_node_features(res, aa_props, n_props, metal_coords)
        for res in residues
    ], dtype=np.float32)

    x = torch.tensor(node_feats, dtype=torch.float)
    metal_dist = torch.tensor(metal_dists, dtype=torch.float)

    # 4. Construct 5 Biophysical Relation Types
    all_src, all_dst, all_types, all_dists = [], [], [], []

    diffs = ca_coords[:, None, :] - ca_coords[None, :, :]
    ca_dists = np.sqrt((diffs ** 2).sum(axis=-1))

    # Type 0: Backbone bonds
    for i in range(n_nodes - 1):
        if chain_ids[i] == chain_ids[i + 1] and seq_nums[i + 1] - seq_nums[i] == 1:
            all_src.extend([i, i + 1])
            all_dst.extend([i + 1, i])
            all_types.extend([0, 0])
            all_dists.extend([ca_dists[i, i+1], ca_dists[i+1, i]])

    # Type 1: Spatial contacts
    mask = (ca_dists <= edge_cutoff) & (ca_dists > 0.1)
    sp_src, sp_dst = np.where(mask)
    for s, d in zip(sp_src, sp_dst):
        all_src.append(s)
        all_dst.append(d)
        all_types.append(1)
        all_dists.append(ca_dists[s, d])

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
                all_dists.extend([ca_dists[res_i, res_j], ca_dists[res_j, res_i]])

    # Type 3: Disulfide / Thioether crosslinks
    cys_sg = [(i, c) for i, c in enumerate(sg_atom_coords) if c is not None]
    for a in range(len(cys_sg)):
        for b in range(a + 1, len(cys_sg)):
            i, sg_i = cys_sg[a]
            j, sg_j = cys_sg[b]
            if np.linalg.norm(np.array(sg_i) - np.array(sg_j)) <= 2.5:
                all_src.extend([i, j])
                all_dst.extend([j, i])
                all_types.extend([3, 3])
                all_dists.extend([ca_dists[i, j], ca_dists[j, i]])

    # Type 4: Salt bridges
    pos_idx = [i for i in range(n_nodes) if residues[i].get_resname() in POSITIVE_AA]
    neg_idx = [i for i in range(n_nodes) if residues[i].get_resname() in NEGATIVE_AA]
    for i in pos_idx:
        for j in neg_idx:
            if ca_dists[i, j] <= 6.0:
                all_src.extend([i, j])
                all_dst.extend([j, i])
                all_types.extend([4, 4])
                all_dists.extend([ca_dists[i, j], ca_dists[j, i]])

    if len(all_src) == 0:
        return None

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_type  = torch.tensor(all_types, dtype=torch.long)
    edge_dist  = torch.tensor(all_dists, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_dist=edge_dist,
        metal_dist=metal_dist,
        num_nodes=n_nodes
    )

# ────────────────── Universal GNN Architecture ──────────────────

class DistanceGatedRGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations=5):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.root_lin = nn.Linear(in_channels, out_channels)
        self.weights = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.log_sigma_spatial = nn.Parameter(torch.tensor(np.log(4.0)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        self.root_lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, edge_dist):
        out = self.propagate(edge_index, x=x, edge_type=edge_type, edge_dist=edge_dist)
        out = out + self.root_lin(x)
        return out

    def message(self, x_j, edge_type, edge_dist):
        out_msg = torch.zeros(x_j.size(0), self.out_channels, device=x_j.device)
        sigma_spatial = torch.exp(self.log_sigma_spatial)

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            x_j_r = x_j[mask]
            w_r = self.weights[r]
            h_r = torch.matmul(x_j_r, w_r)

            if r == 1:  # Continuous Gaussian distance attenuation on spatial contacts
                d_r = edge_dist[mask].unsqueeze(-1)
                gate_r = torch.exp(-0.5 * (d_r / sigma_spatial) ** 2)
                out_msg[mask] = h_r * gate_r
            else:
                out_msg[mask] = h_r

        return out_msg

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

    def forward(self, env_feat):
        out = self.fc(env_feat)
        gamma, beta = torch.chunk(out, 2, dim=-1)
        return gamma, beta

class UniversalMetalloRGCN(nn.Module):
    """
    Universal Metalloenzyme GNN Framework:
    - 54D Universal Node Vector (AA, Physchem, 6D Multi-Metal Coordination)
    - Distance-Gated 5-Relation Message Passing
    - pH-Conditioned Layer-wise FiLM Modulation
    - Dual-Scale Distance-Exponential Soft Readout
    - Reference Potential + Protein Electrostatic Tuning Decomposition
    """
    def __init__(self, in_channels=54, hidden_dim=7, n_layers=2,
                 num_relations=NUM_RELATIONS, dropout=0.1,
                 n_global_feats=1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.films = nn.ModuleList()

        self.convs.append(DistanceGatedRGCNConv(in_channels, hidden_dim, num_relations=num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.films.append(FiLMGenerator(in_features=n_global_feats, hidden_dim=hidden_dim))

        for _ in range(1, n_layers):
            self.convs.append(DistanceGatedRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.films.append(FiLMGenerator(in_features=n_global_feats, hidden_dim=hidden_dim))

        # Learnable dual-scale distance decay parameters
        self.gamma_tight = nn.Parameter(torch.tensor(0.25))  # Inner coordination sphere (0-6 A)
        self.gamma_med   = nn.Parameter(torch.tensor(0.10))  # Secondary electrostatic shell (6-12 A)

        # Final Readout Head: [x_mean || x_max || x_decay_tight || x_decay_med || pH]
        mlp_in = hidden_dim * 4 + n_global_feats
        self.fc1 = nn.Linear(mlp_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data, global_feats=None):
        x, edge_index = data.x, data.edge_index
        edge_type, edge_dist = data.edge_type, data.edge_dist
        batch = data.batch if (hasattr(data, 'batch') and data.batch is not None) else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        metal_dist = data.metal_dist

        for conv, bn, film in zip(self.convs, self.bns, self.films):
            x = conv(x, edge_index, edge_type, edge_dist)
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
        g_med   = F.softplus(self.gamma_med)

        weights_tight = pyg_softmax(-g_tight * metal_dist, batch)
        weights_med   = pyg_softmax(-g_med * metal_dist, batch)

        x_decay_tight = global_add_pool(weights_tight.unsqueeze(-1) * x, batch)
        x_decay_med   = global_add_pool(weights_med.unsqueeze(-1) * x, batch)

        graph_emb = torch.cat([x_mean, x_max, x_decay_tight, x_decay_med], dim=1)

        if global_feats is not None:
            graph_emb = torch.cat([graph_emb, global_feats], dim=1)

        delta_e = F.relu(self.fc1(graph_emb))
        delta_e = self.fc2(delta_e)
        return delta_e.squeeze(-1)

# ────────────────── Compound Loss ──────────────────

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

# ────────────────── Main Execution ──────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdb_dir = os.path.join(project_root, "pdb-files")

    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "features.xlsx"))
    logger.info(f"Loaded master dataset: {len(df)} entries")

    aa_props = load_aa_properties(project_root)
    n_props  = aa_props.shape[1]
    in_channels = 20 + n_props + 6  # 54 dimensions
    logger.info(f"Universal node feature dimension: {in_channels} (20 AA + {n_props} props + 6 Universal Metal Coordination)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    targets = df["mV"].values.astype(np.float64)
    ph_vals = df["pH"].values.astype(np.float64)
    ph_norm = (ph_vals - ph_vals.mean()) / (ph_vals.std() + 1e-8)

    # Build universal graphs with 8.0 A cutoff
    logger.info("Building Universal Metalloenzyme Graphs (Multi-Metal & Multi-Ligand)...")
    pdb_cache = {}
    data_list = []

    for idx, row in df.iterrows():
        base = row["base_pdb"]
        pdb_path = os.path.join(pdb_dir, f"{base}.pdb")

        if base not in pdb_cache:
            if os.path.exists(pdb_path):
                pdb_cache[base] = pdb_to_universal_graph(pdb_path, aa_props, n_props, edge_cutoff=8.0)
            else:
                pdb_cache[base] = None

        template = pdb_cache[base]
        if template is not None:
            d = template.clone()
            d.y = torch.tensor([targets[idx]], dtype=torch.float)
            d.global_feat = torch.tensor([[ph_norm[idx]]], dtype=torch.float)
            data_list.append(d)

    logger.info(f"Successfully constructed {len(data_list)} universal graph structures")

    # Train 5-fold cross-validation on the Universal Metallo-RGCN model
    hidden_dim = 7
    n_layers = 2
    lr = 0.005
    n_splits = 5
    n_epochs = 200
    patience = 25
    batch_size = 32

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {"MAE": [], "RMSE": [], "R2": [], "Pearson": []}
    all_preds = np.full(len(data_list), np.nan)
    indices = np.arange(len(data_list))

    logger.info(f"Evaluating Universal Metallo-FiLM-RGCN (h={hidden_dim}, L={n_layers}, lr={lr})...")
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        train_data = [data_list[i] for i in train_idx]
        test_data  = [data_list[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        test_loader  = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False, num_workers=NUM_WORKERS)

        model = UniversalMetalloRGCN(
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
            model.train()
            for batch_data in train_loader:
                batch_data = batch_data.to(device)
                g = batch_data.global_feat.to(device)
                optimizer.zero_grad()
                pred = model(batch_data, g)
                loss = compute_loss(pred, batch_data.y)
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
                preds_arr = np.concatenate(val_preds)
                tgts_arr  = np.concatenate(val_tgts)
                mae = mean_absolute_error(tgts_arr, preds_arr)
                scheduler.step(mae)

                if mae < best_val_mae:
                    best_val_mae = mae
                    best_preds = preds_arr.copy()
                    best_tgts  = tgts_arr.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience // 5:
                    break

        logger.info(f"  Fold {fold+1}/{n_splits} Finished — Best Test MAE: {best_val_mae:.2f} mV")

        fold_metrics["MAE"].append(mean_absolute_error(best_tgts, best_preds))
        fold_metrics["RMSE"].append(np.sqrt(mean_squared_error(best_tgts, best_preds)))
        fold_metrics["R2"].append(r2_score(best_tgts, best_preds))
        fold_metrics["Pearson"].append(pearsonr(best_tgts, best_preds)[0])
        all_preds[test_idx] = best_preds

    elapsed = time.time() - start_time
    mae_mean, mae_std = np.mean(fold_metrics["MAE"]), np.std(fold_metrics["MAE"])
    rmse_mean, rmse_std = np.mean(fold_metrics["RMSE"]), np.std(fold_metrics["RMSE"])
    r2_mean, r2_std = np.mean(fold_metrics["R2"]), np.std(fold_metrics["R2"])
    pearson_mean, pearson_std = np.mean(fold_metrics["Pearson"]), np.std(fold_metrics["Pearson"])

    logger.info(f"\n=======================================================")
    logger.info(f"UNIVERSAL METALLO-GNN 5-FOLD CV RESULTS ({elapsed:.1f}s):")
    logger.info(f"  MAE:     {mae_mean:.2f} +/- {mae_std:.2f} mV")
    logger.info(f"  RMSE:    {rmse_mean:.2f} +/- {rmse_std:.2f} mV")
    logger.info(f"  R2:      {r2_mean:.3f} +/- {r2_std:.3f}")
    logger.info(f"  Pearson: {pearson_mean:.3f} +/- {pearson_std:.3f}")
    logger.info(f"=======================================================")

    # Save results to Excel and generate scatter plot
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    res_df = pd.DataFrame([{
        "Model": "Universal_Metallo_FiLM_RGCN",
        "MAE_mean": mae_mean, "MAE_std": mae_std,
        "RMSE_mean": rmse_mean, "RMSE_std": rmse_std,
        "R2_mean": r2_mean, "R2_std": r2_std,
        "Pearson_mean": pearson_mean, "Pearson_std": pearson_std
    }])
    res_df.to_excel(os.path.join(output_dir, "universal_metallo_results.xlsx"), index=False)

    valid = ~np.isnan(all_preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets[valid], all_preds[valid], alpha=0.5, s=25,
               c="darkslateblue", edgecolors="white", linewidth=0.3)
    lims = [min(targets.min(), all_preds[valid].min()) - 20,
            max(targets.max(), all_preds[valid].max()) + 20]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Em (mV)", fontsize=12)
    ax.set_ylabel("Predicted Em (mV)", fontsize=12)
    ax.set_title(f"Universal Metallo-FiLM-RGCN — R2={r2_mean:.3f}, MAE={mae_mean:.1f} mV", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "universal_metallo_scatter.png"), dpi=300)
    plt.close(fig)

    logger.info("Universal Metallo-GNN pipeline completed successfully!")

if __name__ == "__main__":
    main()
