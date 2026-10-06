"""
Multi-Head Graph Attention Network (GAT) with Weighted Top Features
====================================================================
Builds protein contact graphs where node features corresponding to
top XGBoost descriptors are heavily scaled up:
  - Flexibility: scaled by 5.0
  - CYS, TRP, PRO one-hot: scaled by 3.0
  - Coordinating Cys indicator: added as a dedicated feature and scaled by 5.0

Message passing is performed using Multi-Head GAT layers to capture
anisotropic spatial relationships.
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
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from Bio.PDB import PDBParser

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_WORKERS = 0

# ────────────────────── Constants ──────────────────────

STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

# ────────────────── Node Feature Builder ──────────────────

def load_aa_properties(project_root):
    """Load and normalize amino acid properties from tableAmm.txt."""
    path = os.path.join(project_root, "tableAmm.txt")
    table = pd.read_csv(path, sep="\t", index_col=1).iloc[:, 1:]
    table = table.loc[table.index.isin(STANDARD_AA)]
    for col in table.columns:
        col_min, col_max = table[col].min(), table[col].max()
        if col_max > col_min:
            table[col] = (table[col] - col_min) / (col_max - col_min)
        else:
            table[col] = 0.0
    return table


def get_weighted_node_features(res, aa_props, n_props, is_coordinating):
    """
    Build a custom feature vector for a residue.
    Weights are increased for the top features found by XGBoost:
      - CYS, TRP, PRO one-hot dimensions scaled by 3.0
      - Flexibility property dimension scaled by 5.0
      - Coordinating Cys indicator scaled by 5.0
    """
    resname = res.get_resname()
    one_hot = np.zeros(20, dtype=np.float32)
    props = np.zeros(n_props, dtype=np.float32)

    if resname in AA_TO_IDX:
        # Scale top one-hot residues
        scale = 3.0 if resname in {"CYS", "TRP", "PRO"} else 1.0
        one_hot[AA_TO_IDX[resname]] = scale

        if resname in aa_props.index:
            row = aa_props.loc[resname].copy()
            # Scale Flexibility property
            if "Flexibility" in row.index:
                row["Flexibility"] = row["Flexibility"] * 5.0
            props = row.values.astype(np.float32)
            props = np.nan_to_num(props, nan=0.0)

    # Coordinating Cys indicator feature
    cys_indicator = np.array([5.0 if (resname == "CYS" and is_coordinating) else 0.0], dtype=np.float32)

    return np.concatenate([one_hot, props, cys_indicator])


# ────────────────── Graph Construction ──────────────────

def pdb_to_graph_weighted(pdb_path, aa_props, n_props, edge_cutoff=8.0):
    """
    Convert PDB to a PyG contact graph with custom weighted node features.
    Coordinating Cysteines are identified by proximity (< 3.0 Å) to any Fe atom.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    # Step 1: Locate all Fe atoms in the structure
    fe_coords = []
    for chain in model:
        for res in chain.get_residues():
            resname = res.get_resname()
            if resname in {"FES", "FE", "SF4", "F3S"}:
                for atom in res:
                    if "FE" in atom.get_name():
                        fe_coords.append(atom.get_vector().get_array())
    fe_coords = np.array(fe_coords) if fe_coords else np.empty((0, 3))

    # Step 2: Collect standard amino acid residues
    residues = []
    coords = []
    for chain in model:
        for res in chain.get_residues():
            resname = res.get_resname()
            if resname in STANDARD_AA and "CA" in res:
                residues.append(res)
                coords.append(res["CA"].get_vector().get_array())

    if len(residues) < 3:
        return None

    coords = np.array(coords, dtype=np.float32)
    n_nodes = len(residues)

    # Step 3: Determine which Cysteines coordinate the Fe atoms
    coordinating_flags = []
    for res in residues:
        is_coord = False
        if res.get_resname() == "CYS" and len(fe_coords) > 0:
            # Check distance of any Cys side-chain atom (SG usually) to any Fe atom
            for atom in res:
                atom_coord = atom.get_vector().get_array()
                dists_to_fe = np.linalg.norm(fe_coords - atom_coord, axis=1)
                if len(dists_to_fe) > 0 and np.min(dists_to_fe) <= 3.0:
                    is_coord = True
                    break
        coordinating_flags.append(is_coord)

    # Step 4: Build weighted node features
    node_feats = []
    for res, is_coord in zip(residues, coordinating_flags):
        feat = get_weighted_node_features(res, aa_props, n_props, is_coord)
        node_feats.append(feat)
    x = torch.tensor(np.array(node_feats), dtype=torch.float)

    # Step 5: Construct edges based on spatial contact
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))

    mask = (dists <= edge_cutoff) & (dists > 0.1)
    src, dst = np.where(mask)
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    edge_attr = torch.tensor(
        dists[src, dst].reshape(-1, 1) / edge_cutoff, dtype=torch.float
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)


# ────────────────── GAT Model with Multi-Head Attention ──────────────────

class ProteinGAT(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_layers=2, heads=4, dropout=0.1, n_global_feats=1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First GAT layer (multi-head)
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        # Intermediate GAT layers
        for i in range(1, n_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        # Final GAT layer (averages heads to output hidden_dim)
        self.convs.append(GATConv(hidden_dim * heads if n_layers > 1 else in_channels, hidden_dim, heads=1, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output MLP Prediction Head
        mlp_input = hidden_dim * 2 + n_global_feats  # Mean + Max Pool + pH
        self.fc1 = nn.Linear(mlp_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data, global_feats=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing layers
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_emb = torch.cat([x_mean, x_max], dim=1)

        # Append global pH
        if global_feats is not None:
            graph_emb = torch.cat([graph_emb, global_feats], dim=1)

        out = F.relu(self.fc1(graph_emb))
        out = self.fc2(out)
        return out.squeeze(-1)


# ────────────────── Training/Evaluation Loop ──────────────────

def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for data in loader:
        data = data.to(device)
        y = data.y.to(device)
        g = data.global_feat.to(device)

        optimizer.zero_grad()
        pred = model(data, g)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for data in loader:
        data = data.to(device)
        g = data.global_feat.to(device)
        pred = model(data, g)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(data.y.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ────────────────── Main ──────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdb_dir = os.path.join(project_root, "pdb-files")

    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "features.xlsx"))
    logger.info(f"Loaded master dataset: {len(df)} entries")

    aa_props = load_aa_properties(project_root)
    n_props = aa_props.shape[1]
    in_channels = 20 + n_props + 1  # 20 one-hot + 28 properties + 1 coordinating Cys indicator
    logger.info(f"Weighted node features dimension: {in_channels}")

    # Config sweeps
    hidden_dims = [5, 6, 7, 8]
    n_layers_list = [2, 3]
    lr_list = [0.001, 0.005]
    cutoff_list = [8.0, 10.0]

    configs = list(itertools.product(hidden_dims, n_layers_list, lr_list, cutoff_list))
    logger.info(f"Grid search: {len(configs)} configuration sweeps of Multi-Head GAT")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    targets = df["mV"].values.astype(np.float64)
    ph_vals = df["pH"].values.astype(np.float64)
    ph_norm = (ph_vals - ph_vals.mean()) / (ph_vals.std() + 1e-8)

    # Pre-build contact graphs for each cutoff
    graph_cache = {}
    for cutoff in cutoff_list:
        logger.info(f"Building custom weighted graphs with cutoff={cutoff}Å...")
        pdb_graph_cache = {}
        data_list = []

        for idx, row in df.iterrows():
            base = row["base_pdb"]
            pdb_path = os.path.join(pdb_dir, f"{base}.pdb")

            if base not in pdb_graph_cache:
                if os.path.exists(pdb_path):
                    g = pdb_to_graph_weighted(pdb_path, aa_props, n_props, edge_cutoff=cutoff)
                    pdb_graph_cache[base] = g
                else:
                    pdb_graph_cache[base] = None

            template = pdb_graph_cache[base]
            if template is not None:
                d = template.clone()
                d.y = torch.tensor([targets[idx]], dtype=torch.float)
                d.global_feat = torch.tensor([[ph_norm[idx]]], dtype=torch.float)
                data_list.append(d)

        graph_cache[cutoff] = data_list
        logger.info(f"  Built {len(data_list)} graph-datapoints")

    n_epochs = 200
    n_splits = 5
    patience = 25
    batch_size = 32

    all_results = []
    start_time = time.time()

    for cfg_idx, (hidden_dim, n_layers, lr, cutoff) in enumerate(configs):
        config_name = f"GAT_h{hidden_dim}_L{n_layers}_lr{lr}_c{int(cutoff)}"
        logger.info(f"\n[{cfg_idx+1}/{len(configs)}] Config: {config_name}")

        data_list = graph_cache[cutoff]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = {"MAE": [], "RMSE": [], "R2": [], "Pearson": []}
        all_preds = np.full(len(data_list), np.nan)
        indices = np.arange(len(data_list))

        for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]

            logger.info(f"    Fold {fold+1}/{n_splits} starting...")
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
            test_loader = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

            model = ProteinGAT(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                heads=4,
                dropout=0.1,
                n_global_feats=1,
            ).to(device)

            optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

            best_val_mae = float("inf")
            best_preds = None
            best_tgts = None
            patience_counter = 0

            for epoch in range(n_epochs):
                train_loss = train_epoch(model, optimizer, train_loader, device)

                if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                    preds, tgts = evaluate(model, test_loader, device)
                    mae = mean_absolute_error(tgts, preds)
                    scheduler.step(mae)

                    if mae < best_val_mae:
                        best_val_mae = mae
                        best_preds = preds.copy()
                        best_tgts = tgts.copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience // 5:
                        break

            logger.info(f"    Fold {fold+1}/{n_splits} finished - Best Val MAE: {best_val_mae:.1f} mV")
            fold_metrics["MAE"].append(mean_absolute_error(best_tgts, best_preds))
            fold_metrics["RMSE"].append(np.sqrt(mean_squared_error(best_tgts, best_preds)))
            fold_metrics["R2"].append(r2_score(best_tgts, best_preds))
            fold_metrics["Pearson"].append(pearsonr(best_tgts, best_preds)[0] if len(best_tgts) > 2 else 0)
            all_preds[test_idx] = best_preds

            del train_loader, test_loader

        elapsed = time.time() - start_time
        result = {
            "config": config_name,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "lr": lr,
            "cutoff": cutoff,
            "MAE_mean": np.mean(fold_metrics["MAE"]),
            "MAE_std": np.std(fold_metrics["MAE"]),
            "RMSE_mean": np.mean(fold_metrics["RMSE"]),
            "RMSE_std": np.std(fold_metrics["RMSE"]),
            "R2_mean": np.mean(fold_metrics["R2"]),
            "R2_std": np.std(fold_metrics["R2"]),
            "Pearson_mean": np.mean(fold_metrics["Pearson"]),
            "Pearson_std": np.std(fold_metrics["Pearson"]),
        }
        all_results.append(result)
        result["preds"] = all_preds.copy()

        logger.info(f"  MAE={result['MAE_mean']:.1f}±{result['MAE_std']:.1f}  "
                     f"R²={result['R2_mean']:.3f}±{result['R2_std']:.3f}  "
                     f"[{elapsed:.0f}s elapsed]")

        # Incremental save
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in all_results])
        results_df = results_df.sort_values("MAE_mean").reset_index(drop=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
        results_df.to_excel(
            os.path.join(os.path.dirname(__file__), "outputs", "gat_weighted_grid_results.xlsx"),
            index=False,
        )

    # Plot best scatter
    best_res = all_results[np.argsort([r["MAE_mean"] for r in all_results])[0]]
    preds = best_res["preds"]
    valid = ~np.isnan(preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets[valid], preds[valid], alpha=0.5, s=25, c="firebrick", edgecolors="white", linewidth=0.3)
    lims = [min(targets.min(), preds[valid].min()) - 20, max(targets.max(), preds[valid].max()) + 20]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    r2 = r2_score(targets[valid], preds[valid])
    mae = mean_absolute_error(targets[valid], preds[valid])
    ax.set_xlabel("Actual Em (mV)", fontsize=12)
    ax.set_ylabel("Predicted Em (mV)", fontsize=12)
    ax.set_title(f"Weighted GAT Model — R²={r2:.3f}, MAE={mae:.1f} mV\n({best_res['config']})", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "outputs", "gat_weighted_best_scatter.png"), dpi=300)
    plt.close(fig)

    logger.info("Grid search completed successfully. Best results saved.")


if __name__ == "__main__":
    main()
