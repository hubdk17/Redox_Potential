"""
Graph Neural Network (GNN) for Fe2S2 Redox Potential Prediction
================================================================
Builds protein graphs from PDB structures:
  - Nodes = residues (Cα atoms), features = AA physicochemical properties
  - Edges = spatial contacts between Cα atoms within a cutoff distance
  - Global feature: pH (appended after graph pooling)

Grid search over:
  - Hidden dimensions: 5, 6, 7, 8
  - GNN layers: 2, 3
  - Learning rate: 0.001, 0.005
  - Edge cutoff: 8Å, 10Å
  - GNN type: GCN, GAT

Uses GPU (cuda:0) with num_workers=4 for DataLoader.
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
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
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

COFACTOR_NAMES = {"FES", "FE", "SF4", "F3S", "FEO", "HEC"}

# ────────────────── AA Property Features ──────────────────

def load_aa_properties(project_root):
    """Load physicochemical properties for the 20 standard AAs."""
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


def get_node_features(resname, aa_props, n_props):
    """Get feature vector for a residue: one-hot(20) + physicochemical properties."""
    one_hot = np.zeros(20, dtype=np.float32)
    props = np.zeros(n_props, dtype=np.float32)

    if resname in AA_TO_IDX:
        one_hot[AA_TO_IDX[resname]] = 1.0
        if resname in aa_props.index:
            props = aa_props.loc[resname].values.astype(np.float32)
            props = np.nan_to_num(props, nan=0.0)

    return np.concatenate([one_hot, props])


# ────────────────── Graph Construction ──────────────────

def pdb_to_graph(pdb_path, aa_props, n_props, edge_cutoff=8.0):
    """
    Convert a PDB file to a PyG Data object.
    Nodes = standard AA residues (Cα atoms).
    Edges = pairs of residues with Cα-Cα distance <= cutoff.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

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

    # Node features
    node_feats = []
    for res in residues:
        feat = get_node_features(res.get_resname(), aa_props, n_props)
        node_feats.append(feat)
    x = torch.tensor(np.array(node_feats), dtype=torch.float)

    # Edge construction
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))

    mask = (dists <= edge_cutoff) & (dists > 0.1)
    src, dst = np.where(mask)
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    edge_attr = torch.tensor(
        dists[src, dst].reshape(-1, 1) / edge_cutoff, dtype=torch.float
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)


# ────────────────── GNN Model ──────────────────

class ProteinGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_layers, gnn_type="GCN",
                 dropout=0.1, n_global_feats=1):
        super().__init__()
        self.gnn_type = gnn_type
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            if gnn_type == "GCN":
                self.convs.append(GCNConv(in_ch, hidden_dim))
            elif gnn_type == "GAT":
                self.convs.append(GATConv(in_ch, hidden_dim, heads=1))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # MLP head: pooled graph features + global features → prediction
        mlp_input = hidden_dim * 2 + n_global_feats
        self.fc1 = nn.Linear(mlp_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data, global_feats=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_emb = torch.cat([x_mean, x_max], dim=1)

        if global_feats is not None:
            graph_emb = torch.cat([graph_emb, global_feats], dim=1)

        out = F.relu(self.fc1(graph_emb))
        out = self.fc2(out)
        return out.squeeze(-1)


# ────────────────── Training with DataLoader ──────────────────

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

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    return preds, targets, mae, rmse, r2


# ────────────────── Main ──────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdb_dir = os.path.join(project_root, "pdb-files")

    # Load master dataset
    features_path = os.path.join(os.path.dirname(__file__), "features.xlsx")
    df = pd.read_excel(features_path)
    logger.info(f"Master dataset: {len(df)} datapoints")

    # Load AA properties
    aa_props = load_aa_properties(project_root)
    n_props = aa_props.shape[1]
    in_channels = 20 + n_props
    logger.info(f"Node feature dim: {in_channels} (20 one-hot + {n_props} properties)")

    # ── Grid search config ──
    hidden_dims = [5, 6, 7, 8]
    n_layers_list = [2, 3]
    lr_list = [0.001, 0.005]
    cutoff_list = [8.0, 10.0]
    gnn_types = ["GCN", "GAT"]

    configs = list(itertools.product(hidden_dims, n_layers_list, lr_list, cutoff_list, gnn_types))
    logger.info(f"Grid search: {len(configs)} configurations")

    # ── Device setup ──
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        torch.backends.cudnn.benchmark = True
    else:
        logger.info(f"Device: {device} (no GPU detected, running on CPU)")

    # ── Pre-build graphs for each cutoff ──
    # Store target and global features directly in the Data object for DataLoader
    targets = df["mV"].values.astype(np.float64)
    ph_vals = df["pH"].values.astype(np.float64)
    ph_norm = (ph_vals - ph_vals.mean()) / (ph_vals.std() + 1e-8)

    graph_cache = {}  # cutoff -> list of Data objects (one per datapoint)
    for cutoff in cutoff_list:
        logger.info(f"Building graphs with cutoff={cutoff}Å...")
        pdb_graph_cache = {}  # base_pdb -> Data template
        data_list = []

        for idx, row in df.iterrows():
            base = row["base_pdb"]
            pdb_path = os.path.join(pdb_dir, f"{base}.pdb")

            if base not in pdb_graph_cache:
                if os.path.exists(pdb_path):
                    g = pdb_to_graph(pdb_path, aa_props, n_props, edge_cutoff=cutoff)
                    pdb_graph_cache[base] = g
                else:
                    pdb_graph_cache[base] = None

            template = pdb_graph_cache[base]
            if template is not None:
                # Clone the graph and attach target + global features
                d = template.clone()
                d.y = torch.tensor([targets[idx]], dtype=torch.float)
                d.global_feat = torch.tensor([[ph_norm[idx]]], dtype=torch.float)
                d.sample_idx = idx  # track original index
                data_list.append(d)

        graph_cache[cutoff] = data_list
        logger.info(f"  Built {len(data_list)} graph-datapoints (cutoff={cutoff}Å)")

    # ── Run grid search with 5-fold CV ──
    n_epochs = 200
    n_splits = 5
    patience = 25
    batch_size = 32

    all_results = []
    start_time = time.time()

    for cfg_idx, (hidden_dim, n_layers, lr, cutoff, gnn_type) in enumerate(configs):
        config_name = f"h{hidden_dim}_L{n_layers}_lr{lr}_{gnn_type}_c{int(cutoff)}"
        logger.info(f"\n[{cfg_idx+1}/{len(configs)}] Config: {config_name}")

        data_list = graph_cache[cutoff]
        if len(data_list) < 50:
            logger.warning(f"  Too few valid graphs ({len(data_list)}), skipping")
            continue

        # 5-fold CV
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = {"MAE": [], "RMSE": [], "R2": [], "Pearson": []}
        all_preds = np.full(len(data_list), np.nan)

        indices = np.arange(len(data_list))

        for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]

            logger.info(f"    Fold {fold+1}/{n_splits} starting...")
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=False,
                persistent_workers=(NUM_WORKERS > 0),
            )
            test_loader = DataLoader(
                test_data, batch_size=batch_size * 2, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=False,
                persistent_workers=(NUM_WORKERS > 0),
            )

            model = ProteinGNN(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                gnn_type=gnn_type,
                dropout=0.1,
                n_global_feats=1,
            ).to(device)

            optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )

            best_val_mae = float("inf")
            best_preds = None
            patience_counter = 0

            for epoch in range(n_epochs):
                train_loss = train_epoch(model, optimizer, train_loader, device)

                if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                    preds, tgts, mae, rmse, r2 = evaluate(model, test_loader, device)
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
            r2 = r2_score(best_tgts, best_preds)
            fold_metrics["R2"].append(r2)
            pear = pearsonr(best_tgts, best_preds)[0] if len(best_tgts) > 2 else 0
            fold_metrics["Pearson"].append(pear)
            all_preds[test_idx] = best_preds

            # Clean up DataLoader workers
            del train_loader, test_loader

        elapsed = time.time() - start_time
        result = {
            "config": config_name,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "lr": lr,
            "cutoff": cutoff,
            "gnn_type": gnn_type,
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
                     f"Pearson={result['Pearson_mean']:.3f}  "
                     f"[{elapsed:.0f}s elapsed]")

        # ── Save incremental results ──
        results_df_temp = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in all_results])
        results_df_temp = results_df_temp.sort_values("MAE_mean").reset_index(drop=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
        results_df_temp.to_excel(
            os.path.join(os.path.dirname(__file__), "outputs", "gnn_grid_search_results.xlsx"),
            index=False,
        )

    total_time = time.time() - start_time
    logger.info(f"\nTotal grid search time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── Results ──
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in all_results])
    results_df = results_df.sort_values("MAE_mean").reset_index(drop=True)

    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
    results_df.to_excel(
        os.path.join(os.path.dirname(__file__), "outputs", "gnn_grid_search_results.xlsx"),
        index=False,
    )

    print("\n" + "=" * 100)
    print("GNN GRID SEARCH RESULTS (sorted by MAE)")
    print("=" * 100)
    pd.set_option("display.width", 150)
    pd.set_option("display.max_rows", 80)
    print(results_df.to_string(index=True))

    # ── Best config per hidden dim ──
    print("\n" + "=" * 80)
    print("BEST CONFIG PER HIDDEN DIMENSION")
    print("=" * 80)
    for hd in hidden_dims:
        subset = results_df[results_df["hidden_dim"] == hd]
        if len(subset) > 0:
            best = subset.iloc[0]
            print(f"  Hidden={hd}: MAE={best['MAE_mean']:.1f}±{best['MAE_std']:.1f}  "
                  f"R²={best['R2_mean']:.3f}  "
                  f"Config: L={int(best['n_layers'])}, lr={best['lr']}, "
                  f"cutoff={int(best['cutoff'])}Å, {best['gnn_type']}")

    # ── Comparison with XGBoost ──
    print("\n" + "=" * 80)
    print("COMPARISON: GNN vs XGBoost")
    print("=" * 80)
    best_gnn = results_df.iloc[0]
    print(f"  Best GNN:       MAE={best_gnn['MAE_mean']:.1f}±{best_gnn['MAE_std']:.1f}  "
          f"R²={best_gnn['R2_mean']:.3f}  ({best_gnn['config']})")
    print(f"  XGBoost(96D):   MAE=48.3±3.6     R²=0.912")
    print(f"  XGBoost(PCA5D): MAE=53.2±7.8     R²=0.880")

    # ── Plot 1: Best MAE and R2 per hidden dim ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for hd in hidden_dims:
        subset = results_df[results_df["hidden_dim"] == hd]
        if len(subset) > 0:
            best = subset.iloc[0]
            axes[0].bar(hd, best["MAE_mean"], yerr=best["MAE_std"],
                       capsize=4, color=plt.cm.viridis(hd / 10),
                       edgecolor="white", width=0.6)
            axes[1].bar(hd, best["R2_mean"], yerr=best["R2_std"],
                       capsize=4, color=plt.cm.viridis(hd / 10),
                       edgecolor="white", width=0.6)

    axes[0].set_xlabel("Hidden Dimension", fontsize=12)
    axes[0].set_ylabel("MAE (mV)", fontsize=12)
    axes[0].set_title("Best GNN MAE per Hidden Dimension", fontsize=13)
    axes[0].set_xticks(hidden_dims)
    axes[0].axhline(y=48.3, color="red", linestyle="--", alpha=0.7, label="XGBoost (48.3 mV)")
    axes[0].legend()

    axes[1].set_xlabel("Hidden Dimension", fontsize=12)
    axes[1].set_ylabel("R² Score", fontsize=12)
    axes[1].set_title("Best GNN R² per Hidden Dimension", fontsize=13)
    axes[1].set_xticks(hidden_dims)
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=0.912, color="red", linestyle="--", alpha=0.7, label="XGBoost (0.912)")
    axes[1].legend()

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("GNN Grid Search: Performance by Hidden Dimension", fontsize=14)
    plt.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "outputs", "gnn_hidden_dim_comparison.png"),
        dpi=300,
    )
    plt.close(fig)

    # ── Plot 2: Heatmaps (GCN vs GAT × layers × hidden_dim) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for gidx, gtype in enumerate(gnn_types):
        ax = axes[gidx]
        heatmap_data = np.full((len(n_layers_list), len(hidden_dims)), np.nan)
        for li, nl in enumerate(n_layers_list):
            for hi, hd in enumerate(hidden_dims):
                subset = results_df[
                    (results_df["gnn_type"] == gtype) &
                    (results_df["n_layers"] == nl) &
                    (results_df["hidden_dim"] == hd)
                ]
                if len(subset) > 0:
                    heatmap_data[li, hi] = subset.iloc[0]["MAE_mean"]

        im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(hidden_dims)))
        ax.set_xticklabels(hidden_dims)
        ax.set_yticks(range(len(n_layers_list)))
        ax.set_yticklabels([f"{n}L" for n in n_layers_list])
        ax.set_xlabel("Hidden Dim")
        ax.set_ylabel("Layers")
        ax.set_title(f"{gtype} — MAE (mV)")
        for li in range(len(n_layers_list)):
            for hi in range(len(hidden_dims)):
                val = heatmap_data[li, hi]
                if not np.isnan(val):
                    ax.text(hi, li, f"{val:.0f}", ha="center", va="center",
                           fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax)

    plt.suptitle("GNN Grid Search Heatmap (Best MAE per config)", fontsize=14)
    plt.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "outputs", "gnn_heatmap.png"),
        dpi=300,
    )
    plt.close(fig)

    # ── Plot 3: Best model scatter ──
    best_result = all_results[results_df.index[0]]
    if "preds" in best_result:
        preds = best_result["preds"]
        valid_targets = np.array([data_list[i].y.item() for i in range(len(data_list))])
        valid = ~np.isnan(preds)
        if valid.sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(valid_targets[valid], preds[valid],
                      alpha=0.5, s=25, c="darkorange", edgecolors="white", linewidth=0.3)
            lims = [min(valid_targets.min(), preds[valid].min()) - 20,
                    max(valid_targets.max(), preds[valid].max()) + 20]
            ax.plot(lims, lims, "k--", alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            r2 = r2_score(valid_targets[valid], preds[valid])
            mae = mean_absolute_error(valid_targets[valid], preds[valid])
            ax.set_xlabel("Actual Em (mV)", fontsize=12)
            ax.set_ylabel("Predicted Em (mV)", fontsize=12)
            ax.set_title(f"Best GNN — R²={r2:.3f}, MAE={mae:.1f} mV\n"
                        f"({best_result['config']})", fontsize=13)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            fig.savefig(
                os.path.join(os.path.dirname(__file__), "outputs", "gnn_best_scatter.png"),
                dpi=300,
            )
            plt.close(fig)

    logger.info("All GNN outputs saved to Custom/outputs/")


if __name__ == "__main__":
    main()
