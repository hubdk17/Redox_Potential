import os
import shutil
import logging
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from Bio.PDB import PDBParser

# Set up logging and warning suppressions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Define GCN Layer in PyTorch
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # adj: (N, N) normalized adjacency matrix
        # x: (N, in_features) node features
        out = torch.matmul(adj, x)
        out = self.linear(out)
        return out

# Define the full Protein GNN Regressor
class ProteinGNN(nn.Module):
    def __init__(self, num_features):
        super(ProteinGNN, self).__init__()
        self.gcn1 = GCNLayer(num_features, 64)
        self.gcn2 = GCNLayer(64, 32)
        self.fc1 = nn.Linear(32 + 1, 16)  # 32 (GNN pooling) + 1 (pH)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x, adj, ph):
        # x: (N, num_features), adj: (N, N), ph: float
        h = torch.relu(self.gcn1(x, adj))
        h = torch.relu(self.gcn2(h, adj))
        
        # Global mean pooling
        g = torch.mean(h, dim=0)  # (32,)
        
        # Concatenate pH
        ph_tensor = torch.tensor([ph], dtype=torch.float32)
        g_combined = torch.cat([g, ph_tensor], dim=0)  # (33,)
        
        out = torch.relu(self.fc1(g_combined))
        out = self.fc2(out)
        return out

# Helper function to get residue one-letter mapping
amm_names = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]

# Read amino acid features table
def load_table_amm():
    table_path = os.path.join(os.path.dirname(__file__), "../tableAmm.txt")
    table_amm = pd.read_csv(table_path, sep="\t", index_col=1)
    table_amm = table_amm.iloc[:, 1:]
    return table_amm

# Parse PDB structure and construct PyTorch graph tensors
def build_protein_graph(pdb_id, table_amm, path_pdb=None):
    if path_pdb is None:
        path_pdb = os.path.join(os.path.dirname(__file__), "../pdb-files")
    pdb_path = os.path.join(path_pdb, f"{pdb_id.lower()}.pdb")
    if not os.path.isfile(pdb_path):
        return None
        
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as e:
        logger.warning(f"Error parsing {pdb_id}: {str(e)}")
        return None
        
    # Find cofactor FES or FE atoms
    cofactor_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.resname in ["FES", "FE"]:
                    cofactor_atoms.extend([atom.coord for atom in residue.get_atoms()])
                    
    if not cofactor_atoms:
        return None
        
    # Compute cofactor barycenter
    barycenter = np.mean(cofactor_atoms, axis=0)
    
    # Collect nodes (residues within 15.0 Å of cofactor barycenter)
    nodes = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.resname in amm_names and "CA" in residue:
                    ca_atom = residue["CA"]
                    dist = np.linalg.norm(ca_atom.coord - barycenter)
                    if dist <= 15.0:
                        nodes.append(residue)
                        
    if len(nodes) < 2:
        return None
        
    # Build node features tensor
    node_features = []
    for node in nodes:
        feats = table_amm.loc[node.resname].values
        node_features.append(feats)
    x_tensor = torch.tensor(np.array(node_features), dtype=torch.float32)
    
    # Build adjacency matrix (dist <= 8.0 Å between C-alpha atoms)
    num_nodes = len(nodes)
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist = np.linalg.norm(nodes[i]["CA"].coord - nodes[j]["CA"].coord)
            if dist <= 8.0:
                adj[i, j] = 1.0
                
    # Add self-loops
    adj = adj + np.eye(num_nodes)
    
    # Normalize adjacency matrix D^(-1/2) * A * D^(-1/2)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = np.dot(D_inv_sqrt, np.dot(adj, D_inv_sqrt))
    
    adj_tensor = torch.tensor(adj_norm, dtype=torch.float32)
    
    return x_tensor, adj_tensor

def main():
    logger.info("Loading dataset and amino acid features...")
    table_amm = load_table_amm()
    
    path_dataset = os.path.join(os.path.dirname(__file__), "../Database Redox Pot Fe2S2 proteins.xlsx")
    df = pd.read_excel(path_dataset).dropna(subset=["PDB Code(s)", "mV", "pH"])
    
    # Construct all graphs
    logger.info("Constructing protein graphs...")
    dataset_graphs = []
    
    for idx, row in df.iterrows():
        pdb_id = row["PDB Code(s)"].strip().lower()
        mv = row["mV"]
        ph = row["pH"]
        
        graph_data = build_protein_graph(pdb_id, table_amm)
        if graph_data is not None:
            x, adj = graph_data
            dataset_graphs.append({
                "pdb_id": pdb_id,
                "x": x,
                "adj": adj,
                "ph": ph,
                "mv": mv
            })
            
    logger.info(f"Successfully constructed {len(dataset_graphs)} graphs.")
    
    if len(dataset_graphs) == 0:
        logger.error("No graphs constructed! Exiting.")
        return

    # Define 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    mae_list, rmse_list, r2_list = [], [], []
    pearson_list, spearman_list = [], []
    
    startTime = time.time()
    
    # 5-fold CV loop
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_graphs)):
        logger.info(f"--- Training Fold {fold + 1}/5 ---")
        
        train_set = [dataset_graphs[i] for i in train_idx]
        test_set = [dataset_graphs[i] for i in test_idx]
        
        # Initialize GNN model
        num_features = dataset_graphs[0]["x"].shape[1]
        model = ProteinGNN(num_features)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.L1Loss()  # MAE Loss
        
        # Train for 10 epochs
        model.train()
        for epoch in range(100):
            epoch_loss = 0.0
            for graph in train_set:
                optimizer.zero_grad()
                pred = model(graph["x"], graph["adj"], graph["ph"])
                target = torch.tensor([graph["mv"]], dtype=torch.float32)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # print(f"Fold {fold+1}, Epoch {epoch+1}/10, Loss: {epoch_loss/len(train_set):.4f}")
            
        # Evaluate on test set
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for graph in test_set:
                pred = model(graph["x"], graph["adj"], graph["ph"])
                preds.append(pred.item())
                targets.append(graph["mv"])
                
        preds = np.array(preds)
        targets = np.array(targets)
        
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        r2 = r2_score(targets, preds)
        
        p_corr, _ = pearsonr(targets, preds)
        s_corr, _ = spearmanr(targets, preds)
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        pearson_list.append(p_corr)
        spearman_list.append(s_corr)
        
        logger.info(f"Fold {fold+1} Results - MAE: {mae:.2f} mV, RMSE: {rmse:.2f} mV, R2: {r2:.3f}, Pearson: {p_corr:.3f}")
        
    executionTime = time.time() - startTime
    
    # Save results to DataFrame and Excel
    results_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)] + ["Mean", "Std"],
        "MAE": mae_list + [np.mean(mae_list), np.std(mae_list)],
        "RMSE": rmse_list + [np.mean(rmse_list), np.std(rmse_list)],
        "R2": r2_list + [np.mean(r2_list), np.std(r2_list)],
        "Pearson": pearson_list + [np.mean(pearson_list), np.std(pearson_list)],
        "Spearman": spearman_list + [np.mean(spearman_list), np.std(spearman_list)]
    })
    
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, "results_dataset.xlsx")
    results_df.to_excel(output_path, index=False)
    
    print("\n--- GNN 10-Epoch CV Results Summary ---")
    print(results_df.to_string(index=False))
    print(f"\nExecution time in seconds: {executionTime:.2f}")

if __name__ == "__main__":
    main()
