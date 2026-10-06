"""
Custom Feature Extractor for Fe2S2 Protein Redox Potential Prediction
=====================================================================
Extracts ~95 structural, geometric, and physicochemical features from
PDB files of Fe2S2 iron-sulfur cluster proteins.

Author: Custom pipeline (built from scratch)
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict
from Bio.PDB import PDBParser, NeighborSearch
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────── Constants ───────────────────────────

STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

COFACTOR_NAMES = {"FES", "FE"}
COFACTOR_ATOMS_FES = ["FE1", "FE2", "S1", "S2"]
COFACTOR_ATOMS_FE = ["FE"]

# Amino acid molecular weights (Da)
AA_MW = {
    "ALA": 89.1, "ARG": 174.2, "ASN": 132.1, "ASP": 133.1, "CYS": 121.2,
    "GLN": 146.1, "GLU": 147.1, "GLY": 75.0, "HIS": 155.2, "ILE": 131.2,
    "LEU": 131.2, "LYS": 146.2, "MET": 149.2, "PHE": 165.2, "PRO": 115.1,
    "SER": 105.1, "THR": 119.1, "TRP": 204.2, "TYR": 181.2, "VAL": 117.1,
}

# Residue categories
POLAR = {"ASP", "GLU", "SER", "THR", "CYS", "TYR", "ASN", "GLN", "HIS", "LYS", "ARG"}
NONPOLAR = {"ALA", "PHE", "GLY", "ILE", "LEU", "MET", "PRO", "TRP", "VAL"}
CHARGED_POS = {"ARG", "LYS", "HIS"}
CHARGED_NEG = {"ASP", "GLU"}
AROMATIC = {"PHE", "TRP", "TYR", "HIS"}
HBOND_DONORS = {"ARG", "LYS", "HIS", "ASN", "GLN", "SER", "THR", "TRP", "TYR", "CYS"}
HBOND_ACCEPTORS = {"ASP", "GLU", "ASN", "GLN", "SER", "THR", "TYR", "HIS"}

# Formal charge at physiological pH
FORMAL_CHARGE = {"ARG": +1, "LYS": +1, "ASP": -1, "GLU": -1, "HIS": 0}


def load_aa_properties(project_root):
    """Load amino acid physicochemical properties from tableAmm.txt.
    Only loads the 20 standard amino acids (no cofactors)."""
    path = os.path.join(project_root, "tableAmm.txt")
    table = pd.read_csv(path, sep="\t", index_col=1).iloc[:, 1:]
    # Keep only standard AAs
    table = table.loc[table.index.isin(STANDARD_AA)]
    return table


def get_residue_ca(residue):
    """Get Cα coordinate of a residue, or centroid of all atoms as fallback."""
    if "CA" in residue:
        return residue["CA"].get_vector().get_array()
    coords = [a.get_vector().get_array() for a in residue.get_atoms()]
    return np.mean(coords, axis=0) if coords else None


def get_residue_centroid(residue):
    """Get centroid of all atoms in a residue."""
    coords = [a.get_vector().get_array() for a in residue.get_atoms()]
    return np.mean(coords, axis=0) if coords else None


def tetrahedron_volume(p1, p2, p3, p4):
    """Volume of tetrahedron from 4 points."""
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    return abs(np.dot(v1, np.cross(v2, v3))) / 6.0


def planarity_rmsd(points):
    """RMSD of points from their best-fit plane."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # smallest singular value direction
    dists = np.dot(centered, normal)
    return np.sqrt(np.mean(dists ** 2))


def residues_within_radius(chain_residues, center, radius):
    """Return standard AA residues whose Cα is within `radius` of `center`."""
    result = []
    for res in chain_residues:
        ca = get_residue_ca(res)
        if ca is not None:
            if np.linalg.norm(ca - center) <= radius:
                result.append(res)
    return result


def compute_aa_composition(residues, total_for_norm=None):
    """Compute fractional amino acid composition."""
    counts = {aa: 0 for aa in STANDARD_AA}
    for res in residues:
        name = res.get_resname()
        if name in counts:
            counts[name] += 1
    n = total_for_norm if total_for_norm else sum(counts.values())
    if n == 0:
        n = 1
    fracs = {f"frac_{aa}": counts[aa] / n for aa in STANDARD_AA}
    return fracs


def compute_weighted_properties(residues, aa_props):
    """Compute weighted-average physicochemical properties over a set of residues.
    Uses nanmean to avoid NaN propagation."""
    # Select 9 key properties
    key_props = [
        "Volume", "Hydrophobicity", "Isoelectric point",
        "P(helix)", "P(sheet)", "Flexibility",
        "Steric hindrance", "Log(Solubility (m at 20°C))", "nH-bonds"
    ]
    # Map display-safe column names
    # The actual column names in tableAmm may have encoding issues, so match flexibly
    available_cols = list(aa_props.columns)

    # Build mapping: short name -> actual column name
    col_map = {}
    for kp in key_props:
        for ac in available_cols:
            if kp.lower().replace("°", "").replace(" ", "") in ac.lower().replace("°", "").replace(" ", "").replace("�", ""):
                col_map[kp] = ac
                break

    results = {}
    aa_names = [res.get_resname() for res in residues if res.get_resname() in STANDARD_AA]
    if not aa_names:
        for kp in key_props:
            safe_name = kp.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            results[f"wavg_{safe_name}"] = 0.0
        return results

    for kp in key_props:
        safe_name = kp.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        if kp in col_map:
            col = col_map[kp]
            vals = [aa_props.loc[aa, col] for aa in aa_names if aa in aa_props.index]
            results[f"wavg_{safe_name}"] = float(np.nanmean(vals)) if vals else 0.0
        else:
            results[f"wavg_{safe_name}"] = 0.0

    return results


# ─────────────────────────── Main Feature Extractor ───────────────────────────

def extract_features_for_chain(chain, structure, cofactor_atoms, aa_props):
    """
    Extract all features for a single chain containing an Fe2S2 cofactor.

    Parameters
    ----------
    chain : Bio.PDB.Chain
    structure : Bio.PDB.Structure
    cofactor_atoms : dict mapping atom_id -> np.array(coord)
    aa_props : pd.DataFrame of amino acid properties

    Returns
    -------
    dict : feature name -> value
    """
    features = {}

    # ── Collect standard AA residues in this chain ──
    std_residues = [
        res for res in chain.get_residues()
        if res.get_resname() in STANDARD_AA and res.id[0] == " "
    ]
    n_residues = len(std_residues)
    features["n_residues"] = n_residues

    # ── Global structure info ──
    all_chains = list(structure[0].get_chains())
    features["n_chains"] = len(all_chains)

    total_residues_structure = sum(
        1 for ch in all_chains for r in ch.get_residues()
        if r.get_resname() in STANDARD_AA and r.id[0] == " "
    )
    features["chain_length_ratio"] = n_residues / max(total_residues_structure, 1)

    # Molecular weight
    features["molecular_weight"] = sum(
        AA_MW.get(res.get_resname(), 0) for res in std_residues
    )

    # B-factors
    bfactors = []
    for res in std_residues:
        for atom in res.get_atoms():
            bfactors.append(atom.get_bfactor())
    if bfactors:
        features["avg_bfactor"] = np.mean(bfactors)
        features["max_bfactor"] = np.max(bfactors)
        features["min_bfactor"] = np.min(bfactors)
        features["std_bfactor"] = np.std(bfactors)
    else:
        features["avg_bfactor"] = 0
        features["max_bfactor"] = 0
        features["min_bfactor"] = 0
        features["std_bfactor"] = 0

    # Resolution from header
    try:
        resolution = structure.header.get("resolution", None)
        features["resolution"] = float(resolution) if resolution else 0.0
    except (ValueError, TypeError):
        features["resolution"] = 0.0

    # ── Cofactor geometry ──
    atom_names = sorted(cofactor_atoms.keys())
    atom_coords = {name: cofactor_atoms[name] for name in atom_names}
    cofactor_center = np.mean(list(atom_coords.values()), axis=0)

    # Fe-Fe and S-S distances (if FES cluster with 4 atoms)
    fe_atoms = [n for n in atom_names if n.startswith("FE")]
    s_atoms = [n for n in atom_names if n.startswith("S")]

    if len(fe_atoms) >= 2:
        features["fe_fe_distance"] = np.linalg.norm(
            atom_coords[fe_atoms[0]] - atom_coords[fe_atoms[1]]
        )
    else:
        features["fe_fe_distance"] = 0.0

    if len(s_atoms) >= 2:
        features["s_s_distance"] = np.linalg.norm(
            atom_coords[s_atoms[0]] - atom_coords[s_atoms[1]]
        )
    else:
        features["s_s_distance"] = 0.0

    # All pairwise Fe-S distances
    pair_idx = 0
    for fe in fe_atoms:
        for s in s_atoms:
            features[f"fe_s_dist_{pair_idx}"] = np.linalg.norm(
                atom_coords[fe] - atom_coords[s]
            )
            pair_idx += 1
    # Pad if fewer than 4 pairs
    while pair_idx < 4:
        features[f"fe_s_dist_{pair_idx}"] = 0.0
        pair_idx += 1

    # Cluster volume (tetrahedron)
    if len(atom_names) >= 4:
        pts = [atom_coords[n] for n in atom_names[:4]]
        features["cluster_volume"] = tetrahedron_volume(*pts)
        features["cluster_planarity"] = planarity_rmsd(np.array(pts))
    else:
        features["cluster_volume"] = 0.0
        features["cluster_planarity"] = 0.0

    # ── Cofactor environment (shell-based) ──
    for radius in [5.0, 8.0, 12.0]:
        shell_residues = residues_within_radius(std_residues, cofactor_center, radius)
        tag = f"{int(radius)}A"
        features[f"n_residues_{tag}"] = len(shell_residues)

        # Net charge in shell
        net_charge = 0
        for res in shell_residues:
            name = res.get_resname()
            net_charge += FORMAL_CHARGE.get(name, 0)
        features[f"net_charge_{tag}"] = net_charge

    # H-bond donors/acceptors within 5 Å
    shell_5 = residues_within_radius(std_residues, cofactor_center, 5.0)
    features["n_hbond_donors_5A"] = sum(
        1 for r in shell_5 if r.get_resname() in HBOND_DONORS
    )
    features["n_hbond_acceptors_5A"] = sum(
        1 for r in shell_5 if r.get_resname() in HBOND_ACCEPTORS
    )

    # Coordinating Cys count (Sγ within 3 Å of any Fe)
    n_cys_coord = 0
    for res in std_residues:
        if res.get_resname() == "CYS" and "SG" in res:
            sg_coord = res["SG"].get_vector().get_array()
            for fe in fe_atoms:
                if np.linalg.norm(sg_coord - atom_coords[fe]) < 3.0:
                    n_cys_coord += 1
                    break
    features["n_cys_coordinating"] = n_cys_coord

    # Average distance to cofactor
    dists_to_cof = []
    for res in std_residues:
        ca = get_residue_ca(res)
        if ca is not None:
            dists_to_cof.append(np.linalg.norm(ca - cofactor_center))
    features["avg_dist_to_cofactor"] = np.mean(dists_to_cof) if dists_to_cof else 0.0

    # Cofactor burial depth (distance to farthest Cα as proxy for surface)
    features["cofactor_depth"] = np.min(dists_to_cof) if dists_to_cof else 0.0

    # Distance to nearest charged residues
    for charge_set, label in [(CHARGED_POS, "pos"), (CHARGED_NEG, "neg")]:
        min_d = 999.0
        for res in std_residues:
            if res.get_resname() in charge_set:
                ca = get_residue_ca(res)
                if ca is not None:
                    d = np.linalg.norm(ca - cofactor_center)
                    if d < min_d:
                        min_d = d
        features[f"nearest_charged_{label}_dist"] = min_d if min_d < 999.0 else 0.0

    # Solvent exposure proxy (cofactor atoms with few protein neighbors)
    all_protein_atoms = [a for res in std_residues for a in res.get_atoms()]
    if all_protein_atoms:
        ns = NeighborSearch(all_protein_atoms)
        exposed_count = 0
        for name in atom_names:
            neighbors = ns.search(atom_coords[name], 4.0, "A")
            if len(neighbors) < 5:  # few neighbors = more exposed
                exposed_count += 1
        features["cofactor_exposure"] = exposed_count / max(len(atom_names), 1)
    else:
        features["cofactor_exposure"] = 1.0

    # Number of cofactors in chain
    n_cofactors = 0
    for res in chain.get_residues():
        if res.get_resname() in COFACTOR_NAMES:
            n_cofactors += 1
    features["n_cofactors"] = n_cofactors

    # ── Amino acid composition (whole chain) ──
    chain_comp = compute_aa_composition(std_residues)
    for k, v in chain_comp.items():
        features[f"chain_{k}"] = v

    # Category fractions
    features["chain_frac_polar"] = sum(
        1 for r in std_residues if r.get_resname() in POLAR
    ) / max(n_residues, 1)
    features["chain_frac_nonpolar"] = sum(
        1 for r in std_residues if r.get_resname() in NONPOLAR
    ) / max(n_residues, 1)
    features["chain_frac_charged"] = sum(
        1 for r in std_residues if r.get_resname() in (CHARGED_POS | CHARGED_NEG)
    ) / max(n_residues, 1)
    features["chain_frac_aromatic"] = sum(
        1 for r in std_residues if r.get_resname() in AROMATIC
    ) / max(n_residues, 1)

    # ── Shell composition (8 Å) ──
    shell_8 = residues_within_radius(std_residues, cofactor_center, 8.0)
    shell_comp = compute_aa_composition(shell_8)
    for k, v in shell_comp.items():
        features[f"shell8_{k}"] = v

    # ── Weighted physicochemical properties ──
    # Whole chain
    chain_props = compute_weighted_properties(std_residues, aa_props)
    for k, v in chain_props.items():
        features[f"chain_{k}"] = v

    # 8 Å shell
    shell_props = compute_weighted_properties(shell_8, aa_props)
    for k, v in shell_props.items():
        features[f"shell8_{k}"] = v

    return features


# ─────────────────────────── Pipeline ───────────────────────────

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdb_dir = os.path.join(project_root, "pdb-files")

    # Load amino acid properties
    aa_props = load_aa_properties(project_root)
    logger.info(f"Loaded AA properties: {aa_props.shape}")

    # Load master dataset
    dataset_path = os.path.join(project_root, "Database Redox Pot Fe2S2 proteins.xlsx")
    ds = pd.read_excel(dataset_path, index_col=0)
    ds = ds[["PDB Code(s)", "mV", "pH"]].dropna(subset=["PDB Code(s)", "mV", "pH"])
    ds["base_pdb"] = ds["PDB Code(s)"].apply(lambda x: x[:4] if len(x) > 4 else x)

    # Get available PDB files
    pdb_files = {f.replace(".pdb", "") for f in os.listdir(pdb_dir) if f.endswith(".pdb")}
    logger.info(f"PDB files on disk: {len(pdb_files)}")

    # Filter dataset to rows with available PDB
    ds_available = ds[ds["base_pdb"].isin(pdb_files)].copy()
    unique_pdbs = sorted(ds_available["base_pdb"].unique())
    logger.info(f"Unique PDB structures to process: {len(unique_pdbs)}")
    logger.info(f"Total datapoints: {len(ds_available)}")

    parser = PDBParser(PERMISSIVE=True, QUIET=True)

    # Cache: base_pdb -> features dict (per chain, averaged)
    pdb_features_cache = {}
    skipped = []

    for pdb_id in unique_pdbs:
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.isfile(pdb_path):
            skipped.append(pdb_id)
            continue

        try:
            structure = parser.get_structure(pdb_id, pdb_path)
        except Exception as e:
            logger.warning(f"Failed to parse {pdb_id}: {e}")
            skipped.append(pdb_id)
            continue

        # Process each chain looking for cofactor
        chain_features_list = []
        model = structure[0]

        for chain in model.get_chains():
            cofactor_atoms = {}
            for residue in chain.get_residues():
                if residue.get_resname() in COFACTOR_NAMES:
                    for atom in residue.get_atoms():
                        if residue.get_resname() == "FES":
                            if atom.get_id() in COFACTOR_ATOMS_FES:
                                cofactor_atoms[atom.get_id()] = atom.get_vector().get_array()
                        elif residue.get_resname() == "FE":
                            if atom.get_id() in COFACTOR_ATOMS_FE:
                                cofactor_atoms[atom.get_id()] = atom.get_vector().get_array()

            if not cofactor_atoms:
                continue

            try:
                feats = extract_features_for_chain(chain, structure, cofactor_atoms, aa_props)
                chain_features_list.append(feats)
            except Exception as e:
                logger.warning(f"Error extracting features for {pdb_id} chain {chain.id}: {e}")

        if not chain_features_list:
            logger.warning(f"No cofactor-containing chains found for {pdb_id}")
            skipped.append(pdb_id)
            continue

        # Average features across chains (same as original pipeline)
        avg_features = {}
        all_keys = chain_features_list[0].keys()
        for key in all_keys:
            vals = [cf[key] for cf in chain_features_list if key in cf]
            avg_features[key] = np.mean(vals)

        avg_features["n_cofactor_chains"] = len(chain_features_list)
        pdb_features_cache[pdb_id] = avg_features

    logger.info(f"Successfully extracted features for {len(pdb_features_cache)} structures")
    if skipped:
        logger.info(f"Skipped {len(skipped)} structures: {skipped[:10]}...")

    # Build final dataset: merge features with experimental conditions
    rows = []
    for _, row in ds_available.iterrows():
        base = row["base_pdb"]
        if base not in pdb_features_cache:
            continue
        feat_row = pdb_features_cache[base].copy()
        feat_row["PDB_Code"] = row["PDB Code(s)"]
        feat_row["base_pdb"] = base
        feat_row["mV"] = row["mV"]
        feat_row["pH"] = row["pH"]
        rows.append(feat_row)

    df_final = pd.DataFrame(rows)

    # Reorder columns: identifiers first, then target, then pH, then features
    id_cols = ["PDB_Code", "base_pdb"]
    target_col = ["mV"]
    cond_col = ["pH"]
    feat_cols = [c for c in df_final.columns if c not in id_cols + target_col + cond_col]
    feat_cols = sorted(feat_cols)
    df_final = df_final[id_cols + target_col + cond_col + feat_cols]

    # Report
    n_features = len(cond_col) + len(feat_cols)
    logger.info(f"Final dataset shape: {df_final.shape}")
    logger.info(f"Total features (pH + structural): {n_features}")
    logger.info(f"NaN count: {df_final[cond_col + feat_cols].isnull().sum().sum()}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "features.xlsx")
    df_final.to_excel(output_path, index=False)
    logger.info(f"Saved features to {output_path}")

    # Print summary statistics for a few key features
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Structures processed: {len(pdb_features_cache)}")
    print(f"Datapoints:           {len(df_final)}")
    print(f"Features:             {n_features}")
    print(f"NaN values:           {df_final[cond_col + feat_cols].isnull().sum().sum()}")
    print()
    print("Sample feature stats (first 15):")
    for col in (cond_col + feat_cols)[:15]:
        vals = df_final[col]
        print(f"  {col:40s}  mean={vals.mean():10.3f}  std={vals.std():10.3f}  "
              f"min={vals.min():10.3f}  max={vals.max():10.3f}")


if __name__ == "__main__":
    main()
