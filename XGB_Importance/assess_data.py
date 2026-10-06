"""Assess available data: PDB files with matching redox potential labels."""
import pandas as pd
import os
from collections import OrderedDict

# Load master dataset
ds = pd.read_excel('Database Redox Pot Fe2S2 proteins.xlsx', index_col=0)
ds = ds[['PDB Code(s)', 'mV', 'pH']].dropna(subset=['PDB Code(s)', 'mV', 'pH'])

# Get all PDB files we have
pdb_dir = 'pdb-files'
pdb_files = [f.replace('.pdb', '') for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
print(f"PDB files downloaded: {len(pdb_files)}")
print(f"Total dataset rows: {len(ds)}")
print(f"Unique PDB codes in dataset: {ds['PDB Code(s)'].nunique()}")
print()

# Map extended IDs to their wild-type base
ds['base_pdb'] = ds['PDB Code(s)'].apply(lambda x: x[:4] if len(x) > 4 else x)

# Which base PDBs do we have files for?
matched = ds[ds['base_pdb'].isin(pdb_files)]
print(f"Rows where base PDB file exists: {len(matched)}")
print(f"Unique base PDB structures: {matched['base_pdb'].nunique()}")
print()

# Show the distribution
print("Rows per base PDB (top 15):")
print(matched['base_pdb'].value_counts().head(15))
print()

# Show mV distribution for matched data
print("mV stats for matched rows:")
print(matched['mV'].describe())
print()

print("pH stats for matched rows:")
print(matched['pH'].describe())
print()

# Show list of available base PDBs
print("All available base PDB IDs:")
available = sorted(matched['base_pdb'].unique())
print(available)
print(f"\nTotal usable datapoints: {len(matched)}")
