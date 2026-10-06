"""Reproduce the bug in the feature calculator's mapped property computation."""
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath('.'))
from utils import amm_names

# Load table_amm exactly as features_calculator does
table_amm = pd.read_csv("tableAmm.txt", sep="\t", index_col=1).iloc[:, 1:]

# Simulate dict_cont as it would be after feature_conteggio()
# Using real counts similar to what 4fxc would have
dict_cont = {}
for aa in amm_names:
    dict_cont["Protein." + aa] = 0

# Set some typical non-zero amino acid counts
dict_cont["Protein.ALA"] = 10
dict_cont["Protein.GLY"] = 8
dict_cont["Protein.LEU"] = 12
dict_cont["Protein.CYS"] = 4

print("=== Before mapped feature computation ===")
print(f"Protein.ALA = {dict_cont['Protein.ALA']}")
print(f"table_amm.loc['ALA', 'Volume'] = {table_amm.loc['ALA', 'Volume']}")
print(f"Expected ALA contribution to Volume = {table_amm.loc['ALA', 'Volume'] * dict_cont['Protein.ALA']}")
print()

# Now do EXACTLY what features_calculator.py lines 197-205 does
name = "Protein"
for col in table_amm.columns:
    values = table_amm[col]
    val_feature = np.sum(
        [
            values[nome] * dict_cont[name + f".{nome}"]
            for nome in amm_names
        ]
    )
    dict_cont[name + f".{col}"] = val_feature

print("=== After mapped feature computation ===")
print(f"Protein.Volume = {dict_cont['Protein.Volume']}")
print(f"Protein.Hydrophobicity = {dict_cont['Protein.Hydrophobicity']}")
print()

# Debug: the issue may be that "Protein.ALA" gets OVERWRITTEN
# because amm_names contains ALA, and table_amm.columns might 
# create keys that collide. Let's check:
print("=== Checking collision ===")
print(f"amm_names = {amm_names}")
print()
print(f"table_amm.columns = {list(table_amm.columns)}")
print()

# Let's step through the loop manually for 'Volume'
col = "Volume"
values = table_amm[col]
print(f"=== Manual step-through for col='{col}' ===")
for nome in amm_names:
    v = values[nome]
    c = dict_cont.get(name + f".{nome}", "KEY_MISSING")
    product = v * c if isinstance(c, (int, float)) else "N/A"
    if product != 0 and product != "N/A":
        print(f"  {nome}: values={v}, count={c}, product={product}")

# THE BIG QUESTION: what is dict_cont["Protein.ALA"] at this point?
print()
print(f"=== dict_cont['Protein.ALA'] AFTER loop = {dict_cont.get('Protein.ALA', 'MISSING')} ===")
print()

# Check if the loop OVERWRITES Protein.ALA before Volume is computed
# Let's check which column names in table_amm could cause this
for col in table_amm.columns:
    key = name + f".{col}"
    if key in [name + "." + aa for aa in amm_names]:
        print(f"COLLISION FOUND: col='{col}' creates key '{key}' which is an amino acid count key!")
