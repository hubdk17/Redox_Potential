"""Diagnostic script to understand the feature dataset."""
import pandas as pd
import numpy as np

df = pd.read_excel('A/dataset_features/dataset_protein_8_3.xlsx', index_col=0)
df = df.set_index('PDB Code(s)')

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape} ({df.shape[0]} proteins x {df.shape[1]} features)")
print(f"Total NaN values: {df.isnull().sum().sum()}")
print(f"Rows with any NaN: {df.isnull().any(axis=1).sum()} / {len(df)}")
print(f"Rows with zero NaN: {(~df.isnull().any(axis=1)).sum()} / {len(df)}")
print()

# Count NaN per row to understand which rows are affected
nan_per_row = df.isnull().sum(axis=1)
print("NaN count per row distribution:")
print(nan_per_row.value_counts().sort_index().head(10))
print()

# Which columns are always NaN vs never NaN vs mixed?
always_nan = df.columns[df.isnull().all()]
never_nan = df.columns[df.notnull().all()]
mixed_nan = df.columns[df.isnull().any() & df.notnull().any()]
print(f"Columns always NaN: {len(always_nan)}")
print(f"Columns never NaN: {len(never_nan)}")
print(f"Columns mixed NaN: {len(mixed_nan)}")
print()

# Which prefixes exist?
prefixes = set()
for c in df.columns:
    parts = c.split('.')
    if len(parts) >= 2:
        prefixes.add(parts[0])
print("Feature prefixes:", sorted(prefixes))
print()

# Check Volume columns specifically
vol_cols = [c for c in df.columns if 'Volume' in c]
print("Volume columns:")
for c in vol_cols:
    non_zero = df[c].ne(0).sum()
    nan_count = df[c].isna().sum()
    mean_val = df[c].mean()
    print(f"  {c}: non-zero={non_zero}, NaN={nan_count}, mean={mean_val:.4f}")
print()

# Check what features are populated for a row that has data
# (pick first row with fewest NaN)
best_row_idx = nan_per_row.idxmin()
best_row = df.loc[best_row_idx]
print(f"Best row (fewest NaN): {best_row_idx}, NaN count: {nan_per_row[best_row_idx]}")
print()

# Check amino acid counts vs mapped properties for that row
table_amm = pd.read_csv('tableAmm.txt', sep='\t', index_col=1).iloc[:, 1:]
std_aa = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
          'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

for prefix in ['Protein', 'Bar']:
    print(f"--- {prefix} features for {best_row_idx} ---")
    # Amino acid counts
    aa_counts = {aa: best_row.get(f'{prefix}.{aa}', 'MISSING') for aa in std_aa}
    total_count = sum(v for v in aa_counts.values() if isinstance(v, (int, float)) and not np.isnan(v))
    print(f"  Total AA count: {total_count}")
    
    # Manual Volume calculation
    manual_vol = sum(table_amm.loc[aa, 'Volume'] * aa_counts[aa] for aa in std_aa if isinstance(aa_counts.get(aa), (int, float)))
    stored_vol = best_row.get(f'{prefix}.Volume', 'MISSING')
    print(f"  Manual Volume (sum of count*vol): {manual_vol:.2f}")
    print(f"  Stored Volume in dataset:         {stored_vol}")
    print(f"  MATCH: {abs(manual_vol - stored_vol) < 0.1 if isinstance(stored_vol, (int, float)) and not np.isnan(stored_vol) else 'N/A'}")
    print()

# Show non-zero, non-NaN feature stats for the "never NaN" columns
print("=" * 70)
print("NEVER-NaN COLUMNS (first 30):")
print("=" * 70)
for c in sorted(never_nan)[:30]:
    vals = df[c]
    print(f"  {c}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
          f"min={vals.min():.4f}, max={vals.max():.4f}, "
          f"all_zero={vals.eq(0).all()}")

print()
print("=" * 70)
print("ALWAYS-NaN COLUMNS (first 30):")
print("=" * 70)
for c in sorted(always_nan)[:30]:
    print(f"  {c}")

# Check which prefixes are always NaN
print()
print("=" * 70)
print("ALWAYS-NaN COLUMNS by PREFIX:")
print("=" * 70)
nan_prefixes = {}
for c in always_nan:
    parts = c.split('.')
    p = parts[0] if len(parts) >= 2 else 'NO_PREFIX'
    nan_prefixes[p] = nan_prefixes.get(p, 0) + 1
for p, count in sorted(nan_prefixes.items(), key=lambda x: -x[1]):
    print(f"  {p}: {count} columns")
