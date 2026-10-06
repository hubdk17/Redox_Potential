import pandas as pd

# Check ANN dataset for same Volume = 0 bug
df = pd.read_excel('ANN/dataset_features/dataset_protein_8_3.xlsx', index_col=0)
print("ANN dataset shape:", df.shape)
print()

if 'n_chains' in df.columns:
    non_nan = df[df['n_chains'].notna()]
    print(f"Rows with n_chains present: {len(non_nan)} / {len(df)}")
    print()
    if 'Protein.Volume' in df.columns:
        print("Protein.Volume stats (non-NaN rows):")
        print(non_nan['Protein.Volume'].describe())
        print()
        print("Protein.ALA stats (non-NaN rows):")
        print(non_nan['Protein.ALA'].describe())
        print()
        # Check if Volume is always 0
        print(f"Protein.Volume all zeros? {(non_nan['Protein.Volume'] == 0).all()}")
        print(f"Bar.Volume all zeros? {(non_nan['Bar.Volume'] == 0).all()}")
