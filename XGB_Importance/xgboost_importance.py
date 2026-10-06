import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Append workspace root for imports
sys.path.append(os.path.abspath("../"))
from utils import RHCF

def main():
    logger.info("Loading dataset...")
    # Load 8_3 dataset
    dataset_path = "../A/dataset_features/dataset_protein_8_3.xlsx"
    df = pd.read_excel(dataset_path, index_col=0)
    
    # Clean index
    if "PDB Code(s)" in df.columns:
        df = df.set_index("PDB Code(s)")
    elif df.index.name != "PDB Code(s)":
        df.index.name = "PDB Code(s)"
        
    df = df.dropna(subset=["mV"])
    
    # Separate features and target
    X_df = df.iloc[:, 1:]  # Exclude mV
    y = df.iloc[:, 0].values  # mV
    
    feature_names = X_df.columns.tolist()
    X = X_df.values
    
    logger.info(f"Loaded {X.shape[0]} samples and {X.shape[1]} raw features.")
    
    # Impute missing values first (so correlation analysis works)
    logger.info("Imputing missing values...")
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_imputed = imp.fit_transform(X)
    
    # Filter highly correlated features (covariation >= 0.99)
    logger.info("Filtering highly correlated features...")
    rhcf = RHCF(covariation=0.99)
    rhcf.fit(X_imputed)
    X_filtered = rhcf.transform(X_imputed)
    
    # Map back to remaining feature names
    selected_indices = rhcf.to_keep
    filtered_feature_names = [feature_names[idx] for idx in selected_indices]
    
    logger.info(f"Features remaining after correlation filter: {X_filtered.shape[1]}")
    
    # Setup 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Arrays to store importance metrics across folds
    importances_perm = np.zeros((5, X_filtered.shape[1]))
    importances_builtin = np.zeros((5, X_filtered.shape[1]))
    
    logger.info("Starting 5-fold cross-validation and importance calculations...")
    startTime = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_filtered)):
        logger.info(f"Processing Fold {fold + 1}/5...")
        
        X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train XGBRegressor with tuned hyperparameters
        model = XGBRegressor(
            learning_rate=0.01,
            max_depth=3,
            n_estimators=100,
            min_child_weight=1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 1. Built-in Feature Importance (Information Gain)
        importances_builtin[fold, :] = model.feature_importances_
        
        # 2. Permutation Feature Importance on test fold
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        # We take the absolute value of the change to represent positive importance.
        importances_perm[fold, :] = np.abs(result.importances_mean)
        
    executionTime = time.time() - startTime
    logger.info(f"Feature importance calculations finished in {executionTime:.2f} seconds.")
    
    # Compute mean and standard dev across folds
    mean_perm = np.mean(importances_perm, axis=0)
    std_perm = np.std(importances_perm, axis=0)
    mean_builtin = np.mean(importances_builtin, axis=0)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "Feature": filtered_feature_names,
        "Permutation_Importance_MAE_increase": mean_perm,
        "Std_Permutation_Importance": std_perm,
        "Builtin_Importance_Gain": mean_builtin
    })
    
    # Sort by mean permutation importance descending
    results_df = results_df.sort_values(by="Permutation_Importance_MAE_increase", ascending=False).reset_index(drop=True)
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Save Excel sheet
    excel_path = "outputs/permutation_importance.xlsx"
    results_df.to_excel(excel_path, index=False)
    logger.info(f"Excel summary saved to {excel_path}.")
    
    # Plot top 50 features
    top_n = 50
    top_features = results_df.head(top_n)
    
    plt.figure(figsize=(14, 14))
    y_pos = np.arange(top_n)
    
    plt.barh(
        y_pos,
        top_features["Permutation_Importance_MAE_increase"],
        xerr=top_features["Std_Permutation_Importance"],
        align="center",
        color="skyblue",
        ecolor="gray",
        capsize=3
    )
    plt.yticks(y_pos, top_features["Feature"], fontsize=8)
    plt.gca().invert_yaxis()  # top-down order
    plt.xlabel("Permutation Importance (MAE Increase in mV)")
    plt.title("XGBoost Permutation Feature Importance (Top 50 Descriptors)")
    plt.tight_layout()
    
    plot_path = "outputs/feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Importance plot saved to {plot_path}.")
    
    # Print Top 50 features
    print("\n--- Top 50 Most Important Descriptors ---")
    pd.set_option('display.max_rows', 100)
    print(top_features[["Feature", "Permutation_Importance_MAE_increase", "Builtin_Importance_Gain"]].to_string(index=True))

if __name__ == "__main__":
    main()
