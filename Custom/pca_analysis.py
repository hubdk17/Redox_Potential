"""
PCA Analysis (5, 6, 7, 8 components) on Custom Features
========================================================
Runs PCA dimensionality reduction, then trains XGBoost on each
reduced representation to compare predictive performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor

# GPU Detection
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

def main():
    # Load dataset
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "features.xlsx"))
    id_cols = ["PDB_Code", "base_pdb"]
    target_col = "mV"
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    X_raw = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    feature_names = feature_cols

    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    dims = [5, 6, 7, 8]
    results = {}

    for n_comp in dims:
        print(f"\n{'='*60}")
        print(f"PCA with {n_comp} components")
        print(f"{'='*60}")

        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        print(f"Explained variance per PC: {np.round(explained * 100, 2)}")
        print(f"Cumulative variance:       {np.round(cumulative * 100, 2)}")
        print(f"Total explained:           {cumulative[-1]*100:.2f}%")

        # Print top contributing features per PC
        print(f"\nTop 5 contributing features per PC:")
        for pc_idx in range(n_comp):
            loadings = pca.components_[pc_idx]
            top_idx = np.argsort(np.abs(loadings))[::-1][:5]
            contributions = [(feature_names[i], loadings[i]) for i in top_idx]
            print(f"  PC{pc_idx+1}: ", end="")
            print(", ".join(f"{name}({val:+.3f})" for name, val in contributions))

        # XGBoost CV on PCA-reduced data
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics = {"MAE": [], "RMSE": [], "R2": [], "Pearson": [], "Spearman": []}
        all_preds = np.full(len(y), np.nan)

        for train_idx, test_idx in kf.split(X_pca):
            X_tr, X_te = X_pca[train_idx], X_pca[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            xgb_params = dict(
                learning_rate=0.1, max_depth=3, n_estimators=500,
                min_child_weight=1, subsample=1.0, random_state=42,
                verbosity=0, tree_method="hist",
            )
            if USE_GPU:
                xgb_params["device"] = "cuda"
            model = XGBRegressor(**xgb_params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            metrics["MAE"].append(mean_absolute_error(y_te, y_pred))
            metrics["RMSE"].append(np.sqrt(mean_squared_error(y_te, y_pred)))
            metrics["R2"].append(r2_score(y_te, y_pred))
            metrics["Pearson"].append(pearsonr(y_te, y_pred)[0])
            metrics["Spearman"].append(spearmanr(y_te, y_pred)[0])
            all_preds[test_idx] = y_pred

        res = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
        results[n_comp] = res

        print(f"\nXGBoost 5-fold CV Results ({n_comp}D PCA):")
        for k, (m, s) in res.items():
            print(f"  {k:12s}  {m:8.3f} ± {s:.3f}")

        # Store predictions for scatter
        results[n_comp]["preds"] = all_preds.copy()
        results[n_comp]["pca_obj"] = pca
        results[n_comp]["explained"] = explained

    # ── Comparison table ──
    print(f"\n{'='*60}")
    print("COMPARISON ACROSS PCA DIMENSIONS")
    print(f"{'='*60}")
    print(f"{'Dims':>6s} | {'Var Explained':>14s} | {'MAE (mV)':>12s} | {'RMSE (mV)':>12s} | {'R²':>12s} | {'Pearson':>12s}")
    print("-" * 80)
    for d in dims:
        r = results[d]
        var_exp = np.sum(results[d]["explained"]) * 100
        print(f"{d:>6d} | {var_exp:>13.1f}% | "
              f"{r['MAE'][0]:>5.1f} ± {r['MAE'][1]:>4.1f} | "
              f"{r['RMSE'][0]:>5.1f} ± {r['RMSE'][1]:>4.1f} | "
              f"{r['R2'][0]:>5.3f} ± {r['R2'][1]:>.3f} | "
              f"{r['Pearson'][0]:>5.3f} ± {r['Pearson'][1]:>.3f}")

    # Also add full (96-dim) baseline for reference
    print(f"{'96':>6s} | {'100.0':>13s}% | {'48.3':>5s} ± {'3.6':>4s} | "
          f"{'71.4':>5s} ± {'11.0':>4s} | {'0.912':>5s} ± {'0.028':>5s} | "
          f"{'0.957':>5s} ± {'0.014':>5s}   (full features, from earlier run)")

    # ── Plots ──
    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)

    # Plot 1: Explained variance bar chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for idx, d in enumerate(dims):
        ax = axes[idx]
        exp = results[d]["explained"] * 100
        cum = np.cumsum(exp)
        bars = ax.bar(range(1, d+1), exp, color=plt.cm.viridis(np.linspace(0.3, 0.9, d)),
                      edgecolor="white", linewidth=0.5, label="Individual")
        ax.plot(range(1, d+1), cum, "ro-", markersize=5, label="Cumulative")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained (%)")
        ax.set_title(f"PCA {d}D ({cum[-1]:.1f}% total)")
        ax.set_xticks(range(1, d+1))
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
    plt.suptitle("Explained Variance by PCA Dimension", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "outputs", "pca_variance.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: 2x2 scatter plots (predicted vs actual for each PCA dim)
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    for idx, d in enumerate(dims):
        ax = axes[idx // 2][idx % 2]
        preds = results[d]["preds"]
        valid = ~np.isnan(preds)
        r2 = r2_score(y[valid], preds[valid])
        mae = mean_absolute_error(y[valid], preds[valid])

        ax.scatter(y[valid], preds[valid], alpha=0.5, s=25, c="steelblue",
                   edgecolors="white", linewidth=0.3)
        lims = [min(y.min(), preds[valid].min()) - 20,
                max(y.max(), preds[valid].max()) + 20]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual Em (mV)", fontsize=11)
        ax.set_ylabel("Predicted Em (mV)", fontsize=11)
        ax.set_title(f"PCA {d}D — R² = {r2:.3f}, MAE = {mae:.1f} mV", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("XGBoost Predictions with PCA-Reduced Features", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "outputs", "pca_scatter.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dims) + 1)
    labels = [f"{d}D" for d in dims] + ["96D\n(full)"]
    r2_means = [results[d]["R2"][0] for d in dims] + [0.912]
    r2_stds = [results[d]["R2"][1] for d in dims] + [0.028]
    mae_means = [results[d]["MAE"][0] for d in dims] + [48.3]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(x)))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=4, color=colors,
                  edgecolor="white", linewidth=0.5)
    # Annotate MAE on each bar
    for i, (bar, mae) in enumerate(zip(bars, mae_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + r2_stds[i] + 0.01,
                f"MAE={mae:.1f}", ha="center", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("XGBoost Performance: PCA Dimensions vs Full Features", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "outputs", "pca_comparison.png"),
                dpi=300)
    plt.close(fig)

    print("\nPlots saved to Custom/outputs/")


if __name__ == "__main__":
    main()
