"""
XGBoost Cross-Validated Prediction + Permutation Feature Importance
===================================================================
Uses the custom-extracted features to train XGBoost, evaluate via
nested cross-validation, and rank all features by permutation importance.
"""

import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── GPU Detection ──
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False
logger = logging.getLogger(__name__)


def main():
    # ── Load dataset ──
    features_path = os.path.join(os.path.dirname(__file__), "features.xlsx")
    df = pd.read_excel(features_path)
    logger.info(f"Loaded dataset: {df.shape}")

    # Separate identifiers, target, and features
    id_cols = ["PDB_Code", "base_pdb"]
    target_col = "mV"
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    feature_names = feature_cols

    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logger.info(f"GPU acceleration: {'ENABLED (cuda)' if USE_GPU else 'DISABLED (cpu)'}")

    # ── XGBoost base params (GPU if available) ──
    xgb_base_params = {
        "random_state": 42,
        "verbosity": 0,
        "tree_method": "hist",
    }
    if USE_GPU:
        xgb_base_params["device"] = "cuda"

    # ── Hyperparameter grid ──
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300, 500],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 1.0],
    }

    # ── Cross-validation setup ──
    n_repeats = 5
    n_splits = 5
    total_folds = n_repeats * n_splits

    # Storage
    importances_perm = np.zeros((total_folds, X.shape[1]))
    importances_builtin = np.zeros((total_folds, X.shape[1]))
    metrics = {
        "MAE_train": [], "MAE_test": [], "RMSE_test": [],
        "R2_test": [], "Pearson_test": [], "Spearman_test": [],
    }
    all_predictions = np.full(len(y), np.nan)
    best_params_list = []

    logger.info(f"Starting {n_repeats}x {n_splits}-fold nested CV ({total_folds} total folds)...")
    start_time = time.time()

    fold_idx = 0
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat * 42)

        for train_idx, test_idx in kf.split(X):
            fold_idx += 1
            logger.info(f"Fold {fold_idx}/{total_folds} (repeat {repeat+1})")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV for hyperparameter tuning
            model = XGBRegressor(**xgb_base_params)
            grid_search = GridSearchCV(
                model,
                param_grid,
                scoring="neg_mean_absolute_error",
                cv=3,
                n_jobs=4,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params_list.append(grid_search.best_params_)

            # Predict
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            # Metrics
            metrics["MAE_train"].append(mean_absolute_error(y_train, y_pred_train))
            metrics["MAE_test"].append(mean_absolute_error(y_test, y_pred_test))
            metrics["RMSE_test"].append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            metrics["R2_test"].append(r2_score(y_test, y_pred_test))
            metrics["Pearson_test"].append(pearsonr(y_test, y_pred_test)[0])
            metrics["Spearman_test"].append(spearmanr(y_test, y_pred_test)[0])

            # Built-in importance
            importances_builtin[fold_idx - 1, :] = best_model.feature_importances_

            # Permutation importance on test set
            perm_result = permutation_importance(
                best_model, X_test, y_test,
                scoring="neg_mean_absolute_error",
                n_repeats=10,
                random_state=42,
                n_jobs=4,
            )
            importances_perm[fold_idx - 1, :] = np.abs(perm_result.importances_mean)

            # Store last-repeat predictions for scatter plot
            if repeat == n_repeats - 1:
                all_predictions[test_idx] = y_pred_test

    elapsed = time.time() - start_time
    logger.info(f"CV completed in {elapsed:.1f} seconds")

    # ── Aggregate metrics ──
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    cv_summary = {}
    for metric_name, values in metrics.items():
        mean_v = np.mean(values)
        std_v = np.std(values)
        cv_summary[metric_name] = {"mean": mean_v, "std": std_v}
        print(f"  {metric_name:20s}  {mean_v:8.3f} ± {std_v:6.3f}")

    # Most common best params
    from collections import Counter
    params_str = [str(p) for p in best_params_list]
    most_common = Counter(params_str).most_common(1)[0]
    print(f"\n  Most common best params ({most_common[1]}/{total_folds} folds):")
    print(f"  {most_common[0]}")

    # ── Feature importance ──
    mean_perm = np.mean(importances_perm, axis=0)
    std_perm = np.std(importances_perm, axis=0)
    mean_builtin = np.mean(importances_builtin, axis=0)

    results_df = pd.DataFrame({
        "Feature": feature_names,
        "Perm_Importance_MAE_increase": np.round(mean_perm, 4),
        "Perm_Importance_Std": np.round(std_perm, 4),
        "Builtin_Gain_Importance": np.round(mean_builtin, 4),
    }).sort_values("Perm_Importance_MAE_increase", ascending=False).reset_index(drop=True)

    # Save full importance table
    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
    results_df.to_excel(
        os.path.join(os.path.dirname(__file__), "outputs", "feature_importance_full.xlsx"),
        index=False,
    )

    # ── Print Top 50 ──
    top50 = results_df.head(50)
    print("\n" + "=" * 60)
    print("TOP 50 FEATURES BY PERMUTATION IMPORTANCE")
    print("=" * 60)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 120)
    print(top50.to_string(index=True))

    # Save top 50
    top50.to_excel(
        os.path.join(os.path.dirname(__file__), "outputs", "top50_features.xlsx"),
        index=False,
    )

    # ── Plot 1: Top 50 Feature Importance Bar Chart ──
    fig, ax = plt.subplots(figsize=(14, 14))
    y_pos = np.arange(len(top50))
    ax.barh(
        y_pos,
        top50["Perm_Importance_MAE_increase"],
        xerr=top50["Perm_Importance_Std"],
        align="center",
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top50))),
        edgecolor="white",
        linewidth=0.5,
        ecolor="gray",
        capsize=2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top50["Feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (MAE Increase, mV)", fontsize=11)
    ax.set_title("XGBoost Permutation Feature Importance — Top 50 Custom Descriptors", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "outputs", "feature_importance.png"),
        dpi=300,
    )
    plt.close(fig)
    logger.info("Saved feature importance plot.")

    # ── Plot 2: Predicted vs Actual Scatter ──
    valid_mask = ~np.isnan(all_predictions)
    if valid_mask.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            y[valid_mask], all_predictions[valid_mask],
            alpha=0.5, s=30, c="steelblue", edgecolors="white", linewidth=0.3,
        )
        lims = [min(y.min(), all_predictions[valid_mask].min()) - 20,
                max(y.max(), all_predictions[valid_mask].max()) + 20]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual Em (mV)", fontsize=12)
        ax.set_ylabel("Predicted Em (mV)", fontsize=12)
        r2 = r2_score(y[valid_mask], all_predictions[valid_mask])
        mae = mean_absolute_error(y[valid_mask], all_predictions[valid_mask])
        ax.set_title(f"Predicted vs Actual — R² = {r2:.3f}, MAE = {mae:.1f} mV", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(
            os.path.join(os.path.dirname(__file__), "outputs", "predicted_vs_actual.png"),
            dpi=300,
        )
        plt.close(fig)
        logger.info("Saved predicted vs actual scatter plot.")

    # ── Save CV metrics ──
    cv_df = pd.DataFrame({
        k: [f"{v['mean']:.3f} ± {v['std']:.3f}"] for k, v in cv_summary.items()
    })
    cv_df.to_excel(
        os.path.join(os.path.dirname(__file__), "outputs", "cv_results.xlsx"),
        index=False,
    )
    logger.info("All outputs saved to Custom/outputs/")


if __name__ == "__main__":
    main()
