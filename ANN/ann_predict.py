# ann_predict.py: ML pipeline using Artificial Neural Network (MLPRegressor)

import os
import logging
import warnings
import time
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor

sys.path.append(os.path.abspath("../"))
from utils import RHCF, RemoveOutliar

warnings.filterwarnings("ignore")

# initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model dictionary for ANN
models_dict = {
    "ANN": [
        MLPRegressor(max_iter=10, random_state=42, early_stopping=False),
        {
            "regressor__feature_selection__estimator__alpha": [10],
            "regressor__feature_selection__estimator__l1_ratio": [0.75],
            "regressor__regressor__hidden_layer_sizes": [(100,)],
            "regressor__regressor__activation": ["relu"],
            "regressor__regressor__alpha": [0.001],
        }
    ]
}

name_models = ["ANN"]
models_with_scaling = ["ANN"]
models_with_fs = ["ANN"]
features_list = []

n_jobs = 4
path_inputs = "dataset_features/"
name_output_proteins = "analysis_dataset"
name_output_result = "results_dataset"
if "outputs" not in os.listdir():
    os.mkdir("outputs")
path_dir_output = "outputs/"

# Radii combination matching features_calculator.py
list_Bar_radius = [8, 12, 16]
list_CofAtom_radius = [3, 4]

# ML parameters
n_repeat = 5  # out nested cross validation (using 5 for speed while maintaining statistical rigour)
split_tuning = 5  # inner nested cross validation used for hyperparameters tuning
CV = 5  # cross validation for hyperparameters tuning
opt = "neg_mean_absolute_error"
covariation = 0.99
alpha = 10
l1_ratio = 0.75
max_iter = 1000

data = pd.read_excel("../Database Redox Pot Fe2S2 proteins.xlsx", index_col=0)
proteins = list(data["PDB Code(s)"])

# Inizialize results DataFrame
models_scan = pd.DataFrame(
    columns=[
        "name_model_radius",
        "estimator",
        "bar_radius",
        "CofAtom_radius",
        "MAE_train",
        "sd_train",
        "MAE_test",
        "sd_test",
        "mean_error_kfold",
        "sd_error_kfold",
        "RMSE",
        "sd_RMSE",
        "R2",
        "sd_R2",
        "Pearson",
        "sd_Pearson",
        "Spearman",
        "sd_Spearman",
        "Mae_test_list",
        "Error_Kfold_list",
        "RMSE_list",
        "R2_test_list",
        "Pearson_test_list",
        "Spearman_test_list",
        "Best_params",
        "selected_features",
        "n_features",
    ]
)

dict_proteins = dict()
dict_proteins_em = dict()

en = ElasticNet(max_iter=max_iter, alpha=alpha, l1_ratio=l1_ratio)
estimator_feature = SelectFromModel(en)
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
df_proteins = None
df_proteins_em = None
index_line = 0

for bar_radius in list_Bar_radius:
    for CofAtom_radius in list_CofAtom_radius:
        file_name = f"dataset_protein_{str(bar_radius)}_{str(CofAtom_radius)}.xlsx"
        full_path = os.path.join(path_inputs, file_name)
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {file_name}. Skipping.")
            continue
            
        logger.info(f"processing: {file_name}")

        df_pmOrig = pd.read_excel(full_path, sheet_name="Sheet1", index_col=0).set_index("PDB Code(s)")
        df_pm = df_pmOrig.copy()
        
        for estimator in name_models:
            dict_proteins[estimator] = dict()
            dict_proteins_em[estimator] = dict()
            cont = 0
            for key in df_pm.index:
                dict_proteins[estimator][f"{key}_{str(cont)}"] = 0
                dict_proteins_em[estimator][f"{key}_{str(cont)}"] = []
                cont += 1

        if len(features_list) != 0:
            df_pm = df_pm.loc[:, features_list]

        X = df_pm.iloc[:, 1:].values
        y = df_pm.iloc[:, 0].values
        labels = list(dict_proteins[name_models[0]].keys())
        labels = np.array(labels)

        logger.info("Hyperparameter Tuning")

        if df_proteins is None:
            df_proteins = pd.DataFrame(index=labels)

        if df_proteins_em is None:
            df_proteins_em = pd.DataFrame(index=labels)

        DATA_dict = {
            name_model: {
                "mae_train": [],
                "mae_test": [],
                "RMSE": [],
                "R2": [],
                "Pearson": [],
                "Spearman": [],
                "EOKfold": [],
            }
            for name_model in name_models
        }

        startTime = time.time()

        i = 0
        for j in range(n_repeat):
            kfold = KFold(n_splits=split_tuning, random_state=j, shuffle=True)
            for train_index, test_index in kfold.split(X):
                logger.info(f"Split_train_test: {i}")
                i = i + 1
                X_train = X[train_index, :]
                y_train = y[train_index]
                labels_train = labels[train_index]
                X_test = X[test_index, :]
                y_test = y[test_index]
                labels_test = labels[test_index]

                # preprocess
                fillph = imp.fit(X_train)
                X_train = fillph.transform(X_train)
                X_test = fillph.transform(X_test)
                
                remove_hcf = RHCF(covariation=covariation).fit(X_train)
                X_train = remove_hcf.transform(X_train)
                X_test = remove_hcf.transform(X_test)
                
                X_train2 = X_train.copy()
                X_test2 = X_test.copy()

                for estimator in name_models:
                    if estimator in models_with_scaling:
                        sc = StandardScaler().fit(X_train2)
                        X_train2 = sc.transform(X_train2)
                        X_test2 = sc.transform(X_test2)

                    pipeline = []
                    if estimator in models_with_fs:
                        pipeline.append(("feature_selection", estimator_feature))

                    pipeline.append(("regressor", models_dict[estimator][0]))
                    pipe = Pipeline(pipeline)

                    treg = TransformedTargetRegressor(regressor=pipe, transformer=None)

                    optEstimator = GridSearchCV(
                        treg,
                        models_dict[estimator][1],
                        scoring=opt,
                        cv=CV,
                        n_jobs=n_jobs,
                    )

                    _ = optEstimator.fit(X_train2, y_train)
                    
                    DATA_dict[estimator]["EOKfold"].append(np.abs(optEstimator.best_score_))
                    DATA_dict[estimator]["mae_train"].append(mean_absolute_error(optEstimator.predict(X_train2), y_train))
                    DATA_dict[estimator]["mae_test"].append(mean_absolute_error(optEstimator.predict(X_test2), y_test))
                    DATA_dict[estimator]["RMSE"].append(np.sqrt(mean_squared_error(optEstimator.predict(X_test2), y_test)))
                    DATA_dict[estimator]["R2"].append(r2_score(y_test, optEstimator.predict(X_test2)))
                    DATA_dict[estimator]["Pearson"].append(pearsonr(y_test, optEstimator.predict(X_test2))[0])
                    DATA_dict[estimator]["Spearman"].append(spearmanr(y_test, optEstimator.predict(X_test2))[0])

                    values = list(np.abs(optEstimator.predict(X_test2) - y_test))
                    for label, value in zip(labels_test, values):
                        dict_proteins[estimator][label] += value

                    em_values = list(optEstimator.predict(X_test2))
                    for label, value in zip(labels_test, em_values):
                        dict_proteins_em[estimator][label].append(value)

        for label in labels:
            for estimator in name_models:
                dict_proteins[estimator][label] = (dict_proteins[estimator][label] / n_repeat)

        for estimator in name_models:
            df_proteins[f"{estimator}_{str(bar_radius)}_{str(CofAtom_radius)}"] = [
                dict_proteins[estimator][key] for key in dict_proteins[estimator].keys()
            ]
            df_proteins_em[f"{estimator}_{str(bar_radius)}_{str(CofAtom_radius)}"] = [
                dict_proteins_em[estimator][key] for key in dict_proteins_em[estimator].keys()
            ]

        if "mV" not in df_proteins.columns:
            df_proteins.insert(loc=0, column="mV", value=df_pm["mV"].values)

        if "mV" not in df_proteins_em.columns:
            df_proteins_em.insert(loc=0, column="mV", value=df_pm["mV"].values)

        df_proteins.to_csv(path_dir_output + name_output_proteins + ".csv")
        df_proteins_em.to_csv(path_dir_output + name_output_proteins + "_em.csv")
        
        model_line = 0
        for estimator in name_models:
            X2 = X.copy()
            fillph = imp.fit(X2)
            X2 = fillph.transform(X2)
            remove_hcf = RHCF(covariation=covariation).fit(X2)
            X2 = remove_hcf.transform(X2)
            if estimator in models_with_scaling:
                X2 = StandardScaler().fit(X2).transform(X2)

            pipeline = []
            if estimator in models_with_fs:
                pipeline.append(("feature_selection", estimator_feature))

            pipeline.append(("regressor", models_dict[estimator][0]))
            pipe = Pipeline(pipeline)
            treg = TransformedTargetRegressor(regressor=pipe, transformer=None)

            optEstimator = GridSearchCV(treg, models_dict[estimator][1], scoring=opt, cv=CV, n_jobs=n_jobs)
            best_model = optEstimator.fit(X2, y)
            best_params = best_model.best_params_

            if estimator in models_with_fs:
                alphaTot = best_model.best_params_["regressor__feature_selection__estimator__alpha"]
                l1_ratioTot = best_model.best_params_["regressor__feature_selection__estimator__l1_ratio"]
                enTot = ElasticNet(max_iter=max_iter, alpha=alphaTot, l1_ratio=l1_ratioTot)
                estimator_featureTot = SelectFromModel(enTot)
                fs = estimator_featureTot.fit(X2, y)

                C = np.array(np.abs(fs.estimator_.coef_) > 0)
                selected_features = []
                features_index = [1 + el for el in remove_hcf.to_keep]
                for el, el2 in zip(df_pm.columns[features_index], C):
                    if el2 == True:
                        selected_features.append(el)
            else:
                selected_features = []

            models_scan.loc[(index_line + model_line)] = [
                estimator+"_"+str(bar_radius)+"_"+str(CofAtom_radius),
                estimator,
                int(bar_radius),
                int(CofAtom_radius),
                np.mean(DATA_dict[estimator]["mae_train"]),
                np.std(DATA_dict[estimator]["mae_train"]),
                np.mean(DATA_dict[estimator]["mae_test"]),
                np.std(DATA_dict[estimator]["mae_test"]),
                np.mean(DATA_dict[estimator]["EOKfold"]),
                np.std(DATA_dict[estimator]["EOKfold"]),
                np.mean(DATA_dict[estimator]["RMSE"]),
                np.std(DATA_dict[estimator]["RMSE"]),
                np.mean(DATA_dict[estimator]["R2"]),
                np.std(DATA_dict[estimator]["R2"]),
                np.mean(DATA_dict[estimator]["Pearson"]),
                np.std(DATA_dict[estimator]["Pearson"]),
                np.mean(DATA_dict[estimator]["Spearman"]),
                np.std(DATA_dict[estimator]["Spearman"]),
                list(DATA_dict[estimator]["mae_test"]),
                list(DATA_dict[estimator]["EOKfold"]),
                list(DATA_dict[estimator]["RMSE"]),
                list(DATA_dict[estimator]["R2"]),
                list(DATA_dict[estimator]["Pearson"]),
                list(DATA_dict[estimator]["Spearman"]),
                best_params,
                list(selected_features),
                len(selected_features),
            ]
            model_line += 1

        executionTime = time.time() - startTime
        print(f"Dataset {file_name} processed. Execution time in seconds: {executionTime}")

        models_scan.to_excel(os.path.join(path_dir_output, f"{name_output_result}.xlsx"))
        index_line += len(name_models)

print("ANN predictions completed successfully. All results stored in outputs/.")
