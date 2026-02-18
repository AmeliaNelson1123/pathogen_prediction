# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import backend as K
import re
from pathlib import Path, PurePosixPath
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
from sklearn.inspection import permutation_importance
import tempfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ----------------------------------------------
# --- Global Variables -------------------------
# ----------------------------------------------

# STEP 2: Edit any of these variables in these sections

# saving global variables
TEST_SIZE = .22  # for validation (Pick a number between .1-.99 to reflect percentage)
RANDOM_STATE = 42  # for repeatability
# developing results table to plot
FILE_INFO = Path("ListeriaSoil_clean_log.csv") # Path("INPUT FILE NAME HERE") i.e. Path("SalCampChicken_clean.csv")

# TODO OPTIONS:
# OPTIONS FOR LISTERIA:
    # Number of Listia Isolates: origianl predictor present in dataset
    # binary_listeria_presense:
# Y_COL = "Salmon_or_camp_test"
Y_COL = "binary_listeria_presense"

# If want all strings/catagorical data to be encoded in 1-hot vectors 
# (aka want to transform arbitrary strings into integer values)
ENCODE_STR = False

# ----------------------------------------------
# --- Data Prep --------------------------------
# ----------------------------------------------

# Data Preperation
def data_prep():
    """
    ----- inputs -----
    ----- outputs ----
    df: pandas df
        processed anonymzied data (string columns representing intervals split into min and max, then put as minimum and maximum values for those columns)
    """

    df = pd.read_csv(Path(FILE_INFO.name))

    # if salcamp column, then adding it to the dataset
    if Y_COL == "Salmon_or_camp_test":
        df["Salmon_or_camp_test"] = (df["CampylobacterAnalysis30ml"] == "Positive") | (df["SalmonellaSPAnalysis"] == "Positive")
    if Y_COL == "binary_listeria_presense":
        df['binary_listeria_presense'] = [row_val if row_val == 0 else 1 for row_val in df['Number of Listeria isolates obtained']]

    # switching missing values and weird failures in writing to np.inf bc pandas didnt handle properly
    df = df.replace("#NAME?", -np.inf)
    df = df.fillna(-np.inf)

    # replacing inf with max number that is not max number + 100 in dict (FOR NOT JUST 99999999)
    df = df.replace(np.inf, 99999)
    # replacing -inf with min number (not -inf) - 100 in dict (FOR NOT JUST -99999999)
    df = df.replace(-np.inf, -99999)

    df = df.dropna(axis=1, how="all")

    if ENCODE_STR:
        df = pd.get_dummies(df)
    return df

# ----------------------------------------------
# --- Train-Test -------------------------------
# ----------------------------------------------

def get_train_test(
    df, y_col=Y_COL, scaling_used=True
):
    """
    ----- inputs -----
    df: pandas dict
        processed data (all numerics)
    Y_COL: str
        string of y labels
    TEST_SIZE: int
        % want test set to be of full data
    scaling_used: boolean
        whether to test scaled data and original data (True) or only original data (False)
    ----- outputs ----
    data_testing: dict[str=scalingType][str=y/X train/test label][pd.DataFrame]
        dictionary contianing
            * string of scaling type (standard scalar, orig)
                * string of what dataset grabbing (X_train, X_test, y_train, y_test)
                    * corresponding data in a pandas dataframe
        "
    """

    # indexes for test set
    X = df.drop(columns=Y_COL)
    y = df[Y_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # data columns
    data_columns = X.columns

    if scaling_used:  # if want to run on scaled and original data
        # testing all with and without scaled data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_test_scaled = scaler.transform(X_test.values)

        data_testing = {
            "columns": data_columns,
            "standard_scalar": {
                "X_train": X_train_scaled,
                "X_test": X_test_scaled,
                "y_train": y_train,  # using unscaled y
                "y_test": y_test,  # using unscaled y
            },
            "orig": {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            },
        }

        return data_testing

    else:  # if only want to run on original data
        data_testing = {
            "columns": data_columns,
            "orig": {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
        }
        return data_testing

# ----------------------------------------------
# --- Modeling Section -------------------------
# ----------------------------------------------

def test_logistic_reg(data_testing):
    """
    ----- inputs -----
    data_testing: dict[str=scalingType][str=y/X train/test label][pd.DataFrame]
        dictionary contianing
            * string of scaling type (standard scalar, orig)
                * string of what dataset grabbing (X_train, X_test, y_train, y_test)
                    * corresponding data in a pandas dataframe
    ----- outputs ----

    """

    log_reg_results = []

    # STEP 3: can edit HYPERPARAMETERS: for logistic regression variables
    # c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 2, 4, 8, 32, 56, 100]
    c_vals = [0.01, 0.1, 1, 4, 8]
    # lr_ratios = [0, 0.5, 1] # 0 = l2 penalty, 1 = l1 penalty, 0.5 = elasticnet penalty (both L1 and L2)
    lr_ratios = [
        0,
        1,
    ]  # 0 = l2 penalty, 1 = l1 penalty, 0.5 = elasticnet penalty (both L1 and L2)

    # grid searching model results for log reg on all types of data with all types of inputs
    for scalar_type in tqdm(data_testing.keys(), desc="logistic regression scaled vs original"):

        # breakpoint()

        if scalar_type == 'columns':
            continue
        # print("data testing is :", type(data_testing), " \n", data_testing)
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]
        feature_names = data_testing["columns"].tolist()

        # breakpoint()

        # going through possible log reg combos
        for c_val in c_vals:
            for lr_rat in lr_ratios:
                # modeling portion
                model = LogisticRegression(C=c_val, l1_ratio=lr_rat)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # validation
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)

                # print(f"Accuracy: {accuracy}")
                # print(f"Precision: {precision}")
                # print(f"Recall: {recall}")
                # print(f"F1 Score: {f1}")
                # print(f"Confusion Matrix:\n{conf_matrix}")

                # getting feature importance
                coefficients = model.coef_.ravel()
                feature_imp = dict(zip(feature_names, coefficients))
                feature_imp_json = json.dumps({k: float(v) for k, v in feature_imp.items()})

                # getting permutation importance
                perm = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=10,
                    RANDOM_STATE=RANDOM_STATE,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # STEP 3, add any hyperparameters to each of these results/outputs: saving results to dictfile_path: str
                log_reg_results.append(
                    {
                        "file name": FILE_INFO.name,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": TEST_SIZE,
                        "random state": RANDOM_STATE,
                        "scalar_status": scalar_type,
                        "y variable used": Y_COL,
                        "model used": "logistic regression",
                        "logistic_reg_c": c_val,
                        "lr_ratios": lr_rat,
                        "nn_layers": np.nan,
                        "nn_neurons": np.nan,
                        "nn_batch_size": np.nan,
                        "nn_epochs": np.nan,
                        "dt_max_depth": np.nan,
                        "dt_min_samples_split": np.nan,
                        "svm_c_val": np.nan,
                        "svm_kernel": np.nan,
                        "knn_n_neighbor": np.nan,
                        "knn_weights": np.nan,
                        "gbm_learning_rate": np.nan,
                        "gbm_n_estimator": np.nan,
                        "coefficient_importance": feature_imp_json,
                        "permutation_importance": perm_imp_json,
                    }
                )

    return log_reg_results

def test_svm(data_testing):
    """
    ----- inputs -----
    data_testing: dict[str=scalingType][str=y/X train/test label][pd.DataFrame]
        dictionary contianing
            * string of scaling type (standard scalar, orig)
                * string of what dataset grabbing (X_train, X_test, y_train, y_test)
                    * corresponding data in a pandas dataframe
    ----- outputs ----

    """

    # results table
    svm_results = []

    # STEP 3: can edit HYPERPARAMETERs: for svm variables
    c_vals = [0.01, 0.1, 1, 4, 8]
    svm_kernels = ['linear', 'rbf', 'sigmoid']

    # grid searching model results for svm on all types of data with all types of inputs
    for scalar_type in tqdm(data_testing.keys(), desc="svm scaled vs original"):
        if scalar_type == 'columns':
            continue
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]
        feature_names = data_testing["columns"].tolist()

        # going through possible svm combos
        for c_val in c_vals:
            for svm_kernel in svm_kernels:
                # modeling portion
                model = SVC(C=c_val, kernel=svm_kernel)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # validation
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)

                # getting feature importance
                coefficients = np.nan
                feature_imp = np.nan
                feature_imp_json = np.nan
                if svm_kernel == 'linear':
                    coefficients = model.coef_.ravel()
                    feature_imp = dict(zip(feature_names, coefficients))
                    feature_imp_json = json.dumps({k: float(v) for k, v in feature_imp.items()})

                # getting permutation importance
                perm = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=10,
                    RANDOM_STATE=RANDOM_STATE,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # STEP 3, add any hyperparameters to each of these results/outputs: saving results to dict
                svm_results.append(
                    {
                        "file name": FILE_INFO.name,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": TEST_SIZE,
                        "random state": RANDOM_STATE,
                        "scalar_status": scalar_type,
                        "y variable used": Y_COL,
                        "model used": "svm",
                        "logistic_reg_c": np.nan,
                        "lr_ratios": np.nan,
                        "nn_layers": np.nan,
                        "nn_neurons": np.nan,
                        "nn_batch_size": np.nan,
                        "nn_epochs": np.nan,
                        "dt_max_depth": np.nan,
                        "dt_min_samples_split": np.nan,
                        "svm_c_val": c_val,
                        "svm_kernel": svm_kernel,
                        "knn_weights": np.nan,
                        "gbm_learning_rate": np.nan,
                        "gbm_n_estimator": np.nan,
                        "coefficient_importance": feature_imp_json,
                        "permutation_importance": perm_imp_json,
                    }
                )

    return svm_results

def test_knn(data_testing):
    """
    ----- inputs -----
    data_testing: dict[str=scalingType][str=y/X train/test label][pd.DataFrame]
        dictionary contianing
            * string of scaling type (standard scalar, orig)
                * string of what dataset grabbing (X_train, X_test, y_train, y_test)
                    * corresponding data in a pandas dataframe
    ----- outputs ----

    """

    # results table
    knn_results = []

    # STEP 3: can edit HYPERPARAMETERS: for knn variables
    knn_n_neighbors = [2, 5, 10, 15, 20]
    weights = ['uniform', 'distance']

    # grid searching model results for knn on all types of data with all types of inputs
    for scalar_type in tqdm(data_testing.keys(), desc="knn scaled vs original"):
        if scalar_type == 'columns':
            continue
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]
        feature_names = data_testing["columns"].tolist()

        # going through possible KNN combos
        for knn_n_neighbor in knn_n_neighbors:
            for weight in weights:
                # modeling portion
                model = KNeighborsClassifier(
                    n_neighbors=knn_n_neighbor, weights=weight
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # validation
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)

                # getting permutation importance
                perm = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=3, # reduced number to speed things up significantly
                    RANDOM_STATE=RANDOM_STATE,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # STEP 3, add any hyperparameters to each of these results/outputs: saving results to dict
                knn_results.append(
                    {
                        "file name": FILE_INFO.name,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": TEST_SIZE,
                        "random state": RANDOM_STATE,
                        "scalar_status": scalar_type,
                        "y variable used": Y_COL,
                        "model used": "knn",
                        "logistic_reg_c": np.nan,
                        "lr_ratios": np.nan,
                        "nn_layers": np.nan,
                        "nn_neurons": np.nan,
                        "nn_batch_size": np.nan,
                        "nn_epochs": np.nan,
                        "dt_max_depth": np.nan,
                        "dt_min_samples_split": np.nan,
                        "svm_c_val": np.nan,
                        "svm_kernel": np.nan,
                        "knn_n_neighbor": knn_n_neighbor,
                        "knn_weights": weight,
                        "gbm_learning_rate": np.nan,
                        "gbm_n_estimator": np.nan,
                        "coefficient_importance": np.nan,
                        "permutation_importance": perm_imp_json,
                    }
                )

    return knn_results

def test_neural_net(data_testing):
    """
    ----- inputs -----
    data_testing: dict[str=scalingType][str=y/X train/test label][pd.DataFrame]
        dictionary contianing
            * string of scaling type (standard scalar, orig)
                * string of what dataset grabbing (X_train, X_test, y_train, y_test)
                    * corresponding data in a pandas dataframe
    ----- outputs ----

    """
    # results table
    neur_net_results = []

    #  STEP 3: can edit HYPERPARAMETERS: for neural net
    nn_layers_list = [1, 2, 3, 4]
    nn_neurons_list = [16, 32, 64, 128, 256]
    nn_batch_size_list = [32, 64, 128, 256]
    nn_epochs_list = [5, 10, 20]

    # grid searching model results for neural net on all types of data with all types of inputs
    for scalar_type in tqdm(data_testing.keys(), desc="neural net scaled vs original"):
        if scalar_type == 'columns':
            continue
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]

        # going through possible neural net combos
        for nn_layers in nn_layers_list:
            for nn_neurons in nn_neurons_list:
                for nn_batch_size in nn_batch_size_list:
                    for nn_epochs in nn_epochs_list:
                        # modeling portion
                        model = Sequential()
                        model.add(Input(shape=(X_train.shape[1],)))
                        for _ in range(nn_layers):
                            model.add(Dense(nn_neurons, activation="relu"))
                        model.add(Dense(1, activation="sigmoid"))
                        model.compile(
                            optimizer="adam",
                            loss="binary_crossentropy",
                            metrics=["accuracy"],
                        )
                        model.fit(
                            X_train,
                            y_train,
                            epochs=nn_epochs,
                            batch_size=nn_batch_size,
                            verbose=0,
                        )
                        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

                        # validation
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        conf_matrix = confusion_matrix(y_test, y_pred)

                        # STEP 3, add any hyperparameters to each of these results/outputs: saving results to dict
                        neur_net_results.append(
                            {
                                "file name": FILE_INFO.name,
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "confusion matrix": conf_matrix,
                                "test size": TEST_SIZE,
                                "random state": RANDOM_STATE,
                                "scalar_status": scalar_type,
                                "y variable used": Y_COL,
                                "model used": "neural net",
                                "logistic_reg_c": np.nan,
                                "lr_ratios": np.nan,
                                "nn_layers": nn_layers,
                                "nn_neurons": nn_neurons,
                                "nn_batch_size": nn_batch_size,
                                "nn_epochs": nn_epochs,
                                "dt_max_depth": np.nan,
                                "dt_min_samples_split": np.nan,
                                "svm_c_val": np.nan,
                                "svm_kernel": np.nan,
                                "knn_n_neighbor": np.nan,
                                "knn_weights": np.nan,
                                "gbm_learning_rate": np.nan,
                                "gbm_n_estimator": np.nan,
                                "coefficient_importance": np.nan,
                                "permutation_importance": np.nan,
                            }
                        )

    return neur_net_results

def test_gbm(data_testing):
    """
    ----- inputs -----
    data_testing: dict[str=scalingType][str=y/X train/test label][pd.DataFrame]
        dictionary contianing
            * string of scaling type (standard scalar, orig)
                * string of what dataset grabbing (X_train, X_test, y_train, y_test)
                    * corresponding data in a pandas dataframe
    ----- outputs ----

    """
    # results table
    gbm_results = []

    #  STEP 3: can edit HYPERPARAMETERS: for gbm variables
    gbm_learning_rates = [0.01, 0.05, 0.1, 0.2]
    gbm_n_estimators = [100, 200, 400, 800]

    # grid searching model results for gbm on all types of data with all types of inputs
    for scalar_type in tqdm(data_testing.keys(), desc="gbm scaled vs original"):
        if scalar_type == 'columns':
            continue
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]
        feature_names = data_testing["columns"].tolist()

        # going through possible gbm combos
        for gbm_learning_rate in gbm_learning_rates:
            for gbm_n_estimator in gbm_n_estimators:
                # modeling portion
                model = GradientBoostingClassifier(
                    learning_rate=gbm_learning_rate, n_estimators=gbm_n_estimator
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # validation
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)

                # getting feature importance
                gbm_imp = model.feature_importances_
                feature_imp = dict(zip(feature_names, gbm_imp))
                feature_imp_json = json.dumps({k: float(v) for k, v in feature_imp.items()})

                # getting permutation importance
                perm = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=10,
                    RANDOM_STATE=RANDOM_STATE,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # STEP 3, add any hyperparameters to each of these results/outputs: saving results to dict
                gbm_results.append(
                    {
                        "file name": FILE_INFO.name,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": TEST_SIZE,
                        "random state": RANDOM_STATE,
                        "scalar_status": scalar_type,
                        "y variable used": Y_COL,
                        "model used": "gbm",
                        "logistic_reg_c": np.nan,
                        "lr_ratios": np.nan,
                        "nn_layers": np.nan,
                        "nn_neurons": np.nan,
                        "nn_batch_size": np.nan,
                        "nn_epochs": np.nan,
                        "dt_max_depth": np.nan,
                        "dt_min_samples_split": np.nan,
                        "svm_c_val": np.nan,
                        "svm_kernel": np.nan,
                        "knn_n_neighbor": np.nan,
                        "knn_weights": np.nan,
                        "gbm_learning_rate": gbm_learning_rate,
                        "gbm_n_estimator": gbm_n_estimator,
                        "coefficient_importance": feature_imp_json,
                        "permutation_importance": perm_imp_json,
                    }
                )

    return gbm_results

def test_decision_tree(data_testing):
    """
    ----- inputs -----
    data_testing: dict[str=scalingType][str=y/X train/test label][pd.DataFrame]
        dictionary contianing
            * string of scaling type (standard scalar, orig)
                * string of what dataset grabbing (X_train, X_test, y_train, y_test)
                    * corresponding data in a pandas dataframe
    ----- outputs ----

    """
    # results table
    dec_tree_results = []

    #  STEP 3: can edit HYPERPARAMETERS: for decision tree variables
    dt_max_depths = [50, 100, 200, 400, None]
    dt_min_samples_splits = [2, 10, 20, 50]

    # grid searching model results for decision tree on all types of data with all types of inputs
    for scalar_type in tqdm(data_testing.keys(), desc="decision tree scaled vs original"):
        if scalar_type == 'columns':
            continue
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]
        feature_names = data_testing["columns"].tolist()

        # going through possible decision tree combos
        for dt_min_samples_split in dt_min_samples_splits:
            for dt_max_depth in dt_max_depths:
                # modeling portion
                model = DecisionTreeClassifier(
                    max_depth=dt_max_depth, min_samples_split=dt_min_samples_split
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # validation
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)

                # getting feature importance
                tree_imp = model.feature_importances_
                feature_imp = dict(zip(feature_names, tree_imp))
                feature_imp_json = json.dumps({k: float(v) for k, v in feature_imp.items()})

                # getting permutation importance
                perm = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=10,
                    RANDOM_STATE=RANDOM_STATE,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # STEP 3, add any hyperparameters to each of these results/outputs
                dec_tree_results.append(
                    {
                        "file name": FILE_INFO.name,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": TEST_SIZE,
                        "random state": RANDOM_STATE,
                        "scalar_status": scalar_type,
                        "y variable used": Y_COL,
                        "model used": "decision_tree",
                        "logistic_reg_c": np.nan,
                        "lr_ratios": np.nan,
                        "nn_layers": np.nan,
                        "nn_neurons": np.nan,
                        "nn_batch_size": np.nan,
                        "nn_epochs": np.nan,
                        "dt_max_depth": dt_max_depth,
                        "dt_min_samples_split": dt_min_samples_split,
                        "svm_c_val": np.nan,
                        "svm_kernel": np.nan,
                        "knn_n_neighbor": np.nan,
                        "knn_weights": np.nan,
                        "gbm_learning_rate": np.nan,
                        "gbm_n_estimator": np.nan,
                        "coefficient_importance": feature_imp_json,
                        "permutation_importance": perm_imp_json,
                    }
                )

    return dec_tree_results


# ----------------------------------------------
# --- Paralllel / Original ---------------------
# ----------------------------------------------

def run_models_for_file() -> list:
    """
    Goal: return model results for file

    Paramaters:
        
    Outputs:
        all_rows: list
            list of the dictionary model results
    """

    df = data_prep()
    if df.empty:
        return []

    data_testing = get_train_test(df, Y_COL=Y_COL, scaling_used=True)

    # STEP 5: Choosing which model results to run (comment or uncomment models as needed)
    # STEP 4: Add model name here if needed
    model_fns = [
        test_logistic_reg,
        # test_neural_net,
        # test_knn,
        test_decision_tree,
        # test_svm,
        # test_gbm,
    ]

    # running each model in the model funcs list to return the results
    all_rows = []
    for fn in model_fns:
        rows = fn(data_testing) # running each function
        if rows:
            all_rows.extend(rows)

    return all_rows

# ----------------------------------------------
# --- Main -------------------------------------
# ----------------------------------------------

def main():
    # running each model to get results
    rows_results = run_models_for_file() # calling function to run models
    dataframe_rows_results = pd.DataFrame(rows_results) # converting into a dataframe, so that we can save it
    dataframe_rows_results.to_csv(f'results_for_{FILE_INFO.name}.csv') # saving it into the files section as a CSV
    print("\n\nCOMPLETED: k\n\n")


if __name__ == "__main__":
    main()