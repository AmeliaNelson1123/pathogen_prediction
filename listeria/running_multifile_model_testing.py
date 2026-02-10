# -------------------------------------------------------------------------
# -Set Up  and global variables -------------------------------------------
# -------------------------------------------------------------------------

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
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
from sklearn.inspection import permutation_importance

# -------------------------------------------------------------------------
# - Configuration / Metadata ----------------------------------------------
# -------------------------------------------------------------------------

# test_size = [10, 15, 30]
test_size = .17  # for validation
random_state = 42  # for repeatability
# developing results table to plot
all_results = {}
y_col = "binary_listeria_presense"
# columns
results_columns = [
    "file name",
    "anonymization type",
    "k-level",
    "l-diversity level",
    "t-closeness level",
    "suppression level",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion matrix",
    "test size",
    "random state",
    "scalar_status",
    "file name of original data (non-anonymized)",
    "y variable used",
    "model used",
    "logistic_reg_c",
    "lr_ratios",
    "nn_layers",
    "nn_neurons",
    "nn_batch_size",
    "nn_epochs",
    "dt_max_depth",
    "dt_min_samples_split",
    "svm_c_val",
    "svm_kernel",
    "knn_n_neighbor",
    "knn_weights",
    "gbm_learning_rate",
    "gbm_n_estimator",
    "coefficient_importance",
    "permutation_importance", 
]

@dataclass(frozen=True)
class FileInfo:
    file_path: str
    orig_file: str
    anonymization: str # "None" | "k" | "l" | "t"
    k: str | float
    l: str | float
    t: str | float
    suppression_level: str | float

    @property
    def is_original_file(self) -> bool:
        return self.file_path == self.orig_file

# -------------------------------------------------------------------------
# - Data Processing -------------------------------------------------------
# -------------------------------------------------------------------------

# Data Preperation
def data_prep(file_info: FileInfo):
    """
    ----- inputs -----
    file_path: FileInfo Object
        file wanting to process, and its relevant information
    ----- outputs ----
    df: pandas df
        processed anonymzied data (string columns representing intervals split into min and max, then put as minimum and maximum values for those columns)
    """
    # breakpoint() # TEMP debugger

    df = None
    if file_info.is_original_file:
        df = pd.read_csv(Path(file_info.file_path))

        # switching missing values and weird failures in writing to np.inf bc pandas didnt handle properly
        df = df.replace("#NAME?", -np.inf)
        df = df.fillna(-np.inf)

        # replacing inf with max number that is not max number + 100 in dict (FOR NOT JUST 99999999)
        df = df.replace(np.inf, 99999)
        # replacing -inf with min number (not -inf) - 100 in dict (FOR NOT JUST -99999999)
        df = df.replace(-np.inf, -99999)

        df = df.dropna(axis=1, how="all")
        return df
    else:
        df = pd.read_parquet(Path(file_info.file_path))

    # breakpoint() # TEMP debugger

    # data is currently in an interval or suppressed format, so swapping it to readable information
    interval_cols = df.columns[(df.columns != "binary_listeria_presense") & (df.columns != "index")]

    # creating min and max columns for intervals

    # quick line of regex
    _interval_re = re.compile(r"^\s*\[\s*([^,]+)\s*,\s*([^\)\]]+)\s*[\)\]]\s*$")

    def interval_to_bounds(x):
        s = str(x).strip()

        # quick check for missing values
        if s in {"*", "MISSING", "[MISSING, MISSING)", "[MISSING, MISSING]"}:
            return (np.nan, np.nan)

        # replacing infs
        s = s.replace("+inf", "inf")

        # performing regex to get the 2 parts of the string interval
        m = _interval_re.match(s)

        if not m:
            print("problem with m: ", m)
            return (np.nan, np.nan)

        # getting raw intervals from parse
        raw_0, raw_1 = m.group(1).strip(), m.group(2).strip()

        def parse_num(t):
            t = t.lower().strip()
            if t in {"-inf", "-infinity"}:
                return -np.inf
            if t in {"inf", "infinity", "+inf", "+infinity"}:
                return np.inf
            return float(t)

        return (parse_num(raw_0), parse_num(raw_1))

    def add_min_max_columns(df, cols):
        for col in cols:
            # checking for int only cols
            if df.empty:
                raise ValueError("DataFrame is empty after reading CSV (0 rows). Cannot create min/max columns.")
    
            if isinstance(df[col].iloc[0], (int, float, np.integer, np.floating)):
                continue

            bounds = df[col].apply(interval_to_bounds)
            # print(bounds)

            # adding min and max columns
            df[f"min_{col}"] = bounds.apply(lambda i: i[0])
            df[f"max_{col}"] = bounds.apply(lambda i: i[1])

        return df

    # convreting df
    if not df.empty:
        df = add_min_max_columns(df, df.columns)
    else:
        return df

    # dropping original cols
    df = df.drop(columns=interval_cols)
    # dropping nans
    df = df.dropna(axis=1, how="all")

    # replacing inf with max number that is not max number + 100 in dict (FOR NOT JUST 99999999)
    df = df.replace(np.inf, 99999)
    # replacing -inf with min number (not -inf) - 100 in dict (FOR NOT JUST -99999999)
    df = df.replace(-np.inf, -99999)

    # print(df.head())

    return df


# -------------------------------------------------------------------------
# - Splitting into Train and Test sets ------------------------------------
# -------------------------------------------------------------------------

def get_train_test(
    df, idx_path, y_col="binary_listeria_presense", scaling_used=True
):
    """
    ----- inputs -----
    df: pandas dict
        processed data (all numerics)
    y_col: str
        string of y labels
    test_size: int
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
    # breakpoint() # TEMP debugger
    
    # indexes for test set 
    test_ind = pd.read_csv(Path(idx_path))

    # seperating testing and training
    test_set = df[df['index'].isin(test_ind['index'])]
    train_set = df[~ df['index'].isin(test_ind["index"])]

    data_columns = df.columns

    # getting x and y
    y_test = test_set[y_col]
    y_test = y_test.values

    X_test = test_set.drop(columns=[y_col, 'index'])
    X_test = X_test.values

    y_train = train_set[y_col]
    y_train = y_train.values

    X_train = train_set.drop(columns=[y_col, 'index'])
    X_train = X_train.values

    if scaling_used:  # if want to run on scaled and original data
        # testing all with and without scaled data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

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


# -------------------------------------------------------------------------
# - Modeling Section ------------------------------------------------------
# -------------------------------------------------------------------------

def test_logistic_reg(data_testing, file_info):
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

    # for logistic regression
    # c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 2, 4, 8, 32, 56, 100]
    c_vals = [0.01, 0.1, 1, 4, 8]
    # lr_ratios = [0, 0.5, 1] # 0 = l2 penalty, 1 = l1 penalty, 0.5 = elasticnet penalty (both L1 and L2)
    lr_ratios = [
        0,
        1,
    ]  # 0 = l2 penalty, 1 = l1 penalty, 0.5 = elasticnet penalty (both L1 and L2)

    # grid searching model results for log reg on all types of data with all types of inputs
    for scalar_type in data_testing.keys():
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
                    random_state=random_state,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # saving results to dictfile_path: str
                log_reg_results.append(
                    {
                        "file name": file_info.file_path,
                        "anonymization type": file_info.anonymization,
                        "k-level": file_info.k,
                        "l-diversity level": file_info.l,
                        "t-closeness level": file_info.t,
                        "suppression level": file_info.suppression_level,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info.orig_file,
                        "y variable used": "binary_listeria_presense",
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

def test_neural_net(data_testing, file_info):
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

    # for neural net
    nn_layers_list = [1, 2, 3, 4]
    nn_neurons_list = [16, 32, 64, 128, 256]
    nn_batch_size_list = [32, 64, 128, 256]
    nn_epochs_list = [5, 10, 20]

    # grid searching model results for neural net on all types of data with all types of inputs
    for scalar_type in data_testing.keys():
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

                        # saving results to dict
                        neur_net_results.append(
                            {
                                "file name": file_info.file_path,
                                "anonymization type": file_info.anonymization,
                                "k-level": file_info.k,
                                "l-diversity level": file_info.l,
                                "t-closeness level": file_info.t,
                                "suppression level": file_info.suppression_level,
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "confusion matrix": conf_matrix,
                                "test size": test_size,
                                "random state": random_state,
                                "scalar_status": scalar_type,
                                "file name of original data (non-anonymized)": file_info.orig_file,
                                "y variable used": "binary_listeria_presense",
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

def test_decision_tree(data_testing, file_info):
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

    # for decision tree variables
    dt_max_depths = [50, 100, 200, 400, None]
    dt_min_samples_splits = [2, 10, 20, 50]

    # grid searching model results for decision tree on all types of data with all types of inputs
    for scalar_type in data_testing.keys():
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
                    random_state=random_state,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                dec_tree_results.append(
                    {
                        "file name": file_info.file_path,
                        "anonymization type": file_info.anonymization,
                        "k-level": file_info.k,
                        "l-diversity level": file_info.l,
                        "t-closeness level": file_info.t,
                        "suppression level": file_info.suppression_level,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info.orig_file,
                        "y variable used": "binary_listeria_presense",
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

def test_svm(data_testing, file_info):
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

    # for svm variables
    c_vals = [0.01, 0.1, 1, 4, 8]
    svm_kernels = ['linear', 'rbf', 'sigmoid']

    # grid searching model results for svm on all types of data with all types of inputs
    for scalar_type in data_testing.keys():
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
                    random_state=random_state,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # saving results to dict
                svm_results.append(
                    {
                        "file name": file_info.file_path,
                        "anonymization type": file_info.anonymization,
                        "k-level": file_info.k,
                        "l-diversity level": file_info.l,
                        "t-closeness level": file_info.t,
                        "suppression level": file_info.suppression_level,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info.orig_file,
                        "y variable used": "binary_listeria_presense",
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

def test_knn(data_testing, file_info):
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

    # for KNN variables
    knn_n_neighbors = [2, 5, 10, 15, 20]
    weights = ['uniform', 'distance']

    # grid searching model results for knn on all types of data with all types of inputs
    for scalar_type in data_testing.keys():
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
                    random_state=random_state,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # saving results to dict
                knn_results.append(
                    {
                        "file name": file_info.file_path,
                        "anonymization type": file_info.anonymization,
                        "k-level": file_info.k,
                        "l-diversity level": file_info.l,
                        "t-closeness level": file_info.t,
                        "suppression level": file_info.suppression_level,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info.orig_file,
                        "y variable used": "binary_listeria_presense",
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

def test_gbm(data_testing, file_info):
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

    # for gbm variables
    gbm_learning_rates = [0.01, 0.05, 0.1, 0.2]
    gbm_n_estimators = [100, 200, 400, 800]

    # grid searching model results for gbm on all types of data with all types of inputs
    for scalar_type in data_testing.keys():
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
                    random_state=random_state,
                    scoring="f1"  # or "accuracy"
                )

                perm_imp = dict(zip(feature_names, perm.importances_mean))
                perm_imp_json = json.dumps({k: float(v) for k, v in perm_imp.items()})

                # saving results to dict
                gbm_results.append(
                    {
                        "file name": file_info.file_path,
                        "anonymization type": file_info.anonymization,
                        "k-level": file_info.k,
                        "l-diversity level": file_info.l,
                        "t-closeness level": file_info.t,
                        "suppression level": file_info.suppression_level,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info.orig_file,
                        "y variable used": "binary_listeria_presense",
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

    # adding all gbm results to the all results
    all_results["gbm"] = gbm_results

    return gbm_results


# -------------------------------------------------------------------------
# - Parse, Call Models, and Parallelize -----------------------------------
# -------------------------------------------------------------------------

def parse_file_info(file_path: str, orig_file: str, idx_file: None | str = None) -> FileInfo:
    """
    Parse metadata from anonymized parquet filename convention.
    Adjust this in ONE place if naming changes.

    Parameters:
        file_path: str
            string to parse information out of (uses pre-determined naming conventions to parse)
        orig_file: str
            string for if it is an original string or not
        idx_file: None | str
            IF ORIGNAL FILE, str string to parse information out of (uses pre-determined naming conventions to parse)
            IF not original file = None
    """
    if file_path == orig_file:
        # processing index file to parse
        p = Path(idx_file)
        
        folder = p.parts[1]  # i.e., "t_parquet" (since parts[0] is "data")
        anonym = folder.split("_")[0]  # i.e. "t"
        
        # Tokenize the filename by "_" and get suppression level
        supp_level = p.name.split("_")[-1]
        return FileInfo(
            file_path=file_path,
            orig_file=orig_file,
            anonymization=anonym,
            k=np.nan,
            l=np.nan,
            t=np.nan,
            suppression_level=supp_level,
        )

    # Example path: data/t_parquet/t_k=2_t=0.3_suplevel=10_anon.parquet
    p = Path(file_path)
    folder = p.parts[1]  # i.e., "t_parquet" (since parts[0] is "data")
    anonym = folder.split("_")[0]  # i.e. "t"

    # Tokenize the filename by "_" and pick out key=value chunks
    tokens = p.name.split("_")
    kv = {}
    for tok in tokens:
        if "=" in tok:
            k_, v_ = tok.split("=", 1)
            kv[k_] = v_.replace(".parquet", "")

    return FileInfo(
        file_path=file_path,
        orig_file=orig_file,
        anonymization=anonym,
        k=kv.get("k"),
        l=kv.get("l"),
        t=kv.get("t"),
        suppression_level=kv.get("suplevel"),
    )

def index_path_for(info: FileInfo) -> str:
    """
    Centralizes where index files live.
    If structure changes, edit here only.

    Parameters:
        info: FileInfo object
            takes in file information to decide on which file to use
    """
    if info.is_original_file:
        # baseline index -> difficult to know/find because it depends on the suppression level, and for normal files, that is fine, but for original files, how would we know which index file to choose bc there is no suppression... so it depends on the surrounding files's suppression methods?
        # or do we make 3 index paths (one for 10, one for 25, and one for 50)???
        # For now: require explicit index for original, too, to be consistent.
        raise ValueError("Original file needs an idx_path strategy (choose a baseline index file).")

    if info.suppression_level is None:
        raise ValueError(f"Suppression level missing for file: {info.file_path}")

    idx_path = f"data/{info.anonymization}_parquet/indexes_to_use_{info.suppression_level}.csv"
    if not Path(idx_path).exists():
        raise FileNotFoundError(f"Index file missing: {idx_path}")
    return idx_path

def run_models_for_file(file_info: FileInfo, idx_path: str) -> list:
    """
    Goal: return model results for file
    
    Paramaters:
        file: str
            the name of the file want to model
        idx_path: str
            the name of the file to pull the test/train indexes from
        file_info: FileInfo object
            parsed information about the file

    Outputs:
        all_rows: list
            list of the dictionary model results
    """

    # breakpoint() # TEMP debugger 

    df = data_prep(file_info)
    if df.empty:
        return []

    # breakpoint() # TEMP debugger 

    data_testing = get_train_test(df, idx_path, y_col=y_col, scaling_used=True)
    
    # breakpoint() # TEMP debugger 

    model_fns = [
        test_logistic_reg,
        # test_neural_net,
        test_knn,
        test_decision_tree,
        test_svm,
        test_gbm,
    ]

    # running each model in the model funcs list to return the results
    all_rows = []
    for fn in model_fns:
        rows = fn(data_testing, file_info)
        # breakpoint() # TEMP for debugging
        if rows:
            all_rows.extend(rows)
        
    return all_rows

def run_one_file(file_path: str, orig_file: str) -> list[dict]:
    # getting information from file
    file_info = parse_file_info(file_path, orig_file)

    # Choose index file
    idx_path = index_path_for(file_info)

    # breakpoint()
    # running the models for each file
    to_return = run_models_for_file(file_info, idx_path)
    # breakpoint()

    # user feedback
    print("    completed file")
    return to_return

def run_one_original_file(file_path: str, idx_path: str) -> list[dict]:
    # getting information from file
    file_info = parse_file_info(file_path, file_path, idx_path)
    # breakpoint() # TEMP debugger 
    
    # running the models for each file
    to_return = run_models_for_file(file_info, idx_path)
    # breakpoint() # TEMP debugger

    # user feedback
    print("    completed original file")
    return to_return

def run_files_parallel(files: list[str], orig_file: str, max_workers: int = 2) -> list[dict]:
    results: list[dict] = []

    # NOTE IMPORTANT: keep max_workers low due to RAM
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        jobs_for_file = {ex.submit(run_one_file, f, orig_file): f for f in files}

        # adding the completed file results to a results list, then will return that result
        for job_x in as_completed(jobs_for_file):
            file = jobs_for_file[job_x]
            try:
                rows = job_x.result()
                if rows:
                    results.extend(rows)
                print(f"[OK] {file} -> {len(rows)} rows")
            except Exception as e:
                print(f"[FAIL] {file}: {e}")

    return results

def run_original_parallel(original_file_to_test: str, max_workers: int = 2) -> list[dict]:
    """
    Run all possible index combinations for the original file. (Goal: creates comparable results for each suppression level)

    Parameters:
        original_file_to_test: str
            the name of the original file to test
    
    Outputs:
        results: list[dict]
            rows to add to files for each type of privacy and threshold
            
    """
    thresholds = [10, 25, 50]
    privacy_types = ["k", "l", "t"]

    # grabbing an original file for 
    list_idx_files = [f"data/{p}_parquet/indexes_to_use_{thr}.csv" for thr in thresholds for p in privacy_types]
    print(len(list_idx_files))

    results = []
    # NOTE IMPORTANT: keep max_workers low due to RAM
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        jobs_for_file = {ex.submit(run_one_original_file, original_file_to_test, idx_file): idx_file for idx_file in list_idx_files}

        # adding the completed file results to a results list, then will return that result
        for job_x in as_completed(jobs_for_file):
            file = jobs_for_file[job_x]
            try:
                rows = job_x.result()
                if rows:
                    results.extend(rows)
                print(f"[OK] {file} -> {len(rows)} rows")
            except Exception as e:
                print(f"[FAIL] {file}: {e}")

    return results

# -------------------------------------------------------------------------
# - Main and Running ------------------------------------------------------
# -------------------------------------------------------------------------


def main():
    original_file_to_test = "data/ListeriaSoil_clean_log.csv"
    
    files_to_test_k = [str(p) for p in Path("data/k_parquet").rglob("*.parquet")]
    # files_to_test_k = []

    files_to_test_l = [str(p) for p in Path("data/l_parquet").rglob("*.parquet")]
    # files_to_test_l = []

    # files_to_test_t = [f"data/t_parquet/{f.name}" for f in Path('data/t_parquet').iterdir() if (f.is_file() and f.suffix == '.parquet')]
    files_to_test_t = [str(p) for p in Path("data/t_parquet").rglob("*.parquet")]
    # files_to_test_t = []

    # creating a baseline for the original model results (with a quick sanity check)
    # orig_file_results = run_original_parallel(original_file_to_test, max_workers=2)
    
    # if len(orig_file_results) == 0:
    #     raise Exception(f"ERROR: FAILED TO GET RESULTS FOR ORIGINAL FILE OF {original_file_to_test}")

    # saving results as a checkpoint
    # pd.DataFrame(orig_file_results).to_csv("results_orig.csv", index=False)
    
    # Start with 1 worker because already at ~80% RAM on a single run
    rows_k = run_files_parallel(files_to_test_k, original_file_to_test, max_workers=1)
    if len(rows_k) != 0:
        # rows_k.extend(orig_file_results) # extending with original file results to integrate original data results
        pd.DataFrame(rows_k).to_csv("results_k_with_features.csv", index=False)
    print("\n\nCOMPLETED: k\n\n")

    # Start with 1 worker because already at ~80% RAM on a single run
    rows_l = run_files_parallel(files_to_test_l, original_file_to_test, max_workers=1)
    if len(rows_l) != 0:
        # rows_l.extend(orig_file_results) # extending with original file results to integrate original data results
        pd.DataFrame(rows_l).to_csv("results_l_with_features.csv", index=False)
    print("\n\nCOMPLETED: l\n\n")

    # Start with 1 worker because already at ~80% RAM on a single run
    rows_t = run_files_parallel(files_to_test_t, original_file_to_test, max_workers=1)
    if len(rows_t) != 0:
        # rows_t.extend(orig_file_results) # extending with original file results to integrate original data results
        pd.DataFrame(rows_t).to_csv("results_t_with_features.csv", index=False)
    print("\n\nCOMPLETED: t\n\n")

if __name__ == "__main__":
    main()


