# -------------------------------------------------------------------------
# -Set Up  and global variables -------------------------------------------
# -------------------------------------------------------------------------

# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
import re
from pathlib import Path

# test_size = [10, 15, 30]
test_size = 15  # for validation
random_state = 42  # for repeatability
# developing results table to plot
all_results = {}

# -------------------------------------------------------------------------
# - Data Processing -------------------------------------------------------
# -------------------------------------------------------------------------


# Data Preperation
def data_prep(file_path):
    """
    ----- inputs -----
    file_path: str
        file wanting to process
    ----- outputs ----
    df: pandas df
        processed anonymzied data (string columns representing intervals split into min and max, then put as minimum and maximum values for those columns)
    """

    df = pd.read_csv(Path(file_path))

    # data is currently in an interval or suppressed format, so swapping it to readable information
    interval_cols = df.columns[df.columns != "combined_label"]

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
            if isinstance(df[col].iloc[0], (int, float, np.integer, np.floating)):
                continue

            bounds = df[col].apply(interval_to_bounds)
            # print(bounds)

            # adding min and max columns
            df[f"min_{col}"] = bounds.apply(lambda i: i[0])
            df[f"max_{col}"] = bounds.apply(lambda i: i[1])

        return df

    # convreting df
    df = add_min_max_columns(df, df.columns)

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
    df, y_col="combined_label", test_size=test_size, scaling_used=True
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

    # getting x and y
    y = df[y_col]
    y = y.values

    X = df.drop(columns=y_col)
    X = X.values

    # getting train, test splits
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scaling_used:  # if want to run on scaled and original data
        # testing all with and without scaled data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_orig)
        X_test_scaled = scaler.transform(X_test_orig)

        data_testing = {
            "standard_scalar": {
                "X_train": X_train_scaled,
                "X_test": X_test_scaled,
                "y_train": y_train_orig,  # using unscaled y
                "y_test": y_test_orig,  # using unscaled y
            },
            "orig": {
                "X_train": X_train_orig,
                "X_test": X_test_orig,
                "y_train": y_train_orig,
                "y_test": y_test_orig,
            },
        }

        return data_testing

    else:  # if only want to run on original data
        data_testing = {
            "orig": {
                "X_train": X_train_orig,
                "X_test": X_test_orig,
                "y_train": y_train_orig,
                "y_test": y_test_orig,
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
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]

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

                # saving results to dict
                log_reg_results.append(
                    {
                        "file name": file_info["file_path"],
                        "anonymization type": file_info["anonymization"],
                        "k-level": file_info["k"],
                        "l-diversity level": file_info["l"],
                        "t-closeness level": file_info["t"],
                        "suppression level": file_info["suppression_level"],
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info[
                            "orig_file"
                        ],
                        "y variable used": "combined_label",
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
                            loss="combined_label",
                            metrics=["accuracy"],
                        )
                        model.fit(
                            X_train,
                            y_train,
                            epochs=nn_epochs,
                            batch_size=nn_batch_size,
                            verbose=1,
                        )
                        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

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

                        # saving results to dict
                        neur_net_results.append(
                            {
                                "file name": file_info["file_path"],
                                "anonymization type": file_info["anonymization"],
                                "k-level": file_info["k"],
                                "l-diversity level": file_info["l"],
                                "t-closeness level": file_info["t"],
                                "suppression level": file_info["suppression_level"],
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "confusion matrix": conf_matrix,
                                "test size": test_size,
                                "random state": random_state,
                                "scalar_status": scalar_type,
                                "file name of original data (non-anonymized)": file_info[
                                    "orig_file"
                                ],
                                "y variable used": "combined_label",
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
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]

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

                # print(f"Accuracy: {accuracy}")
                # print(f"Precision: {precision}")
                # print(f"Recall: {recall}")
                # print(f"F1 Score: {f1}")
                # print(f"Confusion Matrix:\n{conf_matrix}")

                dec_tree_results.append(
                    {
                        "file name": file_info["file_path"],
                        "anonymization type": file_info["anonymization"],
                        "k-level": file_info["k"],
                        "l-diversity level": file_info["l"],
                        "t-closeness level": file_info["t"],
                        "suppression level": file_info["suppression_level"],
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info[
                            "orig_file"
                        ],
                        "y variable used": "combined_label",
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
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]
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

                # print(f"Accuracy: {accuracy}")
                # print(f"Precision: {precision}")
                # print(f"Recall: {recall}")
                # print(f"F1 Score: {f1}")
                # print(f"Confusion Matrix:\n{conf_matrix}")

                # saving results to dict
                svm_results.append(
                    {
                        "file name": file_info["file_path"],
                        "anonymization type": file_info["anonymization"],
                        "k-level": file_info["k"],
                        "l-diversity level": file_info["l"],
                        "t-closeness level": file_info["t"],
                        "suppression level": file_info["suppression_level"],
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info[
                            "orig_file"
                        ],
                        "y variable used": "combined_label",
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
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]

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

                # print(f"Accuracy: {accuracy}")
                # print(f"Precision: {precision}")
                # print(f"Recall: {recall}")
                # print(f"F1 Score: {f1}")
                # print(f"Confusion Matrix:\n{conf_matrix}")

                # saving results to dict
                knn_results.append(
                    {
                        "file name": file_info["file_path"],
                        "anonymization type": file_info["anonymization"],
                        "k-level": file_info["k"],
                        "l-diversity level": file_info["l"],
                        "t-closeness level": file_info["t"],
                        "suppression level": file_info["suppression_level"],
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info[
                            "orig_file"
                        ],
                        "y variable used": "combined_label",
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
        X_test = data_testing[scalar_type]["X_test"]
        X_train = data_testing[scalar_type]["X_train"]
        y_train = data_testing[scalar_type]["y_train"]
        y_test = data_testing[scalar_type]["y_test"]

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

                # print(f"Accuracy: {accuracy}")
                # print(f"Precision: {precision}")
                # print(f"Recall: {recall}")
                # print(f"F1 Score: {f1}")
                # print(f"Confusion Matrix:\n{conf_matrix}")

                # saving results to dict
                gbm_results.append(
                    {
                        "file name": file_info["file_path"],
                        "anonymization type": file_info["anonymization"],
                        "k-level": file_info["k"],
                        "l-diversity level": file_info["l"],
                        "t-closeness level": file_info["t"],
                        "suppression level": file_info["suppression_level"],
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion matrix": conf_matrix,
                        "test size": test_size,
                        "random state": random_state,
                        "scalar_status": scalar_type,
                        "file name of original data (non-anonymized)": file_info[
                            "orig_file"
                        ],
                        "y variable used": "combined_label",
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
                    }
                )

    # adding all gbm results to the all results
    all_results["gbm"] = gbm_results

    return gbm_results


# -------------------------------------------------------------------------
# - Running Section -------------------------------------------------------
# -------------------------------------------------------------------------


def main():
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
    ]

    # will be results combo (these are the rows)
    extended_model_results = []

    files_to_test_k = [f"data/k/{f.name}" for f in Path('data/k').iterdir() if f.is_file()]

    files_to_test_l = [f"data/l/{f.name}" for f in Path('data/l').iterdir() if f.is_file()]

    files_to_test_t = [f"data/t/{f.name}" for f in Path('data/t').iterdir() if f.is_file()]


    # going through k possibilities
    for file in files_to_test_k:
        # print(f"running: {file}")
        # getting info out of file path naming structure
        file_info = file.split("_")
        # print('file_info: ', file_info)
        anonymization_strat = file_info[0]
        k_level = file_info[1].split("=")[1]
        suppression_level = file_info[-2].split("=")[1]

        file_information = {
            "file_path": file,
            "anonymization": anonymization_strat,
            "k": k_level,
            "l": np.nan,
            "t": np.nan,
            "suppression_level": suppression_level,
            "orig_file": "ListeriaSoil_clean_log.csv",
        }

        df = data_prep(file)
        data_testing = get_train_test(df, y_col='combined_label', test_size=test_size, scaling_used=True)
        log_results = test_logistic_reg(data_testing, file_information)
        # neural_net_results = test_neural_net(data_testing, file_information)
        knn_results = test_knn(data_testing, file_information)
        dec_results = test_decision_tree(data_testing, file_information)
        svm_results = test_svm(data_testing, file_information)
        gbm_results = test_gbm(data_testing, file_information)

        # combining the lists
        extended_model_results.extend(log_results)
        # extended_model_results.extend(neural_net_results)
        extended_model_results.extend(knn_results)
        extended_model_results.extend(dec_results)
        extended_model_results.extend(svm_results)
        extended_model_results.extend(gbm_results)

    # going through k, l possibilities
    for file in files_to_test_l:
        # getting info out of file path naming structure
        file_info = file.split('_')
        anonymization_strat = file_info[0]
        k_level = file_info[1].split('=')[1]
        ldiv_level = file_info[2].split('=')[1]
        suppression_level = file_info[-2].split('=')[1]

        file_information = {
            'file_path': file,
            'anonymization': anonymization_strat,
            'k': k_level,
            'l': ldiv_level,
            't': np.nan,
            'suppression_level': suppression_level,
            'orig_file': "ListeriaSoil_clean_log.csv",
        }
        # print(file_information)

        df = data_prep(file)
        data_testing = get_train_test(df, y_col='combined_label', test_size=test_size, scaling_used=True)
        log_results = test_logistic_reg(data_testing, file_information)
        # neural_net_results = test_neural_net(data_testing, file_information)
        knn_results = test_knn(data_testing, file_information)
        dec_results = test_decision_tree(data_testing, file_information)
        svm_results = test_svm(data_testing, file_information)
        gbm_results = test_gbm(data_testing, file_information)

        # combining the lists
        extended_model_results.extend(log_results)
        # extended_model_results.extend(neural_net_results)
        extended_model_results.extend(knn_results)
        extended_model_results.extend(dec_results)
        extended_model_results.extend(svm_results)
        extended_model_results.extend(gbm_results)

    # going through k, t possibilities
    for file in files_to_test_t:
        # getting info out of file path naming structure
        file_info = file.split("_")
        # print('file_info: ', file_info)
        anonymization_strat = file_info[0]
        k_level = file_info[1].split("=")[1]
        tclose_level = file_info[2].split('=')[1]
        suppression_level = file_info[-2].split("=")[1]

        file_information = {
            'file_path': file,
            'anonymization': anonymization_strat,
            'k': k_level,
            'l': np.nan,
            't': tclose_level,
            'suppression_level': suppression_level,
            'orig_file': "ListeriaSoil_clean_log.csv",
        }

        df = data_prep(file)
        data_testing = get_train_test(df, y_col='combined_label', test_size=test_size, scaling_used=True)
        log_results = test_logistic_reg(data_testing, file_information)
        # neural_net_results = test_neural_net(data_testing, file_information)
        knn_results = test_knn(data_testing, file_information)
        dec_results = test_decision_tree(data_testing, file_information)
        svm_results = test_svm(data_testing, file_information)
        gbm_results = test_gbm(data_testing, file_information)

        # combining the lists
        extended_model_results.extend(log_results)
        # extended_model_results.extend(neural_net_results)
        extended_model_results.extend(knn_results)
        extended_model_results.extend(dec_results)
        extended_model_results.extend(svm_results)
        extended_model_results.extend(gbm_results)

    # appending all mosdel results into a large dataframe
    results_combo = pd.DataFrame(extended_model_results, columns=results_columns)
    print(results_combo.head())
    results_combo.to_csv("results.csv")


if __name__ == "__main__":
    main()