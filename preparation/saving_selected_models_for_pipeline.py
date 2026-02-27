# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    PrecisionRecallDisplay
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K
import re
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import pickle
import joblib


# ----------------------------------
# --- Global Variables -------------
# ----------------------------------

soil_vars_only = ['pH', 'Copper (mg/Kg)', 'Molybdenum (mg/Kg)', 'log of Sulfur (mg/Kg)', 'log of Moisture',
       'log of Manganese (mg/Kg)', 'log of Aluminum (mg/Kg)', 'log of Potassium (mg/Kg)', 'log of Total carbon (%)'
       'log of Total nitrogen (%)', 'double log of Zinc (mg/Kg)', 'log of Organic matter (%)', 'log of Phosphorus (mg/Kg)',
       'log of Iron (mg/Kg)', 'log of Magnesium (mg/Kg)', 'log of Sodium (mg/Kg)', 'log of Calcium (mg/Kg)', 'scaled_cluster_kmeans', 'cluster_kmeans']
long_lat_vars_only = ['Latitude', 'Longitude',
       'Precipitation (mm)', 'Max temperature (℃ )', 'Min temperature (℃ )',
       'Wind speed (m/s)', 'Barren (%)', 'Forest (%)', 'Pasture (%)',
       'log of Grassland (%)', 'log of Shrubland (%)', 'log of Open water (%)',
       'log of Developed open space (> 20% Impervious Cover) (%)', 'log of Elevation (m)',
        'log of Cropland (%)', 'log of Wetland (%)', "log of Developed open space (< 20% Impervious Cover) (%)",
       'scaled_cluster_kmeans', 'cluster_kmeans']


# saving global variables
TEST_SIZE = 0.22 # default for validity
RANDOM_STATE = 42  # for repeatability
# developing results table to plot

# grabbing parent directories so can process input and output files correctly
ROOT = Path.cwd()
if ROOT.name == "preparation":
    ROOT = ROOT.parent
DATA_PATH = ROOT / "data"
OUTPUT_PATH = ROOT / "website" / "backend" / "models"

# getting in file path
try:
    file_info = Path(DATA_PATH / "ListeriaSoil_clean_log.csv")
except:
    try:
        file_info = Path("ListeriaSoil_clean_log.csv")
    except Exception as e:
        raise e

Y_COL = "binary_listeria_presense"

# If want all strings/catagorical data to be encoded in 1-hot vectors
# (aka want to transform arbitrary strings into integer values)
ENCODE_STR = False

# ----------------------------------
# --- Data Preparation -------------
# ----------------------------------

def data_prep(file_info):
    """
    ----- inputs -----
    file_info: Path object
        file wanting to process
    ----- outputs ----
    df: pandas df
        processed anonymzied data (string columns representing intervals split into min and max, then put as minimum and maximum values for those columns)
    """

    df = pd.read_csv(file_info)

    # Drop 'index' column if it exists, as it's typically an artifact and not a feature
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    # if salcamp column, then adding it to the dataset
    if Y_COL == "Salmon_or_camp_test":
        df["Salmon_or_camp_test"] = (df["CampylobacterAnalysis30ml"] == "Positive") | (df["SalmonellaSPAnalysis"] == "Positive")
    if Y_COL == "binary_listeria_presense":
        original_listeria_col = 'Number of Listeria isolates obtained'
        df['binary_listeria_presense'] = [row_val if row_val == 0 else 1 for row_val in df[original_listeria_col]]
        # Drop the original column to prevent data leakage
        if original_listeria_col in df.columns:
            df = df.drop(columns=[original_listeria_col])

    # switching missing values and weird failures in writing to np.inf bc pandas didnt handle properly
    df = df.replace("#NAME?", -np.inf)
    df = df.fillna(-np.inf)

    # replacing inf with max number that is not max number + 100 in dict (FOR NOT JUST 99999999)
    df = df.replace(np.inf, 99999)
    # replacing -inf with min number (not -inf) - 100 in dict (FOR NOT JUST -99999999)
    df = df.replace(-np.inf, -99999)

    # Drop unwanted cluster columns if they exist
    cols_to_drop = ['scaled_cluster_kmeans', 'cluster_kmeans']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = df.dropna(axis=1, how="all")

    if ENCODE_STR:
        df = pd.get_dummies(df)
    return df

# ----------------------------------
# --- Train-Test Split -------------
# ----------------------------------

def get_train_test(
    df, scaling_used=True, test_size=TEST_SIZE, file_path=""
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

    # indexes for test set
    X = df.drop(columns=Y_COL)
    y = df[Y_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    # data columns
    data_columns = X.columns

    if scaling_used:  # if want to run on scaled and original data
        # testing all with and without scaled data
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = scaler.transform(X_test)

        # saving the scaler used on training to be able to replicate in the pipeline
        scaler_file_path = OUTPUT_PATH / f"scaler_file_{file_path}.joblib"
        joblib.dump(scaler, scaler_file_path)
        

        # getting all possible standard and original data splits accordingly
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

# ----------------------------------
# --- Saving models ----------------
# ----------------------------------

def save_model(data_testing, scalar_type, file_path):
    """
    Trains a Gradient Boosting Machine (GBM) model and saves it to a file.

    Parameters:
        data_testing: dict
            Dictionary containing processed data for training and testing.
        scalar_type: str
            The type of scaling used ('standard_scalar' or 'orig').
        gbm_learning_rate: float
            The learning rate for the GBM model.
        gbm_n_estimator: int
            The number of boosting stages to perform for the GBM model.
        file_path: str
            The path where the trained model will be saved.
    """

    X_train = data_testing[scalar_type]["X_train"]
    if "log of index" in X_train.columns:
        print('dropping log of index')
        X_train = X_train.drop(columns=["log of index"])
    if "index" in X_train.columns:
        print('dropping index')
        X_train = X_train.drop(columns=["index"])
    if "Unnamed: 0" in X_train.columns:
        print('dropping unnamed 0')
        X_train = X_train.drop(columns=["Unnamed: 0"])

    print("after dropping, ", X_train.columns)
    
    y_train = data_testing[scalar_type]["y_train"]

    # ---------------------------------------------
    #  saving gbm model
    # hyperparams
    gbm_learning_rate = 0.1
    gbm_n_estimator = 100
    # model set up
    model = GradientBoostingClassifier(
        learning_rate=gbm_learning_rate, n_estimators=gbm_n_estimator, random_state=RANDOM_STATE
    )

    # Initialize and train the model with the specified hyperparameters
    model.fit(X_train, y_train)
    print(X_train.columns)

    # file path
    full_file_path = OUTPUT_PATH / f"gbm_{file_path}.pkl"

    with open(full_file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"gbm model saved to {full_file_path}")

    # ---------------------------------------------
    # Neural Net model set up
    # hyperparams
    nn_layers = 1
    nn_epochs = 10
    nn_neurons = 32
    nn_batch_size = 10

    # modeling portion
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    
    for _ in range(nn_layers):
        model.add(Dense(nn_neurons, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    model.fit(
        X_train,
        y_train,
        epochs=nn_epochs,
        batch_size=nn_batch_size,
        verbose=0,
    )

    # file path
    full_file_path = OUTPUT_PATH / f"neural_net_{file_path}.pkl"

    with open(full_file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"nueral net model saved to {full_file_path}")


    # ---------------------------------------------
    # SVC setup
    # hyperparams
    c_val = 1
    svm_kernel = "rbf"
    
    # SVC needs probability=True to use predict_proba, which can be slower
    model = SVC(C=c_val, kernel=svm_kernel, max_iter=20000, probability=True)
    model.fit(X_train, y_train)
    
    # file path
    full_file_path = OUTPUT_PATH / f"svm_{file_path}.pkl"
    
    # Save the trained model using pickle
    with open(full_file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"svm model saved to {full_file_path}")

    print("reiterated training cols at end: ", X_train.columns)


# all data
def save_soil_longlat_models():
    # setup
    print("cwd:", Path.cwd())
    print("file_info:", file_info)
    print("exists:", Path(file_info).exists())
    df = data_prep(file_info)

    # hyperparams
    scalar_type = "standard_scalar"

    # training and testing
    data_testing = get_train_test(df, scaling_used=True, test_size=TEST_SIZE, file_path='main')

    save_model(data_testing, scalar_type, "main")
    

# only latitude longitude accessible data
def save_longlat_models():
    # (elevation, long/lat, and weather), aka no soil
    # setup
    print("cwd:", Path.cwd())
    print("file_info:", file_info)
    print("exists:", Path(file_info).exists())
    df = data_prep(file_info)

    # hyperparams
    scalar_type = "standard_scalar"
    
    # Filter soil_vars_only to only include columns that exist in df
    # NOTE: Added this filter to prevent KeyError if columns were already dropped by data_prep.
    columns_to_drop = [col for col in soil_vars_only if col in df.columns]
    df = df.drop(columns=columns_to_drop) # dropping soil variables to create a long lat only
    print("\n\ncol length of new longlat mode: ", len(df.columns), "   ", df.columns)
    # also drops clusters because those were developed with full variables.
    data_testing = get_train_test(df, scaling_used=True, test_size=TEST_SIZE, file_path='longlat_only')

    save_model(data_testing, scalar_type, "longlat_only")

# only soil data
def save_soil_models():
    # (moisture, soil) aka no weather or long/lat
    # setup
    print("cwd:", Path.cwd())
    print("file_info:", file_info)
    print("exists:", Path(file_info).exists())
    df = data_prep(file_info)

    # hyperparams
    scalar_type = "standard_scalar"

    # Filter long_lat_vars_only to only include columns that exist in df
    # NOTE: Added this filter to prevent KeyError if columns were already dropped by data_prep.
    columns_to_drop = [col for col in long_lat_vars_only if col in df.columns]
    df = df.drop(columns=columns_to_drop) # dropping long lat only
    print("\n\ncol length of new soil mode: ", len(df.columns), "   ", df.columns)
    # also drops clusters because those were developed with full variables.
    data_testing = get_train_test(df, scaling_used=True, test_size=TEST_SIZE, file_path='soil_only')

    # file path
    save_model(data_testing, scalar_type, "soil_only")

# -----------------------------
# --- Main --------------------
# -----------------------------

def main():
    # saving models trained on only latitude-longitude available data (weather, elevation)
    save_longlat_models()

    # saving models trained on ALL (soil and latitude-longitude available data (weather, elevation))
    save_soil_longlat_models()

    # saving models trained on only soil available data
    save_soil_models()

if __name__ == "__main__":
    main()