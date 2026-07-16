"""Shared, leakage-free modeling utilities for the Listeria soil project.

Single source of truth imported by the training notebook, the analysis
notebook, the deploy script, and (indirectly) the backend.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.22
N_CLUSTERS = 3
Y_COL = "binary_listeria_presense"
DATA_FILENAME = "ListeriaSoil_clean_log.csv"
RAW_COUNT_COL = "Number of Listeria isolates obtained"

# Columns that must never be used as features:
# - index artifacts (leak row order), and
# - precomputed KMeans labels (fit on the FULL dataset => leakage; we recompute
#   them fold-safely inside the pipeline instead).
LEAK_COLS = [
    "index",
    "log of index",
    "Unnamed: 0",
    "cluster_kmeans",
    "scaled_cluster_kmeans",
]

SOIL_VARS_ONLY = [
    "pH", "Copper (mg/Kg)", "Molybdenum (mg/Kg)", "log of Sulfur (mg/Kg)",
    "log of Moisture", "log of Manganese (mg/Kg)", "log of Aluminum (mg/Kg)",
    "log of Potassium (mg/Kg)", "log of Total carbon (%)", "log of Total nitrogen (%)",
    "double log of Zinc (mg/Kg)", "log of Organic matter (%)", "log of Phosphorus (mg/Kg)",
    "log of Iron (mg/Kg)", "log of Magnesium (mg/Kg)", "log of Sodium (mg/Kg)",
    "log of Calcium (mg/Kg)",
]
LONGLAT_VARS_ONLY = [
    "Latitude", "Longitude", "Precipitation (mm)", "Max temperature (℃ )",
    "Min temperature (℃ )", "Wind speed (m/s)", "Barren (%)", "Forest (%)",
    "Pasture (%)", "log of Grassland (%)", "log of Shrubland (%)", "log of Open water (%)",
    "log of Developed open space (> 20% Impervious Cover) (%)", "log of Elevation (m)",
    "log of Cropland (%)", "log of Wetland (%)",
    "log of Developed open space (< 20% Impervious Cover) (%)",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_path() -> Path:
    return project_root() / "data" / DATA_FILENAME


def set_seeds(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def load_and_prep(path: Path | None = None) -> pd.DataFrame:
    """Load the CSV and return numeric features + binary target.

    - Builds the binary target from the raw isolate count, then drops the count
      column (prevents target leakage).
    - Drops index artifacts and precomputed cluster columns (see LEAK_COLS).
    - Coerces feature columns to numeric; junk like "#NAME?" becomes NaN and is
      left for the in-pipeline median imputer. NO +/-99999 sentinel fill.
    """
    path = path or data_path()
    df = pd.read_csv(path)

    # Build binary target from the raw count column.
    if RAW_COUNT_COL in df.columns:
        df[Y_COL] = (df[RAW_COUNT_COL] != 0).astype(int)
        df = df.drop(columns=[RAW_COUNT_COL])
    if Y_COL not in df.columns:
        raise ValueError(f"Neither {RAW_COUNT_COL!r} nor {Y_COL!r} present in {path}")
    df[Y_COL] = df[Y_COL].astype(int)

    # Drop leakage / artifact columns if present.
    df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])

    # Coerce features to numeric (turn "#NAME?" etc. into NaN); keep target intact.
    feature_cols = [c for c in df.columns if c != Y_COL]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Drop columns that are entirely missing.
    df = df.dropna(axis=1, how="all")
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[Y_COL])
    y = df[Y_COL]
    return X, y


def make_train_test(df: pd.DataFrame):
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


class _InfToNanConverter(BaseEstimator, TransformerMixin):
    """Convert infinity values to NaN for downstream imputation."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[np.isinf(X)] = np.nan
        return X


def make_preprocessor(add_clusters: bool = True) -> Pipeline:
    """Build a preprocessing pipeline: impute → scale → optionally cluster.

    Steps:
    1. _InfToNanConverter() — converts infinity values to NaN
    2. SimpleImputer(strategy="median") — fills NaNs with column medians
    3. StandardScaler() — zero-mean, unit-variance scaling
    4. (if add_clusters) ClusterFeatureAdder() — appends KMeans cluster id

    Returns:
        sklearn.pipeline.Pipeline with all steps fit-then-transform-safe.
    """
    steps = [
        ("inf_to_nan", _InfToNanConverter()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if add_clusters:
        steps.append(("cluster", ClusterFeatureAdder()))
    return Pipeline(steps)


class ClusterFeatureAdder(BaseEstimator, TransformerMixin):
    """Append an unsupervised KMeans cluster id as an extra feature column.

    KMeans is fit on the training data passed to `fit` ONLY. Because this
    transformer sits inside the pipeline, cross-validation refits it on each
    fold's training portion, so cluster centroids never see validation/test
    rows (unlike the old precomputed cluster columns, which were fit on the
    full dataset). The cluster id encodes which broad soil/geo profile group a
    sample resembles.
    """

    def __init__(self, n_clusters: int = N_CLUSTERS, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=20
        ).fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        labels = self.kmeans_.predict(X).reshape(-1, 1)
        return np.hstack([X, labels])


def _pipe(estimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(add_clusters=True)), ("clf", estimator)])


def sklearn_search_spaces() -> dict[str, tuple[Pipeline, dict]]:
    """Registry of sklearn model pipelines + grids for CV-based selection.

    Each pipeline is preprocessing (impute -> scale -> cluster, fold-safe) +
    a classifier, so `GridSearchCV` can cross-validate the whole thing without
    leaking test/validation rows into imputation, scaling, or clustering.

    LogisticRegression uses solver="saga" with penalty="elasticnet" so that
    `l1_ratio` is a valid parameter (the old default solver rejected it on
    modern sklearn).
    """
    return {
        "logistic_regression": (
            _pipe(LogisticRegression(
                solver="saga", penalty="elasticnet", max_iter=5000,
                random_state=RANDOM_STATE)),
            {"clf__C": [0.01, 0.1, 1, 4, 8], "clf__l1_ratio": [0.0, 1.0]},
        ),
        "knn": (
            _pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [2, 5, 10, 15, 20], "clf__weights": ["uniform", "distance"]},
        ),
        "decision_tree": (
            _pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            {"clf__max_depth": [50, 100, 200, None], "clf__min_samples_split": [2, 10, 20, 50]},
        ),
        "random_forest": (
            _pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            {"clf__n_estimators": [100, 300, 500], "clf__max_depth": [None, 10, 50],
             "clf__min_samples_leaf": [1, 2, 4]},
        ),
        "svm": (
            _pipe(SVC(probability=True, max_iter=20000, random_state=RANDOM_STATE)),
            {"clf__C": [1, 4], "clf__kernel": ["linear", "rbf"]},
        ),
        "gbm": (
            _pipe(GradientBoostingClassifier(random_state=RANDOM_STATE)),
            {"clf__learning_rate": [0.01, 0.05, 0.1, 0.2], "clf__n_estimators": [100, 200, 400, 800]},
        ),
    }


def build_nn(input_dim: int, n_layers: int, n_neurons: int):
    """Build and compile a small feed-forward classifier.

    Dense relu stack (n_layers x n_neurons) -> single sigmoid output unit,
    trained with adam / binary_crossentropy for binary classification.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(Dense(n_neurons, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def nn_grid() -> list[dict]:
    """Small hyperparameter grid for the neural net (vs. an exhaustive sweep).

    Kept deliberately tiny so the manual CV loop in `run_nn_selection` finishes
    in a few minutes instead of hours.
    """
    return [
        {"n_layers": 1, "n_neurons": 32, "epochs": 20, "batch_size": 64},
        {"n_layers": 2, "n_neurons": 64, "epochs": 20, "batch_size": 64},
        {"n_layers": 3, "n_neurons": 128, "epochs": 20, "batch_size": 64},
        {"n_layers": 4, "n_neurons": 128, "epochs": 20, "batch_size": 64},
    ]


def run_nn_selection(X_train, y_train, scoring: str = "accuracy") -> dict:
    """Select the best NN hyperparameters via a manual, leakage-free CV loop.

    GridSearchCV is intentionally NOT used here (that would require a scikeras
    wrapper). Instead we manually loop over `nn_grid()` configs and a seeded
    `StratifiedKFold(5)`, re-fitting `make_preprocessor()` on each fold's
    TRAINING split only (never on the validation split) and reseeding via
    `set_seeds()` before building/fitting each fold's model, so results are
    both leakage-free and reproducible.
    """
    from sklearn.metrics import accuracy_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_arr = np.asarray(y_train)
    X_df = pd.DataFrame(X_train).reset_index(drop=True)
    y_ser = pd.Series(y_arr).reset_index(drop=True)

    best = {"params": None, "cv_best": -1.0}
    for cfg in nn_grid():
        fold_scores = []
        for tr_idx, va_idx in cv.split(X_df, y_ser):
            set_seeds()
            pre = make_preprocessor(add_clusters=True)
            X_tr = pre.fit_transform(X_df.iloc[tr_idx])
            X_va = pre.transform(X_df.iloc[va_idx])
            model = build_nn(X_tr.shape[1], cfg["n_layers"], cfg["n_neurons"])
            model.fit(X_tr, y_ser.iloc[tr_idx], epochs=cfg["epochs"],
                      batch_size=cfg["batch_size"], verbose=0)
            proba = model.predict(X_va, verbose=0).flatten()
            fold_scores.append(accuracy_score(y_ser.iloc[va_idx], (proba > 0.5).astype(int)))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best["cv_best"]:
            best = {"params": cfg, "cv_best": mean_score}
    return best


def run_sklearn_selection(X_train, y_train, scoring: str = "accuracy") -> dict:
    """Select the best hyperparameters per model via CV on the TRAINING set only.

    Uses `GridSearchCV` with a seeded `StratifiedKFold(5)` so every candidate
    is scored purely on training-data cross-validation; the test set is never
    touched here (evaluation on held-out test data happens later, elsewhere).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, (pipe, grid) in sklearn_search_spaces().items():
        search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        results[name] = {
            "estimator": search.best_estimator_,
            "cv_best": float(search.best_score_),
            "params": dict(search.best_params_),
        }
    return results
