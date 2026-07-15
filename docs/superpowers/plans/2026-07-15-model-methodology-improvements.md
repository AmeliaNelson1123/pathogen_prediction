# Model Methodology Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace test-set model selection with leakage-free cross-validation in a single shared module, so the project reports accurate, reproducible metrics and the deployed models match the report.

**Architecture:** All modeling logic moves into one importable module, `preparation/pipeline_utils.py`. Preprocessing (median imputation → standardization → KMeans cluster feature) lives inside an sklearn `Pipeline` so it is refit on training folds only. sklearn models are tuned with `GridSearchCV`; the neural net is tuned with a small manual `StratifiedKFold` loop. A single stratified hold-out test set is scored once. The two notebooks become thin drivers that import the module (no notebook-to-notebook dependency). The deploy script and FastAPI backend consume the same module and a single `best_configs.json` so the report and the served models cannot drift.

**Tech Stack:** Python 3.12 (arm64/Apple Silicon), scikit-learn, TensorFlow 2.16 (bundles Keras 3), pandas, numpy, matplotlib/seaborn/plotly, pytest, jupyter/nbconvert, FastAPI (existing backend).

## Global Constraints

- **Python version / arch:** this machine is **Apple Silicon (arm64)**; the working interpreter is **Python 3.12** (`/usr/local/bin/python3.12`, universal2). Env already built at `.venv` (Task 1 done). Supported range 3.10–3.12.
- **TensorFlow pin:** **`tensorflow==2.16.2`** — `2.15.1` has NO arm64 macOS wheel (its Intel wheel needs AVX and aborts under Rosetta). 2.16.2 is the earliest main-package version with arm64 wheels and is compatible with `numpy==1.26.4` / `scikit-learn==1.4.0`. TF 2.16 bundles **Keras 3** (no separate `keras` pin). `.keras` save/load and `tf.keras` Sequential/Dense/Input all work under Keras 3.
- **Determinism:** seed everything with `RANDOM_STATE = 42` — `random`, `numpy`, `tensorflow`, every estimator that accepts `random_state`, and every split.
- **Selection metric:** **accuracy**, measured by cross-validation on the **training data only**. The hold-out test set is scored exactly once, at the end.
- **Reported metrics:** always report accuracy **plus** precision, recall, F1, ROC-AUC, PR-AUC, and the confusion matrix. Accuracy is never removed.
- **No leakage:** imputation, scaling, and KMeans must be fit inside CV folds via the pipeline. The precomputed `cluster_kmeans` / `scaled_cluster_kmeans` columns and `log of index` must be dropped from features.
- **Single source of truth:** hyperparameters used for deployment come from `best_configs.json` produced by the module — never hardcoded separately.
- **Attribution:** do NOT add any `Co-Authored-By: Claude`/"Generated with Claude" trailer to commits or PRs, and do not mention Claude/Anthropic in any file.
- **Pinning:** pin every dependency version in `requirements.txt`; keep `scikit-learn` and `numpy` versions identical between the training env and `website/requirements.txt` so pickled models load in the backend.

---

## File Structure

**Created:**
- `preparation/pipeline_utils.py` — all shared modeling logic (constants, seeding, data prep, split, cluster transformer, preprocessor, model registry, CV selection, evaluation, orchestration, config I/O).
- `preparation/tests/__init__.py` — empty, makes tests importable.
- `preparation/tests/test_pipeline_utils.py` — pytest unit/integration tests for the module.
- `preparation/tests/conftest.py` — shared fixtures (synthetic frame, path to real data).
- `preparation/data_results/best_configs.json` — CV-selected best hyperparameters per model and per data variant (single source of truth for deployment).
- `dev-requirements.txt` — dev-only tooling (pytest, jupyter, nbconvert).

**Modified:**
- `requirements.txt` — dedupe, drop stdlib `pathlib`, add `plotly`, pin all versions.
- `preparation/Run_and_Test_Models.ipynb` — becomes a thin driver importing `pipeline_utils`.
- `preparation/Analyze_Models.ipynb` — thin driver; adds recall/PR-AUC/calibration/DT-vs-RF distribution; no dependency on the training notebook.
- `preparation/saving_selected_models_for_pipeline.py` — trains from `best_configs.json`; saves preprocess pipeline + estimators; NN via Keras format.
- `website/backend/main.py` — load NN via Keras; use the saved preprocess pipeline; remove the dormant manual KMeans/`ADD_CLUSTERS` path.
- `website/requirements.txt` — align `scikit-learn`/`numpy` pins with training env.
- `ReadMe.md` — corrected Table 1 metrics and Python-version note.

**Module public API (names other tasks depend on — defined in Task 2 onward):**
```
RANDOM_STATE = 42
TEST_SIZE = 0.22
N_CLUSTERS = 3
Y_COL = "binary_listeria_presense"
DATA_FILENAME = "ListeriaSoil_clean_log.csv"
LEAK_COLS = ["index", "log of index", "Unnamed: 0", "cluster_kmeans", "scaled_cluster_kmeans"]
RAW_COUNT_COL = "Number of Listeria isolates obtained"
SOIL_VARS_ONLY: list[str]
LONGLAT_VARS_ONLY: list[str]

project_root() -> Path
data_path() -> Path
set_seeds(seed: int = RANDOM_STATE) -> None
load_and_prep(path: Path | None = None) -> pd.DataFrame          # returns df incl. Y_COL, numeric, NaNs kept
split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]
make_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
class ClusterFeatureAdder(BaseEstimator, TransformerMixin)       # n_clusters, random_state
make_preprocessor(add_clusters: bool = True) -> Pipeline         # imputer -> scaler -> [cluster]
build_nn(input_dim: int, n_layers: int, n_neurons: int) -> keras.Model
sklearn_search_spaces() -> dict[str, tuple[Pipeline, dict]]      # name -> (pipeline, param_grid)
nn_grid() -> list[dict]                                          # small list of {n_layers,n_neurons,epochs,batch_size}
run_sklearn_selection(X_train, y_train, scoring="accuracy") -> dict[str, dict]  # name -> {estimator, cv_best, params}
run_nn_selection(X_train, y_train, scoring="accuracy") -> dict   # {estimator_config, cv_best, params}
predict_proba_any(estimator, X) -> np.ndarray
evaluate(y_true, y_proba, threshold=0.5) -> dict                 # accuracy,precision,recall,f1,roc_auc,pr_auc,confusion_matrix
cv_accuracy_distribution(pipeline, X_train, y_train) -> dict     # {"mean","std","scores"}
select_and_evaluate(df, add_clusters=True) -> dict               # full run: per-model holdout metrics + best params
save_best_configs(configs: dict, path: Path | None = None) -> None
load_best_configs(path: Path | None = None) -> dict
```

---

## Task 1: Reproducible environment & dependency pinning — ✅ COMPLETED

Done directly by the controller (env bootstrap is iterative/interactive, not a good subagent fit). What was actually done, and what later tasks must use:

- **Interpreter:** `/usr/local/bin/python3.12` (universal2) forced to arm64. The venv is `.venv/` (already matched by `.gitignore`'s `.venv/` pattern — no gitignore change needed). Created with `arch -arm64 /usr/local/bin/python3.12 -m venv .venv`. The venv runs arm64 by default (no `arch` prefix needed in later commands).
- **Activation for every later task:** `source .venv/bin/activate`.
- **Jupyter kernel name (for nbconvert):** `pathogen-venv`.
- **`requirements.txt`** (committed) now reads exactly:
  ```
  pandas==2.2.0
  numpy==1.26.4
  scipy==1.13.0
  matplotlib==3.9.0
  seaborn==0.13.0
  plotly==5.22.0
  scikit-learn==1.4.0
  tensorflow==2.16.2
  joblib==1.4.2
  ```
  (Removed duplicate `numpy`, stdlib `pathlib`, and the invalid `keras==2.15.0` pin — TF 2.16 bundles Keras 3.)
- **`dev-requirements.txt`** (committed): `pytest==8.2.0`, `jupyter==1.0.0`, `nbconvert==7.16.4`, `ipykernel==6.29.0`.
- **Verified:** `machine=arm64`, `tf 2.16.2 | numpy 1.26.4 | sklearn 1.4.0 | pandas 2.2.0`; a tiny TF build+fit+predict succeeds (no AVX abort).
- **Committed:** `72ab338 build: pin arm64-compatible deps (Python 3.12, TF 2.16.2) and add dev tooling`.

---

## Task 2: Module scaffold — constants, seeding, paths

**Files:**
- Create: `preparation/pipeline_utils.py`
- Create: `preparation/tests/__init__.py` (empty)
- Create: `preparation/tests/conftest.py`
- Create: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Produces: `RANDOM_STATE`, `TEST_SIZE`, `N_CLUSTERS`, `Y_COL`, `DATA_FILENAME`, `LEAK_COLS`, `RAW_COUNT_COL`, `SOIL_VARS_ONLY`, `LONGLAT_VARS_ONLY`, `project_root()`, `data_path()`, `set_seeds()`.

- [ ] **Step 1: Write the failing test**

`preparation/tests/conftest.py`:
```python
from pathlib import Path
import pytest

@pytest.fixture
def real_data_path():
    return Path(__file__).resolve().parents[2] / "data" / "ListeriaSoil_clean_log.csv"
```

`preparation/tests/test_pipeline_utils.py`:
```python
import numpy as np
import preparation.pipeline_utils as pu


def test_constants_present():
    assert pu.RANDOM_STATE == 42
    assert pu.TEST_SIZE == 0.22
    assert pu.Y_COL == "binary_listeria_presense"
    assert "log of index" in pu.LEAK_COLS
    assert "cluster_kmeans" in pu.LEAK_COLS


def test_data_path_exists(real_data_path):
    assert pu.data_path() == real_data_path
    assert pu.data_path().exists()


def test_set_seeds_is_deterministic():
    pu.set_seeds(42)
    a = np.random.rand(5)
    pu.set_seeds(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'preparation.pipeline_utils'` (or attribute errors).

Note: run pytest from the repo root so `preparation` is importable. If import fails, add an empty `preparation/__init__.py`.

- [ ] **Step 3: Write minimal implementation**

Create `preparation/pipeline_utils.py`:
```python
"""Shared, leakage-free modeling utilities for the Listeria soil project.

Single source of truth imported by the training notebook, the analysis
notebook, the deploy script, and (indirectly) the backend.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

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
```

Also create empty `preparation/tests/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/
git commit -m "feat: scaffold pipeline_utils with constants, seeding, paths"
```

---

## Task 3: `load_and_prep` — leakage-free data preparation

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Consumes: `LEAK_COLS`, `RAW_COUNT_COL`, `Y_COL`, `data_path()`.
- Produces: `load_and_prep(path=None) -> pd.DataFrame` (numeric features + `Y_COL`; NaNs preserved for the pipeline imputer; no sentinel fill; leak columns dropped).

- [ ] **Step 1: Write the failing test**

Add to `test_pipeline_utils.py`:
```python
import pandas as pd
import preparation.pipeline_utils as pu


def test_load_and_prep_target_is_binary(real_data_path):
    df = pu.load_and_prep()
    assert pu.Y_COL in df.columns
    assert set(df[pu.Y_COL].unique()) <= {0, 1}
    # dataset is balanced 50/50 (311/311)
    assert df[pu.Y_COL].sum() == 311
    assert len(df) == 622


def test_load_and_prep_drops_leak_and_raw_columns():
    df = pu.load_and_prep()
    for c in pu.LEAK_COLS + [pu.RAW_COUNT_COL]:
        assert c not in df.columns, f"{c} should have been dropped"


def test_load_and_prep_features_are_numeric_and_keep_nans():
    df = pu.load_and_prep()
    X = df.drop(columns=[pu.Y_COL])
    # every feature column numeric
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    # NaNs are preserved (NOT replaced with +/-99999 sentinels)
    assert (X.abs() == 99999).sum().sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k load_and_prep -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'load_and_prep'`.

- [ ] **Step 3: Write minimal implementation**

Add to `pipeline_utils.py` (add `import pandas as pd` at top):
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k load_and_prep -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: leakage-free load_and_prep (drop leak cols, keep NaNs)"
```

---

## Task 4: Stratified train/test split helpers

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Consumes: `Y_COL`, `TEST_SIZE`, `RANDOM_STATE`, `load_and_prep`.
- Produces: `split_xy(df) -> (X, y)`, `make_train_test(df) -> (X_train, X_test, y_train, y_test)` (stratified, seeded).

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_split_xy_separates_target():
    df = pu.load_and_prep()
    X, y = pu.split_xy(df)
    assert pu.Y_COL not in X.columns
    assert y.name == pu.Y_COL
    assert len(X) == len(y) == 622


def test_make_train_test_is_stratified_and_deterministic():
    df = pu.load_and_prep()
    Xtr1, Xte1, ytr1, yte1 = pu.make_train_test(df)
    Xtr2, Xte2, ytr2, yte2 = pu.make_train_test(df)
    # deterministic
    assert list(ytr1.index) == list(ytr2.index)
    # test size ~22%
    assert abs(len(Xte1) / 622 - 0.22) < 0.01
    # stratified: train/test prevalence within 3pp of overall 0.5
    assert abs(ytr1.mean() - 0.5) < 0.03
    assert abs(yte1.mean() - 0.5) < 0.03
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "split_xy or train_test" -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

Add (add `from sklearn.model_selection import train_test_split` at top):
```python
def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[Y_COL])
    y = df[Y_COL]
    return X, y


def make_train_test(df: pd.DataFrame):
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "split_xy or train_test" -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: stratified seeded train/test split helpers"
```

---

## Task 5: `ClusterFeatureAdder` — leakage-safe KMeans feature

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Consumes: `N_CLUSTERS`, `RANDOM_STATE`.
- Produces: `class ClusterFeatureAdder(BaseEstimator, TransformerMixin)` with `__init__(self, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)`; `fit(X, y=None)` fits KMeans on `X` only; `transform(X)` appends one integer cluster-id column. Operates on numpy arrays (placed after the scaler).

- [ ] **Step 1: Write the failing test**

Add:
```python
import numpy as np


def test_cluster_adder_appends_one_column():
    X = np.random.RandomState(0).rand(50, 4)
    adder = pu.ClusterFeatureAdder(n_clusters=3)
    Xt = adder.fit_transform(X)
    assert Xt.shape == (50, 5)
    labels = Xt[:, -1]
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_cluster_adder_is_fit_on_training_only():
    # centroids depend ONLY on fit data, not on later transform inputs
    rng = np.random.RandomState(1)
    Xtrain = rng.rand(60, 4)
    Xother = rng.rand(10, 4) + 100.0  # far away block
    adder = pu.ClusterFeatureAdder(n_clusters=3).fit(Xtrain)
    centroids_before = adder.kmeans_.cluster_centers_.copy()
    _ = adder.transform(Xother)  # transform must NOT refit
    assert np.allclose(centroids_before, adder.kmeans_.cluster_centers_)


def test_cluster_adder_deterministic():
    X = np.random.RandomState(2).rand(40, 3)
    a = pu.ClusterFeatureAdder().fit_transform(X)
    b = pu.ClusterFeatureAdder().fit_transform(X)
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k cluster_adder -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

Add (add `from sklearn.base import BaseEstimator, TransformerMixin` and `from sklearn.cluster import KMeans` at top):
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k cluster_adder -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: leakage-safe ClusterFeatureAdder (KMeans fit on train fold only)"
```

---

## Task 6: `make_preprocessor` — impute → scale → cluster

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Consumes: `ClusterFeatureAdder`.
- Produces: `make_preprocessor(add_clusters=True) -> sklearn.pipeline.Pipeline` (steps: `imputer` = `SimpleImputer(strategy="median")`, `scaler` = `StandardScaler()`, and if `add_clusters`, `cluster` = `ClusterFeatureAdder()`). Output has no NaNs.

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_preprocessor_removes_nans_and_adds_cluster_col():
    df = pu.load_and_prep()
    X, y = pu.split_xy(df)
    pre = pu.make_preprocessor(add_clusters=True)
    Xt = pre.fit_transform(X)
    assert not np.isnan(Xt).any()
    # one extra column for the cluster id
    assert Xt.shape[1] == X.shape[1] + 1


def test_preprocessor_without_clusters_matches_feature_count():
    df = pu.load_and_prep()
    X, _ = pu.split_xy(df)
    pre = pu.make_preprocessor(add_clusters=False)
    Xt = pre.fit_transform(X)
    assert Xt.shape[1] == X.shape[1]
    assert not np.isnan(Xt).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k preprocessor -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

Add (add `from sklearn.pipeline import Pipeline`, `from sklearn.impute import SimpleImputer`, `from sklearn.preprocessing import StandardScaler` at top):
```python
def make_preprocessor(add_clusters: bool = True) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if add_clusters:
        steps.append(("cluster", ClusterFeatureAdder()))
    return Pipeline(steps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k preprocessor -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: make_preprocessor (median impute -> scale -> cluster) in-pipeline"
```

---

## Task 7: sklearn model registry + CV selection (incl. LogisticRegression fix)

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Consumes: `make_preprocessor`, `RANDOM_STATE`.
- Produces:
  - `sklearn_search_spaces() -> dict[str, tuple[Pipeline, dict]]` — keys: `logistic_regression`, `knn`, `decision_tree`, `random_forest`, `svm`, `gbm`. Each pipeline is `Pipeline([("prep", make_preprocessor()), ("clf", <estimator>)])` and the grid keys are `clf__*`.
  - `run_sklearn_selection(X_train, y_train, scoring="accuracy") -> dict[str, dict]` — per model `{"estimator": fitted_best_pipeline, "cv_best": float, "params": {clf__..: val}}`. Uses `GridSearchCV(..., cv=StratifiedKFold(5, shuffle=True, random_state=42))`.

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_search_spaces_have_all_models():
    spaces = pu.sklearn_search_spaces()
    assert set(spaces) == {
        "logistic_regression", "knn", "decision_tree",
        "random_forest", "svm", "gbm",
    }
    # LogisticRegression must be configured so l1_ratio is valid (saga+elasticnet)
    lr_pipe, lr_grid = spaces["logistic_regression"]
    clf = lr_pipe.named_steps["clf"]
    assert clf.get_params()["solver"] == "saga"
    assert clf.get_params()["penalty"] == "elasticnet"


def test_logistic_regression_fits_without_api_error():
    # the exact call that crashed on modern sklearn must now work
    df = pu.load_and_prep()
    Xtr, Xte, ytr, yte = pu.make_train_test(df)
    lr_pipe, lr_grid = pu.sklearn_search_spaces()["logistic_regression"]
    lr_pipe.set_params(clf__l1_ratio=1.0, clf__C=1.0).fit(Xtr, ytr)  # must not raise


def test_run_sklearn_selection_returns_fitted_best(monkeypatch):
    df = pu.load_and_prep()
    Xtr, Xte, ytr, yte = pu.make_train_test(df)
    # keep it fast: restrict to two cheap models via a small monkeypatched space
    small = {k: pu.sklearn_search_spaces()[k] for k in ["decision_tree", "gbm"]}
    monkeypatch.setattr(pu, "sklearn_search_spaces", lambda: small)
    out = pu.run_sklearn_selection(Xtr, ytr)
    assert set(out) == {"decision_tree", "gbm"}
    for name, res in out.items():
        assert 0.5 <= res["cv_best"] <= 1.0
        # fitted: can predict
        assert res["estimator"].predict(Xte).shape[0] == len(yte)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "search_spaces or logistic or run_sklearn" -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

Add (add imports for the estimators and `GridSearchCV`, `StratifiedKFold`):
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def _pipe(estimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(add_clusters=True)), ("clf", estimator)])


def sklearn_search_spaces():
    return {
        "logistic_regression": (
            # saga + elasticnet makes l1_ratio valid (the old lbfgs+l1_ratio call crashed)
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


def run_sklearn_selection(X_train, y_train, scoring: str = "accuracy") -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, (pipe, grid) in sklearn_search_spaces().items():
        search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        results[name] = {
            "estimator": search.best_estimator_,
            "cv_best": float(search.best_score_),
            "params": {k: v for k, v in search.best_params_.items()},
        }
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "search_spaces or logistic or run_sklearn" -v`
Expected: 3 passed (the `run_sklearn` test may take ~1–2 min for GBM; that's fine).

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: sklearn model registry + GridSearchCV selection (fix LogisticRegression API)"
```

---

## Task 8: Neural net builder + manual CV selection

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Consumes: `make_preprocessor`, `set_seeds`, `RANDOM_STATE`.
- Produces:
  - `build_nn(input_dim, n_layers, n_neurons) -> keras.Model` (Dense relu stack → sigmoid; adam/binary_crossentropy).
  - `nn_grid() -> list[dict]` — small grid, keys `n_layers`, `n_neurons`, `epochs`, `batch_size`.
  - `run_nn_selection(X_train, y_train, scoring="accuracy") -> dict` — `{"params": {...}, "cv_best": float}`. Manual `StratifiedKFold(5)`; the preprocessor is fit per fold on the fold's train split.

Rationale: the NN is tuned with a manual CV loop (not GridSearchCV) to avoid a scikeras dependency and to keep TF seeding explicit. The grid is deliberately small so it runs in minutes.

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_build_nn_shapes():
    pu.set_seeds()
    model = pu.build_nn(input_dim=10, n_layers=2, n_neurons=16)
    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 1)


def test_nn_grid_is_small():
    grid = pu.nn_grid()
    assert 1 <= len(grid) <= 12  # kept small on purpose
    assert all({"n_layers", "n_neurons", "epochs", "batch_size"} <= set(g) for g in grid)


def test_run_nn_selection_returns_params():
    df = pu.load_and_prep()
    Xtr, Xte, ytr, yte = pu.make_train_test(df)
    out = pu.run_nn_selection(Xtr, ytr)
    assert 0.5 <= out["cv_best"] <= 1.0
    assert {"n_layers", "n_neurons", "epochs", "batch_size"} <= set(out["params"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "build_nn or nn_grid or nn_selection" -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

Add:
```python
def build_nn(input_dim: int, n_layers: int, n_neurons: int):
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
    # Deliberately small (vs. the old 480-config exhaustive grid) so CV is tractable.
    return [
        {"n_layers": 1, "n_neurons": 32, "epochs": 20, "batch_size": 64},
        {"n_layers": 2, "n_neurons": 64, "epochs": 20, "batch_size": 64},
        {"n_layers": 3, "n_neurons": 128, "epochs": 20, "batch_size": 64},
        {"n_layers": 4, "n_neurons": 128, "epochs": 20, "batch_size": 64},
    ]


def run_nn_selection(X_train, y_train, scoring: str = "accuracy") -> dict:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "build_nn or nn_grid or nn_selection" -v`
Expected: 3 passed (the selection test may take a few minutes).

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: neural net builder + seeded manual-CV selection"
```

---

## Task 9: `evaluate` — full metric set on the hold-out

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Produces:
  - `predict_proba_any(estimator, X) -> np.ndarray` (handles sklearn `predict_proba[:,1]` and Keras `.predict().flatten()`).
  - `evaluate(y_true, y_proba, threshold=0.5) -> dict` with keys `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `confusion_matrix` (2×2 list).

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_evaluate_perfect_and_keys():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]
    m = pu.evaluate(y_true, y_proba)
    assert set(m) == {"accuracy", "precision", "recall", "f1",
                      "roc_auc", "pr_auc", "confusion_matrix"}
    assert m["accuracy"] == 1.0
    assert m["recall"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_predict_proba_any_sklearn():
    df = pu.load_and_prep()
    Xtr, Xte, ytr, yte = pu.make_train_test(df)
    _, (pipe, _) = "gbm", (pu.sklearn_search_spaces()["gbm"])
    pipe.fit(Xtr, ytr)
    proba = pu.predict_proba_any(pipe, Xte)
    assert proba.shape[0] == len(yte)
    assert ((proba >= 0) & (proba <= 1)).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "evaluate or predict_proba" -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

Add (add metric imports):
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)


def predict_proba_any(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    # Keras model
    return np.asarray(estimator.predict(X, verbose=0)).flatten()


def evaluate(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "evaluate or predict_proba" -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: evaluate() with full metric set + predict_proba_any"
```

---

## Task 10: `cv_accuracy_distribution` — honest DT-vs-RF comparison

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Consumes: `RANDOM_STATE`.
- Produces: `cv_accuracy_distribution(pipeline, X_train, y_train) -> dict` with `{"mean": float, "std": float, "scores": list[float]}` via `cross_val_score` on `StratifiedKFold(5)`. Used to show DT and RF overlap within noise.

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_cv_distribution_shape():
    df = pu.load_and_prep()
    Xtr, _, ytr, _ = pu.make_train_test(df)
    pipe, _ = pu.sklearn_search_spaces()["decision_tree"]
    dist = pu.cv_accuracy_distribution(pipe, Xtr, ytr)
    assert set(dist) == {"mean", "std", "scores"}
    assert len(dist["scores"]) == 5
    assert 0.5 <= dist["mean"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k cv_distribution -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

Add (add `from sklearn.model_selection import cross_val_score`):
```python
def cv_accuracy_distribution(pipeline, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1)
    return {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores.tolist()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k cv_distribution -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: cv_accuracy_distribution for honest model comparison"
```

---

## Task 11: Orchestration + artifacts (`select_and_evaluate`, `best_configs.json`)

**Files:**
- Modify: `preparation/pipeline_utils.py`
- Modify: `preparation/tests/test_pipeline_utils.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `select_and_evaluate(df, add_clusters=True) -> dict` — runs sklearn + NN selection on the train split, scores each model's best config once on the hold-out via `evaluate`, and returns `{model_name: {"cv_best": float, "params": {...}, "holdout": {metrics...}}}`. Deterministic.
  - `save_best_configs(configs, path=None)` / `load_best_configs(path=None)` — JSON round-trip to `preparation/data_results/best_configs.json`.

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_save_load_best_configs_roundtrip(tmp_path):
    cfg = {"gbm": {"cv_best": 0.87, "params": {"clf__learning_rate": 0.2}}}
    p = tmp_path / "best_configs.json"
    pu.save_best_configs(cfg, p)
    assert pu.load_best_configs(p) == cfg


def test_select_and_evaluate_is_deterministic_and_reproduces_known_range():
    df = pu.load_and_prep()
    r1 = pu.select_and_evaluate(df)
    r2 = pu.select_and_evaluate(df)
    # deterministic hold-out accuracy for GBM across repeated runs
    assert abs(r1["gbm"]["holdout"]["accuracy"] - r2["gbm"]["holdout"]["accuracy"]) < 1e-9
    # sanity: matches the ~0.86-0.90 range found in the repo (NOT 0.94)
    for name in ["gbm", "neural_net"]:
        assert 0.80 <= r1[name]["holdout"]["accuracy"] <= 0.93
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "best_configs or select_and_evaluate" -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

Add (add `import json`):
```python
def _default_configs_path() -> Path:
    return project_root() / "preparation" / "data_results" / "best_configs.json"


def save_best_configs(configs: dict, path: Path | None = None) -> None:
    path = path or _default_configs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(configs, f, indent=2)


def load_best_configs(path: Path | None = None) -> dict:
    with open(path or _default_configs_path()) as f:
        return json.load(f)


def select_and_evaluate(df: pd.DataFrame, add_clusters: bool = True) -> dict:
    set_seeds()
    Xtr, Xte, ytr, yte = make_train_test(df)

    results: dict = {}

    # sklearn families
    sk = run_sklearn_selection(Xtr, ytr, scoring="accuracy")
    for name, res in sk.items():
        proba = predict_proba_any(res["estimator"], Xte)
        results[name] = {
            "cv_best": res["cv_best"],
            "params": res["params"],
            "holdout": evaluate(yte, proba),
        }

    # neural net: select params, then refit best on full train split and score once
    nn = run_nn_selection(Xtr, ytr, scoring="accuracy")
    set_seeds()
    pre = make_preprocessor(add_clusters=add_clusters)
    Xtr_t = pre.fit_transform(Xtr)
    Xte_t = pre.transform(Xte)
    model = build_nn(Xtr_t.shape[1], nn["params"]["n_layers"], nn["params"]["n_neurons"])
    model.fit(Xtr_t, np.asarray(ytr), epochs=nn["params"]["epochs"],
              batch_size=nn["params"]["batch_size"], verbose=0)
    proba = np.asarray(model.predict(Xte_t, verbose=0)).flatten()
    results["neural_net"] = {
        "cv_best": nn["cv_best"],
        "params": nn["params"],
        "holdout": evaluate(yte, proba),
    }
    return results
```

Note on NN determinism: TF ops are not always bit-identical across runs; if the determinism assertion for `neural_net` proves flaky, the test only asserts it for `gbm` (already the case). Keep the NN seed call, and document residual NN variance in Task 13's write-up cell.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest preparation/tests/test_pipeline_utils.py -k "best_configs or select_and_evaluate" -v`
Expected: 2 passed. (`select_and_evaluate` runs the full search twice — allow several minutes. If too slow for routine test runs, mark it `@pytest.mark.slow`.)

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest preparation/tests/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add preparation/pipeline_utils.py preparation/tests/test_pipeline_utils.py
git commit -m "feat: select_and_evaluate orchestration + best_configs.json I/O"
```

---

## Task 12: Rewrite `Run_and_Test_Models.ipynb` as a thin driver + regenerate results

**Files:**
- Modify: `preparation/Run_and_Test_Models.ipynb`
- Modify: `preparation/data_results/results_for_ListeriaSoil_clean_log.csv` (regenerated)
- Modify: `preparation/data_results/top_3_scaled_models_summary.csv` (regenerated)
- Create: `preparation/data_results/best_configs.json` (written by the run)

**Interfaces:**
- Consumes: `pipeline_utils.select_and_evaluate`, `save_best_configs`.

- [ ] **Step 1: Replace the notebook body with driver cells**

Replace all code cells with these (keep/adjust the intro markdown). The notebook must add the repo root to `sys.path` so it can import the module, then run and persist:

Cell 1 (imports + path):
```python
import sys, json
from pathlib import Path
import pandas as pd
ROOT = Path.cwd().parents[0] if Path.cwd().name == "preparation" else Path.cwd()
sys.path.insert(0, str(ROOT))
import preparation.pipeline_utils as pu
pu.set_seeds()
```

Cell 2 (run selection + evaluation):
```python
df = pu.load_and_prep()
print("rows:", len(df), "| prevalence:", round(df[pu.Y_COL].mean(), 3))
results = pu.select_and_evaluate(df, add_clusters=True)
```

Cell 3 (persist results table — one row per model with holdout metrics + params):
```python
rows = []
for name, r in results.items():
    row = {"model used": name, "cv_accuracy": r["cv_best"], **r["holdout"]}
    row.update({f"param.{k}": v for k, v in r["params"].items()})
    rows.append(row)
results_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
out = ROOT / "preparation" / "data_results" / "results_for_ListeriaSoil_clean_log.csv"
results_df.to_csv(out, index=False)
results_df[["model used", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
```

Cell 4 (top-3 summary + best_configs.json):
```python
top3 = results_df.head(3)
top3.to_csv(ROOT / "preparation" / "data_results" / "top_3_scaled_models_summary.csv", index=False)
pu.save_best_configs({k: {"cv_best": v["cv_best"], "params": v["params"]} for k, v in results.items()})
top3[["model used", "accuracy", "recall", "roc_auc"]]
```

- [ ] **Step 2: Execute the notebook headless**

Run:
```bash
cd /Users/yeonjinjung/Documents/pathogen_prediction
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=pathogen-venv \
  preparation/Run_and_Test_Models.ipynb
```
Expected: exit 0, no traceback in output cells.

- [ ] **Step 3: Verify regenerated results are sane**

Run:
```bash
python -c "import pandas as pd; d=pd.read_csv('preparation/data_results/top_3_scaled_models_summary.csv'); print(d[['model used','accuracy']]); assert (d['accuracy']<0.93).all(), 'accuracy unexpectedly high (>0.93) — check for leakage'"
cat preparation/data_results/best_configs.json
```
Expected: top-3 accuracies in the ~0.85–0.92 range; `best_configs.json` contains params per model. **No value near 0.94.**

- [ ] **Step 4: Commit**

```bash
git add preparation/Run_and_Test_Models.ipynb preparation/data_results/
git commit -m "refactor: training notebook uses shared module; regenerate honest results"
```

---

## Task 13: Rewrite `Analyze_Models.ipynb` (recall/PR-AUC + calibration + DT-vs-RF)

**Files:**
- Modify: `preparation/Analyze_Models.ipynb`

**Interfaces:**
- Consumes: `pipeline_utils` (`load_and_prep`, `make_train_test`, `sklearn_search_spaces`, `cv_accuracy_distribution`, `predict_proba_any`), `preparation/data_results/results_for_ListeriaSoil_clean_log.csv`.

- [ ] **Step 1: Replace the notebook body with self-contained analysis cells**

The notebook must NOT depend on functions defined in the training notebook — it imports everything from `pipeline_utils`.

Cell 1 (imports + path): same header as Task 12 Cell 1, plus:
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
```

Cell 2 (load the regenerated results table and show the full metric set):
```python
results_df = pd.read_csv(ROOT / "preparation" / "data_results" / "results_for_ListeriaSoil_clean_log.csv")
results_df[["model used", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
```

Cell 3 (honest DT-vs-RF comparison via CV distributions):
```python
df = pu.load_and_prep(); Xtr, Xte, ytr, yte = pu.make_train_test(df)
for name in ["decision_tree", "random_forest"]:
    pipe, _ = pu.sklearn_search_spaces()[name]
    dist = pu.cv_accuracy_distribution(pipe, Xtr, ytr)
    print(f"{name:15s} CV accuracy = {dist['mean']:.4f} +/- {dist['std']:.4f}")
print("\\nInterpretation: overlapping mean+/-std => the DT-vs-RF gap is noise, "
      "not a real ranking (answers the reviewers' question).")
```

Cell 4 (probability calibration of the best model — feeds the website risk score):
```python
best_name = results_df.iloc[0]["model used"]
if best_name == "neural_net":
    best_name = results_df[results_df["model used"] != "neural_net"].iloc[0]["model used"]
pipe, _ = pu.sklearn_search_spaces()[best_name]
pipe.fit(Xtr, ytr)
proba = pu.predict_proba_any(pipe, Xte)
frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10)
plt.plot(mean_pred, frac_pos, "o-", label=best_name)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
plt.title(f"Calibration — {best_name} (Brier={brier_score_loss(yte, proba):.3f})")
plt.legend(); plt.show()
print("Caveat: training data is balanced 50/50; real-world soil prevalence is much lower, "
      "so deployed probabilities are relative risk unless base-rate-corrected.")
```

- [ ] **Step 2: Execute the notebook headless**

Run:
```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=pathogen-venv \
  preparation/Analyze_Models.ipynb
```
Expected: exit 0, no traceback; a calibration plot renders; DT and RF CV means print within ~1 std of each other.

- [ ] **Step 3: Commit**

```bash
git add preparation/Analyze_Models.ipynb
git commit -m "refactor: analysis notebook self-contained; add recall/PR-AUC, calibration, DT-vs-RF"
```

---

## Task 14: Correct README Table 1 + Python-version note; add cluster note

**Files:**
- Modify: `ReadMe.md`

**Interfaces:**
- Consumes: `preparation/data_results/top_3_scaled_models_summary.csv`.

- [ ] **Step 1: Read the regenerated numbers**

Run: `python -c "import pandas as pd; print(pd.read_csv('preparation/data_results/top_3_scaled_models_summary.csv').to_string(index=False))"`

- [ ] **Step 2: Update the model results table in `ReadMe.md`**

Find the existing Table 1 (the one reporting GBM 94.16%/94.2%) and replace the numbers with the regenerated hold-out metrics (accuracy, recall, ROC-AUC per model). Remove any "94.2%" claim.

- [ ] **Step 3: Fix the Python-version claim and add the cluster note**

Replace the "Python 3.10-3.13 compatible" claim with:
```
Requires Python 3.10–3.12 (uses tensorflow 2.16.x). On Apple Silicon, install with a
native-arm64 Python — the Intel tensorflow wheel aborts under Rosetta (AVX).
```
Add one line near the methods/data description:
```
Cluster feature: an unsupervised KMeans (k=3) grouping of soil/geographic profiles,
fit inside each cross-validation fold on training data only (no leakage).
```

- [ ] **Step 4: Verify no stale numbers remain**

Run: `grep -n "94.2\|94.16\|3.13 compatible\|3.10-3.13" ReadMe.md || echo "clean"`
Expected: `clean`.

- [ ] **Step 5: Commit**

```bash
git add ReadMe.md
git commit -m "docs: correct Table 1 metrics, fix Python-version constraint, note cluster feature"
```

---

## Task 15: Deploy the CV-selected models (one source of truth, Keras format)

**Files:**
- Modify: `preparation/saving_selected_models_for_pipeline.py`

**Interfaces:**
- Consumes: `pipeline_utils` (`load_and_prep`, `make_train_test`, `load_best_configs`, `make_preprocessor`, `build_nn`, feature lists), `best_configs.json`.
- Produces (in `website/backend/models/`): `preprocess_<variant>.joblib`, `gbm_<variant>.joblib`, `svm_<variant>.joblib`, `neural_net_<variant>.keras` for `variant in {main, longlat_only, soil_only}`.

- [ ] **Step 1: Rewrite the script to train from `best_configs.json`**

Replace the file contents with a script that, for each of the three feature variants, builds the feature frame (all / longlat-only / soil-only), fits one shared preprocessor, and trains GBM + SVM + NN using the params from `best_configs.json`:
```python
"""Train and persist the deployed models from the CV-selected best configs.

Single source of truth: reads preparation/data_results/best_configs.json so the
served models match the reported metrics. Run AFTER Run_and_Test_Models.ipynb.
"""
from pathlib import Path
import joblib, numpy as np
import preparation.pipeline_utils as pu

OUTPUT = pu.project_root() / "website" / "backend" / "models"
OUTPUT.mkdir(parents=True, exist_ok=True)


def _feature_frame(df, variant):
    X = df.drop(columns=[pu.Y_COL])
    if variant == "longlat_only":
        keep = [c for c in pu.LONGLAT_VARS_ONLY if c in X.columns]
        X = X[keep]
    elif variant == "soil_only":
        keep = [c for c in pu.SOIL_VARS_ONLY if c in X.columns]
        X = X[keep]
    out = X.copy(); out[pu.Y_COL] = df[pu.Y_COL].values
    return out


def _clf_params(cfg):
    return {k.replace("clf__", ""): v for k, v in cfg["params"].items()}


def train_variant(df, variant, configs):
    pu.set_seeds()
    sub = _feature_frame(df, variant)
    Xtr, _, ytr, _ = pu.make_train_test(sub)

    pre = pu.make_preprocessor(add_clusters=(variant == "main"))
    Xtr_t = pre.fit_transform(Xtr)
    joblib.dump(pre, OUTPUT / f"preprocess_{variant}.joblib")

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    gbm = GradientBoostingClassifier(random_state=pu.RANDOM_STATE, **_clf_params(configs["gbm"]))
    gbm.fit(Xtr_t, ytr); joblib.dump(gbm, OUTPUT / f"gbm_{variant}.joblib")

    svm = SVC(probability=True, max_iter=20000, random_state=pu.RANDOM_STATE, **_clf_params(configs["svm"]))
    svm.fit(Xtr_t, ytr); joblib.dump(svm, OUTPUT / f"svm_{variant}.joblib")

    nn_p = configs["neural_net"]["params"]
    nn = pu.build_nn(Xtr_t.shape[1], nn_p["n_layers"], nn_p["n_neurons"])
    nn.fit(Xtr_t, np.asarray(ytr), epochs=nn_p["epochs"], batch_size=nn_p["batch_size"], verbose=0)
    nn.save(OUTPUT / f"neural_net_{variant}.keras")
    print(f"saved {variant}: gbm, svm, neural_net, preprocess")


def main():
    df = pu.load_and_prep()
    configs = pu.load_best_configs()
    for variant in ["main", "longlat_only", "soil_only"]:
        train_variant(df, variant, configs)


if __name__ == "__main__":
    main()
```
Note: `best_configs.json` stores sklearn params as `clf__*` (from GridSearchCV) and NN params as plain keys; `_clf_params` strips the `clf__` prefix for direct estimator construction.

- [ ] **Step 2: Run the script**

Run:
```bash
cd /Users/yeonjinjung/Documents/pathogen_prediction && source .venv/bin/activate
python -m preparation.saving_selected_models_for_pipeline
ls -1 website/backend/models/ | grep -E "preprocess_|_main|_soil_only|_longlat_only"
```
Expected: for each variant, `preprocess_*.joblib`, `gbm_*.joblib`, `svm_*.joblib`, `neural_net_*.keras` exist.

- [ ] **Step 3: Smoke-test that saved artifacts load and predict**

Run:
```bash
python -c "
import joblib, numpy as np, tensorflow as tf
import preparation.pipeline_utils as pu
pre = joblib.load('website/backend/models/preprocess_main.joblib')
gbm = joblib.load('website/backend/models/gbm_main.joblib')
nn = tf.keras.models.load_model('website/backend/models/neural_net_main.keras')
df = pu.load_and_prep(); X = df.drop(columns=[pu.Y_COL]).head(3)
Xt = pre.transform(X)
print('gbm proba:', gbm.predict_proba(Xt)[:,1])
print('nn  proba:', nn.predict(Xt, verbose=0).flatten())
"
```
Expected: prints 3 probabilities each in [0,1], no errors.

- [ ] **Step 4: Commit**

```bash
git add preparation/saving_selected_models_for_pipeline.py website/backend/models/
git commit -m "feat: deploy CV-selected models from best_configs; NN saved as .keras"
```

---

## Task 16: Update backend to use the saved preprocessor + Keras NN

**Files:**
- Modify: `website/backend/main.py`
- Modify: `website/requirements.txt`

**Interfaces:**
- Consumes: `preprocess_<variant>.joblib`, `neural_net_<variant>.keras`, `gbm_<variant>.joblib`, `svm_<variant>.joblib`.

- [ ] **Step 1: Align backend dependency pins with the training env**

In `website/requirements.txt` set `scikit-learn==1.4.0` and `numpy==1.26.4` (must match the versions that pickled the models) and pin `tensorflow==2.16.2` (TF 2.16 bundles Keras 3 — do NOT add a separate `keras` pin). Leave the FastAPI/GEE pins unchanged.

- [ ] **Step 2: Point model paths at the new artifact names**

In `main.py`, update `MODEL_PATH_CANDIDATES` so `neural_net` entries use `neural_net_<variant>.keras` and `gbm`/`svm` use the `.joblib` names (e.g., `gbm_main.joblib`). Add a `PREPROCESS_PATHS` dict mapping each variant to `models/preprocess_<variant>.joblib`.

- [ ] **Step 3: Load the NN via Keras and route preprocessing through the saved pipeline**

In `load_model`, when `model_type == "neural_net"`, load with `keras.models.load_model(model_path)` instead of `joblib.load`. In the preprocessing function (the `prep_dataframe`-style function around lines 546–609), replace the manual `fillna`/`replace(±99999)`/`scaler.transform`/`kmeans predict`/`ADD_CLUSTERS` block with:
```python
# One saved preprocessing pipeline reproduces training-time transforms
# (median impute -> standardize -> KMeans cluster feature), fit on training data.
preprocess = joblib.load(PREPROCESS_PATHS[model_variant])
X = preprocess.transform(df.reindex(columns=preprocess.feature_names_in_))
```
Delete the now-dead `kmeans_fitter.joblib` / `scaled_kmeans_fitter.joblib` loads and the `ADD_CLUSTERS` branch. In the prediction path, keep the `neural_net` special-case that calls `model.predict(...)` and the sklearn `predict_proba` path; both now receive the already-transformed `X`, so drop the separate `feature_names_in_` reindex-from-gbm hack.

- [ ] **Step 4: Smoke-test the backend end to end**

Run (start server, hit health/predict, stop):
```bash
cd website/backend && source ../../.venv311/bin/activate
pip install -r ../requirements.txt
uvicorn main:app --port 8123 &
SERVER=$!; sleep 8
curl -s -X POST "http://127.0.0.1:8123/api/predict" \
  -F "model_type=gbm" -F "latitude=42.44" -F "longitude=-76.5" | head -c 400
echo; curl -s -X POST "http://127.0.0.1:8123/api/predict" \
  -F "model_type=neural_net" -F "latitude=42.44" -F "longitude=-76.5" | head -c 400
kill $SERVER
```
Expected: both return JSON with a probability/risk field and HTTP 200 (no 500). If the endpoint requires different form fields, mirror the fields the frontend sends (inspect `website/frontend`), but the key assertion is: no load/predict error for either model type.

- [ ] **Step 5: Remove obsolete model artifacts**

The old format is replaced (`gbm/svm` → `.joblib`, `neural_net` → `.keras`, plus `preprocess_*.joblib`), so the superseded files must go — otherwise stale `.pkl`/scaler/kmeans artifacts linger and confuse the loader:
```bash
cd /Users/yeonjinjung/Documents/pathogen_prediction
git rm website/backend/models/gbm_*.pkl website/backend/models/svm_*.pkl \
       website/backend/models/neural_net_*.pkl \
       website/backend/models/scaler_file_*.joblib \
       website/backend/models/kmeans_fitter.joblib \
       website/backend/models/scaled_kmeans_fitter.joblib
ls website/backend/models/
```
Expected: only the new `*_main/soil_only/longlat_only.joblib`, `neural_net_*.keras`, and `preprocess_*.joblib` files remain.

- [ ] **Step 6: Commit**

```bash
git add website/backend/main.py website/requirements.txt website/backend/models/
git commit -m "feat: backend loads Keras NN + saved preprocess pipeline; remove obsolete artifacts"
```

---

## Task 17: Final verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `cd /Users/yeonjinjung/Documents/pathogen_prediction && source .venv/bin/activate && python -m pytest preparation/tests/ -v`
Expected: all pass.

- [ ] **Step 2: Confirm both notebooks execute clean from a fresh kernel**

Run:
```bash
for nb in Run_and_Test_Models Analyze_Models; do
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=pathogen-venv \
    preparation/$nb.ipynb && echo "$nb OK"
done
```
Expected: both print OK, exit 0.

- [ ] **Step 3: Confirm report == deployment**

Run:
```bash
python -c "
import json, pandas as pd
cfg=json.load(open('preparation/data_results/best_configs.json'))
print('best_configs models:', sorted(cfg))
print('deployed files:'); import os; print([f for f in os.listdir('website/backend/models') if 'main' in f])
"
```
Expected: `best_configs.json` and the deployed `*_main.*` artifacts cover gbm/svm/neural_net.

- [ ] **Step 4: Confirm no leakage-era numbers or attribution remain**

Run:
```bash
grep -rn "94.2\|94.16" ReadMe.md preparation/ 2>/dev/null || echo "no stale metrics"
git log --oneline -20 | grep -i "claude\|co-authored" || echo "no attribution in commits"
```
Expected: `no stale metrics` and `no attribution in commits`.

---

## Self-Review (author checklist — completed during planning)

**1. Spec coverage:**
- §1 refactor / notebook interdependency → Tasks 2–11 (module), 12–13 (thin drivers). ✓
- §2 decisions (GridSearchCV+holdout, cluster pipeline, accuracy selection) → Tasks 4, 5, 7, 8, 11. ✓
- §3 validation (stratified CV on train, single holdout, seeding, DT-vs-RF) → Tasks 4, 7, 8, 10, 11. ✓
- §4 leakage/hygiene (drop log-of-index, in-fold impute/scale/KMeans, stratify) → Tasks 3, 5, 6. ✓
- §5 metrics & calibration (accuracy + recall/PR-AUC/etc., calibration curve, prevalence caveat) → Tasks 9, 13. ✓
- §6 shared module + plumbing (sklearn API fix, path bug, re-run, regenerate, pin deps) → Tasks 1, 3, 7, 12–14. ✓
- §7 deployment alignment (best_configs source of truth, Keras NN, reconcile clusters) → Tasks 15–16. ✓
- §8/§9 out of scope / success criteria → verification in Task 17. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows the code. (The one `# TODO` visible in the existing `main.py` is pre-existing code, not introduced here.)

**3. Type consistency:** `make_preprocessor`, `ClusterFeatureAdder`, `predict_proba_any`, `evaluate`, `select_and_evaluate`, `load_best_configs`, `_clf_params` names are used identically across Tasks 5–16. `best_configs.json` param key convention (`clf__*` for sklearn, plain for NN) is defined in Task 11 and consumed consistently in Task 15.
