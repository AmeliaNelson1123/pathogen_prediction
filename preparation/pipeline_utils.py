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
