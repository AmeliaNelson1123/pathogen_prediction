import io
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import ee
import numpy as np

# -------------------------------------
# --- Global variables  and Set Up ----
# -------------------------------------

# starting an call to start web interface
app = FastAPI()

# connects frontend and backend (even though from different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"], # allows all actions (get, post, etc) methods
    allow_headers=["*"], # allows for stuff like content-type and authorization to transfer over
)

# creating base variabels
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "farm-app" / "dist"
# creating options for model files by input variant and model family
# each entry is a list of fallback candidates, first-existing path is used.
MODEL_PATH_CANDIDATES: dict[str, dict[str, list[Path]]] = {
    "soil_longlat": {
        "gbm": BASE_DIR / "models" / "gbm_main.pkl",
        "neural_net": BASE_DIR / "models" / "neural_net_main.pkl",
        "svm": BASE_DIR / "models" / "svm_main.pkl",
    },
    "longlat_only": {
        "gbm": BASE_DIR / "models" / "gbm_longlat_only.pkl",
        "neural_net": BASE_DIR / "models" / "neural_net_longlat_only.pkl",
        "svm": BASE_DIR / "models" / "svm_longlat_only.pkl",
    },
    "soil_only": {
        "gbm": BASE_DIR / "models" / "gbm_soil_only.pkl",
        "neural_net": BASE_DIR / "models" / "neural_net_soil_only.pkl",
        "svm": BASE_DIR / "models" / "svm_soil_only.pkl",
    },
}

# scaler paths if wanted for the models
SCALER_PATH_CANDIDATES: dict[str, dict[str, list[Path]]] = {
    "soil_longlat": BASE_DIR / "models" / "scaler_file_main.joblib",
    "longlat_only": BASE_DIR / "models" / "scaler_file_longlat_only.joblib",
    "soil_only": BASE_DIR / "models" / "scaler_file_soil_only.joblib",
}

# creating a dictonary to store the loaded models
MODEL_CACHE: dict[str, Any] = {}
EE_INITIALIZED = False

# Soil and Long/lat variables to group
SOIL_VARS = ['pH', 'Copper (mg/Kg)', 'Molybdenum (mg/Kg)', 'Sulfur (mg/Kg)', 'Moisture',
       'Manganese (mg/Kg)', 'Aluminum (mg/Kg)', 'Potassium (mg/Kg)',
       'Total nitrogen (%)', 'Zinc (mg/Kg)', 'Organic matter (%)', 'Phosphorus (mg/Kg)',
       'Iron (mg/Kg)', 'Magnesium (mg/Kg)', 'Sodium (mg/Kg)', 'Calcium (mg/Kg)', 'Total carbon (%)']
LOG_SOIL_VARS = ['Sulfur (mg/Kg)', 'Moisture',
       'Manganese (mg/Kg)', 'Aluminum (mg/Kg)', 'Potassium (mg/Kg)',
       'Total nitrogen (%)', 'Organic matter (%)', 'Phosphorus (mg/Kg)',
       'Iron (mg/Kg)', 'Magnesium (mg/Kg)', 'Sodium (mg/Kg)', 'Calcium (mg/Kg)', 'Total carbon (%)']
DOUBLE_LOG_SOIL_VARS = ['Zinc (mg/Kg)']

LONGLAT_VARS = ['Latitude', 'Longitude', 
       'Precipitation (mm)', 'Max temperature (℃ )', 'Min temperature (℃ )',
       'Wind speed (m/s)', 'Barren (%)', 'Forest (%)', 'Pasture (%)', 'Grassland (%)', 'Shrubland (%)',
       'Open water (%)', 'Developed open space (> 20% Impervious Cover) (%)', 'Elevation (m)',
        'Cropland (%)', 'Wetland (%)', "Developed open space (< 20% Impervious Cover) (%)"]
LOG_LONGLAT_VARS = ['Grassland (%)', 'Shrubland (%)', 'Open water (%)',
        'Developed open space (> 20% Impervious Cover) (%)', 'Elevation (m)',
        'Cropland (%)', 'Wetland (%)', "Developed open space (< 20% Impervious Cover) (%)"]
CLUSTER_VARS = ['cluster_kmeans', 'scaled_cluster_kmeans']

# Optional management-factor effects are applied as odds multipliers.
IRRIGATION_MULTIPLIER: dict[str, float] = {
    "none": 1.0, # NOT PICK CATOROGY 
    "144_rain_window": 1.0, # more than 144 hours since last rained/irrigated
    "72_rain_window": 2.5, # 72 hours since last rained/irrigated
    "48_rain_window": 2.1, # 48 hours since last rained/irrigated
    "24_rain_window": 7.7, # 24 hours since last rained/irrigated
}

WILDLIFE_MULTIPLIER: dict[str, float] = {
    "none": 1.0, # no seelcted
    "no_risk_wildlife": 1.0, # wildlife never seen in field
    "low_risk_wildlife": 1.0, # wildlife seen 8-30 days in field
    "moderate_risk_wildlife": 0.8, # wildlife seen in field witi
    "high_risk_wildlife": 4.4, # wildlife seen in field within last 3 days
}

MANURE_MULTIPLIER: dict[str, float] = {
    "none": 1.0, # not applied and the no selected varaible
    "no_manure": 1.0, # manure never spread on field
    "manure_over_365_days": 0.6, # over 365 days since spread
    "manure_within_365_days": 7.0, # manure spread on field within 365 days of harvest
}

BUFFER_ZONE_MULTIPLIER: dict[str, float] = {
    "none": 1.0, # no selected
    "no_buffer_zone": 1.0, # field does not have a buffer zone and 
    "buffer_zone": 0.5, # field has a buffer zone
}

RISK_THRESHOLDS = {
    "high": 0.85, # find source to make it not arbitrary
    "moderate": 0.65,  # find source to make it not arbitrary
    "low": 0.59,  # find source to make it not arbitrary
}

# data prep variables in case model pipeline changes
ADD_CLUSTERS = False # if the model requires standardized and non-standardized clusters
ENCODE_STR = False # if the model requires the one-hot encoding of columns of strings/catagories
USE_SCALER = True # if want to run models that were trained on scaled data

# ------------------------------
# --- Helper Functions ---------
# ------------------------------

# the bounding box for the US (used to limit prediction space to US-only data)
US_BBOX = {
    "lat_min": 24.5,
    "lat_max": 49.5,
    "lon_min": -125.0,
    "lon_max": -67.0,
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast" # for likely the same day, and future days weather data (also has elevation)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive" # for historical weather data (and elevation)


def to_one_item_list(value: Any, column_name: str) -> list[Any]:
    """Normalize scalar-like and nested singleton values into a one-item list."""
    cur = value

    # Unwrap nested singleton containers until we hit a scalar-like value.
    while True:
        if isinstance(cur, np.ndarray):
            arr = np.asarray(cur)
            if arr.size != 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{column_name}' must be scalar/size-1, got ndarray "
                        f"shape={arr.shape}, dtype={arr.dtype}."
                    ),
                )
            cur = arr.reshape(-1)[0]
            continue

        if isinstance(cur, (list, tuple)):
            if len(cur) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{column_name}' must contain one value, got "
                        f"{type(cur).__name__} length={len(cur)}."
                    ),
                )
            cur = cur[0]
            continue

        # Convert numpy scalar types (np.float64, np.int64, etc.) to Python scalars.
        if isinstance(cur, np.generic):
            cur = cur.item()
            continue

        break

    return [cur]

# ------------------------------
# --- Earth Engine Config ------
# ------------------------------

# Environment-first configuration for production deploys.
# Fallbacks keep local runs simple (no terminal input needed).
EE_PROJECT = os.getenv("EE_PROJECT", "listeria-prediction-tool")
EE_SERVICE_ACCOUNT = os.getenv(
    "EE_SERVICE_ACCOUNT",
    "temp-for-iafp-competition@listeria-prediction-tool.iam.gserviceaccount.com",
)
EE_PRIVATE_KEY_PATH = Path(
    os.getenv("EE_PRIVATE_KEY_PATH", str(BASE_DIR / "gee-service-account-key.json"))
)
EE_PRIVATE_KEY_JSON = os.getenv("EE_PRIVATE_KEY_JSON")

# checking if the geo points provided are within the US
# (this is where the match is made for land cover)
def in_us_bbox(lat: float, lon: float) -> bool:
    return (
        US_BBOX["lat_min"] <= lat <= US_BBOX["lat_max"]
        and US_BBOX["lon_min"] <= lon <= US_BBOX["lon_max"]
    )

# initializeing the google earth engine. Only doing this if the longitude and latitude are recieved to save time
def initialize_earth_engine() -> None:
    """Initializing the Earth Engine. Only once per process!!!!!"""
    # updating/creating a global variable to reduce times the process is initialized
    global EE_INITIALIZED
    if EE_INITIALIZED: # checking if initialized
        return

    # grabbing account stuff to be able to access google earth engine (ee)
    service_account = EE_SERVICE_ACCOUNT
    private_key_path = EE_PRIVATE_KEY_PATH
    ee_project = EE_PROJECT

    # initializing the google earth engine based on a service account. Lots of user feedback just in case
    try:
        # Preferred in production: inject full JSON via secret env var.
        if service_account and EE_PRIVATE_KEY_JSON:
            credentials = ee.ServiceAccountCredentials(
                service_account,
                key_data=EE_PRIVATE_KEY_JSON,
            )
            ee.Initialize(credentials, project=ee_project)
        elif service_account and private_key_path:
            key_path = Path(private_key_path)
            if not key_path.exists():
                raise FileNotFoundError(
                    f"Earth Engine key file was not found: {key_path}"
                )
            
            credentials = ee.ServiceAccountCredentials(service_account, str(key_path))
            ee.Initialize(credentials, project=ee_project)
        else:
            # Fallback if prefer local EE auth on this machine.
            ee.Initialize(project=ee_project)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Google Earth Engine initialization failed. "
                "Set EE_PROJECT / EE_SERVICE_ACCOUNT and either EE_PRIVATE_KEY_JSON or EE_PRIVATE_KEY_PATH. "
                "or use local EE auth fallback. If you are recieving this message, please notify the github owner"
                f"Underlying error: {exc}"
            ),
        ) from exc

    EE_INITIALIZED = True


# ------------------------------
# ---  Earth Engine - NLCD -----
# ------------------------------

# the link to the NLCD image referenced using google earth engine
NLCD_IMAGE_ID = "USGS/NLCD_RELEASES/2021_REL/NLCD/2021"

# mapping of requested output categories to NLCD class codes.
# basically, there can be multiple codes from NLCD that map to the data in our training set, so we are combining them when needed. 
# Keeping them all in a tuple format helps with debugging and keeps us from needing 2 different processed depending on the type (tuple vs non-tuple) 
NLCD_CATEGORY_CODES: dict[str, tuple[int, ...]] = {
    "Open water (%)": (11,),
    "Developed open space (< 20% Impervious Cover) (%)": (21,),
    "Developed open space (> 20% Impervious Cover) (%)": (22, 23, 24),
    "Barren (%)": (31,),
    "Forest (%)": (41, 42, 43),
    "Shrubland (%)": (51, 52),
    "Grassland (%)": (71,),
    "Cropland (%)": (82,),
    "Pasture (%)": (81,),
    "Wetland (%)": (90, 95),
}

# this is how we are getting the land cover percentages through NLCD (accessed via earth engine)
def get_nlcd_percentages(lat: float, lon: float, buffer_m: int = 1000) -> dict[str, float]:
    """return grouped NLCD percentages for the requested classes."""
    initialize_earth_engine() # initializing if not already done

    # setup
    nlcd = ee.Image(NLCD_IMAGE_ID)
    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(buffer_m)

    # getting results from api at scale 30
    stats = nlcd.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=buffer,
        scale=30,
        maxPixels=1e9,
    )

    # getting the landcover variables
    try:
        histogram = stats.get("landcover").getInfo()
    except Exception as exc:
        # user feedback if needed
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch NLCD landcover histogram from Earth Engine: {exc}",
        ) from exc

    # Always return all categories, even when data is missing!!!!!
    grouped_percentages: dict[str, float] = {key: 0.0 for key in NLCD_CATEGORY_CODES}
    if not histogram: # skipping if none are available
        return grouped_percentages

    # getting the counts!
    total_pixels = sum(histogram.values())
    if total_pixels <= 0:
        return grouped_percentages

    # getting the class percentages 
    class_percentages: dict[int, float] = {
        int(class_code): (pixel_count / total_pixels) * 100.0
        for class_code, pixel_count in histogram.items()
    }

    #doing secondary class sums to match how the data was outputed originally
    for category, class_codes in NLCD_CATEGORY_CODES.items():
        grouped_percentages[category] = round(
            sum(class_percentages.get(code, 0.0) for code in class_codes),
            6,
        )
    return grouped_percentages


# ------------------------------
# --- Weather forecasting ------
# ------------------------------

# grabbing and parsing the format
def parse_forecast_date_mmddyyyy(raw_date: str) -> date:
    """Parsing the  MM/DD/YYYY date string."""
    try:
        # parsing date
        parsed = datetime.strptime(raw_date, "%m/%d/%Y").date()
    except ValueError as exc:
        # more user feedback
        raise HTTPException(
            status_code=400,
            detail="Date must be in MM/DD/YYYY format. For example, Febuary 21, 2026 is inputted as 02/21/2026",
        ) from exc
    min_allowed_date = date(2010, 1, 1)
    if parsed < min_allowed_date:
        raise HTTPException(
            status_code=400,
            detail="Date cannot be earlier than 01/01/2010.",
        )
    return parsed

# grabbing api data for the daily weather data
def get_open_meteo_daily(lat: float, lon: float, target_date: date) -> dict[str, float]:
    """
    Getting Open-Meteo API weather data for the target_date

    !!! forecasts are retrieved for today/future and archive for historical dates.
    """

    # the utc datetime object 
    today_utc = datetime.now(timezone.utc).date()

    # quick check for a date within the bounds of what Open-Meteo can do
    max_forecast_date = today_utc + timedelta(days=14)
    if target_date > max_forecast_date:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unfortunately, dates can only be up to 14 days after the current one. Select a date on or before {max_forecast_date.isoformat()} (UTC)."
            ),
        )

    # quickly checking if we should use a historical or forcasting query
    weather_base_url = OPEN_METEO_ARCHIVE_URL if target_date < today_utc else OPEN_METEO_URL
    query = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            # only grabbing the temp, precip, and wind because that is what was trained on
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "timezone": "UTC",
            "wind_speed_unit": "ms", # grabbing the correct unit
        }
    )
    request_url = f"{weather_base_url}?{query}"

    # connecting to open meteo
    try:
        # timeout set to 15 just in case
        with urlopen(request_url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    # handling the weird errors in more interpretable ways for the user!!!
    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The Open-Meteo request failed with HTTP: {exc.code}. Try refreshing the page and try again.",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The Open-Meteo request failed: {exc.reason}. It seems like there was a proble with the weather API. Try again in a few minutes, or select the model to run with soil only",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The Open-Meteo parsing failed: {exc}. USER SHOULD NOT SEE THIS ERROR",
        ) from exc

    # grabbing daily data
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    # quick sanity check! just in case open-meteo had a missing record. 
    if not dates:
        raise HTTPException(
            status_code=502,
            detail="Weather API (Open-Meteo) did not have any data for the selected date and long/lat. Please choose again.",
        )

    # grabbing the relative index (which should be the anticipated 0 for one date)
    idx = dates.index(target_date.isoformat())
    try:
        # getting all the variables needed for weather and renaiming them to match model variables
        weather = {
            "Max temperature (℃ )": float(daily["temperature_2m_max"][idx]),
            "Min temperature (℃ )": float(daily["temperature_2m_min"][idx]),
            "Precipitation (mm)": float(daily["precipitation_sum"][idx]),
            "Wind speed (m/s)": float(daily["wind_speed_10m_max"][idx]),
            "Elevation (m)": float(payload["elevation"]),
        }
    # a just in case, if everything fails, type of adventure
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Open-Meteo response missing expected weather fields: {exc}",
        ) from exc

    return weather


# -----------------------------------------
# --- Data Preparation for Modeling -------
# -----------------------------------------

# dropping all non-modeling columns
def log_transform_vars(df: pd.DataFrame, model_variant: str) -> pd.DataFrame:
    df = df.copy()
    # doing log transformations of all models if needed
    # comment out if choose a model that does not have log transformations
    with np.errstate(divide="ignore", invalid="ignore"):
        if model_variant == "soil_only":
            for col in LOG_SOIL_VARS:
                df[f"log of {col}"] = np.log1p(df[col])

            for col in DOUBLE_LOG_SOIL_VARS:
                df[f"double log of {col}"] = np.log1p(np.log(df[col]))

            df = df.drop(columns=LOG_SOIL_VARS)
            df = df.drop(columns=DOUBLE_LOG_SOIL_VARS)
        if model_variant == "longlat_only":
            for col in LOG_LONGLAT_VARS:
                df[f"log of {col}"] = np.log1p(df[col])

            df = df.drop(columns=LOG_LONGLAT_VARS)
        if model_variant == "soil_longlat":
            for col in LOG_LONGLAT_VARS:
                df[f"log of {col}"] = np.log1p(df[col])
            for col in LOG_SOIL_VARS:
                df[f"log of {col}"] = np.log1p(df[col])
            for col in DOUBLE_LOG_SOIL_VARS:
                df[f"double log of {col}"] = np.log1p(np.log1p(df[col]))

            df = df.drop(columns=LOG_SOIL_VARS)
            df = df.drop(columns=LOG_LONGLAT_VARS)
            df = df.drop(columns=DOUBLE_LOG_SOIL_VARS)
    return df

# dropping all non-modeling columns
def keep_only_allowed_columns(df: pd.DataFrame, model_variant: str) -> pd.DataFrame:
    # quickly mapping all of the allowed columns and using sets for faster processing
    allowed_by_variant = {
        "soil_only": set(SOIL_VARS),
        "longlat_only": set(LONGLAT_VARS),
        "soil_longlat": set(SOIL_VARS) | set(LONGLAT_VARS),# want both soil and longlat vars
    }
    # checking the variant vs what can and cant drop
    allowed = allowed_by_variant[model_variant]
    keep = [c for c in df.columns if c in allowed] # columns to keep
    dropped = [c for c in df.columns if c not in allowed] # columns to drop (for sanity checks if need to print)
    return df.loc[:, keep].copy()

# processing all data
def data_prep(df, model_variant):
    """
    ----- inputs -----
    df: pandas dataframe
        dataframe wanting to process
    ----- outputs ----
    df: pandas df
        processed anonymzied data (string columns representing intervals split into min and max, then put as minimum and maximum values for those columns)
    """

    # Fail early with clear missing-column feedback before transforms.
    if model_variant == "soil_only":
        required_cols = set(LOG_SOIL_VARS + DOUBLE_LOG_SOIL_VARS)
    elif model_variant == "longlat_only":
        required_cols = set(LOG_LONGLAT_VARS)
    elif model_variant == "soil_longlat":
        required_cols = set(LOG_SOIL_VARS + DOUBLE_LOG_SOIL_VARS + LOG_LONGLAT_VARS)
    else:
        required_cols = set()
    # checking to make sure all the required columns are here
    missing_cols = sorted(col for col in required_cols if col not in df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                "Input data is missing required columns for preprocessing: "
                + ", ".join(missing_cols)
            ),
        )
    
    # handling non-modeling variables
    df = keep_only_allowed_columns(df=df, model_variant=model_variant)
    # performing log transform where needed
    df = log_transform_vars(df=df, model_variant=model_variant)
    
    # switching missing values and weird failures in writing to np.inf bc pandas didnt handle properly
    df = df.replace("#NAME?", -np.inf)
    df = df.fillna(-np.inf)

    # replacing inf with max number that is not max number + 100 in dict (FOR NOT JUST 99999999)
    df = df.replace(np.inf, 99999)
    # replacing -inf with min number (not -inf) - 100 in dict (FOR NOT JUST -99999999)
    df = df.replace(-np.inf, -99999)

    df = df.dropna(axis=1, how="all")

    # Drop 'index' column if it exists, as it's typically an artifact and not a feature
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    if 'log of index' in df.columns:
        df = df.drop(columns=['log of index'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # grabbing the scaler to use if necessray
    scaler = joblib.load(SCALER_PATH_CANDIDATES[model_variant])
    # if the model is trained on scaled data, this should be true
    if USE_SCALER:
        # Reindex to scaler features and fill any absent columns with a safe default.
        df_copy = df.reindex(columns=scaler.feature_names_in_, fill_value=0.0)
        df = scaler.transform(df_copy) # only for scaled kmeans
        df = pd.DataFrame(df, columns=df_copy.columns, index=df_copy.index)

    # dealing with kmeans clusters used in the full model, and also the scaler value
    if model_variant == "soil_longlat":
        kmeans_raw = joblib.load(BASE_DIR / "models" / "kmeans_fitter.joblib")
        kmeans_scaled = joblib.load(BASE_DIR / "models" / "scaled_kmeans_fitter.joblib")

        if ADD_CLUSTERS:
            # quickly dropping columns not trained on for clusters
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            if 'log of index' in df.columns:
                df = df.drop(columns=['log of index'])
            # dropping other necessary cols for clusters
            df_for_kmeans = df.drop(columns=['log of Open water (%)', 'log of Phosphorus (mg/Kg)', 'log of Wetland (%)', 'double log of Zinc (mg/Kg)', 'log of Aluminum (mg/Kg)', 'log of Cropland (%)', 'log of Developed open space (< 20% Impervious Cover) (%)', 'log of Developed open space (> 20% Impervious Cover) (%)'])
            df_for_kmeans_scaled = df.drop(columns=['log of Open water (%)', 'log of Phosphorus (mg/Kg)', 'log of Wetland (%)', 'double log of Zinc (mg/Kg)', 'log of Aluminum (mg/Kg)', 'log of Cropland (%)', 'log of Developed open space (< 20% Impervious Cover) (%)', 'log of Developed open space (> 20% Impervious Cover) (%)'])
            
            # grabbing scaled and non-scaled cluster values
            df["cluster_kmeans"] = kmeans_raw.predict(df_for_kmeans)
            df["scaled_cluster_kmeans"] = kmeans_scaled.predict(df_for_kmeans_scaled)

    # quickly dropping columns not trained on for clusters
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'log of index' in df.columns:
        df = df.drop(columns=['log of index'])

    # quickly encoding strings using one-hot if required by model
    if ENCODE_STR:
        df = pd.get_dummies(df)

    # returning :)
    return df


# -----------------------------------------
# --- Loading and Checking Models ---------
# -----------------------------------------

# loading in the model to run
def load_model(model_variant: str, model_type: str) -> Any:
    
    # just a quick check for some unknown model, basically a fall-back
    if (
        model_variant not in MODEL_PATH_CANDIDATES
        or model_type not in MODEL_PATH_CANDIDATES[model_variant]
    ):
        raise HTTPException(status_code=400, detail="Need to select a model. Invalid model_type.")

    # grabbing the right model option path
    cache_key = f"{model_variant}:{model_type}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    # getting the first available model file path from candidate list
    model_path = MODEL_PATH_CANDIDATES[model_variant][model_type]

    # Only require keras/tensorflow when a neural net model is actually requested.
    if model_type == "neural_net":
        try:
            import keras  # noqa: F401
        except ModuleNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Neural net model requested, but keras/tensorflow is not installed in backend env. "
                    "Install with: pip install tensorflow"
                ),
            )

    # loading the model to run!
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not load model {model_path.name}: {exc}" # user feedback
        ) from exc

    # returning the model!
    MODEL_CACHE[cache_key] = model
    return model


# -----------------------------------------
# -- Handeling results and probality ------
# -----------------------------------------

# this is where we adjust the odds from the probability
# TODO: MAKE SURE THIS IS CONVERTED TO A PROBABILITY MULTIPLIER
def apply_odds_multiplier(p: np.ndarray, multiplier: float) -> np.ndarray:
    """
    multiplicative effect on odds instead of directly adding to probability.
    
    This keeps adjusted probabilities bounded in [0, 1]!
    """
    # making sure no 0s
    clipped = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    # getting odds ratio with safe value
    odds = clipped / (1.0 - clipped)

    # multiplying the odds (adjusting the odds because adjusting probabilities close to 0 and close to 1 will cause problems)
    adjusted_odds = odds * multiplier

    # returning the probability!
    return adjusted_odds / (1.0 + adjusted_odds)

# place to make sure actually grabbing the right index... otherwise buggy, and sometimes pulls the wrong index
def get_binary_class_indices(model: Any, proba: np.ndarray, model_type: str) -> tuple[int, int]:
    """
    Return (absence_idx, presence_idx), where absence is class 0 and presence is class 1.

    Falls back safely to 0/1 when class labels are unavailable.
    """

    # Neural-net path: assume binary probabilities are ordered [class_0, class_1].
    if model_type == "neural_net":
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(
                f"Neural net probability output must have shape (n,2+) but got {proba.shape}"
            )
        return 0, 1

    # Non-neural path: read labels from sklearn model metadata.
    classes = model.classes_
    
    # grabbing the corresponding index for present or absent. 
    # there is a sanity check to make sure the label was indeed 0 or 1
    class_labels = [str(c).strip() for c in classes]
    if "0" in class_labels and "1" in class_labels:
        return class_labels.index("0"), class_labels.index("1")

    # raising error if this is wrong
    raise ValueError(f"Model was not correctly provided. Model type: {model}, and probability was {proba}")

# Grabbing the risk class associated. using a global variable and function to reduce places for future users to look.
def risk_class_from_probability(p: float) -> str:
    if p >= RISK_THRESHOLDS["high"]:
        return "High Risk"
    if p >= RISK_THRESHOLDS["moderate"]:
        return "Moderate Risk"
    if p >= RISK_THRESHOLDS["low"]:
        return "Low Risk"
    return "Very Low Risk"


# -------------------------------
# --- Handleing Actions ---------
# -------------------------------

# quick check endpoint so we can test Google Earth Engine auth independently
@app.get("/health/earth-engine")
@app.get("/api/health/earth-engine")
def earth_engine_health() -> dict[str, Any]:
    initialize_earth_engine()
    return {"success": True, "earth_engine_initialized": True}


# Now, we are running the model
@app.post("/predict")
@app.post("/api/predict")
async def predict(
    file: UploadFile | None = File(default=None),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None),
    forecast_date: str | None = Form(default=None),
    model_type: str = Form(...),
    longlat_mode: str = Form(default="longlat_only"),
    irrigation_mode: str = Form(default="none"),
    wildlife_mode: str = Form(default="none"),
    manure_mode: str = Form(default="none"),
    buffer_zone_mode: str = Form(default="none"),
) -> dict[str, Any]:
    model_type = model_type
    nlcd_percentages: dict[str, float] | None = None
    weather_data: dict[str, float] | None = None
    gis_loaded = False
    gis_fetch_ms: int | None = None
    forecast_date_utc: str | None = None

    # message list to display to user for user feedback
    add_message = []

    # feedback for the user if they did not input anything (csv file and long/lat)
    if file is None and (lat is None or lon is None):
        raise HTTPException(
            status_code=400, detail="Provide either a CSV file or coordinates+date."
        )

    # feedback for user if they did not input the model type they want to run
    if model_type not in ("gbm", "neural_net", "svm"):
        raise HTTPException(
            status_code=400, detail="the model type must be one of: 'gbm', 'neural_net', or 'svm'."
        )

    # more checks just in case something spooky goes on
    if irrigation_mode not in IRRIGATION_MULTIPLIER:
        raise HTTPException(
            status_code=400,
            detail="Invalid irrigation mode.",
        )

    # more checks just in case something spooky goes on
    if wildlife_mode not in WILDLIFE_MULTIPLIER:
        raise HTTPException(
            status_code=400,
            detail="Invalid wildlife mode.",
        )
    if manure_mode not in MANURE_MULTIPLIER:
        raise HTTPException(
            status_code=400,
            detail="Invalid manure mode.",
        )
    if buffer_zone_mode not in BUFFER_ZONE_MULTIPLIER:
        raise HTTPException(
            status_code=400,
            detail="Invalid buffer zone mode.",
        )

    # automatically working with if there is a csv inputed or not and running with and without soil
    if longlat_mode == "soil_only":
        # adding message to say input long for better model accuracy
        add_message.append("Add Longitude, Latitude, and Date information to improve model (and select Soil and Longitude and Latitude Model) and more accurately assess your risk of Listeria. (Coordinates and dates will automatically import the elevation and weather data which will improve the model's ability to access the risk of Listeria)")

        # attempting to read in the csv
        try:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents), index_col=0)
            X = data_prep(df, longlat_mode)
            # Drop 'index' column if it exists, as it's typically an artifact and not a feature
            if 'index' in df.columns:
                df = df.drop(columns=['index'])
            if 'log of index' in df.columns:
                df = df.drop(columns=['log of index'])
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"We could not correctly process your CSV. Likely, there is a problem with the column name. Please use column names that match the Sample CSVs: \nError with: {exc}"
            ) from exc
        # making sure the longitude and latitude is inputed 
        # (because cannot run on no data, so either enviro from long/lat or soil is required)
        model_variant = "soil_only"
    elif longlat_mode == "longlat_only":
        # adding message to say input long for better model accuracy
        add_message.append("Add Soil CSV information to improve model (and select the Soil and Longitude and Latitude Model), and more accurately assess your risk of Listeria.")

        # handeling api calls to get data with long and lat
        dict_df = {
            "Latitude": [lat],
            "Longitude": [lon],
        }
        # checking to make sure the long lat is only inside the USA
        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="The latitude and longitude provided are not completed or correct")
        if not in_us_bbox(lat, lon):
            raise HTTPException(status_code=400, detail=f"Coordinates must be inside US. (Must be within {US_BBOX})")

        # checking again to make sure the date is not none
        if forecast_date is None:
            raise HTTPException(
                status_code=400,
                detail="Date must be in MM/DD/YYYY format. For example, Febuary 21, 2026 is inputted as 02/21/2026",
            )
        forecast_dt = parse_forecast_date_mmddyyyy(forecast_date) # parsing date
        forecast_date_utc = forecast_dt.isoformat() # reformating it

        # quick checks for length of time it took to grab API calls
        gis_start = perf_counter()
        nlcd_percentages = get_nlcd_percentages(lat=lat, lon=lon, buffer_m=1000)
        weather_data = get_open_meteo_daily(lat=lat, lon=lon, target_date=forecast_dt)
        # if the input dict's items are not in list format, we need to convert them so that pandas can read it
        weather_data = {k: to_one_item_list(v, k) for k, v in weather_data.items()}
        nlcd_percentages = {k: to_one_item_list(v, k) for k, v in nlcd_percentages.items()}

        # updating the dictionary to have the matching model keys 
        dict_df.update(nlcd_percentages)
        dict_df.update(weather_data)

        try:
            df = pd.DataFrame(dict_df)
        except Exception as exc:
            col_types = {
                k: {
                    "outer": type(v).__name__,
                    "len": (len(v) if hasattr(v, "__len__") else None),
                    "inner": (type(v[0]).__name__ if isinstance(v, list) and v else None),
                }
                for k, v in dict_df.items()
            }
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not build dataframe from coordinate/weather input: "
                    f"{exc}. Column type summary: {col_types}"
                ),
            ) from exc
        X = data_prep(df, longlat_mode)
        
        gis_fetch_ms = int((perf_counter() - gis_start) * 1000)
        gis_loaded = True
        # Drop 'index' column if it exists, as it's typically an artifact and not a feature
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        if 'log of index' in df.columns:
            df = df.drop(columns=['log of index'])
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        model_variant = "longlat_only"
    elif longlat_mode == "soil_longlat":
        # handeling api calls to get data with long and lat
        dict_df = {
            "Latitude": [lat],
            "Longitude": [lon],
        }
        # checking to make sure the long lat is only inside the USA
        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="The latitude and longitude provided are not completed or correct")
        if not in_us_bbox(lat, lon):
            raise HTTPException(status_code=400, detail=f"Coordinates must be inside US. (Must be within {US_BBOX})")

        if forecast_date is None:
            raise HTTPException(
                status_code=400,
                detail="Coordinate entries need a date as well. Date must be in MM/DD/YYYY format. For example, Febuary 21, 2026 is inputted as 02/21/2026",
            )
        forecast_dt = parse_forecast_date_mmddyyyy(forecast_date)
        forecast_date_utc = forecast_dt.isoformat()

        gis_start = perf_counter()
        nlcd_percentages = get_nlcd_percentages(lat=lat, lon=lon, buffer_m=1000)
        weather_data = get_open_meteo_daily(lat=lat, lon=lon, target_date=forecast_dt)
        
        # if the input dict's items are not in list format, we need to convert them so that pandas can read it
        weather_data = {k: to_one_item_list(v, k) for k, v in weather_data.items()}
        nlcd_percentages = {k: to_one_item_list(v, k) for k, v in nlcd_percentages.items()}
        
        # updating the dictionary to have the matching model keys 
        dict_df.update(nlcd_percentages)
        dict_df.update(weather_data)
        dict_df = pd.DataFrame(dict_df)
        
        gis_fetch_ms = int((perf_counter() - gis_start) * 1000)
        gis_loaded = True

        # attempting to read in the csv
        try:
            contents = await file.read()
            pd_df = pd.read_csv(io.BytesIO(contents), index_col=0)
            # Uploaded CSVs may already contain API-derived env columns.
            # Drop them to avoid _x/_y suffix collisions during cross-merge.
            overlap_cols = [c for c in LONGLAT_VARS if c in pd_df.columns]
            overlap_log_cols = [f"log of {c}" for c in LOG_LONGLAT_VARS if f"log of {c}" in pd_df.columns]
            overlap_double_log_cols = [
                f"double log of {c}" for c in DOUBLE_LOG_SOIL_VARS
                if f"double log of {c}" in pd_df.columns
            ]
            cols_to_drop_before_merge = overlap_cols + overlap_log_cols + overlap_double_log_cols
            if cols_to_drop_before_merge:
                pd_df = pd_df.drop(columns=cols_to_drop_before_merge)

            # Broadcast one-row GIS values to every uploaded soil row.
            df = pd_df.merge(dict_df, how="cross")
            X = data_prep(df, longlat_mode)

        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"We could not correctly process your CSV. Likely, there is a problem with the column name. Please use column names that match the Sample CSVs: \nError with: {exc}"
            ) from exc
        
        model_variant = "soil_longlat"
    else:
        raise HTTPException(
            status_code=400,
            detail="You need to choose to include coordinates (and a date), or soil data, or both.",
        )

    model = load_model(model_variant, model_type)
    # quick drops as sanity check
    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])
    if 'log of index' in X.columns:
        X = X.drop(columns=['log of index'])
    if 'index' in X.columns:
        X = X.drop(columns=['index'])

    # neural net models dont have feature_names_in_ variable, so loading the gbm model which will have the same feature order
    if model_type == "neural_net":
        model_for_features_to_reindex = load_model(model_variant, 'gbm')
        features_to_reindex = model_for_features_to_reindex.feature_names_in_
        X = X.reindex(columns=features_to_reindex, fill_value=0.0)
        X = X.fillna(0.0)
    else: # svm and gbm models have feature_names_in_
        X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)
        X = X.fillna(0.0)

    # columns that contain at least one NaN
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Input still contains NaN after preprocessing in columns: {nan_cols}",
        )

    # running a prediction model!
    result = ""
    try:
        # neural net special case because no predict_proba
        if model_type == "neural_net":
            # Keras path
            proba = model.predict(X, verbose=0)
            if proba.ndim == 2 and proba.shape[1] == 1:
                prob_class_1 = proba[:, 0].astype(float)
                result = np.column_stack([1.0 - prob_class_1, prob_class_1])
            elif proba.ndim == 2 and proba.shape[1] >= 2:
                result = proba.astype(float)
            else:
                raise ValueError(f"Unexpected neural net output shape: {proba.shape}")

        else: # svm and gbm models
            result = model.predict_proba(X)
    except Exception as exc:
        # user feedback if the model fails
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    # class 0 = absence, class 1 = presence
    # using odds instead of probability so that we can accurately adjust the probability increase or decrease of listeria risk!
    # (otherwise, probabilities near 1 or 0 at the beginning will act weird)
    absence_idx, presence_idx = get_binary_class_indices(model=model, proba=result, model_type=model_type)
    base_presence_proba = result[:, presence_idx].astype(float)
    base_absence_proba = result[:, absence_idx].astype(float)

    # finding the addjustments for if the user inputed an irrigation and wildlife option
    irrigation_multiplier = IRRIGATION_MULTIPLIER[irrigation_mode]
    wildlife_multiplier = WILDLIFE_MULTIPLIER[wildlife_mode]
    manure_multiplier = MANURE_MULTIPLIER[manure_mode]
    buffer_zone_multiplier = BUFFER_ZONE_MULTIPLIER[buffer_zone_mode]
    combined_multiplier = (
        irrigation_multiplier
        * wildlife_multiplier
        * manure_multiplier
        * buffer_zone_multiplier
    )

    # adjusting the probabilities, yay!
    adjusted_presence_proba = apply_odds_multiplier(base_presence_proba, combined_multiplier)
    adjusted_absence_proba = 1.0 - adjusted_presence_proba

    # now, making an user-facing summary: classify by the highest adjusted risk row.
    displayed_result = float(np.max(adjusted_presence_proba))
    to_return_risk_class = risk_class_from_probability(displayed_result)
    # adding messages to provide feedback and hopefully helpful remarks
    if irrigation_mode == "24_rain_window":
        add_message.append("Irrigation/rain in the last 24 hours raises risk significantly. Consider additional verification, such as Listeria testing, before harvest. Make sure to clean equipment and consider waiting at least 144 hours before harvesting.")
    elif irrigation_mode in ("48_rain_window", "72_rain_window"):
        add_message.append("Recent irrigation/rain can increase risk. Consider additional verification, such as Listeria testing, before harvest. Make sure to clean equipment and consider waiting at least 144 hours before harvesting.")

    if wildlife_mode == "high_risk_wildlife":
        add_message.append("Active wildlife traffic increases contamination risk. It is recommend to create a buffer zone, exclude visibly affected zones, increase field scouting, and/or strengthen deterrents/barriers.")
    elif wildlife_mode == "moderate_risk_wildlife":
        add_message.append("Wildlife evidence suggests moderate risk. Intensify monitoring and create a buffer zone, and/or strengthen detterents/barriers.")
    
    if manure_mode == "manure_within_365_days":
        add_message.append("Manure application within 365 days of harvest increases risk. Ensure manure is correctly handled and processed before spreading.")
    elif manure_mode == "manure_over_365_days":
        add_message.append("Historic manure use may still influence risk. Continue routine monitoring and sanitation controls.")

    if buffer_zone_mode == "no_buffer_zone":
        add_message.append("No buffer zone can increase contamination risk from adjacent areas or wildlife.")
    
    if to_return_risk_class == "High Risk":
        add_message.append("High risk of Listeria presense in soil: targeted environmental testing is recommended, and impliment strategies to combat Listeria risk, such as sanitation or harvesting 144+ hours after rain/irrigation.")
    elif to_return_risk_class == "Moderate Risk":
        add_message.append("Moderate risk of Listeria presense in soil: increase monitoring frequency and tighten irrigation and harvest hygiene controls.")
    elif to_return_risk_class == "Low Risk":
        add_message.append("Low risk of Listeria presense in soil: continue routine controls and verification sampling.")
    else:
        add_message.append("Very low likelihood of finding Listeria in soil: No special measures are needed. Maintain current non-elevated controls and periodic verification.")
    
    # returning results if pressent!
    return {
        "success": True,
        "model_variant": model_variant,
        "model_type": model_type,
        "longlat_mode": longlat_mode,
        "irrigation_mode": irrigation_mode,
        "wildlife_mode": wildlife_mode,
        "manure_mode": manure_mode,
        "buffer_zone_mode": buffer_zone_mode,
        "forecast_date_utc": forecast_date_utc,
        "gis_loaded": gis_loaded,
        "gis_fetch_ms": gis_fetch_ms,
        "result": result.tolist(),
        "probability_absence_base": base_absence_proba.tolist(),
        "probability_presence_base": base_presence_proba.tolist(),
        "probability_absence_adjusted": adjusted_absence_proba.tolist(),
        "probability_presence_adjusted": adjusted_presence_proba.tolist(),
        "probability_adjustment_multiplier": combined_multiplier,
        "to_return_risk_class": to_return_risk_class,
        "displayed_result": displayed_result,
        "add_message": add_message,
        "nlcd_percentages": nlcd_percentages,
        "weather_data": weather_data,
    }


if FRONTEND_DIST_DIR.exists():
    # Serve prebuilt frontend so end users can run with Python only.
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
