import io
import json
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

import ee
import numpy as np

# ------------------------------
# --- Global variables ---------
# ------------------------------

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
# creating options for models to run if farmers choose to upload different information
MODEL_PATHS: dict[str, dict[str, Path]] = {
    "soil_longlat": {
        "prediction": BASE_DIR / "input_model_file_here.pkl",
        "risk": BASE_DIR / "risk_mode.pkl",
    },
    "longlat_only": {
        "prediction": BASE_DIR / "input_model_file_only_longlat.pkl",
        "risk": BASE_DIR / "risk_model_only_longlat.pkl",
    },
    "soil_only": {
        "prediction": BASE_DIR / "input_model_file_only_soil.pkl",
        "risk": BASE_DIR / "risk_model_only_soil.pkl",
    },
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

# ------------------------------
# --- Helper Functions ---------
# ------------------------------

US_BBOX = {
    "lat_min": 24.5,
    "lat_max": 49.5,
    "lon_min": -125.0,
    "lon_max": -67.0,
}

NLCD_IMAGE_ID = "USGS/NLCD_RELEASES/2021_REL/NLCD/2021"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ------------------------------
# --- Earth Engine Config ------
# ------------------------------
# Set these directly in this file so auth works without terminal env vars.
# Recommended: use a service account JSON key.
EE_PROJECT = "listeria-prediction-tool"
EE_SERVICE_ACCOUNT = "temp-for-iafp-competition@listeria-prediction-tool.iam.gserviceaccount.com"
EE_PRIVATE_KEY_PATH = BASE_DIR / "gee-service-account-key.json"

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

# checking if the geo points provided are within the US
# (this is where the match is made for land cover)
def in_us_bbox(lat: float, lon: float) -> bool:
    return (
        US_BBOX["lat_min"] <= lat <= US_BBOX["lat_max"]
        and US_BBOX["lon_min"] <= lon <= US_BBOX["lon_max"]
    )

# initializeing the google earth engine. Only doing this if the longitude and latitude are recieved to save time
def initialize_earth_engine() -> None:
    """Initializing the Earth Engine. Only once per process."""
    # updating/creating a global variable to reduce times the process is initialized
    global EE_INITIALIZED
    if EE_INITIALIZED: # checking if initialized
        return

    # grabbing account stuff to be able to access google earth engine (ee)
    service_account = EE_SERVICE_ACCOUNT
    private_key_path = EE_PRIVATE_KEY_PATH
    ee_project = EE_PROJECT

    try:
        if service_account and private_key_path:
            key_path = Path(private_key_path)
            if not key_path.exists():
                raise FileNotFoundError(
                    f"Earth Engine key file was not found: {key_path}"
                )

            credentials = ee.ServiceAccountCredentials(service_account, str(key_path))
            ee.Initialize(credentials, project=ee_project)
        else:
            # Fallback if you prefer local EE auth on this machine.
            ee.Initialize(project=ee_project)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Google Earth Engine initialization failed. "
                "Update EE_PROJECT / EE_SERVICE_ACCOUNT / EE_PRIVATE_KEY_PATH in main.py, "
                "or use local EE auth fallback. If you are recieving this message, please notify the github owner"
                f"Underlying error: {exc}"
            ),
        ) from exc

    EE_INITIALIZED = True

# ------------------------------
# ---  Earth Engine - NLCD -----
# ------------------------------

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
    print(f"\n\nGrouped percentages:\n{grouped_percentages}\n")
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
    # doing log transformations of all models if needed
    # comment out if choose a model that does not have log transformations
    if model_variant == "soil_only":
        for col in LOG_SOIL_VARS:
            # print(col)
            df[f"log of {col}"] = np.log1p(df[col])

        for col in DOUBLE_LOG_SOIL_VARS:
            df[f"double log of {col}"] = np.log1p(np.log(df[col]))

        df = df.drop(columns=LOG_SOIL_VARS)
        df = df.drop(columns=DOUBLE_LOG_SOIL_VARS)
    if model_variant == "longlat_only":
        for col in LOG_LONGLAT_VARS:
            # print(col)
            df[f"log of {col}"] = np.log1p(df[col])

        df = df.drop(columns=LOG_LONGLAT_VARS)
    if model_variant == "soil_longlat":
        for col in LOG_LONGLAT_VARS:
            # print(col)
            df[f"log of {col}"] = np.log1p(df[col])
        for col in LOG_SOIL_VARS:
            # print(col)
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
    allowed = allowed_by_variant[model_variant]
    keep = [c for c in df.columns if c in allowed]
    dropped = [c for c in df.columns if c not in allowed]
    print("columns to drop is :", len(dropped))
    print("columns to keep: ", len(dropped))
    # printing to only terminal for debugging reasons
    if dropped:
        print(f"Dropping non-model columns ({model_variant}): {dropped}")
    return df[keep]

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

    print("model variant passed to data prep: ", model_variant)
    print("\n\nlen of df as passed to dataprep: ", len(df.columns))

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

    # dealing with kmeans clusters used in the full model, and also the scaler value
    if model_variant == "soil_longlat":
        scaler = joblib.load(BASE_DIR / "scaler_file.joblib")
        kmeans_raw = joblib.load(BASE_DIR / "kmeans_fitter.joblib")
        kmeans_scaled = joblib.load(BASE_DIR / "scaled_kmeans_fitter.joblib")

        # quickly reindexing so that the scaler can actually work
        df = df.reindex(columns=scaler.feature_names_in_)
        X_scaled = scaler.transform(df) # only for scaled kmeans
        X_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

        # quickly dropping columns not trained on for clusters
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        if 'log of index' in df.columns:
            df = df.drop(columns=['log of index'])
        if 'Unnamed: 0' in X_scaled.columns:
            X_scaled = X_scaled.drop(columns=['Unnamed: 0'])
        if 'log of index' in X_scaled.columns:
            X_scaled = X_scaled.drop(columns=['log of index'])

        df_for_kmeans = df.drop(columns=['log of Open water (%)', 'log of Phosphorus (mg/Kg)', 'log of Wetland (%)', 'double log of Zinc (mg/Kg)', 'log of Aluminum (mg/Kg)', 'log of Cropland (%)', 'log of Developed open space (< 20% Impervious Cover) (%)', 'log of Developed open space (> 20% Impervious Cover) (%)'])
        df_for_kmeans_scaled = X_scaled.drop(columns=['log of Open water (%)', 'log of Phosphorus (mg/Kg)', 'log of Wetland (%)', 'double log of Zinc (mg/Kg)', 'log of Aluminum (mg/Kg)', 'log of Cropland (%)', 'log of Developed open space (< 20% Impervious Cover) (%)', 'log of Developed open space (> 20% Impervious Cover) (%)'])
        
        # grabbing scaled and non-scaled cluster values
        X_scaled["cluster_kmeans"] = kmeans_raw.predict(df_for_kmeans)
        X_scaled["scaled_cluster_kmeans"] = kmeans_scaled.predict(df_for_kmeans_scaled)
        print('got kmeans scaled')
        return X_scaled

    # if ENCODE_STR:
    #     df = pd.get_dummies(df)
    return df

# -----------------------------------------
# --- Loading and Checking Models ---------
# -----------------------------------------

# loading in the model to run
def load_model(model_variant: str, model_type: str) -> Any:
    # just a quick check for some unknown model, basically a fall-back 
    if model_variant not in MODEL_PATHS or model_type not in MODEL_PATHS[model_variant]:
        raise HTTPException(status_code=400, detail="Need to select a model. Invalid model_type.")

    # grabbing the right model option path
    cache_key = f"{model_variant}:{model_type}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    # getting the model path to run it. also quickly checking that the model file is not a dud
    model_path = MODEL_PATHS[model_variant][model_type]
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise HTTPException(
            status_code=503,
            detail=f"Model file not ready: {model_path.name}.", # user feedback
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


# -------------------------------
# --- Handleing Actions ---------
# -------------------------------

# quick check endpoint so we can test Google Earth Engine auth independently
@app.get("/health/earth-engine")
def earth_engine_health() -> dict[str, Any]:
    initialize_earth_engine()
    return {"success": True, "earth_engine_initialized": True}


# Now, we are running the model
@app.post("/predict")
async def predict(
    file: UploadFile | None = File(default=None),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None),
    forecast_date: str | None = Form(default=None),
    model_type: str = Form(...),
    longlat_mode: str = Form(default="longlat_only"),
) -> dict[str, Any]:
    nlcd_percentages: dict[str, float] | None = None
    weather_data: dict[str, float] | None = None
    gis_loaded = False
    gis_fetch_ms: int | None = None
    forecast_date_utc: str | None = None

    # feedback for the user if they did not input anything (csv file and long/lat)
    if file is None and (lat is None or lon is None):
        raise HTTPException(
            status_code=400, detail="Provide either a CSV file or coordinates+date."
        )

    # feedback for user if they did not input the model type they want to run
    if model_type not in ("prediction", "risk"):
        raise HTTPException(
            status_code=400, detail="the model type must be 'prediction' or 'risk'."
        )

    # automatically working with if there is a csv inputed or not and running with and without soil
    if longlat_mode == "soil_only":
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
        print(type(nlcd_percentages))

        # updating the dictionary to have the matching model keys 
        dict_df.update(nlcd_percentages)
        dict_df.update(weather_data)

        df = pd.DataFrame(dict_df)
        print(df.head())
        X = data_prep(df, longlat_mode)
        
        gis_fetch_ms = int((perf_counter() - gis_start) * 1000)
        gis_loaded = True
        print("GIS LOADED: ", gis_loaded)
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
        print("line 609")
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
        print(perf_counter())
        nlcd_percentages = get_nlcd_percentages(lat=lat, lon=lon, buffer_m=1000)
        weather_data = get_open_meteo_daily(lat=lat, lon=lon, target_date=forecast_dt)
        
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
            # Broadcast one-row GIS values to every uploaded soil row.
            print("about to broadcast the weather db: ", len(pd_df.columns))
            df = pd_df.merge(dict_df, how="cross")
            # print(df.head())
            print("starting data prep, df len = ", len(df.columns))
            X = data_prep(df, longlat_mode)
            print("finished data prep")

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

    print('running the model variant: ', model_variant)
    print('running model type: ', model_type)
    # print("cur have    :", list(X.columns))

    model = load_model(model_variant, model_type)
    print("Running: ", model)
    # quick drops as sanity check
    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])
    if 'log of index' in X.columns:
        X = X.drop(columns=['log of index'])
    if 'index' in X.columns:
        X = X.drop(columns=['index'])

    print(X.columns)
    print(X)
    print("model expects:", list(model.feature_names_in_))
    X = X.reindex(columns=model.feature_names_in_)
    # print("cur have    :", list(X.columns))


    # running a prediction model!
    try:
        print(X.columns)
        result = model.predict(X)
    except Exception as exc:
        # user feedback if the model fails
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    # returning results if pressent!
    return {
        "success": True,
        "model_variant": model_variant,
        "longlat_mode": longlat_mode,
        "forecast_date_utc": forecast_date_utc,
        "gis_loaded": gis_loaded,
        "gis_fetch_ms": gis_fetch_ms,
        "result": result.tolist(),
        "nlcd_percentages": nlcd_percentages,
        "weather_data": weather_data,
    }
