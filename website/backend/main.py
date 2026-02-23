import io
import json
from pathlib import Path
from time import perf_counter
from typing import Any

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
        "risk": BASE_DIR / "risk_model_without_soil.pkl",
    },
    "soil_only": {
        "prediction": BASE_DIR / "input_model_file_only_soil.pkl",
        "risk": BASE_DIR / "risk_model_without_longlat.pkl",
    },
}

# creating a dictonary to store the loaded models
MODEL_CACHE: dict[str, Any] = {}
EE_INITIALIZED = False

# Soil and Long/lat variables to group
SOIL_VARS = ['pH', 'Copper (mg/Kg)', 'Molybdenum (mg/Kg)', 'Sulfur (mg/Kg)', 'Moisture',
       'Manganese (mg/Kg)', 'Aluminum (mg/Kg)', 'Potassium (mg/Kg)',
       'Total nitrogen (%)', 'double Zinc (mg/Kg)', 'Organic matter (%)', 'Phosphorus (mg/Kg)',
       'Iron (mg/Kg)', 'Magnesium (mg/Kg)', 'Sodium (mg/Kg)', 'Calcium (mg/Kg)']
LOG_SOIL_VARS = ['Sulfur (mg/Kg)', 'Moisture',
       'Manganese (mg/Kg)', 'Aluminum (mg/Kg)', 'Potassium (mg/Kg)',
       'Total nitrogen (%)', 'Organic matter (%)', 'Phosphorus (mg/Kg)',
       'Iron (mg/Kg)', 'Magnesium (mg/Kg)', 'Sodium (mg/Kg)', 'Calcium (mg/Kg)']
DOUBLE_LOG_SOIL_VARS = ['double Zinc (mg/Kg)']

LONGLAT_VARS = ['Latitude', 'Longitude', 
       'Precipitation (mm)', 'Max temperature (℃ )', 'Min temperature (℃ )',
       'Wind speed (m/s)', 'Barren (%)', 'Forest (%)', 'Pasture (%)',
       'Grassland (%)', 'Shrubland (%)',
       'Open water (%)', 'Total carbon (%)',
       'Developed open space (> 20% Impervious Cover) (%)', 'Elevation (m)',
        'Cropland (%)', 'Wetland (%)', "Developed open space (< 20% Impervious Cover) (%)"]
LOG_LONGLAT_VARS = ['Grassland (%)', 'Shrubland (%)',
        'Open water (%)', 'Total carbon (%)',
        'Developed open space (> 20% Impervious Cover) (%)', 'Elevation (m)',
        'Cropland (%)', 'Wetland (%)', "Developed open space (< 20% Impervious Cover) (%)"]
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
    "open_water_pct": (11,),
    "developed_open_space_lt_20_pct": (21,),
    "developed_gt_20_pct": (22, 23, 24),
    "barren_pct": (31,),
    "forest_pct": (41, 42, 43),
    "shrubland_pct": (51, 52),
    "grassland_pct": (71,),
    "cropland_pct": (82,),
    "pasture_pct": (81,),
    "wetland_pct": (90, 95),
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
                "or use local EE auth fallback. "
                f"Underlying error: {exc}"
            ),
        ) from exc

    EE_INITIALIZED = True

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



# -----------------------------------------
# --- Data Preparation for Modeling -------
# -----------------------------------------

def data_prep(file_info, model_variant):
    """
    ----- inputs -----
    file_info: Path object
        file wanting to process
    ----- outputs ----
    df: pandas df
        processed anonymzied data (string columns representing intervals split into min and max, then put as minimum and maximum values for those columns)
    """

    df = pd.read_csv(file_info)

    # doing log transformations of all models if needed
    # comment out if choose a model that does not have log transformations
    if model_variant == "soil_only":
        for col in LOG_SOIL_VARS:
            # print(col)
            df[f"log of {col}"] = np.log(df[col])

        df = df.drop(columns=LOG_SOIL_VARS)
    if model_variant == "longlat_only":
        for col in LOG_LONGLAT_VARS:
            # print(col)
            df[f"log of {col}"] = np.log(df[col])

        df = df.drop(columns=LOG_LONGLAT_VARS)
    if model_variant == "soil_longlat":
        for col in LOG_LONGLAT_VARS:
            # print(col)
            df[f"log of {col}"] = np.log(df[col])
        for col in LOG_SOIL_VARS:
            # print(col)
            df[f"log of {col}"] = np.log(df[col])
        df = df.drop(columns=LOG_SOIL_VARS)
        df = df.drop(columns=LOG_LONGLAT_VARS)


    # Drop 'index' column if it exists, as it's typically an artifact and not a feature
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    # switching missing values and weird failures in writing to np.inf bc pandas didnt handle properly
    df = df.replace("#NAME?", -np.inf)
    df = df.fillna(-np.inf)

    # replacing inf with max number that is not max number + 100 in dict (FOR NOT JUST 99999999)
    df = df.replace(np.inf, 99999)
    # replacing -inf with min number (not -inf) - 100 in dict (FOR NOT JUST -99999999)
    df = df.replace(-np.inf, -99999)

    df = df.dropna(axis=1, how="all")

    # scale if needed (manually change if non-scaled model is chosen)

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
    model_type: str = Form(...),
    longlat_mode: str = Form(default="with_longlat"),
) -> dict[str, Any]:
    nlcd_percentages: dict[str, float] | None = None
    gis_loaded = False
    gis_fetch_ms: int | None = None

    # feedback for the user if they did not input anything (csv file and long/lat)
    if file is None and (lat is None or lon is None):
        raise HTTPException(
            status_code=400, detail="Provide either a CSV file or coordinates."
        )

    # feedback for user if they did not input the model type they want to run
    if model_type not in ("prediction", "risk"):
        raise HTTPException(
            status_code=400, detail="the model type must be 'prediction' or 'risk'."
        )
    
    # Normalize aliases so frontend naming changes do not break backend routing.
    longlat_mode_aliases = {
        "with_longlat": "longlat_only",
        "with_soil": "soil_only",
        "soil_longlat": "soil_longlat",
        "longlat_only": "longlat_only",
        "soil_only": "soil_only",
    }
    if longlat_mode not in longlat_mode_aliases:
        raise HTTPException(
            status_code=400,
            detail="Must select one of the model options. Try reselecting from the dropdown.",
        )
    normalized_mode = longlat_mode_aliases[longlat_mode]

    # automatically working with if there is a csv inputed or not and running with and without soil
    if normalized_mode == "soil_only":
        # attempting to read in the csv
        try:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
            X = df.to_numpy()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not parse CSV upload: {exc}"
            ) from exc
        # making sure the longitude and latitude is inputed 
        # (because cannot run on no data, so either enviro from long/lat or soil is required)
        model_variant = "soil_only"
    elif normalized_mode == "longlat_only":
        # handeling api calls to get data with long and lat
        X = {
            "latitude": lat,
            "longitude": lon,
        }
        # checking to make sure the long lat is only inside the USA
        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="The latitude and longitude provided are not completed or correct")
        if not in_us_bbox(lat, lon):
            raise HTTPException(status_code=400, detail=f"Coordinates must be inside US. (Must be within {US_BBOX})")

        gis_start = perf_counter()
        nlcd_percentages = get_nlcd_percentages(lat=lat, lon=lon, buffer_m=1000)
        print(type(nlcd_percentages))
        
        gis_fetch_ms = int((perf_counter() - gis_start) * 1000)
        gis_loaded = True
        model_variant = "longlat_only"
    elif normalized_mode == "soil_longlat":
        # handeling api calls to get data with long and lat
        X = {
            "latitude": lat,
            "longitude": lon,
        }
        
        # checking to make sure the long lat is only inside the USA
        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="The latitude and longitude provided are not completed or correct")
        if not in_us_bbox(lat, lon):
            raise HTTPException(status_code=400, detail=f"Coordinates must be inside US. (Must be within {US_BBOX})")

        gis_start = perf_counter()
        nlcd_percentages = get_nlcd_percentages(lat=lat, lon=lon, buffer_m=1000)
        print(type(nlcd_percentages))
        gis_fetch_ms = int((perf_counter() - gis_start) * 1000)
        gis_loaded = True

        print("hit line 315")

        # attempting to read in the csv
        try:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
            print("made it in the try")
            X = df.to_numpy()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not parse CSV upload: {exc}"
            ) from exc
        
        model_variant = "soil_longlat"

    
    else:
        raise HTTPException(
            status_code=400,
            detail="If you do not input longitude and latitude, you need to input a CSV with soil data.",
        )

    model = load_model(model_variant, model_type)
    print("Running: ", model)
    # running a prediction model!
    try:
        result = model.predict(X)
    except Exception as exc:
        # user feedback if the model fails
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    # returning results if pressent!
    return {
        "success": True,
        "model_variant": model_variant,
        "longlat_mode": longlat_mode,
        "gis_loaded": gis_loaded,
        "gis_fetch_ms": gis_fetch_ms,
        "result": result.tolist(),
        "nlcd_percentages": nlcd_percentages,
    }
