import io
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# starting an api call
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
    "with_soil": {
        "prediction": Path(
            os.getenv(
                "PREDICTION_MODEL_WITH_SOIL_PATH",
                BASE_DIR / "input_model_file_here.pkl",
            )
        ),
        "risk": Path(
            os.getenv(
                "RISK_MODEL_WITH_SOIL_PATH",
                BASE_DIR / "risk_mode.pkl",
            )
        ),
    },
    "without_soil": {
        "prediction": Path(
            os.getenv(
                "PREDICTION_MODEL_WITHOUT_SOIL_PATH",
                BASE_DIR / "input_model_file_without_soil.pkl",
            )
        ),
        "risk": Path(
            os.getenv(
                "RISK_MODEL_WITHOUT_SOIL_PATH",
                BASE_DIR / "risk_model_without_soil.pkl",
            )
        ),
    },
    "without_longlat": {
        "prediction": Path(
            os.getenv(
                "PREDICTION_MODEL_WITHOUT_LONGLAT_PATH",
                BASE_DIR / "input_model_file_without_longlat.pkl",
            )
        ),
        "risk": Path(
            os.getenv(
                "RISK_MODEL_WITHOUT_LONGLAT_PATH",
                BASE_DIR / "risk_model_without_longlat.pkl",
            )
        ),
    },
}

# creating a dictonary to store the loaded models
MODEL_CACHE: dict[str, Any] = {}

# loading in the model to run
def load_model(model_variant: str, model_type: str) -> Any:
    # just a quick check for some unknown model, basically a fall-back 
    if model_variant not in MODEL_PATHS or model_type not in MODEL_PATHS[model_variant]:
        raise HTTPException(status_code=400, detail="Invalid model_type.")

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

# Now, we are running the model
@app.post("/predict")
async def predict(
    file: UploadFile | None = File(default=None),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None),
    model_type: str = Form(...),
    longlat_mode: str = Form(default="with_longlat"),
) -> dict[str, Any]:
    # feedback for the user if they did not input anything (csv file and long/lat)
    if file is None and (lat is None or lon is None):
        raise HTTPException(
            status_code=400, detail="Provide either a CSV file or both lat and lon."
        )

    # feedback for user if they did not input the model type they want to run
    if model_type not in ("prediction", "risk"):
        raise HTTPException(
            status_code=400, detail="model_type must be 'prediction' or 'risk'."
        )
    
    # feedback for user if they did not input the long lat option
    if longlat_mode not in ("with_longlat", "without_longlat"):
        raise HTTPException(
            status_code=400,
            detail="longlat_mode must be 'with_longlat' or 'without_longlat'.",
        )

    # automatically working with if there is a csv inputed or not and running with and without soil
    if file is not None:
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
        model_variant = "without_longlat" if longlat_mode == "without_longlat" else "with_soil"
    else:
        if longlat_mode == "without_longlat":
            raise HTTPException(
                status_code=400,
                detail="without_longlat mode requires CSV input.",
            )
        X = [[lat, lon]]
        model_variant = "without_soil"

    model = load_model(model_variant, model_type)

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
        "result": result.tolist(),
    }
