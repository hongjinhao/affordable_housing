from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import pandas as pd
from pydantic import BaseModel

from affordable_housing.config import MODELS_DIR
from affordable_housing.modeling.predict import predict

app = FastAPI(title="Affordable Housing Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://13.57.238.154", "*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model for input data
class PredictionInput(BaseModel):
    avg_targeted_affordability: float
    CDLAC_total_points_score: int
    CDLAC_tie_breaker_self_score: float
    bond_request_amount: float
    homeless_percent: float
    construction_type: str
    housing_type: str
    CDLAC_pool_type: str
    new_construction_set_aside: str
    CDLAC_region: str


# Pydantic model for output data
class PredictionOutput(BaseModel):
    prediction: int  # 1 for "Yes", 0 for "No"
    probability: float  # probability of award


@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input: PredictionInput):
    """Predict whether a housing project will receive funding."""
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([input.dict()])

        # Load model and perform inference
        model_path = MODELS_DIR / "model.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=500, detail="Model file not found")

        result = predict(input_data)

        # Format response
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}
