from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PredictionRequest(BaseModel):
    N: float = Field(..., ge=0, le=140, description="Nitrogen content in soil")
    P: float = Field(..., ge=5, le=145, description="Phosphorous content in soil")
    K: float = Field(..., ge=5, le=205, description="Potassium content in soil")
    temperature: float = Field(..., ge=5, le=50, description="Temperature in Celsius")
    humidity: float = Field(..., ge=10, le=100, description="Relative humidity in %")
    ph: float = Field(..., ge=0, le=14, description="pH value of the soil")
    rainfall: float = Field(..., ge=20, le=300, description="Rainfall in mm")

class SinglePrediction(BaseModel):
    model: str
    crop: str
    confidence: float
    explanation: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    consensus_crop: str
    predictions: List[SinglePrediction]
