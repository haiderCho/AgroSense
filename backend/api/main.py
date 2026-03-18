import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from collections import Counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.concurrency import run_in_threadpool
import uvicorn

from backend.api.schemas import PredictionRequest, PredictionResponse
from backend.inference.predictor import MultiModelPredictor

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("agrosense-api")

# Global Predictor Instance
predictor: Optional[MultiModelPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifecyle management replacing deprecated on_event."""
    global predictor
    try:
        predictor = MultiModelPredictor()
        logger.info("AgroSense Predictor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
    yield
    # Cleanup if needed
    predictor = None
    logger.info("AgroSense Predictor shut down.")

app = FastAPI(
    title="AgroSense API",
    description="Multi-model Crop Recommendation Engine with xAI",
    version="2.1",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Restrict to frontend local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header for performance monitoring."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
def index():
    return {"message": "AgroSense API is running", "version": "2.1"}

@app.get("/health")
def health_check():
    """Detailed health check for orchestration."""
    return {
        "status": "online" if predictor else "initialization_failed",
        "models_loaded": list(predictor.models.keys()) if predictor else [],
        "timestamp": time.time()
    }

@app.get("/status")
def get_status():
    """Returns model loading status for frontend indicator."""
    if not predictor:
        return {
            "status": "loading",
            "models_loaded": 0,
            "models_total": 7,
            "message": "Initializing predictor..."
        }
    
    loaded_models = list(predictor.models.keys())
    total_expected = 7
    
    status = "ready" if len(loaded_models) >= total_expected else "partial"
    
    return {
        "status": status,
        "models_loaded": len(loaded_models),
        "models_total": total_expected,
        "message": "All models loaded" if status == "ready" else f"Loaded {len(loaded_models)}/{total_expected} models"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor is starting up or failed to initialize")
    
    try:
        # Convert Request model to Dict (Pydantic v2 modern way)
        input_data = request.model_dump()
        
        # Run Inference in threadpool to avoid blocking event loop
        result = await run_in_threadpool(predictor.predict, input_data)
        
        # Add input data to response for simulation/tracking
        result["input_data"] = input_data
        
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
