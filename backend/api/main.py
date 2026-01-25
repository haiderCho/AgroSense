from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.api.schemas import PredictionRequest, PredictionResponse
from backend.inference.predictor import MultiModelPredictor
import uvicorn
import os

app = FastAPI(
    title="AgroSense API",
    description="Multi-model Crop Recommendation Engine with xAI",
    version="2.0"
)

# CORS (Allow Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
# We utilize the "lifespan" or just lazy load on startup. 
# Since Render/Container startup allows it, we initialize globally.
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = MultiModelPredictor()
        print("AgroSense Predictor initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "models_loaded": list(predictor.models.keys()) if predictor else []}

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
    total_expected = 7  # Number of expected models
    
    if len(loaded_models) >= total_expected:
        return {
            "status": "ready",
            "models_loaded": len(loaded_models),
            "models_total": total_expected,
            "message": "All models loaded"
        }
    else:
        return {
            "status": "partial",
            "models_loaded": len(loaded_models),
            "models_total": total_expected,
            "message": f"Loaded {len(loaded_models)}/{total_expected} models"
        }

from fastapi.concurrency import run_in_threadpool

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor is starting up or failed to initialize")
    
    try:
        # Convert Request model to Dict
        input_data = request.dict()
        
        # Run Inference in threadpool to avoid blocking event loop
        result = await run_in_threadpool(predictor.predict, input_data)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
