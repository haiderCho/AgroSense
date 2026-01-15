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

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor is starting up or failed to initialize")
    
    try:
        # Convert Request model to Dict
        input_data = request.dict()
        
        # Run Inference
        result = predictor.predict(input_data)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
