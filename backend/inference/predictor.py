import os
import joblib
import pandas as pd
import numpy as np
import mlflow.sklearn
from typing import Dict, Any, Union
from backend.models.config import PATHS

class CropPredictor:
    def __init__(self, model_uri: str = None, local_model_path: str = None):
        """
        Initializes the predictor.
        Args:
            model_uri: MLflow model URI (e.g., "runs:/<run_id>/model").
            local_model_path: Path to a local .joblib model file (fallback).
        """
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        self.load_resources()
        self.load_model(model_uri, local_model_path)

    def load_resources(self):
        """Loads preprocessing artifacts (scaler, label encoder)."""
        try:
            self.label_encoder = joblib.load(PATHS["lab_encoders"])
            self.scaler = joblib.load(PATHS["scalers"])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Preprocessing artifacts not found at defined paths: {e}")

    def load_model(self, model_uri: str, local_model_path: str):
        """Loads the model from MLflow or local disk."""
        if model_uri:
            print(f"Loading model from MLflow URI: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
        elif local_model_path and os.path.exists(local_model_path):
            print(f"Loading model from local path: {local_model_path}")
            self.model = joblib.load(local_model_path)
        else:
            raise ValueError("No valid model URI or local path provided.")

    def predict(self, input_data: Union[Dict[str, float], pd.DataFrame]) -> Dict[str, Any]:
        """
        Predicts the crop based on input features.
        
        Args:
            input_data: Dictionary or DataFrame containing features:
                        ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        Returns:
            Dictionary containing 'predicted_crop' and 'confidence'.
        """
        # Ensure input is DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
            
        # Feature order must match training
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Check for missing columns
        missing = [col for col in feature_order if col not in input_df.columns]
        if missing:
            raise ValueError(f"Input data missing required features: {missing}")
            
        # Reorder columns
        input_df = input_df[feature_order]
        
        # Scale features
        scaled_features = self.scaler.transform(input_df)
        
        # Predict
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled_features)
            predicted_idx = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)
        else:
            predicted_idx = self.model.predict(scaled_features)
            confidence = np.ones(len(predicted_idx)) # Placeholder if no proba

        # Decode Label
        predicted_labels = self.label_encoder.inverse_transform(predicted_idx)
        
        results = []
        for label, conf in zip(predicted_labels, confidence):
            results.append({
                "predicted_crop": label,
                "confidence": float(conf)
            })
            
        return results if len(results) > 1 else results[0]
