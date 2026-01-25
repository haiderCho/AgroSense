import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Union, List
from pathlib import Path
from backend.models.config import PATHS, MLflowConfig

class MultiModelPredictor:
    """
    Multi-model predictor that loads models from named directories (joblib format).
    Falls back to MLflow if joblib models are not available.
    """
    
    def __init__(self, experiment_name: str = None):
        """
        Initializes the predictor by loading all available models.
        Priority: joblib exports > MLflow runs
        """
        self.models: Dict[str, Any] = {}
        self.label_encoder = None
        self.scaler = None
        self.experiment_name = experiment_name or MLflowConfig.experiment_name
        
        self.load_resources()
        
        # Try loading from joblib first, fallback to MLflow
        if not self.load_models_from_disk():
            print("No joblib models found, falling back to MLflow...")
            # Set URI explicitly for the fallback
            mlflow.set_tracking_uri(MLflowConfig.tracking_uri)
            self.load_latest_models()

    def load_resources(self):
        """Loads preprocessing artifacts (scaler, label encoder)."""
        try:
            encoder_path = PATHS["lab_encoders"]
            scaler_path = PATHS["scalers"]
            
            self.label_encoder = joblib.load(encoder_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Resources loaded from {Path(encoder_path).parent}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Preprocessing artifacts not found: {e}")

    def load_models_from_disk(self) -> bool:
        """
        Loads models from the named models directory (joblib format).
        Returns True if at least one model was loaded.
        """
        # Ensure it's a Path object
        models_dir = Path(PATHS["models"])
        
        if not models_dir.exists():
            print(f"Models directory not found: {models_dir}")
            return False
        
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                model_path = model_dir / "model.joblib"
                if model_path.exists():
                    try:
                        model_name = model_dir.name
                        self.models[model_name] = joblib.load(model_path)
                        print(f"Loaded {model_name} from {model_path}")
                    except Exception as e:
                        print(f"Failed to load {model_dir.name}: {e}")
        
        if self.models:
            print(f"Loaded models from disk: {list(self.models.keys())}")
            return True
        return False

    def load_latest_models(self):
        """Scans MLflow for the latest run of each model type and loads them."""
        print(f"Scanning MLflow experiment '{self.experiment_name}' for models...")
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                print(f"Experiment {self.experiment_name} not found.")
                return

            # Find runs that have the 'model_type' param logged
            # We want the latest run for each model_type
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",
                order_by=["start_time DESC"]
            )
            
            if runs.empty:
                print("No runs found.")
                return

            loaded_types = set()
            
            # Iterate through runs to find unique model types
            for _, run in runs.iterrows():
                # Check if 'params.model_type' exists (mlflow prefixes params with 'params.')
                model_type = run.get("params.model_type")
                
                if model_type and model_type not in loaded_types:
                    run_id = run.run_id
                    uri = f"runs:/{run_id}/model"
                    print(f"Loading {model_type} from {uri}...")
                    
                    try:
                        model = mlflow.sklearn.load_model(uri)
                        self.models[model_type] = model
                        loaded_types.add(model_type)
                    except Exception as e:
                        print(f"Failed to load {model_type}: {e}")
            
            print(f"Loaded models from MLflow: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"Error scanning MLflow: {e}")

    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extracts feature importance if available (Tree models)."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                # Normalize
                total = sum(importances)
                if total > 0:
                    importances = importances / total
                    
                # Pair with names
                return {
                    name: float(score) 
                    for name, score in zip(feature_names, importances)
                }
            elif hasattr(model, "coef_"):
                # Linear models (take absolute value of coefs for 'importance')
                # Note: This is a simplification for xAI
                coefs = np.abs(model.coef_[0])
                total = sum(coefs)
                if total > 0:
                    coefs = coefs / total
                return {
                    name: float(score) 
                    for name, score in zip(feature_names, coefs)
                }
        except Exception:
            pass
        return {}

    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicts crop using all loaded models.
        Returns consensus and individual predictions with xAI.
        """
        # Prepare Input
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame([input_data])[feature_order]
        
        # Scale and maintain feature names to avoid sklearn warnings
        scaled_array = self.scaler.transform(input_df)
        scaled_features = pd.DataFrame(scaled_array, columns=feature_order)

        predictions_list = []
        votes = []

        for name, model in self.models.items():
            # Predict
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(scaled_features)
                pred_idx = np.argmax(probs, axis=1)[0]
                confidence = float(np.max(probs, axis=1)[0])
            else:
                pred_idx = model.predict(scaled_features)[0]
                confidence = 1.0 # Fallback

            predicted_label = self.label_encoder.inverse_transform([pred_idx])[0]
            votes.append(predicted_label)
            
            # xAI
            explanation = self._get_feature_importance(model, feature_order)
            
            predictions_list.append({
                "model": name,
                "crop": predicted_label,
                "confidence": confidence,
                "explanation": explanation
            })

        # Determine Consensus (Vote)
        if votes:
            from collections import Counter
            consensus_crop = Counter(votes).most_common(1)[0][0]
        else:
            consensus_crop = "Unknown"

        return {
            "consensus_crop": consensus_crop,
            "predictions": predictions_list
        }

