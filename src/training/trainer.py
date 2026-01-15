import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from src.models.factory import ModelFactory
from src.models.config import PATHS, MLflowConfig

class Trainer:
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or MLflowConfig.experiment_name
        mlflow.set_experiment(self.experiment_name)
        self.load_data()

    def load_data(self):
        """Loads processed data from defined paths."""
        data_dir = PATHS["processed_data"]
        self.X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
        self.X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
        self.y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv")).values.ravel()
        self.X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        self.y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    def train_model(self, model_key: str):
        """Trains a specific model and logs it to MLflow."""
        print(f"\nStarting training for: {model_key}")
        
        with mlflow.start_run(run_name=f"train_{model_key}"):
            # Instantiate model
            model = ModelFactory.get_model(model_key)
            
            # Log hierarchy params
            mlflow.log_param("model_type", model_key)
            mlflow.log_params(model.get_params())
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Evaluate on Validation
            preds = model.predict(self.X_val)
            
            # Metrics
            metrics = {
                "accuracy": accuracy_score(self.y_val, preds),
                "f1_macro": f1_score(self.y_val, preds, average="macro"),
                "precision_macro": precision_score(self.y_val, preds, average="macro"),
                "recall_macro": recall_score(self.y_val, preds, average="macro")
            }
            
            mlflow.log_metrics(metrics)
            print(f"Metrics for {model_key}: {metrics}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_val, preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix: {model_key}")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = f"confusion_matrix_{model_key}.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            os.remove(cm_path) # Clean up local file
            plt.close()

            # Save Model
            mlflow.sklearn.log_model(model, "model")
            
            return model, metrics

    def run_experiment(self, models_to_run: list = None):
        """Runs training for a list of model keys."""
        if models_to_run is None:
            models_to_run = ["rf", "xgb"] # Default subset
            
        results = {}
        for model_key in models_to_run:
            try:
                model, metrics = self.train_model(model_key)
                results[model_key] = metrics
            except Exception as e:
                print(f"Error training {model_key}: {e}")
                
        return results
