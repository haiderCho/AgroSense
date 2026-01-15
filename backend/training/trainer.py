import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from backend.models.factory import ModelFactory
from backend.models.config import PATHS, MLflowConfig

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

    def tune_model(self, model_key: str, n_iter: int = 10, cv: int = 3):
        """
        Tunes hyperparameters using RandomizedSearchCV and returns the best model.
        """
        print(f"\n--- Tuning {model_key} ---")
        from sklearn.model_selection import RandomizedSearchCV
        from backend.models.config import PARAM_GRIDS
        
        if model_key not in PARAM_GRIDS:
            print(f"No parameter grid found for {model_key}, skipping tuning and using default.")
            return self.train_model(model_key)

        with mlflow.start_run(run_name=f"tune_{model_key}"):
            # Instantiate base model
            base_model = ModelFactory.get_model(model_key)
            param_grid = PARAM_GRIDS[model_key]
            
            # Setup Random Search
            # n_jobs=-1 uses all processors
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                verbose=1,
                random_state=42,
                n_jobs=-1,
                scoring='accuracy'
            )
            
            # Run Search
            print(f"Running RandomizedSearchCV for {n_iter} iterations...")
            search.fit(self.X_train, self.y_train)
            
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_
            
            print(f"Best Params: {best_params}")
            print(f"Best CV Score: {best_score:.4f}")
            
            # Log best params
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_accuracy", best_score)
            
            # Final Evaluation on Held-out Validation Set
            preds = best_model.predict(self.X_val)
            metrics = {
                "accuracy": accuracy_score(self.y_val, preds),
                "f1_macro": f1_score(self.y_val, preds, average="macro"),
                "precision_macro": precision_score(self.y_val, preds, average="macro"),
                "recall_macro": recall_score(self.y_val, preds, average="macro")
            }
            mlflow.log_metrics(metrics)
            print(f"Validation Metrics (Best Model): {metrics}")
            
            # Log Confusion Matrix
            cm = confusion_matrix(self.y_val, preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix: {model_key} (Tuned)")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = f"confusion_matrix_{model_key}_tuned.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)

            # Save Best Model
            mlflow.sklearn.log_model(best_model, "model")
            
            return best_model, metrics

    def run_experiment(self, models_to_run: list = None, tune: bool = False):
        """Runs training for a list of model keys, optionally with tuning."""
        if models_to_run is None:
            models_to_run = ["rf", "xgb"] # Default subset
            
        results = {}
        for model_key in models_to_run:
            try:
                if tune:
                    model, metrics = self.tune_model(model_key)
                else:
                    model, metrics = self.train_model(model_key)
                results[model_key] = metrics
            except Exception as e:
                print(f"Error training {model_key}: {e}")
                import traceback
                traceback.print_exc()
                
        return results
