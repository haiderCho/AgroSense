from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class MLflowConfig:
    experiment_name: str = "AgroSense_Crop_Recommendation"
    tracking_uri: str = "file:./mlflow_data"  # Local file-based tracking

@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

# Default configurations for proposed models
MODELS = {
    "rf": ModelConfig(
        name="RandomForest",
        params={
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42
        },
        description="Random Forest Classifier"
    ),
    "xgb": ModelConfig(
        name="XGBoost",
        params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "objective": "multi:softprob",
            "random_state": 42,
            "eval_metric": "mlogloss"
        },
        description="XGBoost Classifier"
    ),
    "svm": ModelConfig(
        name="SVM",
        params={
            "C": 1.0,
            "kernel": "rbf",
            "probability": True,
            "random_state": 42
        },
        description="Support Vector Machine"
    ),
    "catboost": ModelConfig(
        name="CatBoost",
        params={
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 6,
            "loss_function": "MultiClass",
            "verbose": False,
            "random_seed": 42
        },
        description="CatBoost Classifier"
    ),
    "lr": ModelConfig(
        name="LogisticRegression",
        params={
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs", 
            "multi_class": "multinomial" 
        },
        description="Logistic Regression"
    ),
    # Ensembles require special handling in factory, params here might be meta-learner generic
    "ensemble": ModelConfig(
        name="VotingEnsemble",
        params={
            "voting": "soft"
        },
        description="Voting Classifier using RF, XGB, SVM"
    ),
    "stacking": ModelConfig(
        name="StackingClassifier",
        params={
            "cv": 5
        },
        description="Stacking Classifier with Logistic Regression Final Estimator"
    )
}

# Hyperparameter Grids for Tuning
PARAM_GRIDS = {
    "rf": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    },
    "xgb": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    },
    "svm": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"]
    },
    "catboost": {
        "iterations": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7]
    },
    "lr": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "saga"],
        "penalty": ["l2"] 
    }
}

PATHS = {
    "processed_data": "data/processed",
    "lab_encoders": "data/processed/label_encoder.joblib",
    "scalers": "data/processed/scaler.joblib",
    "models": "models",  # Root level models directory for portability
    "artifacts": "mlflow_data/artifacts" # Artifact store
}

MODEL_NAME_MAP = {
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "svm": "SVM",
    "catboost": "CatBoost",
    "lr": "LogisticRegression",
    "ensemble": "VotingEnsemble",
    "stacking": "StackingClassifier"
}
