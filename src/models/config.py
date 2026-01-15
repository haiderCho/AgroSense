from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class MLflowConfig:
    experiment_name: str = "crop_recommendation"
    tracking_uri: Optional[str] = None  # Use default local if None

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

PATHS = {
    "processed_data": "data/processed",
    "lab_encoders": "data/processed/label_encoder.joblib",
    "scalers": "data/processed/scaler.joblib"
}
