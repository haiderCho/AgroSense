from typing import Any
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from .config import MODELS, ModelConfig

class ModelFactory:
    @staticmethod
    def get_model(model_key: str, config: ModelConfig = None) -> Any:
        if config is None:
            if model_key not in MODELS:
                raise ValueError(f"Model key '{model_key}' not found in default configuration.")
            config = MODELS[model_key]
        
        params = config.params.copy()
        
        if model_key == "rf":
            return RandomForestClassifier(**params)
        
        elif model_key == "xgb":
            return XGBClassifier(**params)
        
        elif model_key == "svm":
            return SVC(**params)
        
        elif model_key == "catboost":
            return CatBoostClassifier(**params)
        
        elif model_key == "lr":
            return LogisticRegression(**params)
        
        elif model_key == "ensemble":
            # Instantiate base models for voting
            estimators = [
                ('rf', ModelFactory.get_model('rf')),
                ('xgb', ModelFactory.get_model('xgb')),
                ('svm', ModelFactory.get_model('svm')),
                ('catboost', ModelFactory.get_model('catboost'))
            ]
            return VotingClassifier(estimators=estimators, **params)
        
        elif model_key == "stacking":
            # Instantiate base models for stacking
            estimators = [
                ('rf', ModelFactory.get_model('rf')),
                ('xgb', ModelFactory.get_model('xgb')),
                ('svm', ModelFactory.get_model('svm')),
                ('catboost', ModelFactory.get_model('catboost'))
            ]
            # Meta-learner is Logistic Regression by default for Stacking
            final_estimator = LogisticRegression()
            return StackingClassifier(estimators=estimators, final_estimator=final_estimator, **params)
            
        else:
            raise ValueError(f"Model type '{model_key}' is not implemented in Factory.")
