
# %% [markdown]
# # Integrated Training Pipeline
# 
# This script implements a modular, class-based training pipeline for the AgroSense project.
# It performs the following steps:
# 1.  **Data Loading**: Loads processed data and splits it into training and test sets.
# 2.  **Model Tuning**: Optimizes hyperparameters for multiple models (Random Forest, XGBoost, LightGBM, CatBoost, AdaBoost) using Optuna.
# 3.  **Ensemble Building**: Creates Voting and Stacking ensembles from the best tuned models.
# 4.  **Evaluation**: Calculates metrics (Accuracy, F1, Top-3, etc.), generates confusion matrices, and compares model performance.
# 5.  **Persistence**: Saves the best models and metrics.

# %%
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import argparse

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, top_k_accuracy_score, classification_report
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from optuna_integration import XGBoostPruningCallback

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = '../data/processed'
MODELS_DIR = '../models'
RESULTS_DIR = '../results'

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Data Loader

# %%
class DataLoader:
    def __init__(self, dataset_type='crop'):
        self.dataset_type = dataset_type
        self.processed_data_path = f'{DATA_DIR}/{dataset_type}_recommendation.csv'
        self.preprocessing_pipeline_path = f'{MODELS_DIR}/pipeline_{dataset_type}.joblib'
        self.target_col = 'label'
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        self.label_encoder = None
        self.target_names = None

    def load_data(self):
        """Loads processed data and splits it."""
        logger.info(f"Loading data from {self.processed_data_path}...")
        
        if not os.path.exists(self.processed_data_path):
            raise FileNotFoundError(f"Processed data not found at {self.processed_data_path}. Run ml/preprocessing.py first.")
            
        df = pd.read_csv(self.processed_data_path)
        
        if self.target_col not in df.columns:
             raise ValueError(f"Column '{self.target_col}' not found in {self.processed_data_path}")
             
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Stratified Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Label Encoding
        if self.y_train.dtype == 'object':
            self.label_encoder = LabelEncoder()
            self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            self.y_test_encoded = self.label_encoder.transform(self.y_test)
            self.target_names = list(map(str, self.label_encoder.classes_))
            
            # Save the LabelEncoder for later inference use
            joblib.dump(self.label_encoder, f'{MODELS_DIR}/label_encoder_{self.dataset_type}.joblib')
        else:
            self.y_train_encoded = self.y_train
            self.y_test_encoded = self.y_test
            self.target_names = [str(i) for i in sorted(y.unique())]
            
        logger.info(f"Data Loaded. shapes: X_train={self.X_train.shape}, X_test={self.X_test.shape}")
        
    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.y_train_encoded, self.y_test_encoded, self.target_names

# %% [markdown]
# ## 2. Model Tuner

# %%
class ModelTuner:
    def __init__(self, X_train, y_train, n_trials=50, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_models = {}
        self.le = None
        self.classes_ = None

        if hasattr(y_train, 'dtype') and y_train.dtype == 'object':
             # Fallback if passed raw strings, typically we pass encoded now
             pass

    def optimize_rf(self) -> RandomForestClassifier:
        logger.info("Starting Random Forest Optimization...")
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # 5-fold CV
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=5, scoring='f1_macro')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best RF params: {study.best_params}")
        best_rf = RandomForestClassifier(**study.best_params, random_state=self.random_state, n_jobs=-1)
        best_rf.fit(self.X_train, self.y_train)
        self.best_models['RandomForest'] = best_rf
        return best_rf

    def optimize_xgb(self) -> xgb.XGBClassifier:
        logger.info("Starting XGBoost Optimization...")
        
        # Split train again for early stopping validation within the optimization
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, stratify=self.y_train, random_state=self.random_state
        )

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'mlogloss',
                'enable_categorical': True 
            }
            
            # Pruning callback
            pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
            
            clf = xgb.XGBClassifier(
                **params,
                callbacks=[pruning_callback],
                early_stopping_rounds=50
            )
            
            clf.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            preds = clf.predict(X_val)
            # Use macro f1 for optimization goal
            return precision_recall_fscore_support(y_val, preds, average='macro')[2]


        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best XGB params: {study.best_params}")
        best_xgb = xgb.XGBClassifier(**study.best_params, random_state=self.random_state, n_jobs=-1, enable_categorical=True)
        # For final fit, we don't use early stopping/callbacks to leverage full data
        best_xgb.fit(self.X_train, self.y_train)
        self.best_models['XGBoost'] = best_xgb
        return best_xgb

    def optimize_lgbm(self) -> lgb.LGBMClassifier:
        logger.info("Starting LightGBM Optimization...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }
            
            clf = lgb.LGBMClassifier(**params)
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=5, scoring='f1_macro')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best LGBM params: {study.best_params}")
        best_lgbm = lgb.LGBMClassifier(**study.best_params, class_weight='balanced', random_state=self.random_state, n_jobs=-1, verbose=-1)
        best_lgbm.fit(self.X_train, self.y_train)
        self.best_models['LightGBM'] = best_lgbm
        return best_lgbm

    def optimize_catboost(self) -> CatBoostClassifier:
        logger.info("Starting CatBoost Optimization...")
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
                'random_seed': self.random_state,
                'verbose': False,
                'allow_writing_files': False
            }
            
            clf = CatBoostClassifier(**params)
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=3, scoring='f1_macro') # Reduced CV for speed
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best CatBoost params: {study.best_params}")
        best_cat = CatBoostClassifier(**study.best_params, random_seed=self.random_state, verbose=False, allow_writing_files=False)
        best_cat.fit(self.X_train, self.y_train)
        self.best_models['CatBoost'] = best_cat
        return best_cat

    def optimize_adaboost(self) -> AdaBoostClassifier:
        logger.info("Starting AdaBoost Optimization...")
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 2.0, log=True)
            
            clf = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.random_state,
                algorithm="SAMME" # Default in newer sklearn versions
            )
            
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=5, scoring='f1_macro')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best AdaBoost params: {study.best_params}")
        best_ada = AdaBoostClassifier(**study.best_params, random_state=self.random_state, algorithm="SAMME")
        best_ada.fit(self.X_train, self.y_train)
        self.best_models['AdaBoost'] = best_ada
        return best_ada

    def get_best_models(self) -> Dict[str, BaseEstimator]:
        return self.best_models

# %% [markdown]
# ## 3. Ensemble Factory

# %%
class EnsembleFactory:
    def __init__(self, models: Dict[str, BaseEstimator], random_state=42):
        self.models = models
        self.random_state = random_state

    def create_voting_classifier(self) -> VotingClassifier:
        logger.info("Building VotingClassifier (Soft)...")
        estimators = [(name, model) for name, model in self.models.items()]
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        return voting_clf

    def create_stacking_classifier(self, final_estimator=None) -> StackingClassifier:
        logger.info("Building StackingClassifier...")
        if final_estimator is None:
            final_estimator = LogisticRegression()
            
        estimators = [(name, model) for name, model in self.models.items()]
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )
        return stacking_clf

# %% [markdown]
# ## 4. Evaluator

# %%
class Evaluator:
    def __init__(self, X_test, y_test, class_names=None):
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.metrics = {}

    def evaluate_model(self, model, name_prefix):
        logger.info(f"Evaluating {name_prefix}...")
        
        try:
            y_pred = model.predict(self.X_test)
        except Exception as e:
            logger.error(f"Prediction failed for {name_prefix}:{e}")
            return

        # Calculate standard metrics
        acc = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='macro', zero_division=0)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(self.X_test)
            # Top-3 Accuracy
            try:
                # Only if n_classes > 2
                if len(self.class_names) > 2:
                    top3 = top_k_accuracy_score(self.y_test, y_proba, k=3, labels=sorted(np.unique(self.y_test)))
                else:
                    top3 = acc # For binary, it's just acc
            except Exception:
                top3 = 0.0 
        else:
            top3 = 0.0

        self.metrics[name_prefix] = {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Top-3 Accuracy': top3
        }
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm, name_prefix)
        
        # Classification Report
        report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
        self.save_individual_metrics(name_prefix, report)
        
    def plot_confusion_matrix(self, cm, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names if self.class_names is not None else "auto",
                    yticklabels=self.class_names if self.class_names is not None else "auto")
        plt.title(f'Confusion Matrix - {title}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/confusion_matrix_{title}.png')
        plt.close()

    def compare_models(self):
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df.to_csv(f'{RESULTS_DIR}/model_comparison.csv')
        
        # Plot comparison
        if not comparison_df.empty:
            comparison_df[['Accuracy', 'F1 Score', 'Top-3 Accuracy']].plot(kind='bar', figsize=(14, 7))
            plt.title('Model Comparison')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/model_comparison.png')
            plt.close()
        
        return comparison_df

    def save_individual_metrics(self, name, report):
        with open(f'{RESULTS_DIR}/metrics_{name}.json', 'w') as f:
            json.dump(report, f, indent=4)

    def save_final_results(self):
        with open(f'{RESULTS_DIR}/all_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)

# %% [markdown]
# ## 5. Main Execution

# %%
def main():
    parser = argparse.ArgumentParser(description='AgroSense Training Pipeline')
    parser.add_argument('--dataset', type=str, default='crop', choices=['crop', 'fertilizer'], help='Dataset to train on')
    parser.add_argument('--model', type=str, default='all', choices=['rf', 'xgb', 'lgbm', 'cat', 'ada', 'all'], help='Model to train')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Training Pipeline for {args.dataset.upper()} recommendation...")
    
    try:
        # Load Data
        loader = DataLoader(dataset_type=args.dataset)
        loader.load_data()
        X_train, X_test, _, y_test, y_train_enc, y_test_enc, target_names = loader.get_data()
        
        # Initialize Tuner
        tuner = ModelTuner(X_train, y_train_enc, n_trials=args.trials)
        if hasattr(loader, 'label_encoder') and loader.label_encoder:
             tuner.le = loader.label_encoder
             tuner.classes_ = loader.label_encoder.classes_
        
        # Helper to train and save
        def train_and_save(model_name, optimize_func):
            model = optimize_func()
            joblib.dump(model, f'{MODELS_DIR}/best_{model_name}_{args.dataset}.joblib')
            return model

        best_models = {}
        
        # Train Models
        if args.model == 'rf' or args.model == 'all':
            best_models['RandomForest'] = train_and_save('RandomForest', tuner.optimize_rf)
        if args.model == 'xgb' or args.model == 'all':
            best_models['XGBoost'] = train_and_save('XGBoost', tuner.optimize_xgb)
        if args.model == 'lgbm' or args.model == 'all':
            best_models['LightGBM'] = train_and_save('LightGBM', tuner.optimize_lgbm)
        if args.model == 'cat' or args.model == 'all':
            best_models['CatBoost'] = train_and_save('CatBoost', tuner.optimize_catboost)
        if args.model == 'ada' or args.model == 'all':
            best_models['AdaBoost'] = train_and_save('AdaBoost', tuner.optimize_adaboost)

        # Initialize Evaluator
        evaluator = Evaluator(X_test, y_test_enc, class_names=target_names)
        
        # Evaluate Individual Models
        for name, model in best_models.items():
            evaluator.evaluate_model(model, name)
            
        # Build Ensembles
        if args.model == 'all':
            ensemble_factory = EnsembleFactory(best_models, X_train, y_train_enc)
            
            voting_clf = ensemble_factory.create_voting_classifier()
            if voting_clf:
                voting_clf.fit(X_train, y_train_enc)
                evaluator.evaluate_model(voting_clf, "VotingEnsemble")
                joblib.dump(voting_clf, f'{MODELS_DIR}/voting_ensemble_{args.dataset}.joblib')
                
            stacking_clf = ensemble_factory.create_stacking_classifier()
            if stacking_clf:
                stacking_clf.fit(X_train, y_train_enc)
                evaluator.evaluate_model(stacking_clf, "StackingEnsemble")
                joblib.dump(stacking_clf, f'{MODELS_DIR}/stacking_ensemble_{args.dataset}.joblib')
        
        # Save Results
        if args.model == 'all':
            evaluator.save_final_results()
            evaluator.compare_models()
        else:
             print(f"Metrics: {evaluator.metrics}")
             
        logger.info(f"Pipeline Completed for {args.dataset}.")
        
    except Exception as e:
        logger.error(f"Pipeline Failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
