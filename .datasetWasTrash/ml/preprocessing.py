
# %% [markdown]
# # Production-Ready Preprocessing & Feature Engineering Module
# 
# This module implements a robust, scikit-learn compatible preprocessing pipeline for agricultural datasets.
# It is designed to be modular, testable, and production-ready.
#
# **Key Components:**
# 1.  **Global Configuration:** Centralized definition of feature constraints.
# 2.  **Custom Transformers:** Scikit-learn compatible classes for cleaning and feature engineering.
# 3.  **Pipeline Construction:** Automated assembly of preprocessing steps.
# 4.  **Verification:** Built-in unit tests to validate logic.

# %%
import logging
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

# Configure Logging
log_dir = "logs"
if not os.path.exists(log_dir):
    if os.path.exists("../logs"):
        log_dir = "../logs"
    else:
        os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'preprocessing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# %% [markdown]
# ## 1. Global Configuration Block

# %%
def load_constraints(config_path: str = '../config/constraints.json', profile: str = 'default') -> Dict[str, Tuple[float, float]]:
    """Load feature constraints from JSON config."""
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, config_path)
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get(profile, {})
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using fallback defaults.")
        # Relaxed constraints: Min set to 0 to avoid clipping valid zeros
        return {
            'N': (0, 300), 'P': (0, 300), 'K': (0, 300),
            'temperature': (0, 60), 'humidity': (0, 100),
            'ph': (0, 14), 'rainfall': (0, 500)
        }

FEATURE_CONSTRAINTS = load_constraints()

# %% [markdown]
# ## 2. Step 1: Custom Transformers (The Logic)

# %%
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, constraints: Dict[str, Tuple[float, float]] = FEATURE_CONSTRAINTS):
        self.constraints = constraints
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Clean data by handling duplicates and clipping values."""
        X = X.copy()
        
        # 1. Handle Duplicates
        initial_shape = X.shape
        X = X.drop_duplicates()
        if X.shape[0] < initial_shape[0]:
            logger.info(f"DataCleaner: Dropped {initial_shape[0] - X.shape[0]} duplicate rows.")
            
        # 2. Range Validation (Clipping)
        # Only clip if value is truly out of sane bounds. Zeros should remain zeros.
        for col, (min_val, max_val) in self.constraints.items():
            if col in X.columns:
                # Log clipping count for debugging
                lower_clip = (X[col] < min_val).sum()
                upper_clip = (X[col] > max_val).sum()
                if lower_clip > 0 or upper_clip > 0:
                    logger.debug(f"DataCleaner: {col} - clipped {lower_clip} low, {upper_clip} high.")
                
                X[col] = X[col].clip(lower=min_val, upper=max_val)
                
        logger.info(f"DataCleaner: Output shape {X.shape}")
        return X

    def set_output(self, *, transform=None):
        return self

# %%
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Refactored: Pure pass-through. No synthetic features.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Pure pass-through. No ratios.
        return X.copy()

    def set_output(self, *, transform=None):
        return self

# %% [markdown]
# ## 3. Step 2: The Scikit-Learn Pipeline

# %%
def create_training_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Creates the full end-to-end preprocessing pipeline."""
    
    # Numeric Transformer Pipeline
    # Using StandardScaler instead of PowerTransformer for stability if distribution is weird
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Transformer Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # No derived features
    full_numeric_features = numeric_features
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, full_numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False,
        remainder='drop'
    )
    
    pipeline = Pipeline(steps=[
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()), # No-op now
        ('preprocessor', preprocessor)
    ])
    
    pipeline.set_output(transform="pandas")
    return pipeline

# %% [markdown]
# ## 4. Step 3: Logging & Persistence

# %%
def save_pipeline(pipeline: Pipeline, filepath: str):
    try:
        joblib.dump(pipeline, filepath)
        logger.info(f"Pipeline successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save pipeline: {e}")

# %% [markdown]
# ## 5. Step 4: Unit Tests

# %%
def test_pipeline_logic():
    print("Running Unit Tests...")
    data = pd.DataFrame({
        'N': [60, 50, 400], 'P': [20, 40, 40], 'K': [55, 60, 60],
        'temperature': [25, 35, 10], 'humidity': [50, 80, 45],
        'soil_type': ['Sandy', 'Loamy', 'Clay']
    })
    
    cleaner = DataCleaner()
    cleaned = cleaner.transform(data)
    assert cleaned['N'].max() <= 300, "DataCleaner clipping failed"
    
    engineer = FeatureEngineer()
    engineered = engineer.transform(cleaned)
    # assert 'ratio_N_P' in engineered.columns # Removed check
    
    pipeline = create_training_pipeline(['N', 'P', 'K', 'temperature', 'humidity'], ['soil_type'])
    transformed = pipeline.fit_transform(data)
    
    assert isinstance(transformed, pd.DataFrame), "Pipeline output not DataFrame"
    print("[PASS] All Tests Passed!")

# %% [markdown]
# ## 6. Main Execution Block

# %%
def process_full_dataset(input_path: str, output_dir: str, model_dir: str):
    logger.info(f"Loading raw data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return

    # Standardize Column Names
    rename_map = {
        'Nitrogen': 'N', 'Phosphorous': 'P', 'Potassium': 'K',
        'Temparature': 'temperature', 'Humidity': 'humidity',
        'Moisture': 'moisture', 'Soil Type': 'soil_type',
        'Crop Type': 'crop_type', 'Fertilizer Name': 'fertilizer_name'
    }
    df = df.rename(columns=rename_map)
    
    # --- PIPELINE 1: CROP RECOMMENDATION ---
    logger.info("--- Processing for CROP Recommendation ---")
    target_crop = 'crop_type'
    if target_crop in df.columns:
        # Features: N, P, K, Temp, Humidity, Moisture, Soil Type
        # Exclude: fertilizer_name (Target Leakage/Irrelevant), crop_type (Target)
        
        X_crop = df.drop(columns=[target_crop, 'fertilizer_name'], errors='ignore')
        y_crop = df[target_crop]
        
        num_feats = ['N', 'P', 'K', 'temperature', 'humidity', 'moisture']
        cat_feats = ['soil_type']
        
        # Verify columns exist
        num_feats = [c for c in num_feats if c in X_crop.columns]
        cat_feats = [c for c in cat_feats if c in X_crop.columns]
        
        pipeline_crop = create_training_pipeline(num_feats, cat_feats)
        X_crop_processed = pipeline_crop.fit_transform(X_crop)
        X_crop_processed['label'] = y_crop # Standardize target name for training script
        
        # Save
        crop_out = os.path.join(output_dir, 'crop_recommendation.csv')
        X_crop_processed.to_csv(crop_out, index=False)
        save_pipeline(pipeline_crop, os.path.join(model_dir, 'pipeline_crop.joblib'))
    else:
        logger.warning("Crop Type column not found, skipping Crop pipeline.")

    # --- PIPELINE 2: FERTILIZER RECOMMENDATION ---
    logger.info("--- Processing for FERTILIZER Recommendation ---")
    target_fert = 'fertilizer_name'
    if target_fert in df.columns:
        # Features: N, P, K, Temp, Humidity, Moisture, Soil Type, Crop Type
        # Exclude: fertilizer_name (Target)
        
        X_fert = df.drop(columns=[target_fert], errors='ignore')
        y_fert = df[target_fert]
        
        num_feats = ['N', 'P', 'K', 'temperature', 'humidity', 'moisture']
        cat_feats = ['soil_type', 'crop_type'] # Crop Type is input here
        
        # Verify
        num_feats = [c for c in num_feats if c in X_fert.columns]
        cat_feats = [c for c in cat_feats if c in X_fert.columns]
        
        pipeline_fert = create_training_pipeline(num_feats, cat_feats)
        X_fert_processed = pipeline_fert.fit_transform(X_fert)
        X_fert_processed['label'] = y_fert
        
        # Save
        fert_out = os.path.join(output_dir, 'fertilizer_recommendation.csv')
        X_fert_processed.to_csv(fert_out, index=False)
        save_pipeline(pipeline_fert, os.path.join(model_dir, 'pipeline_fertilizer.joblib'))
    else:
        logger.warning("Fertilizer Name column not found, skipping Fertilizer pipeline.")

if __name__ == "__main__":
    test_pipeline_logic()
    
    base_dir = os.path.dirname(__file__)
    # Corrected path assumption assuming script is run from root or ml/
    # If run from root (python ml/preprocessing.py), __file__ is ml/preprocessing.py. 
    # dirname is ml. ../data/raw relative to ml is correct.
    raw_path = os.path.join(base_dir, '../data/raw/crop_data.csv')
    proc_dir = os.path.join(base_dir, '../data/processed')
    model_dir = os.path.join(base_dir, '../models')
    
    if os.path.exists(raw_path):
        process_full_dataset(raw_path, proc_dir, model_dir)
