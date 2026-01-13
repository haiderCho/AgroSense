# %% [markdown]
# # Crop Data Preprocessing Pipeline
#
# Interactive notebook-style script to preprocess raw data for ML training.
# Run each cell sequentially in VS Code Interactive or export to Jupyter.
#
# **Input:** `data/raw/Crop_recommendation.csv`  
# **Output:** `data/processed/` (train/val/test splits, scalers, encoder)

# %% [markdown]
# ---
# ## Setup & Configuration

# %%
import sys
import os

# Ensure scripts directory is in path for imports
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
    sys.path.insert(0, "scripts")
elif os.path.basename(os.getcwd()) == "AgroSense":
    sys.path.insert(0, "scripts")

from preprocessing import CropPreprocessor, PreprocessingConfig

print("Imports successful!")
print(f"Working directory: {os.getcwd()}")

# %%
# Configure the pipeline
config = PreprocessingConfig(
    input_path="data/raw/Crop_recommendation.csv",
    output_dir="data/processed",
    target_column="label",
    val_size=0.15,                # Validation set for hyperparameter tuning
    test_size=0.15,               # Test set for final evaluation
    random_state=42,
    scaler_type="standard",       # Options: 'standard', 'minmax', 'robust'
    outlier_method="clip",        # Options: 'clip', 'remove', None
    outlier_threshold=1.5,        # IQR multiplier
    missing_strategy="median",    # Options: 'median', 'mean', 'drop'
)

print("Configuration:")
print(f"  Input:           {config.input_path}")
print(f"  Output:          {config.output_dir}")
print(f"  Val size:        {config.val_size} ({config.val_size*100:.0f}%)")
print(f"  Test size:       {config.test_size} ({config.test_size*100:.0f}%)")
print(f"  Train size:      {1-config.val_size-config.test_size:.2f} ({(1-config.val_size-config.test_size)*100:.0f}%)")
print(f"  Scaler:          {config.scaler_type}")
print(f"  Outlier method:  {config.outlier_method}")

# %% [markdown]
# ---
# ## Step 1: Load Raw Data

# %%
preprocessor = CropPreprocessor(config)
df = preprocessor.load_data()

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
df.head()

# %%
print("Data Types:")
print(df.dtypes)
print(f"\nFeature columns: {preprocessor.feature_columns}")

# %% [markdown]
# ---
# ## Step 2: Check & Handle Missing Values

# %%
print("Missing values per column:")
print(df.isnull().sum())

df = preprocessor.handle_missing_values(df)

# %% [markdown]
# ---
# ## Step 3: Outlier Detection & Treatment

# %%
# Detect outliers
outlier_mask = preprocessor.detect_outliers(df)
print("Outliers per column:")
print(outlier_mask.sum())
print(f"\nTotal rows with outliers: {outlier_mask.any(axis=1).sum()}")

# %%
# Treat outliers
df = preprocessor.treat_outliers(df)
print(f"Shape after outlier treatment: {df.shape}")

# %% [markdown]
# ---
# ## Step 4: Feature/Target Split

# %%
X = df[preprocessor.feature_columns].copy()
y = df[config.target_column].copy()

print(f"Features shape: {X.shape}")
print(f"Target shape:   {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts())

# %% [markdown]
# ---
# ## Step 5: Encode Target Labels

# %%
y_encoded = preprocessor.encode_target(y, fit=True)

print(f"Label mapping:")
for label, idx in zip(preprocessor.label_encoder.classes_, range(len(preprocessor.label_encoder.classes_))):
    print(f"  {idx}: {label}")

# %% [markdown]
# ---
# ## Step 6: Train/Val/Test Split (Stratified)

# %%
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y_encoded)

print(f"X_train: {X_train.shape}  ({len(X_train)/len(X)*100:.1f}%)")
print(f"X_val:   {X_val.shape}  ({len(X_val)/len(X)*100:.1f}%)")
print(f"X_test:  {X_test.shape}  ({len(X_test)/len(X)*100:.1f}%)")

# %% [markdown]
# ---
# ## Step 7: Feature Scaling (Fit on Train Only)

# %%
# Fit scaler on training data only (prevents data leakage)
X_train_scaled = preprocessor.scale_features(X_train.copy(), fit=True)

# Transform val and test data with the fitted scaler
X_val_scaled = preprocessor.scale_features(X_val.copy(), fit=False)
X_test_scaled = preprocessor.scale_features(X_test.copy(), fit=False)

print("Scaled training data stats:")
print(X_train_scaled.describe().round(2))

# %% [markdown]
# ---
# ## Step 8: Save Artifacts

# %%
output_dir = preprocessor.save_artifacts(
    X_train_scaled, X_val_scaled, X_test_scaled, 
    y_train, y_val, y_test
)

print(f"\nArtifacts saved to: {output_dir}")
print("\nGenerated files:")
for f in sorted(output_dir.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:30s} ({size_kb:.1f} KB)")

# %% [markdown]
# ---
# ## Summary

# %%
import json

with open(output_dir / "preprocessing_report.json") as f:
    report = json.load(f)

print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"\nInput:  {report['input_shape'][0]} rows × {report['input_shape'][1]} columns")
print(f"\nSplit breakdown:")
print(f"  Train: {report['split']['train_samples']} samples ({report['split']['train_samples']/report['input_shape'][0]*100:.1f}%)")
print(f"  Val:   {report['split']['val_samples']} samples ({report['split']['val_samples']/report['input_shape'][0]*100:.1f}%)")
print(f"  Test:  {report['split']['test_samples']} samples ({report['split']['test_samples']/report['input_shape'][0]*100:.1f}%)")
print(f"\nTarget classes: {report['encoding']['n_classes']}")
print(f"Scaler type:    {report['scaling']['type']}")
print(f"Outliers:       {report['outliers']['affected_rows']} rows affected ({report['outliers']['method']})")
print(f"\nTimestamp: {report['timestamp']}")
