# %% [markdown]
# # Multi-Model Training Pipeline
# This script trains multiple models using the `Trainer` class and logs results to MLflow.
# It is designed to be run interactively (VS Code / Jupyter) or as a script.

# %%
import sys
import os

# Ensure the project root is in sys.path
# Assuming script is run from project root or scripts/ dir
current_dir = os.getcwd()
if "scripts" in current_dir:
    project_root = os.path.dirname(current_dir)
    os.chdir(project_root)
    sys.path.append(project_root)
else:
    sys.path.append(current_dir)

print(f"Working Directory: {os.getcwd()}")

# %%
from backend.training.trainer import Trainer
from backend.models.config import MODELS

# %% [markdown]
# ## Initialize Trainer
# This handles data loading and MLflow experiment setup.

# %%
trainer = Trainer(experiment_name="AgroSense_Crop_Recommendation")

# %% [markdown]
# ## Select Models to Train
# You can define a list of model keys from `src/models/config.py`.
# Available: "rf", "xgb", "svm", "catboost", "lr", "ensemble", "stacking"

# %%
models_to_train = ["rf", "xgb", "svm", "catboost", "lr", "ensemble", "stacking"]
# models_to_train = ["rf"] # Uncomment for quick test

# %% [markdown]
# ## Run Experiments
# Train all selected models and collect metric results.
# Set `tune=True` to enable hyperparameter optimization via RandomizedSearchCV.

# %%
results = trainer.run_experiment(models_to_train, tune=True)

# %% [markdown]
# ## Summary of Results

# %%
print("\n--- Training Summary ---")
for model, metrics in results.items():
    print(f"\nModel: {model}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# %% [markdown]
# ## Next Steps
# - Check MLflow UI (`mlflow ui`) to see detailed logs and artifacts.
# - The best model can be deployed using `CropPredictor`.
