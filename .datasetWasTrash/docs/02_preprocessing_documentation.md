# Developer Documentation: Preprocessing & Feature Engineering Module

## Executive Summary

The `preprocessing.py` module serves as the primary data transformation engine for the AgroSense ML pipeline. It is architected to ensure data integrity, consistent feature generation, and reproducibility across training and inference environments.

**Design Philosophy**:

* **Production-First**: Prioritizes robustness (e.g., outlier clipping) over raw performance.
* **Configurable**: Externalizes domain-specific constraints to support multi-regional deployments.
* **Scientifically Grounded**: Implements agronomic principles (e.g., Liebig's Law) directly into the feature engineering layer.

---

## Technical Stack

* **Orchestration**: `scikit-learn` Pipeline & ColumnTransformer
* **Imputation Strategy**: `KNNImputer` (k=5) to preserve multivariate correlations in soil data.
* **Normalization**: `PowerTransformer` (Yeo-Johnson) to handle skewed distributions (e.g., Rainfall) and approximate Gaussian input.
* **Persistence**: `joblib` for full pipeline serialization.
* **Configuration**: JSON-based constraint loading.

---

## Architectural Components

### 1. Data Sanitization (`DataCleaner`)

**Responsibility**: Enforce data quality before it enters the ML stream.

* **Deduplication**: Eliminates redundant rows to prevent data leakage between train/test splits.
* **Constraint Enforcement**: Validates physical ranges against `config/constraints.json`. Values exceeding defined thresholds (e.g., pH > 14) are clipped rather than dropped, preserving the sample while mitigating sensor errors.

### 2. Feature Engineering (`FeatureEngineer`)

**Responsibility**: Transform raw metrics into domain-relevant features.

* **Nutrient Limiting Factor**:
  * *Implementation*: `np.min(normalized_nutrients)`
  * *Rationale*: Adheres to Liebig's Law of the Minimum, correcting previous models that incorrectly assumed nutrient compensation (weighted sums).
* **Climate Stress Index**:
  * *Implementation*: `(Temperature * Humidity) / 100`
  * *Rationale*: Models the synergistic effect of heat and moisture on pathogen growth.

### 3. Pipeline Construction (`create_training_pipeline`)

**Responsibility**: Assembly and specific handling of data types.

* **Numeric Pipeline**: `KNNImputer` $\rightarrow$ `StandardScaler`.
* **Categorical Pipeline**: `SimpleImputer` (Mode) $\rightarrow$ `OneHotEncoder`.
* **Output**: Configured to return Pandas DataFrames for improved debuggability during development.

---

## Implementation Details & Constraints

### Data Ingestion Mapping

The raw dataset (`crop_data.csv`) utilizes non-standard column headers (e.g., `Temparature`, `Nitrogen`).

* **Solution**: A strict remapping layer in `process_dataset()` standardizes these to the internal schema (`temperature`, `N`, `P`, `K`) prior to pipeline entry.
* **Maintenance Note**: Changes to upstream data schemas require updates to the `rename_map` dictionary.

### Dynamic Configuration

Physiological constraints are defined in `config/constraints.json`.

* **Default Profile**: Based on standard agricultural ranges (e.g., N max 300).
* **Extensibility**: New profiles (e.g., `volcanic_soil`) can be added to the JSON configuration without code modification, facilitating region-specific model tuning.

---

## Operations Guide

**Testing**:
Execute the module directly to run embedded unit tests:

```bash
python notebooks/preprocessing.py
```

*Verification: Ensure `[PASS]` is logged to the console.*

**Execution**:
Running the script triggers the batch processing workflow:

1. Loads `data/raw/crop_data.csv`.
2. Applies the transformation pipeline.
3. Artifact Generation:
    * **Processed Data**: `data/processed/processed_crop_data.csv`
    * **Serialized Model**: `models/preprocessing_pipeline.joblib`

---

## Technical Debt & Optimization Roadmap

1. **Interaction Features**: The current engineering phase focuses on primary interactions. Explicitly modeling secondary interactions (e.g., `pH * Phosphorus` availability curves) could further improve model performance.
