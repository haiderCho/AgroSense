# %% [markdown]
# # Crop Data Preprocessing Pipeline
#
# Production-ready preprocessing module for ML model training.
# Transforms `data/raw/Crop_recommendation.csv` → ML-ready artifacts.
#
# **Features:**
# - Missing value imputation
# - Outlier detection & treatment (IQR-based)
# - Feature scaling (StandardScaler)
# - Label encoding
# - Stratified train/val/test split
# - Artifact persistence (.joblib)

# %%
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# %%
@dataclass
class PreprocessingConfig:
    """
    Configuration for the preprocessing pipeline.

    Attributes:
        input_path: Path to raw CSV data.
        output_dir: Directory for processed artifacts.
        target_column: Name of the target/label column.
        val_size: Fraction of data for validation set (0.0 - 1.0).
        test_size: Fraction of data for test set (0.0 - 1.0).
        random_state: Random seed for reproducibility.
        scaler_type: Type of feature scaler ('standard', 'minmax', 'robust').
        outlier_method: Outlier handling ('clip', 'remove', None).
        outlier_threshold: IQR multiplier for outlier detection.
        missing_strategy: Strategy for missing values ('median', 'mean', 'drop').
    """

    input_path: str = "data/raw/Crop_recommendation.csv"
    output_dir: str = "data/processed"
    target_column: str = "label"
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    scaler_type: Literal["standard", "minmax", "robust"] = "standard"
    outlier_method: Literal["clip", "remove"] | None = "clip"
    outlier_threshold: float = 1.5
    missing_strategy: Literal["median", "mean", "drop"] = "median"


# %%
class CropPreprocessor:
    """
    End-to-end preprocessing pipeline for the Crop Recommendation dataset.

    Handles data loading, validation, cleaning, scaling, encoding,
    and train/val/test splitting with full artifact persistence.

    Example:
        >>> config = PreprocessingConfig()
        >>> preprocessor = CropPreprocessor(config)
        >>> result = preprocessor.run_pipeline()
        >>> print(result['X_train'].shape)
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None
        self.label_encoder: LabelEncoder | None = None
        self.feature_columns: list[str] = []
        self._report: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Data Loading & Validation
    # -------------------------------------------------------------------------
    def load_data(self, path: str | None = None) -> pd.DataFrame:
        """
        Load CSV data with validation.

        Args:
            path: Optional override for input path.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the input file doesn't exist.
            ValueError: If required columns are missing.
        """
        filepath = Path(path or self.config.input_path)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

        # Validate target column exists
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found")

        # Store feature columns
        self.feature_columns = [
            col for col in df.columns if col != self.config.target_column
        ]

        self._report["input_shape"] = list(df.shape)
        self._report["input_path"] = str(filepath.absolute())

        return df

    # -------------------------------------------------------------------------
    # Missing Value Handling
    # -------------------------------------------------------------------------
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on configured strategy.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        missing_before = df.isnull().sum().sum()

        if missing_before == 0:
            logger.info("No missing values detected")
            self._report["missing_values"] = {"before": 0, "after": 0, "strategy": None}
            return df

        strategy = self.config.missing_strategy
        logger.info(f"Handling {missing_before} missing values with '{strategy}' strategy")

        if strategy == "drop":
            df = df.dropna()
        elif strategy in ("mean", "median"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    fill_value = df[col].mean() if strategy == "mean" else df[col].median()
                    df[col] = df[col].fillna(fill_value)

        missing_after = df.isnull().sum().sum()
        self._report["missing_values"] = {
            "before": int(missing_before),
            "after": int(missing_after),
            "strategy": strategy,
        }

        return df

    # -------------------------------------------------------------------------
    # Outlier Detection & Treatment
    # -------------------------------------------------------------------------
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using IQR method.

        Returns:
            Boolean DataFrame marking outlier positions.
        """
        numeric_cols = df[self.feature_columns].select_dtypes(include=[np.number]).columns
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.config.outlier_threshold * IQR
            upper = Q3 + self.config.outlier_threshold * IQR
            outlier_mask[col] = (df[col] < lower) | (df[col] > upper)

        return outlier_mask

    def treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Treat outliers based on configured method.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with outliers treated.
        """
        if self.config.outlier_method is None:
            logger.info("Outlier treatment disabled")
            self._report["outliers"] = {"method": None, "affected_rows": 0}
            return df

        outlier_mask = self.detect_outliers(df)
        affected_rows = outlier_mask.any(axis=1).sum()
        logger.info(f"Detected {affected_rows} rows with outliers")

        method = self.config.outlier_method
        numeric_cols = outlier_mask.columns.tolist()

        if method == "remove":
            df = df[~outlier_mask.any(axis=1)].reset_index(drop=True)
            logger.info(f"Removed {affected_rows} outlier rows")
        elif method == "clip":
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.config.outlier_threshold * IQR
                upper = Q3 + self.config.outlier_threshold * IQR
                df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(f"Clipped outliers in {len(numeric_cols)} columns")

        self._report["outliers"] = {
            "method": method,
            "affected_rows": int(affected_rows),
            "threshold": self.config.outlier_threshold,
        }

        return df

    # -------------------------------------------------------------------------
    # Feature Scaling
    # -------------------------------------------------------------------------
    def scale_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame.
            fit: If True, fit the scaler. If False, use pre-fitted scaler.

        Returns:
            DataFrame with scaled features.
        """
        numeric_cols = df[self.feature_columns].select_dtypes(include=[np.number]).columns.tolist()

        if fit:
            scaler_map = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
                "robust": RobustScaler,
            }
            self.scaler = scaler_map[self.config.scaler_type]()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logger.info(f"Fitted {self.config.scaler_type} scaler on {len(numeric_cols)} features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call scale_features with fit=True first.")
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            logger.info(f"Transformed {len(numeric_cols)} features with pre-fitted scaler")

        self._report["scaling"] = {
            "type": self.config.scaler_type,
            "columns": numeric_cols,
        }

        return df

    # -------------------------------------------------------------------------
    # Target Encoding
    # -------------------------------------------------------------------------
    def encode_target(self, series: pd.Series, fit: bool = True) -> pd.Series:
        """
        Encode categorical target labels.

        Args:
            series: Target series.
            fit: If True, fit the encoder.

        Returns:
            Encoded target series.
        """
        if fit:
            self.label_encoder = LabelEncoder()
            encoded = pd.Series(
                self.label_encoder.fit_transform(series),
                index=series.index,
                name=series.name,
            )
            classes = self.label_encoder.classes_.tolist()
            logger.info(f"Encoded {len(classes)} target classes: {classes[:5]}...")
            self._report["encoding"] = {"classes": classes, "n_classes": len(classes)}
        else:
            if self.label_encoder is None:
                raise ValueError("LabelEncoder not fitted.")
            encoded = pd.Series(
                self.label_encoder.transform(series),
                index=series.index,
                name=series.name,
            )

        return encoded

    # -------------------------------------------------------------------------
    # Train/Val/Test Split
    # -------------------------------------------------------------------------
    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets with stratification.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Second split: separate validation from training
        # Adjust val_size relative to remaining data
        val_relative = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_relative,
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        logger.info(
            f"Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}"
        )

        self._report["split"] = {
            "val_size": self.config.val_size,
            "test_size": self.config.test_size,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "stratified": True,
        }

        return X_train, X_val, X_test, y_train, y_val, y_test

    # -------------------------------------------------------------------------
    # Artifact Persistence
    # -------------------------------------------------------------------------
    def save_artifacts(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> Path:
        """
        Save all processed artifacts to disk.

        Returns:
            Path to output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save data splits
        X_train.to_csv(output_dir / "X_train.csv", index=False)
        X_val.to_csv(output_dir / "X_val.csv", index=False)
        X_test.to_csv(output_dir / "X_test.csv", index=False)
        y_train.to_csv(output_dir / "y_train.csv", index=False)
        y_val.to_csv(output_dir / "y_val.csv", index=False)
        y_test.to_csv(output_dir / "y_test.csv", index=False)

        # Save fitted transformers
        if self.scaler:
            joblib.dump(self.scaler, output_dir / "scaler.joblib")
        if self.label_encoder:
            joblib.dump(self.label_encoder, output_dir / "label_encoder.joblib")

        # Save preprocessing report
        self._report["timestamp"] = datetime.now().isoformat()
        self._report["config"] = {
            "val_size": self.config.val_size,
            "test_size": self.config.test_size,
            "random_state": self.config.random_state,
            "scaler_type": self.config.scaler_type,
            "outlier_method": self.config.outlier_method,
            "missing_strategy": self.config.missing_strategy,
        }

        with open(output_dir / "preprocessing_report.json", "w") as f:
            json.dump(self._report, f, indent=2)

        logger.info(f"Saved artifacts to: {output_dir.absolute()}")

        return output_dir

    # -------------------------------------------------------------------------
    # Full Pipeline Execution
    # -------------------------------------------------------------------------
    def run_pipeline(self) -> dict[str, Any]:
        """
        Execute the complete preprocessing pipeline.

        Returns:
            Dictionary containing processed data and metadata.
        """
        logger.info("=" * 60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load data
        df = self.load_data()

        # Step 2: Handle missing values
        df = self.handle_missing_values(df)

        # Step 3: Treat outliers
        df = self.treat_outliers(df)

        # Step 4: Split features and target
        X = df[self.feature_columns].copy()
        y = df[self.config.target_column].copy()

        # Step 5: Encode target (before split for consistent encoding)
        y_encoded = self.encode_target(y, fit=True)

        # Step 6: Split data (BEFORE scaling to prevent data leakage)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y_encoded)

        # Step 7: Scale features (fit on train only, transform all)
        X_train = self.scale_features(X_train.copy(), fit=True)
        X_val = self.scale_features(X_val.copy(), fit=False)
        X_test = self.scale_features(X_test.copy(), fit=False)

        # Step 8: Save artifacts
        output_dir = self.save_artifacts(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_columns": self.feature_columns,
            "output_dir": output_dir,
            "report": self._report,
        }


# %%
if __name__ == "__main__":
    config = PreprocessingConfig()
    preprocessor = CropPreprocessor(config)
    result = preprocessor.run_pipeline()

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Train features shape: {result['X_train'].shape}")
    print(f"Val features shape:   {result['X_val'].shape}")
    print(f"Test features shape:  {result['X_test'].shape}")
    print(f"Target classes:       {result['label_encoder'].classes_.tolist()}")
    print(f"Output directory:     {result['output_dir']}")
