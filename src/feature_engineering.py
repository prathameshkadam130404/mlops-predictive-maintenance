"""
Feature Engineering Pipeline
============================

Transforms raw sensor time-series into tabular features for tree-based models.

Core component: `FeaturePipeline` class — a serializable pipeline used in both
training (`fit_transform`) and inference (`transform`) to ensure identical
transformations.

Feature categories:
    1. RUL Labeling:       Piecewise-linear remaining useful life (capped)
    2. Sensor Filtering:   Drop near-constant sensors
    3. Normalization:      MinMaxScaler fit on train, transform on both
    4. Rolling Statistics:  Per-engine rolling mean/std/min/max
    5. Lag Features:       Per-engine autoregressive lag values
    6. Difference Features: First-order differences
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual Feature Functions
# ---------------------------------------------------------------------------


def compute_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """Compute piecewise-linear Remaining Useful Life for each engine.

    RUL at cycle t = max_cycle - t, capped at `rul_cap`.

    Args:
        df: DataFrame with 'unit_number' and 'time_cycles' columns.
        rul_cap: Maximum RUL value (default 125).

    Returns:
        DataFrame with 'rul' column added.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_number")["time_cycles"].transform("max")
    df["rul"] = max_cycles - df["time_cycles"]
    df["rul"] = df["rul"].clip(upper=rul_cap)
    return df


def drop_constant_sensors(
    df: pd.DataFrame, sensors_to_drop: list[int]
) -> pd.DataFrame:
    """Remove sensors with near-constant variance.

    Sensors 1, 5, 10, 16, 18, 19 in C-MAPSS FD001 have near-zero variance.

    Args:
        df: Sensor DataFrame.
        sensors_to_drop: List of sensor indices to remove.

    Returns:
        DataFrame with specified sensor columns removed.
    """
    cols_to_drop = [f"s_{i}" for i in sensors_to_drop if f"s_{i}" in df.columns]
    if cols_to_drop:
        logger.info("Dropping %d near-constant sensors: %s", len(cols_to_drop), cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_sizes: list[int],
) -> pd.DataFrame:
    """Compute per-engine rolling window statistics for sensor readings.

    For each sensor × window size, computes rolling mean, std, min, max.
    Grouped by `unit_number` to prevent cross-engine leakage.
    Uses `min_periods=1` for early cycles with insufficient history.

    Args:
        df: DataFrame with sensor columns and 'unit_number'.
        sensor_cols: List of sensor column names.
        window_sizes: List of window sizes (in cycles).

    Returns:
        DataFrame with added rolling feature columns.
    """
    df = df.copy()

    for window in window_sizes:
        logger.info("Computing rolling features with window=%d for %d sensors", window, len(sensor_cols))
        for sensor in sensor_cols:
            grouped = df.groupby("unit_number")[sensor]

            df[f"{sensor}_roll_mean_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(window=w, min_periods=1).mean()
            )
            df[f"{sensor}_roll_std_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(window=w, min_periods=1).std().fillna(0)
            )
            df[f"{sensor}_roll_min_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(window=w, min_periods=1).min()
            )
            df[f"{sensor}_roll_max_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(window=w, min_periods=1).max()
            )

    return df


def add_lag_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    lag_steps: list[int],
) -> pd.DataFrame:
    """Create per-engine lag features.

    Forward-fills NaN values at the start of each engine's trajectory.

    Args:
        df: DataFrame with sensor columns and 'unit_number'.
        sensor_cols: List of sensor column names.
        lag_steps: List of lag step sizes.

    Returns:
        DataFrame with added lag feature columns.
    """
    df = df.copy()

    for lag in lag_steps:
        for sensor in sensor_cols:
            df[f"{sensor}_lag_{lag}"] = df.groupby("unit_number")[sensor].shift(lag)

    # Forward-fill NaN from initial lags (first few cycles of each engine)
    lag_cols = [c for c in df.columns if "_lag_" in c]
    df[lag_cols] = df.groupby("unit_number")[lag_cols].transform(
        lambda x: x.fillna(method="bfill")
    )
    # If still NaN (shouldn't happen but safety net), fill with column mean
    df[lag_cols] = df[lag_cols].fillna(df[lag_cols].mean())

    return df


def add_diff_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
) -> pd.DataFrame:
    """Compute first-order difference features.

    Captures the rate of change per sensor per engine.

    Args:
        df: DataFrame with sensor columns and 'unit_number'.
        sensor_cols: List of sensor column names.

    Returns:
        DataFrame with added difference feature columns.
    """
    df = df.copy()

    for sensor in sensor_cols:
        df[f"{sensor}_diff"] = df.groupby("unit_number")[sensor].diff().fillna(0)

    return df


# ---------------------------------------------------------------------------
# FeaturePipeline Class
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """Unified feature engineering pipeline for training and inference.

    Ensures identical transformations at training and serving time by
    serializing the fitted scaler, feature list, and config.

    Attributes:
        config: Feature engineering configuration from params.yaml.
        scaler: Fitted MinMaxScaler (None before fit).
        feature_columns: Ordered list of feature column names.
        sensor_cols: Active sensor column names.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the feature pipeline with configuration.

        Args:
            config: Feature engineering config dict from params.yaml.
        """
        self.config = config
        self.scaler: MinMaxScaler | None = None
        self.feature_columns: list[str] | None = None
        self.sensor_cols: list[str] | None = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers on training data and apply all transformations.

        This method is called ONCE during training. It:
            1. Drops constant sensors
            2. Identifies active sensor columns
            3. Fits the MinMaxScaler on training sensor values
            4. Applies normalization
            5. Computes rolling, lag, and difference features
            6. Records the final feature column list

        Args:
            df: Raw training DataFrame with 'unit_number', 'time_cycles',
                settings, and sensor columns.

        Returns:
            Transformed DataFrame ready for model training.
        """
        logger.info("FeaturePipeline.fit_transform() — fitting on %d rows", len(df))

        # Step 1: Compute RUL target
        df = compute_rul(df, rul_cap=self.config.get("rul_cap", 125))

        # Step 2: Drop constant sensors
        drop_sensors = self.config.get("drop_sensors", [])
        df = drop_constant_sensors(df, drop_sensors)

        # Step 3: Identify active sensor columns
        self.sensor_cols = [c for c in df.columns if c.startswith("s_")]
        logger.info("Active sensors after filtering: %d", len(self.sensor_cols))

        # Step 4: Fit and apply normalization
        self.scaler = MinMaxScaler()
        df[self.sensor_cols] = self.scaler.fit_transform(df[self.sensor_cols])

        # Step 5: Compute rolling features
        window_sizes = self.config.get("rolling_window_sizes", [5, 10, 20])
        df = add_rolling_features(df, self.sensor_cols, window_sizes)

        # Step 6: Compute lag features
        lag_steps = self.config.get("lag_steps", [1, 3, 5])
        df = add_lag_features(df, self.sensor_cols, lag_steps)

        # Step 7: Compute difference features
        df = add_diff_features(df, self.sensor_cols)

        # Step 8: Drop non-feature columns and record feature list
        non_feature_cols = ["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
        feature_cols = [c for c in df.columns if c not in non_feature_cols and c != "rul"]
        self.feature_columns = sorted(feature_cols)

        logger.info(
            "Feature engineering complete: %d features, %d rows",
            len(self.feature_columns),
            len(df),
        )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to new data (test or inference).

        Uses the scaler and configuration from `fit_transform()`.
        Does not re-fit.

        Args:
            df: New DataFrame to transform (test data or API input).

        Returns:
            Transformed DataFrame with the same feature columns as training.

        Raises:
            RuntimeError: If called before fit_transform() or load().
        """
        if self.scaler is None or self.sensor_cols is None:
            raise RuntimeError(
                "FeaturePipeline has not been fitted. "
                "Call fit_transform() first, or load a saved pipeline."
            )

        logger.info("FeaturePipeline.transform() — transforming %d rows", len(df))

        # Drop constant sensors (same ones as training)
        drop_sensors = self.config.get("drop_sensors", [])
        df = drop_constant_sensors(df, drop_sensors)

        # Apply fitted scaler (NOT re-fitting)
        available_sensors = [c for c in self.sensor_cols if c in df.columns]
        df[available_sensors] = self.scaler.transform(df[available_sensors])

        # Compute rolling features (same window sizes as training)
        window_sizes = self.config.get("rolling_window_sizes", [5, 10, 20])
        df = add_rolling_features(df, available_sensors, window_sizes)

        # Compute lag features (same lag steps as training)
        lag_steps = self.config.get("lag_steps", [1, 3, 5])
        df = add_lag_features(df, available_sensors, lag_steps)

        # Compute difference features
        df = add_diff_features(df, available_sensors)

        return df

    def save(self, path: str | Path) -> None:
        """Serialize the fitted pipeline to disk.

        Args:
            path: Filepath to save the pipeline artifact.
        """
        if self.scaler is None:
            raise RuntimeError("Cannot save unfitted pipeline.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "sensor_cols": self.sensor_cols,
            "config": self.config,
        }
        joblib.dump(artifact, path)
        logger.info("FeaturePipeline saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> FeaturePipeline:
        """Load a fitted pipeline from disk.

        Args:
            path: Filepath to the saved pipeline artifact.

        Returns:
            Fitted FeaturePipeline instance.

        Raises:
            FileNotFoundError: If the artifact doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found: {path}")

        artifact = joblib.load(path)
        pipeline = cls(config=artifact["config"])
        pipeline.scaler = artifact["scaler"]
        pipeline.feature_columns = artifact["feature_columns"]
        pipeline.sensor_cols = artifact["sensor_cols"]

        logger.info(
            "FeaturePipeline loaded from %s (%d features)",
            path,
            len(pipeline.feature_columns) if pipeline.feature_columns else 0,
        )
        return pipeline

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the final feature matrix from a transformed DataFrame.

        Selects only the feature columns in the exact order established
        during training. This ensures model input consistency.

        Args:
            df: Transformed DataFrame (output of fit_transform or transform).

        Returns:
            DataFrame with only feature columns in canonical order.

        Raises:
            RuntimeError: If feature_columns is not set.
        """
        if self.feature_columns is None:
            raise RuntimeError("Feature columns not established. Fit the pipeline first.")

        # Select available columns (handles edge case where some features
        # might be missing from very short inference sequences)
        available = [c for c in self.feature_columns if c in df.columns]
        missing = set(self.feature_columns) - set(available)
        if missing:
            logger.warning(
                "Missing %d features during extraction: %s. Filling with 0.",
                len(missing),
                list(missing)[:5],
            )
            for col in missing:
                df[col] = 0.0

        return df[self.feature_columns]


# ---------------------------------------------------------------------------
# CLI Entrypoint — DVC Pipeline Stage
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the feature engineering DVC stage.

    Loads processed data, applies the full FeaturePipeline, and saves
    the engineered features and fitted pipeline artifact.
    """
    parser = argparse.ArgumentParser(
        description="Engineer features from processed C-MAPSS sensor data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/params.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    feature_config = config["features"]
    processed_dir = Path(data_config["processed_dir"])
    features_dir = Path(data_config["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING STAGE")
    logger.info("=" * 60)

    # Load processed data
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    logger.info("Train data: %d rows, %d columns", *train_df.shape)
    logger.info("Test data: %d rows, %d columns", *test_df.shape)

    # Initialize and fit pipeline on training data
    pipeline = FeaturePipeline(feature_config)
    train_transformed = pipeline.fit_transform(train_df)

    # Transform test data using fitted pipeline
    test_transformed = pipeline.transform(test_df)

    # Extract feature matrices
    x_train = pipeline.get_feature_matrix(train_transformed)
    y_train = train_transformed["rul"]
    train_features = pd.concat([x_train, y_train], axis=1)

    x_test = pipeline.get_feature_matrix(test_transformed)
    test_features = x_test  # No RUL in test features (ground truth is separate)

    # Save feature matrices
    train_features.to_parquet(features_dir / "train_features.parquet", index=False)
    test_features.to_parquet(features_dir / "test_features.parquet", index=False)

    # Save fitted pipeline artifact
    pipeline.save("models/feature_pipeline.joblib")

    logger.info(
        "Feature engineering complete: %d train features, %d test features",
        len(x_train.columns),
        len(x_test.columns),
    )


if __name__ == "__main__":
    main()
