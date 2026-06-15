"""
NASA C-MAPSS Data Loader
========================

Parses the raw NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
space-separated text files into structured Pandas DataFrames.

The C-MAPSS dataset provides run-to-failure multivariate time-series data from
simulated turbofan engine degradation. Each engine (unit) runs from a healthy
state to failure, with 21 sensor readings captured every operational cycle.

Dataset format:
    - Space-separated .txt files, NO headers
    - 26 columns: [unit_number, time_cycles, setting_1..3, s_1..s_21]
    - Training data: full run-to-failure trajectories
    - Test data: truncated trajectories (predict remaining cycles)
    - RUL file: ground-truth remaining useful life for test engines

Reference:
    A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
    NASA Ames Prognostics Data Repository.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column Definitions
# ---------------------------------------------------------------------------
INDEX_COLUMNS: list[str] = ["unit_number", "time_cycles"]
SETTING_COLUMNS: list[str] = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS: list[str] = [f"s_{i}" for i in range(1, 22)]
ALL_COLUMNS: list[str] = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

EXPECTED_COLUMN_COUNT: int = 26


def load_cmapss_train(dataset: str, data_dir: str | Path) -> pd.DataFrame:
    """Load and parse a C-MAPSS training dataset.

    Training files contain full run-to-failure trajectories for each engine unit.
    Each row represents one operational cycle of one engine.

    Args:
        dataset: Sub-dataset identifier (e.g., "FD001", "FD002", "FD003", "FD004").
        data_dir: Path to the directory containing raw .txt files.

    Returns:
        DataFrame with named columns: unit_number, time_cycles, setting_1..3, s_1..s_21.

    Raises:
        FileNotFoundError: If the training file does not exist.
        ValueError: If column count does not match expected 26.
    """
    data_dir = Path(data_dir)
    filepath = data_dir / f"train_{dataset}.txt"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Training data file not found: {filepath}. "
            f"Ensure the NASA C-MAPSS dataset is downloaded to '{data_dir}'."
        )

    logger.info("Loading training data from %s", filepath)

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
        engine="python",
    )

    _validate_raw_dataframe(df, filepath)

    logger.info(
        "Loaded training data: %d rows, %d engines, %d max cycles",
        len(df),
        df["unit_number"].nunique(),
        df["time_cycles"].max(),
    )

    return df


def load_cmapss_test(
    dataset: str, data_dir: str | Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and parse a C-MAPSS test dataset with ground-truth RUL.

    Test files contain truncated trajectories (cut off before failure).
    The corresponding RUL file provides the true remaining cycles for
    each engine at its last observed cycle.

    Args:
        dataset: Sub-dataset identifier (e.g., "FD001").
        data_dir: Path to the directory containing raw .txt files.

    Returns:
        Tuple of:
            - DataFrame with test sensor readings (same schema as training)
            - Series with ground-truth RUL values (one per engine, indexed 1..N)

    Raises:
        FileNotFoundError: If test or RUL files do not exist.
        ValueError: If column count mismatch or RUL count mismatch.
    """
    data_dir = Path(data_dir)
    test_filepath = data_dir / f"test_{dataset}.txt"
    rul_filepath = data_dir / f"RUL_{dataset}.txt"

    if not test_filepath.exists():
        raise FileNotFoundError(f"Test data file not found: {test_filepath}")
    if not rul_filepath.exists():
        raise FileNotFoundError(f"RUL ground-truth file not found: {rul_filepath}")

    # --- Load test sensor data ---
    logger.info("Loading test data from %s", test_filepath)
    df_test = pd.read_csv(
        test_filepath,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
        engine="python",
    )
    _validate_raw_dataframe(df_test, test_filepath)

    # --- Load ground-truth RUL ---
    logger.info("Loading RUL ground truth from %s", rul_filepath)
    rul_series = pd.read_csv(
        rul_filepath,
        sep=r"\s+",
        header=None,
        names=["rul"],
        engine="python",
    )["rul"]

    # Validate RUL count matches number of test engines
    n_test_engines = df_test["unit_number"].nunique()
    if len(rul_series) != n_test_engines:
        raise ValueError(
            f"RUL count mismatch: {len(rul_series)} RUL values "
            f"for {n_test_engines} test engines in {rul_filepath}"
        )

    logger.info(
        "Loaded test data: %d rows, %d engines | RUL range: [%d, %d]",
        len(df_test),
        n_test_engines,
        rul_series.min(),
        rul_series.max(),
    )

    return df_test, rul_series


def _validate_raw_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """Perform basic validation on a raw C-MAPSS DataFrame.

    Checks:
        1. Correct number of columns (26)
        2. No NaN values in raw data (C-MAPSS should have none)
        3. unit_number and time_cycles are positive integers

    Args:
        df: Raw DataFrame to validate.
        filepath: Source filepath (for error messages).

    Raises:
        ValueError: If any validation check fails.
    """
    # Check column count
    if len(df.columns) != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"Column count mismatch in {filepath}: "
            f"expected {EXPECTED_COLUMN_COUNT}, got {len(df.columns)}"
        )

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"Found {nan_count} NaN values in {filepath}, columns: {nan_cols}"
        )

    # Check unit_number validity
    if (df["unit_number"] < 1).any():
        raise ValueError(f"Invalid unit_number values (<1) in {filepath}")

    # Check time_cycles validity
    if (df["time_cycles"] < 1).any():
        raise ValueError(f"Invalid time_cycles values (<1) in {filepath}")


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_rul: pd.Series,
    output_dir: str | Path,
) -> None:
    """Save parsed DataFrames to Parquet format for downstream pipeline stages.

    Parquet is chosen over CSV for:
        - Type preservation (no float→string→float round-trip issues)
        - Compression (smaller files for DVC versioning)
        - Faster read/write for pandas operations

    Args:
        train_df: Parsed training DataFrame.
        test_df: Parsed test DataFrame.
        test_rul: Ground-truth RUL Series.
        output_dir: Directory to save processed files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    rul_path = output_dir / "test_rul.csv"

    train_df.to_parquet(train_path, index=False, engine="pyarrow")
    test_df.to_parquet(test_path, index=False, engine="pyarrow")
    test_rul.to_csv(rul_path, index=False, header=["rul"])

    logger.info(
        "Saved processed data: train=%s (%d rows), test=%s (%d rows), rul=%s (%d values)",
        train_path,
        len(train_df),
        test_path,
        len(test_df),
        rul_path,
        len(test_rul),
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint — Used by DVC pipeline stage
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entrypoint for the data loading DVC stage.

    Reads raw C-MAPSS .txt files and saves them as structured Parquet files.
    Configuration is loaded from the specified YAML config file.
    """
    parser = argparse.ArgumentParser(
        description="Parse raw NASA C-MAPSS data into structured Parquet format."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/params.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    dataset = data_config["dataset"]
    raw_dir = data_config["raw_dir"]
    processed_dir = data_config["processed_dir"]

    logger.info("=" * 60)
    logger.info("DATA LOADING STAGE — Dataset: %s", dataset)
    logger.info("=" * 60)

    # Load raw data
    train_df = load_cmapss_train(dataset, raw_dir)
    test_df, test_rul = load_cmapss_test(dataset, raw_dir)

    # Save as Parquet
    save_processed_data(train_df, test_df, test_rul, processed_dir)

    logger.info("Data loading stage completed successfully.")


if __name__ == "__main__":
    main()
