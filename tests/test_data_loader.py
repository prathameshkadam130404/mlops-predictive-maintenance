"""Tests for the C-MAPSS data loader module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    ALL_COLUMNS,
    EXPECTED_COLUMN_COUNT,
    SENSOR_COLUMNS,
    _validate_raw_dataframe,
    load_cmapss_train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raw_data(tmp_path: Path) -> Path:
    """Create a minimal valid C-MAPSS format training file."""
    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    # 3 engines, varying cycle lengths
    rows = []
    for unit in [1, 2, 3]:
        n_cycles = 10 + unit * 5
        for cycle in range(1, n_cycles + 1):
            row = [unit, cycle]
            row += [0.001, 0.002, 100.0]  # 3 settings
            row += [500.0 + i * 10 for i in range(21)]  # 21 sensors
            rows.append(row)

    df = pd.DataFrame(rows)
    filepath = data_dir / "train_FD001.txt"
    df.to_csv(filepath, sep=" ", header=False, index=False)

    # Also create RUL file
    rul_df = pd.DataFrame({"rul": [50, 60, 70]})
    rul_filepath = data_dir / "RUL_FD001.txt"
    rul_df.to_csv(rul_filepath, sep=" ", header=False, index=False)

    # Test data file
    test_filepath = data_dir / "test_FD001.txt"
    test_rows = rows[:30]  # Use subset as test
    pd.DataFrame(test_rows).to_csv(test_filepath, sep=" ", header=False, index=False)

    return data_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadCmapssTrain:
    """Tests for load_cmapss_train function."""

    def test_loads_correct_columns(self, sample_raw_data: Path) -> None:
        """Verify all 26 expected columns are present and correctly named."""
        df = load_cmapss_train("FD001", sample_raw_data)
        assert list(df.columns) == ALL_COLUMNS
        assert len(df.columns) == EXPECTED_COLUMN_COUNT

    def test_no_nan_values(self, sample_raw_data: Path) -> None:
        """Verify no NaN values exist in loaded data."""
        df = load_cmapss_train("FD001", sample_raw_data)
        assert df.isna().sum().sum() == 0

    def test_correct_engine_count(self, sample_raw_data: Path) -> None:
        """Verify correct number of unique engines."""
        df = load_cmapss_train("FD001", sample_raw_data)
        assert df["unit_number"].nunique() == 3

    def test_positive_time_cycles(self, sample_raw_data: Path) -> None:
        """Verify all time_cycles are positive."""
        df = load_cmapss_train("FD001", sample_raw_data)
        assert (df["time_cycles"] >= 1).all()

    def test_positive_unit_numbers(self, sample_raw_data: Path) -> None:
        """Verify all unit_numbers are positive."""
        df = load_cmapss_train("FD001", sample_raw_data)
        assert (df["unit_number"] >= 1).all()

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for missing data files."""
        with pytest.raises(FileNotFoundError, match="Training data file not found"):
            load_cmapss_train("FD001", tmp_path)

    def test_sensor_columns_are_numeric(self, sample_raw_data: Path) -> None:
        """Verify all sensor columns are numeric (float)."""
        df = load_cmapss_train("FD001", sample_raw_data)
        for col in SENSOR_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


class TestValidateRawDataframe:
    """Tests for _validate_raw_dataframe."""

    def test_passes_valid_data(self, sample_raw_data: Path) -> None:
        """Valid data should pass without errors."""
        df = load_cmapss_train("FD001", sample_raw_data)
        _validate_raw_dataframe(df, Path("test.txt"))  # Should not raise

    def test_fails_on_wrong_column_count(self) -> None:
        """Data with wrong column count should raise ValueError."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Column count mismatch"):
            _validate_raw_dataframe(df, Path("test.txt"))

    def test_fails_on_nan_values(self, sample_raw_data: Path) -> None:
        """Data with NaN values should raise ValueError."""
        df = load_cmapss_train("FD001", sample_raw_data)
        df.loc[0, "s_1"] = np.nan
        with pytest.raises(ValueError, match="NaN values"):
            _validate_raw_dataframe(df, Path("test.txt"))

    def test_fails_on_negative_unit_number(self, sample_raw_data: Path) -> None:
        """Negative unit_number should raise ValueError."""
        df = load_cmapss_train("FD001", sample_raw_data)
        df.loc[0, "unit_number"] = 0
        with pytest.raises(ValueError, match="Invalid unit_number"):
            _validate_raw_dataframe(df, Path("test.txt"))
