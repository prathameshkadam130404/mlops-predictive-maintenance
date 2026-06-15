"""Tests for the Pandera data validation module."""

import numpy as np
import pandas as pd
import pytest

from src.validate import (
    ValidationResult,
    build_raw_sensor_schema,
    validate_features,
    validate_raw_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_sensor_data() -> pd.DataFrame:
    """Create a minimal valid sensor DataFrame matching C-MAPSS FD001 schema."""
    n_rows = 50
    rng = np.random.default_rng(42)

    data = {
        "unit_number": np.repeat([1, 2], n_rows // 2),
        "time_cycles": np.tile(np.arange(1, n_rows // 2 + 1), 2),
        "setting_1": rng.uniform(-0.01, 0.01, n_rows),
        "setting_2": rng.uniform(-0.01, 0.01, n_rows),
        "setting_3": rng.uniform(99.0, 101.0, n_rows),
    }

    # Add sensor columns with values within physical bounds
    sensor_defaults = {
        "s_1": (480, 5), "s_2": (642, 2), "s_3": (1580, 10), "s_4": (1350, 20),
        "s_5": (14, 2), "s_6": (21, 1), "s_7": (450, 30), "s_8": (2388, 0.5),
        "s_9": (9050, 50), "s_10": (1.3, 0.01), "s_11": (45, 2), "s_12": (460, 30),
        "s_13": (2388, 0.5), "s_14": (8140, 20), "s_15": (8.4, 0.1),
        "s_16": (0.03, 0.005), "s_17": (350, 20), "s_18": (2388, 1),
        "s_19": (100, 0.01), "s_20": (30, 5), "s_21": (23, 0.5),
    }

    for sensor, (mean, std) in sensor_defaults.items():
        data[sensor] = rng.normal(mean, std, n_rows)

    return pd.DataFrame(data)


@pytest.fixture
def valid_feature_data() -> pd.DataFrame:
    """Create a minimal valid feature DataFrame."""
    n_rows = 100
    rng = np.random.default_rng(42)

    data = {"rul": rng.integers(0, 125, n_rows)}
    # Add 50 numeric features
    for i in range(50):
        data[f"feature_{i}"] = rng.normal(0, 1, n_rows)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateRawData:
    """Tests for raw data validation."""

    def test_valid_data_passes(self, valid_sensor_data: pd.DataFrame) -> None:
        """Valid C-MAPSS-like data should pass schema validation."""
        result = validate_raw_data(valid_sensor_data)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_result_contains_metadata(self, valid_sensor_data: pd.DataFrame) -> None:
        """Validation result should contain row/column counts."""
        result = validate_raw_data(valid_sensor_data)
        assert result.n_rows == len(valid_sensor_data)
        assert result.n_columns == len(valid_sensor_data.columns)

    def test_null_values_fail(self, valid_sensor_data: pd.DataFrame) -> None:
        """Data with null values should fail validation."""
        valid_sensor_data.loc[0, "s_3"] = None
        result = validate_raw_data(valid_sensor_data)
        assert result.passed is False

    def test_negative_unit_number_fails(self, valid_sensor_data: pd.DataFrame) -> None:
        """Negative unit_number should fail validation."""
        valid_sensor_data.loc[0, "unit_number"] = -1
        result = validate_raw_data(valid_sensor_data)
        assert result.passed is False

    def test_serialization(self, valid_sensor_data: pd.DataFrame) -> None:
        """ValidationResult should serialize to dict cleanly."""
        result = validate_raw_data(valid_sensor_data)
        result_dict = result.to_dict()
        assert "passed" in result_dict
        assert "n_rows" in result_dict
        assert "errors" in result_dict


class TestValidateFeatures:
    """Tests for feature validation."""

    def test_valid_features_pass(self, valid_feature_data: pd.DataFrame) -> None:
        """Valid feature DataFrame should pass validation."""
        result = validate_features(valid_feature_data)
        assert result.passed is True

    def test_nan_features_fail(self, valid_feature_data: pd.DataFrame) -> None:
        """Features with NaN should fail validation."""
        valid_feature_data.loc[0, "feature_0"] = np.nan
        result = validate_features(valid_feature_data)
        assert result.passed is False
        assert any("NaN" in e["detail"] for e in result.errors)

    def test_negative_rul_fails(self, valid_feature_data: pd.DataFrame) -> None:
        """Negative RUL values should fail validation."""
        valid_feature_data.loc[0, "rul"] = -5
        result = validate_features(valid_feature_data)
        assert result.passed is False

    def test_rul_exceeding_cap_fails(self, valid_feature_data: pd.DataFrame) -> None:
        """RUL values above cap should fail validation."""
        valid_feature_data.loc[0, "rul"] = 200
        result = validate_features(valid_feature_data, rul_cap=125)
        assert result.passed is False

    def test_infinity_fails(self, valid_feature_data: pd.DataFrame) -> None:
        """Infinite values should fail validation."""
        valid_feature_data.loc[0, "feature_0"] = np.inf
        result = validate_features(valid_feature_data)
        assert result.passed is False

    def test_low_feature_count_warns(self) -> None:
        """Very few features should produce a warning."""
        df = pd.DataFrame({"rul": [50, 60], "f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        result = validate_features(df)
        assert len(result.warnings) > 0
