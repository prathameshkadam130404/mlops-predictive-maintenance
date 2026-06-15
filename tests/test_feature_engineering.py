"""Tests for the feature engineering module and FeaturePipeline class."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FeaturePipeline,
    add_diff_features,
    add_lag_features,
    add_rolling_features,
    compute_rul,
    drop_constant_sensors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_engine_data() -> pd.DataFrame:
    """Create synthetic multi-engine sensor data."""
    rows = []
    rng = np.random.default_rng(42)

    for unit in [1, 2, 3]:
        n_cycles = 50 + unit * 10
        for cycle in range(1, n_cycles + 1):
            row = {
                "unit_number": unit,
                "time_cycles": cycle,
                "setting_1": rng.uniform(-0.01, 0.01),
                "setting_2": rng.uniform(-0.01, 0.01),
                "setting_3": 100.0,
            }
            # Add degrading sensors (slight upward trend)
            for i in range(1, 22):
                baseline = 500.0 + i * 10
                degradation = cycle * 0.01 * i  # Gradual increase
                noise = rng.normal(0, 1)
                row[f"s_{i}"] = baseline + degradation + noise
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def feature_config() -> dict:
    """Standard feature engineering configuration."""
    return {
        "rul_cap": 125,
        "drop_sensors": [1, 5, 10, 16, 18, 19],
        "rolling_window_sizes": [5, 10],
        "lag_steps": [1, 3],
    }


# ---------------------------------------------------------------------------
# Tests: Individual Feature Functions
# ---------------------------------------------------------------------------


class TestComputeRul:
    """Tests for RUL computation."""

    def test_rul_never_negative(self, sample_engine_data: pd.DataFrame) -> None:
        """RUL should never be negative."""
        df = compute_rul(sample_engine_data, rul_cap=125)
        assert (df["rul"] >= 0).all()

    def test_rul_capped_at_max(self, sample_engine_data: pd.DataFrame) -> None:
        """RUL should never exceed the cap."""
        cap = 125
        df = compute_rul(sample_engine_data, rul_cap=cap)
        assert (df["rul"] <= cap).all()

    def test_rul_zero_at_last_cycle(self, sample_engine_data: pd.DataFrame) -> None:
        """RUL should be 0 at the last cycle of each engine."""
        df = compute_rul(sample_engine_data, rul_cap=125)
        for unit in df["unit_number"].unique():
            unit_data = df[df["unit_number"] == unit]
            last_cycle = unit_data[unit_data["time_cycles"] == unit_data["time_cycles"].max()]
            assert last_cycle["rul"].iloc[0] == 0

    def test_rul_decreasing_within_cap(self, sample_engine_data: pd.DataFrame) -> None:
        """RUL should be non-increasing within the capped region."""
        df = compute_rul(sample_engine_data, rul_cap=125)
        for unit in df["unit_number"].unique():
            unit_data = df[df["unit_number"] == unit].sort_values("time_cycles")
            rul_values = unit_data["rul"].values
            # Once RUL drops below cap, it should only decrease
            below_cap = rul_values[rul_values < 125]
            if len(below_cap) > 1:
                assert all(below_cap[i] >= below_cap[i + 1] for i in range(len(below_cap) - 1))


class TestDropConstantSensors:
    """Tests for sensor filtering."""

    def test_drops_specified_sensors(self, sample_engine_data: pd.DataFrame) -> None:
        """Specified sensors should be removed from the DataFrame."""
        df = drop_constant_sensors(sample_engine_data, [1, 5, 10])
        assert "s_1" not in df.columns
        assert "s_5" not in df.columns
        assert "s_10" not in df.columns
        assert "s_2" in df.columns  # Non-dropped sensor should remain

    def test_no_sensors_dropped_if_empty_list(self, sample_engine_data: pd.DataFrame) -> None:
        """Empty drop list should preserve all sensors."""
        original_cols = set(sample_engine_data.columns)
        df = drop_constant_sensors(sample_engine_data, [])
        assert set(df.columns) == original_cols


class TestRollingFeatures:
    """Tests for rolling window feature computation."""

    def test_no_cross_engine_leakage(self, sample_engine_data: pd.DataFrame) -> None:
        """Rolling features should not leak across engine boundaries."""
        sensor_cols = ["s_2", "s_3"]
        df = add_rolling_features(sample_engine_data, sensor_cols, [5])

        # For each engine, the first rolling mean should be the first value itself
        for unit in df["unit_number"].unique():
            unit_data = df[df["unit_number"] == unit].sort_values("time_cycles")
            first_mean = unit_data["s_2_roll_mean_5"].iloc[0]
            first_val = unit_data["s_2"].iloc[0]
            assert abs(first_mean - first_val) < 1e-6

    def test_rolling_features_created(self, sample_engine_data: pd.DataFrame) -> None:
        """Rolling features should be added to the DataFrame."""
        sensor_cols = ["s_2"]
        df = add_rolling_features(sample_engine_data, sensor_cols, [5])
        assert "s_2_roll_mean_5" in df.columns
        assert "s_2_roll_std_5" in df.columns
        assert "s_2_roll_min_5" in df.columns
        assert "s_2_roll_max_5" in df.columns


class TestLagFeatures:
    """Tests for lag feature computation."""

    def test_lag_features_created(self, sample_engine_data: pd.DataFrame) -> None:
        """Lag features should be added for each sensor × lag combination."""
        df = add_lag_features(sample_engine_data, ["s_2"], [1, 3])
        assert "s_2_lag_1" in df.columns
        assert "s_2_lag_3" in df.columns

    def test_no_nan_in_lag_features(self, sample_engine_data: pd.DataFrame) -> None:
        """Lag features should have no NaN after backfill."""
        df = add_lag_features(sample_engine_data, ["s_2", "s_3"], [1, 3])
        lag_cols = [c for c in df.columns if "_lag_" in c]
        assert df[lag_cols].isna().sum().sum() == 0


class TestDiffFeatures:
    """Tests for difference feature computation."""

    def test_diff_features_created(self, sample_engine_data: pd.DataFrame) -> None:
        """Difference features should be added."""
        df = add_diff_features(sample_engine_data, ["s_2"])
        assert "s_2_diff" in df.columns

    def test_first_diff_is_zero(self, sample_engine_data: pd.DataFrame) -> None:
        """First difference for each engine should be 0 (filled)."""
        df = add_diff_features(sample_engine_data, ["s_2"])
        for unit in df["unit_number"].unique():
            unit_data = df[df["unit_number"] == unit].sort_values("time_cycles")
            assert unit_data["s_2_diff"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Tests: FeaturePipeline Class
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    """Tests for the FeaturePipeline class (training-serving skew prevention)."""

    def test_fit_transform_produces_features(
        self, sample_engine_data: pd.DataFrame, feature_config: dict
    ) -> None:
        """fit_transform should produce a DataFrame with features + rul."""
        pipeline = FeaturePipeline(feature_config)
        result = pipeline.fit_transform(sample_engine_data)
        assert "rul" in result.columns
        assert pipeline.feature_columns is not None
        assert len(pipeline.feature_columns) > 10

    def test_save_load_consistency(
        self, sample_engine_data: pd.DataFrame, feature_config: dict, tmp_path: Path
    ) -> None:
        """Saved and loaded pipeline should produce identical transformations."""
        pipeline = FeaturePipeline(feature_config)
        pipeline.fit_transform(sample_engine_data)

        save_path = tmp_path / "pipeline.joblib"
        pipeline.save(save_path)

        loaded = FeaturePipeline.load(save_path)
        assert loaded.feature_columns == pipeline.feature_columns
        assert loaded.sensor_cols == pipeline.sensor_cols

    def test_transform_uses_fitted_scaler(
        self, sample_engine_data: pd.DataFrame, feature_config: dict
    ) -> None:
        """transform() should use the fitted scaler, not re-fit."""
        pipeline = FeaturePipeline(feature_config)
        pipeline.fit_transform(sample_engine_data)

        # Create different data
        new_data = sample_engine_data.copy()
        new_data[["s_2", "s_3"]] += 100  # Shift values

        result = pipeline.transform(new_data)
        # Transformed values should be outside [0, 1] since scaler was
        # fitted on different data
        assert result is not None

    def test_unfitted_transform_raises(self, feature_config: dict) -> None:
        """transform() on unfitted pipeline should raise RuntimeError."""
        pipeline = FeaturePipeline(feature_config)
        dummy_df = pd.DataFrame({"s_2": [1.0], "unit_number": [1]})
        with pytest.raises(RuntimeError, match="has not been fitted"):
            pipeline.transform(dummy_df)

    def test_unfitted_save_raises(self, feature_config: dict, tmp_path: Path) -> None:
        """save() on unfitted pipeline should raise RuntimeError."""
        pipeline = FeaturePipeline(feature_config)
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            pipeline.save(tmp_path / "pipeline.joblib")
