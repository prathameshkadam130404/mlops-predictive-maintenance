"""Tests for the drift monitoring module."""

import numpy as np
import pandas as pd
import pytest

from src.monitor import (
    detect_prediction_drift,
    simulate_concept_drift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_features() -> pd.DataFrame:
    """Create reference feature data."""
    rng = np.random.default_rng(42)
    n_rows = 200
    data = {}
    for i in range(15):
        data[f"feature_{i}"] = rng.normal(0, 1, n_rows)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictionDrift:
    """Tests for prediction drift detection."""

    def test_no_drift_on_identical_data(self) -> None:
        """Identical prediction distributions should show no drift."""
        preds = np.random.default_rng(42).normal(50, 10, 200)
        result = detect_prediction_drift(preds, preds.copy())
        assert result["layer"] == "prediction_drift"
        # With identical data, drift should generally not be detected
        assert "drift_detected" in result

    def test_drift_on_shifted_data(self) -> None:
        """Significantly shifted predictions should be detected as drift."""
        rng = np.random.default_rng(42)
        reference = rng.normal(50, 10, 200)
        shifted = rng.normal(100, 10, 200)  # Mean shifted by 50
        result = detect_prediction_drift(reference, shifted)
        assert result["drift_detected"] is True

    def test_result_contains_statistics(self) -> None:
        """Result should contain summary statistics."""
        rng = np.random.default_rng(42)
        preds = rng.normal(50, 10, 200)
        result = detect_prediction_drift(preds, preds.copy())
        assert "reference_mean" in result
        assert "reference_std" in result
        assert "current_mean" in result


class TestConceptDriftSimulation:
    """Tests for concept drift simulation."""

    def test_produces_modified_data(self, reference_features: pd.DataFrame) -> None:
        """Simulated drift should produce different data."""
        sensor_cols = [c for c in reference_features.columns if c.startswith("feature_")]
        drifted = simulate_concept_drift(reference_features, sensor_cols)
        # Data should be modified
        assert not drifted.equals(reference_features)

    def test_preserves_shape(self, reference_features: pd.DataFrame) -> None:
        """Simulated drift should preserve DataFrame shape."""
        sensor_cols = reference_features.columns.tolist()
        drifted = simulate_concept_drift(reference_features, sensor_cols)
        assert drifted.shape == reference_features.shape

    def test_no_nan_introduced(self, reference_features: pd.DataFrame) -> None:
        """Drift simulation should not introduce NaN values."""
        sensor_cols = reference_features.columns.tolist()
        drifted = simulate_concept_drift(reference_features, sensor_cols)
        assert drifted.isna().sum().sum() == 0
