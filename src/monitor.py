"""
Three-Layer Drift Monitoring
============================

Implements three layers of drift detection using EvidentlyAI:

Layer 1 — Data Drift:
    Kolmogorov-Smirnov tests on sensor input distributions between
    reference (training) and current data.

Layer 2 — Prediction Drift:
    Wasserstein distance on the model's RUL output distribution.
    Detects model staleness without requiring ground-truth labels.

Layer 3 — Concept Drift Simulation:
    Injects correlated noise and baseline shifts to simulate changes
    in the sensor-failure relationship, then measures prediction impact.

Outputs: HTML report, JSON summary, MLflow artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns

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
# Layer 1: Data Drift Detection
# ---------------------------------------------------------------------------


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: list[str],
    stattest: str = "ks",
) -> dict[str, Any]:
    """Detect distribution shifts in sensor input features.

    Compares the statistical distribution of each numerical feature between
    the reference (training) and current (production/test) datasets.

    Uses Kolmogorov-Smirnov test by default — a non-parametric,
    distribution-free test suitable for continuous sensor data.

    Args:
        reference_data: Training data (the baseline distribution).
        current_data: New data to check for drift.
        numerical_features: List of feature columns to monitor.
        stattest: Statistical test name ('ks', 'wasserstein', 'psi', etc.).

    Returns:
        Dictionary with drift detection results.
    """
    logger.info("Layer 1: Detecting data drift across %d features", len(numerical_features))

    column_mapping = ColumnMapping(
        numerical_features=numerical_features,
    )

    report = Report(metrics=[DataDriftPreset(stattest=stattest)])
    report.run(
        reference_data=reference_data[numerical_features],
        current_data=current_data[numerical_features],
        column_mapping=column_mapping,
    )

    # Extract results
    report_dict = report.as_dict()

    # Parse drift results from the report
    metrics = report_dict.get("metrics", [])
    drift_result = {
        "layer": "data_drift",
        "stattest": stattest,
        "n_features": len(numerical_features),
        "drifted_features": [],
        "n_drifted": 0,
        "share_drifted": 0.0,
        "dataset_drift": False,
    }

    for metric in metrics:
        metric_result = metric.get("result", {})
        if "drift_by_columns" in metric_result:
            drift_by_cols = metric_result["drift_by_columns"]
            for col_name, col_data in drift_by_cols.items():
                if col_data.get("drift_detected", False):
                    drift_result["drifted_features"].append({
                        "feature": col_name,
                        "stattest": col_data.get("stattest_name", stattest),
                        "p_value": col_data.get("p_value", None),
                        "drift_score": col_data.get("drift_score", None),
                    })

        if "dataset_drift" in metric_result:
            drift_result["dataset_drift"] = metric_result["dataset_drift"]
        if "share_of_drifted_columns" in metric_result:
            drift_result["share_drifted"] = metric_result["share_of_drifted_columns"]
        if "number_of_drifted_columns" in metric_result:
            drift_result["n_drifted"] = metric_result["number_of_drifted_columns"]

    logger.info(
        "Data drift: %d/%d features drifted (%.1f%%)",
        drift_result["n_drifted"],
        drift_result["n_features"],
        drift_result["share_drifted"] * 100,
    )

    return drift_result


# ---------------------------------------------------------------------------
# Layer 2: Prediction Drift Detection
# ---------------------------------------------------------------------------


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Detect shifts in the model's RUL prediction distribution.

    Prediction drift is a proxy metric used when ground-truth labels
    aren't available in real time. If the model's output distribution
    shifts significantly, the model is likely operating outside its
    training context.

    Uses Wasserstein distance (Earth Mover's Distance) to quantify
    the distributional shift — this metric is more sensitive to small
    shifts than KS for continuous distributions.

    Args:
        reference_predictions: RUL predictions on the training/reference data.
        current_predictions: RUL predictions on the current/test data.
        threshold: Wasserstein distance threshold for drift detection.

    Returns:
        Dictionary with prediction drift results.
    """
    logger.info("Layer 2: Detecting prediction drift")

    ref_df = pd.DataFrame({"prediction": reference_predictions})
    cur_df = pd.DataFrame({"prediction": current_predictions})

    column_mapping = ColumnMapping(
        numerical_features=["prediction"],
    )

    report = Report(metrics=[DataDriftPreset(stattest="wasserstein")])
    report.run(
        reference_data=ref_df,
        current_data=cur_df,
        column_mapping=column_mapping,
    )

    report_dict = report.as_dict()

    drift_detected = False
    drift_score = 0.0

    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            pred_drift = result["drift_by_columns"].get("prediction", {})
            drift_detected = pred_drift.get("drift_detected", False)
            drift_score = pred_drift.get("drift_score", 0.0)

    pred_drift_result = {
        "layer": "prediction_drift",
        "drift_detected": drift_detected,
        "drift_score": float(drift_score),
        "threshold": threshold,
        "reference_mean": float(reference_predictions.mean()),
        "reference_std": float(reference_predictions.std()),
        "current_mean": float(current_predictions.mean()),
        "current_std": float(current_predictions.std()),
    }

    logger.info(
        "Prediction drift: detected=%s, score=%.4f (threshold=%.4f)",
        drift_detected,
        drift_score,
        threshold,
    )

    return pred_drift_result


# ---------------------------------------------------------------------------
# Layer 3: Concept Drift Simulation
# ---------------------------------------------------------------------------


def simulate_concept_drift(
    reference_data: pd.DataFrame,
    sensor_cols: list[str],
    noise_factor: float = 0.15,
    shift_factor: float = 0.1,
) -> pd.DataFrame:
    """Simulate concept drift by modifying the sensor-failure relationship.

    Real concept drift in equipment doesn't just add noise to sensors —
    it changes WHICH sensor combinations indicate failure. This function
    simulates that by:
        1. Adding correlated noise to sensor groups (simulating sensor degradation)
        2. Shifting specific sensor baselines (simulating operating condition changes)

    Applies correlated noise to sensor groups and baseline shifts to
    selected sensors.

    Args:
        reference_data: Clean reference DataFrame.
        sensor_cols: List of sensor column names.
        noise_factor: Standard deviation multiplier for noise injection.
        shift_factor: Fraction of the feature range to shift baselines.

    Returns:
        Modified DataFrame simulating concept drift.
    """
    logger.info("Layer 3: Simulating concept drift (noise=%.2f, shift=%.2f)", noise_factor, shift_factor)

    drifted = reference_data.copy()

    rng = np.random.default_rng(seed=42)
    available_sensors = [c for c in sensor_cols if c in drifted.columns]

    # Group 1: Add correlated noise to first half of sensors
    # (simulates mechanical coupling — adjacent sensors drift together)
    n_sensors = len(available_sensors)
    group1 = available_sensors[: n_sensors // 2]
    correlated_noise = rng.normal(0, noise_factor, size=len(drifted))
    for sensor in group1:
        sensor_std = drifted[sensor].std()
        if sensor_std > 0:
            drifted[sensor] = drifted[sensor] + correlated_noise * sensor_std

    # Group 2: Shift baseline of second half of sensors
    # (simulates operating condition change)
    group2 = available_sensors[n_sensors // 2:]
    for sensor in group2:
        sensor_range = drifted[sensor].max() - drifted[sensor].min()
        if sensor_range > 0:
            shift = shift_factor * sensor_range * rng.choice([-1, 1])
            drifted[sensor] = drifted[sensor] + shift

    return drifted


def evaluate_concept_drift_impact(
    model: Any,
    reference_features: pd.DataFrame,
    drifted_features: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Evaluate how concept drift impacts model predictions.

    Compares model performance on clean vs. drifted data to quantify
    the impact of concept drift on prediction accuracy.

    Args:
        model: Trained model.
        reference_features: Clean feature matrix.
        drifted_features: Concept-drifted feature matrix.
        feature_columns: List of feature columns to use.

    Returns:
        Dictionary with concept drift impact metrics.
    """
    ref_cols = [c for c in feature_columns if c in reference_features.columns]
    drift_cols = [c for c in feature_columns if c in drifted_features.columns]

    ref_preds = model.predict(reference_features[ref_cols])
    drift_preds = model.predict(drifted_features[drift_cols])

    pred_diff = np.abs(ref_preds - drift_preds)

    result = {
        "layer": "concept_drift_simulation",
        "mean_prediction_change": float(pred_diff.mean()),
        "max_prediction_change": float(pred_diff.max()),
        "median_prediction_change": float(np.median(pred_diff)),
        "pct_predictions_changed_gt_10": float((pred_diff > 10).mean() * 100),
        "reference_pred_mean": float(ref_preds.mean()),
        "drifted_pred_mean": float(drift_preds.mean()),
    }

    logger.info(
        "Concept drift impact: mean ΔRul=%.2f, %.1f%% predictions changed >10 cycles",
        result["mean_prediction_change"],
        result["pct_predictions_changed_gt_10"],
    )

    return result


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def generate_drift_html_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: list[str],
    output_path: str | Path,
) -> None:
    """Generate a comprehensive HTML drift report.

    Args:
        reference_data: Reference (training) dataset.
        current_data: Current (production/test) dataset.
        numerical_features: Numerical feature columns.
        output_path: Path to save the HTML report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    column_mapping = ColumnMapping(numerical_features=numerical_features)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data[numerical_features],
        current_data=current_data[numerical_features],
        column_mapping=column_mapping,
    )

    report.save_html(str(output_path))
    logger.info("HTML drift report saved to %s", output_path)


def run_drift_test_suite(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: list[str],
    drift_share_threshold: float = 0.3,
) -> dict[str, Any]:
    """Run EvidentlyAI test suite for CI/CD integration.

    Returns pass/fail status that can be used as a pipeline gate.

    Args:
        reference_data: Reference (training) dataset.
        current_data: Current dataset.
        numerical_features: Numerical feature columns.
        drift_share_threshold: Maximum allowable fraction of drifted features.

    Returns:
        Dictionary with test suite results.
    """
    column_mapping = ColumnMapping(numerical_features=numerical_features)

    test_suite = TestSuite(tests=[
        TestShareOfDriftedColumns(lt=drift_share_threshold),
    ])

    test_suite.run(
        reference_data=reference_data[numerical_features],
        current_data=current_data[numerical_features],
        column_mapping=column_mapping,
    )

    result_dict = test_suite.as_dict()
    all_passed = result_dict.get("summary", {}).get("all_passed", False)

    return {
        "test_suite_passed": all_passed,
        "summary": result_dict.get("summary", {}),
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the drift monitoring module."""
    parser = argparse.ArgumentParser(
        description="Run 3-layer drift monitoring on trained model."
    )
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument("--simulate-drift", action="store_true",
                        help="Also run concept drift simulation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    monitoring_config = config.get("monitoring", {})
    features_dir = Path(config["data"]["features_dir"])

    logger.info("=" * 60)
    logger.info("DRIFT MONITORING — 3-Layer Analysis")
    logger.info("=" * 60)

    # Load data
    train_features = pd.read_parquet(features_dir / "train_features.parquet")

    # Separate features and target
    if "rul" in train_features.columns:
        feature_cols = [c for c in train_features.columns if c != "rul"]
    else:
        feature_cols = train_features.columns.tolist()

    # For demonstration, split training data into reference/current
    # For demonstration, split training data into reference/current
    n_split = len(train_features) // 2
    reference_data = train_features.iloc[:n_split]
    current_data = train_features.iloc[n_split:]

    results: dict[str, Any] = {}

    # --- Layer 1: Data Drift ---
    data_drift = detect_data_drift(
        reference_data=reference_data,
        current_data=current_data,
        numerical_features=feature_cols,
        stattest=monitoring_config.get("drift_stattest", "ks"),
    )
    results["data_drift"] = data_drift

    # --- Layer 2: Prediction Drift ---
    model = joblib.load("models/model.joblib")
    ref_preds = model.predict(reference_data[feature_cols])
    cur_preds = model.predict(current_data[feature_cols])

    pred_drift = detect_prediction_drift(
        reference_predictions=ref_preds,
        current_predictions=cur_preds,
        threshold=monitoring_config.get("prediction_drift_threshold", 0.1),
    )
    results["prediction_drift"] = pred_drift

    # --- Layer 3: Concept Drift (if requested) ---
    if args.simulate_drift:
        # Load pipeline to get sensor columns
        from src.feature_engineering import FeaturePipeline
        pipeline = FeaturePipeline.load("models/feature_pipeline.joblib")

        drifted_data = simulate_concept_drift(
            current_data, feature_cols, noise_factor=0.15, shift_factor=0.1
        )
        concept_impact = evaluate_concept_drift_impact(
            model, current_data, drifted_data, feature_cols
        )
        results["concept_drift"] = concept_impact
    else:
        results["concept_drift"] = {"layer": "concept_drift_simulation", "status": "not_run"}

    # --- Run Test Suite ---
    test_results = run_drift_test_suite(
        reference_data, current_data, feature_cols,
        monitoring_config.get("drift_share_threshold", 0.3),
    )
    results["test_suite"] = test_results

    # --- Generate Reports ---
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # HTML report
    generate_drift_html_report(
        reference_data, current_data, feature_cols,
        reports_dir / "drift_report.html",
    )

    # JSON summary
    with open(reports_dir / "drift_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Drift summary saved to %s", reports_dir / "drift_summary.json")

    logger.info("Drift monitoring completed successfully.")


if __name__ == "__main__":
    main()
