"""
Data Validation Layer
=====================

Implements Pandera schema-based validation for the predictive maintenance pipeline.
Enforces data quality as a pipeline gate — if validation fails, downstream
stages (feature engineering, training) are blocked.

Three validation schemas:
    1. RawSensorSchema   — validates parsed C-MAPSS sensor data
    2. FeatureSchema     — validates engineered features (post-processing)
    3. InferenceSchema   — validates API input at serving time

All schemas produce structured validation reports (JSON) logged to MLflow
and consumed by the CI pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera as pa
import yaml
from pandera import Check, Column, DataFrameSchema

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
# Validation Result Container
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Structured output of a validation run.

    Attributes:
        passed: Whether all validation checks passed.
        schema_name: Name of the schema that was applied.
        n_rows: Number of rows validated.
        n_columns: Number of columns validated.
        errors: List of error details (empty if passed).
        warnings: List of non-fatal warnings.
    """

    passed: bool
    schema_name: str
    n_rows: int
    n_columns: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "passed": self.passed,
            "schema_name": self.schema_name,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "error_count": len(self.errors),
            "errors": self.errors[:20],  # Cap at 20 for readability
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

# Physically plausible bounds for C-MAPSS FD001 sensors.
# Derived from training data statistics (min/max with 10% margin).
# These bounds catch gross anomalies (negative temperatures, impossible pressures).
SENSOR_BOUNDS: dict[str, tuple[float, float]] = {
    "s_1": (440.0, 520.0),
    "s_2": (637.0, 650.0),
    "s_3": (1555.0, 1620.0),
    "s_4": (1280.0, 1420.0),
    "s_5": (0.0, 25.0),
    "s_6": (0.0, 25.0),
    "s_7": (340.0, 560.0),
    "s_8": (2387.0, 2390.0),
    "s_9": (9000.0, 9500.0),
    "s_10": (1.25, 1.35),
    "s_11": (40.0, 50.0),
    "s_12": (375.0, 560.0),
    "s_13": (2387.0, 2390.0),
    "s_14": (8090.0, 8200.0),
    "s_15": (8.0, 9.0),
    "s_16": (0.01, 0.05),
    "s_17": (300.0, 400.0),
    "s_18": (2380.0, 2400.0),
    "s_19": (100.0, 100.1),
    "s_20": (14.0, 40.0),
    "s_21": (20.0, 25.0),
}


def build_raw_sensor_schema() -> DataFrameSchema:
    """Build the Pandera schema for raw C-MAPSS sensor data.

    Validates:
        - unit_number: positive integer
        - time_cycles: positive integer
        - setting_1..3: float (operational conditions)
        - s_1..s_21: float within physically plausible bounds, no nulls

    Returns:
        Pandera DataFrameSchema for raw sensor data validation.
    """
    columns: dict[str, Column] = {
        "unit_number": Column(
            int,
            Check.ge(1),
            nullable=False,
            description="Engine unit identifier",
        ),
        "time_cycles": Column(
            int,
            Check.ge(1),
            nullable=False,
            description="Operational cycle count",
        ),
        "setting_1": Column(float, nullable=False, coerce=True),
        "setting_2": Column(float, nullable=False, coerce=True),
        "setting_3": Column(float, nullable=False, coerce=True),
    }

    # Add sensor columns with physical bounds
    for sensor_name, (low, high) in SENSOR_BOUNDS.items():
        columns[sensor_name] = Column(
            float,
            checks=[
                Check.ge(low * 0.5, error=f"{sensor_name} below physical minimum"),
                Check.le(high * 2.0, error=f"{sensor_name} above physical maximum"),
            ],
            nullable=False,
            coerce=True,
            description=f"Sensor reading: {sensor_name}",
        )

    return DataFrameSchema(
        columns=columns,
        name="RawSensorSchema",
        strict=False,  # Allow extra columns without failing
        coerce=True,
    )


def build_inference_schema() -> DataFrameSchema:
    """Build the Pandera schema for API inference input validation.

    This schema is used in the FastAPI /predict endpoint to validate
    incoming sensor readings before they reach the model.

    Stricter than the raw schema — enforces tighter bounds to catch
    clearly anomalous inputs that the model was never trained on.

    Returns:
        Pandera DataFrameSchema for inference input validation.
    """
    columns: dict[str, Column] = {
        "time_cycles": Column(
            int,
            Check.ge(1),
            nullable=False,
            coerce=True,
        ),
        "setting_1": Column(float, nullable=False, coerce=True),
        "setting_2": Column(float, nullable=False, coerce=True),
        "setting_3": Column(float, nullable=False, coerce=True),
    }

    for sensor_name, (low, high) in SENSOR_BOUNDS.items():
        columns[sensor_name] = Column(
            float,
            checks=[
                Check.ge(low * 0.8, error=f"{sensor_name} below expected range"),
                Check.le(high * 1.2, error=f"{sensor_name} above expected range"),
            ],
            nullable=False,
            coerce=True,
        )

    return DataFrameSchema(
        columns=columns,
        name="InferenceInputSchema",
        strict=False,
        coerce=True,
    )


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def validate_raw_data(df: pd.DataFrame) -> ValidationResult:
    """Validate a raw C-MAPSS DataFrame against the RawSensorSchema.

    Args:
        df: Raw sensor DataFrame to validate.

    Returns:
        ValidationResult with pass/fail status and error details.
    """
    schema = build_raw_sensor_schema()
    return _run_validation(df, schema, "RawSensorSchema")


def validate_inference_input(df: pd.DataFrame) -> ValidationResult:
    """Validate inference input data against the InferenceInputSchema.

    Used by the FastAPI /predict endpoint to gate bad inputs.

    Args:
        df: Input sensor DataFrame from API request.

    Returns:
        ValidationResult with pass/fail status and error details.
    """
    schema = build_inference_schema()
    return _run_validation(df, schema, "InferenceInputSchema")


def validate_features(df: pd.DataFrame, rul_cap: int = 125) -> ValidationResult:
    """Validate engineered feature DataFrame.

    Checks:
        - No NaN values (rolling features produce NaN for early cycles;
          these should have been handled by min_periods or forward-fill)
        - RUL target is within [0, rul_cap]
        - No infinite values
        - Feature count is reasonable

    Args:
        df: Engineered feature DataFrame.
        rul_cap: Maximum RUL value (from params.yaml).

    Returns:
        ValidationResult with pass/fail status.
    """
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []

    # Check for NaN
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        for col, count in nan_cols.items():
            errors.append({
                "check": "no_nan",
                "column": str(col),
                "detail": f"{count} NaN values found",
            })

    # Check for infinity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        for col, count in inf_cols.items():
            errors.append({
                "check": "no_infinity",
                "column": str(col),
                "detail": f"{count} infinite values found",
            })

    # Check RUL bounds (if RUL column exists)
    if "rul" in df.columns:
        if (df["rul"] < 0).any():
            errors.append({
                "check": "rul_non_negative",
                "column": "rul",
                "detail": f"Found {(df['rul'] < 0).sum()} negative RUL values",
            })
        if (df["rul"] > rul_cap).any():
            errors.append({
                "check": "rul_cap",
                "column": "rul",
                "detail": f"Found {(df['rul'] > rul_cap).sum()} RUL values > {rul_cap}",
            })

    # Warn if feature count seems low
    if len(df.columns) < 30:
        warnings.append(
            f"Feature count ({len(df.columns)}) seems low for a rolling-window pipeline. "
            "Expected 50+ features with multiple window sizes."
        )

    passed = len(errors) == 0
    return ValidationResult(
        passed=passed,
        schema_name="FeatureSchema",
        n_rows=len(df),
        n_columns=len(df.columns),
        errors=errors,
        warnings=warnings,
    )


def _run_validation(
    df: pd.DataFrame, schema: DataFrameSchema, schema_name: str
) -> ValidationResult:
    """Execute a Pandera schema validation and capture results.

    Args:
        df: DataFrame to validate.
        schema: Pandera schema to apply.
        schema_name: Human-readable schema name for the report.

    Returns:
        ValidationResult with structured error details.
    """
    try:
        schema.validate(df, lazy=True)
        logger.info("✅ Validation PASSED: %s (%d rows, %d cols)", schema_name, len(df), len(df.columns))
        return ValidationResult(
            passed=True,
            schema_name=schema_name,
            n_rows=len(df),
            n_columns=len(df.columns),
        )
    except pa.errors.SchemaErrors as exc:
        errors = []
        for _, row in exc.failure_cases.iterrows():
            errors.append({
                "check": str(row.get("check", "unknown")),
                "column": str(row.get("column", "unknown")),
                "index": str(row.get("index", "unknown")),
                "failure_value": str(row.get("failure_case", "unknown")),
            })

        logger.error(
            "❌ Validation FAILED: %s — %d errors found", schema_name, len(errors)
        )
        return ValidationResult(
            passed=False,
            schema_name=schema_name,
            n_rows=len(df),
            n_columns=len(df.columns),
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def save_validation_report(
    results: list[ValidationResult], output_path: str | Path
) -> None:
    """Save validation results as a structured JSON report.

    This report is:
        - Consumed by the DVC pipeline as a stage output
        - Logged as an MLflow artifact for audit trail
        - Parsed by GitHub Actions CI for PR summary reporting

    Args:
        results: List of ValidationResult objects (one per dataset validated).
        output_path: Path to save the JSON report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "overall_passed": all(r.passed for r in results),
        "validations": [r.to_dict() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Validation report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI Entrypoint — DVC Pipeline Stage
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the data validation DVC stage.

    Validates processed training and test data against the RawSensorSchema.
    Produces a JSON validation report that gates downstream pipeline stages.
    """
    parser = argparse.ArgumentParser(
        description="Validate processed C-MAPSS data against quality schemas."
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

    processed_dir = Path(config["data"]["processed_dir"])
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    logger.info("=" * 60)
    logger.info("DATA VALIDATION STAGE")
    logger.info("=" * 60)

    results: list[ValidationResult] = []

    # Validate training data
    if train_path.exists():
        train_df = pd.read_parquet(train_path)
        result = validate_raw_data(train_df)
        results.append(result)
    else:
        logger.error("Training data not found: %s", train_path)
        results.append(
            ValidationResult(
                passed=False,
                schema_name="RawSensorSchema (train)",
                n_rows=0,
                n_columns=0,
                errors=[{"check": "file_exists", "detail": f"File not found: {train_path}"}],
            )
        )

    # Validate test data
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        result = validate_raw_data(test_df)
        results.append(result)
    else:
        logger.error("Test data not found: %s", test_path)
        results.append(
            ValidationResult(
                passed=False,
                schema_name="RawSensorSchema (test)",
                n_rows=0,
                n_columns=0,
                errors=[{"check": "file_exists", "detail": f"File not found: {test_path}"}],
            )
        )

    # Save report
    save_validation_report(results, "reports/validation_report.json")

    # Exit with non-zero if validation failed (gates DVC pipeline)
    if not all(r.passed for r in results):
        logger.error("DATA VALIDATION FAILED — pipeline halted.")
        raise SystemExit(1)

    logger.info("Data validation stage completed successfully.")


if __name__ == "__main__":
    main()
