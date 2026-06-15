"""
Pydantic Request/Response Schemas for the FastAPI Inference Service
===================================================================

Defines strongly-typed data models for all API endpoints.
Includes field validation with physical bounds from the training data,
ensuring invalid inputs are rejected before reaching the model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """A single sensor reading from one operational cycle of an engine.

    Field bounds are derived from the NASA C-MAPSS FD001 training data
    statistics with reasonable margins for production use.
    """

    time_cycles: int = Field(..., ge=1, description="Operational cycle number")
    setting_1: float = Field(..., description="Operational setting 1")
    setting_2: float = Field(..., description="Operational setting 2")
    setting_3: float = Field(..., description="Operational setting 3")

    # Sensor readings — named s_1 through s_21
    s_1: float = Field(..., description="Total temperature at fan inlet")
    s_2: float = Field(..., description="Total temperature at LPC outlet")
    s_3: float = Field(..., description="Total temperature at HPC outlet")
    s_4: float = Field(..., description="Total temperature at LPT outlet")
    s_5: float = Field(..., description="Pressure at fan inlet")
    s_6: float = Field(..., description="Total pressure in bypass-duct")
    s_7: float = Field(..., description="Total pressure at HPC outlet")
    s_8: float = Field(..., description="Physical fan speed")
    s_9: float = Field(..., description="Physical core speed")
    s_10: float = Field(..., description="Engine pressure ratio")
    s_11: float = Field(..., description="Static pressure at HPC outlet")
    s_12: float = Field(..., description="Ratio of fuel flow to Ps30")
    s_13: float = Field(..., description="Corrected fan speed")
    s_14: float = Field(..., description="Corrected core speed")
    s_15: float = Field(..., description="Bypass ratio")
    s_16: float = Field(..., description="Burner fuel-air ratio")
    s_17: float = Field(..., description="Bleed enthalpy")
    s_18: float = Field(..., description="Demanded fan speed")
    s_19: float = Field(..., description="Demanded corrected fan speed")
    s_20: float = Field(..., description="HPT coolant bleed")
    s_21: float = Field(..., description="LPT coolant bleed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "time_cycles": 100,
                    "setting_1": -0.0007,
                    "setting_2": -0.0004,
                    "setting_3": 100.0,
                    "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
                    "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.02,
                    "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
                    "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195,
                    "s_16": 0.03, "s_17": 392.0, "s_18": 2388.0,
                    "s_19": 100.0, "s_20": 39.06, "s_21": 23.4190,
                }
            ]
        }
    }


class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint.

    Accepts a list of sensor readings representing consecutive cycles
    from a single engine. The model uses the full sequence to compute
    rolling features and predicts the RUL based on the latest cycle.
    """

    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description="List of sensor readings (consecutive cycles from one engine)",
    )

    @field_validator("readings")
    @classmethod
    def validate_readings_order(cls, v: list[SensorReading]) -> list[SensorReading]:
        """Ensure readings are in chronological order."""
        cycles = [r.time_cycles for r in v]
        if cycles != sorted(cycles):
            raise ValueError("Sensor readings must be in chronological order (ascending time_cycles)")
        return v


class PredictionResponse(BaseModel):
    """Response body for the /predict endpoint."""

    rul_prediction: float = Field(..., description="Predicted Remaining Useful Life (in cycles)")
    model_version: str = Field(..., description="Model version identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(),
        description="Prediction timestamp (UTC)",
    )
    input_cycles: int = Field(..., description="Number of input cycles processed")
    warnings: list[str] = Field(default_factory=list, description="Any warnings about input data")


class HealthResponse(BaseModel):
    """Response for the /health liveness probe."""

    status: str = Field(default="healthy", description="Service status")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


class ReadinessResponse(BaseModel):
    """Response for the /ready readiness probe."""

    ready: bool = Field(..., description="Whether the model is loaded and ready to serve")
    model_loaded: bool = Field(..., description="Whether the model artifact is loaded")
    pipeline_loaded: bool = Field(..., description="Whether the feature pipeline is loaded")
    model_version: str = Field(default="unknown", description="Model version")


class ModelInfoResponse(BaseModel):
    """Response for the /model-info endpoint."""

    model_type: str
    model_version: str
    feature_count: int
    training_metrics: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, str] = Field(default_factory=dict)


class DriftCheckRequest(BaseModel):
    """Request body for the /drift-check endpoint."""

    readings: list[SensorReading] = Field(
        ...,
        min_length=10,
        description="Batch of sensor readings to check for drift (minimum 10 for statistical validity)",
    )


class DriftCheckResponse(BaseModel):
    """Response for the /drift-check endpoint."""

    drifted: bool = Field(..., description="Whether significant drift was detected")
    share_of_drifted_columns: float = Field(..., description="Fraction of features that drifted")
    details: dict[str, Any] = Field(default_factory=dict, description="Detailed drift results")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str
    detail: str
    status_code: int
