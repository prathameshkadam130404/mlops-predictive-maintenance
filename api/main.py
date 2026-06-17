"""
FastAPI Inference Service
=========================

Production-grade REST API for RUL prediction with:
    - Lifespan-managed model loading (load once at startup)
    - Pydantic input validation
    - Pandera schema enforcement on sensor data
    - Shared FeaturePipeline (training-serving skew prevention)
    - Health/readiness probes for container orchestration
    - Drift check endpoint for online monitoring
    - Structured error handling and logging
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import ModelArtifacts
from api.schemas import (
    DriftCheckRequest,
    DriftCheckResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)

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
# Lifespan — Load model at startup, cleanup at shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Loads model artifacts at startup to avoid per-request loading overhead.
    If model files don't exist (e.g., before first training run), the API
    starts in a degraded state where /ready returns false.
    """
    artifacts = ModelArtifacts.get_instance()
    try:
        artifacts.load()
        logger.info("Model artifacts loaded — API is ready to serve predictions")
    except FileNotFoundError as e:
        logger.warning("Model artifacts not found: %s — API starting in degraded mode", e)
    except Exception as e:
        logger.error("Failed to load model artifacts: %s", e)

    yield  # Application runs

    logger.info("Shutting down — cleaning up resources")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "REST API for turbofan engine Remaining Useful Life (RUL) prediction. "
        "Uses XGBoost model trained on NASA C-MAPSS FD001 dataset with "
        "rolling window sensor features. Includes drift monitoring endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & Readiness Probes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health_check() -> HealthResponse:
    """Liveness probe — confirms the service process is alive.

    Always returns 200 OK. Used by container orchestrators (Docker, K8s)
    to determine if the container should be restarted.
    """
    return HealthResponse()


@app.get("/ready", response_model=ReadinessResponse, tags=["Infrastructure"])
async def readiness_check() -> ReadinessResponse:
    """Readiness probe — confirms the model is loaded and ready to serve.

    Returns 200 with ready=true only if model and pipeline are loaded.
    Used to gate traffic routing — don't send prediction requests
    until the model is ready.
    """
    artifacts = ModelArtifacts.get_instance()
    return ReadinessResponse(
        ready=artifacts.is_loaded,
        model_loaded=artifacts.model is not None,
        pipeline_loaded=artifacts.pipeline is not None,
        model_version=artifacts.version,
    )


# ---------------------------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict Remaining Useful Life from sensor readings.

    Accepts a sequence of sensor readings (consecutive cycles from one engine).
    The readings are transformed using the SAME FeaturePipeline that was used
    during training (preventing training-serving skew), and the model predicts
    the RUL based on the latest available features.

    Args:
        request: PredictionRequest with list of SensorReading objects.

    Returns:
        PredictionResponse with predicted RUL and metadata.

    Raises:
        HTTPException 503: If model is not loaded.
        HTTPException 422: If input validation fails.
    """
    artifacts = ModelArtifacts.get_instance()

    if not artifacts.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first (dvc repro).",
        )

    warnings: list[str] = []

    try:
        # Convert Pydantic models to DataFrame
        readings_dicts = [r.model_dump() for r in request.readings]
        df = pd.DataFrame(readings_dicts)

        # Add a dummy unit_number for the pipeline (single-engine prediction)
        df["unit_number"] = 1

        # Validate input with Pandera (optional, graceful degradation)
        try:
            from src.validate import validate_inference_input
            val_result = validate_inference_input(df)
            if not val_result.passed:
                warnings.append(
                    f"Input validation warnings: {len(val_result.errors)} issues detected. "
                    "Predictions may be unreliable for out-of-distribution inputs."
                )
        except Exception as e:
            logger.warning("Input validation skipped: %s", e)

        # Transform using the SAME FeaturePipeline (skew prevention)
        transformed = artifacts.pipeline.transform(df)  # type: ignore[union-attr]
        feature_matrix = artifacts.pipeline.get_feature_matrix(transformed)  # type: ignore[union-attr]

        # Predict RUL (use the last row — most recent cycle)
        last_features = feature_matrix.iloc[[-1]]
        rul_prediction = float(artifacts.model.predict(last_features)[0])

        # Clamp prediction to non-negative
        rul_prediction = max(0.0, rul_prediction)

        if len(request.readings) < 5:
            warnings.append(
                "Fewer than 5 cycles provided. Rolling window features may be "
                "unreliable. Provide at least 20 cycles for best accuracy."
            )

        return PredictionResponse(
            rul_prediction=round(rul_prediction, 2),
            model_version=artifacts.version,
            input_cycles=len(request.readings),
            warnings=warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") from e


# ---------------------------------------------------------------------------
# Model Info Endpoint
# ---------------------------------------------------------------------------


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Return model metadata, version, training metrics, and lineage.

    Provides transparency into which model is currently serving,
    its performance characteristics, and traceability information.
    """
    artifacts = ModelArtifacts.get_instance()

    if not artifacts.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return ModelInfoResponse(
        model_type=artifacts.model_type,
        model_version=artifacts.version,
        feature_count=artifacts.feature_count,
        training_metrics=artifacts.metrics,
        lineage={
            "model_file": "models/model.joblib",
            "pipeline_file": "models/feature_pipeline.joblib",
            "config_file": "configs/params.yaml",
        },
    )


# ---------------------------------------------------------------------------
# Drift Check Endpoint
# ---------------------------------------------------------------------------


@app.post("/drift-check", response_model=DriftCheckResponse, tags=["Monitoring"])
async def drift_check(request: DriftCheckRequest) -> DriftCheckResponse:
    """Check a batch of sensor readings for data drift.

    Compares incoming data distribution against the training reference.
    Uses EvidentlyAI under the hood for statistical drift detection.

    Requires at least 10 readings for statistical validity.
    """
    artifacts = ModelArtifacts.get_instance()

    if not artifacts.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Convert to DataFrame
        readings_dicts = [r.model_dump() for r in request.readings]
        current_df = pd.DataFrame(readings_dicts)

        # Load reference data for comparison
        reference_path = "data/features/train_features.parquet"
        try:
            reference_df = pd.read_parquet(reference_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Reference data not available. Run training pipeline first.",
            ) from exc

        # Get overlapping feature columns
        feature_cols = [c for c in artifacts.pipeline.feature_columns  # type: ignore[union-attr]
                        if c in current_df.columns and c in reference_df.columns]

        if len(feature_cols) < 5:
            raise HTTPException(
                status_code=422,
                detail="Insufficient overlapping features for drift detection.",
            )

        from src.monitor import detect_data_drift
        drift_result = detect_data_drift(
            reference_data=reference_df,
            current_data=current_df,
            numerical_features=feature_cols,
        )

        return DriftCheckResponse(
            drifted=drift_result.get("dataset_drift", False),
            share_of_drifted_columns=drift_result.get("share_drifted", 0.0),
            details=drift_result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Drift check failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Drift check error: {str(e)}") from e
