"""
Model Loader — Singleton Artifact Manager
==========================================

Manages loading and caching of model artifacts (trained model +
FeaturePipeline) at application startup. Uses the SAME FeaturePipeline
class from src/feature_engineering.py, ensuring zero training-serving skew.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

from src.feature_engineering import FeaturePipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class ModelArtifacts:
    """Singleton container for loaded model artifacts.

    Loaded once at FastAPI startup via lifespan events.
    Reused across all request handlers without reloading.

    Attributes:
        model: Trained sklearn/xgboost model.
        pipeline: Fitted FeaturePipeline instance.
        version: Model version string.
        metrics: Training/evaluation metrics.
        is_loaded: Whether all artifacts are successfully loaded.
    """

    _instance: ModelArtifacts | None = None

    def __init__(self) -> None:
        self.model: Any = None
        self.pipeline: FeaturePipeline | None = None
        self.version: str = "unknown"
        self.metrics: dict[str, Any] = {}
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> ModelArtifacts:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(
        self,
        model_path: str | Path = "models/model.joblib",
        pipeline_path: str | Path = "models/feature_pipeline.joblib",
        metrics_path: str | Path = "metrics/eval_metrics.json",
    ) -> None:
        """Load all model artifacts from disk.

        Args:
            model_path: Path to the trained model.
            pipeline_path: Path to the fitted FeaturePipeline.
            metrics_path: Path to evaluation metrics JSON.

        Raises:
            FileNotFoundError: If required artifact files don't exist.
        """
        model_path = Path(model_path)
        pipeline_path = Path(pipeline_path)
        metrics_path = Path(metrics_path)

        # Load trained model
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        self.model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)

        # Load fitted feature pipeline (SAME class as training)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found: {pipeline_path}")
        self.pipeline = FeaturePipeline.load(pipeline_path)
        logger.info("Feature pipeline loaded from %s", pipeline_path)

        # Load metrics (optional)
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.metrics = json.load(f)
            logger.info("Metrics loaded from %s", metrics_path)

        # Derive version from model hash or metrics
        self.version = f"v1.0-{hash(str(model_path.stat().st_mtime))}"

        self.is_loaded = True
        logger.info("All model artifacts loaded successfully (version: %s)", self.version)

    @property
    def feature_count(self) -> int:
        """Number of features the model expects."""
        if self.pipeline and self.pipeline.feature_columns:
            return len(self.pipeline.feature_columns)
        return 0

    @property
    def model_type(self) -> str:
        """String name of the model type."""
        if self.model is not None:
            return type(self.model).__name__
        return "unknown"
