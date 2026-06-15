"""
Automated Retraining Trigger (Differentiator #6)
==================================================

Evaluates whether model retraining is needed based on drift signals.
Completes the MLOps loop: deploy → monitor → detect degradation → retrain.

Most portfolio projects end at "deploy model." This module continues to
"detect degradation → recommend retraining," demonstrating understanding
of Continuous Training (CT) — not just CI/CD.

The module:
    1. Reads drift summary from the monitoring module
    2. Evaluates retraining conditions against configurable thresholds
    3. Returns a structured decision with reasoning
    4. Exits with code 1 if retraining is needed (CI/CD integration)

In production, this would trigger:
    - An automated retraining workflow (dvc repro with new data)
    - Slack/email alert to the ML engineering team
    - A model registry stage transition (Production → Staging)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
# Decision Container
# ---------------------------------------------------------------------------

@dataclass
class RetrainDecision:
    """Structured output of a retraining evaluation.

    Attributes:
        should_retrain: Whether retraining is recommended.
        reasons: List of reasons supporting the decision.
        drift_summary: Raw drift metrics from the monitoring module.
        exit_code: 0 = no retrain needed, 1 = retrain recommended.
    """
    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    drift_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        """Exit code for CI/CD integration."""
        return 1 if self.should_retrain else 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "should_retrain": self.should_retrain,
            "reasons": self.reasons,
            "exit_code": self.exit_code,
        }


# ---------------------------------------------------------------------------
# Retraining Logic
# ---------------------------------------------------------------------------


def check_retraining_needed(
    drift_summary_path: str | Path = "reports/drift_summary.json",
    config_path: str | Path = "configs/params.yaml",
) -> RetrainDecision:
    """Evaluate whether model retraining is needed based on drift signals.

    Decision logic (any condition triggers retrain):
        1. Data drift: >30% of features drifted (configurable)
        2. Prediction drift: detected by Wasserstein distance test
        3. Concept drift: mean prediction change >10 cycles (if simulation was run)
        4. Test suite: EvidentlyAI test suite failed

    Args:
        drift_summary_path: Path to the drift monitoring JSON output.
        config_path: Path to params.yaml for threshold configuration.

    Returns:
        RetrainDecision with recommendation and reasoning.
    """
    drift_summary_path = Path(drift_summary_path)
    config_path = Path(config_path)

    # Load drift summary
    if not drift_summary_path.exists():
        logger.warning("Drift summary not found: %s — cannot evaluate", drift_summary_path)
        return RetrainDecision(
            should_retrain=False,
            reasons=["Drift summary not available — run monitoring first"],
        )

    with open(drift_summary_path) as f:
        drift_summary = json.load(f)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    retrain_config = config.get("retraining", {})
    monitoring_config = config.get("monitoring", {})

    reasons: list[str] = []
    should_retrain = False

    # --- Check 1: Data Drift ---
    if retrain_config.get("trigger_on_data_drift", True):
        data_drift = drift_summary.get("data_drift", {})
        share_drifted = data_drift.get("share_drifted", 0.0)
        threshold = retrain_config.get("min_drift_share", 0.3)

        if share_drifted > threshold:
            should_retrain = True
            reasons.append(
                f"DATA DRIFT: {share_drifted:.1%} of features drifted "
                f"(threshold: {threshold:.1%})"
            )
            drifted_features = data_drift.get("drifted_features", [])
            if drifted_features:
                top_drifted = [f["feature"] for f in drifted_features[:5]]
                reasons.append(f"  Top drifted features: {top_drifted}")
        else:
            reasons.append(
                f"Data drift OK: {share_drifted:.1%} < {threshold:.1%} threshold"
            )

    # --- Check 2: Prediction Drift ---
    if retrain_config.get("trigger_on_prediction_drift", True):
        pred_drift = drift_summary.get("prediction_drift", {})
        drift_detected = pred_drift.get("drift_detected", False)

        if drift_detected:
            should_retrain = True
            drift_score = pred_drift.get("drift_score", 0.0)
            reasons.append(
                f"PREDICTION DRIFT: Detected (score={drift_score:.4f}). "
                f"Model output distribution has shifted."
            )
        else:
            reasons.append("Prediction drift OK: not detected")

    # --- Check 3: Concept Drift Impact (if available) ---
    concept_drift = drift_summary.get("concept_drift", {})
    if concept_drift.get("layer") == "concept_drift_simulation":
        mean_change = concept_drift.get("mean_prediction_change", 0)
        pct_changed = concept_drift.get("pct_predictions_changed_gt_10", 0)
        if mean_change > 10:
            should_retrain = True
            reasons.append(
                f"CONCEPT DRIFT RISK: Mean prediction change = {mean_change:.1f} cycles, "
                f"{pct_changed:.1f}% of predictions changed >10 cycles"
            )

    # --- Check 4: Test Suite ---
    test_suite = drift_summary.get("test_suite", {})
    test_passed = test_suite.get("test_suite_passed", True)
    if not test_passed:
        should_retrain = True
        reasons.append("DRIFT TEST SUITE: FAILED — automated quality gate breached")

    decision = RetrainDecision(
        should_retrain=should_retrain,
        reasons=reasons,
        drift_summary=drift_summary,
    )

    return decision


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for retraining check.

    Exit codes:
        0 = Model is stable, no retraining needed
        1 = Drift detected, retraining recommended
    """
    parser = argparse.ArgumentParser(
        description="Check if model retraining is needed based on drift signals."
    )
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument(
        "--drift-summary",
        type=str,
        default="reports/drift_summary.json",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RETRAINING CHECK")
    logger.info("=" * 60)

    decision = check_retraining_needed(
        drift_summary_path=args.drift_summary,
        config_path=args.config,
    )

    # Log decision
    for reason in decision.reasons:
        logger.info("  %s", reason)

    if decision.should_retrain:
        logger.warning("🔄 RECOMMENDATION: Retraining needed")
        logger.warning("  Exit code: %d", decision.exit_code)
    else:
        logger.info("✅ Model is stable — no retraining needed")

    # Save decision report
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "retrain_decision.json", "w") as f:
        json.dump(decision.to_dict(), f, indent=2)

    raise SystemExit(decision.exit_code)


if __name__ == "__main__":
    main()
