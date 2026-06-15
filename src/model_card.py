"""
Automated Model Card Generation (Differentiator #5)
=====================================================

Auto-generates a MODEL_CARD.md following Google's Model Cards framework.
This is a documentation artifact that almost nobody includes in portfolio
projects, but signals awareness of model governance and responsible AI.

The model card is generated from:
    - Training metrics (metrics/train_metrics.json)
    - Evaluation metrics (metrics/eval_metrics.json)
    - Model configuration (configs/params.yaml)
    - MLflow run metadata
    - Data validation report

Reference:
    Mitchell et al. (2019). "Model Cards for Model Reporting."
    https://arxiv.org/abs/1810.03993
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.train import get_git_hash, get_dvc_data_hash

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_model_card(
    config_path: str | Path = "configs/params.yaml",
    train_metrics_path: str | Path = "metrics/train_metrics.json",
    eval_metrics_path: str | Path = "metrics/eval_metrics.json",
    validation_report_path: str | Path = "reports/validation_report.json",
    output_path: str | Path = "MODEL_CARD.md",
) -> str:
    """Generate a MODEL_CARD.md from training artifacts.

    Args:
        config_path: Path to params.yaml.
        train_metrics_path: Path to training metrics JSON.
        eval_metrics_path: Path to evaluation metrics JSON.
        validation_report_path: Path to data validation report JSON.
        output_path: Path to write the MODEL_CARD.md.

    Returns:
        The model card content as a string.
    """
    # Load available artifacts
    config = _load_json_or_yaml(config_path)
    train_metrics = _load_json_safe(train_metrics_path)
    eval_metrics = _load_json_safe(eval_metrics_path)
    validation_report = _load_json_safe(validation_report_path)

    model_type = config.get("model", {}).get("type", "unknown")
    dataset = config.get("data", {}).get("dataset", "FD001")
    rul_cap = config.get("features", {}).get("rul_cap", 125)
    git_hash = get_git_hash()
    data_hash = get_dvc_data_hash()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Format metrics
    def fmt(val: float | None, decimals: int = 4) -> str:
        return f"{val:.{decimals}f}" if val is not None else "N/A"

    card = f"""# Model Card: Predictive Maintenance RUL Predictor

> Auto-generated on {timestamp} by `src/model_card.py`
> Following Google's Model Cards framework (Mitchell et al., 2019)

---

## Model Details

| Property | Value |
|---|---|
| **Model Type** | {model_type.upper()} Regressor |
| **Task** | Remaining Useful Life (RUL) Prediction |
| **Framework** | scikit-learn / XGBoost |
| **Git Commit** | `{git_hash}` |
| **Data Version (DVC)** | `{data_hash}` |
| **Generation Date** | {timestamp} |

### Hyperparameters

"""

    if model_type == "xgboost":
        xgb_config = config.get("model", {}).get("xgboost", {})
        for k, v in xgb_config.items():
            card += f"- **{k}**: `{v}`\n"
    elif model_type == "ridge":
        ridge_config = config.get("model", {}).get("ridge", {})
        for k, v in ridge_config.items():
            card += f"- **{k}**: `{v}`\n"

    card += f"""
---

## Intended Use

### Primary Use Case
Predict the remaining number of operational cycles before a turbofan engine
fails, enabling proactive maintenance scheduling and preventing unplanned
downtime.

### Intended Users
- Maintenance engineers reviewing fleet health dashboards
- Automated maintenance scheduling systems
- Equipment reliability analysts

### Out-of-Scope Uses
- **Real-time autonomous control**: This model provides advisory predictions,
  not safety-critical control signals. Human-in-the-loop review is required.
- **Different equipment types**: Trained exclusively on turbofan engines.
  Do not apply to pumps, compressors, or other rotating machinery without
  retraining and revalidation.
- **Safety-critical decisions without engineering review**: Model predictions
  should augment, not replace, qualified engineering judgment.

---

## Training Data

| Property | Value |
|---|---|
| **Dataset** | NASA C-MAPSS {dataset} |
| **Source** | NASA Prognostics Center of Excellence |
| **Training Engines** | 100 units (run-to-failure trajectories) |
| **Total Training Cycles** | ~20,631 |
| **Sensors** | 21 measurements per cycle |
| **Operating Conditions** | 1 (sea level) |
| **Fault Modes** | 1 (HPC degradation) |
| **RUL Cap** | {rul_cap} cycles |

### Data Validation Status
"""

    if validation_report:
        val_passed = validation_report.get("overall_passed", "unknown")
        card += f"- **Schema validation**: {'✅ PASSED' if val_passed else '❌ FAILED'}\n"
    else:
        card += "- **Schema validation**: Not available\n"

    card += f"""
---

## Performance Metrics

### Training Metrics

| Metric | Value |
|---|---|
| **RMSE** | {fmt(train_metrics.get('train_rmse'))} |
| **MAE** | {fmt(train_metrics.get('train_mae'))} |
| **R²** | {fmt(train_metrics.get('train_r2'))} |
| **Asymmetric PHM Score** | {fmt(train_metrics.get('train_asymmetric_score'), 2)} |
| **CV RMSE (mean ± std)** | {fmt(train_metrics.get('cv_rmse_mean'))} ± {fmt(train_metrics.get('cv_rmse_std'))} |

### Test Metrics (Held-out Evaluation)

| Metric | Value |
|---|---|
| **RMSE** | {fmt(eval_metrics.get('test_rmse'))} |
| **MAE** | {fmt(eval_metrics.get('test_mae'))} |
| **R²** | {fmt(eval_metrics.get('test_r2'))} |
| **Asymmetric PHM Score** | {fmt(eval_metrics.get('test_asymmetric_score'), 2)} |
| **Test Samples** | {eval_metrics.get('test_samples', 'N/A')} |

### Scoring Function Note
The asymmetric PHM score penalizes late predictions (predicting more life
than remains) exponentially more than early predictions. This reflects the
real-world cost asymmetry: a late prediction could result in an in-flight
engine failure, while an early prediction only causes unnecessary maintenance.

---

## Limitations

1. **Single operating condition**: Trained on sea-level conditions only.
   Performance at altitude or varying conditions (FD002/FD004) is untested.

2. **Single fault mode**: Only HPC degradation is modeled. Other failure
   mechanisms (fan degradation, LPC issues) are not captured.

3. **Piecewise-linear RUL assumption**: The model assumes engines are
   "equally healthy" for the first {rul_cap} cycles. This simplification
   may miss early degradation signals.

4. **No concept drift adaptation**: The model's performance degrades if
   the sensor-failure relationship changes (e.g., due to equipment
   modifications or sensor recalibration). Retraining is required.

5. **Minimum history requirement**: Rolling window features require at
   least 5 cycles of sensor history. Predictions for engines with fewer
   than 5 cycles may be unreliable.

---

## Ethical Considerations

- **False negatives are costly**: Missing a true failure (predicting high
  RUL when the engine is about to fail) has significantly higher consequences
  than a false positive (predicting low RUL for a healthy engine). The
  asymmetric scoring function reflects this, but operators should still
  apply conservative safety margins.

- **Human-in-the-loop**: This model is designed to support maintenance
  decisions, not make them autonomously. All maintenance actions should
  be reviewed by qualified engineers.

- **Bias in simulation data**: The training data is simulated (C-MAPSS),
  not from real engines. Deployment on real equipment requires transfer
  learning validation and domain adaptation.

---

## Reproducibility

To reproduce this model from scratch:

```bash
git checkout {git_hash}
pip install -r requirements.txt
dvc repro
```

All experiment artifacts are logged in MLflow and can be inspected via:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---

## Model Lineage

| Component | Version/Hash |
|---|---|
| **Code** | Git commit `{git_hash}` |
| **Data** | DVC hash `{data_hash}` |
| **Config** | `configs/params.yaml` |
| **Pipeline** | `dvc.yaml` (5 stages) |
| **Tracking** | MLflow experiment `predictive-maintenance-rul` |
"""

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card)

    logger.info("Model card generated: %s", output_path)
    return card


def _load_json_safe(path: str | Path) -> dict:
    """Load a JSON file, returning empty dict if not found."""
    path = Path(path)
    if not path.exists():
        logger.warning("File not found: %s — using empty dict", path)
        return {}
    with open(path) as f:
        return json.load(f)


def _load_json_or_yaml(path: str | Path) -> dict:
    """Load either a JSON or YAML file based on extension."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for model card generation."""
    parser = argparse.ArgumentParser(description="Generate MODEL_CARD.md")
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument("--output", type=str, default="MODEL_CARD.md")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MODEL CARD GENERATION")
    logger.info("=" * 60)

    generate_model_card(config_path=args.config, output_path=args.output)


if __name__ == "__main__":
    main()
