# Predictive Maintenance MLOps Pipeline

End-to-end production ML system for turbofan engine **Remaining Useful Life (RUL)** prediction, built on the NASA C-MAPSS FD001 dataset.

Covers the full MLOps lifecycle: data validation, feature engineering with skew prevention, experiment tracking, model serving, 3-layer drift monitoring, and automated retraining triggers — all orchestrated through DVC and CI/CD.

---

## Architecture

```
                         ┌──────────────┐
                         │  Raw Sensor  │
                         │    Data      │
                         └──────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │   Pandera Schema      │
                    │   Validation (gate)   │
                    └───────────┬───────────┘
                                │ pass/fail
                    ┌───────────▼───────────┐
                    │   FeaturePipeline     │
                    │   .fit_transform()    │──── serialized ────┐
                    └───────────┬───────────┘                    │
                                │                                │
                    ┌───────────▼───────────┐       ┌────────────▼──────────┐
                    │   XGBoost Training    │       │   FastAPI Service     │
                    │   + MLflow Audit      │       │   .transform()       │
                    └───────────┬───────────┘       │   .predict()         │
                                │                   └────────────┬─────────┘
                    ┌───────────▼───────────┐                    │
                    │   Evaluation          │       ┌────────────▼─────────┐
                    │   (PHM Scoring)       │       │  3-Layer Drift       │
                    └───────────────────────┘       │  Monitoring          │
                                                    └────────────┬─────────┘
                                                                 │
                                                    ┌────────────▼─────────┐
                                                    │  Retraining Trigger  │
                                                    │  (CI exit codes)     │
                                                    └──────────────────────┘
```

---

## Key Engineering Decisions

| Decision | Implementation | Rationale |
|----------|---------------|-----------|
| **Data contracts** | Pandera schema with physical sensor bounds as a DVC pipeline gate | Catches silent data corruption before it reaches the model |
| **Skew prevention** | Single `FeaturePipeline` class shared between training (`fit_transform`) and serving (`transform`) | Eliminates training-serving skew — the most common silent failure mode in production ML |
| **3-layer monitoring** | Data drift (KS test) + Prediction drift (Wasserstein) + Concept drift simulation | Detects degradation at the input, output, and relationship levels independently |
| **Full audit trail** | Git hash + DVC data hash + config hash + validation status logged per MLflow run | Any prediction can be traced back to the exact code, data, and configuration that produced the model |
| **Model governance** | Auto-generated `MODEL_CARD.md` following Google's framework | Documents intended use, limitations, ethical considerations, and reproducibility steps |
| **Closed-loop retraining** | Drift signals evaluated against configurable thresholds; returns non-zero CI exit codes | Automates the deploy → monitor → detect → retrain cycle |

---

## Results

### Training Performance (20,631 samples)

| Metric | Value |
|--------|-------|
| RMSE | 3.43 |
| MAE | 2.50 |
| R² | 0.993 |
| CV-RMSE (5-fold) | 18.00 ± 1.40 |

### Test Performance (100 engines, held-out)

| Metric | Value |
|--------|-------|
| RMSE | 59.47 |
| MAE | 47.60 |
| PHM Asymmetric Score | 824,360 |

> The gap between train and test metrics is expected for C-MAPSS FD001. The test set uses only the **last cycle** per engine with no temporal context, while training uses full degradation trajectories. This is a well-documented evaluation challenge in the PHM literature, not overfitting.

---

## Project Structure

```
├── configs/
│   └── params.yaml                 # Single source of truth for all parameters
├── src/
│   ├── data_loader.py              # NASA C-MAPSS text parser → Parquet
│   ├── validate.py                 # Pandera schema enforcement (pipeline gate)
│   ├── feature_engineering.py      # FeaturePipeline: rolling stats, lags, scaling
│   ├── train.py                    # XGBoost/Ridge + MLflow audit trail
│   ├── evaluate.py                 # Held-out evaluation with PHM scoring
│   ├── monitor.py                  # 3-layer drift monitoring (EvidentlyAI)
│   ├── model_card.py               # Auto-generated MODEL_CARD.md
│   └── retrain_check.py            # Automated retraining trigger
├── api/
│   ├── main.py                     # FastAPI with health/readiness probes
│   ├── schemas.py                  # Pydantic request/response validation
│   └── model_loader.py             # Singleton artifact loader
├── tests/                          # 64 unit and integration tests (pytest)
├── docker/
│   └── Dockerfile                  # Multi-stage production image
├── .github/
│   └── workflows/ci.yml            # Lint → Test → Train → Drift → Report
├── dvc.yaml                        # 5-stage reproducible pipeline
├── Makefile                        # Developer commands
├── requirements.txt                # Pinned dependencies
├── render.yaml                     # Render deployment blueprint
└── MODEL_CARD.md                   # Auto-generated model documentation
```

---

## Setup

### Prerequisites

- Python 3.11
- Git

### Installation

```bash
git clone https://github.com/prathameshkadam130404/mlops-predictive-maintenance.git
cd mlops-predictive-maintenance

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install dvc ruff mypy pytest-cov
```

### Dataset

Download the [NASA C-MAPSS FD001](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) dataset (~6 MB) and place the extracted `CMaps/` folder inside `data/raw/`:

```
data/raw/CMaps/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

### Run the Pipeline

```bash
dvc init
dvc repro
```

This executes 5 stages in sequence: `prepare → validate → featurize → train → evaluate`.

---

## Usage

### Experiment Tracking

```bash
mlflow ui --port 5000
```

Open `http://localhost:5000` and select the `predictive-maintenance-rul` experiment.

### Drift Monitoring

```bash
# Standard drift report
python -m src.monitor --config configs/params.yaml

# With simulated concept drift
python -m src.monitor --config configs/params.yaml --simulate-drift
```

Generates `reports/drift_summary.json` and an interactive `reports/drift_report.html`.

### Model Card

```bash
python -m src.model_card --config configs/params.yaml
```

Generates `MODEL_CARD.md` with metrics, lineage hashes, and limitations.

### Inference API

```bash
uvicorn api.main:app --reload --port 8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (model loaded) |
| `POST` | `/predict` | Predict RUL from sensor readings |
| `GET` | `/model-info` | Model metadata and evaluation metrics |
| `POST` | `/drift-check` | Batch drift detection |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [{
      "time_cycles": 1,
      "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
      "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
      "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.02,
      "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
      "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195,
      "s_16": 0.03, "s_17": 392.0, "s_18": 2388.0,
      "s_19": 100.0, "s_20": 39.06, "s_21": 23.4190
    }]
  }'
```

**Example response:**

```json
{
  "rul_prediction": 124.39,
  "model_version": "v1.0--3863682945317363046",
  "timestamp": "2026-06-16T02:26:42.392387+00:00",
  "input_cycles": 1,
  "warnings": [
    "Fewer than 5 cycles provided. Rolling window features may be unreliable."
  ]
}
```

### Tests

```bash
pytest tests/ -v --tb=short --cov=src --cov=api --cov-report=term-missing
```

64 tests, 41% coverage. Covers data loading, schema validation, feature engineering, training utilities, API endpoints, and drift monitoring.

---

## Docker

```bash
docker build -f docker/Dockerfile -t pred-maint:latest .
docker run -p 8000:8000 pred-maint:latest
```

Multi-stage build with non-root user, built-in health checks, and Gunicorn + Uvicorn process management.

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to `main`:

1. **Lint & Test** — Ruff linting + pytest with coverage
2. **Train & Report** — Full DVC pipeline + drift check + model card generation
3. **PR Summary** — Metrics, drift status, and retraining decision posted to the GitHub Actions summary

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | XGBoost, scikit-learn |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | DVC |
| Data Validation | Pandera |
| Drift Monitoring | EvidentlyAI |
| API Serving | FastAPI, Gunicorn, Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Code Quality | Ruff, mypy, pytest |

---

## License

MIT
