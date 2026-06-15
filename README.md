# 🔧 Predictive Maintenance MLOps Pipeline

**End-to-end production-grade ML system for turbofan engine Remaining Useful Life (RUL) prediction.**

Built on the **NASA C-MAPSS FD001** dataset, this project demonstrates a complete MLOps lifecycle — from raw sensor time-series to a deployed API with automated drift monitoring and retraining triggers.

> **Not a Jupyter notebook.** Not a Kaggle kernel. This is an MLOps engineering project built with production infrastructure, data contracts, and continuous training in mind.

---

## 🏗️ Architecture

```
Sensor Data → Data Validation → Feature Engineering → Model Training → Evaluation
                    ↓ (gate)           ↓ (same class)         ↓ (audit trail)
              Schema Enforcement   FeaturePipeline.fit()   MLflow + Git + DVC hash
                                        ↕
                              FeaturePipeline.load()
                                        ↓
              FastAPI Service ← Model + Pipeline Artifacts
                    ↓
             Drift Monitoring (3-layer)
                    ↓
            Retraining Trigger (automated)
```

---

## 🔴 6 Production Differentiators

These are the features that distinguish this from a typical portfolio project:

| # | Feature | What it does | Why it matters |
|---|---------|-------------|---------------|
| 1 | **Pandera Schema Validation** | Enforces data contracts with physical sensor bounds as a pipeline gate | Catches silent data corruption before it reaches the model |
| 2 | **Shared FeaturePipeline** | Same Python class for training `fit_transform()` and serving `transform()` | Eliminates training-serving skew — the #1 silent failure in production ML |
| 3 | **3-Layer Drift Monitoring** | Data drift + Prediction drift + Concept drift simulation | Goes beyond basic monitoring most portfolios skip entirely |
| 4 | **Full Audit Trail** | Git hash + DVC hash + config hash + validation status in every MLflow run | Complete lineage from any prediction back to exact code + data + config |
| 5 | **Auto Model Card** | Generates `MODEL_CARD.md` following Google's framework | Shows awareness of model governance and responsible AI |
| 6 | **Retraining Trigger** | Evaluates drift signals against thresholds, returns CI exit codes | Closes the MLOps loop: deploy → monitor → detect → retrain |

---

## 📁 Project Structure

```
MLOPS/
├── configs/
│   └── params.yaml              # Single source of truth for all parameters
├── src/
│   ├── data_loader.py           # NASA C-MAPSS parser
│   ├── validate.py              # 🔴 Pandera schema enforcement
│   ├── feature_engineering.py   # 🔴 Shared FeaturePipeline class
│   ├── train.py                 # 🔴 XGBoost training + MLflow audit trail
│   ├── evaluate.py              # Test set evaluation with PHM scoring
│   ├── monitor.py               # 🔴 3-layer drift monitoring
│   ├── model_card.py            # 🔴 Auto-generated MODEL_CARD.md
│   └── retrain_check.py         # 🔴 Automated retraining trigger
├── api/
│   ├── main.py                  # FastAPI app with health/ready probes
│   ├── schemas.py               # Pydantic request/response models
│   └── model_loader.py          # Singleton artifact manager
├── tests/                       # pytest test suite (40+ tests)
├── docker/
│   └── Dockerfile               # Multi-stage production image
├── .github/
│   └── workflows/ci.yml         # Auto-train, test, report on push
├── dvc.yaml                     # 5-stage reproducible pipeline
├── pyproject.toml               # Project metadata + tool config
├── requirements.txt             # Pinned production dependencies
├── Makefile                     # Developer convenience commands
└── render.yaml                  # One-click Render deployment
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate conda environment
conda activate ncm

# Install missing dependencies
pip install xgboost>=2.1 mlflow>=2.17 "dvc>=3.56" evidently>=0.5 \
    gunicorn>=23.0 pandera>=0.21 ruff>=0.8 mypy>=1.13 pytest-cov>=5.0
```

### 2. Download Dataset

Download the **NASA C-MAPSS FD001** dataset and place the following files in `data/raw/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

> Dataset is ~6MB. Available from [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) or [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).

### 3. Run the Pipeline

```bash
# Initialize DVC
dvc init

# Run the full pipeline: prepare → validate → featurize → train → evaluate
dvc repro
```

### 4. View Results

```bash
# Launch MLflow UI
mlflow ui --port 5000

# Run drift monitoring
python -m src.monitor --config configs/params.yaml --simulate-drift

# Generate model card
python -m src.model_card

# Start API server
uvicorn api.main:app --reload --port 8000
```

### 5. Run Tests

```bash
pytest tests/ -v --tb=short --cov=src --cov=api --cov-report=term-missing
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe (always 200) |
| `GET` | `/ready` | Readiness probe (model loaded?) |
| `POST` | `/predict` | Predict RUL from sensor readings |
| `GET` | `/model-info` | Model metadata and metrics |
| `POST` | `/drift-check` | Check batch for data drift |
| `GET` | `/docs` | Interactive Swagger UI |

### Example Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {
        "time_cycles": 1,
        "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
        "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
        "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.02,
        "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
        "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195,
        "s_16": 0.03, "s_17": 392.0, "s_18": 2388.0,
        "s_19": 100.0, "s_20": 39.06, "s_21": 23.4190
      }
    ]
  }'
```

---

## 🐳 Docker

```bash
# Build
docker build -f docker/Dockerfile -t pred-maint:latest .

# Run
docker run -p 8000:8000 pred-maint:latest
```

---

## 📊 DVC Pipeline DAG

```
prepare → validate → featurize → train → evaluate
            ↓ (gate)
     Blocks if schema fails
```

```bash
# Reproduce full pipeline
dvc repro

# Show pipeline DAG
dvc dag

# Compare metrics across experiments
dvc metrics diff
```

---

## 🛠️ Tech Stack

| Category | Tool | Purpose |
|----------|------|---------|
| **ML** | XGBoost, scikit-learn | Model training |
| **Tracking** | MLflow | Experiment tracking, model registry |
| **Pipeline** | DVC | Reproducible pipeline orchestration |
| **Validation** | Pandera | Schema-based data quality enforcement |
| **Monitoring** | EvidentlyAI | 3-layer drift detection |
| **Serving** | FastAPI, Gunicorn, Uvicorn | Production API |
| **Containerization** | Docker | Deployment packaging |
| **CI/CD** | GitHub Actions | Auto-train, test, report |
| **Linting** | Ruff, mypy | Code quality, type checking |
| **Testing** | pytest | 40+ unit and integration tests |

---

## 📝 License

MIT

---

*Built by Prathamesh Kadam*
