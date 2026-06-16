# Model Card: Predictive Maintenance RUL Predictor

> Auto-generated on 2026-06-16 02:17 UTC by `src/model_card.py`
> Following Google's Model Cards framework (Mitchell et al., 2019)

---

## Model Details

| Property | Value |
|---|---|
| **Model Type** | XGBOOST Regressor |
| **Task** | Remaining Useful Life (RUL) Prediction |
| **Framework** | scikit-learn / XGBoost |
| **Git Commit** | `b82d0d7` |
| **Data Version (DVC)** | `05b575bd7547` |
| **Generation Date** | 2026-06-16 02:17 UTC |

### Hyperparameters

- **n_estimators**: `200`
- **max_depth**: `6`
- **learning_rate**: `0.1`
- **subsample**: `0.8`
- **colsample_bytree**: `0.8`
- **early_stopping_rounds**: `20`
- **eval_metric**: `rmse`
- **random_state**: `42`
- **n_jobs**: `-1`

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
| **Dataset** | NASA C-MAPSS FD001 |
| **Source** | NASA Prognostics Center of Excellence |
| **Training Engines** | 100 units (run-to-failure trajectories) |
| **Total Training Cycles** | ~20,631 |
| **Sensors** | 21 measurements per cycle |
| **Operating Conditions** | 1 (sea level) |
| **Fault Modes** | 1 (HPC degradation) |
| **RUL Cap** | 125 cycles |

### Data Validation Status
- **Schema validation**: ✅ PASSED

---

## Performance Metrics

### Training Metrics

| Metric | Value |
|---|---|
| **RMSE** | 3.4273 |
| **MAE** | 2.4978 |
| **R²** | 0.9932 |
| **Asymmetric PHM Score** | 5823.13 |
| **CV RMSE (mean ± std)** | 17.9997 ± 1.3982 |

### Test Metrics (Held-out Evaluation)

| Metric | Value |
|---|---|
| **RMSE** | 59.4723 |
| **MAE** | 47.5996 |
| **R²** | -1.0482 |
| **Asymmetric PHM Score** | 824360.11 |
| **Test Samples** | 100 |

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
   "equally healthy" for the first 125 cycles. This simplification
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
git checkout b82d0d7
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
| **Code** | Git commit `b82d0d7` |
| **Data** | DVC hash `05b575bd7547` |
| **Config** | `configs/params.yaml` |
| **Pipeline** | `dvc.yaml` (5 stages) |
| **Tracking** | MLflow experiment `predictive-maintenance-rul` |
