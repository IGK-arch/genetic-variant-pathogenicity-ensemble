# RUVEGO Pathogenicity Prediction Pipeline

This repository implements the submitted PDR idea as a reproducible training pipeline for
anonymized genetic variant pathogenicity prediction.

## What It Runs

- Four provided datasets: `MASTER`, `KANSER`, `PAH`, `CFTR`
- Robust preprocessing:
  - median imputation for numeric features
  - missing-value indicators
  - robust scaling
  - categorical imputation and one-hot encoding
- Ensemble learning:
  - LightGBM and CatBoost are used automatically if installed
  - sklearn `HistGradientBoosting`, `ExtraTrees`, and `RandomForest` are always available fallbacks
- Panel adaptation:
  - for each panel, a global foundation model is trained on `MASTER`
  - overlapping panel `Variant_ID` values are removed from `MASTER` before fitting the foundation model
  - foundation pathogenicity probability is added as `GLOBAL_FOUNDATION_PROBA`
- Repeated stratified cross-validation
- Threshold tuning on out-of-fold predictions
- PR-AUC, ROC-AUC, Macro F1, class-specific precision/recall/F1
- Calibration plots, PR curves, feature importance, low-confidence samples, near-duplicate scan
- Adversarial validation between `MASTER` and each panel as a covariate-shift check

## Quick Run

```bash
python3 run_experiments.py --fast
```

## Isolated RUVEGO Environment

The full run was executed in a project-local environment named `.venv-ruvego`:

```bash
UV_CACHE_DIR=.uv-cache-ruvego \
UV_SYSTEM_CERTS=1 \
UV_INDEX_URL=https://artifactory.tcsbank.ru/artifactory/api/pypi/python-all/simple \
UV_HTTP_TIMEOUT=120 \
UV_HTTP_CONNECT_TIMEOUT=20 \
UV_HTTP_RETRIES=5 \
uv pip install pandas numpy scikit-learn matplotlib joblib lightgbm catboost shap optuna \
  --python .venv-ruvego/bin/python
```

This environment is independent from other projects and from the earlier sklearn-only run.

Full PDR-style repeated CV:

```bash
python3 run_experiments.py --folds 5 --repeats 3
```

Full optional-stack run with LightGBM, CatBoost, SHAP, and Optuna:

```bash
MPLCONFIGDIR=tmp/matplotlib \
NUMBA_CACHE_DIR=tmp/numba \
.venv-ruvego/bin/python run_experiments.py \
  --folds 5 \
  --repeats 3 \
  --output-dir outputs_full
```

To skip slower diagnostics:

```bash
python3 run_experiments.py --fast --skip-diagnostics --skip-adversarial
```

## Outputs

Artifacts are written to `outputs/`:

- `outputs/metrics/cv_summary.csv` - main panel-level results
- `outputs/metrics/fold_metrics_all.csv` - fold-level validation metrics
- `outputs/metrics/*_threshold_report.json` - selected threshold per dataset
- `outputs/predictions/*_oof_predictions.csv` - out-of-fold probabilities
- `outputs/models/*_final_model.joblib` - final trained models
- `outputs/models/*_foundation_model.joblib` - global foundation models for panels
- `outputs/reports/*_feature_importance.csv` - tree-based feature importance
- `outputs/reports/*_low_confidence.csv` - likely noisy / hard samples
- `outputs/reports/*_near_duplicates.csv` - nearest-neighbor similarity checks
- `outputs/figures/*_pr_curve.png` and `*_calibration.png`

## Predict On A New CSV

After training:

```bash
python3 predict.py \
  --dataset KANSER \
  --input path/to/test.csv \
  --output outputs/predictions/kanser_test_predictions.csv
```

For panel datasets, `predict.py` automatically loads the matching foundation model and adds
the `GLOBAL_FOUNDATION_PROBA` feature.

## Notes

The current local environment has sklearn but not LightGBM/CatBoost/SHAP/Optuna. The code is
therefore designed to run immediately with sklearn fallbacks while still activating the PDR
stack when optional packages are installed.
