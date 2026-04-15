# Predicting U.S. Domestic Flight Delays Using Machine Learning

**EECE 5644 — Introduction to Machine Learning | Northeastern University | Spring 2026**  
Ananth Bharathwaj Thirumalai Anandanpillai · Mithila Krishna Shashikala

---

## Overview

Binary classification task: predict whether a U.S. domestic flight will be delayed ≥15 minutes using only pre-departure schedule information (no weather, no real-time data).

Best result: **holdout AUC 0.6517** (XGBoost + Optuna). Performance is bounded at ~0.65 by the pre-departure feature constraint — literature benchmarks of 0.78–0.85 use METAR weather and FAA ground-delay data.

---

## Dataset

[2015 U.S. DOT/BTS On-Time Performance](https://www.kaggle.com/datasets/usdot/flight-delays) — download `flights.csv` and place it in the project root before running. The notebook subsamples 200,000 flights via stratified sampling; the full file (~500 MB) is not included in this repo.

---

## Results

| Model | Holdout AUC | Holdout F1 |
|---|---|---|
| XGBoost (Optuna) | **0.6517** | 0.369 |
| CatBoost | 0.6514 | 0.368 |
| LightGBM | 0.6492 | 0.366 |
| Stacking (XGB+LGB+RF) | 0.6477 | 0.366 |
| Random Forest | 0.6465 | 0.326 |
| Baselines (LR, NB) | ~0.51 | — |

Key finding: standard cross-validation inflated AUC by **+0.084** due to target-encoding leakage. A custom `run_cv_leakfree()` harness corrects this by re-deriving all target-encoded features inside each fold.

---

## Repository Structure

```
├── MLPR_Project.ipynb          # Main notebook — full pipeline
├── data/
│   ├── X_train_enc.parquet     # Encoded training features (160k rows)
│   ├── X_test_enc.parquet      # Encoded test features (40k rows)
│   ├── y_train.parquet
│   ├── y_test.parquet
│   ├── results_all_models.csv  # Full model comparison table
│   └── threshold_optimization.csv
├── artifacts/
│   ├── encoding_maps.pkl       # Fitted target-encoding maps
│   ├── inference_bundle.joblib # Serialised best model
│   └── lightgbm.joblib
├── attachments/                # All generated figures (PNG)
├── slides.tex                  # LaTeX presentation source
└── project_requirements.pdf
```

---

## Reproducing the Results

### Environment

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna imbalanced-learn shap pyarrow joblib matplotlib seaborn
```

The notebook detects Google Colab automatically and adjusts paths. To run locally, no changes are needed — outputs are written relative to the notebook location.

### Running

Open `MLPR_Project.ipynb` and run all cells in order. Pre-encoded data is available in `data/` so the full pipeline (including encoding) can be skipped by loading the parquet files directly if needed.

Approximate runtimes on CPU:
- EDA + preprocessing: ~2 min
- Baseline models: ~3 min
- XGBoost Optuna tuning (50 trials): ~15 min
- Full notebook end-to-end: ~45 min

---

## Key Engineered Features

| Feature | Description |
|---|---|
| `ROUTE_DELAY_RATE` | Historical delay rate per origin–destination pair (top SHAP predictor) |
| `HOUR_AIRLINE` | Smoothed target encoding of carrier × departure hour |
| `ORIGIN_HOUR_VOLUME` | Flight count at origin airport per hour (congestion proxy, no target signal) |
| `CARRIER_DELAY_RATE` | Historical delay rate per airline |
| Cyclic encodings | Sin/cos pairs for hour, day-of-week, month |