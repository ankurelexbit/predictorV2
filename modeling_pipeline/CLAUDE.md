# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Football match prediction pipeline predicting 1X2 outcomes (Home/Draw/Away) for 6 major European leagues using Elo ratings, Dixon-Coles Poisson models, and XGBoost ensemble.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run full pipeline (data collection through evaluation)
./run_pipeline.sh full

# Retrain models only (when features already exist)
./run_pipeline.sh update

# Make predictions
python 09_prediction_pipeline.py --date 2024-01-20
python 09_prediction_pipeline.py --home "Liverpool" --away "Man City"
```

## Pipeline Architecture

```
01_data_collection.py     → data/raw/*.csv (download from football-data.co.uk)
02_process_raw_data.py    → data/processed/matches.csv
03_feature_engineering.py → data/processed/features.csv (53 features)
03d_data_driven_features.py → data/processed/features_data_driven.csv (73 features)
        ↓
04_model_baseline_elo.py  → models/elo_model.joblib
05_model_dixon_coles.py   → models/dixon_coles_model.joblib
06_model_xgboost.py       → models/xgboost_model.joblib
07_model_ensemble.py      → models/ensemble_model.joblib
        ↓
08_evaluation.py          → Metrics, plots, ROI analysis
09_prediction_pipeline.py → Production predictions
```

**All model scripts (04-08) use `features_data_driven.csv`** which contains both base and advanced features.

## Key Files

- **config.py**: All settings - paths, API keys, league mappings, model hyperparameters, train/val/test season splits
- **utils.py**: Shared utilities - logging, odds conversion, validation metrics
- **data/processed/**: CSV outputs at each pipeline stage
- **models/**: Serialized models (.joblib files)

## Models

1. **Elo**: Simple baseline converting rating differences to probabilities (K=20, home advantage=100)
2. **Dixon-Coles**: Bivariate Poisson with attack/defense strengths and low-score correction (ρ parameter)
3. **XGBoost**: Gradient boosting classifier on all 73 features
4. **Ensemble**: Weighted average (Elo 0.2, Dixon-Coles 0.3, XGBoost 0.5)

## Critical Implementation Notes

- **Time-series validation**: Always use season-based splits (defined in config.py), never random splits
- **Feature file**: Models read from `features_data_driven.csv`, not `features.csv`
- **Probability calibration**: Use isotonic regression after training (see CALIBRATION_METHOD in config)
- **Date parsing**: Use `dayfirst=True` for European date format in CSVs
- **CSV-only workflow**: Default pipeline uses CSV files only; Supabase support is optional

## Configuration (config.py)

```python
# Key settings you may need to adjust
TRAIN_SEASONS = ["2019-2020", "2020-2021", "2021-2022"]
VALIDATION_SEASONS = ["2022-2023"]
TEST_SEASONS = ["2023-2024"]

ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 100
ELO_SEASON_REGRESSION = 0.1  # 10% regression toward mean between seasons

ENSEMBLE_WEIGHTS = {"elo": 0.2, "dixon_coles": 0.3, "xgboost": 0.5}
```

## Supported Leagues

Premier League, Championship (England), La Liga (Spain), Bundesliga (Germany), Serie A (Italy), Ligue 1 (France)

## Expected Performance

- Log Loss: 0.95-1.05
- Accuracy: 50-55% (3-class classification is inherently difficult)
- ROI: +2-5% on betting simulation
