# Pipeline Workflow

## Complete Flow Diagram

```
┌─────────────────────────┐
│ 01_data_collection.py   │
│ Downloads raw CSV files │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 02_process_raw_data.py  │
│ Creates matches.csv     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 03_feature_engineering  │
│ Base features (Elo,     │
│ form, H2H, rest days)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 03d_data_driven_features│
│ Advanced features from  │
│ data patterns           │
└───────────┬─────────────┘
            │
            ▼
      ┌─────┴─────┬─────────┬──────────┐
      │           │         │          │
      ▼           ▼         ▼          ▼
┌──────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐
│ 04_elo   │ │ 05_dixon │ │ 06_xgb │ │ 07_ens  │
│ baseline │ │  coles   │ │ boost  │ │ emble   │
└──────────┘ └──────────┘ └────────┘ └─────────┘
      │           │         │          │
      └─────┬─────┴─────────┴──────────┘
            │
            ▼
┌─────────────────────────┐
│ 08_evaluation.py        │
│ Compare all models      │
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│ 09_prediction_pipeline  │
│ Make new predictions    │
└─────────────────────────┘
```

## File Dependencies

### Input/Output Chain

1. **01_data_collection.py**
   - Input: None
   - Output: `data/raw/football_data_uk/*.csv`

2. **02_process_raw_data.py**
   - Input: `data/raw/football_data_uk/*.csv`
   - Output: `data/processed/matches.csv`

3. **03_feature_engineering.py**
   - Input: `data/processed/matches.csv`
   - Output: 
     - `data/processed/features.csv`
     - `data/processed/elo_ratings.csv`

4. **03d_data_driven_features.py**
   - Input: 
     - `data/processed/matches.csv`
     - `data/processed/features.csv`
   - Output: `data/processed/features_data_driven.csv`

5. **Model Scripts (04-07)**
   - Input: `data/processed/features_data_driven.csv`
   - Output: `models/*.pkl`

6. **08_evaluation.py**
   - Input: 
     - `data/processed/features_data_driven.csv`
     - `models/*.pkl`
   - Output: Console metrics + plots

## Quick Commands

### Full Pipeline
```bash
# Complete run from scratch
./run_pipeline.sh

# Or manually:
python 01_data_collection.py && \
python 02_process_raw_data.py && \
python 03_feature_engineering.py && \
python 03d_data_driven_features.py && \
python 04_model_baseline_elo.py && \
python 05_model_dixon_coles.py && \
python 06_model_xgboost.py && \
python 07_model_ensemble.py && \
python 08_evaluation.py
```

### Update Only
```bash
# If you already have features, just retrain models
python 04_model_baseline_elo.py && \
python 05_model_dixon_coles.py && \
python 06_model_xgboost.py && \
python 07_model_ensemble.py && \
python 08_evaluation.py
```

### Predictions Only
```bash
# Make predictions with existing models
python 09_prediction_pipeline.py --date tomorrow
```