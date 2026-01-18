# Quick Start Guide - Football Prediction Pipeline

## Prerequisites
- Python 3.8+
- Sportmonks API key (sign up at https://www.sportmonks.com/)

## Step-by-Step Execution

### 1. Setup (One-time)

```bash
# Clone/navigate to project directory
cd /path/to/modeling_pipeline

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `config.py` and add your API key:
```python
SPORTMONKS_API_KEY = "your_api_key_here"
```

### 3. Run Complete Pipeline

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run all steps sequentially
python 01_sportmonks_data_collection.py && \
python 02_sportmonks_feature_engineering.py && \
python 04_model_baseline_elo.py && \
python 05_model_dixon_coles.py && \
python 06_model_xgboost.py && \
python 07_model_ensemble.py && \
python 08_evaluation.py
```

**Total time:** ~10-12 minutes (6-8 min for data collection, 1 min for training)

## Step-by-Step Breakdown

If you prefer to run steps individually:

```bash
# Step 1: Collect data from Sportmonks API (~6-8 minutes (OPTIMIZED - 3-4x faster!))
python 01_sportmonks_data_collection.py

# Step 2: Generate 465 features (~30 seconds)
python 02_sportmonks_feature_engineering.py

# Step 3: Train Elo baseline model (~2 seconds)
python 04_model_baseline_elo.py

# Step 4: Train Dixon-Coles model (~5 seconds)
python 05_model_dixon_coles.py

# Step 5: Train XGBoost model (~5 seconds)
python 06_model_xgboost.py

# Step 6: Create ensemble (~3 seconds)
python 07_model_ensemble.py

# Step 7: Evaluate all models (~15 seconds)
python 08_evaluation.py
```

## Expected Outputs

### After Data Collection:
```
data/raw/sportmonks/
├── fixtures.csv      (3,040 matches)
├── lineups.csv       (111,805 player records)
├── events.csv        (41,501 match events)
├── sidelined.csv     (24,827 injury records)
└── standings.csv     (160 season standings)
```

### After Feature Engineering:
```
data/processed/
└── sportmonks_features.csv  (2,877 matches × 465 features)
```

### After Training:
```
models/
├── xgboost_model.joblib
├── elo_model.joblib
├── dixon_coles_model.joblib
└── ensemble_model.joblib
```

### After Evaluation:
```
models/evaluation/
├── calibration_plots/
├── performance_by_season.csv
├── performance_by_league.csv
└── betting_simulation.csv
```

## Expected Performance

**XGBoost (Primary Model):**
- **Test Log Loss:** 0.998
- **Test Accuracy:** 56.25%
- **Market Log Loss:** 1.476
- **Edge:** -0.478 (47.8% better than market!)

## Verification Commands

Check if everything worked:

```bash
# Check data files exist
ls -lh data/raw/sportmonks/
ls -lh data/processed/

# Check model files exist
ls -lh models/*.joblib

# View feature engineering log
tail -50 feature_engineering.log

# View data collection log
tail -50 collection_2018_2026.log
```

## Update with New Data

To refresh with latest matches:

```bash
source venv/bin/activate
python 01_sportmonks_data_collection.py
python 02_sportmonks_feature_engineering.py
python 06_model_xgboost.py
```

## Troubleshooting

**Problem:** API key error
```bash
# Solution: Verify API key in config.py
grep SPORTMONKS_API_KEY config.py
```

**Problem:** Missing dependencies
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Problem:** Rate limit exceeded
```
# Solution: Wait - script handles this automatically
# It will retry with exponential backoff
```

**Problem:** Import errors
```bash
# Solution: Ensure virtual environment is activated
which python  # Should show venv/bin/python
```

## Quick Reference

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| 01_sportmonks_data_collection.py | Fetch API data | 6-8 min | 5 CSV files |
| 02_sportmonks_feature_engineering.py | Create features | 30 sec | 465 features |
| 04_model_baseline_elo.py | Train Elo | 2 sec | elo_model.joblib |
| 05_model_dixon_coles.py | Train Dixon-Coles | 5 sec | dixon_coles_model.joblib |
| 06_model_xgboost.py | Train XGBoost | 5 sec | xgboost_model.joblib |
| 07_model_ensemble.py | Combine models | 3 sec | ensemble_model.joblib |
| 08_evaluation.py | Evaluate & analyze | 15 sec | Evaluation reports |

## Support

For detailed documentation, see [README.md](README.md)

For issues, check the **Troubleshooting** section in README.md
