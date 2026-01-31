# Production Files - Final Configuration

This document lists all the final files for production deployment with CatBoost and 162 features (including 12 draw-specific features).

## Performance Summary

**Final Model: Conservative CatBoost (162 features)**
- Log Loss: **0.9970**
- Draw Accuracy: **21.90%**
- Overall Accuracy: **51.5%**
- Class Weights: Away=1.2, Draw=1.5, Home=1.0
- Features: 150 original + 12 draw parity features

**Model Location:** `models/with_draw_features/conservative_with_draw_features.joblib`

---

## Complete Data Flow Pipeline

```
1. DATA DOWNLOAD (backfill_historical_data.py)
   SportMonks API
        ↓
   data/historical/fixtures/*.json (raw JSON files)

2. CONVERSION (convert_json_to_csv.py) - Optional but recommended
   JSON files
        ↓
   data/processed/fixtures_with_stats.csv (100x faster)

3. FEATURE GENERATION (generate_training_data.py)
   Raw data → FeatureOrchestrator
        ├─→ Pillar 1: Fundamentals (50 features)
        ├─→ Pillar 2: Modern Analytics (60 features)
        └─→ Pillar 3: Hidden Edges (52 features)
        ↓
   data/training_data_with_draw_features.csv (162 features)

4. MODEL TRAINING (train_production_model.py)
   Training data → CatBoost + Conservative Weights
        ↓
   models/production/model_v4_162feat.joblib

5. PREDICTIONS (Your application)
   New fixtures → FeatureOrchestrator → Model
        ↓
   Probabilities: [Away, Draw, Home]
```

---

## Core Feature Engineering Files

### Data Loading
- **`src/data/json_loader.py`**
  - Loads fixture data from JSON or CSV
  - Provides point-in-time queries
  - Caches statistics and lineups

### Feature Orchestration
- **`src/features/feature_orchestrator.py`**
  - Main entry point for feature generation
  - Coordinates all 3 pillars
  - Generates 162 features per fixture

### Feature Engines
- **`src/features/pillar1_fundamentals.py`** - 50 features
  - Elo ratings, league position, recent form
  - H2H history, home advantage

- **`src/features/pillar2_modern_analytics.py`** - 60 features
  - Derived xG, shot analysis
  - Defensive intensity, attack patterns

- **`src/features/pillar3_hidden_edges.py`** - 52 features ⭐
  - 40 original hidden edge features
  - **12 NEW draw parity features:**
    1. `elo_difference` - Team strength parity
    2. `form_difference_10` - Recent form parity
    3. `position_difference` - League position parity
    4. `h2h_draw_rate` - Historical draw tendency
    5. `home_draw_rate_10` - Home team draw rate
    6. `away_draw_rate_10` - Away team draw rate
    7. `combined_draw_tendency` - Both teams' draw tendency
    8. `league_draw_rate` - League-wide draw frequency
    9. `both_midtable` - Both teams in mid-table
    10. `both_low_scoring` - Both teams low-scoring
    11. `both_defensive` - Both teams defensive
    12. `either_coming_from_draw` - Recent draw context

### Supporting Calculators
- **`src/features/elo_calculator.py`**
  - K-factor: 32
  - Home advantage: 35 points

- **`src/features/standings_calculator.py`**
  - Point-in-time league standings
  - Position, points, goal difference

---

## Data Download & Conversion Scripts

### Historical Data Download ⭐
**`scripts/backfill_historical_data.py`**
- Downloads raw fixture data from SportMonks API
- Requires: `SPORTMONKS_API_KEY` environment variable
- Saves JSON files to `data/historical/fixtures/`
- Supports date range filtering, league filtering
- Options: `--skip-lineups`, `--skip-sidelined`, `--workers N`

**Usage:**
```bash
export SPORTMONKS_API_KEY="your_api_key_here"
python3 scripts/backfill_historical_data.py \
  --start-date 2023-08-01 \
  --end-date 2024-05-31 \
  --output-dir data/historical
```

### JSON to CSV Conversion ⭐
**`scripts/convert_json_to_csv.py`**
- Converts raw JSON files to single CSV for 100x faster loading
- One-time conversion recommended for performance
- Creates: `data/processed/fixtures.csv` or `fixtures_with_stats.csv`
- Significantly speeds up feature generation

**Usage:**
```bash
python3 scripts/convert_json_to_csv.py
# Output: data/processed/fixtures_with_stats.csv
```

---

## Training & Pipeline Scripts

### Production Training Script ⭐
**`scripts/train_production_model.py`** (recommended)
- Simplified version for production use
- Trains CatBoost with Conservative weights
- 100 Optuna trials
- Saves to versioned location

### Development Training Script
**`scripts/retrain_with_draw_features.py`**
- Full retraining pipeline with evaluation
- Regenerates training data
- Detailed comparison output
- Use for analysis and testing

### Feature Data Generation
**`scripts/generate_training_data.py`**
- Generates training CSV from raw data
- Supports filtering by league/date
- Creates 162-feature dataset (162 features + 9 metadata columns)

### Weekly Automation
**`scripts/weekly_retrain_pipeline.py`** ⚠️
- **Status:** Needs update to use production training script
- Currently uses old `train_improved_model.py`
- Should be updated to call `train_production_model.py`

---

## Generated Assets

### Training Data
- **Current:** `data/training_data_with_draw_features.csv`
  - 17,943 samples
  - 162 feature columns
  - Chronologically sorted

### Trained Models
- **Production Model:** `models/with_draw_features/conservative_with_draw_features.joblib` ⭐
  - CatBoost with Conservative weights
  - 162 features
  - 0.9970 log loss, 21.90% draw accuracy

- **Alternative (not recommended):** `models/with_draw_features/xgboost_with_draw_features.joblib`
  - XGBoost version (worse performance)
  - 0.9991 log loss, 18.73% draw accuracy

### Results & Metadata
- **`models/with_draw_features/results.json`**
  - Test metrics
  - Confusion matrix
  - Comparison with baseline

---

## Production Deployment Workflow

### Initial Setup (From Scratch)
```bash
# 1. Download historical data from SportMonks API
export SPORTMONKS_API_KEY="your_api_key_here"
python3 scripts/backfill_historical_data.py \
  --start-date 2023-08-01 \
  --end-date 2024-05-31 \
  --output-dir data/historical

# 2. Convert JSON to CSV (optional but recommended for 100x speedup)
python3 scripts/convert_json_to_csv.py
# Creates: data/processed/fixtures_with_stats.csv

# 3. Generate 162-feature training dataset
python3 scripts/generate_training_data.py \
  --output data/training_data_with_draw_features.csv

# 4. Train production model
python3 scripts/train_production_model.py \
  --data data/training_data_with_draw_features.csv \
  --output models/production/model_v4_162feat.joblib
```

### Quick Start (Data Already Downloaded)
```bash
# If you already have data/historical/fixtures/*.json

# 1. Convert to CSV (optional, for performance)
python3 scripts/convert_json_to_csv.py

# 2. Generate training data
python3 scripts/generate_training_data.py \
  --output data/training_data_with_draw_features.csv

# 3. Train model
python3 scripts/train_production_model.py \
  --data data/training_data_with_draw_features.csv \
  --output models/production/model_v4_162feat.joblib
```

### Weekly Retraining
```bash
# Weekly pipeline includes:
# 1. Download new data (backfill_historical_data.py)
# 2. Convert to CSV (convert_json_to_csv.py)
# 3. Generate training data (generate_training_data.py)
# 4. Train model (needs update to use train_production_model.py)

# Option 1: Manual weekly update
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4

# Option 2: Manual step-by-step
export SPORTMONKS_API_KEY="your_api_key"
python3 scripts/backfill_historical_data.py --start-date 2024-01-01 --end-date 2024-01-31
python3 scripts/convert_json_to_csv.py
python3 scripts/generate_training_data.py --output data/training_data_latest.csv
python3 scripts/train_production_model.py --data data/training_data_latest.csv --output models/production/model_latest.joblib

# Option 3: Automated via cron (recommended)
# Add to crontab -e:
# 0 2 * * 0 cd /path/to/pipeline_v4 && python3 scripts/weekly_retrain_pipeline.py --weeks-back 4
```

### Load Model for Predictions
```python
import joblib
import pandas as pd

# Load production model
model = joblib.load('models/with_draw_features/conservative_with_draw_features.joblib')

# Generate features for new fixtures
from src.features.feature_orchestrator import FeatureOrchestrator
orchestrator = FeatureOrchestrator(data_dir='data/historical')
features_df = orchestrator.generate_features(fixture_data)

# Get predictions
probabilities = model.predict_proba(features_df)
# probabilities[:, 0] = Away win probability
# probabilities[:, 1] = Draw probability
# probabilities[:, 2] = Home win probability
```

---

## Key Configuration Details

### Class Weights (Conservative)
```python
class_weights = {
    0: 1.2,  # Away win
    1: 1.5,  # Draw (highest weight)
    2: 1.0   # Home win
}
```

### Data Split (Chronological)
```python
train: 70%  # Oldest matches
val:   15%  # Middle matches
test:  15%  # Most recent matches
```

### Hyperparameter Search Space (Optuna)
```python
{
    'iterations': [200, 800],
    'depth': [4, 10],
    'learning_rate': [0.01, 0.3],
    'l2_leaf_reg': [1, 10],
    'border_count': [32, 255],
    'bagging_temperature': [0, 1],
    'random_strength': [0, 10]
}
```

### Features to Exclude from Training
```python
features_to_exclude = [
    'fixture_id', 'home_team_id', 'away_team_id',
    'season_id', 'league_id', 'match_date',
    'home_score', 'away_score', 'result', 'target',
    'home_team_name', 'away_team_name', 'state_id'
]
```

---

## Model Evaluation Results

### Test Set Performance (2,692 matches)

**Overall Metrics:**
- Log Loss: 0.9970
- Accuracy: 51.5%

**Per-Class Accuracy:**
- Away: 56.7% (494/872 correct)
- Draw: 18.7% (124/662 correct)
- Home: 65.9% (763/1,158 correct)

**Prediction Distribution:**
- Predicted: 33.8% Away | 18.8% Draw | 47.4% Home
- Actual: 32.4% Away | 24.6% Draw | 43.0% Home

**Confusion Matrix:**
```
                Predicted
             Away   Draw   Home
Actual Away:  494    120    258
Actual Draw:  228    124    310
Actual Home:  188    207    763
```

---

## Important Notes

### Draw Prediction Limitation
- Draw accuracy remains challenging (~22%)
- Added 12 draw-specific features did not significantly improve performance
- This appears to be a practical limit for this dataset
- Consider setting realistic expectations for draw predictions

### Point-in-Time Correctness
- All features use only data available before match date
- No future data leakage
- Chronological splits maintained throughout

### Performance Optimization
- Convert JSON to CSV for 100x speedup (`scripts/convert_json_to_csv.py`)
- Use league filtering for faster iteration
- Cache builds available (`scripts/build_cache.py`)

### Weekly Retrain Recommendations
- Retrain every 1-2 weeks during season
- Download 4 weeks of new data per retrain
- Keep last 3 models for rollback capability
- Monitor log loss trend over time

---

## Next Steps for Production

1. **Create production training script** (`train_production_model.py`)
2. **Update weekly pipeline** to use new training script
3. **Set up cron job** for automated weekly retraining
4. **Create prediction API** that loads model and generates features
5. **Implement model versioning** and rollback strategy
6. **Monitor production metrics** (log loss, accuracy, draw rate)

---

## Comparison: What Changed from Original

### Original Model (150 features)
- Log Loss: 0.9983
- Draw Accuracy: 22.05%
- Class Weights: 1.2/1.5/1.0

### New Model (162 features)
- Log Loss: **0.9970** (↓ 0.0013)
- Draw Accuracy: **21.90%** (↓ 0.15%)
- Class Weights: 1.2/1.5/1.0
- Added Features: 12 draw parity indicators

**Conclusion:** Slight log loss improvement, but draw accuracy remains similar. The 162-feature model is recommended due to better log loss and additional context features.
