# Football Prediction System - Complete End-to-End Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Daily Operations](#daily-operations)
3. [Weekly Operations](#weekly-operations)
4. [From-Scratch Setup](#from-scratch-setup)
5. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
6. [File Reference](#file-reference)

---

## 🎯 System Overview

**Current Model**:
- **Features**: 477 (includes league_id for league-specific patterns)
- **Training**: XGBoost with balanced class weights (Draw 1.5x, Away 1.3x)
- **Optimization**: 30 trials of hyperparameter tuning
- **Calibration**: Isotonic Regression for probability calibration
- **Thresholds**: Home=0.48, Draw=0.35, Away=0.45

**Expected Performance**:
- **ROI**: 25-28%
- **Win Rate**: 64-66%
- **Bets/Day**: 7-8
- **Bet Distribution**: 62% Home, 33% Away, 5% Draw

---

## 📅 DAILY OPERATIONS

### Get Live Predictions

Run this **daily** (or multiple times per day) to get betting recommendations:

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Get predictions for upcoming matches
venv/bin/python run_live_predictions.py
```

**What it does**:
1. Fetches upcoming fixtures (next 24 hours) from SportMonks API
2. For each match:
   - Fetches real-time odds from 19 bookmakers
   - Generates 477 features (including league_id)
   - Makes prediction with calibrated probabilities
   - Applies optimal thresholds
   - Uses best available odds for PnL calculation
3. Saves recommendations to `data/predictions/recommendations_YYYYMMDD_HHMM.json`
4. Displays betting recommendations in console

**Output Example**:
```
================================================================================
LIVE PREDICTION RUN
================================================================================
Time: 2026-01-21 00:31:09
Thresholds: H=0.48, D=0.35, A=0.45

Found 31 upcoming fixtures

[1/31] AZ vs Excelsior
  League: Eredivisie
  ✅ Generated 477 features (includes league_id=564)
  🎯 Predictions: H=51.7% D=27.0% A=21.2%
  💰 RECOMMENDATION: Bet HOME @ 51.7% confidence
  Best Odds: 1.95 (potential profit: $95)

[2/31] Galatasaray vs Atlético Madrid
  League: Champions League
  ✅ Generated 477 features (includes league_id=8)
  🎯 Predictions: H=47.9% D=30.0% A=22.1%
  ⏭️  No bet (no threshold exceeded)

...

SUMMARY:
  Fixtures processed: 31
  Recommendations: 17 bets
  Bet distribution: 14 Home, 3 Away, 0 Draw
```

**View Results**:
```bash
# View latest recommendations
cat data/predictions/recommendations_*.json | tail -1 | python -m json.tool

# Check logs
tail -f logs/live_predictions.log
```

---

## 📆 WEEKLY OPERATIONS

### Option 1: Automated Script (Recommended)

Run the complete weekly pipeline:

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Run complete weekly update
scripts/weekly_model_retraining.sh
```

**What it does**:
1. **Fetch Latest Data** (last 7 days)
   - Calls SportMonks API for new matches
   - Saves to `data/raw/sportmonks/`
   - Includes `league_id`, lineups, events, sidelined ✅

2. **Update Features**
   - Runs `02_sportmonks_feature_engineering.py`
   - Preserves league_id ✅
   - Generates 477 features per match
   - Updates `data/processed/sportmonks_features.csv`

3. **Retrain Model**
   - Runs `tune_for_draws.py`
   - 30 trials of hyperparameter optimization
   - Trains with balanced class weights
   - Applies Isotonic Regression calibration
   - Saves to `models/xgboost_model_draw_tuned.joblib`

4. **Recalibrate Thresholds**
   - Tests model on recent data
   - Optimizes thresholds for maximum ROI
   - Updates `production_thresholds.py` if better

5. **Validate Performance**
   - Tests on validation set
   - Checks ROI, win rate, bet frequency
   - Deploys only if performance is good
   - Keeps backup of old model

6. **Generate Report**
   - Creates weekly performance summary
   - Logs all metrics

**Time**: 15-20 minutes

**Output**: Log saved to `logs/weekly_training_YYYYMMDD_HHMM.log`

---

### Option 2: Manual Steps

If you prefer to run steps individually:

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Step 1: Fetch latest data (last 7 days)
venv/bin/python 01_sportmonks_data_collection.py --update --days 7

# Step 2: Update features (preserves league_id)
venv/bin/python 02_sportmonks_feature_engineering.py

# Step 3: Retrain model (hyperparameter tuning + calibration)
venv/bin/python tune_for_draws.py

# Step 4: Recalibrate thresholds (optional but recommended)
venv/bin/python scripts/recalibrate_thresholds_weekly.py

# Step 5: Validate performance
venv/bin/python scripts/validate_weekly_model.py
```

**When to use manual steps**:
- You want to inspect each step
- You want to skip threshold recalibration
- You're testing changes

---

## 🔄 FROM-SCRATCH SETUP

### Complete Data Re-fetch (2-3 hours)

If you want to rebuild everything from scratch:

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Step 1: Fetch ALL historical data (2-3 hours)
venv/bin/python 01_sportmonks_data_collection.py --full --min-year 2016

# Step 2: Process features (includes league_id automatically)
venv/bin/python 02_sportmonks_feature_engineering.py

# Step 3: Train model (30 trials + calibration)
venv/bin/python tune_for_draws.py

# Step 4: Optimize thresholds
venv/bin/python scripts/recalibrate_thresholds_weekly.py

# Step 5: Test live predictions
venv/bin/python run_live_predictions.py
```

**When to do this**:
- First-time setup
- Major data corruption
- Want to retrain on all historical data
- Testing new features

**Note**: The script automatically fetches fixtures, lineups, events, and sidelined data with `league_id` preserved!

---

## 🤖 AUTOMATED SETUP (Cron Jobs)

### Daily Predictions

Run predictions **4 times per day** (6am, 12pm, 6pm, 12am):

```bash
# Edit crontab
crontab -e

# Add this line
0 6,12,18,0 * * * cd /Users/ankurgupta/code/predictorV2/modeling_pipeline && venv/bin/python run_live_predictions.py >> logs/daily_predictions.log 2>&1
```

### Weekly Model Refresh

Run model update **every Sunday at 2 AM**:

```bash
# Edit crontab
crontab -e

# Add this line
0 2 * * 0 /Users/ankurgupta/code/predictorV2/modeling_pipeline/scripts/weekly_model_retraining.sh >> /Users/ankurgupta/code/predictorV2/modeling_pipeline/logs/weekly_training.log 2>&1
```

### View Cron Jobs

```bash
# List all cron jobs
crontab -l

# Check if cron is running
ps aux | grep cron
```

---

## 🔍 MONITORING & TROUBLESHOOTING

### Check System Status

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# 1. Check model info
venv/bin/python -c "
from predict_live import load_models
model = load_models('xgboost')['xgboost']
print(f'✅ Model loaded')
print(f'   Features: {len(model.feature_columns)}')
print(f'   Has league_id: {\"league_id\" in model.feature_columns}')
"

# 2. Check current thresholds
venv/bin/python -c "
from production_thresholds import get_production_thresholds, EXPECTED_PERFORMANCE
thresholds = get_production_thresholds()
print(f'✅ Thresholds: H={thresholds[\"home\"]:.2f}, D={thresholds[\"draw\"]:.2f}, A={thresholds[\"away\"]:.2f}')
print(f'   Expected ROI: {EXPECTED_PERFORMANCE[\"roi\"]:.1f}%')
print(f'   Expected Win Rate: {EXPECTED_PERFORMANCE[\"win_rate\"]:.1f}%')
"

# 3. Check data files
ls -lh data/processed/sportmonks_features.csv
ls -lh models/xgboost_model_draw_tuned.joblib
ls -lh data/elo_ratings.csv
ls -lh data/team_standings.csv

# 4. Check if league_id is in features
head -1 data/processed/sportmonks_features.csv | grep -o "league_id" && echo "✅ league_id present" || echo "❌ league_id missing"
```

### View Logs

```bash
# Live predictions log
tail -f logs/live_predictions.log

# Weekly training log (latest)
tail -f logs/weekly_training_*.log

# All logs
ls -lht logs/ | head -20
```

### Common Issues

**Issue 1: No predictions generated**
```bash
# Check upcoming fixtures
venv/bin/python -c "
from predict_live import LiveFeatureCalculator
calc = LiveFeatureCalculator()
fixtures = calc.get_upcoming_fixtures()
print(f'Found {len(fixtures)} upcoming fixtures')
for f in fixtures[:5]:
    print(f'  - {f[\"home_team_name\"]} vs {f[\"away_team_name\"]}')
"
```

**Issue 2: API errors**
```bash
# Test API connection
curl "https://api.sportmonks.com/v3/football/fixtures/between/2026-01-21/2026-01-22?api_token=YOUR_API_KEY&include=participants" | head -50

# Check API key in code
grep "API_KEY" predict_live.py | head -5
```

**Issue 3: Model not loading**
```bash
# Check model file
ls -lh models/xgboost_model_draw_tuned.joblib

# If missing, retrain
venv/bin/python tune_for_draws.py
```

**Issue 4: league_id missing**
```bash
# Add league_id to existing data
venv/bin/python add_league_to_features.py

# Retrain model
venv/bin/python tune_for_draws.py
```

**Issue 5: Low bet frequency**
```bash
# Check thresholds (may be too high)
venv/bin/python -c "
from production_thresholds import get_production_thresholds
print(get_production_thresholds())
"

# Lower thresholds if needed (edit production_thresholds.py)
# Or run threshold recalibration
venv/bin/python scripts/recalibrate_thresholds_weekly.py
```

---

## 📁 FILE REFERENCE

### Essential Production Files

**Core Scripts**:
- `run_live_predictions.py` - Main daily prediction script
- `predict_live.py` - Live feature generation (includes league_id)
- `odds_fetcher.py` - Real-time odds fetching
- `production_thresholds.py` - Optimal betting thresholds
- `config.py` - Configuration and paths
- `utils.py` - Helper functions

**Model Files**:
- `06_model_xgboost.py` - XGBoost model class (balanced weights)
- `tune_for_draws.py` - Training script (hyperparameter tuning + calibration)
- `models/xgboost_model_draw_tuned.joblib` - Trained model (477 features)

**Data Files**:
- `data/processed/sportmonks_features.csv` - Training data (with league_id)
- `data/elo_ratings.csv` - Team Elo ratings
- `data/team_standings.csv` - League standings
- `data/player_database.csv` - Player statistics
- `data/raw/sportmonks/fixtures.csv` - Raw fixture data (with league_id)

**Weekly Pipeline**:
- `scripts/weekly_model_retraining.sh` - Automated weekly update
- `01_sportmonks_data_collection.py` - Data fetching (full + weekly updates)
- `02_sportmonks_feature_engineering.py` - Feature engineering (preserves league_id)
- `scripts/recalibrate_thresholds_weekly.py` - Threshold optimization
- `scripts/validate_weekly_model.py` - Performance validation

**One-Time Setup**:
- `add_league_to_features.py` - Add league_id to existing data (one-time)
- `build_player_database.py` - Build player database
- `player_stats_manager.py` - Manage player stats

---

## 🎯 QUICK COMMAND REFERENCE

| Task | Command | Frequency |
|------|---------|-----------|
| **Daily predictions** | `venv/bin/python run_live_predictions.py` | Daily (or 4x/day) |
| **Weekly update** | `scripts/weekly_model_retraining.sh` | Weekly (Sunday) |
| **Check status** | `tail -f logs/live_predictions.log` | As needed |
| **View predictions** | `cat data/predictions/recommendations_*.json \| tail -1` | As needed |
| **Retrain model** | `venv/bin/python tune_for_draws.py` | Weekly (automated) |
| **Fetch data** | `venv/bin/python 01_sportmonks_data_collection.py --update --days 7` | Weekly (automated) |
| **Test API** | `venv/bin/python -c "from predict_live import LiveFeatureCalculator; ..."` | As needed |

---

## 📊 EXPECTED WORKFLOW

### Daily Routine

```bash
# Morning (6-8 AM)
venv/bin/python run_live_predictions.py

# Check recommendations
cat data/predictions/recommendations_*.json | tail -1 | python -m json.tool

# Place bets based on recommendations
# Monitor results
```

### Weekly Routine (Sunday Morning)

```bash
# Automated (cron runs at 2 AM)
# OR manual:
scripts/weekly_model_retraining.sh

# Check logs
tail logs/weekly_training_*.log

# Verify new model
venv/bin/python -c "
from predict_live import load_models
model = load_models('xgboost')['xgboost']
print(f'Model features: {len(model.feature_columns)}')
"

# Test predictions
venv/bin/python run_live_predictions.py
```

---

## ✅ VERIFICATION CHECKLIST

After setup or weekly update, verify:

- [ ] Model has 477 features (includes league_id)
- [ ] Thresholds are set (H=0.48, D=0.35, A=0.45)
- [ ] Data files exist (features, elo, standings, players)
- [ ] Live predictions generate recommendations
- [ ] Logs show no errors
- [ ] API calls are working
- [ ] Cron jobs are scheduled (if using automation)

---

## 🎉 YOU'RE READY!

**Daily**: `venv/bin/python run_live_predictions.py`  
**Weekly**: `scripts/weekly_model_retraining.sh`  

**Expected Results**:
- 7-8 bets per day
- 25-28% ROI
- 64-66% win rate
- Balanced Home/Away/Draw distribution

**Good luck with your predictions!** 🚀
