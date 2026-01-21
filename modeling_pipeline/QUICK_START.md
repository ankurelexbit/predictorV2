# Football Prediction System - Quick Start Guide

## 🎯 Daily & Weekly Commands

This guide shows you exactly what to run for daily predictions and weekly model updates.

---

## 📅 DAILY: Live Predictions

Run this **every day** (or multiple times per day) to get betting recommendations:

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Get today's predictions
venv/bin/python run_live_predictions.py
```

**Output**: Betting recommendations saved to `data/predictions/recommendations_YYYYMMDD_HHMM.json`

**What it does**:
- Fetches upcoming matches (next 24 hours)
- Generates 72 features per match (including league)
- Makes predictions with retrained model
- Applies optimal thresholds (H=0.48, D=0.35, A=0.45)
- Shows betting recommendations

**Expected**: 5-10 bets per day, 25% ROI, 64% win rate

---

## 📆 WEEKLY: Model Refresh

Run this **every Sunday** (or weekly) to update the model with latest data:

### Option 1: Automated Script (Recommended)

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Run complete weekly pipeline
scripts/weekly_model_retraining.sh
```

**What it does**:
1. Fetches last 7 days of data
2. Updates features (preserves league_id)
3. Retrains model with hyperparameter tuning
4. Recalibrates thresholds
5. Validates performance
6. Deploys new model if good

**Time**: ~15-20 minutes

---

### Option 2: Manual Steps

If you prefer to run steps individually:

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Step 1: Fetch latest data (last 7 days)
venv/bin/python scripts/fetch_latest_data.py --days 7

# Step 2: Update features (includes league_id)
venv/bin/python 02_sportmonks_feature_engineering.py

# Step 3: Retrain model (30 trials of hyperparameter tuning)
venv/bin/python tune_for_draws.py

# Step 4: Recalibrate thresholds
venv/bin/python scripts/recalibrate_thresholds_weekly.py

# Step 5: Validate performance
venv/bin/python scripts/validate_weekly_model.py
```

---

## 🤖 AUTOMATED: Set Up Cron Jobs

### Daily Predictions (Every 6 hours)

```bash
# Edit crontab
crontab -e

# Add this line (runs at 6am, 12pm, 6pm, 12am)
0 6,12,18,0 * * * cd /Users/ankurgupta/code/predictorV2/modeling_pipeline && venv/bin/python run_live_predictions.py >> logs/daily_predictions.log 2>&1
```

### Weekly Model Refresh (Every Sunday at 2 AM)

```bash
# Edit crontab
crontab -e

# Add this line
0 2 * * 0 /Users/ankurgupta/code/predictorV2/modeling_pipeline/scripts/weekly_model_retraining.sh >> /Users/ankurgupta/code/predictorV2/modeling_pipeline/logs/weekly_training.log 2>&1
```

---

## 🔍 Check Status

### View Latest Predictions

```bash
# Show latest recommendations
cat data/predictions/recommendations_*.json | tail -1 | python -m json.tool

# Count today's bets
ls -1 data/predictions/recommendations_$(date +%Y%m%d)*.json 2>/dev/null | wc -l
```

### Check Model Info

```bash
# Check model features
venv/bin/python -c "
from predict_live import load_models
model = load_models('xgboost')['xgboost']
print(f'Model features: {len(model.feature_columns)}')
print(f'Has league_id: {\"league_id\" in model.feature_columns}')
"

# Check current thresholds
venv/bin/python -c "
from production_thresholds import get_production_thresholds
print(get_production_thresholds())
"
```

### View Logs

```bash
# Daily predictions log
tail -f logs/live_predictions.log

# Weekly training log
tail -f logs/weekly_training_*.log
```

---

## 📊 Expected Performance

**Current Model** (Retrained with balanced weights + league feature):
- **ROI**: 25-28%
- **Win Rate**: 64-66%
- **Bets/Day**: 7-8
- **Features**: 72 (includes league_id)
- **Thresholds**: H=0.48, D=0.35, A=0.45

**Bet Distribution**:
- Home: 62%
- Away: 33%
- Draw: 5%

---

## 🚨 Troubleshooting

### No predictions generated

```bash
# Check if fixtures are available
venv/bin/python -c "
from predict_live import LiveFeatureCalculator
calc = LiveFeatureCalculator()
fixtures = calc.get_upcoming_fixtures()
print(f'Found {len(fixtures)} upcoming fixtures')
"
```

### API errors

```bash
# Check API key
grep "SPORTMONKS_API_KEY" predict_live.py

# Test API connection
curl "https://api.sportmonks.com/v3/football/fixtures/between/2026-01-21/2026-01-22?api_token=YOUR_KEY&include=participants"
```

### Model not loading

```bash
# Check model file exists
ls -lh models/xgboost_model_draw_tuned.joblib

# If missing, retrain
venv/bin/python tune_for_draws.py
```

---

## 📁 Important Files

**Production Files** (don't delete):
- `models/xgboost_model_draw_tuned.joblib` - Trained model
- `production_thresholds.py` - Optimal thresholds
- `odds_fetcher.py` - Real-time odds
- `predict_live.py` - Feature generation
- `run_live_predictions.py` - Main prediction script

**Data Files** (don't delete):
- `data/elo_ratings.csv` - Team Elo ratings
- `data/team_standings.csv` - League standings
- `data/player_database.csv` - Player stats
- `data/processed/sportmonks_features.csv` - Training data

**Scripts** (for weekly updates):
- `scripts/fetch_latest_data.py` - Data fetching
- `scripts/weekly_model_retraining.sh` - Automated pipeline
- `02_sportmonks_feature_engineering.py` - Feature engineering
- `tune_for_draws.py` - Model training

---

## 🎯 Quick Reference

| Task | Command | Frequency |
|------|---------|-----------|
| **Get predictions** | `venv/bin/python run_live_predictions.py` | Daily |
| **Update model** | `scripts/weekly_model_retraining.sh` | Weekly |
| **Check status** | `tail -f logs/live_predictions.log` | As needed |
| **View bets** | `cat data/predictions/recommendations_*.json \| tail -1` | As needed |

---

## 🚀 First-Time Setup (One-Time Only)

If you haven't added league feature yet:

```bash
# 1. Add league_id to existing data
venv/bin/python add_league_to_features.py

# 2. Retrain model with league
venv/bin/python tune_for_draws.py

# 3. Test live predictions
venv/bin/python run_live_predictions.py
```

After this, use the daily/weekly commands above!

---

## 💡 Tips

1. **Run daily predictions** in the morning (6-8 AM) to catch early matches
2. **Run weekly refresh** on Sunday morning to include weekend results
3. **Monitor first 100 bets** to confirm performance matches expectations
4. **Check logs regularly** for API errors or issues
5. **Keep backups** of model files before weekly updates

---

## 📞 Support

**Logs**: All logs are in `logs/` directory  
**Predictions**: All predictions in `data/predictions/`  
**Models**: Model backups in `models/xgboost_model_draw_tuned_backup_*.joblib`

---

**You're ready to run!** 🎉

**Daily**: `venv/bin/python run_live_predictions.py`  
**Weekly**: `scripts/weekly_model_retraining.sh`
