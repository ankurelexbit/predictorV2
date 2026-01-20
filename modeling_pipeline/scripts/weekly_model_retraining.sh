#!/bin/bash
# Weekly Model Retraining Pipeline
# Runs every Sunday at 2 AM
# 
# This script:
# 1. Fetches latest data (last 7 days)
# 2. Updates features
# 3. Retrains the draw-tuned XGBoost model
# 4. Recalibrates thresholds
# 5. Validates performance
# 6. Deploys new model if performance is good

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_FILE="logs/weekly_training_$(date +%Y%m%d_%H%M).log"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "WEEKLY MODEL RETRAINING PIPELINE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 1: Fetch Latest Data
echo "STEP 1: Fetching latest data..." | tee -a "$LOG_FILE"
venv/bin/python scripts/fetch_latest_data.py --days 7 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Data fetch failed!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ Data fetch complete" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 2: Update Features
echo "STEP 2: Updating features..." | tee -a "$LOG_FILE"
venv/bin/python 01_feature_engineering.py 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Feature engineering failed!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ Features updated" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 2.5: Update Player Stats Database
echo "STEP 2.5: Updating player stats database..." | tee -a "$LOG_FILE"
venv/bin/python scripts/update_player_stats.py 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "⚠️  Player stats update failed (non-critical)" | tee -a "$LOG_FILE"
else
    echo "✅ Player stats updated" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

# Step 3: Retrain Draw-Tuned XGBoost Model (includes hyperparameter tuning)
echo "STEP 3: Retraining draw-tuned XGBoost model with hyperparameter tuning..." | tee -a "$LOG_FILE"
echo "   This includes 30 trials of hyperparameter search" | tee -a "$LOG_FILE"
venv/bin/python tune_for_draws.py 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Model training failed!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ Model retrained" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 4: Recalibrate Thresholds
echo "STEP 4: Recalibrating thresholds..." | tee -a "$LOG_FILE"
venv/bin/python scripts/recalibrate_thresholds_weekly.py 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Threshold recalibration failed!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ Thresholds recalibrated" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 5: Validate Performance
echo "STEP 5: Validating performance..." | tee -a "$LOG_FILE"
venv/bin/python scripts/validate_weekly_model.py 2>&1 | tee -a "$LOG_FILE"

VALIDATION_EXIT=$?

if [ $VALIDATION_EXIT -eq 0 ]; then
    echo "✅ Validation passed - deploying new model" | tee -a "$LOG_FILE"
    
    # Backup old model
    cp models/xgboost_model_draw_tuned.joblib models/xgboost_model_draw_tuned_backup_$(date +%Y%m%d).joblib
    
    # Deploy new model (already saved by tune_for_draws.py)
    echo "✅ New model deployed" | tee -a "$LOG_FILE"
    
elif [ $VALIDATION_EXIT -eq 2 ]; then
    echo "⚠️  Validation warning - model performance acceptable but below target" | tee -a "$LOG_FILE"
    echo "   Deploying with caution..." | tee -a "$LOG_FILE"
    
    # Backup and deploy anyway
    cp models/xgboost_model_draw_tuned.joblib models/xgboost_model_draw_tuned_backup_$(date +%Y%m%d).joblib
    echo "✅ New model deployed with warning" | tee -a "$LOG_FILE"
    
else
    echo "❌ Validation failed - NOT deploying new model" | tee -a "$LOG_FILE"
    echo "   Keeping previous model" | tee -a "$LOG_FILE"
    
    # Restore backup if exists
    LATEST_BACKUP=$(ls -t models/xgboost_model_draw_tuned_backup_*.joblib 2>/dev/null | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        cp "$LATEST_BACKUP" models/xgboost_model_draw_tuned.joblib
        echo "✅ Restored previous model" | tee -a "$LOG_FILE"
    fi
fi

echo "" | tee -a "$LOG_FILE"

# Step 6: Generate Weekly Report
echo "STEP 6: Generating weekly report..." | tee -a "$LOG_FILE"
venv/bin/python scripts/generate_weekly_report.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "WEEKLY RETRAINING COMPLETE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "End Time: $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
