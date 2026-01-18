#!/bin/bash

# Football Prediction Pipeline - Complete Execution Script
# This script runs the entire modeling pipeline from data collection to evaluation

set -e  # Exit on any error

echo "============================================================"
echo "FOOTBALL PREDICTION PIPELINE"
echo "============================================================"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Check if API key is configured
if grep -q "your_api_key_here" config.py 2>/dev/null; then
    echo "⚠️  API key not configured!"
    echo "Please edit config.py and add your Sportmonks API key"
    exit 1
fi

echo "✓ API key configured"
echo "Starting pipeline execution..."
echo ""

# Step 1: Data Collection
echo "============================================================"
echo "STEP 1/7: Collecting data from Sportmonks API"
echo "============================================================"
echo "This will take ~25 minutes (API rate limited)"
echo ""
python 01_sportmonks_data_collection.py 2>&1 | tee collection_$(date +%Y%m%d).log
echo "✓ Data collection complete"
echo ""

# Step 2: Feature Engineering
echo "============================================================"
echo "STEP 2/7: Generating features"
echo "============================================================"
python 02_sportmonks_feature_engineering.py 2>&1 | tee feature_engineering_$(date +%Y%m%d).log
echo "✓ Feature engineering complete"
echo ""

# Step 3: Elo Model
echo "============================================================"
echo "STEP 3/7: Training Elo baseline model"
echo "============================================================"
python 04_model_baseline_elo.py
echo "✓ Elo model trained"
echo ""

# Step 4: Dixon-Coles Model
echo "============================================================"
echo "STEP 4/7: Training Dixon-Coles model"
echo "============================================================"
python 05_model_dixon_coles.py
echo "✓ Dixon-Coles model trained"
echo ""

# Step 5: XGBoost Model
echo "============================================================"
echo "STEP 5/7: Training XGBoost model"
echo "============================================================"
python 06_model_xgboost.py
echo "✓ XGBoost model trained"
echo ""

# Step 6: Ensemble
echo "============================================================"
echo "STEP 6/7: Creating ensemble model"
echo "============================================================"
python 07_model_ensemble.py
echo "✓ Ensemble model created"
echo ""

# Step 7: Evaluation
echo "============================================================"
echo "STEP 7/7: Evaluating models"
echo "============================================================"
python 08_evaluation.py
echo "✓ Evaluation complete"
echo ""

# Summary
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Models saved to: models/"
echo "  - Evaluation results: models/evaluation/"
echo "  - Logs: *.log files"
echo ""
echo "Expected XGBoost performance:"
echo "  - Test Log Loss: ~0.998"
echo "  - Test Accuracy: ~56.25%"
echo "  - Edge over market: ~47.8%"
echo ""
echo "Next steps:"
echo "  - Review evaluation results in models/evaluation/"
echo "  - Check model performance metrics"
echo "  - Use models for predictions on new matches"
echo ""
