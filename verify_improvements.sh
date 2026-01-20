#!/bin/bash
# Quick verification script for high-impact improvements

set -e  # Exit on error

echo "=========================================="
echo "VERIFYING HIGH-IMPACT IMPROVEMENTS"
echo "=========================================="

cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Step 1: Testing Feature Engineering with EMA and Rest Days..."
echo "--------------------------------------------------------------"
python 02_sportmonks_feature_engineering.py

echo ""
echo "Step 2: Training XGBoost with Expanded Hyperparameter Grid..."
echo "--------------------------------------------------------------"
echo "Note: Using 10 trials for quick verification (use --n-trials 30 for full tuning)"
python 06_model_xgboost.py --tune --n-trials 10

echo ""
echo "Step 3: Training Ensemble with Auto Weight Optimization..."
echo "--------------------------------------------------------------"
python 07_model_ensemble.py

echo ""
echo "=========================================="
echo "VERIFICATION COMPLETE!"
echo "=========================================="
echo ""
echo "Check the following files for results:"
echo "  - models/model_comparison.csv (performance metrics)"
echo "  - models/xgboost_model.joblib (trained model)"
echo "  - models/ensemble_model.joblib (optimized ensemble)"
echo ""
echo "Key metrics to look for:"
echo "  ✓ Log Loss < 0.95"
echo "  ✓ Draw predictions > 0"
echo "  ✓ Accuracy > 55%"
