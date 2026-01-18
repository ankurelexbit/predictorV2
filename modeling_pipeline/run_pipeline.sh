#!/bin/bash

# Football Prediction Pipeline Runner
# Usage: ./run_pipeline.sh [full|update|predict]

set -e  # Exit on error

echo "======================================"
echo "Football Prediction Pipeline"
echo "======================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found. Make sure dependencies are installed."
fi

# Default to full run
MODE=${1:-full}

case $MODE in
    full)
        echo "Running FULL pipeline..."
        echo ""
        
        echo "Step 1/9: Collecting data..."
        python 01_data_collection.py
        
        echo -e "\nStep 2/9: Processing raw data..."
        python 02_process_raw_data.py
        
        echo -e "\nStep 3/9: Engineering base features..."
        python 03_feature_engineering.py
        
        echo -e "\nStep 4/9: Adding data-driven features..."
        python 03d_data_driven_features.py
        
        echo -e "\nStep 5/9: Training Elo model..."
        python 04_model_baseline_elo.py
        
        echo -e "\nStep 6/9: Training Dixon-Coles model..."
        python 05_model_dixon_coles.py
        
        echo -e "\nStep 7/9: Training XGBoost model..."
        python 06_model_xgboost.py
        
        echo -e "\nStep 8/9: Creating ensemble..."
        python 07_model_ensemble.py
        
        echo -e "\nStep 9/9: Evaluating models..."
        python 08_evaluation.py
        ;;
        
    update)
        echo "Running UPDATE (models only)..."
        echo ""
        
        echo "Step 1/5: Training Elo model..."
        python 04_model_baseline_elo.py
        
        echo -e "\nStep 2/5: Training Dixon-Coles model..."
        python 05_model_dixon_coles.py
        
        echo -e "\nStep 3/5: Training XGBoost model..."
        python 06_model_xgboost.py
        
        echo -e "\nStep 4/5: Creating ensemble..."
        python 07_model_ensemble.py
        
        echo -e "\nStep 5/5: Evaluating models..."
        python 08_evaluation.py
        ;;
        
    predict)
        echo "Running PREDICTIONS..."
        python 09_prediction_pipeline.py "${@:2}"
        ;;
        
    *)
        echo "Usage: ./run_pipeline.sh [full|update|predict]"
        echo "  full    - Run complete pipeline from data collection"
        echo "  update  - Update models only (assumes features exist)"
        echo "  predict - Make predictions"
        exit 1
        ;;
esac

echo -e "\n======================================"
echo "Pipeline completed successfully!"
echo "======================================"