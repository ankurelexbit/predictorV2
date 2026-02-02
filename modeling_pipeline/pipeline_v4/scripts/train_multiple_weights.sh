#!/bin/bash
# Train 3 models with different class weights in parallel

echo "=============================================================================="
echo "PARALLEL MODEL TRAINING - CLASS WEIGHT COMPARISON"
echo "=============================================================================="
echo ""
echo "Training 3 models with different class weight configurations:"
echo "  Option 1 (Conservative): H=1.0, D=1.3, A=1.0"
echo "  Option 2 (Aggressive):   H=1.3, D=1.5, A=1.2"
echo "  Option 3 (Balanced):     H=1.2, D=1.4, A=1.1"
echo ""
echo "Started: $(date)"
echo ""

# Set data path
DATA_PATH="data/training_data_with_draw_features.csv"

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ ERROR: Training data not found: $DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p models/weight_experiments

# Run 3 training jobs in parallel
echo "🚀 Starting 3 parallel training jobs..."
echo ""

# Option 1: Conservative (reduce draw/away, keep home)
(
    echo "[Option 1] Training Conservative model (H=1.0, D=1.3, A=1.0)..."
    python3 scripts/train_production_model.py \
        --data "$DATA_PATH" \
        --output models/weight_experiments/option1_conservative.joblib \
        --weight-home 1.0 \
        --weight-draw 1.3 \
        --weight-away 1.0 \
        --n-trials 100 \
        > models/weight_experiments/option1_training.log 2>&1
    echo "[Option 1] ✅ Training complete!"
) &
PID1=$!

# Option 2: Aggressive (increase home, keep others)
(
    echo "[Option 2] Training Aggressive model (H=1.3, D=1.5, A=1.2)..."
    python3 scripts/train_production_model.py \
        --data "$DATA_PATH" \
        --output models/weight_experiments/option2_aggressive.joblib \
        --weight-home 1.3 \
        --weight-draw 1.5 \
        --weight-away 1.2 \
        --n-trials 100 \
        > models/weight_experiments/option2_training.log 2>&1
    echo "[Option 2] ✅ Training complete!"
) &
PID2=$!

# Option 3: Balanced (adjust all three)
(
    echo "[Option 3] Training Balanced model (H=1.2, D=1.4, A=1.1)..."
    python3 scripts/train_production_model.py \
        --data "$DATA_PATH" \
        --output models/weight_experiments/option3_balanced.joblib \
        --weight-home 1.2 \
        --weight-draw 1.4 \
        --weight-away 1.1 \
        --n-trials 100 \
        > models/weight_experiments/option3_training.log 2>&1
    echo "[Option 3] ✅ Training complete!"
) &
PID3=$!

# Wait for all jobs to complete
wait $PID1 $PID2 $PID3

echo ""
echo "=============================================================================="
echo "✅ ALL TRAINING JOBS COMPLETE"
echo "=============================================================================="
echo ""
echo "Completed: $(date)"
echo ""
echo "Models saved in: models/weight_experiments/"
echo "  - option1_conservative.joblib"
echo "  - option2_aggressive.joblib"
echo "  - option3_balanced.joblib"
echo ""
echo "Logs saved in: models/weight_experiments/"
echo "  - option1_training.log"
echo "  - option2_training.log"
echo "  - option3_training.log"
echo ""
echo "Next step: Run comparison analysis"
echo "  python3 scripts/compare_model_distributions.py"
echo ""
