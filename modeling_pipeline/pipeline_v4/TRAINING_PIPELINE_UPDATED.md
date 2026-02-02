# Training Pipeline Update - Option 3 Configuration

**Date:** February 2, 2026
**Status:** ✅ COMPLETE

---

## Summary

The training pipeline has been updated to match the deployed production configuration (Option 3: Balanced). All class weights, model settings, and defaults are now consistent between training and inference.

---

## Configuration Details

### Model: Option 3 (Balanced)

**Class Weights:**
- Away (0): 1.1
- Draw (1): 1.4
- Home (2): 1.2

**Hyperparameters:**
- Trials: 100 (Optuna optimization)
- Random Seed: 42
- Loss Function: MultiClass

**Data Split:**
- Train: 70%
- Validation: 15%
- Test: 15%

---

## Weekly Retraining

### Basic Usage (with default Option 3 weights):

```bash
# Train with latest data
python3 scripts/train_production_model.py \
  --data data/training_data_latest.csv \
  --output models/production/option3_$(date +%Y%m%d).joblib
```

### Custom Hyperparameter Tuning:

```bash
# More thorough tuning (slower but potentially better)
python3 scripts/train_production_model.py \
  --data data/training_data_latest.csv \
  --output models/production/option3_$(date +%Y%m%d).joblib \
  --n-trials 200
```

### Custom Class Weights (if experimenting):

```bash
# Override default Option 3 weights
python3 scripts/train_production_model.py \
  --data data/training_data_latest.csv \
  --output models/production/custom_model.joblib \
  --weight-home 1.3 \
  --weight-draw 1.5 \
  --weight-away 1.0
```

---

## Files Updated

### 1. `scripts/train_production_model.py`

**Changes:**
- Header documentation updated to reflect Option 3 (Balanced)
- `PRODUCTION_CONFIG` class weights changed from (A=1.2, D=1.5, H=1.0) to (A=1.1, D=1.4, H=1.2)
- Added model metadata: name, version, description
- Enhanced logging to show model configuration and class weights
- Argument parser defaults updated to Option 3 values
- Help text clarified to indicate Option 3 defaults

**Before:**
```python
PRODUCTION_CONFIG = {
    'model_name': 'Conservative',
    'class_weights': {0: 1.2, 1: 1.5, 2: 1.0}  # A/D/H
}
# Defaults: --weight-home 1.0, --weight-draw 1.5, --weight-away 1.2
```

**After:**
```python
PRODUCTION_CONFIG = {
    'model_name': 'Option 3: Balanced',
    'version': 'v4.1',
    'class_weights': {0: 1.1, 1: 1.4, 2: 1.2},  # A/D/H
    'description': 'Balanced weights for optimal draw performance'
}
# Defaults: --weight-home 1.2, --weight-draw 1.4, --weight-away 1.1
```

---

## Validation

### Configuration Check:

```bash
# Display current training configuration
python3 -c "
from scripts.train_production_model import PRODUCTION_CONFIG
print('Class Weights:', PRODUCTION_CONFIG['class_weights'])
print('Model Name:', PRODUCTION_CONFIG['model_name'])
print('Version:', PRODUCTION_CONFIG['version'])
"
```

**Expected Output:**
```
Class Weights: {0: 1.1, 1: 1.4, 2: 1.2}
Model Name: Option 3: Balanced
Version: v4.1
```

### Consistency Verification:

```bash
# Verify training config matches production config
python3 -c "
from scripts.train_production_model import PRODUCTION_CONFIG
from config import production_config
train_weights = PRODUCTION_CONFIG['class_weights']
prod_weights = production_config.MODEL_INFO['class_weights']
match = (
    train_weights[0] == prod_weights['away'] and
    train_weights[1] == prod_weights['draw'] and
    train_weights[2] == prod_weights['home']
)
print('✅ Configs match' if match else '❌ Configs DO NOT match')
"
```

**Expected Output:**
```
✅ Configs match
```

---

## Output Files

When training completes, the script generates:

1. **Model file**: `*.joblib` (CatBoost classifier)
2. **Metadata file**: `*_metadata.json` containing:
   - Timestamp
   - Model type
   - Feature count
   - Class weights used
   - Best hyperparameters found
   - Test set metrics (log loss, accuracy, draw accuracy, prediction distribution)
   - Full production config

### Example Metadata:

```json
{
  "timestamp": "2026-02-02T10:30:00",
  "model_type": "CatBoost",
  "features": 162,
  "class_weights": {"0": 1.1, "1": 1.4, "2": 1.2},
  "best_params": {
    "iterations": 450,
    "depth": 6,
    "learning_rate": 0.08,
    "l2_leaf_reg": 5.2
  },
  "test_metrics": {
    "log_loss": 0.987,
    "accuracy": 0.523,
    "draw_accuracy": 38.5,
    "prediction_distribution": {
      "away_pct": 25.2,
      "draw_pct": 28.1,
      "home_pct": 46.7
    }
  }
}
```

---

## Deployment Workflow

### Step 1: Generate Latest Training Data

```bash
# Download recent data (last 4 weeks)
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4

# Or use backfill for specific date range
python3 scripts/backfill_historical_data.py \
  --start-date 2025-12-01 \
  --end-date 2026-02-02 \
  --output-dir data/historical
```

### Step 2: Train New Model

```bash
# Train with Option 3 defaults
python3 scripts/train_production_model.py \
  --data data/training_data_latest.csv \
  --output models/production/option3_$(date +%Y%m%d).joblib
```

### Step 3: Evaluate Performance

```bash
# Check metadata file for test set performance
cat models/production/option3_*_metadata.json | jq '.test_metrics'
```

**Target Metrics (based on January 2026 backtest):**
- Log Loss: < 1.0
- Overall Accuracy: > 50%
- Draw Accuracy: 35-40%
- Prediction Distribution: ~25% Away, ~30% Draw, ~45% Home

### Step 4: Update Production Config (if deploying)

```bash
# Edit config/production_config.py
MODEL_PATH = "models/production/option3_20260202.joblib"
MODEL_INFO['last_updated'] = '2026-02-02'
```

### Step 5: Test with Real Data

```bash
# Run prediction on upcoming matches
python3 scripts/predict_production.py --days-ahead 7

# Or backtest on recent finished matches
python3 scripts/predict_production.py \
  --start-date 2026-01-20 \
  --end-date 2026-02-02 \
  --include-finished
```

---

## Monitoring & Quality Control

### Expected Model Behavior:

**Prediction Distribution:**
- Away: 20-30% of predictions
- Draw: 25-35% of predictions
- Home: 40-50% of predictions

**After Threshold Filtering (H=0.65, D=0.30, A=0.42):**
- Away: 20-25% of bets
- Draw: 55-60% of bets
- Home: 15-20% of bets

### Warning Signs:

❌ **Log loss > 1.05** → Model may be overfitting or underfitting
❌ **Draw accuracy < 30%** → Draw calibration issue
❌ **Prediction distribution heavily skewed** → Class weight issue
❌ **Test accuracy < 45%** → Model quality problem

### Rollback Conditions:

If new model shows degraded performance:
1. Check test set metrics in metadata
2. Compare against previous model's metadata
3. If log loss increased by >0.05 or accuracy dropped >3%, consider rollback
4. Keep previous production model as backup

---

## Consistency Checklist

✅ **Training pipeline uses Option 3 class weights (A=1.1, D=1.4, H=1.2)**
✅ **Default arguments match production configuration**
✅ **Model version tagged as v4.1**
✅ **Hyperparameter optimization uses same random seed (42)**
✅ **Data split matches production ratio (70/15/15)**
✅ **Metadata includes full production config for traceability**

---

## Additional Resources

- **Production Deployment**: See `PRODUCTION_DEPLOYMENT_SUMMARY.md`
- **Weekly Retraining Guide**: See `docs/WEEKLY_RETRAIN_GUIDE.md`
- **Model Comparison**: See `scripts/compare_models_from_db.py`
- **Threshold Analysis**: See `scripts/analyze_home_threshold_option3.py`

---

## Support

**Configuration Questions:**
- Check `config/production_config.py` for deployed settings
- Run validation script to verify consistency

**Training Issues:**
- Check metadata file for diagnostic info
- Compare metrics against expected ranges
- Verify input data quality

**Deployment Questions:**
- See `PRODUCTION_DEPLOYMENT_SUMMARY.md` for complete deployment guide
- Use rollback plan if performance degrades

---

## ✅ Training Pipeline Ready

Your training pipeline is now configured to reproduce the production model:
- ✅ Option 3 (Balanced) class weights
- ✅ Consistent hyperparameter settings
- ✅ Matching data split ratios
- ✅ Automated metadata tracking

Run weekly retraining to keep the model fresh with latest data.
