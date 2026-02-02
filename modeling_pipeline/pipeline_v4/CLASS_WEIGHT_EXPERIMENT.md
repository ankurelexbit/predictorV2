# Class Weight Experiment Guide

## Overview

This experiment trains 3 models in parallel with different class weight configurations to optimize home predictions.

## Quick Start

### Step 1: Train All 3 Models in Parallel

```bash
./scripts/train_multiple_weights.sh
```

This will train:
- **Option 1 (Conservative)**: H=1.0, D=1.3, A=1.0 - Reduce draw/away weights
- **Option 2 (Aggressive)**: H=1.3, D=1.5, A=1.2 - Increase home weight
- **Option 3 (Balanced)**: H=1.2, D=1.4, A=1.1 - Adjust all three

**Time**: ~30-45 minutes (running in parallel with 50 trials each)

### Step 2: Compare Distributions

```bash
python3 scripts/compare_model_distributions.py
```

This will show:
- Prediction probability distributions (Home/Draw/Away %)
- Calibration analysis (Predicted vs Actual)
- Per-class accuracy
- Side-by-side comparison table
- Recommendations

### Step 3: Select Best Model

Based on the comparison, choose the model that:
1. Has best home calibration (closest to actual ~40%)
2. Maintains draw profitability (draw prob ~30%)
3. Has lowest log loss
4. Increases home predictions without hurting overall accuracy

---

## Manual Training (Single Model)

If you want to train a single model with custom weights:

```bash
python3 scripts/train_production_model.py \
  --data data/training_data_with_draw_features.csv \
  --output models/custom_weights.joblib \
  --weight-home 1.2 \
  --weight-draw 1.4 \
  --weight-away 1.1 \
  --n-trials 100
```

---

## Expected Results

### Current Model (H=1.0, D=1.5, A=1.2)
```
Average Probabilities:
  Home: 39.6%  (Actual: 40.2%, Market: 43.5%)
  Draw: 30.2%  (Actual: 32.2%, Market: 26.2%)
  Away: 30.2%  (Actual: 27.6%, Market: 30.3%)

Issue: Model underestimates home by 3.9% vs market
```

### Option 1: Conservative (H=1.0, D=1.3, A=1.0)
```
Expected:
  Home: 41-42% (closer to actual)
  Draw: 29-30% (slight decrease, still profitable)
  Away: 29-30% (decrease to match actual)

Impact: +3-4% home predictions, safer approach
```

### Option 2: Aggressive (H=1.3, D=1.5, A=1.2)
```
Expected:
  Home: 44-45% (above actual, close to market)
  Draw: 29-30% (maintained)
  Away: 26-27% (maintained)

Impact: +5-6% home predictions, risk of over-predicting
```

### Option 3: Balanced (H=1.2, D=1.4, A=1.1) ⭐ RECOMMENDED
```
Expected:
  Home: 43-44% (matches market)
  Draw: 30-31% (slight decrease)
  Away: 27-28% (matches actual)

Impact: +4-5% home predictions, best overall balance
```

---

## What to Look For

### 1. Home Probability Increase
Target: Move from 39.6% → 43-44% (market level)

### 2. Calibration Improvement
Current: Home -3.9% vs market
Target: Home within ±2% of market

### 3. Draw Performance Preservation
Current: Draw 73% ROI (very profitable)
Target: Keep draw prob ~30% (don't hurt this!)

### 4. Log Loss
Current: ~0.95-1.0
Target: Keep below 1.0 (lower is better)

### 5. Overall Accuracy
Current: ~50-52%
Target: Maintain or improve

---

## Backtest the Winner

Once you select the best model:

```bash
# Test on January 2026 (out-of-sample)
python3 scripts/predict_production.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --model-path models/weight_experiments/option3_balanced.joblib

# Update results
python3 scripts/update_results.py

# Analyze thresholds
python3 scripts/analyze_thresholds.py
```

Compare:
- Home bets: Should increase from 8 → 12-15
- Home ROI: Should increase from 9% → 20-30%
- Overall ROI: Should increase from 35.6% → 38-42%

---

## Next Steps After Selecting Winner

1. **Deploy to production**: Copy winner to production path
   ```bash
   cp models/weight_experiments/option3_balanced.joblib \
      models/with_draw_features/conservative_with_draw_features.joblib
   ```

2. **Run February predictions**: Use new model for upcoming games

3. **Monitor performance**: Track actual vs expected for 2-4 weeks

4. **Iterate**: If needed, fine-tune weights further based on results

---

## Troubleshooting

### Training Fails
- Check data path exists: `data/training_data_with_draw_features.csv`
- Check you have enough RAM (CatBoost can be memory-intensive)
- Reduce `--n-trials` from 50 to 20 for faster testing

### Comparison Script Fails
- Ensure at least one model finished training
- Check model files exist in `models/weight_experiments/`

### Models Look Similar
- Try more extreme weights (e.g., H=1.5 or H=0.8)
- Check class balance in training data
- Verify weights are being applied (check metadata JSON)

---

## Files Created

```
scripts/
├── train_production_model.py     (modified - now accepts weight params)
├── train_multiple_weights.sh     (new - parallel training)
└── compare_model_distributions.py (new - comparison analysis)

models/weight_experiments/
├── option1_conservative.joblib
├── option1_conservative_metadata.json
├── option1_training.log
├── option2_aggressive.joblib
├── option2_aggressive_metadata.json
├── option2_training.log
├── option3_balanced.joblib
├── option3_balanced_metadata.json
└── option3_training.log
```

---

## Summary

This experiment lets you:
1. ✅ Train 3 models in parallel (~30-45 mins)
2. ✅ Compare prediction distributions side-by-side
3. ✅ Select the best weight configuration empirically
4. ✅ Deploy the winner to production

**Recommended**: Start with Option 3 (Balanced) as it adjusts all three weights proportionally.
