# Weekly Retraining + Recalibration Workflow

## Overview

Every week, new matches are played and results are finalized. To keep your model accurate, you need to:
1. **Retrain the model** on expanded dataset (includes new results)
2. **Recalibrate** to fix model overconfidence/underconfidence
3. Model + calibrators must ALWAYS be updated together

## Why Recalibrate Every Week?

**Short answer:** When you retrain the model, it produces slightly different raw probabilities. The old calibrators won't work correctly with the new model.

**Simple analogy:**
- Model is like a thermometer that reads temperatures
- Calibrator is the conversion chart that fixes the thermometer's errors
- If you replace the thermometer (retrain model), you need a new conversion chart (recalibrate)

## Automatic Weekly Pipeline

### Option 1: Run Manually (Recommended for Testing)

```bash
cd /path/to/pipeline_v4
source venv/bin/activate
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4
```

This will:
1. Download last 4 weeks of fixture data
2. Convert JSON to CSV
3. Generate training dataset
4. Train model (auto-increments version: v2.0.0 → v2.1.0)
5. **Recalibrate on last 3 months of predictions** ← NEW STEP
6. Cleanup old files

### Option 2: Automate with Cron (Production)

Run every Sunday at 2am:

```bash
crontab -e
```

Add this line:
```
0 2 * * 0 cd /Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v4 && /path/to/venv/bin/python3 scripts/weekly_retrain_pipeline.py --weeks-back 4 >> logs/weekly_retrain.log 2>&1
```

### Option 3: Manual Recalibration Only

If you already retrained the model and just need to recalibrate:

```bash
python3 scripts/recalibrate_model.py --months-back 3
```

## What Happens During Recalibration?

### Step 1: Fetch Recent Predictions
- Gets last 3 months of predictions from your database
- Only uses predictions with resolved results (actual_result IS NOT NULL)
- Needs minimum 100 predictions (usually have 300-500)

### Step 2: Fit Isotonic Regression
For each outcome (Home/Draw/Away):
- Compares: Model said X% → Actually happened Y%
- Learns the correction mapping
- Example: Model says 60% → Calibrator corrects to 48%

### Step 3: Save Calibrators
- Saves to `models/calibrators.joblib`
- Your prediction script automatically loads these
- Thresholds in config apply to calibrated probs, not raw

## Example Output

```
===============================================================================
MODEL RECALIBRATION
===============================================================================

Calibration period: Last 3 months
Minimum samples required: 100

[1/3] Fetching predictions from 2025-11-04 to 2026-02-04...
   Loaded 539 predictions with resolved results

[2/3] Fitting isotonic regression calibrators...
   Home  — Actual: 0.460 | Raw: 0.447 | Calibrated: 0.460 | MAE: 0.441→0.418
   Draw  — Actual: 0.244 | Raw: 0.253 | Calibrated: 0.244 | MAE: 0.371→0.362
   Away  — Actual: 0.296 | Raw: 0.300 | Calibrated: 0.296 | MAE: 0.374→0.354

[3/3] Saving calibrators to models/calibrators.joblib...
   ✅ Calibrators saved successfully

===============================================================================
CALIBRATION COMPLETE
===============================================================================

✅ Calibrators trained on 539 predictions
✅ Saved to: models/calibrators.joblib
```

## When to Recalibrate (Checklist)

✅ **YES - Recalibrate when:**
- Weekly retrain (model version changes)
- Model performance degrades noticeably
- Routine maintenance every 3-6 months

❌ **NO - Don't recalibrate when:**
- Just changing thresholds (calibrators stay same)
- After a few bad days (variance is normal)
- Model hasn't changed

## Impact of Calibration

Based on your Nov 2025 - Jan 2026 data:

| Strategy | Without Calibration | With Calibration | Improvement |
|----------|---------------------|------------------|-------------|
| H=0.40, D=0.28, A=0.40 | +$5.20 PnL | +$41.90 PnL | **+$36.70 (+706%)** |
| H=0.36, D=0.28, A=0.40 | +$23.98 PnL | +$33.93 PnL | **+$9.95 (+41%)** |

**Without calibration, you lose 40-88% of potential profit!**

## Files Updated by Pipeline

```
models/
├── production/
│   ├── model_v2.0.0.joblib      # Old model
│   ├── model_v2.1.0.joblib      # New model (after retrain)
│   └── LATEST                   # Points to v2.1.0
├── calibrators.joblib           # NEW - Updated after recalibration

data/
├── training_data_20260204_140530.csv  # New training data
└── training_data_latest.csv           # Symlink to above

logs/
└── weekly_retrain.log           # Pipeline execution log
```

## Troubleshooting

### "Not enough data for calibration"

```
❌ ERROR: Not enough data for calibration
   Found: 45 predictions
   Required: 100 predictions
```

**Solution:** Reduce calibration period or wait for more results:
```bash
python3 scripts/recalibrate_model.py --months-back 6  # Use 6 months instead of 3
```

### "Calibration failed in pipeline"

Pipeline will continue but warn you:
```
⚠️  Calibration failed - using old calibrators (may be inaccurate)
```

**Solution:** Run recalibration manually:
```bash
python3 scripts/recalibrate_model.py --months-back 3
```

Check if you have enough resolved predictions in database:
```sql
SELECT COUNT(*) FROM predictions
WHERE match_date >= NOW() - INTERVAL '3 months'
  AND actual_result IS NOT NULL;
```

### "Database connection error"

Make sure DATABASE_URL is set:
```bash
export DATABASE_URL="postgresql://..."
# Or put in .env file
```

## Summary

1. **Run weekly pipeline:** Downloads data + trains model + recalibrates
2. **Model + Calibrators updated together:** They're a matched pair
3. **Calibration adds +$7-37 profit:** Don't skip this step!
4. **Use last 3 months of data:** Good balance of recency and sample size
5. **Automatic versioning:** Old models kept for rollback

## Questions?

**Q: What if I only change thresholds, not model?**
A: No need to recalibrate. Calibrators work with any thresholds.

**Q: Can I use old calibrators with new model?**
A: No! Model and calibrators must match. Always recalibrate after retraining.

**Q: How long does recalibration take?**
A: ~5-10 seconds for 500 predictions. Very fast.

**Q: What if I want to test a new model without affecting production?**
A: Save to different file:
```bash
# Train to test path
python3 scripts/train_production_model.py --output models/test_model.joblib

# Recalibrate to test path
python3 scripts/recalibrate_model.py --output models/test_calibrators.joblib

# Test predictions won't affect production
```
