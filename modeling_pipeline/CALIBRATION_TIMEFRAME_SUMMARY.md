# 📊 Threshold Calibration - Timeframe Summary

**Date**: January 19, 2026
**Calibration Type**: Betting Strategy Threshold Optimization

---

## 🗓️ Calibration Data Timeframe

### Dataset Used
- **File**: `recent_predictions_20260119_164742.csv`
- **Total Matches**: 90
- **Date Range**: January 8-18, 2026 (11 days)
- **Source**: Recent predictions from trained ensemble model

### Daily Breakdown

| Date | Matches |
|------|---------|
| 2026-01-08 | 3 |
| 2026-01-09 | 2 |
| 2026-01-10 | 12 |
| 2026-01-11 | 8 |
| 2026-01-12 | 3 |
| 2026-01-13 | 3 |
| 2026-01-14 | 6 |
| 2026-01-15 | 3 |
| 2026-01-16 | 6 |
| 2026-01-17 | 34 |
| 2026-01-18 | 10 |
| **Total** | **90** |

### Leagues Included
Based on model training configuration (from `config.py`):
- Premier League (England)
- La Liga (Spain)
- Bundesliga (Germany)
- Serie A (Italy)
- Ligue 1 (France)
- Championship (England)

---

## 📈 What Was Calibrated

### Betting Strategy Thresholds

The optimization calibrated 3 parameters:

1. **Away Win Minimum Probability**
   - What it controls: Minimum probability to bet on away wins
   - Search range: 0.25 - 0.60
   - **Calibrated value: 0.48**

2. **Draw Close Threshold**
   - What it controls: Maximum difference between home/away probabilities to bet on draw
   - Search range: 0.05 - 0.25
   - **Calibrated value: 0.16**

3. **Home Win Minimum Probability**
   - What it controls: Minimum probability to bet on home wins
   - Search range: 0.45 - 0.70
   - **Calibrated value: 0.64**

### Optimization Method
- **Algorithm**: Random Search
- **Trials**: 300 threshold combinations tested
- **Objective**: Maximize ROI (Return on Investment)
- **Evaluation**: Each combination backtested on all 90 matches
- **Result**: Best combination selected based on highest ROI

---

## ⚠️ Important Limitations

### 1. Small Sample Size

**Issue**: Only 90 matches used for calibration

**Implications**:
- High variance in results
- Thresholds may not generalize well
- Statistical significance is limited

**Recommendation**: For robust calibration, use 200-500+ matches (30-60 days)

### 2. Short Time Period

**Issue**: Only 11 days of data

**Implications**:
- May not capture seasonal variations
- Doesn't cover different match contexts (cup games, derbies, etc.)
- Limited variety in team match-ups

**Recommendation**: Use full season or multiple months for stable thresholds

### 3. Model Bias Issues

**Critical Finding**: The model used for these predictions has serious issues:

| Issue | Impact |
|-------|--------|
| 0 draw predictions | Model never predicts draws as most likely outcome |
| Home bias | Over-predicts home wins, under-predicts away wins |
| Poor calibration | Probabilities don't match actual outcome distribution |

**Result**: The calibrated thresholds are **defensive** - they work around model weaknesses rather than leveraging model strengths.

### 4. No Cross-Validation

**Issue**: Thresholds optimized on same data they'll be evaluated on

**Implications**:
- Risk of overfitting to this specific 11-day period
- Performance may degrade on new data
- No held-out validation set

**Recommendation**: Use proper train/validation/test split for threshold optimization

---

## 📊 Model Training vs Calibration Timeframes

### Model Training Data

The **models themselves** (Elo, Dixon-Coles, XGBoost) were trained on:

**File**: `data/processed/sportmonks_features.csv` (60MB, ~2,500 matches)

**From `config.py`**:
```python
TRAIN_SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25"
]

VALIDATION_SEASONS = [
    "2025-26"  # Current season
]
```

**Training Data Timeframe**: August 2022 - May 2025 (~3 years)

### Betting Threshold Calibration

**Timeframe**: January 8-18, 2026 (11 days)

**Why Different?**
- Threshold calibration done on **recent predictions** to ensure relevance
- Used latest model predictions with current performance characteristics
- Quick iteration vs full model retraining

---

## 🎯 Confidence Levels

### Statistical Confidence

With 90 matches:
- **95% Confidence Interval**: Very wide (±15-20% on ROI)
- **Required for ±5% CI**: ~385 matches
- **Current Sample**: 23% of required size

### Reliability Assessment

| Metric | Reliability | Notes |
|--------|-------------|-------|
| **ROI** | ⚠️ Low | Small sample, high variance |
| **Win Rate** | ⚠️ Medium | 90 matches provides rough estimate |
| **Bet Count** | ✅ Good | Clearly shows betting frequency |
| **Threshold Direction** | ✅ Good | Higher = more selective (robust finding) |

---

## 💡 Key Findings from Calibration

### Performance by Outcome (on calibration data)

| Bet Type | Bets | Win Rate | ROI | Status |
|----------|------|----------|-----|--------|
| **Home Win** | 0 | N/A | N/A | Threshold too high (0.64) - effectively disabled |
| **Draw** | ~25 | 36% | +34.4% | **Only profitable category** |
| **Away Win** | ~8 | 12.5% | -80.9% | Still losing despite high threshold |

### Why These Numbers?

**High Away Threshold (0.48)**:
- Model predicts away wins 70% of the time (massive away bias)
- But actual away wins are only 42%
- High threshold (0.48) filters out most false away predictions
- Even with filtering, away bets still lose money

**Medium Draw Threshold (0.16)**:
- Model never predicts draws (0%)
- But draw bets using probabilities are profitable (+34.4% ROI)
- Threshold identifies "close matches" where draw is likely
- Works because high draw odds (3.0-4.0x) create value

**Very High Home Threshold (0.64)**:
- Model has 0% success rate on home win bets
- Threshold set so high it effectively disables home betting
- Defensive move to avoid guaranteed losses

---

## 🔄 Recommendation: Re-Calibrate After Model Fix

### Current Situation
- Thresholds work around model weaknesses
- Still negative ROI (-5.05%) even with optimization
- Limited by poor model calibration

### After Model Retraining
Once the model is fixed (draw predictions, home bias), re-calibrate with:

1. **Larger Dataset**: 200-500 matches (30-60 days)
2. **Proper Splits**: Train/validation/test for thresholds
3. **Multiple Timeframes**: Test stability across different periods
4. **League-Specific**: Optimize separately for each league
5. **Cross-Validation**: K-fold CV to reduce overfitting

### Expected Improvement
With properly calibrated model:
- ROI could improve from -5% to +10-20%
- All bet types (home/draw/away) could be profitable
- More bets placed (currently only 33 out of 90 matches)
- Better generalization to new data

---

## 📁 Related Files

### Calibration Files
- **Data**: `recent_predictions_20260119_164742.csv` (90 matches)
- **Script**: `optimize_betting_thresholds.py` (optimization code)
- **Results**: `threshold_optimization_20260119_180247.txt` (full report)
- **Output**: `11_smart_betting_strategy.py` (updated thresholds)

### Model Training Files
- **Data**: `data/processed/sportmonks_features.csv` (~2,500 matches, 3 years)
- **Config**: `config.py` (seasons defined)
- **Models**: `models/` directory (trained models)

---

## 📝 Summary

| Aspect | Details |
|--------|---------|
| **Calibration Period** | January 8-18, 2026 (11 days) |
| **Sample Size** | 90 matches |
| **Leagues** | 6 major European leagues |
| **Method** | Random search (300 trials) |
| **Objective** | Maximize betting ROI |
| **Result** | Improved ROI by +35% but still negative |
| **Limitation** | Small sample, model bias issues |
| **Status** | ⚠️ Valid for this dataset, needs larger sample |

---

## ✅ Action Items

### For Current Use
- [x] Thresholds updated in betting strategy
- [x] Documentation completed
- [ ] Test on new data (next 30 days)
- [ ] Monitor performance and adjust if needed

### For Future Improvement
- [ ] Fix model calibration (draw predictions)
- [ ] Collect 200+ matches for re-calibration
- [ ] Use proper train/val/test splits
- [ ] Implement cross-validation
- [ ] Consider league-specific thresholds

---

**Last Updated**: January 19, 2026
**Calibration Valid**: January 8-18, 2026 data
**Next Re-calibration**: After model improvements or 200+ new matches collected
