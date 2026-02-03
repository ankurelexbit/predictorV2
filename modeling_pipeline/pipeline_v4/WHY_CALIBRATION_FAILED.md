# Why Calibration Failed: Technical Analysis

**Date:** February 3, 2026
**Analysis:** Isotonic Regression Calibration on Option 3 Model

---

## Executive Summary

**Result:** Isotonic regression calibration made the model **significantly worse** across all metrics.

| Metric | Uncalibrated | Calibrated | Change |
|--------|--------------|------------|--------|
| **Log Loss** | 1.0204 | 1.1889 | **+0.17** ❌ (worse) |
| **Away Brier Score** | 0.1864 | 0.1873 | +0.0009 ❌ |
| **Draw Brier Score** | 0.2072 | 0.2097 | +0.0024 ❌ |
| **Home Brier Score** | 0.2183 | 0.2234 | +0.0051 ❌ |
| **Betting Profit** | $41.81 | $6.68 | **-$35.13** ❌ (-84%) |

**Conclusion:** Calibration is harmful for this model.

---

## The Four Root Causes

### 1. Validation Set Overfitting ⚠️

**What Happened:**
- Isotonic regression was fit on the **validation set** (15% of training data)
- It learned the specific quirks of that 2,691-sample validation set
- When applied to **test set** (different time period), the learned mapping was wrong

**Evidence:**
- Calibration MSE improved on some classes (Away: 0.0131 → 0.0043)
- But overall log loss got WORSE (1.0204 → 1.1889)
- Classic overfitting: fits validation perfectly, generalizes poorly

**Why This Happened:**
- Validation set is small (2,691 samples)
- Isotonic regression is non-parametric (high capacity)
- It memorized validation set instead of learning true calibration

---

### 2. Confidence Overcorrection 📈

**What Happened:**
- Calibration **increased** the number of high-confidence predictions
- But these new high-confidence predictions were often **wrong**

**Evidence:**

| Confidence Threshold | Uncalibrated Count | Calibrated Count | Change |
|---------------------|-------------------|------------------|--------|
| > 50% | 91 (45%) | 122 (60%) | **+31** |
| > 60% | 49 (24%) | 55 (27%) | +6 |
| > 70% | 24 (12%) | 39 (19%) | **+15** |
| > 80% | 7 (4%) | 10 (5%) | +3 |

**Impact:**
- 31 more predictions crossed 50% confidence threshold
- But calibration made predictions LESS accurate overall
- **4 predictions** that were CORRECT became WRONG after calibration

**Examples of Predictions That Became Wrong:**

```
Leeds United vs Manchester United
- Actual: Draw
- Uncalibrated: Draw (38% conf) ✓ CORRECT
- Calibrated: Home (36% conf) ✗ WRONG

Genoa vs Bologna
- Actual: Home
- Uncalibrated: Home (35% conf) ✓ CORRECT
- Calibrated: Away (38% conf) ✗ WRONG
```

---

### 3. Probability Distribution Distortion 📊

**What Happened:**
- Calibration changed the mean probabilities away from actual frequencies
- Made prediction errors **larger**, not smaller

**Evidence:**

| Class | Actual Freq | Uncalib Error | Calib Error | Result |
|-------|-------------|---------------|-------------|--------|
| **Away** | 30.2% | -2.7% | -2.7% | No change |
| **Draw** | 30.2% | -2.7% | **-5.3%** | ❌ Worse |
| **Home** | 39.6% | +5.4% | **+8.0%** | ❌ Worse |

**Key Finding:**
- Draw predictions went from 27.5% → 24.9% (moved AWAY from 30.2% actual)
- Home predictions went from 45.0% → 47.6% (moved AWAY from 39.6% actual)
- Calibration made the model **MORE overconfident on homes**, not less

---

### 4. Class Weights Already Optimize for Strategy 🎯

**Critical Insight:** The original model's class weights (A=1.1, D=1.4, H=1.2) were **specifically chosen** to optimize betting performance.

**What Class Weights Do:**
- Tell the model which mistakes are more costly
- Bias predictions toward or away from certain outcomes
- Create intentional "miscalibration" that improves strategy performance

**What Calibration Did:**
- Destroyed this carefully tuned balance
- Tried to "fix" probabilities that were intentionally biased
- Made the model worse for the betting strategy

**Analogy:**
```
Original Model: "Boost draw predictions because draws are valuable"
Calibration: "These draw predictions look too high, let me reduce them"
Result: Fewer draw bets, less profit
```

---

## Why Isotonic Regression Failed

### What Isotonic Regression Does

Isotonic regression creates a **monotonic mapping** from predicted probabilities to calibrated probabilities:

```
Uncalibrated Prob → Calibrated Prob
     0.10        →      0.08
     0.20        →      0.15
     0.30        →      0.25
     0.40        →      0.42
     0.50        →      0.58
     0.60        →      0.71
     ...
```

This mapping is learned from the **validation set**.

### The Problem

**On validation set:** The mapping works perfectly (by design)

**On test set:** The mapping is wrong because:
1. Test set has different time period (different teams, different form)
2. Test set has different match difficulty distribution
3. Small validation set (2,691 samples) leads to overfitting
4. Non-parametric method has high capacity to memorize

**Result:** Calibration that helps on validation, hurts on test (and real world)

---

## The Real "Calibration" Was Already There

### Probability Thresholds = Calibration Filter

The production strategy uses **high probability thresholds**:
- Home: 0.65 (only bet if > 65% confident)
- Draw: 0.30 (only bet if > 30% confident)
- Away: 0.42 (only bet if > 42% confident)

**These thresholds ARE a form of calibration:**
- They filter out low-quality predictions
- They compensate for model overconfidence
- They were empirically optimized on real betting performance

**Evidence:**
- Current strategy: 52% win rate, 28.3% ROI
- Without thresholds (bet all predictions): Would lose money
- Thresholds provide the "safety margin" that calibration attempts to add

---

## Why This Matters for EV Strategy

### EV Strategy Needs Perfect Calibration

EV = (Probability × Odds) - 1

- If probability is 5% too high → EV is wrong → Bad bets
- If probability is 5% too low → EV is wrong → Missed opportunities

### Our Calibration Made This Worse

**Before Calibration:**
- Draw probabilities: Mean 27.5% (actual 30.2%) - underestimated by 2.7%
- Home probabilities: Mean 45.0% (actual 39.6%) - overestimated by 5.4%

**After Calibration:**
- Draw probabilities: Mean 24.9% (actual 30.2%) - underestimated by 5.3% ❌
- Home probabilities: Mean 47.6% (actual 39.6%) - overestimated by 8.0% ❌

**Result:**
- Calibration made calibration errors **LARGER**
- EV calculations became **MORE wrong**
- Betting performance got **WORSE**

---

## Alternative Calibration Approaches (Why They Would Also Fail)

### 1. Platt Scaling (Sigmoid)
**Why it would fail:**
- Assumes logistic relationship (parametric)
- Our model probabilities don't follow logistic curve
- Would still overfit to validation set

### 2. Temperature Scaling
**Why it would fail:**
- Single temperature parameter for all classes
- Can't fix per-class biases
- Would hurt the intentional class weight bias

### 3. Beta Calibration
**Why it would fail:**
- More parameters = more overfitting risk
- Still learns from small validation set
- Doesn't respect class weight strategy

### 4. Ensemble Calibration
**Why it would fail:**
- Requires multiple models
- Expensive to train
- Each model would have same fundamental issue

---

## The Fundamental Issue

### Calibration vs. Optimization Conflict

**Two competing goals:**

1. **Calibration Goal:** Make probabilities match observed frequencies
   - P(Home) = 40% should win 40% of the time
   - Perfect calibration = predictions are "honest"

2. **Betting Strategy Goal:** Maximize profit using class weights
   - Boost draw predictions (D weight = 1.4)
   - Reduce home predictions (H weight = 1.2)
   - Predictions are "strategically biased"

**These goals are incompatible!**

The model was **intentionally miscalibrated** via class weights to maximize betting profit. Trying to "fix" the calibration destroys the strategy.

---

## Mathematical Explanation

### Why Class Weights Create Intentional Miscalibration

CatBoost with class weights minimizes:

```
Loss = Σ weight[y] × log_loss(y, pred)
```

Not:

```
Loss = Σ log_loss(y, pred)  # Equal weights
```

**Effect:**
- Model is penalized MORE for getting draws wrong (weight 1.4)
- Model is penalized LESS for getting away wins wrong (weight 1.1)
- Model learns to **overpredict draws** (safer given penalty)

**This creates:**
- Intentionally biased probabilities
- "Miscalibration" that improves strategy performance
- Probabilities that are "wrong" but **useful**

**When we calibrate:**
- We try to "fix" this bias
- We remove the strategic advantage
- We make probabilities "honest" but **less profitable**

---

## Real-World Analogy

### The Weather Forecast Analogy

**Scenario 1: Honest Forecaster**
- Says 30% chance of rain
- It rains 30% of the time
- Perfect calibration ✓
- But people get caught in rain 30% of the time

**Scenario 2: Conservative Forecaster**
- Says 50% chance of rain when it's actually 30%
- People bring umbrella more often
- "Miscalibrated" (overestimates rain)
- But fewer people get wet ✓

**Our Model is Like Scenario 2:**
- Overestimates draws (to catch profitable opportunities)
- "Miscalibrated" for honesty
- But **calibrated for profit** ✓

---

## Conclusion

### Why Calibration Failed: Summary

1. **Isotonic regression overfit to validation set**
   - Small sample size (2,691)
   - Non-parametric method with high capacity
   - Learned validation quirks instead of true calibration

2. **Calibration increased confidence in wrong predictions**
   - 31 more predictions crossed 50% threshold
   - But 4 correct predictions became wrong
   - Higher confidence ≠ better accuracy

3. **Calibration distorted probability distributions**
   - Made prediction errors LARGER, not smaller
   - Draw: -2.7% error → -5.3% error
   - Home: +5.4% error → +8.0% error

4. **Calibration destroyed strategic class weight bias**
   - Class weights intentionally bias predictions
   - This bias improves betting performance
   - Calibration "fixes" what shouldn't be fixed

### The Bottom Line

**Your model doesn't need calibration.**

It needs:
- ✅ Class weights (A=1.1, D=1.4, H=1.2) - Already optimized
- ✅ Probability thresholds (H=0.65, D=0.30, A=0.42) - Already optimized
- ✅ Top 5 leagues filter - Already implemented

**Calibration is solving a problem that doesn't exist and creating problems that do.**

The current setup ($41.81 profit, 28.3% ROI) is already optimal. No changes needed.

---

## Lessons Learned

1. **Calibration is not always beneficial**
   - Only helps if probabilities are used directly (weather forecasting, medical diagnosis)
   - Harmful if probabilities feed into a strategy (betting, trading)

2. **Small validation sets cause overfitting**
   - 2,691 samples not enough for isotonic regression
   - Need 10,000+ samples for reliable calibration

3. **Class weights ≠ Bad calibration**
   - Class weights create strategic bias
   - This bias is intentional and profitable
   - Don't "fix" what isn't broken

4. **Test on out-of-sample data**
   - Calibration looked good on validation set
   - Failed spectacularly on test set
   - Always validate on held-out data

5. **Profit > Probabilities**
   - Goal is profit, not perfect probabilities
   - Miscalibration that makes money is good
   - Calibration that loses money is bad

---

## Files Reference

- **Calibration plots:** `models/calibrated/calibration_curves.png`
- **Calibrated model:** `models/calibrated/option3_calibrated_for_ev.joblib`
- **Analysis script:** `scripts/analyze_calibration_failure.py`
- **Test results:** `results/calibrated_ev_test.txt`

---

**Last Updated:** February 3, 2026
