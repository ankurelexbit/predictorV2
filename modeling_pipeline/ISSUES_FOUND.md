# Comprehensive Model Issues Report

## Executive Summary

The football prediction model has **39.5% accuracy** (worse than naive "always predict home" at 43.9%). We've identified **6 critical issues** causing this poor performance.

---

## 🚨 CRITICAL ISSUES FOUND

### Issue #1: Elo Model - Home Advantage Too High
**Severity:** CRITICAL  
**Impact:** Makes Elo model predict home win even when away team is stronger

**Details:**
- Home advantage set to **100 Elo points** (default in EloProbabilityModel)
- This is **50-100% higher than realistic** (should be 50-70 points)
- 100 Elo points = 64% expected score boost

**Example:**
- Feyenoord (1456 Elo) vs Sparta Rotterdam (1519 Elo)
- Sparta is 63 points stronger
- After adding 100 HA: -63 + 100 = **+37 net for weaker home team**
- Result: Elo predicts 69.2% home win (Actual: Away win 4-3)

**Fix:**
```python
# In 04_model_baseline_elo.py
class EloProbabilityModel:
    def __init__(self, home_advantage=50):  # Change from 100 to 50-60
```

---

### Issue #2: Dixon-Coles - Team Name Mismatch
**Severity:** CRITICAL  
**Impact:** Dixon-Coles can't find teams, uses default 0.0 parameters, makes all predictions wrong

**Details:**
- API returns: "Paris", "SC Heerenveen", "FC Groningen"
- Model has: "Paris Saint Germain", [missing Dutch teams]
- When team not found → uses attack=0.0, defense=0.0
- This makes predictions completely random

**Evidence:**
- Dixon-Coles predicted 0/3 home wins (100% away win bias)
- Model has 428 teams, but many API team names don't match

**Examples of mismatches:**
- API: "Paris" → Model: "Paris Saint Germain"
- API: "SC Heerenveen" → Model: [NOT FOUND]
- API: "FC Groningen" → Model: [NOT FOUND]

**Fix:**
- Use team IDs instead of team names for predictions
- OR create name mapping layer (Paris → Paris Saint Germain)
- Retrain Dixon-Coles with exact API team names

---

### Issue #3: XGBoost - Learned Home Bias from Training Data
**Severity:** HIGH  
**Impact:** XGBoost always predicts home win

**Details:**
- Training data: 43.9% home wins, 31.6% away wins (1.39x imbalance)
- XGBoost learns this pattern and amplifies it
- Top features are position/points-based, which may correlate with home bias

**Evidence:**
- XGBoost predicted 3/3 home wins (100%)
- Even when Parma (weaker) played Genoa with similar strength

**Fix:**
- Use class weights to balance training:
```python
scale_pos_weight = {
    0: 1.39,  # away wins (underrepresented)
    1: 1.79,  # draws (underrepresented) 
    2: 1.0    # home wins (baseline)
}
```
- Reduce home advantage in training features
- Add more away-win-focused features

---

### Issue #4: Feature Scaling - Vastly Different Ranges
**Severity:** MEDIUM  
**Impact:** XGBoost may over-rely on large-scale features

**Details:**
- Elo ratings: Range 764 points (1236-2001)
- Form: Range 15 points (0-15)
- Goals: Range 6 (0-6)
- Possession: Range 78.8 (0-78.8)

**Scale ratio:** 127.4x difference between largest and smallest

**Current status:**
- XGBoost uses StandardScaler BUT may still be affected
- Elo (large scale) might dominate over form (small scale)

**Fix:**
- Apply MinMaxScaler to normalize all features to [0, 1]
- Or use RobustScaler for outlier resistance

---

### Issue #5: Stacking Ensemble - Inherits All Base Model Issues
**Severity:** HIGH  
**Impact:** Ensemble can't fix broken base models

**Details:**
- Elo: 100% home win bias (home advantage too high)
- Dixon-Coles: 100% away win bias (team name mismatch)
- XGBoost: 100% home win bias (learned from training data)

**Meta-model learns from broken inputs:**
- 2 models say home (Elo + XGBoost)
- 1 model says away (Dixon-Coles, but with wrong teams)
- Result: Meta-model learns "trust Elo + XGBoost" → 100% home bias

**Fix:**
- Fix base models FIRST
- Then retrain stacking meta-model

---

### Issue #6: Training Data Class Imbalance
**Severity:** MEDIUM  
**Impact:** Model learns to predict home wins more often

**Details:**
- Home wins: 43.9%
- Away wins: 31.6%
- Draws: 24.5%

**Imbalance ratio:** 1.39x (home vs away)

**Fix:**
- Use SMOTE or class weights during training
- Or use stratified sampling to balance classes

---

## 📊 COMPARISON: ACTUAL VS EXPECTED

### What SHOULD happen (Sparta vs Feyenoord):
- Sparta: 1519 Elo, 12 form points (excellent)
- Feyenoord: 1456 Elo, 2 form points (terrible)
- **Expected:** Sparta favored (60%+ away win)
- **Actual result:** Sparta won 4-3 ✓

### What DOES happen (our model):
- Elo model: +100 HA overrides -63 Elo diff → 69% home win ✗
- Dixon-Coles: Teams not found → random prediction → away win (by luck)
- XGBoost: Learned home bias → 50% home win ✗
- **Ensemble:** Combines broken models → 69% home win ✗

---

## 🎯 ROOT CAUSES SUMMARY

1. **Elo home advantage** (100) is 50-100% too high
2. **Dixon-Coles** can't match team names from API
3. **XGBoost** learned home bias from imbalanced training data
4. **Features** not properly scaled
5. **Stacking ensemble** can't fix broken base models
6. **Training data** has class imbalance favoring home wins

---

## ✅ RECOMMENDED FIXES (Priority Order)

### Priority 1 (Critical - Do First):
1. **Fix Elo home advantage:** Reduce from 100 to 50-60 points
2. **Fix Dixon-Coles team names:** Use team IDs or create name mapping
3. **Retrain all models** with fixes above

### Priority 2 (High - Do Soon):
4. **Add class weights to XGBoost** to balance training
5. **Improve feature scaling** (MinMaxScaler or RobustScaler)
6. **Retrain stacking ensemble** after base models are fixed

### Priority 3 (Medium - Nice to Have):
7. **Better decision logic:** Don't always pick max probability
   - Only predict when confidence > 45% AND margin > 15%
   - Flag "too close to call" matches
8. **Add confidence scores** to predictions
9. **Collect more away win examples** for training

---

## 📈 EXPECTED IMPROVEMENT

After fixing these issues:
- **Current:** 39.5% accuracy
- **After fixes:** Estimated 48-52% accuracy
- **Target:** 50-55% (betting market level)

The single biggest gain will come from fixing Elo home advantage and Dixon-Coles team names.

---

*Report generated: 2026-01-19*
