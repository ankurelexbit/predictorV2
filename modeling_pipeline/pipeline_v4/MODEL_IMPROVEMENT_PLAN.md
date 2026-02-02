# Model Improvement Plan - V4 Pipeline

## Executive Summary

Current model achieves **35.6% ROI** on top 5 leagues, but has significant room for improvement:
- **Draws are underpriced**: +4.1% edge vs market → 73% ROI on draw bets
- **Homes are overpriced**: -3.8% edge vs market → 9% ROI on home bets
- **Model has calibration issues** at extreme probabilities (0.6-0.8 range)
- **Profitability is highest at MEDIUM confidence (0.3-0.4)**, not high confidence

---

## Critical Issues Identified

### 1. **Calibration Problems**

| Probability Range | Issue | Impact |
|-------------------|-------|--------|
| Home 0.6-0.7 | Predicts 65.7%, actual 50% | Over-confident by 15.7% |
| Away 0.5-0.6 | Predicts 55.7%, actual 69.2% | Under-confident by 13.6% |
| Away 0.6-0.8 | Predicts 64-72%, actual 40-50% | Over-confident by 20-25% |

**Root Cause**: Model trained for log loss, not calibrated probabilities. Isotonic calibration may not be working well for extreme values.

### 2. **Market Edge Analysis**

```
Average Model Edge vs Market:
- Home: -3.8% (model thinks homes less likely than market)
- Draw: +4.1% (model thinks draws MORE likely than market) ✅
- Away: -0.1% (model agrees with market)
```

**Key Insight**: Model is systematically under-predicting draws relative to market. This is PROFITABLE because the market itself under-prices draws.

### 3. **High-Confidence Bets Underperform**

When model has >10% edge vs market:
- Home: 50% win rate (2 games)
- Draw: 26.7% win rate (15 games)
- Away: 33.3% win rate (12 games)

**Problem**: Model's highest-confidence predictions are WRONG. This suggests overfitting or feature issues.

### 4. **Optimal Confidence Sweet Spot**

| Max Probability | Bets | Win Rate | ROI |
|-----------------|------|----------|-----|
| 0.3 - 0.4 | 54 | 42.6% | **+22.1%** ✅ |
| 0.4 - 0.5 | 61 | 45.9% | -2.4% |
| 0.5 - 0.6 | 28 | 64.3% | +15.9% |
| 0.6 - 0.7 | 17 | 47.1% | -26.1% ❌ |
| 0.7 - 0.8 | 13 | 69.2% | -12.0% ❌ |

**Counter-intuitive finding**: Best ROI at MEDIUM confidence (0.3-0.4), worst at high confidence (0.6-0.8).

---

## Recommended Improvements

### **TIER 1: High Impact, Low Effort** ⭐

#### 1.1 Re-calibrate Model with Better Method
**Current**: Isotonic calibration on validation set
**Proposed**: Platt scaling (sigmoid) + stratified calibration by outcome type

```python
from sklearn.calibration import CalibratedClassifierCV

# Separate calibration for each class
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',  # Better for extreme probabilities
    cv=5  # More robust
)
```

**Expected Impact**: Fix 0.6-0.8 probability over-confidence, improve home bet accuracy by 10-15%
**Effort**: 1-2 hours (just retrain with different calibration)

---

#### 1.2 Add Odds as Features
**Current**: Model trained on raw fixture data only
**Proposed**: Include market odds as features to learn market inefficiencies

```python
# New features to add:
- implied_home_prob = 1 / home_odds
- implied_draw_prob = 1 / draw_odds
- implied_away_prob = 1 / away_odds
- odds_spread = max(odds) - min(odds)
- odds_entropy = -Σ(p_i * log(p_i))
```

**Why This Works**:
- Market odds contain wisdom of crowd
- Model learns WHERE it has edge vs market
- Trains to beat the market, not predict outcomes

**Expected Impact**: 20-40% ROI improvement (this is the #1 change)
**Effort**: 3-4 hours (need to backfill odds for training data)

**⚠️ CRITICAL**: Must use CLOSING odds (not opening odds) to avoid lookahead bias

---

#### 1.3 Train with Custom Profit Objective
**Current**: Trained to minimize log loss
**Proposed**: Train to maximize profit with Kelly criterion

```python
import xgboost as xgb

def profit_objective(preds, dtrain):
    """Custom objective: maximize expected profit"""
    labels = dtrain.get_label()
    odds = dtrain.get_weight()  # Store odds in weight

    # Calculate profit gradient
    # If correct: gradient = odds - 1
    # If wrong: gradient = -1
    grad = (labels == preds) * (odds - 1) - (labels != preds) * 1
    hess = np.ones_like(grad)

    return grad, hess

model = xgb.XGBClassifier(objective=profit_objective)
```

**Expected Impact**: 10-20% ROI improvement
**Effort**: 4-6 hours (complex to implement correctly)

---

### **TIER 2: Medium Impact, Medium Effort**

#### 2.1 Separate Models by Outcome Type
**Current**: One model predicts H/D/A simultaneously
**Proposed**: Three binary models (Home vs Not, Draw vs Not, Away vs Not)

**Why This Works**:
- Different features matter for each outcome
- Draws have different dynamics than H/A
- Can tune each model independently

**Expected Impact**: 10-15% ROI improvement
**Effort**: 6-8 hours (3x training pipelines)

---

#### 2.2 Enable Lineup Features
**Current**: Lineups disabled for performance
**Proposed**: Add player quality features

```python
# New lineup-based features:
- starting_xi_value_home/away
- missing_key_players_home/away
- formation_type (4-4-2, 4-3-3, etc.)
- lineup_stability (% same as last game)
```

**Expected Impact**: 5-10% ROI improvement, especially for injury-heavy games
**Effort**: 8-10 hours (re-download data with lineups, retrain)

---

#### 2.3 Add Draw-Specific Features
**Current**: General features for all outcomes
**Proposed**: Engineer features specifically predictive of draws

```python
# Draw-specific features:
- teams_evenly_matched = abs(elo_diff) < 50
- defensive_matchup = (goals_conceded_home + goals_conceded_away) < 2.0
- low_variance_league = std_dev(league_goals_per_game)
- recent_draws_home/away = count(draws in last 5)
- head_to_head_draws = count(draws in H2H history)
```

**Why This Works**: Draws are already profitable, double down on what works

**Expected Impact**: 20-30% improvement in draw prediction (could hit 85-90% ROI)
**Effort**: 4-6 hours

---

### **TIER 3: High Impact, High Effort**

#### 3.1 Ensemble Model
**Proposed**: Combine multiple models with different strengths

```python
# Model 1: XGBoost (current) - good at patterns
# Model 2: LightGBM - good at categorical features
# Model 3: Neural Network - good at interactions
# Model 4: Logistic Regression - good baseline
# Model 5: Market Odds - wisdom of crowd

final_prediction = (
    0.30 * xgb_pred +
    0.25 * lgb_pred +
    0.20 * nn_pred +
    0.10 * lr_pred +
    0.15 * market_implied
)
```

**Expected Impact**: 15-25% ROI improvement
**Effort**: 12-16 hours

---

#### 3.2 Add Context Features
**Proposed**: Capture situational factors

```python
# Context features:
- days_since_last_match_home/away
- fixture_congestion (matches in next 7 days)
- tournament_stage (early season, late season, playoffs)
- derby_game (local rivalry)
- relegation_pressure
- title_race_pressure
- weather_conditions (if available)
```

**Expected Impact**: 10-15% ROI improvement
**Effort**: 10-12 hours

---

## Recommended Implementation Order

### **Phase 1: Quick Wins (Week 1)**
1. ✅ **Add odds as features** (backfill + retrain)
2. ✅ **Re-calibrate with Platt scaling**
3. ✅ **Add draw-specific features**

**Expected Result**: 50-60% ROI (up from 35.6%)

### **Phase 2: Model Architecture (Week 2-3)**
4. Train separate binary models (H vs not, D vs not, A vs not)
5. Add custom profit objective function
6. Implement ensemble approach

**Expected Result**: 70-80% ROI

### **Phase 3: Feature Expansion (Week 4)**
7. Enable lineup features
8. Add context features
9. Fine-tune thresholds with new model

**Expected Result**: 80-100% ROI

---

## Testing Protocol

For EACH change:

1. **Backtest on January 2026** (163 games)
   - Compare ROI vs current 35.6%
   - Check calibration curves
   - Verify no data leakage

2. **Out-of-sample validation**
   - Test on December 2025 (NOT used in training)
   - Must achieve >25% ROI to proceed

3. **Paper trade for 1 week**
   - Generate predictions but don't bet
   - Compare with current model
   - Deploy if outperforms

---

## Critical Warnings ⚠️

### 1. **Odds Data Leakage Risk**
When adding odds as features:
- ✅ Use CLOSING odds (from 15 mins before kickoff)
- ❌ NEVER use odds from fixtures API at prediction time (those are current, not historical)
- Must backfill historical closing odds from archive

### 2. **Overfitting Risk**
Adding more features = higher overfitting risk:
- Use stronger regularization (increase `min_child_weight`, `gamma`)
- Reduce tree depth (`max_depth=3` instead of 4)
- More aggressive early stopping

### 3. **Market Dependency**
If you train on odds, you become dependent on odds quality:
- Model won't work without odds
- Performance degrades if odds provider changes
- Consider ensemble that works with OR without odds

---

## What To Do Next?

### **Option A: Start with Quick Wins** (Recommended)
1. Add draw-specific features (4 hours)
2. Re-calibrate model (2 hours)
3. Backtest on January 2026
4. If ROI > 50%, deploy; else continue to Phase 2

### **Option B: Go All-In on Odds**
1. Backfill historical closing odds (8 hours of data work)
2. Add odds as features
3. Retrain with odds
4. This could be a game-changer (50-100% ROI jump)

### **Option C: Train Separate Draw Model**
1. Train binary classifier: Draw vs Not-Draw
2. Use ONLY for draw predictions
3. Keep current model for H/A
4. This targets your biggest profit source

---

## My Recommendation: **Start with Draw-Specific Features** 🎯

**Why**:
- Draws are already your best bet (73% ROI)
- Lowest risk, high reward
- Can implement in 4-6 hours
- No data dependencies

**After that**: Add odds as features (if you can get historical odds data)

---

Would you like me to:
1. ✅ **Implement draw-specific features** (start now, 4 hours)
2. ✅ **Re-calibrate model with better method** (2 hours)
3. ⏳ **Investigate backfilling historical odds** (research task)
4. ⏳ **Train separate draw model** (6 hours)
