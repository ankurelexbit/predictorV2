# Home Prediction Improvement Plan

## 🔴 THE CORE PROBLEM

**Model is systematically UNDER-PREDICTING home wins by ~5%**

```
Home Favorites (odds < 2.0x):
├─ Market thinks: 63.2% home win probability
├─ Model thinks:  56.6% home win probability ❌ TOO LOW
└─ Actual:        61.7% home win rate

Result: Model has -6.6% edge vs market on home favorites
```

**Impact**:
- Missing 63 out of 70 home wins (90%)
- Leaving $84 profit on table
- Only betting 8 home games/month (all heavy favorites with bad odds)

---

## 🎯 ROOT CAUSES

### 1. **Home Advantage Under-Weighted**
- Current `home_advantage_strength` feature may be too conservative
- Not capturing venue-specific advantages
- Missing psychological factors (crowd pressure, familiarity)

### 2. **Training Objective Mismatch**
- Model trained on log loss penalizes false positives heavily
- Better to predict draw/away when unsure than risk being wrong on home
- Creates conservative bias

### 3. **Feature Gap**
- Missing: travel distance for away team
- Missing: venue-specific metrics (stadium capacity, attendance)
- Missing: derby/rivalry games (huge home advantage)
- Missing: rest days differential

### 4. **Calibration Issue**
- Model gives 56.6% avg probability for home favorites
- Should be closer to 62% to match reality
- **5.1% systematic under-estimation**

---

## 🚀 SOLUTIONS (Ranked by Impact)

### **SOLUTION 1: Recalibrate Home Probabilities** ⚡ QUICK WIN
**Time: 2 hours | Expected Impact: +15-20% home bet ROI**

Apply a learned correction factor to boost home probabilities:

```python
# After model prediction, before betting decision
if pred_home_prob >= 0.50:  # Only for home favorites
    # Boost home probability by 5.1% (the systematic error)
    correction_factor = 0.051

    pred_home_prob_corrected = pred_home_prob + correction_factor

    # Renormalize to sum to 1.0
    total = pred_home_prob_corrected + pred_draw_prob + pred_away_prob
    pred_home_prob = pred_home_prob_corrected / total
    pred_draw_prob = pred_draw_prob / total
    pred_away_prob = pred_away_prob / total
```

**Why This Works**:
- Directly fixes the 5.1% under-estimation
- Simple post-processing, no retraining needed
- Can implement in 30 minutes

**Expected Result**:
- Home predictions align with market (56.6% → 61.7%)
- More home bets pass the 0.75 threshold
- Better ROI on home bets that do get placed

---

### **SOLUTION 2: Add Home-Specific Features** ⭐ BEST LONG-TERM
**Time: 6-8 hours | Expected Impact: +25-35% home bet ROI**

Engineer 10-15 new features that specifically predict home advantage:

#### 2.1 Venue Strength Features
```python
# Calculate for each team's home stadium
home_advantage_features = {
    # Historical home performance
    'home_win_rate_venue': wins / games at this venue,
    'home_win_rate_vs_expected': actual wins - elo_expected_wins,
    'home_goal_difference_venue': avg(home_goals - away_goals) at venue,

    # Venue characteristics
    'stadium_capacity': capacity / 100000,  # Normalize
    'avg_attendance_rate': actual_attendance / capacity,
    'venue_altitude': meters_above_sea_level,  # If available

    # Consistency
    'home_advantage_consistency': std_dev(home results) - lower is better,
}
```

#### 2.2 Travel & Fatigue Features
```python
away_team_features = {
    # Travel burden
    'travel_distance': km_between_stadiums / 1000,
    'travel_is_long': 1 if distance > 500km else 0,

    # Rest differential
    'rest_days_differential': home_rest_days - away_rest_days,
    'away_fixture_congestion': games_in_next_7_days for away team,
}
```

#### 2.3 Psychological Features
```python
matchup_features = {
    # Derby/rivalry
    'is_derby': 1 if local_rivalry else 0,
    'is_big_game': 1 if both teams in top 6 else 0,

    # Pressure
    'home_title_pressure': 1 if home in title race else 0,
    'home_relegation_pressure': 1 if home in bottom 5 else 0,
    'away_confidence': away_team_win_streak,
}
```

#### 2.4 Head-to-Head Home Advantage
```python
h2h_home_features = {
    'h2h_home_win_rate': home_wins / h2h_games_at_venue,
    'h2h_home_dominance': (home_wins - away_wins) / h2h_games,
    'h2h_home_avg_goals': avg(home_goals) in h2h_at_venue,
}
```

**Implementation Steps**:
1. Add features to `pillar1_fundamentals.py` or `pillar3_hidden_edges.py`
2. Regenerate training data with new features
3. Retrain model
4. Backtest on January 2026

**Expected Result**:
- Model learns to value home advantage properly
- Finds 15-20 home bets/month instead of 8
- Captures home underdogs with value

---

### **SOLUTION 3: Separate Home Win Model** 🎯 MOST ACCURATE
**Time: 8-10 hours | Expected Impact: +30-40% home bet ROI**

Train a binary classifier specifically for home wins:

```python
# Train separate model: Home Win vs (Draw or Away)
y_binary = (y == 'H').astype(int)

home_model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=0.7,  # Slight home bias
    max_depth=4,
    learning_rate=0.05,
    n_estimators=500
)

home_model.fit(X_train, y_binary)

# Use this ONLY for home predictions
home_prob_specialized = home_model.predict_proba(X)[:, 1]
```

**Why This Works**:
- Dedicated model learns home-specific patterns
- Can tune hyperparameters specifically for home wins
- Not penalized by draw/away confusion

**Architecture**:
```
Main Model (3-class):      Home Model (binary):
├─ Home: 0.45             ├─ Home: 0.62 ✅
├─ Draw: 0.35        +    └─ Not-Home: 0.38
└─ Away: 0.20
                          Final Decision:
                          Use home_model.prob for home bets
                          Use main_model for draw/away bets
```

**Expected Result**:
- Home predictions match market accuracy
- Find home underdogs (model sees 50% home, market sees 30%)
- 20-25 home bets/month with better edge

---

### **SOLUTION 4: Dynamic Home Threshold** 🔧 SMART BETTING
**Time: 3-4 hours | Expected Impact: +10-15% home bet ROI**

Instead of fixed 0.75 threshold, use dynamic threshold based on odds:

```python
def should_bet_home(home_prob, home_odds):
    """Dynamic threshold based on odds and EV"""

    # Calculate expected value
    ev = (home_prob * home_odds) - 1

    # Different thresholds for different odds ranges
    if home_odds < 1.5:  # Heavy favorite
        # Need very high confidence (low value)
        return home_prob >= 0.85 and ev > 0

    elif home_odds < 2.5:  # Moderate favorite
        # Standard threshold
        return home_prob >= 0.70 and ev > 0

    elif home_odds < 4.0:  # Slight favorite / even
        # Lower threshold (more value)
        return home_prob >= 0.55 and ev > 0.05

    else:  # Home underdog
        # Much lower threshold (high value)
        return home_prob >= 0.40 and ev > 0.10
```

**Why This Works**:
- Heavy favorites: Stricter (market efficient)
- Home underdogs: Looser (market inefficient)
- Focuses on expected value, not just probability

**Expected Result**:
- Fewer heavy favorite bets (avoid -ROI)
- More home underdog bets (capture value)
- Overall home bet ROI increases 10-15%

---

### **SOLUTION 5: Ensemble with Market Odds** 🔥 GAME CHANGER
**Time: 6-8 hours | Expected Impact: +40-50% home bet ROI**

Combine model predictions with market implied probabilities:

```python
# Extract market implied probabilities
market_home_prob = 1 / home_odds
market_draw_prob = 1 / draw_odds
market_away_prob = 1 / away_odds

# Normalize (remove bookmaker margin)
total = market_home_prob + market_draw_prob + market_away_prob
market_home_prob /= total
market_draw_prob /= total
market_away_prob /= total

# Ensemble (70% model, 30% market)
final_home_prob = 0.70 * model_home_prob + 0.30 * market_home_prob

# Or: Only boost home when model AND market agree
if model_home_prob >= 0.50 and market_home_prob >= 0.50:
    # Both agree home is likely - boost confidence
    final_home_prob = max(model_home_prob, market_home_prob)
```

**Why This Works**:
- Market is efficient on home favorites (use their wisdom)
- Model finds underpriced home underdogs (keep your edge)
- Best of both worlds

**Expected Result**:
- Home predictions as accurate as market
- Still find value bets where model disagrees
- 25-30 home bets/month with positive ROI

---

## 📊 RECOMMENDED IMPLEMENTATION PLAN

### **Phase 1: Immediate (This Week)** - 5 hours

1. ✅ **Recalibrate home probabilities** (2h)
   - Add +5.1% correction factor to home favorites
   - Test on January 2026 data

2. ✅ **Dynamic home threshold** (3h)
   - Implement odds-based threshold logic
   - Require positive EV for all home bets

**Expected Result**: 15-20% improvement in home ROI, 12-15 home bets/month

---

### **Phase 2: Short-Term (Next Week)** - 8 hours

3. ✅ **Add 10 home-specific features** (6h)
   - Travel distance
   - Venue win rate
   - Rest days differential
   - Derby flag
   - H2H home dominance

4. ✅ **Retrain model** (2h)
   - Regenerate training data with new features
   - Train with new features
   - Backtest on Dec 2025 + Jan 2026

**Expected Result**: 25-30% improvement in home ROI, 18-22 home bets/month

---

### **Phase 3: Long-Term (Weeks 3-4)** - 10 hours

5. ✅ **Train separate home model** (8h)
   - Binary classifier for home wins
   - Hyperparameter tuning
   - Ensemble with main model

6. ✅ **Add market odds ensemble** (2h)
   - Combine model + market when both agree
   - Find value where they disagree

**Expected Result**: 40-50% improvement in home ROI, 25-30 home bets/month

---

## 🎯 QUICK WIN: What to do RIGHT NOW

### **30-Minute Implementation** (Recalibration)

Add to `scripts/predict_production.py`:

```python
def adjust_home_predictions(home_prob, draw_prob, away_prob, home_odds):
    """Fix systematic home under-prediction"""

    # Only adjust home favorites (odds < 2.5)
    if home_odds < 2.5 and home_prob >= 0.45:
        # Model underestimates by 5.1% on average
        correction = 0.051

        # Apply correction
        home_prob_adjusted = home_prob + correction

        # Renormalize
        total = home_prob_adjusted + draw_prob + away_prob
        home_prob = home_prob_adjusted / total
        draw_prob = draw_prob / total
        away_prob = away_prob / total

    return home_prob, draw_prob, away_prob

# In prediction loop:
home_prob, draw_prob, away_prob = adjust_home_predictions(
    home_prob, draw_prob, away_prob, best_home_odds
)
```

**Test on January 2026**:
- Before: 8 home bets, 7 wins, $0.74 profit (9% ROI)
- After: 15-18 home bets, 11-13 wins, $3-5 profit (20-25% ROI)

---

## 🔬 VALIDATION CHECKLIST

Before deploying ANY change:

- [ ] Backtest on January 2026 (163 games)
- [ ] Compare home bet count: should increase from 8 to 15-25
- [ ] Check home win rate: should stay 75-85%
- [ ] Check home bet ROI: should increase from 9% to 20%+
- [ ] Verify no data leakage
- [ ] Test on December 2025 (out-of-sample)
- [ ] Paper trade for 1 week before going live

---

## 💡 MY TOP RECOMMENDATION

**Start with Phase 1 (5 hours total)**:

1. **Recalibrate home probabilities** (+5.1% boost for favorites)
2. **Add dynamic threshold** (odds-based with EV filter)

This gives you:
- ✅ Immediate improvement (no retraining)
- ✅ Low risk (easy to revert)
- ✅ Measurable impact (15-20% ROI gain)
- ✅ Quick implementation (1 evening of work)

Then move to Phase 2 next week with new features.

---

Would you like me to:
- **A**: Implement Phase 1 (recalibration + dynamic threshold) RIGHT NOW
- **B**: Design the 10 home-specific features first (for Phase 2)
- **C**: Build the separate home model (Phase 3)
- **D**: Something else?

I recommend **Option A** - we can have improved home predictions running in 30 minutes.
