# Do Training Data Fixes Apply to Live Predictions?

**Date:** 2026-02-03
**Answer:** ✅ **YES - All fixes automatically apply to live predictions**

---

## Why All Fixes Apply Automatically

### Architecture: Shared Feature Engines

Both training and prediction pipelines use the **SAME core feature engines**:

```python
# Training Pipeline (generate_training_data.py / feature_orchestrator.py)
pillar1 = Pillar1FundamentalsEngine(data_loader, standings_calc, elo_calc)
pillar2 = Pillar2ModernAnalyticsEngine(data_loader)
pillar3 = Pillar3HiddenEdgesEngine(data_loader, standings_calc, elo_calc)

# Live Prediction Pipeline (predict_live_with_history.py)
self.pillar1_engine = Pillar1FundamentalsEngine(...)  # Lines 447-451
self.pillar2_engine = Pillar2ModernAnalyticsEngine(...)  # Line 452
self.pillar3_engine = Pillar3HiddenEdgesEngine(...)  # Lines 453-457
```

### Feature Generation: Same Methods

Both pipelines call the same `generate_features()` methods:

**Training Pipeline:**
```python
# feature_orchestrator.py
features = {}
features.update(self.pillar1.generate_features(...))
features.update(self.pillar2.generate_features(...))
features.update(self.pillar3.generate_features(...))
```

**Live Prediction Pipeline:**
```python
# predict_live_with_history.py (lines 510-538)
features = {}
features.update(self.pillar1_engine.generate_features(...))  # Same method!
features.update(self.pillar2_engine.generate_features(...))  # Same method!
features.update(self.pillar3_engine.generate_features(...))  # Same method!
```

---

## All Fixes and Where They Live

### ✅ Fix 1: Big Chances Calculation
**File:** `src/features/pillar2_modern_analytics.py:197-217`
**Used By:** Both training and prediction
**Status:** ✅ Automatically applies to live predictions

**Implementation:**
```python
def _extract_team_stats(self, match: pd.Series, is_home: bool) -> Dict:
    shots_total = match.get(f'{prefix}shots_total', 0) or 0
    shots_on_target = match.get(f'{prefix}shots_on_target', 0) or 0
    shots_inside_box = match.get(f'{prefix}shots_inside_box', 0) or 0

    # Calculate big chances
    if shots_total > 0:
        accuracy = shots_on_target / shots_total
        big_chances = shots_inside_box * accuracy * 0.8
    else:
        big_chances = 0

    return {'big_chances_created': big_chances, ...}
```

---

### ✅ Fix 2: xG Trends NaN Handling
**File:** `src/features/pillar2_modern_analytics.py:218-231`
**Used By:** Both training and prediction
**Status:** ✅ Automatically applies to live predictions

**Implementation:**
```python
def _calculate_trend(self, values: list) -> float:
    # Filter out NaN values before trend calculation
    clean_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]

    if len(clean_values) < 2:
        return 0.0

    x = np.arange(len(clean_values))
    try:
        slope = np.polyfit(x, clean_values, 1)[0]
        return float(slope) if not np.isnan(slope) else 0.0
    except:
        return 0.0
```

---

### ✅ Fix 3: Rest Advantage Extreme Value Capping
**File:** `src/features/pillar3_hidden_edges.py:354-371`
**Used By:** Both training and prediction
**Status:** ✅ Automatically applies to live predictions

**Implementation:**
```python
if len(home_last_match) > 0:
    home_rest = (as_of_date - home_last_match.iloc[0]['starting_at']).days
    # Cap at reasonable maximum (60 days) and minimum (2 days)
    home_rest = max(2, min(60, home_rest))
else:
    home_rest = 7
```

---

### ✅ Fix 4: Derby Detection
**File:** `src/features/pillar3_hidden_edges.py:547-595`
**Used By:** Both training and prediction
**Status:** ✅ Automatically applies to live predictions

**Implementation:**
```python
def _is_derby_match(self, home_team_id: int, away_team_id: int) -> int:
    DERBY_PAIRS = {
        frozenset([14, 15]),   # Manchester derby
        frozenset([18, 6]),    # North London derby
        # ... 18 major derbies
    }
    match_pair = frozenset([home_team_id, away_team_id])
    return 1 if match_pair in DERBY_PAIRS else 0
```

---

### ✅ Fix 5: xG vs Top/Bottom Features
**File:** `src/features/pillar3_hidden_edges.py:290-454`
**Used By:** Both training and prediction
**Status:** ✅ Automatically applies to live predictions

**Implementation:**
```python
def _calculate_xg_vs_opposition_groups(...) -> Dict:
    # Get standings and split into top/bottom half
    standings = self.standings_calc.calculate_standings_at_date(...)
    mid_point = len(standings) // 2
    top_half_teams = set(standings.iloc[:mid_point]['team_id'])
    bottom_half_teams = set(standings.iloc[mid_point:]['team_id'])

    # Calculate xG performance against each group
    home_xg_vs_top = self._calculate_xg_vs_opposition_group(
        team_id=home_team_id,
        opposition_group=top_half_teams,
        ...
    )
    ...
```

---

### ✅ Fix 6: Players Unavailable Estimation
**File:** `src/features/player_features.py:303-440`
**Used By:** Both training and prediction
**Status:** ✅ Automatically applies to live predictions

**Implementation:**
```python
def _estimate_sidelined_count(self, team_id: int, as_of_date: datetime) -> int:
    base_unavailable = 2

    # Factor 1: Fixture congestion
    congestion_factor = 0
    # (counts recent matches in last 14 days)

    # Factor 2: Time of season
    season_factor = 0
    if month in [12, 1, 4]:  # High-risk months
        season_factor = 1

    total = base_unavailable + congestion_factor + season_factor
    return max(1, min(5, total))
```

---

## Verification

### How to Verify Live Predictions Use Fixed Features

**Test live prediction after retraining:**

```bash
# 1. Regenerate training data with fixes
python3 scripts/generate_training_data.py --output data/training_data.csv

# 2. Retrain model
python3 scripts/train_production_model.py \
  --data data/training_data.csv \
  --output models/production/v4_with_fixes.joblib

# 3. Initialize live prediction pipeline
export SPORTMONKS_API_KEY="your_key"
python3 scripts/predict_live_with_history.py --verify

# 4. Make a prediction and inspect features
python3 scripts/predict_live_with_history.py --predict-upcoming --days-ahead 7
```

**Check prediction output:**
The prediction will return all features in the `features` dict (line 551 in predict_live_with_history.py):

```python
return {
    'home_prob': float(probs[2]),
    'draw_prob': float(probs[1]),
    'away_prob': float(probs[0]),
    'features': features,  # All features used for this prediction
}
```

You can inspect these features to verify:
- `home_big_chances_per_match_5` is not 0
- `home_xg_trend_10` varies (not always 0)
- `rest_advantage` is in range [-58, +58] (not -2500)
- `is_derby_match` is 1 for derby matches
- `home_xg_vs_top_half` varies (not constant 1.2)
- `home_players_unavailable` is 1-5 (varies by month/congestion)

---

## Summary

### ✅ All Fixes Apply Automatically

| Fix | File | Training | Live Prediction |
|-----|------|----------|-----------------|
| Big chances calculation | `pillar2_modern_analytics.py` | ✅ | ✅ |
| xG trends NaN handling | `pillar2_modern_analytics.py` | ✅ | ✅ |
| Rest advantage capping | `pillar3_hidden_edges.py` | ✅ | ✅ |
| Derby detection | `pillar3_hidden_edges.py` | ✅ | ✅ |
| xG vs top/bottom | `pillar3_hidden_edges.py` | ✅ | ✅ |
| Players unavailable | `player_features.py` | ✅ | ✅ |

### 🎯 Key Takeaway

**You only need to regenerate training data and retrain the model.**

The live prediction pipeline will automatically use all the fixes because it uses the same feature generation code. No changes to prediction scripts are needed.

### 📋 Action Checklist

- [ ] Regenerate training data with fixes
- [ ] Verify new training data has correct feature values
- [ ] Retrain model with new training data
- [ ] Deploy new model to production
- [ ] ✅ Live predictions automatically use all fixes (no action needed!)

---

## Additional Notes

### Why This Architecture is Good

1. **Single Source of Truth**: Feature logic exists in one place
2. **Consistency**: Training and prediction use identical features
3. **No Drift**: Can't have training/serving skew
4. **Easy Maintenance**: Fix once, applies everywhere
5. **Point-in-Time Correctness**: Both pipelines respect temporal constraints

### Future Feature Development

When adding new features:
1. Add to appropriate pillar engine (`pillar1/2/3_*.py`)
2. Feature automatically works in both training and prediction
3. No need to duplicate logic

This is why the V4 pipeline architecture is production-ready! 🚀
