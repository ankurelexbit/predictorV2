# Constant Features Analysis

**Date:** 2026-02-03
**Issue:** 8 features have the same value for ALL 17,943 rows

---

## Summary

**8 features are truly constant (zero information):**

| Feature | Constant Value | Why Constant |
|---------|----------------|--------------|
| `home_xg_vs_top_half` | 1.2 | Not implemented - placeholder |
| `away_xg_vs_top_half` | 1.1 | Not implemented - placeholder |
| `home_xga_vs_bottom_half` | 0.8 | Not implemented - placeholder |
| `away_xga_vs_bottom_half` | 0.9 | Not implemented - placeholder |
| `home_key_players_available` | 4 | No lineup data available |
| `away_key_players_available` | 4 | No lineup data available |
| `home_players_unavailable` | 1 | Sidelined data not used |
| `away_players_unavailable` | 1 | Sidelined data not used |

**Impact on Model:** These features provide **ZERO predictive value** but waste model capacity.

---

## Root Causes

### 1. xG vs Top/Bottom Teams (4 features)

**Location:** `src/features/pillar3_hidden_edges.py:248-251`

**Code:**
```python
return {
    'home_xg_vs_top_half': 1.2,  # Placeholder
    'away_xg_vs_top_half': 1.1,
    'home_xga_vs_bottom_half': 0.8,
    'away_xga_vs_bottom_half': 0.9,
}
```

**Why Constant:**
These features require:
1. Classify teams as "top half" or "bottom half" of table
2. Track xG performance separately against each group
3. Currently not implemented - just hardcoded placeholders

**Can Be Fixed?** YES - Requires implementation

---

### 2. Key Players Available (2 features)

**Location:** `src/features/player_features.py:108-109`

**Code:**
```python
if lineups:
    # Calculate from actual lineup
    features['home_key_players_available'] = self._count_key_players_starting(...)
else:
    # Lineup data not available - use default
    features['home_key_players_available'] = 4
```

**Why Constant:**
- Lineup data is **0% available** in historical fixtures
- Lineups only available 1-2 hours before kickoff
- Historical data doesn't include lineups
- Always uses default fallback: 4

**Can Be Fixed?** NO - Lineup data not in historical dataset

---

### 3. Players Unavailable (2 features)

**Location:** `src/features/player_features.py:316`

**Code:**
```python
def _get_sidelined_count(self, team_id: int, as_of_date: datetime) -> int:
    """Get count of sidelined players (injured/suspended)."""
    # In production, this would check sidelined data
    # For training, we use conservative estimates
    return 1  # Assume 1 player unavailable on average
```

**Why Constant:**
- Sidelined data (injuries/suspensions) not used during training
- Point-in-time sidelined info not available for historical matches
- Returns hardcoded estimate: 1

**Can Be Fixed?** MAYBE - If sidelined data exists in historical JSON

---

## Impact Assessment

### Model Training Impact

**Capacity Waste:**
- 8 constant features out of 162 total (4.9%)
- No predictive value whatsoever
- Model learns to ignore them (zero feature importance)

**Training Time:**
- Minimal impact (XGBoost handles constant features efficiently)
- Slight slowdown from extra columns

**Model Performance:**
- No direct harm (constant features get zero weight)
- Wasted opportunity to use better features

---

## Options

### Option 1: Remove Constant Features ✅ RECOMMENDED

**Action:** Drop these 8 features from training

**Implementation:**
```python
# In scripts/train_production_model.py
CONSTANT_FEATURES = [
    'home_xg_vs_top_half', 'away_xg_vs_top_half',
    'home_xga_vs_bottom_half', 'away_xga_vs_bottom_half',
    'home_key_players_available', 'away_key_players_available',
    'home_players_unavailable', 'away_players_unavailable'
]

# Drop before training
X_train = X_train.drop(columns=CONSTANT_FEATURES, errors='ignore')
X_val = X_val.drop(columns=CONSTANT_FEATURES, errors='ignore')
X_test = X_test.drop(columns=CONSTANT_FEATURES, errors='ignore')
```

**Pros:**
- ✅ Cleaner feature set
- ✅ Slightly faster training
- ✅ Easier to interpret feature importance
- ✅ No wasted model capacity

**Cons:**
- Need to remember to drop them in production predictions too

---

### Option 2: Keep As-Is (Do Nothing)

**Action:** Leave constant features in dataset

**Pros:**
- ✅ No code changes needed
- ✅ Model will assign zero importance automatically
- ✅ Can implement later without retraining

**Cons:**
- ❌ Wasted 4.9% of feature space
- ❌ Cluttered feature set
- ❌ Misleading feature count (162 vs actual 154)

---

### Option 3: Implement Missing Features

**Action:** Properly calculate the 4 xG vs top/bottom features

**Implementation needed:**
```python
def _get_fixture_adjusted_features(...):
    # Get standings at this date
    standings = self.standings_calc.calculate_standings_at_date(...)

    # Split into top/bottom half
    mid_point = len(standings) // 2
    top_half_teams = set(standings.iloc[:mid_point]['team_id'])
    bottom_half_teams = set(standings.iloc[mid_point:]['team_id'])

    # Calculate xG vs each group
    home_xg_vs_top = self._calculate_xg_vs_group(home_team_id, top_half_teams, ...)
    home_xga_vs_bottom = self._calculate_xga_vs_group(home_team_id, bottom_half_teams, ...)

    return {
        'home_xg_vs_top_half': home_xg_vs_top,
        'away_xg_vs_top_half': away_xg_vs_top,
        'home_xga_vs_bottom_half': home_xga_vs_bottom,
        'away_xga_vs_bottom_half': away_xga_vs_bottom,
    }
```

**Effort:** 1-2 hours

**Pros:**
- ✅ Useful features (teams perform differently vs strong/weak opponents)
- ✅ Full 162 features as designed

**Cons:**
- ❌ Requires implementation work
- ❌ More complex code
- ⚠️ Uncertain if worth the effort (might not improve model much)

---

## Recommendation

### SHORT TERM: Option 1 (Remove constant features)

**Reason:**
- Quick fix (5 minutes)
- Cleaner dataset
- No downside

**Implementation:**
Add this to training script:
```python
# Drop constant features that provide zero information
CONSTANT_FEATURES_TO_DROP = [
    'home_xg_vs_top_half', 'away_xg_vs_top_half',
    'home_xga_vs_bottom_half', 'away_xga_vs_bottom_half',
    'home_key_players_available', 'away_key_players_available',
    'home_players_unavailable', 'away_players_unavailable'
]

print(f"Dropping {len(CONSTANT_FEATURES_TO_DROP)} constant features...")
for df in [X_train, X_val, X_test]:
    df.drop(columns=CONSTANT_FEATURES_TO_DROP, errors='ignore', inplace=True)

print(f"Final feature count: {len(X_train.columns)}")
```

### LONG TERM: Option 3 (Implement xG vs top/bottom)

**After model is working well, consider implementing:**
- `home/away_xg_vs_top_half`
- `home/away_xga_vs_bottom_half`

**Skip these (data not available):**
- `home/away_key_players_available` - Need lineup data
- `home/away_players_unavailable` - Need point-in-time sidelined data

---

## Feature Importance After Removal

**Current (with constants):**
- Total features: 162
- Actually useful: 154
- Constant (useless): 8

**After removal:**
- Total features: 154
- All features contribute information
- Cleaner feature importance rankings

---

## Code Changes Needed (Option 1)

### 1. Update Training Script

**File:** `scripts/train_production_model.py`

Add after loading data, before training:

```python
# Drop constant features
CONSTANT_FEATURES = [
    'home_xg_vs_top_half', 'away_xg_vs_top_half',
    'home_xga_vs_bottom_half', 'away_xga_vs_bottom_half',
    'home_key_players_available', 'away_key_players_available',
    'home_players_unavailable', 'away_players_unavailable'
]

logger.info(f"Dropping {len(CONSTANT_FEATURES)} constant features...")
X_train = X_train.drop(columns=CONSTANT_FEATURES, errors='ignore')
X_val = X_val.drop(columns=CONSTANT_FEATURES, errors='ignore')
X_test = X_test.drop(columns=CONSTANT_FEATURES, errors='ignore')
logger.info(f"Final feature count: {X_train.shape[1]}")
```

### 2. Update Prediction Script

**File:** `scripts/predict_live_with_history.py`

Add after feature generation:

```python
# Drop constant features (same as training)
CONSTANT_FEATURES = [
    'home_xg_vs_top_half', 'away_xg_vs_top_half',
    'home_xga_vs_bottom_half', 'away_xga_vs_bottom_half',
    'home_key_players_available', 'away_key_players_available',
    'home_players_unavailable', 'away_players_unavailable'
]

features_df = features_df.drop(columns=CONSTANT_FEATURES, errors='ignore')
```

### 3. Document in README

Add note about constant features being excluded.

---

## Testing After Removal

```bash
# Verify features dropped correctly
python3 -c "
import pandas as pd

df = pd.read_csv('data/training_data.csv')

# Check constant features
constant_features = [
    'home_xg_vs_top_half', 'away_xg_vs_top_half',
    'home_xga_vs_bottom_half', 'away_xga_vs_bottom_half',
    'home_key_players_available', 'away_key_players_available',
    'home_players_unavailable', 'away_players_unavailable'
]

# Verify they're constant
for feat in constant_features:
    unique = df[feat].nunique()
    print(f'{feat}: {unique} unique values')

print(f'\nTotal features in CSV: {len(df.columns)}')
print(f'Will train with: {len(df.columns) - 9 - 8} features')
print(f'  (Minus 9 metadata + 8 constant = {len(df.columns) - 17})')
"
```

---

## Summary

✅ **Confirmed:** 8 features are constant (same value for all 17,943 rows)
✅ **Impact:** Zero predictive value, waste 4.9% of feature space
✅ **Recommendation:** Drop them during training (Option 1)
✅ **Effort:** 5 minutes to implement
✅ **Benefit:** Cleaner model, easier interpretation

**Next step:** Add constant feature dropping to training script, then retrain.
