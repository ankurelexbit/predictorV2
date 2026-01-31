# Model Improvement Strategy

**Based on:** Comprehensive Data Quality Analysis
**Current Status:** 150 features, 35,886 samples, XGBoost baseline
**Goal:** Improve prediction accuracy and log loss

---

## 🔴 Critical Data Quality Issues Found

### 1. **24 Features with >50% Missing Values**

These features have insufficient data and should be removed or imputed:

**High Missing (>60%):**
- All derived xG features (63-67% missing)
- Defensive actions and interceptions (74% missing)
- xG trends (65% missing)
- Shot-related features (56-63% missing)

**Impact:** These are mostly **Pillar 2** (Modern Analytics) features. Missing data reduces model accuracy.

**Solution:**
```python
# Option 1: Remove features with >50% missing
features_to_remove = [
    'away_interceptions_per_90', 'away_defensive_actions_per_90',
    'home_defensive_actions_per_90', 'home_interceptions_per_90',
    'derived_xgd_matchup', 'home_xga_trend_10', 'away_xga_trend_10',
    # ... (see data quality report for full list)
]

# Option 2: Impute with median/mean
df[high_missing_features] = df[high_missing_features].fillna(df[high_missing_features].median())
```

### 2. **22 Constant Features (100% Same Value)**

These features provide zero information and MUST be removed:

- Player features: `home_lineup_avg_rating_5`, `home_top_3_players_rating`, etc. (all constant)
- Context features: `rest_advantage`, `is_derby_match`, `home_days_since_last_match` (all constant)
- xG features: `home_xg_vs_top_half`, `home_xg_trend_10` (all constant)
- Big chances: `home_big_chances_per_match_5`, `away_big_chances_per_match_5` (all zeros)

**Root Cause:** These features are placeholder values that were never properly calculated.

### 3. **41 Highly Correlated Pairs (correlation = 1.0)**

Perfect correlation indicates redundancy:

**Examples:**
- `home_elo` ↔ `home_elo_vs_league_avg` (correlation = 1.0)
- `elo_diff` ↔ `elo_diff_with_home_advantage` (correlation = 1.0)
- All PPDA/tackle/xG corner features perfectly correlated

**Solution:** Remove one feature from each pair.

---

## 🎯 Recommended Action Plan

### Phase 1: Data Cleanup (Immediate)

**Step 1: Remove Bad Features**
```python
# Remove constant features (22)
constant_features = [
    'home_big_chances_per_match_5', 'away_big_chances_per_match_5',
    'home_xg_trend_10', 'away_xg_trend_10',
    'home_xg_vs_top_half', 'away_xg_vs_top_half',
    'home_xga_vs_bottom_half', 'away_xga_vs_bottom_half',
    'home_lineup_avg_rating_5', 'away_lineup_avg_rating_5',
    'home_top_3_players_rating', 'away_top_3_players_rating',
    'home_key_players_available', 'away_key_players_available',
    'home_players_in_form', 'away_players_in_form',
    'home_players_unavailable', 'away_players_unavailable',
    'home_days_since_last_match', 'away_days_since_last_match',
    'rest_advantage', 'is_derby_match'
]

# Remove high missing features (>50%)
high_missing_features = [
    'away_interceptions_per_90', 'away_defensive_actions_per_90',
    'home_defensive_actions_per_90', 'home_interceptions_per_90',
    'derived_xgd_matchup', 'home_xga_trend_10', 'away_xga_trend_10',
    'away_derived_xga_per_match_5', 'away_derived_xgd_5',
    'away_derived_xg_per_match_5', 'away_goals_vs_xg_5',
    'away_ga_vs_xga_5', 'home_derived_xga_per_match_5',
    'home_derived_xgd_5', 'home_ga_vs_xga_5',
    'home_goals_vs_xg_5', 'home_derived_xg_per_match_5',
    'away_shots_on_target_per_match_5', 'home_shots_on_target_conceded_5',
    'home_shots_on_target_per_match_5', 'away_xg_per_shot_5',
    'away_shot_accuracy_5', 'home_xg_per_shot_5', 'home_shot_accuracy_5'
]

# Remove redundant correlated features (keep one from each pair)
redundant_features = [
    'home_elo_vs_league_avg',  # Keep home_elo
    'away_elo_vs_league_avg',  # Keep away_elo
    'elo_diff_with_home_advantage',  # Keep elo_diff
    'home_ppda_5', 'away_ppda_5',  # Keep tackle success rates
    'home_xg_from_corners_5', 'away_xg_from_corners_5',
    'home_big_chance_conversion_5', 'away_big_chance_conversion_5',
    'home_inside_box_xg_ratio', 'away_inside_box_xg_ratio',
]

# Total features removed: ~70
# Effective features: ~80 high-quality features
```

**Expected Impact:** Reduce noise, improve model generalization, faster training

### Phase 2: Feature Engineering Improvements

**Fix Missing Data Sources:**

1. **Derived xG calculation is failing** - Check `pillar2_modern_analytics.py`
   - 63% missing suggests statistics are not available for many fixtures
   - Solution: Improve shot-based xG calculation or use simpler proxy

2. **Player features are not being calculated** - All constant at placeholder values
   - Check `pillar3_hidden_edges.py` player quality functions
   - Either fix or remove entirely

3. **Context features not calculated** - Rest days, derby match all zeros
   - Implement proper logic or remove

**New Features to Add:**

```python
# More robust features that don't rely on missing statistics:

# 1. Goal-based metrics (always available)
- home_goals_per_game_season
- away_goals_conceded_per_game_season
- home_clean_sheets_pct
- away_btts_pct

# 2. Points-based momentum
- home_points_vs_expected (based on position)
- away_points_vs_expected

# 3. Match context
- league_competitiveness (std dev of points)
- home_pressure_index (position - expected position)

# 4. Interaction features
- elo_diff * recent_form_diff
- home_attack_strength * away_defense_weakness
```

### Phase 3: Advanced Modeling Techniques

#### 1. **Better Imputation Strategies**

Instead of dropping features with missing values, use sophisticated imputation:

```python
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer

# MICE (Multiple Imputation by Chained Equations)
imputer = IterativeImputer(random_state=42, max_iter=10)
X_imputed = imputer.fit_transform(X_train)
```

#### 2. **Feature Selection**

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE

# Method 1: Mutual Information
selector = SelectKBest(score_func=mutual_info_classif, k=50)
X_selected = selector.fit_transform(X_train, y_train)

# Method 2: Recursive Feature Elimination
rfe = RFE(estimator=XGBClassifier(), n_features_to_select=50)
X_selected = rfe.fit_transform(X_train, y_train)

# Method 3: XGBoost Feature Importance
model.fit(X_train, y_train)
important_features = model.get_feature_importance().head(50)['feature'].tolist()
```

#### 3. **Ensemble Models** ⭐ **RECOMMENDED**

**A. LightGBM (Usually better than XGBoost for tabular data)**

```python
import lightgbm as lgb

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(X_train, y_train)
```

**Expected Improvement:** 2-5% better accuracy, 0.02-0.05 lower log loss

**B. CatBoost (Handles categorical features natively)**

```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    random_seed=42,
    verbose=False
)

cat_model.fit(X_train, y_train)
```

**Expected Improvement:** Similar to LightGBM, better with categorical data

**C. Stacking Ensemble** ⭐⭐ **BEST APPROACH**

Combine multiple models for superior performance:

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Base models
xgb_model = XGBClassifier(**xgb_params)
lgb_model = lgb.LGBMClassifier(**lgb_params)
cat_model = CatBoostClassifier(**cat_params)

# Stacking
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_model.fit(X_train, y_train)
```

**Expected Improvement:** 5-10% better accuracy, 0.05-0.10 lower log loss

#### 4. **Neural Networks** (If you want to try something different)

```python
import tensorflow as tf
from tensorflow import keras

# Deep neural network
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(n_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=100, batch_size=256, callbacks=[early_stop])
```

**Expected Improvement:** Variable (0-10%), requires more tuning

#### 5. **Hyperparameter Optimization**

Use Optuna for systematic tuning:

```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)
    return log_loss(y_val, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

---

## 📈 Expected Performance Improvements

| Approach | Difficulty | Time | Expected Improvement | Log Loss Reduction |
|----------|-----------|------|---------------------|-------------------|
| **Data Cleanup** | Easy | 1 hour | +2-3% accuracy | -0.02 to -0.03 |
| **Fix Missing Features** | Medium | 1 day | +3-5% accuracy | -0.03 to -0.05 |
| **LightGBM/CatBoost** | Easy | 2 hours | +2-5% accuracy | -0.02 to -0.05 |
| **Stacking Ensemble** | Medium | 1 day | +5-10% accuracy | -0.05 to -0.10 |
| **Neural Network** | Hard | 2-3 days | +0-10% accuracy | -0.00 to -0.10 |
| **Hyperparameter Tuning** | Medium | 1 day | +1-3% accuracy | -0.01 to -0.03 |

**Recommended Path:**
1. **Week 1:** Data cleanup + Fix missing features → +5-8% improvement
2. **Week 2:** Stacking ensemble (XGB + LGB + Cat) → +5-10% improvement
3. **Week 3:** Hyperparameter tuning → +1-3% improvement

**Total Expected Improvement: +11-21% accuracy, -0.08 to -0.18 log loss**

---

## 🚀 Quick Wins (Do This First)

1. **Remove 70 bad features** (constant + high missing + redundant)
   - Expected: +2-3% accuracy immediately

2. **Try LightGBM instead of XGBoost**
   - Same API, often better performance
   - Expected: +2-5% accuracy

3. **Implement proper train/val/test split**
   - Ensure chronological splitting (already done ✓)

4. **Class weights for draw prediction**
   ```python
   from sklearn.utils.class_weight import compute_class_weight

   class_weights = compute_class_weight('balanced', classes=[0,1,2], y=y_train)
   model = XGBClassifier(scale_pos_weight=class_weights)
   ```

5. **Calibration** (already done ✓)
   - Isotonic regression on validation set

---

## 💡 Additional Recommendations

### 1. **Target Engineering**

Instead of predicting H/D/A directly, try:

```python
# Multi-output: Predict goals for both teams
target_home_goals = df['home_score']
target_away_goals = df['away_score']

# Then derive H/D/A from goal predictions
```

### 2. **League-Specific Models**

Train separate models for different leagues:

```python
for league_id in [8, 39, 140, 78]:  # PL, La Liga, Serie A, Bundesliga
    df_league = df[df['league_id'] == league_id]
    model_league = train_model(df_league)
    save_model(model_league, f'model_league_{league_id}.pkl')
```

### 3. **Time-Based Features**

Add temporal patterns:

```python
df['month'] = pd.to_datetime(df['match_date']).dt.month
df['day_of_week'] = pd.to_datetime(df['match_date']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])
```

### 4. **Market Odds Integration**

If you have access to betting odds, they're extremely predictive:

```python
# Convert odds to implied probabilities
df['market_home_prob'] = 1 / df['home_odds']
df['market_draw_prob'] = 1 / df['draw_odds']
df['market_away_prob'] = 1 / df['away_odds']
```

---

## 🎯 Final Recommendations

**Immediate (This Week):**
1. ✅ Run data cleanup script to remove 70 bad features
2. ✅ Try LightGBM with same parameters as XGBoost
3. ✅ Fix feature generation for derived xG and player features

**Short-term (Next 2 Weeks):**
1. ✅ Implement stacking ensemble (XGB + LGB + CatBoost)
2. ✅ Use Optuna for hyperparameter tuning
3. ✅ Add new engineered features (interactions, temporal)

**Long-term (Next Month):**
1. ✅ Explore neural networks (if time permits)
2. ✅ Build league-specific models
3. ✅ Integrate market odds (if available)

**Expected Final Performance:**
- **Current:** ~45% accuracy, ~0.95 log loss (estimated)
- **After improvements:** ~55-65% accuracy, ~0.80-0.85 log loss
- **Best case:** ~65-70% accuracy, ~0.75-0.80 log loss
