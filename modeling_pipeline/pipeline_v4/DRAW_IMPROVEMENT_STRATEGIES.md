# Strategies to Improve Draw Prediction

## Current Performance
- Conservative model: 22.1% draw accuracy (146 correct out of 662 draws)
- Problem: 44% of draws predicted as Home, 34% as Away, only 22% as Draw

## Strategy 1: Increase Draw Class Weights ⭐ **TRY FIRST**

Test higher draw multipliers:
- Current: 1.5x
- Try: 2.0x, 2.5x, 3.0x

**Run**: `python3 scripts/train_high_draw_weights.py`

Expected trade-off: Draw accuracy ↑, Log loss may slightly ↑

---

## Strategy 2: Add Draw-Specific Features

### 2.1 Team Strength Parity Features
Add to `pillar3_hidden_edges.py`:

```python
# Elo difference (closer to 0 = more likely draw)
home_elo_diff = abs(home_elo - away_elo)

# Recent form parity
home_form_diff = abs(home_last_5_points - away_last_5_points)

# League position difference
position_diff = abs(home_position - away_position)

# Head-to-head draw rate
h2h_draw_rate = h2h_draws / total_h2h_matches

# Attack/defense balance similarity
attack_diff = abs(home_avg_goals_scored - away_avg_goals_scored)
defense_diff = abs(home_avg_goals_conceded - away_avg_goals_conceded)
```

### 2.2 Historical Draw Tendency Features

```python
# Team's draw rate in last N games
home_recent_draw_rate = home_draws_last_10 / 10
away_recent_draw_rate = away_draws_last_10 / 10

# Combined draw tendency
combined_draw_tendency = (home_recent_draw_rate + away_recent_draw_rate) / 2

# League draw rate (some leagues have more draws)
league_avg_draw_rate = league_draws / total_league_matches
```

### 2.3 Situational Draw Indicators

```python
# Mid-table teams draw more
home_is_midtable = 1 if 7 <= home_position <= 14 else 0
away_is_midtable = 1 if 7 <= away_position <= 14 else 0
both_midtable = home_is_midtable * away_is_midtable

# Low-scoring teams draw more
both_low_scoring = 1 if (home_avg_goals < 1.2 and away_avg_goals < 1.2) else 0

# Derby/rivalry matches (more defensive)
is_derby = 1 if teams_are_nearby_geographically else 0
```

---

## Strategy 3: Threshold Adjustment (No Retraining)

Instead of `argmax`, use custom thresholds:

```python
def predict_with_thresholds(probabilities, away_th=0.40, draw_th=0.30, home_th=0.40):
    """
    Predict with lower threshold for draws.

    Logic: If draw prob > draw_th AND it's the highest, predict draw
    """
    predictions = []
    for probs in probabilities:
        away_prob, draw_prob, home_prob = probs

        # Lower bar for draw prediction
        if draw_prob >= draw_th and draw_prob == max(probs):
            predictions.append(1)  # Draw
        elif away_prob > home_prob:
            predictions.append(0)  # Away
        else:
            predictions.append(2)  # Home

    return np.array(predictions)
```

Test different thresholds:
- draw_th = 0.25: More aggressive draw prediction
- draw_th = 0.20: Very aggressive
- draw_th = 0.30: Conservative

**Benefit**: No retraining needed, quick to test

---

## Strategy 4: Two-Stage Classifier

Train two models:
1. **Stage 1**: Binary classifier (Draw vs Not-Draw)
2. **Stage 2**: Binary classifier (Home vs Away) - only if Stage 1 predicts Not-Draw

```python
# Stage 1: Train model for draw detection
y_train_binary = (y_train == 1).astype(int)  # 1 if draw, 0 otherwise
draw_detector = cb.CatBoostClassifier(...)
draw_detector.fit(X_train, y_train_binary)

# Stage 2: Train model for home/away (only on non-draws)
non_draw_mask = y_train != 1
X_train_filtered = X_train[non_draw_mask]
y_train_filtered = (y_train[non_draw_mask] == 2).astype(int)  # 1 if home, 0 if away
home_away_classifier = cb.CatBoostClassifier(...)
home_away_classifier.fit(X_train_filtered, y_train_filtered)

# Prediction
draw_probs = draw_detector.predict_proba(X_test)[:, 1]
predictions = []
for i, draw_prob in enumerate(draw_probs):
    if draw_prob > 0.35:  # Threshold for draw
        predictions.append(1)  # Draw
    else:
        home_prob = home_away_classifier.predict_proba(X_test[i:i+1])[0, 1]
        predictions.append(2 if home_prob > 0.5 else 0)
```

---

## Strategy 5: Analyze Feature Importance for Draws

Find which features best separate draws from non-draws:

```python
import shap

# Train model
model = cb.CatBoostClassifier(...)
model.fit(X_train, y_train)

# Get SHAP values for draw predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Analyze features that push predictions toward draw
draw_shap_values = shap_values[:, :, 1]  # SHAP values for class 1 (draw)

# Find most important features for draws
draw_feature_importance = np.abs(draw_shap_values).mean(axis=0)
top_draw_features = sorted(zip(feature_names, draw_feature_importance),
                           key=lambda x: x[1], reverse=True)[:20]

print("Top features influencing draw predictions:")
for feat, imp in top_draw_features:
    print(f"  {feat}: {imp:.4f}")
```

This tells you which existing features help predict draws - you can engineer more features like those.

---

## Strategy 6: SMOTE for Class Imbalance

Use SMOTE to oversample draws in training data:

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(sampling_strategy={0: len(y_train[y_train==0]),
                                  1: int(len(y_train[y_train==1]) * 1.5),  # Increase draws by 50%
                                  2: len(y_train[y_train==2])},
              random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train on resampled data
model.fit(X_train_resampled, y_train_resampled)
```

---

## Strategy 7: Ensemble with Draw Specialist

Create an ensemble where one model specializes in draw detection:

```python
# Model 1: Optimized for overall performance
model_general = cb.CatBoostClassifier(class_weights=[1.2, 1.5, 1.0])
model_general.fit(X_train, y_train)

# Model 2: Optimized specifically for draws
model_draw_specialist = cb.CatBoostClassifier(class_weights=[1.0, 5.0, 1.0])
model_draw_specialist.fit(X_train, y_train)

# Ensemble predictions
probs_general = model_general.predict_proba(X_test)
probs_draw = model_draw_specialist.predict_proba(X_test)

# Weighted average (give more weight to draw specialist for draw class)
final_probs = probs_general.copy()
final_probs[:, 1] = 0.4 * probs_general[:, 1] + 0.6 * probs_draw[:, 1]  # Emphasize draw specialist

# Renormalize
final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
```

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Run `train_high_draw_weights.py` to test draw weights of 2.0x, 2.5x, 3.0x
2. ✅ Test threshold adjustment with current model (no retraining)

### Phase 2: Feature Engineering (2-4 hours)
3. Add team strength parity features (Elo diff, form diff, position diff)
4. Add historical draw tendency features
5. Retrain with new features

### Phase 3: Advanced Methods (4-8 hours)
6. Implement two-stage classifier
7. Try SMOTE resampling
8. Build draw specialist ensemble

### Expected Improvements
- **Phase 1**: Draw accuracy 22% → 30-35% (modest log loss increase)
- **Phase 2**: Draw accuracy 30% → 40-45% (better feature set)
- **Phase 3**: Draw accuracy 40% → 50-55% (specialized approach)

---

## Trade-offs to Consider

| Approach | Draw Accuracy Gain | Log Loss Impact | Complexity |
|----------|-------------------|-----------------|------------|
| Higher weights | +5-15% | +0.01-0.03 | Low |
| Draw features | +10-20% | +0.00-0.02 | Medium |
| Threshold tuning | +5-10% | 0.00 | Very Low |
| Two-stage | +15-25% | +0.02-0.05 | Medium |
| SMOTE | +10-15% | +0.01-0.03 | Low |
| Ensemble | +20-30% | +0.01-0.02 | High |

**Best ROI**: Start with Phase 1 (higher weights + thresholds) - quick to test, low complexity, good gains.
