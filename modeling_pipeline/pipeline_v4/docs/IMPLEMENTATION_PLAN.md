# Implementation Plan: Advanced Modeling Approaches

## Overview

Four approaches to explore for achieving the constraints (WR>50%, ROI>10%, H/D/A+):

1. **Option 1+2**: Market data at 1hr before + Lineup features
2. **Hyperparameter Tuning**: Proper optimization of model parameters
3. **Goal Difference Regression**: Predict goal difference, derive H/D/A
4. **Ensemble of Specialists**: Separate models per outcome

---

## Approach 1: Market Data at 1hr + Lineup Features

### What We Have
- Odds with `created_at` and `latest_bookmaker_update` timestamps
- Lineup data: starters, substitutes, formations, sidelined players
- Player details including ratings (type_id=118)

### Implementation Steps

**Step 1: Extract "1-hour-before" market features**
```python
def extract_1hr_market_features(fixture):
    """Only use odds updated MORE than 1 hour before kickoff"""
    kickoff = datetime.strptime(fixture['starting_at'], '%Y-%m-%d %H:%M:%S')
    one_hour_before = kickoff - timedelta(hours=1)

    # Filter odds to those available at 1hr mark
    valid_odds = []
    for odd in fixture.get('odds', []):
        updated = odd.get('latest_bookmaker_update')
        if updated:
            updated_dt = datetime.strptime(updated[:19], '%Y-%m-%d %H:%M:%S')
            if updated_dt <= one_hour_before:
                valid_odds.append(odd)

    # Calculate features from valid_odds only
    return calculate_market_features(valid_odds)
```

**Step 2: Extract lineup features**
```python
def extract_lineup_features(fixture, team_id, historical_lineups):
    """
    Features based on announced lineup vs historical patterns

    Features to create:
    - num_regular_starters: How many of usual XI are starting
    - key_players_missing: Are top players (by rating) missing
    - formation_change: Different from usual formation
    - rotation_score: How different is this lineup from average
    - goalkeeper_change: Is regular GK playing
    - avg_starter_rating: Average rating of starting XI
    """
    lineups = fixture.get('lineups', [])
    starters = [l for l in lineups if l.get('type_id') == 11 and l.get('team_id') == team_id]

    # Get player IDs
    starter_ids = set(l.get('player_id') for l in starters)

    # Compare to historical most common starters
    usual_starters = get_usual_starters(team_id, historical_lineups)
    overlap = len(starter_ids & usual_starters)

    # Get ratings
    ratings = []
    for s in starters:
        for detail in s.get('details', []):
            if detail.get('type_id') == 118:  # Rating
                ratings.append(detail.get('data', {}).get('value', 0))

    return {
        'num_regular_starters': overlap,
        'rotation_score': 11 - overlap,
        'avg_starter_rating': np.mean(ratings) if ratings else 0,
        'formation': fixture.get('formations', [{}])[0].get('formation', '')
    }
```

**Step 3: Create sidelined/injury features**
```python
def extract_injury_features(fixture, team_id, player_importance):
    """
    Features based on who is injured/suspended

    Features:
    - num_injured: Total players out
    - key_players_injured: Weighted by player importance
    - injury_impact_score: Sum of importance of missing players
    """
    sidelined = fixture.get('sidelined', [])
    team_sidelined = [s for s in sidelined if s.get('participant_id') == team_id]

    # Calculate impact
    impact = 0
    for s in team_sidelined:
        player_id = s.get('player_id')
        importance = player_importance.get(player_id, 0.5)  # Default medium importance
        impact += importance

    return {
        'num_injured': len(team_sidelined),
        'injury_impact_score': impact
    }
```

---

## Approach 2: Hyperparameter Tuning

### Current Parameters (not tuned)
```python
CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=3,
    # ... defaults for everything else
)
```

### Proper Tuning Approach
```python
from sklearn.model_selection import TimeSeriesSplit
import optuna

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
    }

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        model = CatBoostClassifier(**params, verbose=False, random_seed=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Score based on ROI, not just accuracy
        probs = model.predict_proba(X.iloc[val_idx])
        roi = calculate_backtest_roi(probs, y.iloc[val_idx], odds.iloc[val_idx])
        scores.append(roi)

    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Class Weights for Draw
```python
# Option: Weight draws higher since they're hardest to predict
class_weights = {0: 1.0, 1: 1.5, 2: 1.0}  # Higher weight for draws

model = CatBoostClassifier(
    class_weights=class_weights,
    # ... other params
)
```

---

## Approach 3: Goal Difference Regression

### Concept
Instead of classifying H/D/A, predict the continuous goal difference:
- Target: `home_goals - away_goals` (e.g., -2, -1, 0, 1, 2, 3)
- Derive probabilities from the predicted distribution

### Implementation
```python
from catboost import CatBoostRegressor
from scipy.stats import norm

class GoalDifferenceModel:
    def __init__(self):
        self.model = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function='RMSE',
            random_seed=42
        )
        self.residual_std = None  # Learned from validation

    def fit(self, X_train, y_train, X_val, y_val):
        """
        y_train/y_val = home_goals - away_goals
        """
        self.model.fit(X_train, y_train)

        # Learn residual distribution
        val_preds = self.model.predict(X_val)
        residuals = y_val - val_preds
        self.residual_std = np.std(residuals)

    def predict_proba(self, X):
        """
        Returns P(Home), P(Draw), P(Away) derived from goal difference distribution
        """
        # Predicted mean goal difference
        pred_mean = self.model.predict(X)

        probs = []
        for mean in pred_mean:
            # Assume normal distribution around predicted mean
            # P(Home) = P(goal_diff > 0.5)  # Win by 1+
            # P(Draw) = P(-0.5 < goal_diff < 0.5)  # Goal diff = 0
            # P(Away) = P(goal_diff < -0.5)  # Lose by 1+

            p_home = 1 - norm.cdf(0.5, loc=mean, scale=self.residual_std)
            p_away = norm.cdf(-0.5, loc=mean, scale=self.residual_std)
            p_draw = 1 - p_home - p_away

            probs.append([p_away, p_draw, p_home])

        return np.array(probs)

    def predict_asian_handicap(self, X, line):
        """
        Predict Asian Handicap outcome
        e.g., line = -1.5 means home needs to win by 2+
        """
        pred_mean = self.model.predict(X)

        # P(Home covers -1.5) = P(goal_diff > 1.5)
        p_cover = 1 - norm.cdf(line + 0.5, loc=pred_mean, scale=self.residual_std)
        return p_cover
```

### Advantages
- Natural uncertainty quantification
- Same model predicts 1X2, Asian Handicap, and can inform Over/Under
- Draw probability emerges naturally (not forced third class)
- Richer training signal (3-0 and 1-0 both "Home" but model sees difference)

---

## Approach 4: Ensemble of Outcome-Specific Models

### Concept
Train 3 separate binary classifiers, each optimized for one outcome:
- Home Specialist: P(Home) vs P(Not Home)
- Draw Specialist: P(Draw) vs P(Not Draw)
- Away Specialist: P(Away) vs P(Not Away)

### Implementation
```python
class EnsembleSpecialists:
    def __init__(self):
        # Different architectures can be optimal for each outcome
        self.home_model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.03,
            scale_pos_weight=1.3,  # Home is common
        )
        self.draw_model = CatBoostClassifier(
            iterations=800, depth=4, learning_rate=0.02,  # Deeper search for draws
            scale_pos_weight=3.0,  # Draw is rare (25%)
        )
        self.away_model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.03,
            scale_pos_weight=2.0,  # Away less common
        )

        # Can use different feature sets per model
        self.home_features = HOME_FOCUSED_FEATURES
        self.draw_features = DRAW_FOCUSED_FEATURES  # Include combined_draw_tendency, etc.
        self.away_features = AWAY_FOCUSED_FEATURES

    def fit(self, X, y):
        # Binary targets
        y_home = (y == 2).astype(int)
        y_draw = (y == 1).astype(int)
        y_away = (y == 0).astype(int)

        self.home_model.fit(X[self.home_features], y_home)
        self.draw_model.fit(X[self.draw_features], y_draw)
        self.away_model.fit(X[self.away_features], y_away)

    def predict_proba(self, X):
        p_home = self.home_model.predict_proba(X[self.home_features])[:, 1]
        p_draw = self.draw_model.predict_proba(X[self.draw_features])[:, 1]
        p_away = self.away_model.predict_proba(X[self.away_features])[:, 1]

        # Normalize to sum to 1
        total = p_home + p_draw + p_away
        probs = np.column_stack([
            p_away / total,
            p_draw / total,
            p_home / total
        ])

        return probs
```

### Draw-Specific Features
```python
DRAW_FOCUSED_FEATURES = [
    # Standard features
    'elo_diff', 'position_diff', ...

    # Draw-specific
    'combined_draw_tendency',
    'home_draw_rate_10',
    'away_draw_rate_10',
    'h2h_draw_rate',
    'league_draw_rate',
    'both_midtable',  # Midtable teams draw more
    'both_defensive',  # Defensive teams draw more
    'both_low_scoring',
    'either_coming_from_draw',

    # New features to add
    'home_xg_minus_away_xg_abs',  # Close xG = more draws
    'home_league_pos_minus_away_abs',  # Close positions = more draws
    'elo_diff_abs',  # Small diff = more draws
]
```

---

## Recommended Execution Order

1. **Hyperparameter Tuning** (2-3 hours)
   - Quick win, improves all approaches
   - Use Optuna with TimeSeriesSplit

2. **Goal Difference Regression** (1-2 hours)
   - Fundamentally different approach
   - May have better calibration for draws

3. **Ensemble of Specialists** (2-3 hours)
   - Can tune each model independently
   - Draw specialist can use different features

4. **Option 1+2: Market + Lineup** (3-4 hours)
   - Requires extracting new features from raw data
   - Most realistic for production use

---

## Expected Outcomes

| Approach | Complexity | Expected ROI Improvement |
|----------|------------|-------------------------|
| Hyperparameter Tuning | Low | +2-5% |
| Goal Difference Regression | Medium | +3-8% (better draws) |
| Ensemble Specialists | Medium | +3-7% |
| Market + Lineup (1hr) | High | +5-15% (if done right) |

**Combined**: Potentially 10-20% ROI with proper implementation

---

## Current vs Proposed

| Metric | Current (Match Only) | Target |
|--------|---------------------|--------|
| WR | 50.7% | >55% |
| ROI | -6.1% | >10% |
| Draw Accuracy | ~25% | >35% |
