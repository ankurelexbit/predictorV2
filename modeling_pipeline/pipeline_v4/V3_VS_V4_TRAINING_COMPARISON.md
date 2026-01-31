# V3 vs V4 Training Script Comparison

## Overview

**V3** (`train_xgboost_roi_optimized.py`): ROI-optimized betting model
**V4** (`train_model.py`): Log loss optimized draw-focused model

## Major Philosophical Differences

| Aspect | V3 Pipeline | V4 Pipeline |
|--------|-------------|-------------|
| **Primary Goal** | Maximize betting ROI (profit) | Minimize log loss (probability quality) |
| **Optimization Target** | ROI % with confidence thresholds | Log loss + draw prediction balance |
| **Class Weights** | Tests 5 configurations (1.0x to 2.5x draw) | No class weights (relies on hyperparameters) |
| **Model Selection** | Best ROI configuration | Best draw count + log loss |
| **Betting Strategy** | Built-in with confidence thresholds | None (use EV externally) |
| **Calibration** | None | Isotonic calibration on validation set |

## Technical Differences

### 1. Data Splitting

**V3** - Fixed date splits:
```python
train_mask = df['match_date'] < '2024-01-01'
val_mask = (df['match_date'] >= '2024-01-01') & (df['match_date'] < '2025-01-01')
test_mask = df['match_date'] >= '2025-01-01'
```

**V4** - Percentage-based chronological:
```python
train_end = int(n * 0.70)
val_end = int(n * 0.85)
# 70% train / 15% val / 15% test
```

### 2. Model Training Approach

**V3**:
- Tests 5 different class weight configurations
- Fixed hyperparameters for all runs
- Selects best based on ROI
- Manual XGBoost training with DMatrix

**V4**:
- Optional hyperparameter tuning (20 trials)
- Random search over conservative parameter grid
- Selects best based on draw predictions + log loss
- Uses XGBoostFootballModel wrapper

### 3. Class Weights

**V3** - Tests multiple configurations:
```python
configs = [
    {"name": "Baseline", "draw_mult": 1.0, "away_mult": 1.0},
    {"name": "Slight Draw Boost", "draw_mult": 1.3, "away_mult": 1.2},
    {"name": "Current Default", "draw_mult": 1.5, "away_mult": 1.3},
    {"name": "Moderate Draw", "draw_mult": 2.0, "away_mult": 1.5},
    {"name": "High Draw", "draw_mult": 2.5, "away_mult": 1.5},
]
```

**V4** - No class weights, conservative hyperparameters instead:
```python
'max_depth': [3, 4, 5],            # Lower depth
'min_child_weight': [10, 15, 20],  # High weight = conservative
'gamma': [1.0, 2.0, 3.0],          # Regularization
```

### 4. Hyperparameters

**V3** - Fixed for all runs:
```python
params = {
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.9,
    'colsample_bytree': 0.6,
    'min_child_weight': 3,
    'gamma': 0.5,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'n_estimators': 500,
}
```

**V4** - Default parameters (if no tuning):
```python
params = {
    'max_depth': 4,              # More conservative
    'learning_rate': 0.03,
    'subsample': 0.7,            # Lower
    'colsample_bytree': 0.7,     # Higher
    'min_child_weight': 10,      # Much higher (more conservative)
    'gamma': 1.0,                # Higher regularization
    'n_estimators': 1000,        # More trees
}
```

### 5. Evaluation Metrics

**V3**:
- Log loss
- ROI % at different confidence thresholds (0.35-0.60)
- Profit in dollars
- Win rate
- Bets breakdown by outcome

**V4**:
- Log loss (primary)
- Accuracy
- Draws predicted vs actual
- Confusion matrix
- Feature importance

### 6. ROI Calculation

**V3** - Built-in ROI evaluation:
```python
def calculate_roi(y_true, y_pred_proba, confidence_threshold=0.45):
    """
    Calculates profit using typical odds:
    - Away: 3.0
    - Draw: 3.5
    - Home: 2.0
    """
    # Only bet when confidence > threshold
    # Return ROI%, profit, win rate
```

**V4** - None (assumes external EV calculation)

### 7. Model Selection

**V3**:
```python
best_config = max(results, key=lambda x: x['roi'])
# Selects configuration with highest ROI
```

**V4**:
```python
if draw_count > best_draw_count or (draw_count >= best_draw_count and score < best_score):
    # Prioritizes draw prediction, then log loss
```

### 8. Calibration

**V3**: No calibration

**V4**: Isotonic calibration on validation set
```python
final_model.calibrate(X_val, y_val, method='isotonic')
```

## When to Use Each

### Use V3 (`train_xgboost_roi_optimized.py`) when:
- Primary goal is betting profit (ROI)
- You have typical odds assumptions
- You want to test different class weight strategies
- You want built-in confidence threshold optimization
- You're working with V3 feature set

### Use V4 (`train_model.py`) when:
- Primary goal is probability quality (log loss)
- You want calibrated probabilities
- You prefer hyperparameter tuning over class weights
- You'll calculate EV externally with real odds
- You're working with V4 3-pillar feature framework (150+ features)

## Combining Best of Both

For your current situation with V4 pipeline, you could:

1. **Use V4 features + V3 ROI approach**:
   - Load V4 training_data.csv (150+ features)
   - Apply V3's class weight testing
   - Evaluate with ROI metrics

2. **Use V3 ROI evaluation on V4 models**:
   - Train models with V4's approach
   - Add V3's ROI calculation function
   - Compare ROI across your final models

3. **Hybrid approach**:
   - V4's hyperparameter tuning
   - V3's class weights
   - Both log loss and ROI metrics
   - This is essentially what you did with `train_final_models.py`

## Current State

You've actually already evolved beyond both scripts:
- `train_final_models.py` uses CatBoost (better than XGBoost)
- Tests both no weights and conservative weights
- 100 Optuna trials (more thorough than V4's 20)
- Evaluates log loss AND draw accuracy (hybrid approach)

Your final models achieve:
- **0.9851 log loss** (better than both V3 and V4)
- **22.05% draw accuracy** with conservative weights (balanced)

The main thing you're missing from V3 is the ROI backtesting with typical odds, which could be added as a post-processing evaluation step.
