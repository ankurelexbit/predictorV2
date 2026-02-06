#!/usr/bin/env python3
"""
Train Production Model v3.0.0
=============================

Multiclass CatBoost + LightGBM ensemble with balanced class weights.
Saves model for production use.

Recommended thresholds for 14-20 bets/week:
- Home >= 55%
- Away >= 50%
- Draw >= 40%
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"
MODEL_DIR = Path(__file__).parent.parent / "models" / "production"
RANDOM_STATE = 42

META_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target'
]


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)
    df['target'] = df['result'].map({'A': 0, 'D': 1, 'H': 2})

    feature_cols = [c for c in df.columns if c not in META_COLS]

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Train period: {train_df['match_date'].min().date()} to {train_df['match_date'].max().date()}")
    print(f"Test period: {test_df['match_date'].min().date()} to {test_df['match_date'].max().date()}")

    return train_df, val_df, test_df, feature_cols


def train_ensemble(X_train, y_train, X_val, y_val):
    """Train CatBoost + LightGBM ensemble."""
    print("\nTraining CatBoost...")
    catboost = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        auto_class_weights='Balanced',
        random_seed=RANDOM_STATE,
        verbose=False
    )
    catboost.fit(X_train, y_train, eval_set=(X_val, y_val),
                 early_stopping_rounds=50, verbose=False)

    print("Training LightGBM...")
    lightgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        reg_lambda=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lightgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    return catboost, lightgbm


def evaluate_model(catboost, lightgbm, X_test, y_test, y_result):
    """Evaluate ensemble on test set."""
    probs_cat = catboost.predict_proba(X_test)
    probs_lgb = lightgbm.predict_proba(X_test)
    probs_ensemble = (probs_cat + probs_lgb) / 2

    preds = probs_ensemble.argmax(axis=1)

    # Overall metrics
    accuracy = accuracy_score(y_test, preds)
    logloss = log_loss(y_test, probs_ensemble)

    # Per-outcome metrics
    result_map = {'H': 2, 'A': 0, 'D': 1}
    y_numeric = np.array([result_map[r] for r in y_result])

    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'predictions': {
            'home': int((preds == 2).sum()),
            'away': int((preds == 0).sum()),
            'draw': int((preds == 1).sum())
        }
    }

    # Win rate per outcome
    for label, name in [(2, 'home'), (0, 'away'), (1, 'draw')]:
        mask = preds == label
        if mask.sum() > 0:
            correct = (y_test[mask] == label).sum()
            metrics[f'{name}_win_rate'] = float(correct / mask.sum())
        else:
            metrics[f'{name}_win_rate'] = 0.0

    return metrics, probs_ensemble


def evaluate_with_thresholds(probs, y_result, thresholds):
    """Evaluate with confidence thresholds."""
    result_map = {'H': 2, 'A': 0, 'D': 1}
    y_numeric = np.array([result_map[r] for r in y_result])

    results = {
        'home': {'preds': 0, 'correct': 0},
        'away': {'preds': 0, 'correct': 0},
        'draw': {'preds': 0, 'correct': 0}
    }

    for i in range(len(probs)):
        p_away, p_draw, p_home = probs[i]
        true_label = y_numeric[i]

        candidates = []
        if p_home >= thresholds['home']:
            candidates.append(('home', p_home, 2))
        if p_away >= thresholds['away']:
            candidates.append(('away', p_away, 0))
        if p_draw >= thresholds['draw']:
            candidates.append(('draw', p_draw, 1))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            outcome, prob, label = best
            results[outcome]['preds'] += 1
            if label == true_label:
                results[outcome]['correct'] += 1

    return results


def save_model(catboost, lightgbm, feature_cols, metrics, version="3.0.0"):
    """Save ensemble model and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save as a dict containing both models
    model_data = {
        'catboost': catboost,
        'lightgbm': lightgbm,
        'feature_cols': feature_cols,
        'model_type': 'ensemble'
    }

    model_path = MODEL_DIR / f"model_v{version}.joblib"
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save metadata
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'CatBoost+LightGBM Ensemble',
        'features': len(feature_cols),
        'class_weights': 'balanced',
        'recommended_thresholds': {
            'home': 0.55,
            'away': 0.50,
            'draw': 0.40
        },
        'expected_bets_per_week': '14-20',
        'test_metrics': metrics
    }

    metadata_path = MODEL_DIR / f"model_v{version}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # Update LATEST
    latest_path = MODEL_DIR / "LATEST"
    with open(latest_path, 'w') as f:
        f.write(f"model_v{version}.joblib")

    return model_path


def main():
    # Load data
    train_df, val_df, test_df, feature_cols = load_data()

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    y_result = test_df['result'].values

    # Train ensemble
    catboost, lightgbm = train_ensemble(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION (All Predictions)")
    print("="*80)

    metrics, probs = evaluate_model(catboost, lightgbm, X_test, y_test, y_result)

    print(f"Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"\nPrediction distribution:")
    print(f"  Home: {metrics['predictions']['home']} ({metrics['predictions']['home']/len(y_test)*100:.1f}%)")
    print(f"  Away: {metrics['predictions']['away']} ({metrics['predictions']['away']/len(y_test)*100:.1f}%)")
    print(f"  Draw: {metrics['predictions']['draw']} ({metrics['predictions']['draw']/len(y_test)*100:.1f}%)")
    print(f"\nWin rates:")
    print(f"  Home: {metrics['home_win_rate']*100:.1f}%")
    print(f"  Away: {metrics['away_win_rate']*100:.1f}%")
    print(f"  Draw: {metrics['draw_win_rate']*100:.1f}%")

    # Evaluate with recommended thresholds
    print("\n" + "="*80)
    print("EVALUATION (With Recommended Thresholds: H>=55%, A>=50%, D>=40%)")
    print("="*80)

    test_weeks = (test_df['match_date'].max() - test_df['match_date'].min()).days / 7
    thresholds = {'home': 0.55, 'away': 0.50, 'draw': 0.40}
    results = evaluate_with_thresholds(probs, y_result, thresholds)

    total_preds = sum(r['preds'] for r in results.values())
    total_correct = sum(r['correct'] for r in results.values())
    overall_wr = total_correct / total_preds * 100 if total_preds > 0 else 0

    print(f"Total bets: {total_preds}")
    print(f"Bets per week: {total_preds/test_weeks:.1f}")
    print(f"Overall Win Rate: {overall_wr:.1f}%")
    print()

    for outcome in ['home', 'away', 'draw']:
        p = results[outcome]['preds']
        c = results[outcome]['correct']
        wr = c/p*100 if p > 0 else 0
        print(f"  {outcome.upper()}: {p} bets, {c} correct, WR: {wr:.1f}%")

    # Add threshold results to metrics
    metrics['threshold_results'] = {
        'thresholds': thresholds,
        'total_bets': total_preds,
        'bets_per_week': round(total_preds/test_weeks, 1),
        'overall_win_rate': round(overall_wr, 1),
        'home_bets': results['home']['preds'],
        'home_win_rate': round(results['home']['correct']/results['home']['preds']*100 if results['home']['preds'] else 0, 1),
        'away_bets': results['away']['preds'],
        'away_win_rate': round(results['away']['correct']/results['away']['preds']*100 if results['away']['preds'] else 0, 1),
        'draw_bets': results['draw']['preds'],
        'draw_win_rate': round(results['draw']['correct']/results['draw']['preds']*100 if results['draw']['preds'] else 0, 1)
    }

    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    save_model(catboost, lightgbm, feature_cols, metrics, version="3.0.0")

    print("\n" + "="*80)
    print("USAGE")
    print("="*80)
    print("""
To load and use this model:

```python
import joblib
import numpy as np

# Load model
model_data = joblib.load('models/production/model_v3.0.0.joblib')
catboost = model_data['catboost']
lightgbm = model_data['lightgbm']
feature_cols = model_data['feature_cols']

# Get predictions for a match
X = match_features[feature_cols].values  # Shape: (1, 162)
probs_cat = catboost.predict_proba(X)
probs_lgb = lightgbm.predict_proba(X)
probs = (probs_cat + probs_lgb) / 2  # [P(Away), P(Draw), P(Home)]

# Apply thresholds
p_away, p_draw, p_home = probs[0]
if p_home >= 0.55:
    prediction = 'Home'
elif p_away >= 0.50:
    prediction = 'Away'
elif p_draw >= 0.40:
    prediction = 'Draw'
else:
    prediction = 'No Bet'
```
""")


if __name__ == '__main__':
    main()
