#!/usr/bin/env python3
"""
Three Binary Models Ensemble
============================

Architecture:
- Model 1: Home Classifier (Home vs Not-Home)
- Model 2: Away Classifier (Away vs Not-Away)
- Model 3: Draw Classifier (Draw vs Not-Draw)

Each model is trained independently as a binary classifier.
Final prediction: Compare probabilities from all 3 and pick the highest.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"
RANDOM_STATE = 42

META_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target'
]


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Sort by date for chronological split
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    # Create binary targets for each outcome
    df['target_home'] = (df['result'] == 'H').astype(int)  # 1 if Home, 0 otherwise
    df['target_away'] = (df['result'] == 'A').astype(int)  # 1 if Away, 0 otherwise
    df['target_draw'] = (df['result'] == 'D').astype(int)  # 1 if Draw, 0 otherwise

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in META_COLS and not c.startswith('target_')]

    # Split chronologically: 70% train, 15% val, 15% test
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Train: {len(train_df)} matches ({train_df['match_date'].min().date()} to {train_df['match_date'].max().date()})")
    print(f"Val:   {len(val_df)} matches")
    print(f"Test:  {len(test_df)} matches ({test_df['match_date'].min().date()} to {test_df['match_date'].max().date()})")

    return train_df, val_df, test_df, feature_cols


def train_binary_model(X_train, y_train, X_val, y_val, outcome_name):
    """Train a binary classifier for a specific outcome."""
    print(f"\nTraining {outcome_name} Classifier...")
    print(f"  Positive class rate: {y_train.mean()*100:.1f}%")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        auto_class_weights='Balanced',  # Handle class imbalance
        random_seed=RANDOM_STATE,
        verbose=False
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    # Evaluate on validation
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"  Validation AUC: {val_auc:.3f}")

    return model


def ensemble_predict(home_model, away_model, draw_model, X):
    """
    Get predictions from all 3 models and ensemble them.

    Each model outputs P(outcome) for its specific outcome.
    We normalize these to sum to 1 and pick the highest.
    """
    # Get probability of each outcome from its specialist model
    prob_home = home_model.predict_proba(X)[:, 1]  # P(Home)
    prob_away = away_model.predict_proba(X)[:, 1]  # P(Away)
    prob_draw = draw_model.predict_proba(X)[:, 1]  # P(Draw)

    # Stack into matrix
    probs = np.column_stack([prob_away, prob_draw, prob_home])  # Order: A, D, H (0, 1, 2)

    # Normalize so they sum to 1 (optional but makes interpretation easier)
    probs_normalized = probs / probs.sum(axis=1, keepdims=True)

    return probs, probs_normalized


def evaluate_predictions(probs_raw, probs_norm, y_true_result, threshold_mode='highest'):
    """
    Evaluate predictions.

    threshold_mode:
    - 'highest': Always predict the outcome with highest probability
    - 'selective': Only predict if confidence exceeds threshold
    """
    results = {
        'home': {'predictions': 0, 'correct': 0},
        'away': {'predictions': 0, 'correct': 0},
        'draw': {'predictions': 0, 'correct': 0}
    }

    # Map result to numeric: A=0, D=1, H=2
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y_numeric = np.array([result_map[r] for r in y_true_result])

    if threshold_mode == 'highest':
        # Always predict highest probability outcome
        predictions = probs_norm.argmax(axis=1)

        for i, pred in enumerate(predictions):
            true_label = y_numeric[i]
            outcome = ['away', 'draw', 'home'][pred]
            results[outcome]['predictions'] += 1
            if pred == true_label:
                results[outcome]['correct'] += 1

    return results, y_numeric


def main():
    # Load data
    train_df, val_df, test_df, feature_cols = load_data()

    # Prepare features
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Prepare binary targets for each outcome
    y_train_home = train_df['target_home'].values
    y_train_away = train_df['target_away'].values
    y_train_draw = train_df['target_draw'].values

    y_val_home = val_df['target_home'].values
    y_val_away = val_df['target_away'].values
    y_val_draw = val_df['target_draw'].values

    print("\n" + "="*80)
    print("TRAINING 3 BINARY MODELS")
    print("="*80)

    # Train 3 separate binary models
    home_model = train_binary_model(X_train, y_train_home, X_val, y_val_home, "HOME")
    away_model = train_binary_model(X_train, y_train_away, X_val, y_val_away, "AWAY")
    draw_model = train_binary_model(X_train, y_train_draw, X_val, y_val_draw, "DRAW")

    # Ensemble predictions on test set
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTIONS ON TEST SET")
    print("="*80)

    probs_raw, probs_norm = ensemble_predict(home_model, away_model, draw_model, X_test)

    # Show example predictions
    print("\nExample predictions (first 10 matches):")
    print("-" * 80)
    print(f"{'Match':<6} {'P(Home)':<10} {'P(Away)':<10} {'P(Draw)':<10} {'Predicted':<10} {'Actual':<10} {'Correct'}")
    print("-" * 80)

    result_map = {'A': 0, 'D': 1, 'H': 2}
    outcome_names = ['Away', 'Draw', 'Home']

    for i in range(10):
        p_home = probs_norm[i, 2]
        p_away = probs_norm[i, 0]
        p_draw = probs_norm[i, 1]
        pred_idx = probs_norm[i].argmax()
        pred_name = outcome_names[pred_idx]
        actual = test_df.iloc[i]['result']
        correct = '✓' if pred_idx == result_map[actual] else '✗'

        print(f"{i+1:<6} {p_home:<10.1%} {p_away:<10.1%} {p_draw:<10.1%} {pred_name:<10} {actual:<10} {correct}")

    # Evaluate all predictions
    print("\n" + "="*80)
    print("RESULTS: PREDICT ALL MATCHES (Highest Probability)")
    print("="*80)

    results, y_numeric = evaluate_predictions(probs_raw, probs_norm, test_df['result'].values)

    total_correct = sum(r['correct'] for r in results.values())
    total_preds = sum(r['predictions'] for r in results.values())

    print(f"\nOverall: {total_preds} matches, {total_correct} correct, WR: {total_correct/total_preds*100:.1f}%")
    print(f"\nPer-outcome breakdown:")
    for outcome in ['home', 'away', 'draw']:
        preds = results[outcome]['predictions']
        correct = results[outcome]['correct']
        wr = (correct / preds * 100) if preds > 0 else 0
        print(f"  {outcome.upper()}: {preds} predictions, {correct} correct, WR: {wr:.1f}%")

    # Selective predictions with thresholds
    print("\n" + "="*80)
    print("RESULTS: SELECTIVE PREDICTIONS (Confidence Thresholds)")
    print("="*80)

    threshold_configs = [
        {'home': 0.40, 'away': 0.40, 'draw': 0.40, 'name': 'Low (40% all)'},
        {'home': 0.45, 'away': 0.45, 'draw': 0.45, 'name': 'Medium (45% all)'},
        {'home': 0.50, 'away': 0.50, 'draw': 0.50, 'name': 'High (50% all)'},
        {'home': 0.55, 'away': 0.50, 'draw': 0.45, 'name': 'Outcome-specific'},
        {'home': 0.60, 'away': 0.55, 'draw': 0.45, 'name': 'Aggressive'},
    ]

    for config in threshold_configs:
        home_thresh = config['home']
        away_thresh = config['away']
        draw_thresh = config['draw']

        selective_results = {
            'home': {'predictions': 0, 'correct': 0},
            'away': {'predictions': 0, 'correct': 0},
            'draw': {'predictions': 0, 'correct': 0}
        }

        for i in range(len(probs_norm)):
            p_home = probs_norm[i, 2]
            p_away = probs_norm[i, 0]
            p_draw = probs_norm[i, 1]
            true_label = result_map[test_df.iloc[i]['result']]

            # Find candidates that meet threshold
            candidates = []
            if p_home >= home_thresh:
                candidates.append(('home', p_home, 2))
            if p_away >= away_thresh:
                candidates.append(('away', p_away, 0))
            if p_draw >= draw_thresh:
                candidates.append(('draw', p_draw, 1))

            if candidates:
                # Pick highest confidence
                best = max(candidates, key=lambda x: x[1])
                outcome, prob, label = best
                selective_results[outcome]['predictions'] += 1
                if label == true_label:
                    selective_results[outcome]['correct'] += 1

        total_preds = sum(r['predictions'] for r in selective_results.values())
        total_correct = sum(r['correct'] for r in selective_results.values())
        overall_wr = (total_correct / total_preds * 100) if total_preds > 0 else 0

        print(f"\n{config['name']}:")
        print(f"  Total: {total_preds} bets, WR: {overall_wr:.1f}%")
        for outcome in ['home', 'away', 'draw']:
            preds = selective_results[outcome]['predictions']
            correct = selective_results[outcome]['correct']
            wr = (correct / preds * 100) if preds > 0 else 0
            print(f"  {outcome.upper()}: {preds} bets @ {wr:.1f}% WR")

    # Base rates for comparison
    print("\n" + "="*80)
    print("BASE RATES (for comparison)")
    print("="*80)
    base_home = (test_df['result'] == 'H').mean() * 100
    base_away = (test_df['result'] == 'A').mean() * 100
    base_draw = (test_df['result'] == 'D').mean() * 100
    print(f"  Home: {base_home:.1f}%")
    print(f"  Away: {base_away:.1f}%")
    print(f"  Draw: {base_draw:.1f}%")


if __name__ == '__main__':
    main()
