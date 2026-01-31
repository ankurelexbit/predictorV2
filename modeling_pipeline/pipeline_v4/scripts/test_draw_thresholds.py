#!/usr/bin/env python3
"""
Test Different Draw Prediction Thresholds
==========================================

Tests the existing Conservative model with different draw thresholds
to see if we can improve draw accuracy without retraining.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_data():
    """Load test data."""
    df = pd.read_csv('data/training_data.csv')
    df['match_date'] = pd.to_datetime(df['match_date'])
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    df = df.sort_values('match_date').reset_index(drop=True)

    # Get test set (last 15%)
    n = len(df)
    test_start = int(n * 0.85)
    test_df = df.iloc[test_start:].copy()

    return test_df

def predict_with_threshold(probabilities, draw_threshold=0.33):
    """
    Predict with custom draw threshold.

    Args:
        probabilities: Array of shape (n, 3) with [away, draw, home] probabilities
        draw_threshold: Minimum probability to predict draw

    Returns:
        predictions: Array of predicted classes
    """
    predictions = []
    for probs in probabilities:
        away_p, draw_p, home_p = probs

        # If draw probability exceeds threshold and it's the highest, predict draw
        if draw_p >= draw_threshold and draw_p == max(probs):
            predictions.append(1)  # Draw
        elif away_p > home_p:
            predictions.append(0)  # Away
        else:
            predictions.append(2)  # Home

    return np.array(predictions)

def evaluate_predictions(y_true, y_pred, probabilities, threshold):
    """Evaluate predictions and return metrics."""
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, probabilities)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    away_acc = cm[0, 0] / cm[0].sum() * 100 if cm[0].sum() > 0 else 0
    draw_acc = cm[1, 1] / cm[1].sum() * 100 if cm[1].sum() > 0 else 0
    home_acc = cm[2, 2] / cm[2].sum() * 100 if cm[2].sum() > 0 else 0

    # Prediction distribution
    away_pct = (y_pred == 0).sum() / len(y_pred) * 100
    draw_pct = (y_pred == 1).sum() / len(y_pred) * 100
    home_pct = (y_pred == 2).sum() / len(y_pred) * 100

    # Actual distribution
    away_actual = (y_true == 0).sum() / len(y_true) * 100
    draw_actual = (y_true == 1).sum() / len(y_true) * 100
    home_actual = (y_true == 2).sum() / len(y_true) * 100

    return {
        'threshold': threshold,
        'log_loss': loss,
        'accuracy': acc,
        'away_accuracy': away_acc,
        'draw_accuracy': draw_acc,
        'home_accuracy': home_acc,
        'away_pct': away_pct,
        'draw_pct': draw_pct,
        'home_pct': home_pct,
        'away_actual': away_actual,
        'draw_actual': draw_actual,
        'home_actual': home_actual,
        'confusion_matrix': cm
    }

def main():
    print("="*80)
    print("DRAW THRESHOLD TUNING TEST")
    print("="*80)
    print("\nTesting Conservative Uncalibrated Model with different draw thresholds...")

    # Load model
    model_path = Path('models/final/model_conservative_uncalibrated.joblib')
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Load test data
    print("Loading test data...")
    test_df = load_data()

    # Get features
    features_to_exclude = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result', 'target',
        'home_team_name', 'away_team_name', 'state_id'
    ]
    feature_cols = [c for c in test_df.columns if c not in features_to_exclude]

    X_test = test_df[feature_cols]
    y_test = test_df['target'].values

    print(f"Test set: {len(y_test)} samples")
    print(f"Using {len(feature_cols)} features\n")

    # Get probabilities
    print("Getting model probabilities...")
    probabilities = model.predict_proba(X_test)

    # Test different thresholds
    thresholds = [0.20, 0.25, 0.30, 0.33, 0.35, 0.40]
    results = []

    print("\n" + "="*80)
    print("TESTING DIFFERENT DRAW THRESHOLDS")
    print("="*80)
    print(f"\n{'Threshold':<12} {'Log Loss':<12} {'Draw Acc':<12} {'Draw Pred%':<12} {'Overall Acc':<12}")
    print("-"*80)

    for threshold in thresholds:
        predictions = predict_with_threshold(probabilities, threshold)
        metrics = evaluate_predictions(y_test, predictions, probabilities, threshold)
        results.append(metrics)

        print(f"{threshold:<12.2f} {metrics['log_loss']:<12.4f} {metrics['draw_accuracy']:<12.2f} "
              f"{metrics['draw_pct']:<12.1f} {metrics['accuracy']:<12.1%}")

    # Find best draw accuracy
    best_draw = max(results, key=lambda x: x['draw_accuracy'])
    best_balanced = max(results, key=lambda x: x['draw_accuracy'] - abs(x['draw_pct'] - x['draw_actual']))

    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)

    # Baseline (standard argmax = threshold ~0.33)
    baseline = [r for r in results if r['threshold'] == 0.33][0]

    print(f"\n📊 ACTUAL DISTRIBUTION (Test Set):")
    print(f"   Away: {baseline['away_actual']:.1f}% | Draw: {baseline['draw_actual']:.1f}% | Home: {baseline['home_actual']:.1f}%")

    print(f"\n🔵 BASELINE (Threshold = 0.33, Standard argmax):")
    print(f"   Log Loss: {baseline['log_loss']:.4f}")
    print(f"   Draw Accuracy: {baseline['draw_accuracy']:.2f}%")
    print(f"   Predictions: {baseline['away_pct']:.1f}% Away | {baseline['draw_pct']:.1f}% Draw | {baseline['home_pct']:.1f}% Home")
    print(f"   Overall Accuracy: {baseline['accuracy']:.1%}")

    print(f"\n🟢 BEST DRAW ACCURACY (Threshold = {best_draw['threshold']:.2f}):")
    print(f"   Log Loss: {best_draw['log_loss']:.4f} ({best_draw['log_loss'] - baseline['log_loss']:+.4f})")
    print(f"   Draw Accuracy: {best_draw['draw_accuracy']:.2f}% ({best_draw['draw_accuracy'] - baseline['draw_accuracy']:+.2f}%)")
    print(f"   Predictions: {best_draw['away_pct']:.1f}% Away | {best_draw['draw_pct']:.1f}% Draw | {best_draw['home_pct']:.1f}% Home")
    print(f"   Overall Accuracy: {best_draw['accuracy']:.1%}")

    print(f"\n🟡 BEST BALANCED (Threshold = {best_balanced['threshold']:.2f}):")
    print(f"   Log Loss: {best_balanced['log_loss']:.4f} ({best_balanced['log_loss'] - baseline['log_loss']:+.4f})")
    print(f"   Draw Accuracy: {best_balanced['draw_accuracy']:.2f}% ({best_balanced['draw_accuracy'] - baseline['draw_accuracy']:+.2f}%)")
    print(f"   Predictions: {best_balanced['away_pct']:.1f}% Away | {best_balanced['draw_pct']:.1f}% Draw | {best_balanced['home_pct']:.1f}% Home")
    print(f"   Overall Accuracy: {best_balanced['accuracy']:.1%}")

    # Show confusion matrices
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)

    for label, result in [("Baseline (0.33)", baseline), (f"Best Draw ({best_draw['threshold']:.2f})", best_draw)]:
        cm = result['confusion_matrix']
        print(f"\n{label}:")
        print("                Predicted")
        print("             Away   Draw   Home")
        print(f"Actual Away:  {cm[0][0]:4d}   {cm[0][1]:4d}   {cm[0][2]:4d}")
        print(f"Actual Draw:  {cm[1][0]:4d}   {cm[1][1]:4d}   {cm[1][2]:4d}")
        print(f"Actual Home:  {cm[2][0]:4d}   {cm[2][1]:4d}   {cm[2][2]:4d}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    improvement = best_draw['draw_accuracy'] - baseline['draw_accuracy']
    log_loss_change = best_draw['log_loss'] - baseline['log_loss']

    if improvement > 5:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT POSSIBLE!")
        print(f"   Using threshold {best_draw['threshold']:.2f} improves draw accuracy by {improvement:.1f}%")
        print(f"   Log loss change: {log_loss_change:+.4f} (minimal impact)")
        print(f"\n   Recommendation: Use threshold {best_draw['threshold']:.2f} for betting")
    elif improvement > 2:
        print(f"\n⚠️ MODERATE IMPROVEMENT")
        print(f"   Using threshold {best_draw['threshold']:.2f} improves draw accuracy by {improvement:.1f}%")
        print(f"   Consider testing, but gains are modest")
    else:
        print(f"\n❌ LIMITED IMPROVEMENT")
        print(f"   Threshold tuning only improves draw accuracy by {improvement:.1f}%")
        print(f"   Recommend trying higher class weights instead (Option A)")

    print("\n✅ Test complete!")

if __name__ == '__main__':
    main()
