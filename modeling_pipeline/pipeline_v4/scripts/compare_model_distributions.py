#!/usr/bin/env python3
"""
Compare Model Distributions
============================

Compare prediction distributions from models trained with different class weights.
Analyzes how each weight configuration affects home/draw/away predictions.

Usage:
    python3 scripts/compare_model_distributions.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

# Feature columns to exclude
FEATURES_TO_EXCLUDE = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target',
    'home_team_name', 'away_team_name', 'state_id'
]

def load_model_and_metadata(model_path):
    """Load model and its metadata."""
    model = joblib.load(model_path)

    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    return model, metadata

def prepare_test_data(data_path):
    """Load and prepare test data."""
    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in FEATURES_TO_EXCLUDE]

    # Map target
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Sort chronologically
    df = df.sort_values('match_date').reset_index(drop=True)

    # Use last 15% as test set (same as training script)
    n = len(df)
    test_start = int(n * 0.85)
    test_df = df.iloc[test_start:]

    X_test = test_df[feature_cols]
    y_test = test_df['target']

    return X_test, y_test, test_df

def analyze_model(model, X_test, y_test, model_name, class_weights):
    """Analyze a single model's predictions."""
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}")
    print(f"Class Weights: Home={class_weights[2]:.1f}, Draw={class_weights[1]:.1f}, Away={class_weights[0]:.1f}")
    print()

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate metrics
    from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    # Prediction distribution (what model predicts most often)
    away_pred_count = (y_pred == 0).sum()
    draw_pred_count = (y_pred == 1).sum()
    home_pred_count = (y_pred == 2).sum()

    away_pred_pct = away_pred_count / len(y_pred) * 100
    draw_pred_pct = draw_pred_count / len(y_pred) * 100
    home_pred_pct = home_pred_count / len(y_pred) * 100

    # Average probabilities
    avg_home_prob = y_pred_proba[:, 2].mean()
    avg_draw_prob = y_pred_proba[:, 1].mean()
    avg_away_prob = y_pred_proba[:, 0].mean()

    # Actual distribution
    away_actual_count = (y_test == 0).sum()
    draw_actual_count = (y_test == 1).sum()
    home_actual_count = (y_test == 2).sum()

    away_actual_pct = away_actual_count / len(y_test) * 100
    draw_actual_pct = draw_actual_count / len(y_test) * 100
    home_actual_pct = home_actual_count / len(y_test) * 100

    # Per-class accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)
    home_accuracy = conf_matrix[2, 2] / home_actual_count * 100 if home_actual_count > 0 else 0
    draw_accuracy = conf_matrix[1, 1] / draw_actual_count * 100 if draw_actual_count > 0 else 0
    away_accuracy = conf_matrix[0, 0] / away_actual_count * 100 if away_actual_count > 0 else 0

    print(f"📊 PERFORMANCE METRICS:")
    print(f"   Log Loss: {loss:.4f}")
    print(f"   Overall Accuracy: {acc:.1%}")
    print()

    print(f"📈 PREDICTION DISTRIBUTION:")
    print(f"   {'Outcome':<10} {'Predictions':<15} {'Avg Probability':<20} {'Actual':<15}")
    print(f"   {'-'*70}")
    print(f"   {'Home':<10} {home_pred_count:>4} ({home_pred_pct:>5.1f}%)   {avg_home_prob:>6.3f} ({avg_home_prob*100:>5.1f}%)      {home_actual_count:>4} ({home_actual_pct:>5.1f}%)")
    print(f"   {'Draw':<10} {draw_pred_count:>4} ({draw_pred_pct:>5.1f}%)   {avg_draw_prob:>6.3f} ({avg_draw_prob*100:>5.1f}%)      {draw_actual_count:>4} ({draw_actual_pct:>5.1f}%)")
    print(f"   {'Away':<10} {away_pred_count:>4} ({away_pred_pct:>5.1f}%)   {avg_away_prob:>6.3f} ({avg_away_prob*100:>5.1f}%)      {away_actual_count:>4} ({away_actual_pct:>5.1f}%)")
    print()

    print(f"🎯 PER-CLASS ACCURACY:")
    print(f"   Home: {home_accuracy:.1f}%")
    print(f"   Draw: {draw_accuracy:.1f}%")
    print(f"   Away: {away_accuracy:.1f}%")
    print()

    print(f"📊 CALIBRATION ANALYSIS:")
    print(f"   {'Outcome':<10} {'Predicted':<15} {'Actual':<15} {'Calibration':<15}")
    print(f"   {'-'*60}")
    home_cal = home_actual_pct - (avg_home_prob * 100)
    draw_cal = draw_actual_pct - (avg_draw_prob * 100)
    away_cal = away_actual_pct - (avg_away_prob * 100)
    print(f"   {'Home':<10} {avg_home_prob*100:>6.1f}%         {home_actual_pct:>6.1f}%         {home_cal:+.1f}%")
    print(f"   {'Draw':<10} {avg_draw_prob*100:>6.1f}%         {draw_actual_pct:>6.1f}%         {draw_cal:+.1f}%")
    print(f"   {'Away':<10} {avg_away_prob*100:>6.1f}%         {away_actual_pct:>6.1f}%         {away_cal:+.1f}%")

    return {
        'model_name': model_name,
        'class_weights': class_weights,
        'log_loss': loss,
        'accuracy': acc,
        'avg_home_prob': avg_home_prob,
        'avg_draw_prob': avg_draw_prob,
        'avg_away_prob': avg_away_prob,
        'home_pred_pct': home_pred_pct,
        'draw_pred_pct': draw_pred_pct,
        'away_pred_pct': away_pred_pct,
        'home_accuracy': home_accuracy,
        'draw_accuracy': draw_accuracy,
        'away_accuracy': away_accuracy,
        'home_calibration': home_cal,
        'draw_calibration': draw_cal,
        'away_calibration': away_cal
    }

def create_comparison_table(results):
    """Create comparison table of all models."""
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    print()

    print("📊 PREDICTION PROBABILITIES:")
    print(f"{'Model':<25} {'Home Prob':<15} {'Draw Prob':<15} {'Away Prob':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<25} {r['avg_home_prob']:<15.3f} {r['avg_draw_prob']:<15.3f} {r['avg_away_prob']:<15.3f}")
    print()

    print("📈 PREDICTION COUNTS (at max probability):")
    print(f"{'Model':<25} {'Home %':<15} {'Draw %':<15} {'Away %':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<25} {r['home_pred_pct']:<15.1f} {r['draw_pred_pct']:<15.1f} {r['away_pred_pct']:<15.1f}")
    print()

    print("🎯 ACCURACY:")
    print(f"{'Model':<25} {'Overall':<15} {'Home':<15} {'Draw':<15} {'Away':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<25} {r['accuracy']:<15.1%} {r['home_accuracy']:<15.1f} {r['draw_accuracy']:<15.1f} {r['away_accuracy']:<15.1f}")
    print()

    print("📉 LOG LOSS:")
    print(f"{'Model':<25} {'Log Loss':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<25} {r['log_loss']:<15.4f}")
    print()

    print("🎚️  CALIBRATION (Actual - Predicted):")
    print(f"{'Model':<25} {'Home':<15} {'Draw':<15} {'Away':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<25} {r['home_calibration']:+15.1f}% {r['draw_calibration']:+15.1f}% {r['away_calibration']:+15.1f}%")
    print()

    print("💡 RECOMMENDATIONS:")
    print("-"*80)

    # Find best for home calibration
    best_home_cal = min(results, key=lambda x: abs(x['home_calibration']))
    print(f"✅ Best Home Calibration: {best_home_cal['model_name']} ({best_home_cal['home_calibration']:+.1f}%)")

    # Find best for draw calibration
    best_draw_cal = min(results, key=lambda x: abs(x['draw_calibration']))
    print(f"✅ Best Draw Calibration: {best_draw_cal['model_name']} ({best_draw_cal['draw_calibration']:+.1f}%)")

    # Find best overall log loss
    best_loss = min(results, key=lambda x: x['log_loss'])
    print(f"✅ Best Log Loss: {best_loss['model_name']} ({best_loss['log_loss']:.4f})")

    # Find highest home probability
    highest_home = max(results, key=lambda x: x['avg_home_prob'])
    print(f"✅ Highest Home Confidence: {highest_home['model_name']} ({highest_home['avg_home_prob']:.3f})")


def main():
    print("="*80)
    print("MODEL DISTRIBUTION COMPARISON")
    print("="*80)
    print()

    # Model paths
    models = [
        {
            'name': 'Option 1: Conservative',
            'path': Path('models/weight_experiments/option1_conservative.joblib'),
            'weights_name': 'H=1.0, D=1.3, A=1.0'
        },
        {
            'name': 'Option 2: Aggressive',
            'path': Path('models/weight_experiments/option2_aggressive.joblib'),
            'weights_name': 'H=1.3, D=1.5, A=1.2'
        },
        {
            'name': 'Option 3: Balanced',
            'path': Path('models/weight_experiments/option3_balanced.joblib'),
            'weights_name': 'H=1.2, D=1.4, A=1.1'
        },
        {
            'name': 'Current Production',
            'path': Path('models/with_draw_features/conservative_with_draw_features.joblib'),
            'weights_name': 'H=1.0, D=1.5, A=1.2'
        }
    ]

    # Load test data
    data_path = Path('data/training_data_with_draw_features.csv')
    print(f"Loading test data from: {data_path}")
    X_test, y_test, test_df = prepare_test_data(data_path)
    print(f"✅ Test set: {len(X_test)} samples")
    print()

    # Analyze each model
    results = []
    for model_info in models:
        if not model_info['path'].exists():
            print(f"⚠️  Skipping {model_info['name']} - model file not found")
            continue

        print(f"Loading {model_info['name']}...")
        model, metadata = load_model_and_metadata(model_info['path'])

        # Get class weights from metadata or use defaults
        if metadata and 'class_weights' in metadata:
            class_weights = list(metadata['class_weights'].values())
        else:
            # Parse from name
            if 'Option 1' in model_info['name']:
                class_weights = [1.0, 1.3, 1.0]  # A/D/H
            elif 'Option 2' in model_info['name']:
                class_weights = [1.2, 1.5, 1.3]
            elif 'Option 3' in model_info['name']:
                class_weights = [1.1, 1.4, 1.2]
            else:
                class_weights = [1.2, 1.5, 1.0]

        result = analyze_model(model, X_test, y_test, model_info['name'], class_weights)
        results.append(result)

    # Create comparison table
    if len(results) > 1:
        create_comparison_table(results)

    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
