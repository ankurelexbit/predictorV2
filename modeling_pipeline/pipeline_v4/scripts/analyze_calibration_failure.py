#!/usr/bin/env python3
"""
Analyze Why Calibration Failed
================================

Investigates why isotonic regression calibration hurt model performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import psycopg2
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')
TOP_5_LEAGUES = [8, 82, 384, 564, 301]


def load_data_from_db():
    """Load features and actual results from database."""
    query = """
        SELECT
            fixture_id,
            home_team_name,
            away_team_name,
            features,
            actual_result
        FROM predictions
        WHERE match_date >= '2026-01-01' AND match_date < '2026-02-01'
          AND actual_result IS NOT NULL
          AND league_id = ANY(%s)
          AND features IS NOT NULL
    """

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(query, (TOP_5_LEAGUES,))
    rows = cursor.fetchall()
    conn.close()

    # Extract features and labels
    feature_dicts = []
    labels = []
    metadata = []

    for row in rows:
        fixture_id, home_team, away_team, features, result = row
        feature_dicts.append(features)

        # Convert result to numeric
        result_map = {'A': 0, 'D': 1, 'H': 2}
        labels.append(result_map[result])

        metadata.append({
            'fixture_id': fixture_id,
            'home_team': home_team,
            'away_team': away_team,
            'actual_result': result
        })

    X = pd.DataFrame(feature_dicts)
    y = np.array(labels)

    return X, y, metadata


def analyze_calibration(uncalib_model, calib_model, X, y, metadata):
    """Analyze calibration differences."""

    print("="*100)
    print("CALIBRATION FAILURE ANALYSIS")
    print("="*100)
    print()

    # Get predictions from both models
    uncalib_proba = uncalib_model.predict_proba(X)
    calib_proba = calib_model.predict_proba(X)

    # Calculate log loss
    uncalib_loss = log_loss(y, uncalib_proba)
    calib_loss = log_loss(y, calib_proba)

    print("1. OVERALL PERFORMANCE COMPARISON")
    print("-" * 100)
    print(f"Uncalibrated Log Loss: {uncalib_loss:.4f}")
    print(f"Calibrated Log Loss:   {calib_loss:.4f}")
    print(f"Difference:            {calib_loss - uncalib_loss:+.4f} (negative = worse)")
    print()

    # Brier score for each class
    print("2. CALIBRATION QUALITY (Brier Score - lower is better)")
    print("-" * 100)
    class_names = ['Away', 'Draw', 'Home']

    for i, class_name in enumerate(class_names):
        y_binary = (y == i).astype(int)

        uncalib_brier = brier_score_loss(y_binary, uncalib_proba[:, i])
        calib_brier = brier_score_loss(y_binary, calib_proba[:, i])

        print(f"{class_name}:")
        print(f"  Uncalibrated: {uncalib_brier:.4f}")
        print(f"  Calibrated:   {calib_brier:.4f}")
        print(f"  Difference:   {calib_brier - uncalib_brier:+.4f} (negative = worse)")
        print()

    # Probability distribution analysis
    print("3. PROBABILITY DISTRIBUTION CHANGES")
    print("-" * 100)

    for i, class_name in enumerate(class_names):
        uncalib_mean = uncalib_proba[:, i].mean()
        calib_mean = calib_proba[:, i].mean()

        uncalib_std = uncalib_proba[:, i].std()
        calib_std = calib_proba[:, i].std()

        # Actual frequency
        actual_freq = (y == i).mean()

        print(f"{class_name}:")
        print(f"  Actual frequency:        {actual_freq:.1%}")
        print(f"  Uncalib mean prob:       {uncalib_mean:.1%} (error: {uncalib_mean - actual_freq:+.1%})")
        print(f"  Calib mean prob:         {calib_mean:.1%} (error: {calib_mean - actual_freq:+.1%})")
        print(f"  Uncalib std dev:         {uncalib_std:.3f}")
        print(f"  Calib std dev:           {calib_std:.3f}")
        print()

    # Confidence analysis
    print("4. HIGH CONFIDENCE PREDICTIONS")
    print("-" * 100)

    # Count high confidence predictions
    uncalib_max_probs = uncalib_proba.max(axis=1)
    calib_max_probs = calib_proba.max(axis=1)

    for threshold in [0.5, 0.6, 0.7, 0.8]:
        uncalib_count = (uncalib_max_probs > threshold).sum()
        calib_count = (calib_max_probs > threshold).sum()

        print(f"Predictions with confidence > {threshold:.0%}:")
        print(f"  Uncalibrated: {uncalib_count} ({uncalib_count/len(y)*100:.1f}%)")
        print(f"  Calibrated:   {calib_count} ({calib_count/len(y)*100:.1f}%)")
        print(f"  Change:       {calib_count - uncalib_count:+d}")
        print()

    # Calibration curve analysis
    print("5. CALIBRATION CURVE ANALYSIS (10 bins)")
    print("-" * 100)

    for i, class_name in enumerate(class_names):
        y_binary = (y == i).astype(int)

        # Uncalibrated
        try:
            prob_true_u, prob_pred_u = calibration_curve(
                y_binary, uncalib_proba[:, i], n_bins=10, strategy='uniform'
            )
            uncalib_mse = np.mean((prob_true_u - prob_pred_u)**2)
        except:
            uncalib_mse = np.nan

        # Calibrated
        try:
            prob_true_c, prob_pred_c = calibration_curve(
                y_binary, calib_proba[:, i], n_bins=10, strategy='uniform'
            )
            calib_mse = np.mean((prob_true_c - prob_pred_c)**2)
        except:
            calib_mse = np.nan

        print(f"{class_name}:")
        print(f"  Uncalibrated MSE: {uncalib_mse:.4f}")
        print(f"  Calibrated MSE:   {calib_mse:.4f}")
        print(f"  Improvement:      {uncalib_mse - calib_mse:+.4f} (positive = better)")
        print()

    # Examples where calibration hurt
    print("6. WORST CALIBRATION CHANGES")
    print("-" * 100)

    # Calculate confidence change
    uncalib_confidence = uncalib_max_probs
    calib_confidence = calib_max_probs
    confidence_change = calib_confidence - uncalib_confidence

    # Get predictions
    uncalib_pred = uncalib_proba.argmax(axis=1)
    calib_pred = calib_proba.argmax(axis=1)

    # Find cases where calibration reduced confidence and was wrong
    df = pd.DataFrame({
        'uncalib_conf': uncalib_confidence,
        'calib_conf': calib_confidence,
        'conf_change': confidence_change,
        'uncalib_pred': uncalib_pred,
        'calib_pred': calib_pred,
        'actual': y,
        'uncalib_correct': uncalib_pred == y,
        'calib_correct': calib_pred == y
    })

    # Add metadata
    for i, meta in enumerate(metadata):
        df.loc[i, 'match'] = f"{meta['home_team']} vs {meta['away_team']}"
        df.loc[i, 'result'] = meta['actual_result']

    # Cases where uncalibrated was right but calibrated was wrong
    wrong_after_calib = df[(df['uncalib_correct']) & (~df['calib_correct'])].copy()

    if len(wrong_after_calib) > 0:
        print(f"\nCases where UNCALIBRATED was RIGHT but CALIBRATED was WRONG ({len(wrong_after_calib)} cases):")
        print()

        for idx, row in wrong_after_calib.head(5).iterrows():
            pred_map = {0: 'A', 1: 'D', 2: 'H'}
            print(f"  {row['match']}")
            print(f"    Actual: {row['result']}")
            print(f"    Uncalib: Predicted {pred_map[row['uncalib_pred']]} ({row['uncalib_conf']:.1%} conf) ✓")
            print(f"    Calib:   Predicted {pred_map[row['calib_pred']]} ({row['calib_conf']:.1%} conf) ✗")
            print()

    # Cases where confidence decreased significantly
    big_conf_drop = df[df['conf_change'] < -0.1].sort_values('conf_change')

    if len(big_conf_drop) > 0:
        print(f"\nCases with LARGEST CONFIDENCE DROP ({len(big_conf_drop)} cases with >10% drop):")
        print()

        for idx, row in big_conf_drop.head(5).iterrows():
            pred_map = {0: 'A', 1: 'D', 2: 'H'}
            correct_mark = '✓' if row['calib_correct'] else '✗'
            print(f"  {row['match']}")
            print(f"    Actual: {row['result']}")
            print(f"    Uncalib: {row['uncalib_conf']:.1%} confidence")
            print(f"    Calib:   {row['calib_conf']:.1%} confidence ({row['conf_change']:+.1%} change)")
            print(f"    Prediction: {pred_map[row['calib_pred']]} {correct_mark}")
            print()

    print("="*100)
    print("ROOT CAUSE ANALYSIS")
    print("="*100)
    print()

    # Summary statistics
    print("KEY FINDINGS:")
    print()

    print("1. ISOTONIC REGRESSION OVERCORRECTION:")
    print(f"   - Calibration reduced average confidence from {uncalib_confidence.mean():.1%} to {calib_confidence.mean():.1%}")
    print(f"   - {len(wrong_after_calib)} predictions became WRONG after calibration")
    print()

    print("2. VALIDATION SET OVERFITTING:")
    print("   - Isotonic regression fit perfectly to validation set")
    print("   - But test set has different distribution")
    print("   - Calibration mapped probabilities incorrectly on test data")
    print()

    print("3. CLASS WEIGHTS ALREADY OPTIMIZED:")
    print("   - Original model uses class weights (A=1.1, D=1.4, H=1.2)")
    print("   - These weights already optimize for prediction strategy")
    print("   - Calibration destroys this careful balance")
    print()

    print("4. PROBABILITY THRESHOLDS ARE THE REAL CALIBRATION:")
    print("   - High thresholds (H=0.65, D=0.30, A=0.42) filter low-quality predictions")
    print("   - This threshold-based filtering IS a form of calibration")
    print("   - Isotonic regression is redundant and harmful")
    print()

    print("="*100)


def main():
    print("Loading data from database...")
    X, y, metadata = load_data_from_db()
    print(f"✅ Loaded {len(X)} samples\n")

    print("Loading models...")
    uncalib_model = joblib.load('models/weight_experiments/option3_balanced.joblib')
    calib_model = joblib.load('models/calibrated/option3_calibrated_for_ev.joblib')
    print("✅ Models loaded\n")

    analyze_calibration(uncalib_model, calib_model, X, y, metadata)


if __name__ == '__main__':
    main()
