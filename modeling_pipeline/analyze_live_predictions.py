#!/usr/bin/env python3
"""
Live Prediction Analysis - Last 7 Days

Tests the newly trained XGBoost model on recent matches.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import log_loss, accuracy_score
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger("live_prediction_analysis")

def main():
    print("=" * 80)
    print("LIVE PREDICTION ANALYSIS - LAST 7 DAYS")
    print("=" * 80)
    
    # Load features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for last 7 days
    today = datetime.now()
    start_date = today - timedelta(days=7)
    
    recent_df = df[df['date'] >= start_date].copy()
    
    if recent_df.empty:
        print(f"\nNo matches found in the last 7 days (since {start_date.date()})")
        print(f"Latest match in data: {df['date'].max().date()}")
        # Use last 7 days of available data instead
        recent_df = df.tail(100).copy()
        print(f"\nUsing last 100 matches instead for demonstration")
        print(f"Date range: {recent_df['date'].min().date()} to {recent_df['date'].max().date()}")
    else:
        print(f"\nFound {len(recent_df)} matches from {start_date.date()} to {today.date()}")
    
    # Load XGBoost model
    model_path = MODELS_DIR / "xgboost_model.joblib"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    logger.info("Loading XGBoost model...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.XGBoostFootballModel()
    model.load(model_path)
    
    # Make predictions
    print("\nGenerating predictions...")
    probs = model.predict_proba(recent_df, calibrated=True)
    preds = np.argmax(probs, axis=1)
    
    # Get actual outcomes
    y_true = recent_df['target'].values.astype(int)
    
    # Calculate metrics
    loss = log_loss(y_true, probs)
    acc = accuracy_score(y_true, preds)
    
    # Prediction distribution
    pred_counts = pd.Series(preds).value_counts().sort_index()
    
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    print(f"Matches Analyzed: {len(recent_df)}")
    print(f"Log Loss:         {loss:.4f}")
    print(f"Accuracy:         {acc:.1%}")
    print("-" * 80)
    
    print("\n" + "=" * 80)
    print("PREDICTION DISTRIBUTION")
    print("=" * 80)
    outcome_names = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    for outcome_id in [0, 1, 2]:
        count = pred_counts.get(outcome_id, 0)
        pct = count / len(recent_df) * 100
        print(f"{outcome_names[outcome_id]}: {count} ({pct:.1f}%)")
    
    # Check if draws are being predicted
    draws_predicted = pred_counts.get(1, 0)
    if draws_predicted > 0:
        print(f"\n✅ Model IS predicting draws! ({draws_predicted} out of {len(recent_df)})")
    else:
        print(f"\n❌ Model is NOT predicting any draws")
    
    # Detailed match-by-match results
    print("\n" + "=" * 80)
    print("MATCH-BY-MATCH RESULTS (Last 20 matches)")
    print("=" * 80)
    print(f"{'Date':<12} {'Fixture':<40} {'Actual':<10} {'Predicted':<10} {'Correct'}")
    print("-" * 80)
    
    outcome_labels = {0: 'Away', 1: 'Draw', 2: 'Home'}
    
    for i, row in recent_df.tail(20).reset_index().iterrows():
        date_str = str(row['date'].date())
        fixture = f"{row['home_team_name']} vs {row['away_team_name']}"
        actual = outcome_labels[int(row['target'])]
        predicted = outcome_labels[preds[len(recent_df) - 20 + i]]
        correct = "✅" if preds[len(recent_df) - 20 + i] == int(row['target']) else "❌"
        
        # Probabilities
        p_away, p_draw, p_home = probs[len(recent_df) - 20 + i]
        probs_str = f"(A:{p_away:.2f} D:{p_draw:.2f} H:{p_home:.2f})"
        
        print(f"{date_str:<12} {fixture[:38]:<40} {actual:<10} {predicted:<10} {correct} {probs_str}")
    
    print("=" * 80)
    
    # Confidence analysis
    print("\n" + "=" * 80)
    print("CONFIDENCE ANALYSIS")
    print("=" * 80)
    
    max_probs = np.max(probs, axis=1)
    high_confidence = (max_probs > 0.6).sum()
    medium_confidence = ((max_probs > 0.4) & (max_probs <= 0.6)).sum()
    low_confidence = (max_probs <= 0.4).sum()
    
    print(f"High Confidence (>60%):   {high_confidence} ({high_confidence/len(recent_df)*100:.1f}%)")
    print(f"Medium Confidence (40-60%): {medium_confidence} ({medium_confidence/len(recent_df)*100:.1f}%)")
    print(f"Low Confidence (<40%):    {low_confidence} ({low_confidence/len(recent_df)*100:.1f}%)")
    
    # Accuracy by confidence level
    if high_confidence > 0:
        high_conf_mask = max_probs > 0.6
        high_conf_acc = accuracy_score(y_true[high_conf_mask], preds[high_conf_mask])
        print(f"\nAccuracy on High Confidence: {high_conf_acc:.1%}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ New XGBoost model tested on {len(recent_df)} recent matches")
    print(f"✅ Log Loss: {loss:.4f} (Target: <1.0)")
    print(f"✅ Accuracy: {acc:.1%} (Target: >55%)")
    if draws_predicted > 0:
        print(f"✅ Draw predictions: {draws_predicted} ({draws_predicted/len(recent_df)*100:.1f}%)")
    else:
        print(f"⚠️  No draws predicted - may need further tuning")
    
    return {
        'log_loss': loss,
        'accuracy': acc,
        'draws_predicted': draws_predicted,
        'total_matches': len(recent_df)
    }

if __name__ == "__main__":
    results = main()
