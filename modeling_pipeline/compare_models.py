#!/usr/bin/env python3
"""
Compare Original vs Draw-Tuned Model

Tests both models on the last 7 days to see the difference.
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

logger = setup_logger("model_comparison")

def main():
    print("=" * 80)
    print("MODEL COMPARISON - ORIGINAL VS DRAW-TUNED")
    print("=" * 80)
    
    # Load features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for last 7 days
    today = datetime.now()
    start_date = today - timedelta(days=7)
    recent_df = df[df['date'] >= start_date].copy()
    
    print(f"\nAnalyzing {len(recent_df)} matches from {start_date.date()} to {today.date()}")
    
    # Load models
    import importlib.util
    spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    print("\nLoading models...")
    original_model = mod.XGBoostFootballModel()
    original_model.load(MODELS_DIR / "xgboost_model.joblib")
    
    draw_tuned_model = mod.XGBoostFootballModel()
    draw_tuned_model.load(MODELS_DIR / "xgboost_model_draw_tuned.joblib")
    
    # Get predictions
    y_true = recent_df['target'].values.astype(int)
    
    original_probs = original_model.predict_proba(recent_df, calibrated=True)
    original_preds = np.argmax(original_probs, axis=1)
    
    draw_tuned_probs = draw_tuned_model.predict_proba(recent_df, calibrated=True)
    draw_tuned_preds = np.argmax(draw_tuned_probs, axis=1)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    models = {
        'Original Model': (original_preds, original_probs),
        'Draw-Tuned Model': (draw_tuned_preds, draw_tuned_probs)
    }
    
    outcome_names = {0: 'Away', 1: 'Draw', 2: 'Home'}
    
    for model_name, (preds, probs) in models.items():
        loss = log_loss(y_true, probs)
        acc = accuracy_score(y_true, preds)
        
        print(f"\n{model_name}:")
        print(f"  Log Loss:  {loss:.4f}")
        print(f"  Accuracy:  {acc:.1%}")
        
        # Prediction distribution
        print(f"  Predictions:")
        for outcome_id in [0, 1, 2]:
            count = (preds == outcome_id).sum()
            pct = count / len(preds) * 100
            actual_count = (y_true == outcome_id).sum()
            print(f"    {outcome_names[outcome_id]}: {count} ({pct:.1f}%) [Actual: {actual_count}]")
    
    # Detailed comparison
    print("\n" + "=" * 80)
    print("PREDICTION DIFFERENCES (Last 20 matches)")
    print("=" * 80)
    print(f"{'Date':<12} {'Fixture':<35} {'Actual':<8} {'Original':<8} {'Draw-Tuned':<10}")
    print("-" * 80)
    
    for i, row in recent_df.tail(20).reset_index().iterrows():
        idx = len(recent_df) - 20 + i
        date_str = str(row['date'].date())
        fixture = f"{row['home_team_name'][:15]} vs {row['away_team_name'][:15]}"
        actual = outcome_names[int(row['target'])]
        orig_pred = outcome_names[original_preds[idx]]
        draw_pred = outcome_names[draw_tuned_preds[idx]]
        
        # Highlight if predictions differ
        diff_marker = " *" if orig_pred != draw_pred else ""
        
        print(f"{date_str:<12} {fixture:<35} {actual:<8} {orig_pred:<8} {draw_pred:<10}{diff_marker}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    orig_draws = (original_preds == 1).sum()
    tuned_draws = (draw_tuned_preds == 1).sum()
    actual_draws = (y_true == 1).sum()
    
    print(f"Actual Draws:        {actual_draws}")
    print(f"Original Predicted:  {orig_draws}")
    print(f"Draw-Tuned Predicted: {tuned_draws}")
    
    if tuned_draws > orig_draws:
        print(f"\n✅ Draw-tuned model predicts {tuned_draws - orig_draws} more draws!")
    
    # Which model is better?
    orig_loss = log_loss(y_true, original_probs)
    tuned_loss = log_loss(y_true, draw_tuned_probs)
    
    if tuned_loss < orig_loss:
        print(f"✅ Draw-tuned model has better log loss ({tuned_loss:.4f} vs {orig_loss:.4f})")
    else:
        print(f"⚠️  Original model has better log loss ({orig_loss:.4f} vs {tuned_loss:.4f})")
    
    orig_acc = accuracy_score(y_true, original_preds)
    tuned_acc = accuracy_score(y_true, draw_tuned_preds)
    
    if tuned_acc > orig_acc:
        print(f"✅ Draw-tuned model has better accuracy ({tuned_acc:.1%} vs {orig_acc:.1%})")
    else:
        print(f"⚠️  Original model has better accuracy ({orig_acc:.1%} vs {tuned_acc:.1%})")

if __name__ == "__main__":
    main()
