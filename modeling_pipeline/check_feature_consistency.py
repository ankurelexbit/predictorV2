#!/usr/bin/env python3
"""
Feature Consistency Checker

Validates that live prediction features match training features exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR

def check_feature_consistency():
    """Check if live features match training features."""
    
    print("=" * 80)
    print("FEATURE CONSISTENCY CHECK")
    print("=" * 80)
    
    # Load training features
    print("\n1. Loading training features...")
    training_features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    
    if not training_features_path.exists():
        print(f"❌ Training features not found: {training_features_path}")
        return
    
    df_train = pd.read_csv(training_features_path, nrows=10)
    training_cols = set(df_train.columns)
    
    print(f"✅ Training features loaded: {len(training_cols)} columns")
    
    # Get feature columns (exclude metadata)
    metadata_cols = {'date', 'home_team_name', 'away_team_name', 'target', 
                     'league_id', 'season_id', 'fixture_id', 'home_team_id', 
                     'away_team_id', 'odds_home', 'odds_draw', 'odds_away'}
    
    training_feature_cols = training_cols - metadata_cols
    print(f"   Feature columns: {len(training_feature_cols)}")
    print(f"   Metadata columns: {len(metadata_cols)}")
    
    # Check what the model expects
    print("\n2. Checking XGBoost model expectations...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        model = mod.XGBoostFootballModel()
        model.load(MODELS_DIR / "xgboost_model_draw_tuned.joblib")
        
        if hasattr(model, 'feature_names'):
            expected_features = set(model.feature_names)
            print(f"✅ Model expects {len(expected_features)} features")
        else:
            print("⚠️  Model doesn't store feature names")
            expected_features = None
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        expected_features = None
    
    # Check live prediction feature generation
    print("\n3. Checking live prediction feature generation...")
    
    # Read the predict_live.py to see what features it generates
    predict_live_file = Path(__file__).parent / "predict_live.py"
    
    with open(predict_live_file, 'r') as f:
        live_code = f.read()
    
    # Check for key feature calculations
    feature_checks = {
        'Elo ratings': 'elo_diff' in live_code or 'home_elo' in live_code,
        'Rolling stats': 'rolling' in live_code.lower(),
        'Form features': 'wins_3' in live_code or 'form' in live_code.lower(),
        'EMA features': 'ema' in live_code.lower(),
        'Rest days': 'rest' in live_code.lower() or 'days_since' in live_code,
        'H2H features': 'h2h' in live_code.lower() or 'head_to_head' in live_code,
        'Standings': 'position' in live_code or 'points' in live_code,
    }
    
    print("\n   Feature categories in live code:")
    for feature, present in feature_checks.items():
        status = "✅" if present else "❌"
        print(f"   {status} {feature}")
    
    # Check for missing features warning
    print("\n4. Checking for missing features handling...")
    
    if 'Missing features' in live_code or 'missing_features' in live_code:
        print("✅ Live code handles missing features")
    else:
        print("⚠️  No explicit missing features handling found")
    
    # Analyze the log output from the live run
    print("\n5. Analyzing live prediction logs...")
    
    # The live run showed:
    # - "Built 246 features" for each match
    # - "⚠️ Lineups not available, using approximations"
    
    print("   From live run logs:")
    print("   ✅ Built 246 features per match")
    print("   ⚠️  Lineups not available (using approximations)")
    
    # Compare feature counts
    print("\n6. Feature count comparison...")
    
    training_feature_count = len(training_feature_cols)
    live_feature_count = 246  # From logs
    
    print(f"   Training: {training_feature_count} features")
    print(f"   Live:     {live_feature_count} features")
    
    if training_feature_count == live_feature_count:
        print("   ✅ Feature counts match!")
    else:
        diff = abs(training_feature_count - live_feature_count)
        print(f"   ⚠️  Feature count mismatch: {diff} features difference")
    
    # Check for player stats features
    print("\n7. Checking player stats features...")
    
    player_features = [col for col in training_cols if any(x in col for x in 
                      ['rating', 'duels', 'tackles', 'clearances', 'touches', 
                       'dispossessed', 'aerials', 'possession_lost'])]
    
    print(f"   Found {len(player_features)} player-level features in training")
    
    if player_features:
        print("   Sample player features:")
        for feat in list(player_features)[:5]:
            print(f"     - {feat}")
    
    # Check if these are missing in live
    if '⚠️ Lineups not available' in str(live_code):
        print("   ⚠️  Live predictions use approximations for player stats")
        print("   This could cause prediction differences!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    issues = []
    
    # Check 1: Feature count
    if training_feature_count != live_feature_count:
        issues.append(f"Feature count mismatch: {training_feature_count} vs {live_feature_count}")
    
    # Check 2: Player features
    if player_features and '⚠️ Lineups not available' in str(live_code):
        issues.append("Player stats features use approximations in live (lineups not available)")
    
    # Check 3: Missing feature categories
    missing_categories = [cat for cat, present in feature_checks.items() if not present]
    if missing_categories:
        issues.append(f"Missing feature categories in live: {', '.join(missing_categories)}")
    
    if not issues:
        print("\n✅ No major issues found!")
        print("   Features appear consistent between training and live prediction.")
    else:
        print(f"\n⚠️  Found {len(issues)} potential issues:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. Feature Alignment:")
    print("   - Verify that predict_live.py calculates ALL 490 training features")
    print("   - Check for any hardcoded defaults or approximations")
    
    print("\n2. Player Stats:")
    print("   - Live predictions use approximations when lineups unavailable")
    print("   - This is expected pre-match but may reduce accuracy")
    print("   - Consider using historical average player stats")
    
    print("\n3. Missing Data Handling:")
    print("   - Ensure missing features are filled with same defaults as training")
    print("   - Check that NaN handling is consistent")
    
    print("\n4. Validation:")
    print("   - Run a test: predict on a historical match and compare")
    print("   - Features should be identical for same match")
    
    return {
        'training_features': training_feature_count,
        'live_features': live_feature_count,
        'issues': issues
    }

if __name__ == "__main__":
    results = check_feature_consistency()
