#!/usr/bin/env python3
"""
Session Changes Verification Script

Verifies that all improvements from the current session are properly
integrated into the original pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))

def check_feature_engineering():
    """Verify feature engineering has EMA and rest days."""
    print("\n" + "="*80)
    print("1. FEATURE ENGINEERING CHANGES")
    print("="*80)
    
    # Check if functions exist in the file
    fe_file = Path(__file__).parent / "02_sportmonks_feature_engineering.py"
    
    with open(fe_file, 'r') as f:
        content = f.read()
    
    checks = {
        "calculate_ema_features": "def calculate_ema_features(" in content,
        "calculate_rest_days": "def calculate_rest_days(" in content,
        "EMA called in main": "calculate_ema_features(df" in content,
        "Rest days called in main": "calculate_rest_days(df" in content,
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_pass = False
    
    # Check if features are in the output
    try:
        from config import PROCESSED_DATA_DIR
        features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
        
        if features_path.exists():
            df = pd.read_csv(features_path, nrows=10)
            
            # Check for EMA features
            ema_features = [c for c in df.columns if '_ema' in c]
            rest_features = [c for c in df.columns if 'rest' in c.lower()]
            
            print(f"\n  Feature counts in output:")
            print(f"    Total features: {len(df.columns)}")
            print(f"    EMA features: {len(ema_features)} (expected: 20)")
            print(f"    Rest features: {len(rest_features)} (expected: 5-7)")
            
            if len(ema_features) >= 20 and len(rest_features) >= 5:
                print(f"  ✅ Features are present in output file")
            else:
                print(f"  ⚠️  Features may be missing - run feature engineering")
                all_pass = False
        else:
            print(f"  ⚠️  Features file not found - run feature engineering")
            all_pass = False
            
    except Exception as e:
        print(f"  ⚠️  Could not verify features file: {e}")
        all_pass = False
    
    return all_pass

def check_xgboost_params():
    """Verify XGBoost parameters are updated."""
    print("\n" + "="*80)
    print("2. XGBOOST PARAMETER UPDATES")
    print("="*80)
    
    from config import XGBOOST_PARAMS
    
    expected = {
        'min_child_weight': 20,
        'gamma': 5.0,
        'max_depth': 3,
        'learning_rate': 0.03,
        'n_estimators': 500,
    }
    
    all_pass = True
    for param, expected_value in expected.items():
        actual_value = XGBOOST_PARAMS.get(param)
        match = actual_value == expected_value
        status = "✅" if match else "❌"
        print(f"  {status} {param}: {actual_value} (expected: {expected_value})")
        if not match:
            all_pass = False
    
    return all_pass

def check_ensemble_optimization():
    """Verify ensemble optimization is enabled."""
    print("\n" + "="*80)
    print("3. ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*80)
    
    ensemble_file = Path(__file__).parent / "07_model_ensemble.py"
    
    with open(ensemble_file, 'r') as f:
        content = f.read()
    
    checks = {
        "optimize_weights function exists": "def optimize_weights(" in content,
        "Optimization enabled by default": "default=True" in content and "--optimize-weights" in content,
        "Time-based split (not season)": "df.iloc[:train_end]" in content,
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_pass = False
    
    return all_pass

def check_model_files():
    """Verify model files exist."""
    print("\n" + "="*80)
    print("4. MODEL FILES")
    print("="*80)
    
    from config import MODELS_DIR
    
    models = {
        "XGBoost (original)": MODELS_DIR / "xgboost_model.joblib",
        "XGBoost (draw-tuned)": MODELS_DIR / "xgboost_model_draw_tuned.joblib",
        "Elo": MODELS_DIR / "elo_model.joblib",
        "Dixon-Coles": MODELS_DIR / "dixon_coles_model.joblib",
    }
    
    all_pass = True
    for name, path in models.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if exists else ""
        print(f"  {status} {name}: {size}")
        if name == "XGBoost (draw-tuned)" and not exists:
            all_pass = False
    
    return all_pass

def check_hyperparameter_grid():
    """Verify hyperparameter grid is expanded."""
    print("\n" + "="*80)
    print("5. HYPERPARAMETER GRID EXPANSION")
    print("="*80)
    
    xgb_file = Path(__file__).parent / "06_model_xgboost.py"
    
    with open(xgb_file, 'r') as f:
        content = f.read()
    
    # Check if expanded grid exists
    checks = {
        "min_child_weight expanded": "'min_child_weight': [1, 3, 5, 7, 10]" in content,
        "gamma expanded": "'gamma': [0, 0.1, 0.5, 1.0]" in content,
        "Comments about draws": "# Expanded: higher values for more draws" in content or "conservative" in content.lower(),
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_pass = False
    
    return all_pass

def check_documentation():
    """Verify documentation exists."""
    print("\n" + "="*80)
    print("6. DOCUMENTATION")
    print("="*80)
    
    brain_dir = Path(__file__).parent.parent / ".gemini" / "antigravity" / "brain"
    
    # Find the conversation directory
    conv_dirs = list(brain_dir.glob("*"))
    if conv_dirs:
        latest_conv = max(conv_dirs, key=lambda p: p.stat().st_mtime)
        
        docs = {
            "Implementation Plan": latest_conv / "implementation_plan.md",
            "Walkthrough": latest_conv / "walkthrough.md",
            "Task List": latest_conv / "task.md",
            "Model Usage Guide": latest_conv / "model_usage_guide.md",
        }
        
        all_pass = True
        for name, path in docs.items():
            exists = path.exists()
            status = "✅" if exists else "⚠️ "
            print(f"  {status} {name}")
            if name in ["Walkthrough", "Model Usage Guide"] and not exists:
                all_pass = False
    else:
        print("  ⚠️  Could not find documentation directory")
        all_pass = False
    
    return all_pass

def main():
    print("="*80)
    print("SESSION CHANGES VERIFICATION")
    print("="*80)
    print("\nVerifying all improvements from the current session are integrated...")
    
    results = {
        "Feature Engineering (EMA + Rest Days)": check_feature_engineering(),
        "XGBoost Parameters": check_xgboost_params(),
        "Ensemble Optimization": check_ensemble_optimization(),
        "Model Files": check_model_files(),
        "Hyperparameter Grid": check_hyperparameter_grid(),
        "Documentation": check_documentation(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_pass = True
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {component}")
        if not passed:
            all_pass = False
    
    print("\n" + "="*80)
    if all_pass:
        print("✅ ALL CHANGES VERIFIED - Pipeline is ready!")
        print("\nYour pipeline now includes:")
        print("  • 27 new features (20 EMA + 7 rest days)")
        print("  • Optimized XGBoost parameters for draw predictions")
        print("  • Automatic ensemble weight optimization")
        print("  • Expanded hyperparameter search grid")
        print("  • Draw-tuned model (xgboost_model_draw_tuned.joblib)")
    else:
        print("⚠️  SOME CHANGES MISSING - See details above")
        print("\nTo fix:")
        print("  1. Run: python 02_sportmonks_feature_engineering.py")
        print("  2. Run: python 06_model_xgboost.py --tune --n-trials 20")
        print("  3. Run: python 07_model_ensemble.py")
    
    print("="*80)
    
    return all_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
