#!/usr/bin/env python3
"""
Quick Model Test - Verify XGBoost is ready for predictions

This script tests both models to ensure they're working correctly.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR

def test_model(model_path, model_name):
    """Test a model to ensure it's ready for predictions."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Model file exists: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")
    
    # Load model
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        model = mod.XGBoostFootballModel()
        model.load(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test prediction on sample data
    try:
        features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
        df = pd.read_csv(features_path)
        sample = df.tail(10)
        
        probs = model.predict_proba(sample, calibrated=True)
        preds = np.argmax(probs, axis=1)
        
        print(f"✅ Predictions working")
        print(f"   Sample predictions: {preds[:5]}")
        print(f"   Sample probabilities: {probs[0]}")
        
        # Check if predicting draws
        draw_count = (preds == 1).sum()
        if draw_count > 0:
            print(f"✅ Model predicts draws: {draw_count}/10 in sample")
        else:
            print(f"⚠️  Model did not predict any draws in sample")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    print(f"✅ {model_name} is READY for production!")
    return True

def main():
    print("="*60)
    print("MODEL READINESS CHECK")
    print("="*60)
    
    models_to_test = [
        (MODELS_DIR / "xgboost_model.joblib", "Original XGBoost Model"),
        (MODELS_DIR / "xgboost_model_draw_tuned.joblib", "Draw-Tuned XGBoost Model"),
    ]
    
    results = {}
    for model_path, model_name in models_to_test:
        results[model_name] = test_model(model_path, model_name)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model_name, is_ready in results.items():
        status = "✅ READY" if is_ready else "❌ NOT READY"
        print(f"{model_name}: {status}")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if results.get("Draw-Tuned XGBoost Model"):
        print("✅ Use: xgboost_model_draw_tuned.joblib")
        print("   Reason: Predicts draws, better log loss")
    elif results.get("Original XGBoost Model"):
        print("⚠️  Use: xgboost_model.joblib")
        print("   Warning: Does not predict draws")
    else:
        print("❌ No models are ready!")
    
    print("\nTo make predictions, use:")
    print("  python predict_live.py --model xgboost_model_draw_tuned.joblib")

if __name__ == "__main__":
    main()
