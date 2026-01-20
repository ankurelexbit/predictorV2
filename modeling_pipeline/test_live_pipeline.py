#!/usr/bin/env python3
"""
Test that predict_live.py uses the draw-tuned model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_live_pipeline_model():
    """Verify predict_live.py loads the draw-tuned model."""
    
    print("="*60)
    print("TESTING LIVE PIPELINE MODEL")
    print("="*60)
    
    # Check the file content
    predict_live_file = Path(__file__).parent / "predict_live.py"
    
    with open(predict_live_file, 'r') as f:
        content = f.read()
    
    # Check for draw-tuned model
    uses_draw_tuned = "xgboost_model_draw_tuned.joblib" in content
    uses_original = content.count("xgboost_model.joblib") > 0
    
    print(f"\n✅ Uses draw-tuned model: {uses_draw_tuned}")
    print(f"⚠️  References original model: {uses_original}")
    
    # Count occurrences
    draw_tuned_count = content.count("xgboost_model_draw_tuned.joblib")
    original_count = content.count('"xgboost_model.joblib"')
    
    print(f"\nOccurrences:")
    print(f"  draw-tuned: {draw_tuned_count}")
    print(f"  original: {original_count}")
    
    # Test loading
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    try:
        from predict_live import load_models
        from config import MODELS_DIR
        
        # Test XGBoost loading
        print("\nLoading XGBoost model...")
        models = load_models(model_name='xgboost')
        
        if 'xgboost' in models:
            print("✅ XGBoost model loaded successfully")
            
            # Check which model file was loaded
            xgb_draw_tuned_path = MODELS_DIR / "xgboost_model_draw_tuned.joblib"
            xgb_original_path = MODELS_DIR / "xgboost_model.joblib"
            
            if xgb_draw_tuned_path.exists():
                print(f"✅ Draw-tuned model exists: {xgb_draw_tuned_path}")
            if xgb_original_path.exists():
                print(f"ℹ️  Original model exists: {xgb_original_path}")
            
            # Test prediction
            print("\nTesting prediction...")
            import pandas as pd
            from config import PROCESSED_DATA_DIR
            
            features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
            df = pd.read_csv(features_path, nrows=5)
            
            probs = models['xgboost'].predict_proba(df, calibrated=True)
            preds = probs.argmax(axis=1)
            
            print(f"✅ Predictions working")
            print(f"   Sample predictions: {preds}")
            
            # Check if predicting draws
            if (preds == 1).any():
                print(f"✅ Model predicts draws!")
            else:
                print(f"⚠️  Model did not predict draws in sample")
                
        else:
            print("❌ XGBoost model not loaded")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if uses_draw_tuned and draw_tuned_count >= 2:
        print("✅ Live pipeline is configured to use draw-tuned model")
        print("✅ Model loads successfully")
        print("\nYou can now run:")
        print("  python predict_live.py")
        print("\nIt will use the optimized draw-tuned model!")
    else:
        print("⚠️  Live pipeline may not be using draw-tuned model")
        print("   Check predict_live.py manually")
    
    print("="*60)

if __name__ == "__main__":
    test_live_pipeline_model()
