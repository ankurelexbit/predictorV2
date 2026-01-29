
import joblib
import sys
import os
from pathlib import Path
import importlib.util

# Load parent class definition dynamically
param_model_path = Path('/Users/ankurgupta/code/predictorV2/modeling_pipeline/06_model_xgboost.py')
spec = importlib.util.spec_from_file_location("xgboost_model", param_model_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
XGBoostFootballModel = mod.XGBoostFootballModel

# Make it available globally for joblib to find
sys.modules['xgboost_model'] = mod
sys.modules['06_model_xgboost'] = mod

def inspect_parent_model():
    model_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/models/xgboost_model_draw_tuned.joblib'
    print(f"Loading {model_path}...")
    
    try:
        model = joblib.load(model_path)
        print(f"Type: {type(model)}")
        print(f"Dir: {dir(model)}")
        
        if hasattr(model, 'feature_columns'):
            print(f"\nFound feature_columns ({len(model.feature_columns)}):")
            for f in sorted(model.feature_columns):
                print(f"  - {f}")
        elif hasattr(model, 'features'):
            print(f"\nFound features ({len(model.features)}):")
            for f in sorted(model.features):
                print(f"  - {f}")
        elif hasattr(model, 'feature_names'):
            print(f"\nFound feature_names ({len(model.feature_names)}):")
            for f in sorted(model.feature_names):
                print(f"  - {f}")
                
        # Also check dictionary
        if isinstance(model, dict):
            print("\nKeys in dict:")
            print(model.keys())
            if 'feature_columns' in model:
                 print(f"\nFound feature_columns in dict ({len(model['feature_columns'])}):")
                 for f in sorted(model['feature_columns']):
                    print(f"  - {f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_parent_model()
