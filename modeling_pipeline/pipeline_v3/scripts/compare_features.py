
import joblib
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project paths
sys.path.append('/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3')
sys.path.append('/Users/ankurgupta/code/predictorV2/modeling_pipeline')

def get_parent_features():
    """Extract features from parent pipeline model"""
    try:
        model_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/models/xgboost_model_draw_tuned.joblib'
        if not os.path.exists(model_path):
            print(f"Parent model not found at {model_path}")
            return set()
            
        model = joblib.load(model_path)
        # Check if it has feature_names or similar
        if hasattr(model, 'feature_columns'):
            return set(model.feature_columns)
        elif hasattr(model, 'feature_names_in_'):
            return set(model.feature_names_in_)
        elif hasattr(model, 'get_booster'):
            return set(model.get_booster().feature_names)
        else:
            print("Could not extract features from parent model object")
            return set()
    except Exception as e:
        print(f"Error loading parent model: {e}")
        return set()

def get_new_features():
    """Extract features from new pipeline model"""
    try:
        model_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3/models/xgboost_model.joblib'
        if not os.path.exists(model_path):
            print(f"New model not found at {model_path}")
            return set()
            
        data = joblib.load(model_path)
        if isinstance(data, dict) and 'feature_columns' in data:
            return set(data['feature_columns'])
        
        # Fallback if it's the object itself
        if hasattr(data, 'feature_columns'):
            return set(data.feature_columns)
            
        print("Could not extract features from new model object")
        return set()
    except Exception as e:
        print(f"Error loading new model: {e}")
        return set()

def main():
    print("EXTRACTING FEATURES FROM MODELS...")
    parent_features = get_parent_features()
    new_features = get_new_features()
    
    print(f"\nParent Model Features: {len(parent_features)}")
    print(f"New Model Features:    {len(new_features)}")
    
    # Compare
    missing_in_new = parent_features - new_features
    extra_in_new = new_features - parent_features
    common = parent_features.intersection(new_features)
    
    print(f"\nCommon Features: {len(common)}")
    
    print(f"\nMISSING IN NEW PIPELINE ({len(missing_in_new)} features):")
    print("-" * 50)
    for f in sorted(list(missing_in_new)):
        print(f"  - {f}")
        
    print(f"\nEXTRA IN NEW PIPELINE ({len(extra_in_new)} features):")
    print("-" * 50)
    for f in sorted(list(extra_in_new))[:20]:  # Show top 20
        print(f"  - {f}")
    if len(extra_in_new) > 20:
        print(f"  ... and {len(extra_in_new)-20} more")

if __name__ == "__main__":
    main()
