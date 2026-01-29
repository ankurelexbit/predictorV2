import joblib
import sys
from pathlib import Path
import pandas as pd

def main():
    root_dir = Path(__file__).parent.parent.parent
    parent_model_path = root_dir / 'models/xgboost_model_draw_tuned.joblib'
    v3_model_path = Path('models/xgboost_model.joblib')
    # If using user's script result: v3_model_path = Path('models/xgboost_selected.pkl') - but we focus on '03' trained model
    
    print(f"Loading Parent Model: {parent_model_path}")
    try:
        parent_model_data = joblib.load(parent_model_path)
        # Check if it's a dict or object
        if isinstance(parent_model_data, dict) and 'feature_columns' in parent_model_data:
            parent_features = parent_model_data['feature_columns']
        elif hasattr(parent_model_data, 'feature_columns'):
            parent_features = parent_model_data.feature_columns
        else:
             # Maybe XGBClassifier?
             if hasattr(parent_model_data, 'feature_names_in_'):
                 parent_features = parent_model_data.feature_names_in_
             else:
                 print("Could not find feature columns in Parent Model")
                 return
    except Exception as e:
        print(f"Error loading Parent Model: {e}")
        return

    print(f"Loading V3 Model: {v3_model_path}")
    try:
        v3_model_data = joblib.load(v3_model_path)
        if isinstance(v3_model_data, dict) and 'feature_columns' in v3_model_data:
            v3_features = v3_model_data['feature_columns']
        elif hasattr(v3_model_data, 'feature_columns'):
            v3_features = v3_model_data.feature_columns
        else:
             print("Could not find feature columns in V3 Model")
             return
    except Exception as e:
        print(f"Error loading V3 Model: {e}")
        return

    print(f"\nParent Features: {len(parent_features)}")
    print(f"V3 Features: {len(v3_features)}")
    
    common = set(parent_features) & set(v3_features)
    parent_only = set(parent_features) - set(v3_features)
    v3_only = set(v3_features) - set(parent_features)
    
    print(f"Common: {len(common)}")
    print(f"Parent Only: {len(parent_only)}")
    print(f"V3 Only: {len(v3_only)}")
    
    # Generate Output Markdown
    report_path = Path('/Users/ankurgupta/.gemini/antigravity/brain/b17befe7-0b46-48c7-8e29-6cb1b85b637c/feature_model_diff_analysis.md')
    with open(report_path, 'w') as f:
        f.write("# Model Feature Diff Analysis\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Parent Model Features: {len(parent_features)}\n")
        f.write(f"- V3 Model Features: {len(v3_features)}\n\n")
        
        f.write("## 1. Parent Only (Dropped in V3)\n")
        for ft in sorted(list(parent_only)):
            f.write(f"- `{ft}`\n")
            
        f.write("\n## 2. V3 Only (New in V3)\n")
        for ft in sorted(list(v3_only)):
            f.write(f"- `{ft}`\n")
            
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
