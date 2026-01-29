
import pandas as pd
import joblib
import sys

def check_feature_metrics():
    # 1. Get Parent Features
    print("Loading parent model features...")
    try:
        model_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/models/xgboost_model_draw_tuned.joblib'
        data = joblib.load(model_path)
        parent_features = set(data['feature_columns'])
        print(f"Parent model uses {len(parent_features)} features")
    except Exception as e:
        print(f"Error loading parent model: {e}")
        return

    # 2. Get Our Data Columns
    print("\nLoading current dataset columns...")
    try:
        df = pd.read_csv('/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3/data/csv/training_data_complete.csv', nrows=5)
        our_columns = set(df.columns)
        print(f"Our dataset has {len(our_columns)} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. Compare
    missing_in_ours = parent_features - our_columns
    print(f"\nMISSING Features in our dataset ({len(missing_in_ours)}):")
    for f in sorted(list(missing_in_ours)):
        print(f"  - {f}")
        
    present_in_ours = parent_features.intersection(our_columns)
    print(f"\nPRESENT Features ({len(present_in_ours)}):")
    # print(sorted(list(present_in_ours)))

if __name__ == "__main__":
    check_feature_metrics()
