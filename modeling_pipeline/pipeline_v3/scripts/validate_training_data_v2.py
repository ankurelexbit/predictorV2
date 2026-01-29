import pandas as pd
import numpy as np

def validate_training_data():
    print("Loading training data...")
    df = pd.read_csv('data/csv/training_data_complete_v2.csv')
    print(f"Loaded {len(df)} rows.")

    # 1. Check New Features
    new_features = [
        'home_pass_accuracy_5', 'home_duels_won_5', 'home_counter_attacks_5',
        'home_position', 'away_position'
    ]
    
    print("\nAttribute Quality (New Features):")
    for col in new_features:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            continue
            
        null_pct = df[col].isnull().mean() * 100
        mean_val = df[col].mean()
        min_val = df[col].min()
        max_val = df[col].max()
        
        print(f"{col}: Null={null_pct:.1f}%, Mean={mean_val:.2f}, Range=[{min_val}, {max_val}]")
        
        if null_pct > 20:
            print(f"  ⚠️ High null rate!")

    # 2. Check Logic (Position)
    # Position should be roughly 1-20
    if 'home_position' in df.columns:
        invalid_pos = df[
            (df['home_position'] < 1) | (df['home_position'] > 30)
        ]
        if len(invalid_pos) > 0:
            print(f"⚠️ {len(invalid_pos)} rows with suspicious position (outside 1-30)")
            
    # 3. Check Zeroed Features (Points)
    # As per user request, derived points features should be 0
    zero_features = ['home_points', 'home_points_at_home']
    print("\nChecking Zeroed Features (User Request):")
    for col in zero_features:
        if col in df.columns:
            non_zero = (df[col] != 0).sum()
            print(f"{col}: Non-zero count = {non_zero}")
            if non_zero > 0:
                print(f"  ❌ ERROR: Expected 0, found values!")

    # 4. Target Check
    print("\nTarget Check:")
    if 'target_home_win' in df.columns:
        print(f"Home Win Rate: {df['target_home_win'].mean():.3f}")
    else:
        print("❌ Missing target!")

if __name__ == "__main__":
    validate_training_data()
