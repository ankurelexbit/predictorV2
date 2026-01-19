"""
Quick script to check prediction distribution after Phase 1 changes
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from config import TEST_SEASONS
from utils import season_based_split

# Load test data
print("Loading data...")
features_df = pd.read_csv('data/processed/sportmonks_features.csv')
features_df['date'] = pd.to_datetime(features_df['date'])

# Filter to test season
_, _, test_df = season_based_split(
    features_df,
    'season_name',
    ["2019/2020", "2020/2021", "2021/2022"],
    ["2022/2023"],
    ["2023/2024"]
)

print(f"Test set: {len(test_df)} matches")

# Actual outcomes
y_true = test_df['target'].values.astype(int)

print("\n" + "="*60)
print("ACTUAL OUTCOME DISTRIBUTION (Test Set)")
print("="*60)

away_actual = (y_true == 0).sum()
draw_actual = (y_true == 1).sum()
home_actual = (y_true == 2).sum()
total = len(y_true)

print(f"Home wins: {home_actual} ({home_actual/total*100:.1f}%)")
print(f"Draws:     {draw_actual} ({draw_actual/total*100:.1f}%)")
print(f"Away wins: {away_actual} ({away_actual/total*100:.1f}%)")

# Load models and get predictions
print("\n" + "="*60)
print("MODEL PREDICTION DISTRIBUTIONS")
print("="*60)

# Elo model
print("\n1. Elo Model:")
# Import directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location("elo_module", "04_model_baseline_elo.py")
elo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elo_module)
EloProbabilityModel = elo_module.EloProbabilityModel

elo_model = EloProbabilityModel()
elo_model.load(Path('models/elo_model.joblib'))
elo_probs = elo_model.predict_proba(test_df, calibrated=True)
elo_preds = np.argmax(elo_probs, axis=1)

away_elo = (elo_preds == 0).sum()
draw_elo = (elo_preds == 1).sum()
home_elo = (elo_preds == 2).sum()

print(f"   Home wins: {home_elo} ({home_elo/total*100:.1f}%)")
print(f"   Draws:     {draw_elo} ({draw_elo/total*100:.1f}%)")
print(f"   Away wins: {away_elo} ({away_elo/total*100:.1f}%)")

# XGBoost model
print("\n2. XGBoost Model:")
spec_xgb = importlib.util.spec_from_file_location("xgb_module", "06_model_xgboost.py")
xgb_module = importlib.util.module_from_spec(spec_xgb)
spec_xgb.loader.exec_module(xgb_module)
XGBoostFootballModel = xgb_module.XGBoostFootballModel

xgb_model = XGBoostFootballModel()
xgb_model.load(Path('models/xgboost_model.joblib'))
xgb_probs = xgb_model.predict_proba(test_df, calibrated=True)
xgb_preds = np.argmax(xgb_probs, axis=1)

away_xgb = (xgb_preds == 0).sum()
draw_xgb = (xgb_preds == 1).sum()
home_xgb = (xgb_preds == 2).sum()

print(f"   Home wins: {home_xgb} ({home_xgb/total*100:.1f}%)")
print(f"   Draws:     {draw_xgb} ({draw_xgb/total*100:.1f}%)")
print(f"   Away wins: {away_xgb} ({away_xgb/total*100:.1f}%)")

# Summary comparison
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"\n{'Outcome':<10} {'Actual':<12} {'Elo':<12} {'XGBoost':<12}")
print("-" * 60)
print(f"{'Home':<10} {home_actual/total*100:>6.1f}%     {home_elo/total*100:>6.1f}%     {home_xgb/total*100:>6.1f}%")
print(f"{'Draw':<10} {draw_actual/total*100:>6.1f}%     {draw_elo/total*100:>6.1f}%     {draw_xgb/total*100:>6.1f}%")
print(f"{'Away':<10} {away_actual/total*100:>6.1f}%     {away_elo/total*100:>6.1f}%     {away_xgb/total*100:>6.1f}%")

# Success criteria
print("\n" + "="*60)
print("PHASE 1 SUCCESS CRITERIA")
print("="*60)

draw_elo_pct = draw_elo/total*100
draw_xgb_pct = draw_xgb/total*100
home_elo_pct = home_elo/total*100
home_xgb_pct = home_xgb/total*100
away_elo_pct = away_elo/total*100
away_xgb_pct = away_xgb/total*100

criteria = {
    "Model predicts draws (>0)": draw_elo > 0 and draw_xgb > 0,
    "Draw rate 22-30%": (22 <= draw_elo_pct <= 30) or (22 <= draw_xgb_pct <= 30),
    "Home rate 38-44%": (38 <= home_elo_pct <= 44) or (38 <= home_xgb_pct <= 44),
    "Away rate 30-36%": (30 <= away_elo_pct <= 36) or (30 <= away_xgb_pct <= 36),
}

for criterion, passed in criteria.items():
    status = "✓" if passed else "✗"
    print(f"{status} {criterion}")

if all(criteria.values()):
    print("\n✓ ALL CRITERIA MET! Phase 1 successful.")
else:
    print(f"\n✗ Some criteria not met. May need further tuning.")

print(f"\nKey improvement: Draw predictions increased from 0 to {draw_elo} (Elo) and {draw_xgb} (XGBoost)")
