import pandas as pd
import numpy as np
import joblib
import importlib.util

df = pd.read_csv('data/processed/sportmonks_features.csv')
df = df[df['target'].notna() & df['season_name'].isin(['2023/2024'])].tail(20)

# Load ensemble
spec_elo = importlib.util.spec_from_file_location("elo", "04_model_baseline_elo.py")
elo_mod = importlib.util.module_from_spec(spec_elo)
spec_elo.loader.exec_module(elo_mod)

spec_xgb = importlib.util.spec_from_file_location("xgb", "06_model_xgboost.py")
xgb_mod = importlib.util.module_from_spec(spec_xgb)
spec_xgb.loader.exec_module(xgb_mod)

spec_ens = importlib.util.spec_from_file_location("ens", "07_model_ensemble.py")
ens_mod = importlib.util.module_from_spec(spec_ens)
spec_ens.loader.exec_module(ens_mod)

ensemble = ens_mod.EnsembleModel()
elo_model = elo_mod.EloProbabilityModel()
elo_model.load('models/elo_model.joblib')
ensemble.add_model('elo', elo_model, 0.2)

class DCWrapper:
    def predict_proba(self, df, calibrated=True):
        return np.ones((len(df), 3)) / 3
ensemble.add_model('dc', DCWrapper(), 0.3)

xgb_model = xgb_mod.XGBoostFootballModel()
xgb_model.load('models/xgboost_model.joblib')
ensemble.add_model('xgb', xgb_model, 0.5)

ens_data = joblib.load('models/ensemble_model.joblib')
if 'calibrators' in ens_data:
    ensemble.calibrators = ens_data['calibrators']
    ensemble.is_calibrated = True
if 'stacking_model' in ens_data:
    ensemble.stacking_model = ens_data['stacking_model']

preds = ensemble.predict_proba(df, calibrated=True)

print("Sample predictions:")
print(f"{'Home':<8} {'Draw':<8} {'Away':<8} Meets criteria?")
for i in range(min(20, len(preds))):
    h, d, a = preds[i, 2], preds[i, 1], preds[i, 0]
    
    # Check criteria
    away_ok = a >= 0.35
    draw_ok = abs(h - a) < 0.15
    home_ok = h >= 0.55
    any_ok = away_ok or draw_ok or home_ok
    
    print(f"{h:>6.1%}   {d:>6.1%}   {a:>6.1%}   ", end='')
    if any_ok:
        if away_ok:
            print("✓ Away", end='')
        if draw_ok:
            print(" ✓ Draw", end='')
        if home_ok:
            print(" ✓ Home", end='')
        print()
    else:
        print("✗ None")

print(f"\nMatches meeting criteria:")
print(f"  Away (≥35%): {(preds[:, 0] >= 0.35).sum()}")
print(f"  Draw (diff <15%): {(np.abs(preds[:, 2] - preds[:, 0]) < 0.15).sum()}")
print(f"  Home (≥55%): {(preds[:, 2] >= 0.55).sum()}")
