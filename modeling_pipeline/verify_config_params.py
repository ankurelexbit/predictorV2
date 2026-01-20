#!/usr/bin/env python3
"""
Verify that config.py has the correct draw-tuned parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import XGBOOST_PARAMS

print("="*60)
print("CURRENT XGBOOST_PARAMS IN CONFIG.PY")
print("="*60)

for key, value in XGBOOST_PARAMS.items():
    print(f"{key:20s}: {value}")

print("\n" + "="*60)
print("RECOMMENDED DRAW-TUNED PARAMETERS")
print("="*60)

recommended = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.6,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,
    "gamma": 5.0,
    "reg_alpha": 2.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
}

for key, value in recommended.items():
    print(f"{key:20s}: {value}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)

key_params = ['min_child_weight', 'gamma', 'max_depth', 'learning_rate']
all_match = True

for key in key_params:
    current = XGBOOST_PARAMS.get(key)
    rec = recommended.get(key)
    match = "✅" if current == rec else "❌"
    print(f"{key:20s}: Current={current:8} | Recommended={rec:8} {match}")
    if current != rec:
        all_match = False

print("\n" + "="*60)
if all_match:
    print("✅ All key parameters match recommended values!")
else:
    print("❌ Parameters need to be updated in config.py")
    print("\nTo update, replace XGBOOST_PARAMS in config.py with:")
    print("\nXGBOOST_PARAMS = {")
    for key, value in recommended.items():
        if isinstance(value, str):
            print(f'    "{key}": "{value}",')
        else:
            print(f'    "{key}": {value},')
    print("}")
