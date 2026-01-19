"""Check home advantage parameters in trained models."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODELS_DIR = Path("models")

print("=" * 80)
print("CHECKING HOME ADVANTAGE PARAMETERS")
print("=" * 80)

# Check Elo model
print("\n1. ELO MODEL:")
elo_data = joblib.load(MODELS_DIR / "elo_model.joblib")
print(f"   Is calibrated: {elo_data['is_calibrated']}")

# Check Dixon-Coles model
print("\n2. DIXON-COLES MODEL:")
dc_data = joblib.load(MODELS_DIR / "dixon_coles_model.joblib")
print(f"   Home advantage: {dc_data['base_model_data']['home_adv']:.3f}")
print(f"   Rho (low score correlation): {dc_data['base_model_data']['rho']:.3f}")
print(f"   In exp() form: e^{dc_data['base_model_data']['home_adv']:.3f} = {np.exp(dc_data['base_model_data']['home_adv']):.3f}x goal multiplier")

# Check training data to see actual home win rate
print("\n3. TRAINING DATA STATISTICS:")
features_df = pd.read_csv('data/processed/sportmonks_features.csv')
features_df = features_df[features_df['target'].notna()]

total = len(features_df)
home_wins = (features_df['target'] == 2).sum()
draws = (features_df['target'] == 1).sum()
away_wins = (features_df['target'] == 0).sum()

print(f"   Total matches: {total}")
print(f"   Home wins: {home_wins} ({home_wins/total:.1%})")
print(f"   Draws: {draws} ({draws/total:.1%})")
print(f"   Away wins: {away_wins} ({away_wins/total:.1%})")

# Check Elo implementation
print("\n4. ELO MODEL HOME ADVANTAGE:")
print("   Checking source code...")
import importlib
elo_module = importlib.import_module('04_model_baseline_elo')
EloProbabilityModel = elo_module.EloProbabilityModel
model = EloProbabilityModel()
print(f"   Default home advantage: {model.home_advantage} Elo points")

# Calculate what this means in probability
elo_diff = 100  # home advantage
exp_home = 1 / (1 + 10 ** (-elo_diff / 400))
print(f"   100 Elo points = {exp_home:.1%} expected score (before draw adjustment)")

# Even with -63 Elo diff, adding 100 gives +37 for home
elo_diff_sparta = -63 + 100
exp_with_ha = 1 / (1 + 10 ** (-elo_diff_sparta / 400))
print(f"\n   Example: Feyenoord vs Sparta (Elo diff -63)")
print(f"   After adding 100 HA: -63 + 100 = +37 Elo for Feyenoord")
print(f"   Expected score: {exp_with_ha:.1%} (favors home despite being weaker!)")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("🚨 100 Elo point home advantage is TOO LARGE!")
print("   It's overriding actual team strength differences up to ~100 Elo points.")
print("   When Sparta is 63 Elo stronger, adding 100 HA makes Feyenoord favored.")
print("\n   Typical home advantage should be ~50-70 Elo points, not 100.")
print("\n   This explains why model predicts home win even when away team is stronger!")
