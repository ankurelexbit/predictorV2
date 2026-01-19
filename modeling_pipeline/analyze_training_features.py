"""Analyze all features used in training."""
import pandas as pd
import numpy as np

# Load training features
features_df = pd.read_csv('data/processed/sportmonks_features.csv')

print("=" * 80)
print("TRAINING FEATURE ANALYSIS")
print("=" * 80)

# Get all columns
all_cols = features_df.columns.tolist()

# Categorize features
metadata = ['fixture_id', 'date', 'season_id', 'home_team_id', 'away_team_id', 
            'home_team_name', 'away_team_name', 'home_goals', 'away_goals']
target = ['target', 'home_win', 'draw', 'away_win']

feature_cols = [c for c in all_cols if c not in metadata + target]

print(f"\nTotal columns: {len(all_cols)}")
print(f"Metadata columns: {len(metadata)}")
print(f"Target columns: {len(target)}")
print(f"Feature columns: {len(feature_cols)}")

# Categorize by type
elo_features = [c for c in feature_cols if 'elo' in c.lower()]
form_features = [c for c in feature_cols if 'form' in c.lower() or 'wins' in c.lower()]
goal_features = [c for c in feature_cols if 'goal' in c.lower()]
xg_features = [c for c in feature_cols if 'xg' in c.lower()]
shot_features = [c for c in feature_cols if 'shot' in c.lower()]
pass_features = [c for c in feature_cols if 'pass' in c.lower()]
possession_features = [c for c in feature_cols if 'possession' in c.lower()]
position_features = [c for c in feature_cols if 'position' in c.lower() or 'points' in c.lower()]
h2h_features = [c for c in feature_cols if 'h2h' in c.lower()]
market_features = [c for c in feature_cols if 'odds' in c.lower() or 'market' in c.lower()]
player_features = [c for c in feature_cols if 'player' in c.lower() or 'rating' in c.lower()]
team_stats = [c for c in feature_cols if any(x in c.lower() for x in ['tackle', 'corner', 'foul', 'card', 'save'])]
attack_defense = [c for c in feature_cols if 'attack' in c.lower() or 'defense' in c.lower() or 'dangerous' in c.lower()]

print("\n" + "=" * 80)
print("FEATURE CATEGORIES")
print("=" * 80)

categories = [
    ("Elo Features", elo_features),
    ("Form & Wins", form_features),
    ("Goals", goal_features),
    ("xG (Expected Goals)", xg_features),
    ("Shots", shot_features),
    ("Passes", pass_features),
    ("Possession", possession_features),
    ("League Position & Points", position_features),
    ("Head-to-Head", h2h_features),
    ("Market/Odds", market_features),
    ("Player Stats", player_features),
    ("Team Stats (tackles, corners, etc)", team_stats),
    ("Attack/Defense Strength", attack_defense),
]

for cat_name, cat_features in categories:
    print(f"\n{cat_name}: {len(cat_features)} features")
    if len(cat_features) > 0 and len(cat_features) <= 10:
        for f in cat_features:
            print(f"  - {f}")
    elif len(cat_features) > 10:
        for f in cat_features[:5]:
            print(f"  - {f}")
        print(f"  ... and {len(cat_features) - 5} more")

# Check which features have the most missing values
print("\n" + "=" * 80)
print("FEATURES WITH MISSING DATA")
print("=" * 80)

missing_pct = (features_df[feature_cols].isnull().sum() / len(features_df) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 10]

print(f"\nFeatures with >10% missing data: {len(high_missing)}")
for feat, pct in high_missing.head(20).items():
    print(f"  {feat}: {pct:.1f}% missing")

# Check features with zero variance
print("\n" + "=" * 80)
print("FEATURES WITH ZERO/LOW VARIANCE")
print("=" * 80)

low_var = []
for col in feature_cols:
    values = features_df[col].dropna()
    if len(values) > 0:
        std = values.std()
        if std == 0 or (std < 0.01 and values.abs().max() < 1):
            low_var.append((col, std, values.nunique()))

if low_var:
    print(f"\nFound {len(low_var)} low-variance features:")
    for feat, std, nunique in low_var[:20]:
        print(f"  {feat}: std={std:.4f}, unique={nunique}")
else:
    print("\nNo low-variance features found")

# Save feature list
with open('data/processed/training_features_list.txt', 'w') as f:
    f.write("TRAINING FEATURES LIST\n")
    f.write("=" * 80 + "\n\n")
    for cat_name, cat_features in categories:
        f.write(f"\n{cat_name} ({len(cat_features)} features):\n")
        for feat in cat_features:
            f.write(f"  {feat}\n")

print(f"\n\nFeature list saved to: data/processed/training_features_list.txt")

