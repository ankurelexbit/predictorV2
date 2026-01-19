"""
Deep diagnostic analysis to find all issues beyond home advantage.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("=" * 80)
print("DEEP DIAGNOSTIC ANALYSIS - FINDING ALL MODEL ISSUES")
print("=" * 80)

# Load training data
features_df = pd.read_csv('data/processed/sportmonks_features.csv')
features_df = features_df[features_df['target'].notna()].copy()

print(f"\nTotal training samples: {len(features_df)}")

# ============================================================================
# ISSUE #1: ELO RATING VARIANCE
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE CHECK #1: ELO RATING VARIANCE")
print("=" * 80)

print("\nHome Elo Statistics:")
print(f"  Mean: {features_df['home_elo'].mean():.1f}")
print(f"  Std Dev: {features_df['home_elo'].std():.1f}")
print(f"  Min: {features_df['home_elo'].min():.1f}")
print(f"  Max: {features_df['home_elo'].max():.1f}")
print(f"  Range: {features_df['home_elo'].max() - features_df['home_elo'].min():.1f}")

print("\nAway Elo Statistics:")
print(f"  Mean: {features_df['away_elo'].mean():.1f}")
print(f"  Std Dev: {features_df['away_elo'].std():.1f}")
print(f"  Min: {features_df['away_elo'].min():.1f}")
print(f"  Max: {features_df['away_elo'].max():.1f}")
print(f"  Range: {features_df['away_elo'].max() - features_df['away_elo'].min():.1f}")

print("\nElo Difference Statistics:")
print(f"  Mean: {features_df['elo_diff'].mean():.1f}")
print(f"  Std Dev: {features_df['elo_diff'].std():.1f}")
print(f"  Min: {features_df['elo_diff'].min():.1f}")
print(f"  Max: {features_df['elo_diff'].max():.1f}")

# Check if Elo is too compressed
if features_df['home_elo'].std() < 100:
    print("\n⚠️  PROBLEM: Elo ratings have LOW variance (std < 100)")
    print("    Ratings are too compressed to distinguish team strengths")
elif features_df['home_elo'].std() > 200:
    print("\n⚠️  PROBLEM: Elo ratings have VERY HIGH variance")
    print("    Ratings may be unstable or incorrectly calculated")
else:
    print("\n✓ Elo variance looks reasonable")

# Check if most matches have similar Elo diff
similar_strength = (features_df['elo_diff'].abs() < 50).sum()
print(f"\nMatches with Elo diff < 50 points: {similar_strength} ({similar_strength/len(features_df):.1%})")
if similar_strength / len(features_df) > 0.7:
    print("⚠️  PROBLEM: Too many matches with similar team strength")
    print("    Elo system may not be differentiating teams properly")

# ============================================================================
# ISSUE #2: FORM CALCULATION VARIANCE
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE CHECK #2: FORM CALCULATION VARIANCE")
print("=" * 80)

print("\nHome Form (5 games) Statistics:")
print(f"  Mean: {features_df['home_form_5'].mean():.2f}")
print(f"  Std Dev: {features_df['home_form_5'].std():.2f}")
print(f"  Min: {features_df['home_form_5'].min():.2f}")
print(f"  Max: {features_df['home_form_5'].max():.2f}")
print(f"  Range: 0-15 possible (0=5 losses, 15=5 wins)")

# Check for form issues
zero_form_home = (features_df['home_form_5'] == 0).sum()
zero_form_away = (features_df['away_form_5'] == 0).sum()
print(f"\nMatches where home form = 0: {zero_form_home} ({zero_form_home/len(features_df):.1%})")
print(f"Matches where away form = 0: {zero_form_away} ({zero_form_away/len(features_df):.1%})")

if zero_form_home / len(features_df) > 0.3:
    print("⚠️  PROBLEM: Too many teams with zero form")
    print("    Form calculation may be broken or data missing")

# Check if form correlates with outcome
form_diff = features_df['home_form_5'] - features_df['away_form_5']
home_wins = features_df[features_df['target'] == 2]
away_wins = features_df[features_df['target'] == 0]

print(f"\nForm diff when home wins: {form_diff[features_df['target']==2].mean():.2f}")
print(f"Form diff when away wins: {form_diff[features_df['target']==0].mean():.2f}")
print(f"Form diff when draw: {form_diff[features_df['target']==1].mean():.2f}")

# ============================================================================
# ISSUE #3: XGBOOST FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE CHECK #3: XGBOOST FEATURE IMPORTANCE")
print("=" * 80)

try:
    xgb_model_data = joblib.load('models/xgboost_model.joblib')
    feature_importance = xgb_model_data.get('feature_importance', {})
    
    if feature_importance:
        # Convert to sorted list
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 15 Most Important Features:")
        for i, (feat, importance) in enumerate(sorted_features[:15], 1):
            print(f"  {i}. {feat}: {importance:.1f}")
        
        # Check for home bias in features
        top_10_features = [f[0] for f in sorted_features[:10]]
        home_features = [f for f in top_10_features if 'home' in f.lower()]
        away_features = [f for f in top_10_features if 'away' in f.lower()]
        
        print(f"\nTop 10 features mentioning 'home': {len(home_features)}")
        print(f"Top 10 features mentioning 'away': {len(away_features)}")
        
        if len(home_features) > len(away_features) * 1.5:
            print("⚠️  PROBLEM: Model relies more heavily on home features")
            print("    This creates home bias in predictions")
except Exception as e:
    print(f"Could not load XGBoost feature importance: {e}")

# ============================================================================
# ISSUE #4: TARGET CLASS DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE CHECK #4: TRAINING DATA CLASS DISTRIBUTION")
print("=" * 80)

home_win_rate = (features_df['target'] == 2).sum() / len(features_df)
draw_rate = (features_df['target'] == 1).sum() / len(features_df)
away_win_rate = (features_df['target'] == 0).sum() / len(features_df)

print(f"\nTraining Data Distribution:")
print(f"  Home wins: {home_win_rate:.1%}")
print(f"  Draws: {draw_rate:.1%}")
print(f"  Away wins: {away_win_rate:.1%}")

print(f"\nClass imbalance ratio: {home_win_rate / away_win_rate:.2f}x")

if home_win_rate / away_win_rate > 1.5:
    print("⚠️  PROBLEM: Significant class imbalance favoring home wins")
    print("    Model will learn to predict home wins more often")

# ============================================================================
# ISSUE #5: FEATURE SCALING
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE CHECK #5: FEATURE SCALING ISSUES")
print("=" * 80)

# Check scales of different features
features_to_check = {
    'home_elo': features_df['home_elo'].values,
    'home_form_5': features_df['home_form_5'].values,
    'home_goals_5': features_df['home_goals_5'].values,
    'home_possession_pct_5': features_df['home_possession_pct_5'].values,
}

print("\nFeature Scale Comparison:")
for name, values in features_to_check.items():
    valid_values = values[~np.isnan(values)]
    if len(valid_values) > 0:
        print(f"  {name}:")
        print(f"    Range: [{valid_values.min():.2f}, {valid_values.max():.2f}]")
        print(f"    Mean: {valid_values.mean():.2f}, Std: {valid_values.std():.2f}")

# Check if scales are vastly different
scales = []
for values in features_to_check.values():
    valid_values = values[~np.isnan(values)]
    if len(valid_values) > 0:
        scales.append(valid_values.max() - valid_values.min())

if max(scales) / min(scales) > 100:
    print("\n⚠️  PROBLEM: Features have vastly different scales")
    print(f"    Scale ratio: {max(scales) / min(scales):.1f}x")
    print("    May need better normalization")

# ============================================================================
# ISSUE #6: HOME VS AWAY FEATURE SYMMETRY
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE CHECK #6: HOME VS AWAY FEATURE SYMMETRY")
print("=" * 80)

# Check if home features are systematically higher than away
home_elo_mean = features_df['home_elo'].mean()
away_elo_mean = features_df['away_elo'].mean()

print(f"\nAverage Elo:")
print(f"  Home: {home_elo_mean:.1f}")
print(f"  Away: {away_elo_mean:.1f}")
print(f"  Difference: {home_elo_mean - away_elo_mean:.1f}")

if abs(home_elo_mean - away_elo_mean) > 20:
    print("⚠️  PROBLEM: Home and away Elo averages differ significantly")
    print("    This suggests Elo calculation has home bias built in")

home_goals = features_df['home_goals_5'].mean()
away_goals = features_df['away_goals_5'].mean()
print(f"\nAverage Goals (last 5 games):")
print(f"  Home teams: {home_goals:.2f}")
print(f"  Away teams: {away_goals:.2f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF ISSUES FOUND")
print("=" * 80)

issues_found = []

if features_df['home_elo'].std() < 100:
    issues_found.append("Elo ratings too compressed")
if similar_strength / len(features_df) > 0.7:
    issues_found.append("Elo not differentiating teams enough")
if zero_form_home / len(features_df) > 0.3:
    issues_found.append("Form calculations may be broken")
if home_win_rate / away_win_rate > 1.5:
    issues_found.append("Training data class imbalance (favors home)")
if abs(home_elo_mean - away_elo_mean) > 20:
    issues_found.append("Elo calculation has built-in home bias")

if issues_found:
    for i, issue in enumerate(issues_found, 1):
        print(f"\n{i}. {issue}")
else:
    print("\nNo major issues detected beyond home advantage parameter")

