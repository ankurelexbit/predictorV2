"""
Feature Distribution Analysis by League.

Analyzes how features vary across different leagues to identify:
- League-specific patterns
- Feature distributions
- Statistical differences
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# League mapping
LEAGUE_NAMES = {
    8: 'Premier League',
    82: 'Bundesliga',
    301: 'Serie A',
    384: 'La Liga',
    564: 'Ligue 1'
}

def analyze_feature_by_league(df, feature_name, league_names):
    """Analyze a single feature across leagues."""
    print(f"\n{feature_name}:")
    print("-" * 80)
    
    for league_id in sorted(df['league_id'].unique()):
        league_data = df[df['league_id'] == league_id][feature_name]
        league_data_clean = league_data.dropna()
        
        if len(league_data_clean) > 0:
            print(f"  {league_names.get(league_id, f'League {league_id}'):20s}: "
                  f"mean={league_data_clean.mean():6.2f}, "
                  f"std={league_data_clean.std():6.2f}, "
                  f"min={league_data_clean.min():6.2f}, "
                  f"max={league_data_clean.max():6.2f}, "
                  f"n={len(league_data_clean):5d}")

def main():
    """Run comprehensive feature distribution analysis."""
    print("=" * 80)
    print("FEATURE DISTRIBUTION ANALYSIS BY LEAGUE")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading training data...")
    df = pd.read_csv('data/training_data.csv')
    print(f"✅ Loaded {len(df)} fixtures")
    
    # Basic league info
    print("\n" + "=" * 80)
    print("1. LEAGUE OVERVIEW")
    print("=" * 80)
    
    for league_id in sorted(df['league_id'].unique()):
        league_fixtures = df[df['league_id'] == league_id]
        league_name = LEAGUE_NAMES.get(league_id, f'League {league_id}')
        
        # Result distribution
        results = league_fixtures['result'].value_counts()
        total = len(league_fixtures)
        
        print(f"\n{league_name} (League {league_id}):")
        print(f"  Total fixtures: {total:,}")
        print(f"  Home wins: {results.get('H', 0):,} ({results.get('H', 0)/total*100:.1f}%)")
        print(f"  Draws: {results.get('D', 0):,} ({results.get('D', 0)/total*100:.1f}%)")
        print(f"  Away wins: {results.get('A', 0):,} ({results.get('A', 0)/total*100:.1f}%)")
        print(f"  Avg goals/match: {(league_fixtures['home_score'].mean() + league_fixtures['away_score'].mean()):.2f}")
    
    # Pillar 1 Features Analysis
    print("\n" + "=" * 80)
    print("2. PILLAR 1 FEATURES - FUNDAMENTALS")
    print("=" * 80)
    
    pillar1_features = [
        'home_elo',
        'away_elo',
        'home_league_position',
        'home_points_last_5',
        'away_points_last_5',
        'home_goals_scored_last_5',
        'away_goals_scored_last_5',
        'home_goals_conceded_last_5',
        'away_goals_conceded_last_5',
    ]
    
    for feature in pillar1_features:
        if feature in df.columns:
            analyze_feature_by_league(df, feature, LEAGUE_NAMES)
    
    # Pillar 2 Features Analysis
    print("\n" + "=" * 80)
    print("3. PILLAR 2 FEATURES - MODERN ANALYTICS")
    print("=" * 80)
    
    pillar2_features = [
        'home_derived_xg_per_match_5',
        'away_derived_xg_per_match_5',
        'home_shots_per_match_5',
        'away_shots_per_match_5',
        'home_possession_pct_5',
        'away_possession_pct_5',
        'home_tackles_per_90',
        'away_tackles_per_90',
        'home_attacks_per_match_5',
        'away_attacks_per_match_5',
    ]
    
    for feature in pillar2_features:
        if feature in df.columns:
            analyze_feature_by_league(df, feature, LEAGUE_NAMES)
    
    # Pillar 3 Features Analysis
    print("\n" + "=" * 80)
    print("4. PILLAR 3 FEATURES - HIDDEN EDGES")
    print("=" * 80)
    
    pillar3_features = [
        'home_win_streak',
        'away_win_streak',
        'home_unbeaten_streak',
        'away_unbeaten_streak',
        'home_weighted_form_5',
        'away_weighted_form_5',
    ]
    
    for feature in pillar3_features:
        if feature in df.columns:
            analyze_feature_by_league(df, feature, LEAGUE_NAMES)
    
    # League-specific insights
    print("\n" + "=" * 80)
    print("5. LEAGUE-SPECIFIC INSIGHTS")
    print("=" * 80)
    
    for league_id in sorted(df['league_id'].unique()):
        league_data = df[df['league_id'] == league_id]
        league_name = LEAGUE_NAMES.get(league_id, f'League {league_id}')
        
        print(f"\n{league_name}:")
        
        # Home advantage
        home_wins = (league_data['result'] == 'H').sum()
        away_wins = (league_data['result'] == 'A').sum()
        home_advantage = (home_wins / (home_wins + away_wins)) * 100 if (home_wins + away_wins) > 0 else 0
        print(f"  Home advantage: {home_advantage:.1f}% (home wins / (home + away wins))")
        
        # Average Elo
        if 'home_elo' in league_data.columns:
            avg_elo = (league_data['home_elo'].mean() + league_data['away_elo'].mean()) / 2
            print(f"  Average Elo: {avg_elo:.1f}")
        
        # Competitiveness (Elo std dev)
        if 'home_elo' in league_data.columns:
            elo_std = league_data['home_elo'].std()
            print(f"  Competitiveness (Elo std): {elo_std:.1f} (lower = more competitive)")
        
        # Goals per match
        avg_goals = league_data['home_score'].mean() + league_data['away_score'].mean()
        print(f"  Goals per match: {avg_goals:.2f}")
        
        # Draw rate
        draws = (league_data['result'] == 'D').sum()
        draw_rate = (draws / len(league_data)) * 100
        print(f"  Draw rate: {draw_rate:.1f}%")
    
    # Statistical comparison
    print("\n" + "=" * 80)
    print("6. STATISTICAL COMPARISON ACROSS LEAGUES")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    
    for league_id in sorted(df['league_id'].unique()):
        league_data = df[df['league_id'] == league_id]
        league_name = LEAGUE_NAMES.get(league_id, f'League {league_id}')
        
        row = {
            'League': league_name,
            'Fixtures': len(league_data),
            'Home Win %': (league_data['result'] == 'H').sum() / len(league_data) * 100,
            'Draw %': (league_data['result'] == 'D').sum() / len(league_data) * 100,
            'Away Win %': (league_data['result'] == 'A').sum() / len(league_data) * 100,
            'Goals/Match': league_data['home_score'].mean() + league_data['away_score'].mean(),
            'Avg Elo': (league_data['home_elo'].mean() + league_data['away_elo'].mean()) / 2 if 'home_elo' in league_data.columns else 0,
            'Elo Std': league_data['home_elo'].std() if 'home_elo' in league_data.columns else 0,
        }
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Key findings
    print("\n" + "=" * 80)
    print("7. KEY FINDINGS")
    print("=" * 80)
    
    # Most competitive league (lowest Elo std)
    most_competitive = comparison_df.loc[comparison_df['Elo Std'].idxmin()]
    print(f"\n✅ Most Competitive: {most_competitive['League']} (Elo std: {most_competitive['Elo Std']:.1f})")
    
    # Highest scoring league
    highest_scoring = comparison_df.loc[comparison_df['Goals/Match'].idxmax()]
    print(f"✅ Highest Scoring: {highest_scoring['League']} ({highest_scoring['Goals/Match']:.2f} goals/match)")
    
    # Strongest home advantage
    strongest_home = comparison_df.loc[comparison_df['Home Win %'].idxmax()]
    print(f"✅ Strongest Home Advantage: {strongest_home['League']} ({strongest_home['Home Win %']:.1f}% home wins)")
    
    # Highest draw rate
    most_draws = comparison_df.loc[comparison_df['Draw %'].idxmax()]
    print(f"✅ Most Draws: {most_draws['League']} ({most_draws['Draw %']:.1f}% draws)")
    
    # Highest average quality (Elo)
    highest_quality = comparison_df.loc[comparison_df['Avg Elo'].idxmax()]
    print(f"✅ Highest Quality: {highest_quality['League']} (Avg Elo: {highest_quality['Avg Elo']:.1f})")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
