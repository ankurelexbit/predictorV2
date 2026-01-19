#!/usr/bin/env python3
"""
Debug feature calculations to understand why predictions are so similar.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from predict_live import LiveFeatureCalculator, get_upcoming_fixtures

warnings.filterwarnings('ignore')


def analyze_feature_variance(date_str='2026-01-18'):
    """Analyze feature variance across multiple matches."""

    print('='*80)
    print('FEATURE VARIANCE ANALYSIS')
    print('='*80)

    # Get fixtures
    fixtures_df = get_upcoming_fixtures(date_str)

    if fixtures_df.empty:
        print('No fixtures found')
        return

    print(f'\nFound {len(fixtures_df)} fixtures for {date_str}')

    # Calculate features for each match
    calculator = LiveFeatureCalculator()
    all_features = []
    match_info = []

    for idx, fixture in fixtures_df.head(10).iterrows():  # Analyze first 10 matches
        print(f'\n--- Match {idx+1}: {fixture["home_team_name"]} vs {fixture["away_team_name"]} ---')

        features = calculator.build_features_for_match(
            fixture['home_team_id'],
            fixture['away_team_id'],
            datetime.fromisoformat(fixture['date'])
        )

        if features:
            all_features.append(features)
            match_info.append(f"{fixture['home_team_name']} vs {fixture['away_team_name']}")

            # Print key features for this match
            print(f"  Home Elo: {features['home_elo']:.1f}, Away Elo: {features['away_elo']:.1f}")
            print(f"  Home Form (5): {features['home_form_5']:.3f}, Away Form (5): {features['away_form_5']:.3f}")
            print(f"  Home Goals (5): {features['home_goals_5']:.2f}, Away Goals (5): {features['away_goals_5']:.2f}")
            print(f"  Home xG (5): {features['home_xg_5']:.3f}, Away xG (5): {features['away_xg_5']:.3f}")
            print(f"  Home Shots (5): {features['home_shots_total_5']:.1f}, Away Shots (5): {features['away_shots_total_5']:.1f}")
            print(f"  Home Possession (5): {features['home_possession_pct_5']:.1f}%, Away Possession (5): {features['away_possession_pct_5']:.1f}%")

    if not all_features:
        print('\nNo features calculated')
        return

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    print('\n' + '='*80)
    print('FEATURE STATISTICS ACROSS ALL MATCHES')
    print('='*80)

    # Key features to analyze
    key_features = [
        'home_elo', 'away_elo', 'elo_diff',
        'home_form_5', 'away_form_5', 'form_diff_5',
        'home_goals_5', 'away_goals_5',
        'home_xg_5', 'away_xg_5',
        'home_shots_total_5', 'away_shots_total_5',
        'home_possession_pct_5', 'away_possession_pct_5',
        'home_dangerous_attacks_5', 'away_dangerous_attacks_5'
    ]

    print('\nFeature Variance (std dev):')
    for feature in key_features:
        if feature in features_df.columns:
            mean_val = features_df[feature].mean()
            std_val = features_df[feature].std()
            min_val = features_df[feature].min()
            max_val = features_df[feature].max()
            cv = (std_val / mean_val * 100) if mean_val != 0 else 0  # Coefficient of variation

            print(f'\n  {feature}:')
            print(f'    Mean: {mean_val:.3f}, Std: {std_val:.3f}')
            print(f'    Min: {min_val:.3f}, Max: {max_val:.3f}')
            print(f'    Coefficient of Variation: {cv:.1f}%')

            # Flag features with very low variance
            if std_val < 0.01 and mean_val > 0:
                print(f'    ⚠️  WARNING: Very low variance!')
            elif cv < 5 and mean_val > 0:
                print(f'    ⚠️  WARNING: Low coefficient of variation (<5%)')

    # Check for constant features
    print('\n' + '='*80)
    print('CONSTANT OR NEAR-CONSTANT FEATURES')
    print('='*80)

    constant_features = []
    for col in features_df.columns:
        if features_df[col].std() < 0.001:
            constant_features.append((col, features_df[col].mean()))

    if constant_features:
        print(f'\nFound {len(constant_features)} features with almost no variance:')
        for feature, value in constant_features[:20]:  # Show first 20
            print(f'  {feature}: {value:.4f}')
    else:
        print('\n✓ No constant features detected')

    # Compare with training data
    print('\n' + '='*80)
    print('COMPARISON WITH TRAINING DATA')
    print('='*80)

    try:
        training_features = pd.read_csv('data/processed/sportmonks_features.csv')

        print(f'\nTraining data shape: {training_features.shape}')
        print(f'Live prediction features shape: {features_df.shape}')

        # Compare key feature distributions
        print('\nFeature Distribution Comparison:')
        for feature in ['home_elo', 'away_elo', 'home_goals_5', 'away_goals_5', 'home_form_5', 'away_form_5']:
            if feature in training_features.columns and feature in features_df.columns:
                train_mean = training_features[feature].mean()
                live_mean = features_df[feature].mean()
                train_std = training_features[feature].std()
                live_std = features_df[feature].std()

                print(f'\n  {feature}:')
                print(f'    Training: mean={train_mean:.3f}, std={train_std:.3f}')
                print(f'    Live:     mean={live_mean:.3f}, std={live_std:.3f}')

                # Check if live data is within reasonable range of training data
                if abs(live_mean - train_mean) > 2 * train_std:
                    print(f'    ⚠️  WARNING: Live mean is outside training distribution!')
                if live_std < 0.1 * train_std:
                    print(f'    ⚠️  WARNING: Live variance is much lower than training!')

    except Exception as e:
        print(f'\nCould not load training data: {e}')

    # Check feature correlation
    print('\n' + '='*80)
    print('HIGHLY CORRELATED FEATURE PAIRS')
    print('='*80)

    # Look for features that are perfectly or nearly perfectly correlated
    corr_matrix = features_df.corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.99:  # Nearly perfect correlation
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))

    if high_corr_pairs:
        print(f'\nFound {len(high_corr_pairs)} highly correlated pairs (r > 0.99):')
        for feat1, feat2, corr in high_corr_pairs[:10]:
            print(f'  {feat1} <-> {feat2}: r={corr:.4f}')
    else:
        print('\n✓ No highly correlated features detected')


def compare_team_features(team1_id, team2_id, team1_name, team2_name):
    """Deep dive into features for two specific teams."""

    print('\n' + '='*80)
    print(f'DETAILED FEATURE COMPARISON: {team1_name} vs {team2_name}')
    print('='*80)

    calculator = LiveFeatureCalculator()

    # Get recent matches
    print(f'\nFetching recent matches for {team1_name}...')
    team1_matches = calculator.get_team_recent_matches(team1_id, limit=15)

    print(f'Fetching recent matches for {team2_name}...')
    team2_matches = calculator.get_team_recent_matches(team2_id, limit=15)

    print(f'\n{team1_name}: {len(team1_matches)} matches')
    print(f'{team2_name}: {len(team2_matches)} matches')

    # Analyze recent form
    if team1_matches:
        print(f'\n{team1_name} Recent Matches:')
        for i, match in enumerate(team1_matches[-5:], 1):  # Last 5 matches
            is_home = match['home_team_id'] == team1_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']
            location = 'H' if is_home else 'A'
            result = 'W' if team_score > opp_score else ('D' if team_score == opp_score else 'L')

            print(f'  {i}. ({location}) {match["home_team_name"]} {match["home_score"]}-{match["away_score"]} {match["away_team_name"]} - {result}')

            # Show some statistics
            stats = match.get('statistics', {})
            if stats:
                side = 'home' if is_home else 'away'
                shots = stats.get('shots_total', {}).get(side, 0)
                possession = stats.get('possession_pct', {}).get(side, 0)
                print(f'      Stats: {shots} shots, {possession}% possession')

    if team2_matches:
        print(f'\n{team2_name} Recent Matches:')
        for i, match in enumerate(team2_matches[-5:], 1):
            is_home = match['home_team_id'] == team2_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']
            location = 'H' if is_home else 'A'
            result = 'W' if team_score > opp_score else ('D' if team_score == opp_score else 'L')

            print(f'  {i}. ({location}) {match["home_team_name"]} {match["home_score"]}-{match["away_score"]} {match["away_team_name"]} - {result}')

            stats = match.get('statistics', {})
            if stats:
                side = 'home' if is_home else 'away'
                shots = stats.get('shots_total', {}).get(side, 0)
                possession = stats.get('possession_pct', {}).get(side, 0)
                print(f'      Stats: {shots} shots, {possession}% possession')

    # Calculate rolling statistics
    team1_matches.sort(key=lambda x: x['date'])
    team2_matches.sort(key=lambda x: x['date'])

    team1_stats_5 = calculator.calculate_rolling_stats(team1_matches, team1_id, window=5)
    team2_stats_5 = calculator.calculate_rolling_stats(team2_matches, team2_id, window=5)

    print(f'\n{team1_name} Rolling Stats (last 5):')
    print(f'  Form: {team1_stats_5.get("form", 0):.3f}')
    print(f'  Goals avg: {team1_stats_5.get("goals_avg", 0):.2f}')
    print(f'  xG avg: {team1_stats_5.get("xg_avg", 0):.3f}')
    print(f'  Shots avg: {team1_stats_5.get("shots_total_avg", 0):.1f}')
    print(f'  Possession avg: {team1_stats_5.get("possession_pct_avg", 0):.1f}%')

    print(f'\n{team2_name} Rolling Stats (last 5):')
    print(f'  Form: {team2_stats_5.get("form", 0):.3f}')
    print(f'  Goals avg: {team2_stats_5.get("goals_avg", 0):.2f}')
    print(f'  xG avg: {team2_stats_5.get("xg_avg", 0):.3f}')
    print(f'  Shots avg: {team2_stats_5.get("shots_total_avg", 0):.1f}')
    print(f'  Possession avg: {team2_stats_5.get("possession_pct_avg", 0):.1f}%')


if __name__ == '__main__':
    # Overall variance analysis
    analyze_feature_variance('2026-01-18')

    # Deep dive into specific match
    print('\n\n' + '='*80)
    print('DEEP DIVE: SC Heerenveen vs FC Groningen')
    print('='*80)
    compare_team_features(1053, 2345, 'SC Heerenveen', 'FC Groningen')
