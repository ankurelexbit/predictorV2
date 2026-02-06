#!/usr/bin/env python3
"""
Deep Analysis of Draw Patterns
==============================

Goal: Find if there are ANY conditions where draws can be predicted at >50% WR

Analyze:
1. League-specific draw rates (some leagues like Serie A have higher draws)
2. Match context (late season, mid-table teams, etc.)
3. Combined filter approach (multiple conditions must be met)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"
RANDOM_STATE = 42

# Known league IDs (from common football data)
LEAGUE_NAMES = {
    8: 'Premier League',
    564: 'La Liga',
    384: 'Serie A',
    82: 'Bundesliga',
    301: 'Ligue 1',
    501: 'Primeira Liga',
    271: 'Eredivisie'
}

META_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target'
]


def load_data():
    """Load and prepare data."""
    df = pd.read_csv(DATA_PATH)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    # Create target
    result_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(result_map)

    # Split chronologically
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, test_df


def analyze_league_draw_rates(test_df):
    """Analyze draw rates by league."""
    print("\n" + "="*80)
    print("DRAW RATES BY LEAGUE")
    print("="*80)

    league_stats = []
    for league_id in test_df['league_id'].unique():
        league_data = test_df[test_df['league_id'] == league_id]
        total = len(league_data)
        draws = (league_data['result'] == 'D').sum()
        draw_rate = draws / total * 100 if total > 0 else 0
        name = LEAGUE_NAMES.get(league_id, f'League {league_id}')

        league_stats.append({
            'league_id': league_id,
            'name': name,
            'total': total,
            'draws': draws,
            'draw_rate': draw_rate
        })

    league_stats = sorted(league_stats, key=lambda x: x['draw_rate'], reverse=True)

    print(f"\n{'League':<25} {'Matches':<10} {'Draws':<10} {'Draw Rate':<10}")
    print("-" * 60)
    for s in league_stats[:15]:
        print(f"{s['name'][:24]:<25} {s['total']:<10} {s['draws']:<10} {s['draw_rate']:.1f}%")

    return league_stats


def analyze_position_patterns(test_df):
    """Analyze draw rates by team positions."""
    print("\n" + "="*80)
    print("DRAW RATES BY POSITION CONTEXT")
    print("="*80)

    # Both teams mid-table (positions 7-14)
    if 'home_league_position' in test_df.columns and 'away_league_position' in test_df.columns:
        mid_table_mask = (
            (test_df['home_league_position'] >= 7) & (test_df['home_league_position'] <= 14) &
            (test_df['away_league_position'] >= 7) & (test_df['away_league_position'] <= 14)
        )
        mid_data = test_df[mid_table_mask]
        mid_draw_rate = (mid_data['result'] == 'D').mean() * 100 if len(mid_data) > 0 else 0
        print(f"\nBoth teams mid-table (7-14): {len(mid_data)} matches, Draw rate: {mid_draw_rate:.1f}%")

        # Very close positions
        close_pos_mask = abs(test_df['home_league_position'] - test_df['away_league_position']) <= 2
        close_data = test_df[close_pos_mask]
        close_draw_rate = (close_data['result'] == 'D').mean() * 100 if len(close_data) > 0 else 0
        print(f"Positions within 2: {len(close_data)} matches, Draw rate: {close_draw_rate:.1f}%")

        # Top vs Top
        top_vs_top = (test_df['home_league_position'] <= 6) & (test_df['away_league_position'] <= 6)
        top_data = test_df[top_vs_top]
        top_draw_rate = (top_data['result'] == 'D').mean() * 100 if len(top_data) > 0 else 0
        print(f"Top 6 vs Top 6: {len(top_data)} matches, Draw rate: {top_draw_rate:.1f}%")

        # Bottom vs Bottom
        bottom_vs_bottom = (test_df['home_league_position'] >= 15) & (test_df['away_league_position'] >= 15)
        bottom_data = test_df[bottom_vs_bottom]
        bottom_draw_rate = (bottom_data['result'] == 'D').mean() * 100 if len(bottom_data) > 0 else 0
        print(f"Bottom 6 vs Bottom 6: {len(bottom_data)} matches, Draw rate: {bottom_draw_rate:.1f}%")


def analyze_elo_patterns(test_df):
    """Analyze draw rates by Elo closeness."""
    print("\n" + "="*80)
    print("DRAW RATES BY ELO CLOSENESS")
    print("="*80)

    if 'home_elo' in test_df.columns and 'away_elo' in test_df.columns:
        test_df = test_df.copy()
        test_df['elo_diff'] = abs(test_df['home_elo'] - test_df['away_elo'])

        bands = [(0, 25), (0, 50), (0, 75), (0, 100), (100, 200), (200, 500)]
        for low, high in bands:
            mask = (test_df['elo_diff'] >= low) & (test_df['elo_diff'] < high)
            data = test_df[mask]
            draw_rate = (data['result'] == 'D').mean() * 100 if len(data) > 0 else 0
            print(f"Elo diff {low}-{high}: {len(data)} matches, Draw rate: {draw_rate:.1f}%")


def analyze_score_patterns(test_df):
    """Analyze draw rates by low-scoring matches."""
    print("\n" + "="*80)
    print("DRAW RATES BY SCORING CONTEXT")
    print("="*80)

    # Use derived xG as a proxy for expected scoring
    if 'home_derived_xg_5' in test_df.columns and 'away_derived_xg_5' in test_df.columns:
        test_df = test_df.copy()

        # Low xG matches (defensive teams)
        low_xg_mask = (test_df['home_derived_xg_5'] < 1.0) & (test_df['away_derived_xg_5'] < 1.0)
        low_data = test_df[low_xg_mask]
        low_draw_rate = (low_data['result'] == 'D').mean() * 100 if len(low_data) > 0 else 0
        print(f"Both teams low xG (<1.0): {len(low_data)} matches, Draw rate: {low_draw_rate:.1f}%")

        # Similar xG teams
        xg_diff = abs(test_df['home_derived_xg_5'] - test_df['away_derived_xg_5'])
        similar_xg = xg_diff < 0.3
        similar_data = test_df[similar_xg]
        similar_draw_rate = (similar_data['result'] == 'D').mean() * 100 if len(similar_data) > 0 else 0
        print(f"Similar xG (diff < 0.3): {len(similar_data)} matches, Draw rate: {similar_draw_rate:.1f}%")


def find_high_draw_conditions(train_df, test_df):
    """Find conditions that maximize draw prediction accuracy."""
    print("\n" + "="*80)
    print("SEARCHING FOR HIGH-DRAW CONDITIONS")
    print("="*80)

    # Train a binary draw detector
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    X_train = train_df[feature_cols].values
    y_train_draw = (train_df['target'] == 1).astype(int).values

    print("\nTraining binary draw detector...")
    model = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.02,
        auto_class_weights='Balanced',
        random_seed=RANDOM_STATE,
        verbose=False
    )
    model.fit(X_train, y_train_draw)

    # Predict on test
    X_test = test_df[feature_cols].values
    y_test_draw = (test_df['target'] == 1).astype(int).values

    draw_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_draw, draw_probs)
    print(f"Draw detector AUC: {auc:.3f}")

    # Analyze by confidence bands
    print("\nDraw Win Rate by Model Confidence:")
    print("-" * 60)

    conf_bands = [
        (0.30, 0.35),
        (0.35, 0.40),
        (0.40, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 1.0)
    ]

    for low, high in conf_bands:
        mask = (draw_probs >= low) & (draw_probs < high)
        if mask.sum() == 0:
            continue

        correct = y_test_draw[mask].sum()
        total = mask.sum()
        wr = correct / total * 100
        print(f"  Confidence {low:.0%}-{high:.0%}: {total} predictions, WR: {wr:.1f}%")

    return model, feature_cols


def combined_filter_analysis(train_df, test_df, draw_model, feature_cols):
    """Analyze combining multiple filters."""
    print("\n" + "="*80)
    print("COMBINED FILTER ANALYSIS")
    print("="*80)

    X_test = test_df[feature_cols].values
    draw_probs = draw_model.predict_proba(X_test)[:, 1]

    # Create condition masks
    conditions = {}

    # Elo closeness
    if 'home_elo' in test_df.columns and 'away_elo' in test_df.columns:
        elo_diff = abs(test_df['home_elo'].values - test_df['away_elo'].values)
        conditions['elo_close'] = elo_diff < 50

    # Position closeness
    if 'home_league_position' in test_df.columns and 'away_league_position' in test_df.columns:
        pos_diff = abs(test_df['home_league_position'].values - test_df['away_league_position'].values)
        conditions['pos_close'] = pos_diff <= 3

    # Mid-table
    if 'home_league_position' in test_df.columns and 'away_league_position' in test_df.columns:
        conditions['mid_table'] = (
            (test_df['home_league_position'].values >= 7) &
            (test_df['home_league_position'].values <= 14) &
            (test_df['away_league_position'].values >= 7) &
            (test_df['away_league_position'].values <= 14)
        )

    # High draw probability
    conditions['high_prob'] = draw_probs >= 0.40
    conditions['very_high_prob'] = draw_probs >= 0.50

    y_test_draw = (test_df['target'] == 1).astype(int).values

    # Test various combinations
    print("\nTesting filter combinations:")
    print("-" * 70)

    combinations = [
        (['high_prob'], 'Model prob >= 40%'),
        (['very_high_prob'], 'Model prob >= 50%'),
        (['elo_close'], 'Elo diff < 50'),
        (['pos_close'], 'Position diff <= 3'),
        (['mid_table'], 'Both mid-table'),
        (['high_prob', 'elo_close'], 'Prob >= 40% + Elo close'),
        (['high_prob', 'pos_close'], 'Prob >= 40% + Position close'),
        (['high_prob', 'mid_table'], 'Prob >= 40% + Mid-table'),
        (['high_prob', 'elo_close', 'pos_close'], 'Prob >= 40% + Elo + Position close'),
        (['very_high_prob', 'elo_close'], 'Prob >= 50% + Elo close'),
        (['very_high_prob', 'pos_close'], 'Prob >= 50% + Position close'),
        (['very_high_prob', 'elo_close', 'pos_close'], 'Prob >= 50% + Elo + Position close'),
    ]

    best_wr = 0
    best_combo = None

    for cond_names, desc in combinations:
        # Check all conditions available
        if not all(c in conditions for c in cond_names):
            continue

        # Combine masks
        mask = np.ones(len(test_df), dtype=bool)
        for c in cond_names:
            mask = mask & conditions[c]

        if mask.sum() == 0:
            continue

        correct = y_test_draw[mask].sum()
        total = mask.sum()
        wr = correct / total * 100

        print(f"  {desc:<45} {total:>5} predictions, WR: {wr:.1f}%")

        if wr > best_wr and total >= 10:
            best_wr = wr
            best_combo = (cond_names, desc, total, wr)

    if best_combo:
        print(f"\nBest combination found:")
        print(f"  {best_combo[1]}: {best_combo[2]} predictions @ {best_combo[3]:.1f}% WR")


def main():
    print("Loading data...")
    train_df, test_df = load_data()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Test period: {test_df['match_date'].min()} to {test_df['match_date'].max()}")

    # Base draw rate
    base_draw_rate = (test_df['result'] == 'D').mean() * 100
    print(f"\nBase draw rate in test set: {base_draw_rate:.1f}%")

    # Analyze patterns
    analyze_league_draw_rates(test_df)
    analyze_position_patterns(test_df)
    analyze_elo_patterns(test_df)
    analyze_score_patterns(test_df)

    # Find high-draw conditions
    draw_model, feature_cols = find_high_draw_conditions(train_df, test_df)

    # Combined filter analysis
    combined_filter_analysis(train_df, test_df, draw_model, feature_cols)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBase draw rate: {base_draw_rate:.1f}%")
    print("\nKey insight: Draws are fundamentally hard to predict because:")
    print("1. Low base rate (~25%) means random guess is already 25%")
    print("2. Even perfectly balanced matches don't dramatically increase draw rate")
    print("3. Football draws are influenced by in-game events (red cards, late goals)")
    print("\nRecommendation: Consider if draw predictions add value to the product")


if __name__ == '__main__':
    main()
