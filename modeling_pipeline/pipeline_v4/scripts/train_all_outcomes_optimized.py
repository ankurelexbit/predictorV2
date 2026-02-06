"""
Optimized Model for ALL Outcomes (Home, Draw, Away)

Key optimizations:
1. Draw-specific features (match balance, closeness indicators)
2. League-specific draw rates as prior
3. Historical team draw tendencies
4. Separate calibration and thresholds per outcome
5. Context-aware betting (find sweet spots for each outcome)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and merge feature sets."""
    print("Loading data...")

    base_df = pd.read_csv('data/training_data_with_market.csv')
    base_df['match_date'] = pd.to_datetime(base_df['match_date'])

    lineup_df = pd.read_csv('data/lineup_features_v2.csv')
    lineup_df['match_date'] = pd.to_datetime(lineup_df['match_date'])

    lineup_cols = [c for c in lineup_df.columns
                   if c not in ['match_date', 'home_team_id', 'away_team_id']]

    df = base_df.merge(lineup_df[lineup_cols], on='fixture_id', how='inner')
    df = df[df['match_date'] >= '2019-01-01'].copy()
    df = df.sort_values('match_date').reset_index(drop=True)

    print(f"  Loaded {len(df):,} fixtures")
    return df


def calculate_team_draw_rates(df):
    """Calculate historical draw rates for each team."""
    print("  Calculating team draw rates...")

    team_draws = defaultdict(list)
    team_home_draws = defaultdict(list)
    team_away_draws = defaultdict(list)

    for _, row in df.iterrows():
        is_draw = 1 if row['result'] == 'D' else 0
        home_id = row['home_team_id']
        away_id = row['away_team_id']

        team_draws[home_id].append(is_draw)
        team_draws[away_id].append(is_draw)
        team_home_draws[home_id].append(is_draw)
        team_away_draws[away_id].append(is_draw)

    # Calculate rates (using last 20 matches)
    team_draw_rate = {}
    team_home_draw_rate = {}
    team_away_draw_rate = {}

    for team_id in team_draws:
        recent = team_draws[team_id][-20:]
        team_draw_rate[team_id] = np.mean(recent) if recent else 0.25

        recent_home = team_home_draws[team_id][-10:]
        team_home_draw_rate[team_id] = np.mean(recent_home) if recent_home else 0.25

        recent_away = team_away_draws[team_id][-10:]
        team_away_draw_rate[team_id] = np.mean(recent_away) if recent_away else 0.25

    return team_draw_rate, team_home_draw_rate, team_away_draw_rate


def calculate_league_draw_rates(df):
    """Calculate draw rates by league."""
    print("  Calculating league draw rates...")

    league_draws = df.groupby('league_id').apply(
        lambda x: (x['result'] == 'D').mean()
    ).to_dict()

    return league_draws


def engineer_draw_features(df):
    """Add features specifically designed to predict draws."""
    print("  Engineering draw-specific features...")

    df = df.copy()
    new_features = []

    # 1. Elo closeness (higher = more likely draw)
    if 'elo_diff' in df.columns:
        df['elo_closeness'] = 1 / (1 + abs(df['elo_diff']) / 50)
        new_features.append('elo_closeness')

        df['elo_diff_abs'] = abs(df['elo_diff'])
        new_features.append('elo_diff_abs')

        # Very close matches
        df['elo_very_close'] = (abs(df['elo_diff']) < 50).astype(int)
        new_features.append('elo_very_close')

    # 2. Position closeness
    if 'position_diff' in df.columns:
        df['position_closeness'] = 1 / (1 + abs(df['position_diff']))
        new_features.append('position_closeness')

        df['position_diff_abs'] = abs(df['position_diff'])
        new_features.append('position_diff_abs')

        # Same half of table
        df['position_very_close'] = (abs(df['position_diff']) <= 3).astype(int)
        new_features.append('position_very_close')

    # 3. Points closeness
    if 'points_diff' in df.columns:
        df['points_closeness'] = 1 / (1 + abs(df['points_diff']) / 5)
        new_features.append('points_closeness')

        df['points_diff_abs'] = abs(df['points_diff'])
        new_features.append('points_diff_abs')

    # 4. Form similarity (both teams in similar form = draw likely)
    form_cols_home = [c for c in df.columns if c.startswith('home_') and 'win_rate' in c.lower()]
    form_cols_away = [c for c in df.columns if c.startswith('away_') and 'win_rate' in c.lower()]

    if form_cols_home and form_cols_away:
        for h_col in form_cols_home[:2]:
            a_col = h_col.replace('home_', 'away_')
            if a_col in df.columns:
                diff_col = f'form_diff_{h_col.replace("home_", "")}'
                df[diff_col] = abs(df[h_col] - df[a_col])
                new_features.append(diff_col)

    # 5. Lineup quality similarity
    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_closeness'] = 1 / (1 + abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating']) * 10)
        new_features.append('lineup_closeness')

        df['lineup_diff_abs'] = abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating'])
        new_features.append('lineup_diff_abs')

    # 6. Combined balance score (higher = more balanced = more likely draw)
    balance_features = ['elo_closeness', 'position_closeness', 'lineup_closeness']
    available_balance = [f for f in balance_features if f in df.columns]
    if available_balance:
        df['match_balance_score'] = df[available_balance].mean(axis=1)
        new_features.append('match_balance_score')

    # 7. Goal scoring patterns (low scoring = more draws)
    goal_cols = [c for c in df.columns if 'goals' in c.lower() or 'xg' in c.lower()]
    for col in goal_cols[:4]:
        if col in df.columns and 'home_' in col:
            away_col = col.replace('home_', 'away_')
            if away_col in df.columns:
                avg_col = f'avg_{col.replace("home_", "")}'
                df[avg_col] = (df[col] + df[away_col]) / 2
                new_features.append(avg_col)

    # 8. Defensive strength (both teams defensive = draw)
    if 'home_goals_conceded_5' in df.columns and 'away_goals_conceded_5' in df.columns:
        df['combined_defensive'] = df['home_goals_conceded_5'] + df['away_goals_conceded_5']
        new_features.append('combined_defensive')

        # Low scoring matches
        df['both_defensive'] = ((df['home_goals_conceded_5'] < 1.2) &
                                (df['away_goals_conceded_5'] < 1.2)).astype(int)
        new_features.append('both_defensive')

    print(f"    Added {len(new_features)} draw-specific features")
    return df, new_features


def engineer_away_features(df):
    """Add features specifically designed to predict away wins."""
    print("  Engineering away-specific features...")

    df = df.copy()
    new_features = []

    # 1. Strong away team indicators
    if 'away_elo' in df.columns and 'home_elo' in df.columns:
        df['away_elo_advantage'] = df['away_elo'] - df['home_elo']
        new_features.append('away_elo_advantage')

        # Away team significantly stronger
        df['away_much_stronger'] = (df['away_elo'] - df['home_elo'] > 100).astype(int)
        new_features.append('away_much_stronger')

    # 2. Away form vs Home form
    if 'away_points_5' in df.columns and 'home_points_5' in df.columns:
        df['away_form_advantage'] = df['away_points_5'] - df['home_points_5']
        new_features.append('away_form_advantage')

    # 3. Weak home team indicators
    if 'home_league_position' in df.columns:
        df['home_bottom_half'] = (df['home_league_position'] > 10).astype(int)
        new_features.append('home_bottom_half')

    # 4. Away team attacking strength
    if 'away_goals_scored_5' in df.columns:
        df['away_high_scoring'] = (df['away_goals_scored_5'] > 1.5).astype(int)
        new_features.append('away_high_scoring')

    print(f"    Added {len(new_features)} away-specific features")
    return df, new_features


def get_base_features(df):
    """Get base statistical features (no odds)."""
    exclude_patterns = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result',
        'odds', 'implied', 'bookmaker', 'sharp', 'soft', 'disagreement',
        'market_home', 'market_away', 'market_draw', 'market_overround',
        'ah_', 'ou_', 'num_bookmakers', 'over_2_5', 'under_2_5',
    ]

    def should_exclude(col):
        return any(p.lower() in col.lower() for p in exclude_patterns)

    feature_cols = [c for c in df.columns
                    if not should_exclude(c)
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return feature_cols


def engineer_standard_features(df, feature_cols):
    """Add standard engineered features."""
    df = df.copy()
    new_features = []

    if 'elo_diff' in df.columns:
        df['elo_diff_sq'] = df['elo_diff'] ** 2 * np.sign(df['elo_diff'])
        new_features.append('elo_diff_sq')

        if 'lineup_rating_diff' in df.columns:
            df['elo_x_lineup'] = df['elo_diff'] * df['lineup_rating_diff']
            new_features.append('elo_x_lineup')

        if 'position_diff' in df.columns:
            df['elo_x_position'] = df['elo_diff'] * df['position_diff']
            new_features.append('elo_x_position')

    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
        new_features.append('elo_ratio')

    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_quality_gap'] = abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating'])
        new_features.append('lineup_quality_gap')

    return df, feature_cols + new_features


def train_model(X_train, y_train, X_val, y_val):
    """Train CatBoost model."""
    print("\nTraining CatBoost model...")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.025,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=75,
        eval_metric='MultiClass'
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    # Calibrate per outcome
    print("\nCalibrating probabilities...")
    raw_probs = model.predict_proba(X_val)

    calibrators = {}
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
        calibrators[outcome].fit(raw_probs[:, idx], (y_val == idx).astype(int))

    return model, calibrators


def apply_calibration(probs, calibrators):
    """Apply isotonic calibration."""
    cal_probs = np.zeros_like(probs)
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(probs[:, idx])

    row_sums = cal_probs.sum(axis=1, keepdims=True)
    cal_probs = cal_probs / np.where(row_sums == 0, 1, row_sums)
    return cal_probs


def find_optimal_thresholds(df_val, cal_probs_val, outcome_name, outcome_idx):
    """Grid search for optimal thresholds for a specific outcome."""
    result_map = {'A': 0, 'D': 1, 'H': 2}

    best_config = None
    best_score = -999

    if outcome_name == 'home':
        prob_range = [0.48, 0.50, 0.52, 0.54]
        edge_range = [0.04, 0.05, 0.06, 0.07]
        min_odds_range = [1.40, 1.50]
        max_odds_range = [2.80, 3.20, 3.50]
    elif outcome_name == 'away':
        prob_range = [0.42, 0.45, 0.48, 0.50]
        edge_range = [0.06, 0.08, 0.10]
        min_odds_range = [2.00, 2.20, 2.50]
        max_odds_range = [4.00, 5.00]
    else:  # draw
        prob_range = [0.28, 0.30, 0.32]
        edge_range = [0.04, 0.06, 0.08]
        min_odds_range = [3.00, 3.20, 3.40]
        max_odds_range = [4.00, 4.50]

    for min_prob in prob_range:
        for min_edge in edge_range:
            for min_odds in min_odds_range:
                for max_odds in max_odds_range:
                    # Simulate bets
                    wins = 0
                    total = 0
                    pnl = 0

                    odds_col = f'{outcome_name}_best_odds'

                    for i in range(len(df_val)):
                        row = df_val.iloc[i]
                        prob = cal_probs_val[i, outcome_idx]
                        odds = row.get(odds_col, 0)

                        if not odds or odds < min_odds or odds > max_odds:
                            continue

                        implied = 1 / odds if odds > 1 else 0.33
                        edge = prob - implied

                        if prob >= min_prob and edge >= min_edge:
                            actual = result_map.get(row['result'], -1)
                            if actual == -1:
                                continue

                            total += 1
                            won = 1 if actual == outcome_idx else 0
                            wins += won
                            pnl += (odds - 1) if won else -1

                    if total >= 20:
                        wr = wins / total
                        roi = pnl / total
                        # Score: prioritize WR, then ROI
                        score = wr * 100 + roi * 10

                        if score > best_score:
                            best_score = score
                            best_config = {
                                'min_prob': min_prob,
                                'min_edge': min_edge,
                                'min_odds': min_odds,
                                'max_odds': max_odds,
                                'val_bets': total,
                                'val_wr': wr * 100,
                                'val_roi': roi * 100
                            }

    return best_config


def backtest_all_outcomes(df_test, cal_probs, params):
    """Backtest with per-outcome parameters."""
    results = []
    result_map = {'A': 0, 'D': 1, 'H': 2}

    for i in range(len(df_test)):
        row = df_test.iloc[i]

        p_away, p_draw, p_home = cal_probs[i]

        h_odds = row.get('home_best_odds', 0)
        d_odds = row.get('draw_best_odds', 0)
        a_odds = row.get('away_best_odds', 0)

        if not h_odds or not d_odds or not a_odds:
            continue

        h_implied = 1 / h_odds if h_odds > 1 else 0.5
        d_implied = 1 / d_odds if d_odds > 1 else 0.25
        a_implied = 1 / a_odds if a_odds > 1 else 0.35

        actual = result_map.get(row['result'], -1)
        if actual == -1:
            continue

        candidates = []

        # Home
        hp = params.get('home', {})
        if hp.get('enabled', False):
            h_edge = p_home - h_implied
            if (hp['min_odds'] <= h_odds <= hp['max_odds'] and
                p_home >= hp['min_prob'] and h_edge >= hp['min_edge']):
                candidates.append({
                    'outcome': 'Home', 'idx': 2,
                    'odds': h_odds, 'prob': p_home, 'edge': h_edge
                })

        # Away
        ap = params.get('away', {})
        if ap.get('enabled', False):
            a_edge = p_away - a_implied
            if (ap['min_odds'] <= a_odds <= ap['max_odds'] and
                p_away >= ap['min_prob'] and a_edge >= ap['min_edge']):
                candidates.append({
                    'outcome': 'Away', 'idx': 0,
                    'odds': a_odds, 'prob': p_away, 'edge': a_edge
                })

        # Draw
        dp = params.get('draw', {})
        if dp.get('enabled', False):
            d_edge = p_draw - d_implied
            if (dp['min_odds'] <= d_odds <= dp['max_odds'] and
                p_draw >= dp['min_prob'] and d_edge >= dp['min_edge']):
                candidates.append({
                    'outcome': 'Draw', 'idx': 1,
                    'odds': d_odds, 'prob': p_draw, 'edge': d_edge
                })

        if not candidates:
            continue

        # Select best by edge
        best = max(candidates, key=lambda x: x['edge'])
        won = 1 if actual == best['idx'] else 0
        pnl = (best['odds'] - 1) if won else -1

        results.append({
            'date': row['match_date'],
            'outcome': best['outcome'],
            'odds': best['odds'],
            'prob': best['prob'],
            'edge': best['edge'],
            'won': won,
            'pnl': pnl,
            'month': pd.to_datetime(row['match_date']).strftime('%Y-%m')
        })

    return pd.DataFrame(results)


def analyze_results(results_df, title):
    """Analyze results."""
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

    if len(results_df) == 0:
        print("No bets placed!")
        return None

    total = len(results_df)
    wins = results_df['won'].sum()
    wr = wins / total * 100
    pnl = results_df['pnl'].sum()
    roi = pnl / total * 100

    print(f"\nOVERALL: {total} bets, {wr:.1f}% WR, {pnl:.2f} PnL, {roi:.1f}% ROI")

    # By outcome
    print("\nBY OUTCOME:")
    outcome_stats = {}
    for outcome in ['Home', 'Away', 'Draw']:
        sub = results_df[results_df['outcome'] == outcome]
        if len(sub) > 0:
            o_wr = sub['won'].mean() * 100
            o_roi = sub['pnl'].sum() / len(sub) * 100
            o_avg_odds = sub['odds'].mean()
            outcome_stats[outcome] = {'bets': len(sub), 'wr': o_wr, 'roi': o_roi}
            print(f"  {outcome}: {len(sub)} bets, {o_wr:.1f}% WR, {o_roi:.1f}% ROI, avg odds {o_avg_odds:.2f}")

    # Monthly
    print("\nMONTHLY:")
    monthly = results_df.groupby('month').agg({
        'won': ['sum', 'count'],
        'pnl': 'sum'
    })
    monthly.columns = ['wins', 'bets', 'pnl']
    monthly['wr'] = monthly['wins'] / monthly['bets'] * 100
    monthly['roi'] = monthly['pnl'] / monthly['bets'] * 100

    wr50 = 0
    profitable = 0
    for month, row in monthly.iterrows():
        wr_ok = "✓" if row['wr'] >= 50 else "✗"
        roi_ok = "✓" if row['roi'] >= 0 else "✗"
        if row['wr'] >= 50: wr50 += 1
        if row['roi'] >= 0: profitable += 1
        print(f"  {month}: {row['bets']:.0f} bets, {row['wr']:.1f}% WR {wr_ok}, {row['roi']:.1f}% ROI {roi_ok}")

    print(f"\nMonths WR>=50%: {wr50}/{len(monthly)} ({wr50/len(monthly)*100:.0f}%)")
    print(f"Profitable months: {profitable}/{len(monthly)} ({profitable/len(monthly)*100:.0f}%)")

    return {
        'total': total, 'wr': wr, 'roi': roi,
        'wr50_pct': wr50/len(monthly)*100,
        'profitable_pct': profitable/len(monthly)*100,
        'outcome_stats': outcome_stats
    }


def main():
    # Load data
    df = load_data()

    # Get base features
    feature_cols = get_base_features(df)

    # Engineer all features
    df, standard_features = engineer_standard_features(df, feature_cols)
    df, draw_features = engineer_draw_features(df)
    df, away_features = engineer_away_features(df)

    all_features = standard_features + draw_features + away_features
    # Remove duplicates
    all_features = list(dict.fromkeys(all_features))

    print(f"\nTotal features: {len(all_features)}")
    print(f"  - Standard features: {len(standard_features)}")
    print(f"  - Draw-specific features: {len(draw_features)}")
    print(f"  - Away-specific features: {len(away_features)}")

    # Prepare target
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y = df['result'].map(result_map)
    X = df[all_features].fillna(0)

    # Chronological split
    train_end = int(len(df) * 0.60)
    val_end = int(len(df) * 0.80)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    print(f"\nSplits:")
    print(f"  Train: {len(X_train):,} ({df_train['match_date'].min().date()} to {df_train['match_date'].max().date()})")
    print(f"  Val: {len(X_val):,} ({df_val['match_date'].min().date()} to {df_val['match_date'].max().date()})")
    print(f"  Test: {len(X_test):,} ({df_test['match_date'].min().date()} to {df_test['match_date'].max().date()})")

    # Train model
    model, calibrators = train_model(X_train, y_train, X_val, y_val)

    # Get calibrated probabilities
    raw_probs_val = model.predict_proba(X_val)
    cal_probs_val = apply_calibration(raw_probs_val, calibrators)

    raw_probs_test = model.predict_proba(X_test)
    cal_probs_test = apply_calibration(raw_probs_test, calibrators)

    # Find optimal thresholds per outcome using validation set
    print("\n" + "="*60)
    print("OPTIMIZING THRESHOLDS PER OUTCOME (on validation set)")
    print("="*60)

    home_config = find_optimal_thresholds(df_val, cal_probs_val, 'home', 2)
    away_config = find_optimal_thresholds(df_val, cal_probs_val, 'away', 0)
    draw_config = find_optimal_thresholds(df_val, cal_probs_val, 'draw', 1)

    print(f"\nHome optimal config: {home_config}")
    print(f"Away optimal config: {away_config}")
    print(f"Draw optimal config: {draw_config}")

    # Build final params
    final_params = {
        'home': {
            'enabled': True,
            'min_prob': home_config['min_prob'] if home_config else 0.52,
            'min_edge': home_config['min_edge'] if home_config else 0.06,
            'min_odds': home_config['min_odds'] if home_config else 1.40,
            'max_odds': home_config['max_odds'] if home_config else 3.50
        },
        'away': {
            'enabled': True,
            'min_prob': away_config['min_prob'] if away_config else 0.45,
            'min_edge': away_config['min_edge'] if away_config else 0.08,
            'min_odds': away_config['min_odds'] if away_config else 2.20,
            'max_odds': away_config['max_odds'] if away_config else 4.50
        },
        'draw': {
            'enabled': True,
            'min_prob': draw_config['min_prob'] if draw_config else 0.30,
            'min_edge': draw_config['min_edge'] if draw_config else 0.06,
            'min_odds': draw_config['min_odds'] if draw_config else 3.20,
            'max_odds': draw_config['max_odds'] if draw_config else 4.50
        }
    }

    # Test evaluation
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    preds = cal_probs_test.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"\nOverall accuracy: {acc:.1%}")

    # Backtest with all outcomes
    results_all = backtest_all_outcomes(df_test, cal_probs_test, final_params)
    stats_all = analyze_results(results_all, "ALL OUTCOMES (Optimized Thresholds)")

    # Also test home-only for comparison
    home_only_params = {
        'home': final_params['home'],
        'away': {'enabled': False},
        'draw': {'enabled': False}
    }
    results_home = backtest_all_outcomes(df_test, cal_probs_test, home_only_params)
    stats_home = analyze_results(results_home, "HOME ONLY (Comparison)")

    # Summary
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Strategy':<20} {'Bets':>6} {'WR':>8} {'ROI':>8} {'WR50%':>8} {'Prof%':>8}")
    print("-"*60)
    if stats_all:
        print(f"{'All Outcomes':<20} {stats_all['total']:>6} {stats_all['wr']:>7.1f}% {stats_all['roi']:>7.1f}% {stats_all['wr50_pct']:>7.0f}% {stats_all['profitable_pct']:>7.0f}%")
    if stats_home:
        print(f"{'Home Only':<20} {stats_home['total']:>6} {stats_home['wr']:>7.1f}% {stats_home['roi']:>7.1f}% {stats_home['wr50_pct']:>7.0f}% {stats_home['profitable_pct']:>7.0f}%")

    # Save model
    model_data = {
        'model': model,
        'calibrators': calibrators,
        'features': all_features,
        'betting_params': final_params,
        'trained_date': datetime.now().isoformat(),
        'performance': stats_all
    }

    Path('models').mkdir(exist_ok=True)
    joblib.dump(model_data, 'models/all_outcomes_optimized.joblib')
    print(f"\n✓ Model saved to: models/all_outcomes_optimized.joblib")

    results_all.to_csv('data/backtest_all_outcomes.csv', index=False)
    print(f"✓ Backtest results saved to: data/backtest_all_outcomes.csv")

    # Print final params
    print("\n" + "="*70)
    print("OPTIMIZED BETTING PARAMETERS")
    print("="*70)
    for outcome, params in final_params.items():
        if params.get('enabled'):
            print(f"\n{outcome.upper()}:")
            print(f"  min_prob: {params['min_prob']}")
            print(f"  min_edge: {params['min_edge']}")
            print(f"  odds range: {params['min_odds']} - {params['max_odds']}")

    return model, calibrators, results_all, final_params


if __name__ == '__main__':
    model, calibrators, results, params = main()
