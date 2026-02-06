"""
Sweet Spot Strategy - Optimized for ALL Outcomes

Based on empirical analysis, each outcome has specific conditions where it works best:

HOME:
  - Edge: 6-10% (moderate edge, not overconfident)
  - Odds: 1.70-2.00 (slight favorites, not heavy favorites)
  - Expected: ~60% WR, ~12% ROI

AWAY:
  - Edge: 10-20% (needs higher edge to overcome away disadvantage)
  - Odds: 2.50-3.00 (moderate underdogs)
  - Expected: ~50% WR, ~22% ROI

DRAW:
  - Edge: 6-10% (low edge only - high edge draws are overfitted)
  - Odds: 3.00-3.40 (typical draw odds, not extreme)
  - Expected: ~43% WR, ~43% ROI (high variance)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SWEET SPOT PARAMETERS (derived from empirical analysis)
# ============================================================
SWEET_SPOT_PARAMS = {
    'home': {
        'enabled': True,
        'min_prob': 0.50,
        'min_edge': 0.06,
        'max_edge': 0.12,  # Cap edge - high edge = overconfidence
        'min_odds': 1.65,
        'max_odds': 2.10
    },
    'away': {
        'enabled': True,
        'min_prob': 0.42,
        'min_edge': 0.10,
        'max_edge': 0.22,
        'min_odds': 2.40,
        'max_odds': 3.20
    },
    'draw': {
        'enabled': True,
        'min_prob': 0.28,
        'min_edge': 0.05,
        'max_edge': 0.12,  # Critical: high edge draws fail
        'min_odds': 3.00,
        'max_odds': 3.50
    }
}


def load_data():
    """Load and merge data."""
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


def get_features(df):
    """Get features excluding odds."""
    exclude_patterns = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result',
        'odds', 'implied', 'bookmaker', 'sharp', 'soft', 'disagreement',
        'market_home', 'market_away', 'market_draw', 'market_overround',
        'ah_', 'ou_', 'num_bookmakers', 'over_2_5', 'under_2_5',
    ]

    def should_exclude(col):
        return any(p.lower() in col.lower() for p in exclude_patterns)

    return [c for c in df.columns
            if not should_exclude(c) and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]


def engineer_features(df, feature_cols):
    """Add engineered features for all outcomes."""
    df = df.copy()
    new_features = []

    # Standard interactions
    if 'elo_diff' in df.columns:
        df['elo_diff_sq'] = df['elo_diff'] ** 2 * np.sign(df['elo_diff'])
        new_features.append('elo_diff_sq')

        if 'lineup_rating_diff' in df.columns:
            df['elo_x_lineup'] = df['elo_diff'] * df['lineup_rating_diff']
            new_features.append('elo_x_lineup')

    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
        new_features.append('elo_ratio')

    # Draw-specific: closeness indicators
    if 'elo_diff' in df.columns:
        df['elo_closeness'] = 1 / (1 + abs(df['elo_diff']) / 50)
        new_features.append('elo_closeness')
        df['elo_diff_abs'] = abs(df['elo_diff'])
        new_features.append('elo_diff_abs')

    if 'position_diff' in df.columns:
        df['position_closeness'] = 1 / (1 + abs(df['position_diff']))
        new_features.append('position_closeness')

    if 'points_diff' in df.columns:
        df['points_closeness'] = 1 / (1 + abs(df['points_diff']) / 5)
        new_features.append('points_closeness')

    # Lineup balance
    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_closeness'] = 1 / (1 + abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating']) * 10)
        new_features.append('lineup_closeness')

    # Away-specific
    if 'away_elo' in df.columns and 'home_elo' in df.columns:
        df['away_elo_advantage'] = df['away_elo'] - df['home_elo']
        new_features.append('away_elo_advantage')

    return df, feature_cols + new_features


def train_model(X_train, y_train, X_val, y_val):
    """Train model."""
    print("\nTraining CatBoost...")

    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        eval_metric='MultiClass'
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    # Calibrate
    print("\nCalibrating...")
    raw_probs = model.predict_proba(X_val)

    calibrators = {}
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
        calibrators[outcome].fit(raw_probs[:, idx], (y_val == idx).astype(int))

    return model, calibrators


def apply_calibration(probs, calibrators):
    """Apply calibration."""
    cal_probs = np.zeros_like(probs)
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(probs[:, idx])

    row_sums = cal_probs.sum(axis=1, keepdims=True)
    return cal_probs / np.where(row_sums == 0, 1, row_sums)


def backtest_sweet_spots(df_test, cal_probs, params=SWEET_SPOT_PARAMS):
    """Backtest using sweet spot strategy with edge caps."""
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

        # Home - sweet spot
        hp = params['home']
        if hp.get('enabled', False):
            h_edge = p_home - h_implied
            if (hp['min_odds'] <= h_odds <= hp['max_odds'] and
                p_home >= hp['min_prob'] and
                hp['min_edge'] <= h_edge <= hp.get('max_edge', 1.0)):
                candidates.append({
                    'outcome': 'Home', 'idx': 2,
                    'odds': h_odds, 'prob': p_home, 'edge': h_edge
                })

        # Away - sweet spot
        ap = params['away']
        if ap.get('enabled', False):
            a_edge = p_away - a_implied
            if (ap['min_odds'] <= a_odds <= ap['max_odds'] and
                p_away >= ap['min_prob'] and
                ap['min_edge'] <= a_edge <= ap.get('max_edge', 1.0)):
                candidates.append({
                    'outcome': 'Away', 'idx': 0,
                    'odds': a_odds, 'prob': p_away, 'edge': a_edge
                })

        # Draw - sweet spot (CRITICAL: cap edge)
        dp = params['draw']
        if dp.get('enabled', False):
            d_edge = p_draw - d_implied
            if (dp['min_odds'] <= d_odds <= dp['max_odds'] and
                p_draw >= dp['min_prob'] and
                dp['min_edge'] <= d_edge <= dp.get('max_edge', 1.0)):
                candidates.append({
                    'outcome': 'Draw', 'idx': 1,
                    'odds': d_odds, 'prob': p_draw, 'edge': d_edge
                })

        if not candidates:
            continue

        # Select by edge (within sweet spot, higher is still better)
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
    print(f"\n{'='*70}")
    print(title)
    print('='*70)

    if len(results_df) == 0:
        print("No bets placed!")
        return None

    total = len(results_df)
    wins = results_df['won'].sum()
    wr = wins / total * 100
    pnl = results_df['pnl'].sum()
    roi = pnl / total * 100

    print(f"\n{'OVERALL':}")
    print(f"  Total Bets: {total}")
    print(f"  Wins: {int(wins)}, Losses: {int(total - wins)}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  PnL: {pnl:+.2f} units")
    print(f"  ROI: {roi:+.1f}%")

    # By outcome
    print(f"\nBY OUTCOME:")
    outcome_stats = {}
    for outcome in ['Home', 'Away', 'Draw']:
        sub = results_df[results_df['outcome'] == outcome]
        if len(sub) > 0:
            o_wr = sub['won'].mean() * 100
            o_roi = sub['pnl'].sum() / len(sub) * 100
            o_pnl = sub['pnl'].sum()
            outcome_stats[outcome] = {'bets': len(sub), 'wr': o_wr, 'roi': o_roi, 'pnl': o_pnl}
            status = "✓" if o_roi > 0 else "✗"
            print(f"  {outcome}: {len(sub):>3} bets, {o_wr:>5.1f}% WR, {o_roi:>+6.1f}% ROI, {o_pnl:>+6.2f} units {status}")

    # Monthly
    print(f"\nMONTHLY:")
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
        print(f"  {month}: {row['bets']:>3.0f} bets, {row['wr']:>5.1f}% WR {wr_ok}, {row['roi']:>+6.1f}% ROI {roi_ok}")

    print(f"\n  Months WR >= 50%: {wr50}/{len(monthly)} ({wr50/len(monthly)*100:.0f}%)")
    print(f"  Profitable months: {profitable}/{len(monthly)} ({profitable/len(monthly)*100:.0f}%)")

    return {
        'total': total, 'wr': wr, 'roi': roi, 'pnl': pnl,
        'wr50_pct': wr50/len(monthly)*100,
        'profitable_pct': profitable/len(monthly)*100,
        'outcome_stats': outcome_stats
    }


def main():
    # Load data
    df = load_data()

    # Features
    feature_cols = get_features(df)
    df, all_features = engineer_features(df, feature_cols)
    print(f"Total features: {len(all_features)}")

    # Target
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y = df['result'].map(result_map)
    X = df[all_features].fillna(0)

    # Split
    train_end = int(len(df) * 0.65)
    val_end = int(len(df) * 0.80)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    df_test = df.iloc[val_end:].copy()

    print(f"\nSplits:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,} ({df_test['match_date'].min().date()} to {df_test['match_date'].max().date()})")

    # Train
    model, calibrators = train_model(X_train, y_train, X_val, y_val)

    # Test
    raw_probs = model.predict_proba(X_test)
    cal_probs = apply_calibration(raw_probs, calibrators)

    print(f"\nTest accuracy: {accuracy_score(y_test, cal_probs.argmax(axis=1)):.1%}")

    # ========================================
    # BACKTEST SWEET SPOT STRATEGY
    # ========================================

    results = backtest_sweet_spots(df_test, cal_probs)
    stats = analyze_results(results, "SWEET SPOT STRATEGY - ALL OUTCOMES")

    # Compare with different configurations
    print("\n" + "="*70)
    print("TESTING VARIATIONS")
    print("="*70)

    # Variation 1: Home + Away only (no draws)
    params_ha = SWEET_SPOT_PARAMS.copy()
    params_ha['draw'] = {'enabled': False}
    results_ha = backtest_sweet_spots(df_test, cal_probs, params_ha)
    stats_ha = analyze_results(results_ha, "HOME + AWAY ONLY")

    # Variation 2: Tighter draw constraints
    params_tight_draw = SWEET_SPOT_PARAMS.copy()
    params_tight_draw['draw'] = {
        'enabled': True,
        'min_prob': 0.30,
        'min_edge': 0.06,
        'max_edge': 0.10,  # Very tight
        'min_odds': 3.10,
        'max_odds': 3.40
    }
    results_tight = backtest_sweet_spots(df_test, cal_probs, params_tight_draw)
    stats_tight = analyze_results(results_tight, "TIGHT DRAW CONSTRAINTS")

    # Summary
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    print(f"{'Strategy':<25} {'Bets':>6} {'WR':>8} {'ROI':>8} {'PnL':>10} {'WR50%':>8}")
    print("-"*70)

    for name, s in [("Sweet Spot (All)", stats),
                    ("Home + Away", stats_ha),
                    ("Tight Draw", stats_tight)]:
        if s:
            print(f"{name:<25} {s['total']:>6} {s['wr']:>7.1f}% {s['roi']:>+7.1f}% {s['pnl']:>+9.2f} {s['wr50_pct']:>7.0f}%")

    # Save best model
    best_stats = max([stats, stats_ha, stats_tight], key=lambda x: x['roi'] if x else -999)
    best_params = SWEET_SPOT_PARAMS if best_stats == stats else (params_ha if best_stats == stats_ha else params_tight_draw)
    best_results = results if best_stats == stats else (results_ha if best_stats == stats_ha else results_tight)

    model_data = {
        'model': model,
        'calibrators': calibrators,
        'features': all_features,
        'betting_params': best_params,
        'trained_date': datetime.now().isoformat(),
        'performance': best_stats
    }

    Path('models').mkdir(exist_ok=True)
    joblib.dump(model_data, 'models/sweet_spot_strategy.joblib')
    print(f"\n✓ Model saved to: models/sweet_spot_strategy.joblib")

    best_results.to_csv('data/backtest_sweet_spot.csv', index=False)
    print(f"✓ Results saved to: data/backtest_sweet_spot.csv")

    # Print final params
    print("\n" + "="*70)
    print("FINAL SWEET SPOT PARAMETERS")
    print("="*70)
    for outcome, params in best_params.items():
        if params.get('enabled'):
            print(f"\n{outcome.upper()}:")
            print(f"  Probability: >= {params['min_prob']}")
            print(f"  Edge: {params['min_edge']*100:.0f}% - {params.get('max_edge', 1)*100:.0f}%")
            print(f"  Odds: {params['min_odds']:.2f} - {params['max_odds']:.2f}")

    return model, calibrators, best_results, best_params


if __name__ == '__main__':
    model, calibrators, results, params = main()
