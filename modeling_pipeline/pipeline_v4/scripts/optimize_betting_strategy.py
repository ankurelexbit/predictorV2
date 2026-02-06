"""
Optimize Betting Strategy Parameters

Systematically search for the best combination of:
- Threshold parameters (min_prob, min_edge, odds range)
- Outcome selection (Home only, Home+Away, All)
- Monthly consistency optimization
"""

import pandas as pd
import numpy as np
import joblib
from itertools import product
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_model_and_data():
    """Load production model and test data."""
    print("Loading model and data...")

    # Load model
    model_data = joblib.load('models/production_statistical_model.joblib')
    model = model_data['model']
    calibrators = model_data['calibrators']
    features = model_data['features']

    # Load data
    base_df = pd.read_csv('data/training_data_with_market.csv')
    base_df['match_date'] = pd.to_datetime(base_df['match_date'])

    lineup_df = pd.read_csv('data/lineup_features_v2.csv')
    lineup_df['match_date'] = pd.to_datetime(lineup_df['match_date'])

    lineup_cols = [c for c in lineup_df.columns
                   if c not in ['match_date', 'home_team_id', 'away_team_id']]

    df = base_df.merge(lineup_df[lineup_cols], on='fixture_id', how='inner')
    df = df[df['match_date'] >= '2019-01-01'].copy()
    df = df.sort_values('match_date').reset_index(drop=True)

    # Same split as training
    val_end = int(len(df) * 0.80)
    df_test = df.iloc[val_end:].copy()

    # Engineer features (same as training)
    if 'elo_diff' in df_test.columns and 'lineup_rating_diff' in df_test.columns:
        df_test['elo_x_lineup'] = df_test['elo_diff'] * df_test['lineup_rating_diff']
    if 'elo_diff' in df_test.columns and 'position_diff' in df_test.columns:
        df_test['elo_x_position'] = df_test['elo_diff'] * df_test['position_diff']
    if 'home_elo' in df_test.columns and 'away_elo' in df_test.columns:
        df_test['elo_ratio'] = df_test['home_elo'] / (df_test['away_elo'] + 1)
    if 'elo_diff' in df_test.columns:
        df_test['elo_diff_sq'] = df_test['elo_diff'] ** 2 * np.sign(df_test['elo_diff'])
    if 'home_lineup_avg_rating' in df_test.columns and 'away_lineup_avg_rating' in df_test.columns:
        df_test['lineup_quality_gap'] = abs(df_test['home_lineup_avg_rating'] - df_test['away_lineup_avg_rating'])

    X_test = df_test[features].fillna(0)

    # Get calibrated probabilities
    raw_probs = model.predict_proba(X_test)

    cal_probs = np.zeros_like(raw_probs)
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(raw_probs[:, idx])
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    cal_probs = cal_probs / np.where(row_sums == 0, 1, row_sums)

    print(f"  Test data: {len(df_test):,} fixtures")
    print(f"  Date range: {df_test['match_date'].min().date()} to {df_test['match_date'].max().date()}")

    return df_test, cal_probs


def backtest_with_params(df_test, cal_probs, params):
    """Backtest with specific parameters."""
    results = []
    result_map = {'A': 0, 'D': 1, 'H': 2}

    for i in range(len(df_test)):
        row = df_test.iloc[i]

        p_away = cal_probs[i, 0]
        p_draw = cal_probs[i, 1]
        p_home = cal_probs[i, 2]

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

        # Home bet
        if params.get('home_enabled', True):
            h_edge = p_home - h_implied
            if (params['home_min_odds'] <= h_odds <= params['home_max_odds'] and
                p_home >= params['home_min_prob'] and
                h_edge >= params['home_min_edge']):
                candidates.append({
                    'outcome': 'Home', 'idx': 2,
                    'odds': h_odds, 'prob': p_home, 'edge': h_edge
                })

        # Away bet
        if params.get('away_enabled', False):
            a_edge = p_away - a_implied
            if (params['away_min_odds'] <= a_odds <= params['away_max_odds'] and
                p_away >= params['away_min_prob'] and
                a_edge >= params['away_min_edge']):
                candidates.append({
                    'outcome': 'Away', 'idx': 0,
                    'odds': a_odds, 'prob': p_away, 'edge': a_edge
                })

        # Draw bet
        if params.get('draw_enabled', False):
            d_edge = p_draw - d_implied
            if (params['draw_min_odds'] <= d_odds <= params['draw_max_odds'] and
                p_draw >= params['draw_min_prob'] and
                d_edge >= params['draw_min_edge']):
                candidates.append({
                    'outcome': 'Draw', 'idx': 1,
                    'odds': d_odds, 'prob': p_draw, 'edge': d_edge
                })

        if not candidates:
            continue

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


def evaluate_results(results_df):
    """Calculate performance metrics."""
    if len(results_df) < 20:
        return None

    total = len(results_df)
    wr = results_df['won'].mean() * 100
    roi = results_df['pnl'].sum() / total * 100

    # Monthly stats
    monthly = results_df.groupby('month').agg({
        'won': ['sum', 'count'],
        'pnl': 'sum'
    })
    monthly.columns = ['wins', 'bets', 'pnl']
    monthly['wr'] = monthly['wins'] / monthly['bets'] * 100
    monthly['roi'] = monthly['pnl'] / monthly['bets'] * 100

    wr50_months = (monthly['wr'] >= 50).sum()
    profitable_months = (monthly['roi'] >= 0).sum()
    total_months = len(monthly)

    return {
        'total_bets': total,
        'win_rate': wr,
        'roi': roi,
        'wr50_months': wr50_months,
        'profitable_months': profitable_months,
        'total_months': total_months,
        'wr50_pct': wr50_months / total_months * 100,
        'profitable_pct': profitable_months / total_months * 100
    }


def grid_search_home_only():
    """Grid search for optimal Home-only parameters."""
    print("\n" + "=" * 70)
    print("GRID SEARCH: HOME ONLY")
    print("=" * 70)

    df_test, cal_probs = load_model_and_data()

    best_results = []

    # Parameter grid
    min_probs = [0.42, 0.45, 0.48, 0.50, 0.52]
    min_edges = [0.04, 0.05, 0.06, 0.07, 0.08]
    min_odds_list = [1.40, 1.50, 1.60]
    max_odds_list = [3.00, 3.50, 4.00]

    total_combos = len(min_probs) * len(min_edges) * len(min_odds_list) * len(max_odds_list)
    print(f"\nTesting {total_combos} combinations...")

    for min_prob, min_edge, min_odds, max_odds in product(min_probs, min_edges, min_odds_list, max_odds_list):
        params = {
            'home_enabled': True,
            'home_min_prob': min_prob,
            'home_min_edge': min_edge,
            'home_min_odds': min_odds,
            'home_max_odds': max_odds,
            'away_enabled': False,
            'draw_enabled': False
        }

        results = backtest_with_params(df_test, cal_probs, params)
        metrics = evaluate_results(results)

        if metrics and metrics['total_bets'] >= 50:
            best_results.append({
                'params': params.copy(),
                'metrics': metrics
            })

    # Sort by composite score (prioritize WR, then ROI)
    def score(r):
        m = r['metrics']
        return m['win_rate'] * 2 + m['roi'] + m['wr50_pct']

    best_results.sort(key=score, reverse=True)

    print(f"\nTop 10 configurations (sorted by composite score):")
    print(f"{'Bets':>6} {'WR':>7} {'ROI':>8} {'WR50%':>7} {'Prof%':>7} | Parameters")
    print("-" * 80)

    for r in best_results[:10]:
        m = r['metrics']
        p = r['params']
        print(f"{m['total_bets']:>6} {m['win_rate']:>6.1f}% {m['roi']:>7.1f}% {m['wr50_pct']:>6.1f}% {m['profitable_pct']:>6.1f}% | "
              f"prob>{p['home_min_prob']:.2f} edge>{p['home_min_edge']:.2f} odds:{p['home_min_odds']:.1f}-{p['home_max_odds']:.1f}")

    return best_results


def grid_search_home_away():
    """Grid search for optimal Home + Away parameters."""
    print("\n" + "=" * 70)
    print("GRID SEARCH: HOME + AWAY")
    print("=" * 70)

    df_test, cal_probs = load_model_and_data()

    best_results = []

    # Simplified parameter grid for Home + Away
    home_configs = [
        {'min_prob': 0.48, 'min_edge': 0.06, 'min_odds': 1.50, 'max_odds': 3.50},
        {'min_prob': 0.50, 'min_edge': 0.05, 'min_odds': 1.50, 'max_odds': 3.00},
        {'min_prob': 0.45, 'min_edge': 0.07, 'min_odds': 1.50, 'max_odds': 3.50},
    ]

    away_configs = [
        {'min_prob': 0.45, 'min_edge': 0.08, 'min_odds': 2.00, 'max_odds': 4.00},
        {'min_prob': 0.48, 'min_edge': 0.10, 'min_odds': 2.20, 'max_odds': 4.50},
        {'min_prob': 0.50, 'min_edge': 0.12, 'min_odds': 2.50, 'max_odds': 5.00},
    ]

    for home_cfg, away_cfg in product(home_configs, away_configs):
        params = {
            'home_enabled': True,
            'home_min_prob': home_cfg['min_prob'],
            'home_min_edge': home_cfg['min_edge'],
            'home_min_odds': home_cfg['min_odds'],
            'home_max_odds': home_cfg['max_odds'],
            'away_enabled': True,
            'away_min_prob': away_cfg['min_prob'],
            'away_min_edge': away_cfg['min_edge'],
            'away_min_odds': away_cfg['min_odds'],
            'away_max_odds': away_cfg['max_odds'],
            'draw_enabled': False
        }

        results = backtest_with_params(df_test, cal_probs, params)
        metrics = evaluate_results(results)

        if metrics and metrics['total_bets'] >= 50:
            best_results.append({
                'params': params.copy(),
                'metrics': metrics
            })

    # Sort by composite score
    def score(r):
        m = r['metrics']
        return m['win_rate'] * 2 + m['roi'] + m['wr50_pct']

    best_results.sort(key=score, reverse=True)

    print(f"\nTop 5 HOME + AWAY configurations:")
    print(f"{'Bets':>6} {'WR':>7} {'ROI':>8} {'WR50%':>7} | Home params | Away params")
    print("-" * 100)

    for r in best_results[:5]:
        m = r['metrics']
        p = r['params']
        print(f"{m['total_bets']:>6} {m['win_rate']:>6.1f}% {m['roi']:>7.1f}% {m['wr50_pct']:>6.1f}% | "
              f"p>{p['home_min_prob']:.2f} e>{p['home_min_edge']:.2f} | "
              f"p>{p['away_min_prob']:.2f} e>{p['away_min_edge']:.2f}")

    return best_results


def test_high_confidence_only():
    """Test extremely selective high-confidence strategy."""
    print("\n" + "=" * 70)
    print("HIGH CONFIDENCE STRATEGY")
    print("=" * 70)

    df_test, cal_probs = load_model_and_data()

    # Very selective parameters
    params = {
        'home_enabled': True,
        'home_min_prob': 0.55,  # Very high probability
        'home_min_edge': 0.10,  # Large edge
        'home_min_odds': 1.60,
        'home_max_odds': 2.80,
        'away_enabled': True,
        'away_min_prob': 0.55,
        'away_min_edge': 0.12,
        'away_min_odds': 2.50,
        'away_max_odds': 4.00,
        'draw_enabled': False
    }

    results = backtest_with_params(df_test, cal_probs, params)

    if len(results) > 0:
        print(f"\nHigh confidence results:")
        print(f"  Bets: {len(results)}")
        print(f"  Win Rate: {results['won'].mean()*100:.1f}%")
        print(f"  ROI: {results['pnl'].sum()/len(results)*100:.1f}%")

        # By outcome
        print("\n  By outcome:")
        for outcome in ['Home', 'Away']:
            sub = results[results['outcome'] == outcome]
            if len(sub) > 0:
                print(f"    {outcome}: {len(sub)} bets, {sub['won'].mean()*100:.1f}% WR, {sub['pnl'].sum()/len(sub)*100:.1f}% ROI")

        # Monthly
        print("\n  Monthly:")
        monthly = results.groupby('month').agg({'won': ['sum', 'count'], 'pnl': 'sum'})
        monthly.columns = ['wins', 'bets', 'pnl']
        monthly['wr'] = monthly['wins'] / monthly['bets'] * 100
        monthly['roi'] = monthly['pnl'] / monthly['bets'] * 100

        for month, row in monthly.iterrows():
            wr_ok = "✓" if row['wr'] >= 50 else "✗"
            roi_ok = "✓" if row['roi'] >= 0 else "✗"
            print(f"    {month}: {row['bets']:.0f} bets, {row['wr']:.1f}% WR {wr_ok}, {row['roi']:.1f}% ROI {roi_ok}")

    return results


def test_value_betting():
    """Test value betting: bet only when edge is very large."""
    print("\n" + "=" * 70)
    print("VALUE BETTING STRATEGY (Large Edge Only)")
    print("=" * 70)

    df_test, cal_probs = load_model_and_data()

    edge_thresholds = [0.08, 0.10, 0.12, 0.15]

    for min_edge in edge_thresholds:
        params = {
            'home_enabled': True,
            'home_min_prob': 0.40,
            'home_min_edge': min_edge,
            'home_min_odds': 1.50,
            'home_max_odds': 4.00,
            'away_enabled': True,
            'away_min_prob': 0.40,
            'away_min_edge': min_edge,
            'away_min_odds': 1.80,
            'away_max_odds': 5.00,
            'draw_enabled': False
        }

        results = backtest_with_params(df_test, cal_probs, params)

        if len(results) >= 20:
            wr = results['won'].mean() * 100
            roi = results['pnl'].sum() / len(results) * 100
            avg_odds = results['odds'].mean()
            avg_edge = results['edge'].mean() * 100
            print(f"  Edge >= {min_edge*100:.0f}%: {len(results):>4} bets, {wr:>5.1f}% WR, {roi:>6.1f}% ROI, avg odds {avg_odds:.2f}, avg edge {avg_edge:.1f}%")


def main():
    # Run all optimizations
    print("=" * 70)
    print("BETTING STRATEGY OPTIMIZATION")
    print("=" * 70)

    # 1. Grid search for home only
    home_results = grid_search_home_only()

    # 2. Grid search for home + away
    ha_results = grid_search_home_away()

    # 3. High confidence strategy
    hc_results = test_high_confidence_only()

    # 4. Value betting
    test_value_betting()

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    if home_results:
        best = home_results[0]
        print(f"\nBest Home-Only Config:")
        print(f"  Bets: {best['metrics']['total_bets']}")
        print(f"  WR: {best['metrics']['win_rate']:.1f}%")
        print(f"  ROI: {best['metrics']['roi']:.1f}%")
        print(f"  WR>=50% months: {best['metrics']['wr50_pct']:.0f}%")
        p = best['params']
        print(f"  Params: prob>={p['home_min_prob']}, edge>={p['home_min_edge']}, odds {p['home_min_odds']}-{p['home_max_odds']}")

    if ha_results:
        best = ha_results[0]
        print(f"\nBest Home+Away Config:")
        print(f"  Bets: {best['metrics']['total_bets']}")
        print(f"  WR: {best['metrics']['win_rate']:.1f}%")
        print(f"  ROI: {best['metrics']['roi']:.1f}%")
        print(f"  WR>=50% months: {best['metrics']['wr50_pct']:.0f}%")


if __name__ == '__main__':
    main()
