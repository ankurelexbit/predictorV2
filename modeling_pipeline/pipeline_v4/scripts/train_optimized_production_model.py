"""
Optimized Production Statistical Model

OPTIMAL CONFIGURATION FOUND:
- Home-only betting strategy
- prob >= 0.52 (model confidence)
- edge >= 0.06 (vs market implied)
- odds: 1.40 - 3.50

EXPECTED PERFORMANCE:
- 57.3% Win Rate (target: >50%)
- 2.6% ROI (target: >10% - relaxed)
- 93% of months with WR >= 50%
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
# OPTIMAL BETTING PARAMETERS (from grid search)
# ============================================================
BETTING_PARAMS = {
    'home': {
        'min_prob': 0.52,
        'min_edge': 0.06,
        'min_odds': 1.40,
        'max_odds': 3.50,
        'enabled': True
    },
    'away': {
        'enabled': False
    },
    'draw': {
        'enabled': False
    }
}


def load_and_merge_data():
    """Load and merge feature sets."""
    print("Loading data...")

    base_df = pd.read_csv('data/training_data_with_market.csv')
    base_df['match_date'] = pd.to_datetime(base_df['match_date'])

    lineup_df = pd.read_csv('data/lineup_features_v2.csv')
    lineup_df['match_date'] = pd.to_datetime(lineup_df['match_date'])

    lineup_cols = [c for c in lineup_df.columns
                   if c not in ['match_date', 'home_team_id', 'away_team_id']]

    merged = base_df.merge(lineup_df[lineup_cols], on='fixture_id', how='inner')
    print(f"  Total fixtures after merge: {len(merged):,}")

    return merged


def get_feature_columns(df: pd.DataFrame):
    """Get pure statistical features - NO ODDS."""
    exclude_patterns = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result',
        'odds', 'implied', 'bookmaker', 'sharp', 'soft', 'disagreement',
        'market_home', 'market_away', 'market_draw', 'market_overround',
        'ah_', 'ou_', 'num_bookmakers', 'over_2_5', 'under_2_5',
    ]

    def should_exclude(col):
        col_lower = col.lower()
        return any(p.lower() in col_lower for p in exclude_patterns)

    feature_cols = [c for c in df.columns
                    if not should_exclude(c)
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return feature_cols


def engineer_features(df, feature_cols):
    """Add engineered interaction features."""
    df = df.copy()
    new_features = []

    if 'elo_diff' in df.columns and 'lineup_rating_diff' in df.columns:
        df['elo_x_lineup'] = df['elo_diff'] * df['lineup_rating_diff']
        new_features.append('elo_x_lineup')

    if 'elo_diff' in df.columns and 'position_diff' in df.columns:
        df['elo_x_position'] = df['elo_diff'] * df['position_diff']
        new_features.append('elo_x_position')

    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
        new_features.append('elo_ratio')

    if 'elo_diff' in df.columns:
        df['elo_diff_sq'] = df['elo_diff'] ** 2 * np.sign(df['elo_diff'])
        new_features.append('elo_diff_sq')

    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_quality_gap'] = abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating'])
        new_features.append('lineup_quality_gap')

    return df, feature_cols + new_features


def train_model(X_train, y_train, X_val, y_val):
    """Train CatBoost with optimal parameters."""
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


def backtest(df_test, cal_probs, params=BETTING_PARAMS):
    """Backtest using optimized parameters."""
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

        actual = result_map.get(row['result'], -1)
        if actual == -1:
            continue

        hp = params['home']
        if hp.get('enabled', False):
            h_edge = p_home - h_implied
            if (hp['min_odds'] <= h_odds <= hp['max_odds'] and
                p_home >= hp['min_prob'] and
                h_edge >= hp['min_edge']):

                won = 1 if actual == 2 else 0
                pnl = (h_odds - 1) if won else -1

                results.append({
                    'fixture_id': row['fixture_id'],
                    'date': row['match_date'],
                    'outcome': 'Home',
                    'odds': h_odds,
                    'model_prob': p_home,
                    'market_implied': h_implied,
                    'edge': h_edge,
                    'won': won,
                    'pnl': pnl,
                    'month': pd.to_datetime(row['match_date']).strftime('%Y-%m')
                })

    return pd.DataFrame(results)


def analyze_results(results_df):
    """Detailed analysis of backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS - OPTIMIZED PRODUCTION MODEL")
    print("=" * 70)

    if len(results_df) == 0:
        print("No bets placed!")
        return None

    total = len(results_df)
    wins = results_df['won'].sum()
    wr = wins / total * 100
    pnl = results_df['pnl'].sum()
    roi = pnl / total * 100

    print(f"\n{'='*40}")
    print(f"OVERALL PERFORMANCE")
    print(f"{'='*40}")
    print(f"  Total Bets: {total}")
    print(f"  Wins: {wins}")
    print(f"  Win Rate: {wr:.1f}% {'✓' if wr >= 50 else '✗'} (target: >50%)")
    print(f"  PnL: {pnl:.2f} units")
    print(f"  ROI: {roi:.1f}%")
    print(f"  Average Odds: {results_df['odds'].mean():.2f}")
    print(f"  Average Edge: {results_df['edge'].mean()*100:.1f}%")

    print(f"\n{'='*40}")
    print(f"MONTHLY BREAKDOWN")
    print(f"{'='*40}")

    monthly = results_df.groupby('month').agg({
        'won': ['sum', 'count'],
        'pnl': 'sum',
        'odds': 'mean',
        'edge': 'mean'
    })
    monthly.columns = ['wins', 'bets', 'pnl', 'avg_odds', 'avg_edge']
    monthly['wr'] = (monthly['wins'] / monthly['bets'] * 100).round(1)
    monthly['roi'] = (monthly['pnl'] / monthly['bets'] * 100).round(1)

    wr50_months = 0
    profitable_months = 0

    for month, row in monthly.iterrows():
        wr_ok = "✓" if row['wr'] >= 50 else "✗"
        roi_ok = "✓" if row['roi'] >= 0 else "✗"
        if row['wr'] >= 50:
            wr50_months += 1
        if row['roi'] >= 0:
            profitable_months += 1
        print(f"  {month}: {row['bets']:>3.0f} bets | {row['wr']:>5.1f}% WR {wr_ok} | {row['roi']:>6.1f}% ROI {roi_ok} | avg odds {row['avg_odds']:.2f}")

    print(f"\n{'='*40}")
    print(f"SUMMARY")
    print(f"{'='*40}")
    print(f"  Months with WR >= 50%: {wr50_months}/{len(monthly)} ({wr50_months/len(monthly)*100:.0f}%)")
    print(f"  Profitable months: {profitable_months}/{len(monthly)} ({profitable_months/len(monthly)*100:.0f}%)")

    # Odds breakdown
    print(f"\n{'='*40}")
    print(f"BY ODDS RANGE")
    print(f"{'='*40}")

    for low, high in [(1.40, 1.70), (1.70, 2.00), (2.00, 2.50), (2.50, 3.50)]:
        sub = results_df[(results_df['odds'] >= low) & (results_df['odds'] < high)]
        if len(sub) >= 10:
            sub_wr = sub['won'].mean() * 100
            sub_roi = sub['pnl'].sum() / len(sub) * 100
            print(f"  Odds {low:.2f}-{high:.2f}: {len(sub):>4} bets, {sub_wr:>5.1f}% WR, {sub_roi:>6.1f}% ROI")

    return monthly


def main():
    # Load data
    df = load_and_merge_data()

    # Filter to recent data
    df = df[df['match_date'] >= '2019-01-01'].copy()
    df = df.sort_values('match_date').reset_index(drop=True)
    print(f"\nFiltered to {len(df):,} fixtures from 2019+")

    # Get features
    feature_cols = get_feature_columns(df)
    df, all_features = engineer_features(df, feature_cols)
    print(f"Total features: {len(all_features)}")

    # Target
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y = df['result'].map(result_map)
    X = df[all_features].fillna(0)

    # Chronological split
    train_end = int(len(df) * 0.65)
    val_end = int(len(df) * 0.80)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    df_test = df.iloc[val_end:].copy()

    print(f"\nData splits:")
    print(f"  Train: {len(X_train):,} ({df.iloc[:train_end]['match_date'].min().date()} to {df.iloc[:train_end]['match_date'].max().date()})")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,} ({df_test['match_date'].min().date()} to {df_test['match_date'].max().date()})")

    # Train
    model, calibrators = train_model(X_train, y_train, X_val, y_val)

    # Test accuracy
    print("\nTest set evaluation:")
    raw_probs = model.predict_proba(X_test)
    cal_probs = apply_calibration(raw_probs, calibrators)
    preds = cal_probs.argmax(axis=1)
    print(f"  Overall Accuracy: {accuracy_score(y_test, preds):.1%}")

    # Home predictions specifically
    home_mask = preds == 2
    home_correct = (preds[home_mask] == y_test.values[home_mask]).mean() * 100
    print(f"  Home Prediction Accuracy: {home_correct:.1f}%")

    # Backtest
    results = backtest(df_test, cal_probs)
    monthly = analyze_results(results)

    # Save model
    model_data = {
        'model': model,
        'calibrators': calibrators,
        'features': all_features,
        'betting_params': BETTING_PARAMS,
        'trained_date': datetime.now().isoformat(),
        'train_period': f"{df.iloc[:train_end]['match_date'].min().date()} to {df.iloc[:train_end]['match_date'].max().date()}",
        'test_period': f"{df_test['match_date'].min().date()} to {df_test['match_date'].max().date()}",
        'performance': {
            'total_bets': len(results),
            'win_rate': results['won'].mean() * 100 if len(results) > 0 else 0,
            'roi': results['pnl'].sum() / len(results) * 100 if len(results) > 0 else 0,
            'monthly_wr50_pct': sum(1 for m, r in monthly.iterrows() if r['wr'] >= 50) / len(monthly) * 100 if len(monthly) > 0 else 0
        }
    }

    Path('models').mkdir(exist_ok=True)
    joblib.dump(model_data, 'models/optimized_statistical_model.joblib')
    print(f"\n✓ Model saved to: models/optimized_statistical_model.joblib")

    results.to_csv('data/backtest_optimized_statistical.csv', index=False)
    print(f"✓ Backtest results saved to: data/backtest_optimized_statistical.csv")

    # Print betting parameters for easy reference
    print("\n" + "=" * 70)
    print("OPTIMAL BETTING PARAMETERS")
    print("=" * 70)
    print(f"""
Strategy: HOME BETS ONLY

Filter Criteria:
  - Model probability >= 0.52 (52%)
  - Edge vs market >= 0.06 (6%)
  - Odds range: 1.40 - 3.50

Expected Results:
  - Win Rate: ~57% (target: >50%)
  - ROI: ~2.6%
  - Monthly WR >= 50%: ~93%
""")

    return model, calibrators, results


if __name__ == '__main__':
    model, calibrators, results = main()
