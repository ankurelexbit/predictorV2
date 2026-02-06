"""
Production Statistical Model (No Odds Features)

Best performing configuration from exploration:
- Home-only betting strategy
- Model: CatBoost with calibration
- Thresholds: prob>=0.48, edge>=0.06, odds 1.5-3.5
- Expected: ~51% WR, ~3% ROI

This model uses only point-in-time correct features:
1. Elo ratings
2. League standings
3. Recent form (last 5/10 matches)
4. Historical player ratings (PIT correct)
5. xG and other match statistics
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# OPTIMAL BETTING PARAMETERS (from exploration)
# ============================================================
BETTING_PARAMS = {
    'home': {
        'min_prob': 0.48,
        'min_edge': 0.06,
        'min_odds': 1.50,
        'max_odds': 3.50,
        'enabled': True
    },
    'away': {
        'min_prob': 0.50,
        'min_edge': 0.08,
        'min_odds': 2.00,
        'max_odds': 4.00,
        'enabled': False  # Disabled - lower performance
    },
    'draw': {
        'min_prob': 0.30,
        'min_edge': 0.10,
        'min_odds': 3.00,
        'max_odds': 4.50,
        'enabled': False  # Disabled - lower performance
    }
}


def load_and_merge_data():
    """Load and merge feature sets."""
    print("Loading data...")

    base_df = pd.read_csv('data/training_data_with_market.csv')
    base_df['match_date'] = pd.to_datetime(base_df['match_date'])
    print(f"  Base data: {len(base_df):,} fixtures")

    lineup_df = pd.read_csv('data/lineup_features_v2.csv')
    lineup_df['match_date'] = pd.to_datetime(lineup_df['match_date'])
    print(f"  Lineup features: {len(lineup_df):,} fixtures")

    lineup_cols = [c for c in lineup_df.columns
                   if c not in ['match_date', 'home_team_id', 'away_team_id']]

    merged = base_df.merge(lineup_df[lineup_cols], on='fixture_id', how='inner')
    print(f"  Merged: {len(merged):,} fixtures")

    return merged


def get_feature_columns(df: pd.DataFrame):
    """
    Get pure statistical features - NO ODDS.

    Strictly excludes all betting market features.
    """
    # Patterns to exclude
    exclude_patterns = [
        # Metadata and targets
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result',
        # ALL odds-related features
        'odds', 'implied', 'bookmaker', 'sharp', 'soft', 'disagreement',
        'market_home', 'market_away', 'market_draw', 'market_overround',
        'ah_', 'ou_', 'num_bookmakers', 'over_2_5', 'under_2_5',
    ]

    def should_exclude(col):
        col_lower = col.lower()
        for pattern in exclude_patterns:
            if pattern.lower() in col_lower:
                return True
        return False

    feature_cols = [c for c in df.columns
                    if not should_exclude(c)
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    # Categorize for logging
    lineup_features = [c for c in feature_cols if 'lineup' in c.lower()]
    elo_features = [c for c in feature_cols if 'elo' in c.lower()]
    form_features = [c for c in feature_cols if any(x in c.lower() for x in ['_5', '_10', 'form', 'streak', 'trend'])]

    print(f"\nFeature breakdown ({len(feature_cols)} total):")
    print(f"  - Elo features: {len(elo_features)}")
    print(f"  - Form features: {len(form_features)}")
    print(f"  - Lineup features (PIT): {len(lineup_features)}")
    print(f"  - Other features: {len(feature_cols) - len(lineup_features) - len(elo_features)}")

    return feature_cols


def engineer_features(df, feature_cols):
    """Add engineered interaction features."""
    df = df.copy()
    new_features = []

    # Elo x Lineup interaction
    if 'elo_diff' in df.columns and 'lineup_rating_diff' in df.columns:
        df['elo_x_lineup'] = df['elo_diff'] * df['lineup_rating_diff']
        new_features.append('elo_x_lineup')

    # Elo x Position interaction
    if 'elo_diff' in df.columns and 'position_diff' in df.columns:
        df['elo_x_position'] = df['elo_diff'] * df['position_diff']
        new_features.append('elo_x_position')

    # Elo ratio
    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
        new_features.append('elo_ratio')

    # Squared elo diff (non-linear)
    if 'elo_diff' in df.columns:
        df['elo_diff_sq'] = df['elo_diff'] ** 2 * np.sign(df['elo_diff'])
        new_features.append('elo_diff_sq')

    # Lineup quality gap
    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_quality_gap'] = abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating'])
        new_features.append('lineup_quality_gap')

    print(f"  Added {len(new_features)} engineered features")

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

    # Calibrate
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


def backtest_optimal_strategy(df_test, cal_probs, params=BETTING_PARAMS):
    """
    Backtest using optimized betting parameters.

    Default: Home-only strategy with tight thresholds.
    """
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
        hp = params['home']
        if hp['enabled']:
            h_edge = p_home - h_implied
            if (hp['min_odds'] <= h_odds <= hp['max_odds'] and
                p_home >= hp['min_prob'] and
                h_edge >= hp['min_edge']):
                candidates.append({
                    'outcome': 'Home', 'idx': 2,
                    'odds': h_odds, 'prob': p_home, 'edge': h_edge
                })

        # Away bet
        ap = params['away']
        if ap['enabled']:
            a_edge = p_away - a_implied
            if (ap['min_odds'] <= a_odds <= ap['max_odds'] and
                p_away >= ap['min_prob'] and
                a_edge >= ap['min_edge']):
                candidates.append({
                    'outcome': 'Away', 'idx': 0,
                    'odds': a_odds, 'prob': p_away, 'edge': a_edge
                })

        # Draw bet
        dp = params['draw']
        if dp['enabled']:
            d_edge = p_draw - d_implied
            if (dp['min_odds'] <= d_odds <= dp['max_odds'] and
                p_draw >= dp['min_prob'] and
                d_edge >= dp['min_edge']):
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
            'model_prob': best['prob'],
            'edge': best['edge'],
            'won': won,
            'pnl': pnl,
            'month': pd.to_datetime(row['match_date']).strftime('%Y-%m')
        })

    return pd.DataFrame(results)


def analyze_results(results_df):
    """Detailed analysis of backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS - Production Statistical Model")
    print("=" * 70)

    if len(results_df) == 0:
        print("No bets placed!")
        return None

    total = len(results_df)
    wins = results_df['won'].sum()
    wr = wins / total * 100
    pnl = results_df['pnl'].sum()
    roi = pnl / total * 100

    print(f"\nOVERALL: {total} bets, {wr:.1f}% WR, {pnl:.2f} units PnL, {roi:.1f}% ROI")

    # Monthly breakdown
    print("\nMONTHLY PERFORMANCE:")
    monthly = results_df.groupby('month').agg({
        'won': ['sum', 'count'],
        'pnl': 'sum'
    })
    monthly.columns = ['wins', 'bets', 'pnl']
    monthly['wr'] = (monthly['wins'] / monthly['bets'] * 100).round(1)
    monthly['roi'] = (monthly['pnl'] / monthly['bets'] * 100).round(1)

    profitable_months = 0
    wr50_months = 0

    for month, row in monthly.iterrows():
        wr_ok = "✓" if row['wr'] >= 50 else "✗"
        roi_ok = "✓" if row['roi'] >= 0 else "✗"
        if row['wr'] >= 50:
            wr50_months += 1
        if row['roi'] >= 0:
            profitable_months += 1
        print(f"  {month}: {row['bets']:.0f} bets, {row['wr']:.1f}% WR {wr_ok}, {row['roi']:.1f}% ROI {roi_ok}")

    print(f"\nMonths with WR >= 50%: {wr50_months}/{len(monthly)}")
    print(f"Profitable months: {profitable_months}/{len(monthly)}")

    # By outcome
    print("\nBY OUTCOME:")
    for outcome in ['Home', 'Away', 'Draw']:
        sub = results_df[results_df['outcome'] == outcome]
        if len(sub) > 0:
            wr = sub['won'].mean() * 100
            roi = sub['pnl'].sum() / len(sub) * 100
            avg_odds = sub['odds'].mean()
            avg_edge = sub['edge'].mean() * 100
            print(f"  {outcome}: {len(sub)} bets, {wr:.1f}% WR, {roi:.1f}% ROI, avg odds {avg_odds:.2f}, avg edge {avg_edge:.1f}%")

    return monthly


def show_feature_importance(model, feature_cols, n=25):
    """Show top features."""
    importance = model.get_feature_importance()
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop {n} Most Important Features:")
    for i, row in imp_df.head(n).iterrows():
        tag = ""
        if 'lineup' in row['feature'].lower():
            tag = "[LINEUP]"
        elif 'elo' in row['feature'].lower():
            tag = "[ELO]"
        elif any(x in row['feature'].lower() for x in ['_5', '_10']):
            tag = "[FORM]"
        print(f"  {row['importance']:6.2f}: {row['feature']} {tag}")

    return imp_df


def main():
    # Load data
    df = load_and_merge_data()

    # Filter to recent data
    df = df[df['match_date'] >= '2019-01-01'].copy()
    df = df.sort_values('match_date').reset_index(drop=True)
    print(f"\nFiltered to {len(df):,} fixtures from 2019+")

    # Get features
    feature_cols = get_feature_columns(df)

    # Engineer additional features
    df, all_features = engineer_features(df, feature_cols)

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

    # Evaluate accuracy
    print("\nTest set evaluation:")
    raw_probs = model.predict_proba(X_test)
    cal_probs = apply_calibration(raw_probs, calibrators)
    preds = cal_probs.argmax(axis=1)
    print(f"  Accuracy: {accuracy_score(y_test, preds):.1%}")

    # Backtest
    results = backtest_optimal_strategy(df_test, cal_probs)
    monthly = analyze_results(results)

    # Feature importance
    imp_df = show_feature_importance(model, all_features)

    # Save production model
    model_data = {
        'model': model,
        'calibrators': calibrators,
        'features': all_features,
        'betting_params': BETTING_PARAMS,
        'trained_date': datetime.now().isoformat(),
        'train_period': f"{df.iloc[:train_end]['match_date'].min().date()} to {df.iloc[:train_end]['match_date'].max().date()}",
        'performance': {
            'total_bets': len(results),
            'win_rate': results['won'].mean() * 100 if len(results) > 0 else 0,
            'roi': results['pnl'].sum() / len(results) * 100 if len(results) > 0 else 0
        }
    }

    Path('models').mkdir(exist_ok=True)
    joblib.dump(model_data, 'models/production_statistical_model.joblib')
    print(f"\n✓ Model saved to: models/production_statistical_model.joblib")

    # Save backtest results
    results.to_csv('data/backtest_production_statistical.csv', index=False)
    imp_df.to_csv('data/feature_importance_production.csv', index=False)
    print(f"✓ Backtest results saved to: data/backtest_production_statistical.csv")

    return model, calibrators, results


if __name__ == '__main__':
    model, calibrators, results = main()
