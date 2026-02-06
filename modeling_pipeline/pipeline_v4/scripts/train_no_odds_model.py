"""
Train Pure Statistical Model (No Odds Features)

Uses only features that are:
1. Point-in-time correct
2. Not derived from betting markets

This isolates the predictive power of:
- Elo ratings
- League standings
- Recent form
- Historical player ratings (from our PIT database)
- xG and other match statistics
"""

import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


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


def prepare_features(df: pd.DataFrame):
    """Prepare features, EXCLUDING all odds-related columns."""

    # Columns to exclude
    exclude_patterns = [
        # Metadata
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result',
        # ALL odds-related features
        'odds', 'implied', 'bookmaker', 'sharp', 'soft', 'disagreement',
        'market_home', 'market_away', 'market_draw', 'ah_', 'ou_',
        'num_bookmakers',
    ]

    def should_exclude(col):
        col_lower = col.lower()
        for pattern in exclude_patterns:
            if pattern.lower() in col_lower:
                return True
        return False

    # Get numeric features that aren't excluded
    feature_cols = [c for c in df.columns
                    if not should_exclude(c)
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"\nUsing {len(feature_cols)} features (NO ODDS)")

    # Categorize
    lineup_features = [c for c in feature_cols if 'lineup' in c.lower()]
    elo_features = [c for c in feature_cols if 'elo' in c.lower()]
    form_features = [c for c in feature_cols if any(x in c.lower() for x in ['_5', '_10', 'form', 'streak', 'trend'])]

    print(f"  - Elo features: {len(elo_features)}")
    print(f"  - Form features: {len(form_features)}")
    print(f"  - Lineup features (PIT): {len(lineup_features)}")
    print(f"  - Other features: {len(feature_cols) - len(lineup_features) - len(elo_features)}")

    # Target
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y = df['result'].map(result_map)

    X = df[feature_cols].fillna(0)

    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val):
    """Train CatBoost."""
    print("\nTraining model...")

    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.02,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
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
    cal_probs = cal_probs / np.where(row_sums == 0, 1, row_sums)
    return cal_probs


def backtest_with_market_odds(df_test, cal_probs, min_edge=0.05, min_prob=0.35):
    """
    Backtest: Use model probabilities but bet against MARKET odds.

    The model predicts without seeing odds.
    We then compare model probability vs market implied probability.
    Bet when model sees higher probability than market.
    """
    results = []

    for i in range(len(df_test)):
        row = df_test.iloc[i]

        p_away = cal_probs[i, 0]
        p_draw = cal_probs[i, 1]
        p_home = cal_probs[i, 2]

        # Market odds (for evaluation only - model didn't see these)
        h_odds = row.get('home_best_odds', 0)
        d_odds = row.get('draw_best_odds', 0)
        a_odds = row.get('away_best_odds', 0)

        if not h_odds or not d_odds or not a_odds:
            continue

        h_implied = 1 / h_odds if h_odds > 1 else 0.5
        d_implied = 1 / d_odds if d_odds > 1 else 0.25
        a_implied = 1 / a_odds if a_odds > 1 else 0.35

        result_map = {'A': 0, 'D': 1, 'H': 2}
        actual = result_map.get(row['result'], -1)
        if actual == -1:
            continue

        candidates = []

        # Home
        h_edge = p_home - h_implied
        if h_odds >= 1.50 and p_home >= min_prob and h_edge >= min_edge:
            candidates.append({
                'outcome': 'Home', 'idx': 2,
                'odds': h_odds, 'prob': p_home, 'edge': h_edge
            })

        # Away
        a_edge = p_away - a_implied
        if a_odds >= 1.50 and p_away >= min_prob and a_edge >= min_edge:
            candidates.append({
                'outcome': 'Away', 'idx': 0,
                'odds': a_odds, 'prob': p_away, 'edge': a_edge
            })

        # Draw
        d_edge = p_draw - d_implied
        if d_odds >= 3.00 and p_draw >= 0.28 and d_edge >= 0.08:
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
    """Analyze results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (Pure Statistical Model - No Odds Features)")
    print("=" * 70)

    if len(results_df) == 0:
        print("No bets placed!")
        return None

    total = len(results_df)
    wins = results_df['won'].sum()
    wr = wins / total * 100
    pnl = results_df['pnl'].sum()
    roi = pnl / total * 100

    print(f"\nOVERALL: {total} bets, {wr:.1f}% WR, ${pnl:.2f} PnL, {roi:.1f}% ROI")

    # Monthly
    print("\nBY MONTH:")
    monthly = results_df.groupby('month').agg({
        'won': ['sum', 'count'],
        'pnl': 'sum'
    })
    monthly.columns = ['wins', 'bets', 'pnl']
    monthly['wr'] = (monthly['wins'] / monthly['bets'] * 100).round(1)
    monthly['roi'] = (monthly['pnl'] / monthly['bets'] * 100).round(1)

    constraints_met = 0
    for month, row in monthly.iterrows():
        wr_ok = "✓" if row['wr'] >= 50 else "✗"
        roi_ok = "✓" if row['roi'] >= 10 else "✗"
        if row['wr'] >= 50 and row['roi'] >= 10:
            constraints_met += 1
        print(f"  {month}: {row['bets']:.0f} bets, {row['wr']:.1f}% WR {wr_ok}, {row['roi']:.1f}% ROI {roi_ok}")

    print(f"\nMonths meeting both constraints: {constraints_met}/{len(monthly)}")

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


def show_feature_importance(model, feature_cols, n=30):
    """Show top features."""
    importance = model.get_feature_importance()
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop {n} Features (No Odds):")
    for _, row in imp_df.head(n).iterrows():
        tag = ""
        if 'lineup' in row['feature'].lower():
            tag = "[LINEUP-PIT]"
        elif 'elo' in row['feature'].lower():
            tag = "[ELO]"
        elif any(x in row['feature'].lower() for x in ['_5', '_10']):
            tag = "[FORM]"
        print(f"  {row['importance']:.2f}: {row['feature']} {tag}")

    return imp_df


def main():
    # Load
    df = load_and_merge_data()

    # Filter recent
    df = df[df['match_date'] >= '2020-01-01'].copy()
    df = df.sort_values('match_date').reset_index(drop=True)
    print(f"\nFiltered to {len(df):,} fixtures from 2020+")

    # Prepare
    X, y, feature_cols = prepare_features(df)

    # Split
    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    df_test = df.iloc[val_end:].copy()

    print(f"\nSplits:")
    print(f"  Train: {len(X_train):,} ({df.iloc[:train_end]['match_date'].min().date()} to {df.iloc[:train_end]['match_date'].max().date()})")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,} ({df_test['match_date'].min().date()} to {df_test['match_date'].max().date()})")

    # Train
    model, calibrators = train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\nTest accuracy:")
    raw_probs = model.predict_proba(X_test)
    cal_probs = apply_calibration(raw_probs, calibrators)
    preds = cal_probs.argmax(axis=1)
    print(f"  {accuracy_score(y_test, preds):.1%}")

    # Backtest
    results = backtest_with_market_odds(df_test, cal_probs)
    analyze_results(results)

    # Feature importance
    imp_df = show_feature_importance(model, feature_cols)

    # Save
    joblib.dump({
        'model': model,
        'calibrators': calibrators,
        'features': feature_cols
    }, 'models/no_odds_model.joblib')

    results.to_csv('data/backtest_no_odds.csv', index=False)
    imp_df.to_csv('data/feature_importance_no_odds.csv', index=False)

    print("\nSaved model and results")

    return model, results, imp_df


if __name__ == '__main__':
    model, results, importance = main()
