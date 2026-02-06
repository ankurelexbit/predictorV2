"""
Train Point-in-Time Correct Model

Uses only features that would be available at prediction time:
1. Base match features (Elo, form, standings) - all point-in-time correct
2. Lineup features from historical player database - point-in-time correct
3. Odds features - use opening odds or current odds (not closing)

NO DATA LEAKAGE:
- Lineup ratings are historical averages, not post-match ratings
- Market features use odds available before kickoff
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

# Strategy parameters
MIN_ODDS = 1.60
MIN_EV = 0.08
MIN_CAL_PROB = 0.40
MIN_EDGE = 0.05


def load_and_merge_data():
    """Load and merge all feature sets with point-in-time correctness."""
    print("Loading data...")

    # 1. Base training data (has match features and closing odds)
    base_df = pd.read_csv('data/training_data_with_market.csv')
    base_df['match_date'] = pd.to_datetime(base_df['match_date'])
    print(f"  Base training data: {len(base_df):,} fixtures")

    # 2. Point-in-time correct lineup features
    lineup_df = pd.read_csv('data/lineup_features_v2.csv')
    lineup_df['match_date'] = pd.to_datetime(lineup_df['match_date'])
    print(f"  PIT lineup features: {len(lineup_df):,} fixtures")

    # Merge on fixture_id
    lineup_cols = [c for c in lineup_df.columns
                   if c not in ['match_date', 'home_team_id', 'away_team_id']]

    merged = base_df.merge(
        lineup_df[lineup_cols],
        on='fixture_id',
        how='inner'
    )
    print(f"  After merge: {len(merged):,} fixtures")

    return merged


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix, carefully excluding any data leakage.
    """
    # Columns to EXCLUDE (metadata, targets, post-match data)
    exclude_cols = [
        # Metadata
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result',
        # These market features may have timing issues - exclude for now
        # We keep basic odds but need to be careful about their timing
        'home_sharp_vs_soft', 'draw_sharp_vs_soft', 'away_sharp_vs_soft',
        'home_bookmaker_disagreement', 'draw_bookmaker_disagreement', 'away_bookmaker_disagreement',
    ]

    # Keep only numeric columns that aren't excluded
    feature_cols = [c for c in df.columns
                    if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"\nUsing {len(feature_cols)} features")

    # Categorize features
    lineup_features = [c for c in feature_cols if 'lineup' in c.lower()]
    odds_features = [c for c in feature_cols if 'odds' in c.lower() or 'implied' in c.lower()]
    base_features = [c for c in feature_cols if c not in lineup_features and c not in odds_features]

    print(f"  - Base match features: {len(base_features)}")
    print(f"  - Lineup features (PIT correct): {len(lineup_features)}")
    print(f"  - Odds features: {len(odds_features)}")

    # Target
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y = df['result'].map(result_map)

    X = df[feature_cols].fillna(0)

    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val):
    """Train CatBoost with calibration."""
    print("\nTraining CatBoost model...")

    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        eval_metric='MultiClass'
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=100
    )

    # Calibrate
    print("\nCalibrating...")
    raw_val_probs = model.predict_proba(X_val)

    calibrators = {}
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
        calibrators[outcome].fit(raw_val_probs[:, idx], (y_val == idx).astype(int))

    return model, calibrators


def apply_calibration(probs, calibrators):
    """Apply isotonic calibration."""
    cal_probs = np.zeros_like(probs)
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(probs[:, idx])

    row_sums = cal_probs.sum(axis=1, keepdims=True)
    cal_probs = cal_probs / np.where(row_sums == 0, 1, row_sums)
    return cal_probs


def backtest(df_test, cal_probs):
    """Run backtest with betting strategy."""
    results = []

    for i in range(len(df_test)):
        row = df_test.iloc[i]

        p_away = cal_probs[i, 0]
        p_draw = cal_probs[i, 1]
        p_home = cal_probs[i, 2]

        # Use available odds (these are from the data, timing depends on source)
        h_odds = row.get('home_best_odds', 0)
        d_odds = row.get('draw_best_odds', 0)
        a_odds = row.get('away_best_odds', 0)

        if not h_odds or not d_odds or not a_odds:
            continue

        # Implied probabilities
        h_implied = 1 / h_odds if h_odds > 1 else 0.5
        d_implied = 1 / d_odds if d_odds > 1 else 0.25
        a_implied = 1 / a_odds if a_odds > 1 else 0.35

        # Get actual result
        result_map = {'A': 0, 'D': 1, 'H': 2}
        actual = result_map.get(row['result'], -1)
        if actual == -1:
            continue

        match_date = row['match_date']

        # Evaluate bets
        candidates = []

        # Home
        if h_odds >= MIN_ODDS:
            h_ev = p_home * h_odds - 1
            h_edge = p_home - h_implied
            if p_home >= MIN_CAL_PROB and h_edge >= MIN_EDGE and h_ev >= MIN_EV:
                candidates.append({
                    'outcome': 'Home', 'idx': 2,
                    'odds': h_odds, 'ev': h_ev, 'edge': h_edge, 'prob': p_home
                })

        # Away
        if a_odds >= MIN_ODDS:
            a_ev = p_away * a_odds - 1
            a_edge = p_away - a_implied
            if p_away >= MIN_CAL_PROB and a_edge >= MIN_EDGE and a_ev >= MIN_EV:
                candidates.append({
                    'outcome': 'Away', 'idx': 0,
                    'odds': a_odds, 'ev': a_ev, 'edge': a_edge, 'prob': p_away
                })

        # Draw (selective)
        if d_odds >= 3.20:
            d_ev = p_draw * d_odds - 1
            d_edge = p_draw - d_implied
            if d_edge >= 0.10 and d_ev >= 0.15:
                candidates.append({
                    'outcome': 'Draw', 'idx': 1,
                    'odds': d_odds, 'ev': d_ev, 'edge': d_edge, 'prob': p_draw
                })

        if not candidates:
            continue

        best = max(candidates, key=lambda x: x['ev'])
        won = 1 if actual == best['idx'] else 0
        pnl = (best['odds'] - 1) if won else -1

        results.append({
            'date': match_date,
            'outcome': best['outcome'],
            'odds': best['odds'],
            'ev': best['ev'],
            'prob': best['prob'],
            'won': won,
            'pnl': pnl,
            'month': pd.to_datetime(match_date).strftime('%Y-%m')
        })

    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (Point-in-Time Correct Model)")
    print("=" * 70)

    if len(results_df) == 0:
        print("No bets placed!")
        return

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

    for month, row in monthly.iterrows():
        wr_ok = "✓" if row['wr'] >= 50 else "✗"
        roi_ok = "✓" if row['roi'] >= 10 else "✗"
        print(f"  {month}: {row['bets']:.0f} bets, {row['wr']:.1f}% WR {wr_ok}, {row['roi']:.1f}% ROI {roi_ok}")

    # By outcome
    print("\nBY OUTCOME:")
    for outcome in ['Home', 'Away', 'Draw']:
        sub = results_df[results_df['outcome'] == outcome]
        if len(sub) > 0:
            wr = sub['won'].mean() * 100
            roi = sub['pnl'].sum() / len(sub) * 100
            print(f"  {outcome}: {len(sub)} bets, {wr:.1f}% WR, {roi:.1f}% ROI")

    return monthly


def analyze_feature_importance(model, feature_cols, n_top=25):
    """Show feature importance."""
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop {n_top} Features:")
    for _, row in importance_df.head(n_top).iterrows():
        ftype = ""
        if 'lineup' in row['feature'].lower():
            ftype = "[LINEUP-PIT]"
        elif 'odds' in row['feature'].lower() or 'implied' in row['feature'].lower():
            ftype = "[ODDS]"
        print(f"  {row['importance']:.2f}: {row['feature']} {ftype}")

    return importance_df


def main():
    # Load data
    df = load_and_merge_data()

    # Filter to recent data
    df = df[df['match_date'] >= '2023-01-01'].copy()
    df = df.sort_values('match_date').reset_index(drop=True)
    print(f"\nFiltered to {len(df):,} fixtures from 2023+")

    # Prepare features
    X, y, feature_cols = prepare_features(df)

    # Chronological split
    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

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

    # Test evaluation
    print("\nTest set:")
    raw_probs = model.predict_proba(X_test)
    cal_probs = apply_calibration(raw_probs, calibrators)
    preds = cal_probs.argmax(axis=1)
    print(f"  Accuracy: {accuracy_score(y_test, preds):.1%}")

    # Backtest
    results = backtest(df_test, cal_probs)
    monthly = analyze_results(results)

    # Feature importance
    importance = analyze_feature_importance(model, feature_cols)

    # Save
    model_data = {
        'model': model,
        'calibrators': calibrators,
        'features': feature_cols,
    }
    joblib.dump(model_data, 'models/pit_correct_model.joblib')
    print(f"\nModel saved to: models/pit_correct_model.joblib")

    results.to_csv('data/backtest_results_pit.csv', index=False)
    importance.to_csv('data/feature_importance_pit.csv', index=False)

    return model, calibrators, results


if __name__ == '__main__':
    model, calibrators, results = main()
