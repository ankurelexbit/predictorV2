"""
Train Model Using 1-Hour-Before Features

Combines:
1. Base match features (Elo, form, standings, etc.)
2. Market features available 1hr before kickoff
3. Lineup features (available at lineup announcement ~1hr before)

This creates a realistic model for pre-match predictions.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Strategy parameters (can be tuned)
MIN_ODDS = 1.60
MIN_EV = 0.08
MIN_CAL_PROB = 0.40
MIN_EDGE = 0.05


def load_and_merge_data():
    """Load and merge all feature sets"""
    print("Loading data...")

    # 1. Base training data
    base_df = pd.read_csv('data/training_data_with_market.csv')
    base_df['match_date'] = pd.to_datetime(base_df['match_date'])
    print(f"  Base training data: {len(base_df):,} fixtures")

    # 2. 1hr market features
    market_1hr = pd.read_csv('data/market_features_1hr.csv')
    market_1hr['match_date'] = pd.to_datetime(market_1hr['match_date'])
    print(f"  1hr market features: {len(market_1hr):,} fixtures")

    # 3. Lineup features
    lineup_df = pd.read_csv('data/lineup_features.csv')
    lineup_df['match_date'] = pd.to_datetime(lineup_df['match_date'])
    print(f"  Lineup features: {len(lineup_df):,} fixtures")

    # Merge on fixture_id
    # First, rename 1hr market columns to avoid conflicts with closing odds
    market_1hr_cols = [c for c in market_1hr.columns if c not in ['fixture_id', 'match_date']]
    market_1hr_renamed = market_1hr.rename(columns={c: c for c in market_1hr.columns})

    # Merge base with 1hr market
    merged = base_df.merge(
        market_1hr_renamed[['fixture_id'] + market_1hr_cols],
        on='fixture_id',
        how='inner'
    )
    print(f"  After market merge: {len(merged):,} fixtures")

    # Merge with lineup features
    lineup_cols = [c for c in lineup_df.columns if c not in ['fixture_id', 'match_date', 'home_team_id', 'away_team_id']]
    merged = merged.merge(
        lineup_df[['fixture_id'] + lineup_cols],
        on='fixture_id',
        how='inner'
    )
    print(f"  After lineup merge: {len(merged):,} fixtures")

    return merged


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix"""
    # Columns to exclude from features
    exclude_cols = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result', 'target', 'match_date_x', 'match_date_y',
        # Exclude closing odds features (data leakage)
        'home_best_odds', 'draw_best_odds', 'away_best_odds',
        'home_implied_prob', 'draw_implied_prob', 'away_implied_prob',
        'home_sharp_vs_soft', 'draw_sharp_vs_soft', 'away_sharp_vs_soft',
        'home_bookmaker_disagreement', 'draw_bookmaker_disagreement', 'away_bookmaker_disagreement',
        'num_bookmakers', 'ah_main_line', 'ou_main_line', 'ou_over_odds', 'ou_under_odds',
        # Formation string (categorical, not numeric)
        'home_formation', 'away_formation'
    ]

    # Keep only numeric columns that aren't excluded
    feature_cols = [c for c in df.columns
                    if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"\nUsing {len(feature_cols)} features")

    # Show 1hr and lineup features specifically
    hr_features = [c for c in feature_cols if '_1hr' in c]
    lineup_features = [c for c in feature_cols if c.startswith(('home_', 'away_')) and
                       any(x in c for x in ['rating', 'sidelined', 'starters', 'formation', 'defender', 'midfielder', 'forward', 'goalkeeper'])]

    print(f"  - 1hr market features: {len(hr_features)}")
    print(f"  - Lineup features: {len(lineup_features)}")

    X = df[feature_cols].fillna(0)
    # Target is 'result' column: H=2, D=1, A=0
    if 'target' in df.columns:
        y = df['target']
    elif 'result' in df.columns:
        # Map H/D/A to numeric
        result_map = {'A': 0, 'D': 1, 'H': 2}
        y = df['result'].map(result_map)
    else:
        raise ValueError("No target column found")

    return X, y, feature_cols


def train_with_calibration(X_train, y_train, X_val, y_val):
    """Train CatBoost with isotonic calibration"""
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
    print("\nCalibrating probabilities...")
    raw_val_probs = model.predict_proba(X_val)

    calibrators = {}
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
        calibrators[outcome].fit(raw_val_probs[:, idx], (y_val == idx).astype(int))

    return model, calibrators


def apply_calibration(probs, calibrators):
    """Apply isotonic calibration"""
    cal_probs = np.zeros_like(probs)
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(probs[:, idx])

    # Normalize
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    cal_probs = cal_probs / np.where(row_sums == 0, 1, row_sums)
    return cal_probs


def backtest_strategy(df_test, cal_probs, features):
    """Backtest the betting strategy"""
    results = []

    for i in range(len(df_test)):
        row = df_test.iloc[i]

        # Calibrated probabilities
        p_away = cal_probs[i, 0]
        p_draw = cal_probs[i, 1]
        p_home = cal_probs[i, 2]

        # Use 1hr odds for evaluation
        h_odds = row.get('home_best_odds_1hr')
        d_odds = row.get('draw_best_odds_1hr')
        a_odds = row.get('away_best_odds_1hr')

        # Skip if no odds
        if pd.isna(h_odds) or pd.isna(d_odds) or pd.isna(a_odds):
            continue

        # Implied probabilities from 1hr odds
        h_implied = 1 / h_odds if h_odds > 1 else 0.5
        d_implied = 1 / d_odds if d_odds > 1 else 0.25
        a_implied = 1 / a_odds if a_odds > 1 else 0.35

        # Get actual result
        if 'target' in df_test.columns:
            actual = int(row['target'])
        else:
            result_map = {'A': 0, 'D': 1, 'H': 2}
            actual = result_map.get(row['result'], -1)
            if actual == -1:
                continue
        match_date = row['match_date']

        # Evaluate each outcome
        candidates = []

        # Home bet
        if h_odds >= MIN_ODDS:
            h_ev = p_home * h_odds - 1
            h_edge = p_home - h_implied
            if p_home >= MIN_CAL_PROB and h_edge >= MIN_EDGE and h_ev >= MIN_EV:
                candidates.append({
                    'outcome': 'Home',
                    'outcome_idx': 2,
                    'odds': h_odds,
                    'ev': h_ev,
                    'edge': h_edge,
                    'prob': p_home
                })

        # Away bet
        if a_odds >= MIN_ODDS:
            a_ev = p_away * a_odds - 1
            a_edge = p_away - a_implied
            if p_away >= MIN_CAL_PROB and a_edge >= MIN_EDGE and a_ev >= MIN_EV:
                candidates.append({
                    'outcome': 'Away',
                    'outcome_idx': 0,
                    'odds': a_odds,
                    'ev': a_ev,
                    'edge': a_edge,
                    'prob': p_away
                })

        # Draw bet (more selective)
        if d_odds >= 3.20:
            d_ev = p_draw * d_odds - 1
            d_edge = p_draw - d_implied
            # Check for sharp book signal from 1hr data
            d_sharp = row.get('draw_sharp_vs_soft_1hr', 0) or 0
            d_disagree = row.get('draw_disagreement_1hr', 0) or 0
            if d_sharp > 0.02 and d_edge >= 0.10 and d_ev >= 0.15 and d_disagree > 0.03:
                candidates.append({
                    'outcome': 'Draw',
                    'outcome_idx': 1,
                    'odds': d_odds,
                    'ev': d_ev,
                    'edge': d_edge,
                    'prob': p_draw
                })

        if not candidates:
            continue

        # Select best by EV
        best = max(candidates, key=lambda x: x['ev'])

        # Determine win/loss
        won = 1 if actual == best['outcome_idx'] else 0
        pnl = (best['odds'] - 1) if won else -1

        results.append({
            'date': match_date,
            'outcome': best['outcome'],
            'odds': best['odds'],
            'ev': best['ev'],
            'edge': best['edge'],
            'prob': best['prob'],
            'won': won,
            'pnl': pnl,
            'month': pd.to_datetime(match_date).strftime('%Y-%m')
        })

    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze backtest results"""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (1hr-Before Model)")
    print("=" * 70)

    if len(results_df) == 0:
        print("No bets placed!")
        return

    # Overall
    total_bets = len(results_df)
    wins = results_df['won'].sum()
    wr = wins / total_bets * 100
    total_pnl = results_df['pnl'].sum()
    roi = total_pnl / total_bets * 100

    print(f"\nOVERALL: {total_bets} bets, {wr:.1f}% WR, ${total_pnl:.2f} PnL, {roi:.1f}% ROI")

    # By month
    print("\nBY MONTH:")
    monthly = results_df.groupby('month').agg({
        'won': ['sum', 'count'],
        'pnl': 'sum'
    })
    monthly.columns = ['wins', 'bets', 'pnl']
    monthly['wr'] = (monthly['wins'] / monthly['bets'] * 100).round(1)
    monthly['roi'] = (monthly['pnl'] / monthly['bets'] * 100).round(1)

    all_constraints_met = True
    for month, row in monthly.iterrows():
        wr_ok = row['wr'] >= 50
        roi_ok = row['roi'] >= 10
        status = "✓" if (wr_ok and roi_ok) else "✗"
        if not (wr_ok and roi_ok):
            all_constraints_met = False
        print(f"  {month}: {row['bets']:.0f} bets, {row['wr']:.1f}% WR {'✓' if wr_ok else '✗'}, {row['roi']:.1f}% ROI {'✓' if roi_ok else '✗'}")

    # By outcome
    print("\nBY OUTCOME:")
    for outcome in ['Home', 'Draw', 'Away']:
        subset = results_df[results_df['outcome'] == outcome]
        if len(subset) > 0:
            wr = subset['won'].mean() * 100
            roi = subset['pnl'].sum() / len(subset) * 100
            print(f"  {outcome}: {len(subset)} bets, {wr:.1f}% WR, {roi:.1f}% ROI")

    print("\n" + "=" * 70)
    if all_constraints_met:
        print("✓ ALL CONSTRAINTS MET (WR≥50%, ROI≥10% per month)")
    else:
        print("✗ CONSTRAINTS NOT MET - need further optimization")
    print("=" * 70)

    return monthly


def analyze_feature_importance(model, feature_cols, n_top=20):
    """Analyze feature importance"""
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop {n_top} Features:")
    for i, row in importance_df.head(n_top).iterrows():
        feature_type = ""
        if '_1hr' in row['feature']:
            feature_type = "[1HR MARKET]"
        elif any(x in row['feature'] for x in ['rating', 'sidelined', 'starters', 'defender', 'midfielder', 'forward', 'goalkeeper']):
            feature_type = "[LINEUP]"
        print(f"  {row['importance']:.1f}: {row['feature']} {feature_type}")

    return importance_df


def main():
    # Load and merge data
    df = load_and_merge_data()

    # Filter to recent data for realistic backtest
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
    print(f"  Train: {len(X_train):,} ({df.iloc[:train_end]['match_date'].min()} to {df.iloc[:train_end]['match_date'].max()})")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,} ({df_test['match_date'].min()} to {df_test['match_date'].max()})")

    # Train model
    model, calibrators = train_with_calibration(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    print("\nTest set evaluation:")
    raw_probs = model.predict_proba(X_test)
    cal_probs = apply_calibration(raw_probs, calibrators)

    preds = cal_probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.1%}")

    # Backtest
    results = backtest_strategy(df_test, cal_probs, feature_cols)
    monthly = analyze_results(results)

    # Feature importance
    importance_df = analyze_feature_importance(model, feature_cols)

    # Save model
    model_data = {
        'model': model,
        'calibrators': calibrators,
        'features': feature_cols,
        'strategy_params': {
            'MIN_ODDS': MIN_ODDS,
            'MIN_EV': MIN_EV,
            'MIN_CAL_PROB': MIN_CAL_PROB,
            'MIN_EDGE': MIN_EDGE
        }
    }
    joblib.dump(model_data, 'models/1hr_model_v1.joblib')
    print(f"\nModel saved to: models/1hr_model_v1.joblib")

    # Save results
    results.to_csv('data/backtest_results_1hr.csv', index=False)
    importance_df.to_csv('data/feature_importance_1hr.csv', index=False)

    return model, calibrators, results, importance_df


if __name__ == '__main__':
    model, calibrators, results, importance = main()
