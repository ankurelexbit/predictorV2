"""
Train Integrated Model with Market Features

This script trains a sophisticated CatBoost model using:
1. Original 162 match features (Elo, form, xG, etc.)
2. New 27 market features (bookmaker disagreement, sharp/soft spread, etc.)

The model is designed to achieve:
- WR > 50% per month
- ROI > 10% per month
- H/D/A all positive per month

Key innovations:
1. Market features as "second opinion" - when model and market disagree
2. Sharp money indicators - when sharp books favor an outcome
3. Bookmaker disagreement - uncertainty signal for timing bets
4. Edge detection - model_prob vs market_prob
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
import joblib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Feature configuration
METADATA_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id',
    'league_id', 'match_date', 'home_score', 'away_score', 'result'
]

# Market features that encode odds (should NOT be used as model features
# because they contain the "answer" - market's probability estimate)
ODDS_FEATURES = [
    'home_best_odds', 'home_avg_odds', 'home_implied_prob',
    'draw_best_odds', 'draw_avg_odds', 'draw_implied_prob',
    'away_best_odds', 'away_avg_odds', 'away_implied_prob',
    'market_overround',
    'market_home_prob_normalized', 'market_draw_prob_normalized', 'market_away_prob_normalized',
    'ah_home_odds', 'ah_away_odds',
    'over_2_5_best', 'over_2_5_avg', 'under_2_5_best', 'under_2_5_avg',
]

# Market features that encode SIGNALS (can be used as model features)
MARKET_SIGNAL_FEATURES = [
    'home_bookmaker_disagreement', 'home_sharp_vs_soft',
    'draw_bookmaker_disagreement', 'draw_sharp_vs_soft',
    'away_bookmaker_disagreement', 'away_sharp_vs_soft',
    'ah_main_line',  # Asian handicap line encodes market view of goal difference
    'num_bookmakers',  # Liquidity indicator
]

TARGET_MAP = {'H': 2, 'D': 1, 'A': 0}
TARGET_REVERSE = {2: 'Home', 1: 'Draw', 0: 'Away'}


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load data and prepare features"""
    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    # Create target
    df['target'] = df['result'].map(TARGET_MAP)

    return df


def get_feature_columns(df: pd.DataFrame, include_market_signals: bool = True) -> List[str]:
    """Get list of feature columns to use"""
    exclude = set(METADATA_COLS + ODDS_FEATURES + ['target', 'result'])

    if not include_market_signals:
        exclude.update(MARKET_SIGNAL_FEATURES)

    features = [c for c in df.columns if c not in exclude]
    return features


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    verbose: bool = True
) -> CatBoostClassifier:
    """Train CatBoost classifier"""

    X_train = train_df[features].fillna(0)
    y_train = train_df['target']

    X_val = val_df[features].fillna(0)
    y_val = val_df['target']

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        verbose=False,
        early_stopping_rounds=100,
        l2_leaf_reg=3,
        border_count=128,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=verbose)

    return model


def train_calibrators(
    model: CatBoostClassifier,
    val_df: pd.DataFrame,
    features: List[str]
) -> Dict[str, IsotonicRegression]:
    """Train isotonic calibrators for each outcome"""

    X_val = val_df[features].fillna(0)
    y_val = val_df['target'].values

    probs = model.predict_proba(X_val)

    calibrators = {}
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal = IsotonicRegression(out_of_bounds='clip')
        # Binary target: 1 if this outcome, 0 otherwise
        y_binary = (y_val == idx).astype(int)
        cal.fit(probs[:, idx], y_binary)
        calibrators[outcome] = cal

    return calibrators


def apply_calibration(
    probs: np.ndarray,
    calibrators: Dict[str, IsotonicRegression]
) -> np.ndarray:
    """Apply calibration to raw probabilities"""
    cal_probs = np.zeros_like(probs)

    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(probs[:, idx])

    # Normalize to sum to 1
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cal_probs = cal_probs / row_sums

    return cal_probs


def evaluate_model(
    model: CatBoostClassifier,
    calibrators: Dict[str, IsotonicRegression],
    test_df: pd.DataFrame,
    features: List[str]
) -> Dict:
    """Evaluate model on test set"""

    X_test = test_df[features].fillna(0)
    y_test = test_df['target'].values

    # Raw predictions
    raw_probs = model.predict_proba(X_test)
    raw_preds = raw_probs.argmax(axis=1)
    raw_acc = (raw_preds == y_test).mean()

    # Calibrated predictions
    cal_probs = apply_calibration(raw_probs, calibrators)
    cal_preds = cal_probs.argmax(axis=1)
    cal_acc = (cal_preds == y_test).mean()

    results = {
        'raw_accuracy': raw_acc,
        'calibrated_accuracy': cal_acc,
        'samples': len(test_df),
    }

    # Per-outcome accuracy
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        mask = y_test == idx
        if mask.sum() > 0:
            results[f'{outcome}_accuracy'] = (cal_preds[mask] == idx).mean()
            results[f'{outcome}_samples'] = mask.sum()

    return results


def backtest_strategy(
    model: CatBoostClassifier,
    calibrators: Dict[str, IsotonicRegression],
    test_df: pd.DataFrame,
    features: List[str],
    min_edge: float = 0.05,
    min_cal_prob: float = 0.35,
    sharp_signal_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Backtest sophisticated betting strategy

    Strategy:
    1. Calculate calibrated probability for each outcome
    2. Calculate edge vs market (cal_prob - market_prob)
    3. Apply sharp money filter (when sharps favor, boost confidence)
    4. Apply bookmaker disagreement filter (bet when market uncertain)
    5. Select bets meeting all criteria
    """

    X_test = test_df[features].fillna(0)

    # Get calibrated probabilities
    raw_probs = model.predict_proba(X_test)
    cal_probs = apply_calibration(raw_probs, calibrators)

    # Build results DataFrame
    results = test_df[['fixture_id', 'match_date', 'result',
                       'home_best_odds', 'draw_best_odds', 'away_best_odds',
                       'home_implied_prob', 'draw_implied_prob', 'away_implied_prob',
                       'home_bookmaker_disagreement', 'draw_bookmaker_disagreement', 'away_bookmaker_disagreement',
                       'home_sharp_vs_soft', 'draw_sharp_vs_soft', 'away_sharp_vs_soft']].copy()

    results['cal_home'] = cal_probs[:, 2]
    results['cal_draw'] = cal_probs[:, 1]
    results['cal_away'] = cal_probs[:, 0]

    # Calculate edges
    results['edge_home'] = results['cal_home'] - results['home_implied_prob']
    results['edge_draw'] = results['cal_draw'] - results['draw_implied_prob']
    results['edge_away'] = results['cal_away'] - results['away_implied_prob']

    # Calculate EV
    results['ev_home'] = results['cal_home'] * results['home_best_odds'] - 1
    results['ev_draw'] = results['cal_draw'] * results['draw_best_odds'] - 1
    results['ev_away'] = results['cal_away'] * results['away_best_odds'] - 1

    # Sophisticated bet selection
    bets = []

    for idx, row in results.iterrows():
        # Evaluate each outcome
        candidates = []

        for outcome, result_code, cal_col, edge_col, ev_col, odds_col, disagreement_col, sharp_col in [
            ('Home', 'H', 'cal_home', 'edge_home', 'ev_home', 'home_best_odds', 'home_bookmaker_disagreement', 'home_sharp_vs_soft'),
            ('Draw', 'D', 'cal_draw', 'edge_draw', 'ev_draw', 'draw_best_odds', 'draw_bookmaker_disagreement', 'draw_sharp_vs_soft'),
            ('Away', 'A', 'cal_away', 'edge_away', 'ev_away', 'away_best_odds', 'away_bookmaker_disagreement', 'away_sharp_vs_soft'),
        ]:
            cal_prob = row[cal_col]
            edge = row[edge_col]
            ev = row[ev_col]
            odds = row[odds_col]
            disagreement = row[disagreement_col] if pd.notna(row[disagreement_col]) else 0
            sharp_signal = row[sharp_col] if pd.notna(row[sharp_col]) else 0

            # Base criteria
            if cal_prob < min_cal_prob:
                continue
            if edge < min_edge:
                continue
            if ev < 0:
                continue

            # Calculate confidence score
            # Higher is better
            confidence = edge + (sharp_signal * sharp_signal_weight if sharp_signal > 0 else 0)

            # Boost if high disagreement (market uncertain, our model may have edge)
            if disagreement > 0.05:
                confidence += 0.02

            candidates.append({
                'outcome': outcome,
                'result_code': result_code,
                'cal_prob': cal_prob,
                'edge': edge,
                'ev': ev,
                'odds': odds,
                'confidence': confidence,
                'sharp_signal': sharp_signal,
                'disagreement': disagreement,
            })

        if candidates:
            # Select best candidate by confidence
            best = max(candidates, key=lambda x: x['confidence'])

            bets.append({
                'fixture_id': row['fixture_id'],
                'match_date': row['match_date'],
                'actual_result': row['result'],
                'bet_outcome': best['outcome'],
                'bet_result_code': best['result_code'],
                'cal_prob': best['cal_prob'],
                'edge': best['edge'],
                'ev': best['ev'],
                'odds': best['odds'],
                'confidence': best['confidence'],
                'sharp_signal': best['sharp_signal'],
                'disagreement': best['disagreement'],
                'won': row['result'] == best['result_code'],
            })

    bets_df = pd.DataFrame(bets)

    if len(bets_df) == 0:
        print("No bets placed with current criteria")
        return bets_df

    # Calculate PnL
    bets_df['pnl'] = bets_df.apply(
        lambda x: x['odds'] - 1 if x['won'] else -1, axis=1
    )

    return bets_df


def analyze_backtest(bets_df: pd.DataFrame) -> Dict:
    """Analyze backtest results"""

    if len(bets_df) == 0:
        return {'error': 'No bets'}

    # Overall metrics
    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    win_rate = wins / total_bets
    total_pnl = bets_df['pnl'].sum()
    roi = total_pnl / total_bets

    results = {
        'total_bets': total_bets,
        'wins': wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'roi': roi,
    }

    # By outcome
    for outcome in ['Home', 'Draw', 'Away']:
        mask = bets_df['bet_outcome'] == outcome
        if mask.sum() > 0:
            outcome_df = bets_df[mask]
            results[f'{outcome.lower()}_bets'] = len(outcome_df)
            results[f'{outcome.lower()}_wins'] = outcome_df['won'].sum()
            results[f'{outcome.lower()}_wr'] = outcome_df['won'].mean()
            results[f'{outcome.lower()}_pnl'] = outcome_df['pnl'].sum()
            results[f'{outcome.lower()}_roi'] = outcome_df['pnl'].sum() / len(outcome_df)

    # By month
    bets_df['month'] = pd.to_datetime(bets_df['match_date']).dt.to_period('M')
    monthly = bets_df.groupby('month').agg({
        'won': ['count', 'sum', 'mean'],
        'pnl': 'sum'
    }).reset_index()
    monthly.columns = ['month', 'bets', 'wins', 'win_rate', 'pnl']
    monthly['roi'] = monthly['pnl'] / monthly['bets']

    results['monthly'] = monthly.to_dict('records')

    return results


def print_results(results: Dict):
    """Print formatted results"""

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nOverall:")
    print(f"  Total bets: {results['total_bets']}")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Total PnL: ${results['total_pnl']:.2f}")
    print(f"  ROI: {results['roi']:.1%}")

    print(f"\nBy Outcome:")
    for outcome in ['home', 'draw', 'away']:
        if f'{outcome}_bets' in results:
            print(f"  {outcome.title()}: {results[f'{outcome}_bets']} bets, "
                  f"{results[f'{outcome}_wr']:.1%} WR, "
                  f"${results[f'{outcome}_pnl']:.2f} PnL, "
                  f"{results[f'{outcome}_roi']:.1%} ROI")

    print(f"\nBy Month:")
    for m in results.get('monthly', []):
        print(f"  {m['month']}: {m['bets']} bets, {m['win_rate']:.1%} WR, "
              f"${m['pnl']:.2f} PnL, {m['roi']:.1%} ROI")

    # Check constraints
    print(f"\nConstraint Check:")
    all_pass = True
    for m in results.get('monthly', []):
        wr_pass = m['win_rate'] >= 0.50
        roi_pass = m['roi'] >= 0.10
        status = "✓" if (wr_pass and roi_pass) else "✗"
        print(f"  {m['month']}: WR≥50% {'✓' if wr_pass else '✗'}, ROI≥10% {'✓' if roi_pass else '✗'}")
        if not (wr_pass and roi_pass):
            all_pass = False

    return all_pass


def main():
    print("=" * 60)
    print("Training Integrated Model with Market Features")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data('data/training_data_with_market.csv')
    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")

    # Get features
    features = get_feature_columns(df, include_market_signals=True)
    print(f"Using {len(features)} features")

    # Chronological split
    # Train on data before Oct 2025, validate on Oct 2025, test on Nov-Jan
    train_end = df[df['match_date'] < '2025-10-01'].index.max()
    val_end = df[df['match_date'] < '2025-11-01'].index.max()

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"\nTrain: {len(train_df)} samples (up to {train_df['match_date'].max()})")
    print(f"Val: {len(val_df)} samples ({val_df['match_date'].min()} to {val_df['match_date'].max()})")
    print(f"Test: {len(test_df)} samples ({test_df['match_date'].min()} to {test_df['match_date'].max()})")

    # Train model
    print("\nTraining model...")
    model = train_model(train_df, val_df, features, verbose=False)
    print("Model trained.")

    # Train calibrators
    print("Training calibrators...")
    calibrators = train_calibrators(model, val_df, features)
    print("Calibrators trained.")

    # Evaluate
    print("\nEvaluating on test set...")
    eval_results = evaluate_model(model, calibrators, test_df, features)
    print(f"Raw accuracy: {eval_results['raw_accuracy']:.1%}")
    print(f"Calibrated accuracy: {eval_results['calibrated_accuracy']:.1%}")

    # Feature importance
    print("\nTop 15 Most Important Features:")
    importance = model.get_feature_importance()
    feat_imp = sorted(zip(features, importance), key=lambda x: -x[1])
    for feat, imp in feat_imp[:15]:
        print(f"  {feat}: {imp:.1f}")

    # Backtest with different parameters
    print("\n" + "=" * 60)
    print("Backtesting Strategies")
    print("=" * 60)

    # Strategy 1: Conservative
    print("\n--- Strategy 1: Conservative (edge≥8%, cal≥40%) ---")
    bets1 = backtest_strategy(model, calibrators, test_df, features,
                              min_edge=0.08, min_cal_prob=0.40, sharp_signal_weight=0.5)
    if len(bets1) > 0:
        results1 = analyze_backtest(bets1)
        print_results(results1)

    # Strategy 2: Moderate
    print("\n--- Strategy 2: Moderate (edge≥5%, cal≥35%) ---")
    bets2 = backtest_strategy(model, calibrators, test_df, features,
                              min_edge=0.05, min_cal_prob=0.35, sharp_signal_weight=0.5)
    if len(bets2) > 0:
        results2 = analyze_backtest(bets2)
        print_results(results2)

    # Strategy 3: Aggressive
    print("\n--- Strategy 3: Aggressive (edge≥3%, cal≥30%) ---")
    bets3 = backtest_strategy(model, calibrators, test_df, features,
                              min_edge=0.03, min_cal_prob=0.30, sharp_signal_weight=0.5)
    if len(bets3) > 0:
        results3 = analyze_backtest(bets3)
        print_results(results3)

    # Save model and calibrators
    print("\nSaving model...")
    joblib.dump({
        'model': model,
        'calibrators': calibrators,
        'features': features,
    }, 'models/integrated_model_v1.joblib')
    print("Saved to: models/integrated_model_v1.joblib")

    return model, calibrators, features


if __name__ == '__main__':
    main()
