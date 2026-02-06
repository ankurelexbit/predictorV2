#!/usr/bin/env python3
"""
Train Bet Selector Model
=========================

Loads raw market features CSV (from extract_market_features.py),
joins with training data to get model probabilities, then trains
a Logistic Regression bet selector with walk-forward backtest.

Usage:
    # Backtest only
    python3 scripts/train_bet_selector.py --dry-run

    # Train and save model
    python3 scripts/train_bet_selector.py

    # Custom settings
    python3 scripts/train_bet_selector.py --folds 10 --min-confidence 0.55
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market.bet_selector import BetSelector
from src.market.market_feature_extractor import MarketFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_MARKET_FEATURES = Path(__file__).parent.parent / 'data' / 'market_features_raw.csv'
TRAINING_DATA = Path(__file__).parent.parent / 'data' / 'training_data.csv'
MODEL_OUTPUT = Path(__file__).parent.parent / 'models' / 'production' / 'bet_selector.joblib'


def generate_cv_probabilities(train: pd.DataFrame, feature_cols: list, n_splits: int = 5) -> pd.DataFrame:
    """Generate cross-validated out-of-sample probabilities.

    Each fixture's probability is predicted by a model that was NOT trained on it.
    This prevents data leakage from overfit in-sample predictions.
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from config.production_config import TRAINING_CONFIG

    logger.info(f"Generating {n_splits}-fold CV out-of-sample probabilities...")

    train = train.sort_values('match_date').reset_index(drop=True)
    X = train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = train['result']

    all_probs = np.zeros((len(train), 3))
    fold_size = len(train) // n_splits

    for fold_idx in range(n_splits):
        test_start = fold_idx * fold_size
        test_end = len(train) if fold_idx == n_splits - 1 else (fold_idx + 1) * fold_size

        # Train on everything outside this fold (chronological, so use all data before test)
        # But to be fair, train only on data BEFORE the test fold
        train_end = test_start
        if train_end < 100:
            # Not enough training data for first fold, skip
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]

        # Train CatBoost
        cat_params = TRAINING_CONFIG.get('catboost', {}).copy()
        cat_params['verbose'] = 0
        cat_params.pop('early_stopping_rounds', None)
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_train, y_train)

        # Train LightGBM
        lgb_params = TRAINING_CONFIG.get('lightgbm', {}).copy()
        lgb_params['verbose'] = -1
        lgb = LGBMClassifier(**lgb_params)
        lgb.fit(X_train, y_train)

        # Ensemble predict
        probs_cat = cat.predict_proba(X_test)
        probs_lgb = lgb.predict_proba(X_test)
        probs = (probs_cat + probs_lgb) / 2

        all_probs[test_start:test_end] = probs
        logger.info(f"  Fold {fold_idx+1}/{n_splits}: trained on {train_end} rows, predicted {test_end-test_start} rows")

    train['pred_away_prob'] = all_probs[:, 0]
    train['pred_draw_prob'] = all_probs[:, 1]
    train['pred_home_prob'] = all_probs[:, 2]

    # Remove rows with no predictions (first fold if too small)
    train = train[train['pred_home_prob'] > 0].copy()
    return train


def build_dataset(market_path: str, training_path: str) -> pd.DataFrame:
    """Join raw market features with CV out-of-sample model probabilities."""
    logger.info("Loading raw market features...")
    mkt = pd.read_csv(market_path)
    logger.info(f"  Market features: {len(mkt)} rows")

    logger.info("Loading training data...")
    train = pd.read_csv(training_path)
    logger.info(f"  Training data: {len(train)} rows")

    # Load model for feature_cols only
    from config.production_config import get_latest_model_path
    model_data = joblib.load(get_latest_model_path())
    feature_cols = model_data['feature_cols']

    # Generate CV out-of-sample probabilities (prevents data leakage)
    train = generate_cv_probabilities(train, feature_cols)

    # Keep only what we need for the join
    prob_df = train[['fixture_id', 'pred_home_prob', 'pred_draw_prob', 'pred_away_prob']].copy()
    prob_df['fixture_id'] = prob_df['fixture_id'].astype(int)
    mkt['fixture_id'] = mkt['fixture_id'].astype(int)

    # Join
    df = mkt.merge(prob_df, on='fixture_id', how='inner')
    logger.info(f"  Joined: {len(df)} rows (market features matched with model probs)")

    if len(df) == 0:
        logger.error("No matches found. Check fixture_id alignment.")
        sys.exit(1)

    # Build full features using the extractor
    extractor = MarketFeatureExtractor()
    feature_names = extractor.get_feature_names()

    # Compute derived features that need model probs
    df['pred_max_prob'] = df[['pred_home_prob', 'pred_draw_prob', 'pred_away_prob']].max(axis=1)

    # Predicted outcome encoded
    def encode_outcome(row):
        probs = {'home': row['pred_home_prob'], 'draw': row['pred_draw_prob'], 'away': row['pred_away_prob']}
        pred = max(probs, key=probs.get)
        return {'home': 0, 'draw': 1, 'away': 2}[pred]

    df['pred_outcome_encoded'] = df.apply(encode_outcome, axis=1)

    # Model-vs-market edge features
    for outcome in ['home', 'draw', 'away']:
        market_prob_col = f'{outcome}_market_prob'
        model_prob_col = f'pred_{outcome}_prob'
        if market_prob_col in df.columns:
            df[f'{outcome}_edge'] = df[model_prob_col] - df[market_prob_col]
        else:
            df[f'{outcome}_edge'] = 0

    # Edge on predicted outcome
    def pred_edge(row):
        probs = {'home': row['pred_home_prob'], 'draw': row['pred_draw_prob'], 'away': row['pred_away_prob']}
        pred = max(probs, key=probs.get)
        return row.get(f'{pred}_edge', 0)

    df['pred_outcome_edge'] = df.apply(pred_edge, axis=1)

    # Add best odds for predicted outcome (for PnL calculation)
    def pred_odds(row):
        probs = {'home': row['pred_home_prob'], 'draw': row['pred_draw_prob'], 'away': row['pred_away_prob']}
        pred = max(probs, key=probs.get)
        return row.get(f'{pred}_best_odds', 0)

    df['best_home_odds'] = df['home_best_odds']
    df['best_draw_odds'] = df['draw_best_odds']
    df['best_away_odds'] = df['away_best_odds']

    df = df.sort_values('match_date').reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description='Train bet selector model')
    parser.add_argument('--market-features', default=str(RAW_MARKET_FEATURES))
    parser.add_argument('--training-data', default=str(TRAINING_DATA))
    parser.add_argument('--output', default=str(MODEL_OUTPUT))
    parser.add_argument('--folds', type=int, default=7)
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--dry-run', action='store_true', help='Backtest only')
    parser.add_argument('--strategy', default=None,
                        help='Strategy to save (e.g. gbm_0.55, hybrid_gbm_0.55). Auto-selects best if not set.')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BET SELECTOR TRAINING")
    logger.info("=" * 60)

    # Build dataset
    df = build_dataset(args.market_features, args.training_data)
    logger.info(f"\nDataset: {len(df)} fixtures")
    logger.info(f"Date range: {df['match_date'].iloc[0]} to {df['match_date'].iloc[-1]}")
    logger.info(f"Results: {df['actual_result'].value_counts().to_dict()}")

    # Verify features exist
    feature_names = MarketFeatureExtractor.get_feature_names()
    present = [f for f in feature_names if f in df.columns]
    missing = [f for f in feature_names if f not in df.columns]
    logger.info(f"Features: {len(present)} present, {len(missing)} missing")
    if missing:
        logger.warning(f"Missing features (will be filled with 0): {missing}")

    # Run walk-forward backtest
    logger.info(f"\nRunning {args.folds}-fold walk-forward backtest (LR + GBM + hybrid)...")
    results = BetSelector.backtest(df, n_folds=args.folds)

    # Print per-fold comparison for key strategies
    fold_details = results['fold_details']
    key_strats = [s for s in fold_details.keys() if s in fold_details]

    logger.info(f"\n{'='*140}")
    logger.info("PER-FOLD BREAKDOWN")
    logger.info(f"{'='*140}")
    header = f"{'Fold':<6} {'Period':<22}"
    for s in key_strats:
        header += f" | {s:>16}"
    logger.info(header)
    logger.info("-" * 140)

    first_key = key_strats[0]
    n_folds = len(fold_details.get(first_key, []))
    for i in range(n_folds):
        period = ""
        line_parts = []
        for s in key_strats:
            folds = fold_details[s]
            if i < len(folds):
                fd = folds[i]
                if not period:
                    period = f"{fd['test_start']} to {fd['test_end']}"
                line_parts.append(f"{fd['bets']:>4}b {fd['roi']:>+5.1f}%")
            else:
                line_parts.append(f"{'N/A':>12}")
        fold_idx = fold_details[first_key][i]['fold']
        line = f"{fold_idx:<6} {period:<22}"
        for part in line_parts:
            line += f" | {part:>16}"
        logger.info(line)
    logger.info("-" * 140)

    # Print all strategies sorted by ROI
    logger.info(f"\n{'='*70}")
    logger.info("ALL STRATEGIES COMPARISON (sorted by ROI)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Strategy':<25} {'Bets':>6} {'WR%':>7} {'Profit':>9} {'ROI%':>7}")
    logger.info("-" * 70)

    strats = results['strategies']
    sorted_strats = sorted(strats.items(), key=lambda x: x[1]['roi'], reverse=True)
    for name, s in sorted_strats:
        if s['bets'] == 0:
            continue
        marker = " <-- BEST" if name == sorted_strats[0][0] else ""
        marker = " <-- baseline" if name == 'threshold_H60' else marker
        logger.info(
            f"{name:<25} {s['bets']:>6} {s['wr']:>6.1f}% ${s['profit']:>7.2f} {s['roi']:>6.1f}%{marker}"
        )

    # Top features
    if results['top_features']:
        logger.info(f"\nTop 10 LR features by coefficient magnitude:")
        for name, coef in results['top_features']:
            logger.info(f"  {name:<30} {coef:>+.4f}")

    # Find best strategy (prefer strategies with >= 200 bets for reliability)
    reliable_strats = [(n, s) for n, s in sorted_strats if s['bets'] >= 200]
    if reliable_strats:
        best_name, best_stats = reliable_strats[0]
    else:
        best_name, best_stats = sorted_strats[0]

    thr_stats = strats.get('threshold_H60', {})
    logger.info(f"\nBest reliable strategy (>=200 bets): {best_name} ({best_stats['roi']:.1f}% ROI, {best_stats['bets']} bets)")
    if thr_stats:
        logger.info(f"vs threshold H60 baseline: {thr_stats['roi']:.1f}% ROI, {thr_stats['bets']} bets")

    # Save model
    if not args.dry_run:
        # Override with --strategy if provided
        save_name = args.strategy or best_name
        if save_name not in strats:
            logger.error(f"Strategy '{save_name}' not found. Available: {list(strats.keys())}")
            sys.exit(1)
        save_stats = strats[save_name]
        logger.info(f"\nSaving strategy: {save_name} ({save_stats['roi']:.1f}% ROI, {save_stats['bets']} bets)")

        use_gbm = 'gbm' in save_name
        model_type = 'gbm' if use_gbm else 'lr'
        best_conf = args.min_confidence
        import re as re_mod
        conf_match = re_mod.search(r'(\d+\.\d+)', save_name)
        if conf_match:
            best_conf = float(conf_match.group(1))

        logger.info(f"\nTraining final {model_type.upper()} model (conf={best_conf:.2f}) on all {len(df)} samples...")

        def get_pred_outcome(row):
            probs = {'home': row['pred_home_prob'], 'draw': row['pred_draw_prob'], 'away': row['pred_away_prob']}
            return max(probs, key=probs.get)

        df['pred_outcome'] = df.apply(get_pred_outcome, axis=1)
        df['pred_outcome_code'] = df['pred_outcome'].map({'home': 'H', 'draw': 'D', 'away': 'A'})
        df['bet_won'] = (df['pred_outcome_code'] == df['actual_result']).astype(int)

        selector = BetSelector(min_confidence=best_conf, model_type=model_type)
        X = df[feature_names].fillna(0)
        y = df['bet_won'].values
        odds = df.apply(lambda r: r[f"best_{r['pred_outcome']}_odds"], axis=1).values
        selector.train(X, y, odds=odds)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        selector.save(args.output)
        logger.info(f"Saved to {args.output}")
    else:
        logger.info("\n[DRY RUN] Model not saved.")


if __name__ == '__main__':
    main()
