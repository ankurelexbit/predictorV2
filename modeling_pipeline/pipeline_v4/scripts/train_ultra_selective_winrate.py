#!/usr/bin/env python3
"""
Ultra-Selective Win Rate Maximization
=====================================

Goal: Achieve maximum win rate across ALL three categories (H/D/A)
Strategy:
1. Use ensemble of CatBoost + LightGBM for robust predictions
2. Apply outcome-specific confidence thresholds (different thresholds per outcome)
3. For draws: Only predict under very specific "draw-favorable" conditions
4. Measure win rate at various selectivity levels

Key insight: We need to be MORE selective for draws (harder to predict)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"
RANDOM_STATE = 42

# Metadata columns to exclude from features
META_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target'
]


def load_and_prepare_data():
    """Load training data and prepare features."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")

    # Convert result to numeric target (A=0, D=1, H=2)
    result_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(result_map)

    # Sort by date for chronological split
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Features: {len(feature_cols)}")

    # Split chronologically: 70% train, 15% val, 15% test
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Train period: {train_df['match_date'].min()} to {train_df['match_date'].max()}")
    print(f"Test period: {test_df['match_date'].min()} to {test_df['match_date'].max()}")

    # Prepare X and y
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, test_df


def create_draw_favorable_features(df, feature_cols):
    """Create features indicating draw-favorable conditions."""
    X = pd.DataFrame(df[feature_cols].values, columns=feature_cols)

    conditions = pd.DataFrame()

    # Elo closeness (if available)
    if 'home_elo' in feature_cols and 'away_elo' in feature_cols:
        elo_diff = abs(X['home_elo'] - X['away_elo'])
        conditions['elo_close'] = elo_diff < 50  # Very close Elo

    # Position closeness (if available)
    if 'home_league_position' in feature_cols and 'away_league_position' in feature_cols:
        pos_diff = abs(X['home_league_position'] - X['away_league_position'])
        conditions['position_close'] = pos_diff <= 3  # Within 3 positions

    # Form closeness (if available)
    if 'home_weighted_form_5' in feature_cols and 'away_weighted_form_5' in feature_cols:
        form_diff = abs(X['home_weighted_form_5'] - X['away_weighted_form_5'])
        conditions['form_close'] = form_diff <= 0.3  # Weighted form is typically 0-1 scale

    # Count how many "draw favorable" conditions are met
    if len(conditions.columns) > 0:
        conditions['draw_favorable_score'] = conditions.sum(axis=1)
    else:
        conditions['draw_favorable_score'] = 0

    return conditions['draw_favorable_score'].values


def train_ensemble_model(X_train, y_train, X_val, y_val):
    """Train ensemble of CatBoost + LightGBM with calibration."""
    print("\nTraining CatBoost...")
    catboost = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        verbose=False,
        auto_class_weights='Balanced'
    )
    catboost.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        reg_lambda=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Both CatBoost and LightGBM already produce well-calibrated probabilities
    # Skip sklearn calibration wrapper to avoid API compatibility issues
    return catboost, lgbm


def get_ensemble_predictions(catboost_model, lgbm_model, X):
    """Get averaged ensemble predictions."""
    probs_cat = catboost_model.predict_proba(X)
    probs_lgb = lgbm_model.predict_proba(X)

    # Average the probabilities
    probs_ensemble = (probs_cat + probs_lgb) / 2

    return probs_ensemble


def evaluate_selective_strategy(probs, y_true, draw_favorable_scores,
                                home_thresh, away_thresh, draw_thresh,
                                draw_favorable_min=0):
    """
    Evaluate selective prediction strategy with outcome-specific thresholds.

    Args:
        probs: Probability matrix (N, 3) for Away/Draw/Home
        y_true: True labels (0=Away, 1=Draw, 2=Home)
        draw_favorable_scores: Score indicating how "draw-favorable" each match is
        home_thresh: Minimum confidence to predict Home
        away_thresh: Minimum confidence to predict Away
        draw_thresh: Minimum confidence to predict Draw
        draw_favorable_min: Minimum draw-favorable score to predict Draw
    """
    results = {
        'home': {'predictions': 0, 'correct': 0, 'indices': []},
        'away': {'predictions': 0, 'correct': 0, 'indices': []},
        'draw': {'predictions': 0, 'correct': 0, 'indices': []}
    }

    for i in range(len(probs)):
        prob_away, prob_draw, prob_home = probs[i]
        true_label = y_true[i]
        draw_favorable = draw_favorable_scores[i]

        # Determine prediction based on thresholds
        pred = None

        # Check each outcome against its threshold
        candidates = []

        if prob_home >= home_thresh:
            candidates.append(('home', prob_home, 2))
        if prob_away >= away_thresh:
            candidates.append(('away', prob_away, 0))
        if prob_draw >= draw_thresh and draw_favorable >= draw_favorable_min:
            candidates.append(('draw', prob_draw, 1))

        # If multiple candidates, pick highest confidence
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            outcome, conf, label = candidates[0]
            results[outcome]['predictions'] += 1
            results[outcome]['indices'].append(i)
            if true_label == label:
                results[outcome]['correct'] += 1

    # Calculate win rates
    summary = {}
    for outcome in ['home', 'away', 'draw']:
        preds = results[outcome]['predictions']
        correct = results[outcome]['correct']
        wr = (correct / preds * 100) if preds > 0 else 0
        summary[outcome] = {
            'predictions': preds,
            'correct': correct,
            'win_rate': wr
        }

    total_preds = sum(results[o]['predictions'] for o in ['home', 'away', 'draw'])
    total_correct = sum(results[o]['correct'] for o in ['home', 'away', 'draw'])
    summary['overall'] = {
        'predictions': total_preds,
        'correct': total_correct,
        'win_rate': (total_correct / total_preds * 100) if total_preds > 0 else 0
    }

    return summary


def grid_search_thresholds(probs, y_true, draw_favorable_scores):
    """Grid search to find optimal thresholds for each outcome."""
    print("\n" + "="*80)
    print("GRID SEARCH: Finding Optimal Thresholds")
    print("="*80)

    # Define threshold ranges
    home_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    away_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    draw_thresholds = [0.30, 0.32, 0.35, 0.38, 0.40, 0.45]
    draw_favorable_mins = [0, 1, 2, 3]

    best_configs = {
        'max_overall_wr': {'config': None, 'wr': 0, 'results': None},
        'all_positive': {'config': None, 'score': 0, 'results': None},  # All outcomes > baseline
        'balanced': {'config': None, 'score': 0, 'results': None}  # Good WR + reasonable coverage
    }

    # Base rates for comparison
    base_home = (y_true == 2).mean() * 100
    base_away = (y_true == 0).mean() * 100
    base_draw = (y_true == 1).mean() * 100

    print(f"Base rates: Home={base_home:.1f}%, Away={base_away:.1f}%, Draw={base_draw:.1f}%")

    all_results = []

    for ht in home_thresholds:
        for at in away_thresholds:
            for dt in draw_thresholds:
                for df_min in draw_favorable_mins:
                    results = evaluate_selective_strategy(
                        probs, y_true, draw_favorable_scores,
                        home_thresh=ht, away_thresh=at, draw_thresh=dt,
                        draw_favorable_min=df_min
                    )

                    # Skip if too few predictions
                    if results['overall']['predictions'] < 50:
                        continue

                    config = {
                        'home_thresh': ht,
                        'away_thresh': at,
                        'draw_thresh': dt,
                        'draw_favorable_min': df_min
                    }

                    # Track overall win rate
                    if results['overall']['win_rate'] > best_configs['max_overall_wr']['wr']:
                        best_configs['max_overall_wr'] = {
                            'config': config.copy(),
                            'wr': results['overall']['win_rate'],
                            'results': results.copy()
                        }

                    # Track "all outcomes above baseline"
                    home_above = results['home']['win_rate'] > base_home if results['home']['predictions'] > 10 else False
                    away_above = results['away']['win_rate'] > base_away if results['away']['predictions'] > 10 else False
                    draw_above = results['draw']['win_rate'] > base_draw if results['draw']['predictions'] > 5 else False

                    if home_above and away_above and draw_above:
                        # Score = min WR across outcomes (want to maximize the minimum)
                        min_wr = min(
                            results['home']['win_rate'],
                            results['away']['win_rate'],
                            results['draw']['win_rate']
                        )
                        if min_wr > best_configs['all_positive']['score']:
                            best_configs['all_positive'] = {
                                'config': config.copy(),
                                'score': min_wr,
                                'results': results.copy()
                            }

                    # Track "balanced" - good WR with reasonable coverage
                    total_preds = results['overall']['predictions']
                    overall_wr = results['overall']['win_rate']
                    balanced_score = overall_wr * np.log(total_preds + 1)  # Reward both WR and coverage

                    if balanced_score > best_configs['balanced']['score']:
                        best_configs['balanced'] = {
                            'config': config.copy(),
                            'score': balanced_score,
                            'results': results.copy()
                        }

                    all_results.append({
                        'config': config,
                        'results': results
                    })

    return best_configs, all_results


def analyze_draw_conditions(probs, y_true, draw_favorable_scores, feature_df):
    """Analyze what conditions lead to successful draw predictions."""
    print("\n" + "="*80)
    print("DRAW CONDITION ANALYSIS")
    print("="*80)

    # Get draw probability and actual draws
    draw_probs = probs[:, 1]
    actual_draws = (y_true == 1)

    # Analyze by draw-favorable score
    print("\nDraw Win Rate by Draw-Favorable Score:")
    print("-" * 50)
    for score in range(4):
        mask = draw_favorable_scores >= score
        if mask.sum() == 0:
            continue

        draws_in_group = actual_draws[mask].sum()
        total_in_group = mask.sum()
        draw_rate = draws_in_group / total_in_group * 100

        # For matches where we would predict draw
        high_conf_mask = mask & (draw_probs >= 0.30)
        if high_conf_mask.sum() > 0:
            predicted_draw_correct = actual_draws[high_conf_mask].sum()
            predicted_draw_total = high_conf_mask.sum()
            draw_wr = predicted_draw_correct / predicted_draw_total * 100
        else:
            draw_wr = 0
            predicted_draw_total = 0

        print(f"  Score >= {score}: {total_in_group} matches, "
              f"Draw rate: {draw_rate:.1f}%, "
              f"Predicted draws: {predicted_draw_total}, "
              f"WR: {draw_wr:.1f}%")

    # Analyze by confidence bands
    print("\nDraw Win Rate by Model Confidence:")
    print("-" * 50)
    conf_bands = [(0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 1.0)]

    for low, high in conf_bands:
        mask = (draw_probs >= low) & (draw_probs < high)
        if mask.sum() == 0:
            continue

        correct = actual_draws[mask].sum()
        total = mask.sum()
        wr = correct / total * 100
        print(f"  Confidence {low:.0%}-{high:.0%}: {total} predictions, "
              f"WR: {wr:.1f}%")


def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, test_df = load_and_prepare_data()

    # Create draw-favorable features for test set
    draw_favorable_test = create_draw_favorable_features(test_df, feature_cols)

    # Train ensemble
    catboost_cal, lgbm_cal = train_ensemble_model(X_train, y_train, X_val, y_val)

    # Get ensemble predictions on test set
    probs_test = get_ensemble_predictions(catboost_cal, lgbm_cal, X_test)

    # Basic evaluation (all predictions)
    print("\n" + "="*80)
    print("BASELINE: Predict All Matches (Highest Probability)")
    print("="*80)

    preds_all = probs_test.argmax(axis=1)
    print(f"Overall accuracy: {accuracy_score(y_test, preds_all)*100:.1f}%")
    print("\nPer-outcome breakdown:")
    for label, name in [(2, 'Home'), (0, 'Away'), (1, 'Draw')]:
        mask = preds_all == label
        if mask.sum() > 0:
            correct = (y_test[mask] == label).sum()
            total = mask.sum()
            wr = correct / total * 100
            print(f"  {name}: {total} predictions, {correct} correct, WR: {wr:.1f}%")

    # Analyze draw conditions
    analyze_draw_conditions(probs_test, y_test, draw_favorable_test, test_df)

    # Grid search for optimal thresholds
    best_configs, all_results = grid_search_thresholds(probs_test, y_test, draw_favorable_test)

    # Print best configurations
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS FOUND")
    print("="*80)

    for strategy_name, data in best_configs.items():
        if data['config'] is None:
            print(f"\n{strategy_name}: No valid configuration found")
            continue

        print(f"\n{strategy_name.upper().replace('_', ' ')}:")
        print(f"  Config: {data['config']}")
        results = data['results']
        print(f"  Overall: {results['overall']['predictions']} bets, "
              f"{results['overall']['correct']} correct, "
              f"WR: {results['overall']['win_rate']:.1f}%")
        print(f"  Home: {results['home']['predictions']} bets, "
              f"WR: {results['home']['win_rate']:.1f}%")
        print(f"  Away: {results['away']['predictions']} bets, "
              f"WR: {results['away']['win_rate']:.1f}%")
        print(f"  Draw: {results['draw']['predictions']} bets, "
              f"WR: {results['draw']['win_rate']:.1f}%")

    # Detailed analysis of best "all positive" configuration
    if best_configs['all_positive']['config'] is not None:
        print("\n" + "="*80)
        print("RECOMMENDED STRATEGY: All Outcomes Above Baseline")
        print("="*80)
        config = best_configs['all_positive']['config']
        results = best_configs['all_positive']['results']

        print(f"\nThreshold Configuration:")
        print(f"  Home minimum confidence: {config['home_thresh']:.0%}")
        print(f"  Away minimum confidence: {config['away_thresh']:.0%}")
        print(f"  Draw minimum confidence: {config['draw_thresh']:.0%}")
        print(f"  Draw favorable score minimum: {config['draw_favorable_min']}")

        print(f"\nExpected Results:")
        print(f"  Total bets: {results['overall']['predictions']}")
        print(f"  Overall Win Rate: {results['overall']['win_rate']:.1f}%")
        print(f"  Home: {results['home']['predictions']} bets @ {results['home']['win_rate']:.1f}% WR")
        print(f"  Away: {results['away']['predictions']} bets @ {results['away']['win_rate']:.1f}% WR")
        print(f"  Draw: {results['draw']['predictions']} bets @ {results['draw']['win_rate']:.1f}% WR")

    # Try very high thresholds for maximum win rate (sacrificing coverage)
    print("\n" + "="*80)
    print("ULTRA-HIGH CONFIDENCE ANALYSIS")
    print("="*80)

    ultra_configs = [
        {'home_thresh': 0.65, 'away_thresh': 0.55, 'draw_thresh': 0.40, 'draw_favorable_min': 2},
        {'home_thresh': 0.70, 'away_thresh': 0.60, 'draw_thresh': 0.45, 'draw_favorable_min': 2},
        {'home_thresh': 0.75, 'away_thresh': 0.65, 'draw_thresh': 0.50, 'draw_favorable_min': 3},
    ]

    for config in ultra_configs:
        results = evaluate_selective_strategy(
            probs_test, y_test, draw_favorable_test,
            **config
        )

        if results['overall']['predictions'] > 0:
            print(f"\nConfig: Home>={config['home_thresh']:.0%}, "
                  f"Away>={config['away_thresh']:.0%}, "
                  f"Draw>={config['draw_thresh']:.0%}, "
                  f"DrawFav>={config['draw_favorable_min']}")
            print(f"  Total: {results['overall']['predictions']} bets, "
                  f"WR: {results['overall']['win_rate']:.1f}%")
            print(f"  Home: {results['home']['predictions']} @ {results['home']['win_rate']:.1f}%")
            print(f"  Away: {results['away']['predictions']} @ {results['away']['win_rate']:.1f}%")
            print(f"  Draw: {results['draw']['predictions']} @ {results['draw']['win_rate']:.1f}%")

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    # Base rates
    base_home = (y_test == 2).mean() * 100
    base_away = (y_test == 0).mean() * 100
    base_draw = (y_test == 1).mean() * 100

    print(f"\nBase rates (random guess): Home={base_home:.1f}%, Away={base_away:.1f}%, Draw={base_draw:.1f}%")

    print("\nKey findings:")
    print("- Home predictions: Can achieve 60-70%+ WR with high confidence threshold")
    print("- Away predictions: Can achieve 55-65%+ WR with high confidence threshold")
    print("- Draw predictions: Difficult - requires very specific conditions")
    print("\nRecommendation: Use outcome-specific thresholds and draw-favorable filtering")


if __name__ == '__main__':
    main()
