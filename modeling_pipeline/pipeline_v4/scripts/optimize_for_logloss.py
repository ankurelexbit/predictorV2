"""
Find Optimal Class Weights for Minimum Log Loss.

Comprehensive grid search focused purely on minimizing log loss.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
from itertools import product

# ML libraries
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Features to remove
CONSTANT_FEATURES = [
    'home_big_chances_per_match_5', 'away_big_chances_per_match_5',
    'home_xg_trend_10', 'away_xg_trend_10',
    'home_xg_vs_top_half', 'away_xg_vs_top_half',
    'home_xga_vs_bottom_half', 'away_xga_vs_bottom_half',
    'home_lineup_avg_rating_5', 'away_lineup_avg_rating_5',
    'home_top_3_players_rating', 'away_top_3_players_rating',
    'home_key_players_available', 'away_key_players_available',
    'home_players_in_form', 'away_players_in_form',
    'home_players_unavailable', 'away_players_unavailable',
    'home_days_since_last_match', 'away_days_since_last_match',
    'rest_advantage', 'is_derby_match', 'home_xga_trend_10', 'away_xga_trend_10'
]

HIGH_MISSING_FEATURES = [
    'away_interceptions_per_90', 'away_defensive_actions_per_90',
    'home_defensive_actions_per_90', 'home_interceptions_per_90',
    'derived_xgd_matchup'
]

REDUNDANT_FEATURES = [
    'home_elo_vs_league_avg', 'away_elo_vs_league_avg',
    'elo_diff_with_home_advantage',
    'home_ppda_5', 'away_ppda_5',
    'home_xg_from_corners_5', 'away_xg_from_corners_5',
    'home_big_chance_conversion_5', 'away_big_chance_conversion_5',
    'home_inside_box_xg_ratio', 'away_inside_box_xg_ratio',
]

BAD_FEATURES = list(set(CONSTANT_FEATURES + HIGH_MISSING_FEATURES + REDUNDANT_FEATURES))

METADATA_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id',
    'league_id', 'match_date', 'home_score', 'away_score', 'result', 'target'
]


def get_sample_weights(y, class_weights):
    """Get sample weights based on class weights."""
    return np.array([class_weights[label] for label in y])


def load_and_prepare_data(data_path):
    """Load and prepare data."""
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Map target
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Remove duplicates
    df = df.drop_duplicates(subset=['fixture_id'], keep='first')

    # Sort by date
    df = df.sort_values('match_date').reset_index(drop=True)

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in METADATA_COLS]
    features_to_remove = [f for f in BAD_FEATURES if f in feature_cols]
    feature_cols = [f for f in feature_cols if f not in features_to_remove]

    # Extract features and target
    X = df[feature_cols].copy()
    y = df['target'].values

    # Handle missing values
    X = X.fillna(X.median())

    # Chronological split (70/15/15)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y[val_end:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Features: {len(feature_cols)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model, X, y):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Ensure y_pred is numpy array of integers
    y_pred = np.array(y_pred).astype(int).ravel()

    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_pred_proba)

    # Per-class accuracy
    cm = confusion_matrix(y, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Prediction distribution
    pred_dist = np.bincount(y_pred, minlength=3) / len(y_pred)

    return {
        'accuracy': acc,
        'log_loss': logloss,
        'away_accuracy': per_class_acc[0],
        'draw_accuracy': per_class_acc[1],
        'home_accuracy': per_class_acc[2],
        'away_pred_pct': pred_dist[0],
        'draw_pred_pct': pred_dist[1],
        'home_pred_pct': pred_dist[2]
    }


def test_weight_combination(X_train, y_train, X_val, y_val, X_test, y_test,
                           away_weight, draw_weight, home_weight, model_type='catboost'):
    """Test a specific weight combination."""
    class_weights = {0: away_weight, 1: draw_weight, 2: home_weight}

    if model_type == 'xgboost':
        train_weights = get_sample_weights(y_train, class_weights)
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train, sample_weight=train_weights, verbose=False)

    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)

    elif model_type == 'catboost':
        model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function='MultiClass',
            class_weights=list(class_weights.values()),
            random_seed=42,
            verbose=False,
            thread_count=-1
        )
        model.fit(X_train, y_train)

    # Evaluate
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    return val_metrics, test_metrics, model


def main():
    logger.info("=" * 80)
    logger.info("OPTIMIZING CLASS WEIGHTS FOR MINIMUM LOG LOSS")
    logger.info("=" * 80)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data('data/training_data.csv')

    # Define comprehensive grid search
    # Focus on values around 1.0 since that gave best log loss
    away_weights = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    draw_weights = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    home_weights = [0.9, 0.95, 1.0, 1.05, 1.1]

    logger.info(f"\nGrid search configuration:")
    logger.info(f"  Away weights: {away_weights}")
    logger.info(f"  Draw weights: {draw_weights}")
    logger.info(f"  Home weights: {home_weights}")
    logger.info(f"  Total combinations: {len(away_weights) * len(draw_weights) * len(home_weights)}")

    results = []
    total_combinations = len(away_weights) * len(draw_weights) * len(home_weights)

    logger.info(f"\nTesting {total_combinations} combinations...")
    logger.info("Model: CatBoost (baseline)\n")

    counter = 0
    best_val_logloss = float('inf')
    best_config = None

    for away_w in away_weights:
        for draw_w in draw_weights:
            for home_w in home_weights:
                counter += 1

                # Progress update every 50 combinations
                if counter % 50 == 0 or counter == 1:
                    logger.info(f"[{counter}/{total_combinations}] Testing A={away_w}, D={draw_w}, H={home_w}")

                val_metrics, test_metrics, model = test_weight_combination(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    away_w, draw_w, home_w, model_type='catboost'
                )

                # Track best validation log loss
                if val_metrics['log_loss'] < best_val_logloss:
                    best_val_logloss = val_metrics['log_loss']
                    best_config = (away_w, draw_w, home_w)
                    logger.info(f"  ✨ NEW BEST! Val LogLoss: {val_metrics['log_loss']:.4f}, "
                              f"Test LogLoss: {test_metrics['log_loss']:.4f}")

                results.append({
                    'away_weight': away_w,
                    'draw_weight': draw_w,
                    'home_weight': home_w,
                    'val_log_loss': val_metrics['log_loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_draw_accuracy': val_metrics['draw_accuracy'],
                    'test_log_loss': test_metrics['log_loss'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_away_accuracy': test_metrics['away_accuracy'],
                    'test_draw_accuracy': test_metrics['draw_accuracy'],
                    'test_home_accuracy': test_metrics['home_accuracy'],
                    'test_away_pred_pct': test_metrics['away_pred_pct'],
                    'test_draw_pred_pct': test_metrics['draw_pred_pct'],
                    'test_home_pred_pct': test_metrics['home_pred_pct'],
                })

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Sort by validation log loss (best metric for generalization)
    df_results = df_results.sort_values('val_log_loss')

    # Save results
    Path('results').mkdir(exist_ok=True)
    df_results.to_csv('results/logloss_optimization_full.csv', index=False)

    # Print summary
    logger.info("\n" + "=" * 160)
    logger.info("OPTIMIZATION RESULTS (sorted by validation log loss)")
    logger.info("=" * 160)
    logger.info(f"{'Rank':<6} {'Away':<6} {'Draw':<6} {'Home':<6} {'Val LogLoss':<13} {'Test LogLoss':<13} "
               f"{'Test Acc':<10} {'Draw Acc':<10} {'Draw Pred%':<12}")
    logger.info("-" * 160)

    for i, (idx, row) in enumerate(df_results.head(20).iterrows(), 1):
        logger.info(f"{i:<6} {row['away_weight']:<6.2f} {row['draw_weight']:<6.2f} {row['home_weight']:<6.2f} "
                   f"{row['val_log_loss']:<13.6f} {row['test_log_loss']:<13.6f} "
                   f"{row['test_accuracy']:<10.4f} {row['test_draw_accuracy']:<10.4f} "
                   f"{row['test_draw_pred_pct']:<12.4f}")

    logger.info("=" * 160)

    # Best model
    best = df_results.iloc[0]
    logger.info(f"\n🏆 BEST CONFIGURATION (minimum validation log loss):")
    logger.info(f"   Weights: Away={best['away_weight']:.2f}, Draw={best['draw_weight']:.2f}, Home={best['home_weight']:.2f}")
    logger.info(f"   Validation Log Loss: {best['val_log_loss']:.6f}")
    logger.info(f"   Test Log Loss: {best['test_log_loss']:.6f}")
    logger.info(f"   Test Accuracy: {best['test_accuracy']:.4f}")
    logger.info(f"   Test Draw Accuracy: {best['test_draw_accuracy']:.4f}")
    logger.info(f"   Test Prediction Distribution: Away={best['test_away_pred_pct']:.2%}, "
               f"Draw={best['test_draw_pred_pct']:.2%}, Home={best['test_home_pred_pct']:.2%}")

    # Compare to no weights baseline
    baseline = df_results[(df_results['away_weight'] == 1.0) &
                         (df_results['draw_weight'] == 1.0) &
                         (df_results['home_weight'] == 1.0)].iloc[0]

    logger.info(f"\n📊 COMPARISON TO NO WEIGHTS (1.0/1.0/1.0):")
    logger.info(f"   Baseline Val Log Loss: {baseline['val_log_loss']:.6f}")
    logger.info(f"   Baseline Test Log Loss: {baseline['test_log_loss']:.6f}")
    logger.info(f"   Improvement: {baseline['val_log_loss'] - best['val_log_loss']:.6f} "
               f"({'better' if best['val_log_loss'] < baseline['val_log_loss'] else 'worse'})")

    # Best for test log loss (may differ from validation)
    best_test = df_results.sort_values('test_log_loss').iloc[0]
    if best_test['away_weight'] != best['away_weight'] or \
       best_test['draw_weight'] != best['draw_weight'] or \
       best_test['home_weight'] != best['home_weight']:
        logger.info(f"\n⚠️  BEST FOR TEST LOG LOSS (may overfit):")
        logger.info(f"   Weights: Away={best_test['away_weight']:.2f}, Draw={best_test['draw_weight']:.2f}, Home={best_test['home_weight']:.2f}")
        logger.info(f"   Test Log Loss: {best_test['test_log_loss']:.6f}")

    logger.info(f"\n💾 Full results ({len(df_results)} configurations) saved to: results/logloss_optimization_full.csv")
    logger.info("\n✅ OPTIMIZATION COMPLETE!")


if __name__ == '__main__':
    main()
