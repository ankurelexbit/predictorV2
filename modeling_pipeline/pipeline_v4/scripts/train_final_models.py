"""
Train Final Production Models with Calibration.

Trains two models:
1. No weights (1.0/1.0/1.0) - Best log loss
2. Conservative weights (1.2/1.5/1.0) - Best for betting

Both with:
- Full hyperparameter optimization (Optuna)
- Isotonic calibration
- Comprehensive evaluation
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import argparse
import json
import warnings
from datetime import datetime

# ML libraries
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    logger.info(f"Loading data from {data_path}...")
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

    logger.info(f"Loaded {len(df):,} samples")
    logger.info(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in METADATA_COLS]
    features_to_remove = [f for f in BAD_FEATURES if f in feature_cols]
    feature_cols = [f for f in feature_cols if f not in features_to_remove]

    logger.info(f"Features: {len(df.columns)} → {len(feature_cols)} (after cleanup)")

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

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(X_train):,} ({len(X_train)/n:.1%})")
    logger.info(f"  Val:   {len(X_val):,} ({len(X_val)/n:.1%})")
    logger.info(f"  Test:  {len(X_test):,} ({len(X_test)/n:.1%})")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def evaluate_model(model, X, y, name=""):
    """Evaluate model comprehensively."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Ensure y_pred is numpy array
    y_pred = np.array(y_pred).astype(int).ravel()

    # Basic metrics
    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_pred_proba)

    # Per-class metrics
    cm = confusion_matrix(y, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Prediction distribution
    pred_dist = np.bincount(y_pred, minlength=3) / len(y_pred)

    # Brier score (calibration metric)
    brier = np.mean([brier_score_loss(y == i, y_pred_proba[:, i]) for i in range(3)])

    metrics = {
        'accuracy': acc,
        'log_loss': logloss,
        'brier_score': brier,
        'away_accuracy': per_class_acc[0],
        'draw_accuracy': per_class_acc[1],
        'home_accuracy': per_class_acc[2],
        'away_pred_pct': pred_dist[0],
        'draw_pred_pct': pred_dist[1],
        'home_pred_pct': pred_dist[2],
        'confusion_matrix': cm
    }

    if name:
        logger.info(f"{name}:")
        logger.info(f"  Accuracy: {acc:.4f}, Log Loss: {logloss:.4f}, Brier: {brier:.4f}")
        logger.info(f"  Per-class Acc: Away={per_class_acc[0]:.3f}, Draw={per_class_acc[1]:.3f}, Home={per_class_acc[2]:.3f}")
        logger.info(f"  Predictions:   Away={pred_dist[0]:.1%}, Draw={pred_dist[1]:.1%}, Home={pred_dist[2]:.1%}")

    return metrics


def optimize_catboost(X_train, y_train, X_val, y_val, class_weights, n_trials=100):
    """Optimize CatBoost with Optuna."""
    logger.info(f"\nOptimizing CatBoost ({n_trials} trials)...")
    logger.info(f"Class weights: {class_weights}")

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'loss_function': 'MultiClass',
            'class_weights': list(class_weights.values()),
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }

        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)
        return log_loss(y_val, y_pred_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best log loss: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Train final model
    best_params = study.best_params.copy()
    best_params.update({
        'loss_function': 'MultiClass',
        'class_weights': list(class_weights.values()),
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1
    })

    model = cb.CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    return model, best_params


def calibrate_model(model, X_cal, y_cal, method='isotonic'):
    """Apply calibration to model."""
    logger.info(f"\nApplying {method} calibration...")

    # CalibratedClassifierCV expects a classifier with predict_proba
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv='prefit'  # Use pre-trained model
    )
    calibrated.fit(X_cal, y_cal)

    logger.info("Calibration complete")
    return calibrated


def main():
    parser = argparse.ArgumentParser(description='Train Final Production Models')
    parser.add_argument('--data', type=str, default='data/training_data.csv',
                       help='Path to training data')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--output-dir', type=str, default='models/final',
                       help='Output directory for models')

    args = parser.parse_args()

    logger.info("=" * 100)
    logger.info("TRAINING FINAL PRODUCTION MODELS")
    logger.info("=" * 100)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_and_prepare_data(args.data)

    # Store results
    results = {}

    # ==================================================================================
    # MODEL 1: NO WEIGHTS (1.0/1.0/1.0) - Best Log Loss
    # ==================================================================================
    logger.info("\n" + "=" * 100)
    logger.info("MODEL 1: NO WEIGHTS (1.0/1.0/1.0) - OPTIMIZED FOR LOG LOSS")
    logger.info("=" * 100)

    class_weights_1 = {0: 1.0, 1: 1.0, 2: 1.0}

    # Train with optimization
    model_1, params_1 = optimize_catboost(
        X_train, y_train, X_val, y_val,
        class_weights_1, n_trials=args.n_trials
    )

    # Evaluate uncalibrated
    logger.info("\n--- Uncalibrated Performance ---")
    train_metrics_1 = evaluate_model(model_1, X_train, y_train, "Train")
    val_metrics_1 = evaluate_model(model_1, X_val, y_val, "Val")
    test_metrics_1 = evaluate_model(model_1, X_test, y_test, "Test")

    # Calibrate
    model_1_cal = calibrate_model(model_1, X_val, y_val, method='isotonic')

    # Evaluate calibrated
    logger.info("\n--- Calibrated Performance ---")
    train_metrics_1_cal = evaluate_model(model_1_cal, X_train, y_train, "Train")
    val_metrics_1_cal = evaluate_model(model_1_cal, X_val, y_val, "Val")
    test_metrics_1_cal = evaluate_model(model_1_cal, X_test, y_test, "Test")

    # Save models
    joblib.dump(model_1, output_dir / 'model_no_weights_uncalibrated.joblib')
    joblib.dump(model_1_cal, output_dir / 'model_no_weights_calibrated.joblib')
    logger.info(f"\n✅ Saved to {output_dir / 'model_no_weights_*.joblib'}")

    results['no_weights'] = {
        'class_weights': class_weights_1,
        'params': params_1,
        'uncalibrated': {
            'train': train_metrics_1,
            'val': val_metrics_1,
            'test': test_metrics_1
        },
        'calibrated': {
            'train': train_metrics_1_cal,
            'val': val_metrics_1_cal,
            'test': test_metrics_1_cal
        }
    }

    # ==================================================================================
    # MODEL 2: CONSERVATIVE WEIGHTS (1.2/1.5/1.0) - Best for Betting
    # ==================================================================================
    logger.info("\n" + "=" * 100)
    logger.info("MODEL 2: CONSERVATIVE WEIGHTS (1.2/1.5/1.0) - OPTIMIZED FOR BETTING")
    logger.info("=" * 100)

    class_weights_2 = {0: 1.2, 1: 1.5, 2: 1.0}

    # Train with optimization
    model_2, params_2 = optimize_catboost(
        X_train, y_train, X_val, y_val,
        class_weights_2, n_trials=args.n_trials
    )

    # Evaluate uncalibrated
    logger.info("\n--- Uncalibrated Performance ---")
    train_metrics_2 = evaluate_model(model_2, X_train, y_train, "Train")
    val_metrics_2 = evaluate_model(model_2, X_val, y_val, "Val")
    test_metrics_2 = evaluate_model(model_2, X_test, y_test, "Test")

    # Calibrate
    model_2_cal = calibrate_model(model_2, X_val, y_val, method='isotonic')

    # Evaluate calibrated
    logger.info("\n--- Calibrated Performance ---")
    train_metrics_2_cal = evaluate_model(model_2_cal, X_train, y_train, "Train")
    val_metrics_2_cal = evaluate_model(model_2_cal, X_val, y_val, "Val")
    test_metrics_2_cal = evaluate_model(model_2_cal, X_test, y_test, "Test")

    # Save models
    joblib.dump(model_2, output_dir / 'model_conservative_uncalibrated.joblib')
    joblib.dump(model_2_cal, output_dir / 'model_conservative_calibrated.joblib')
    logger.info(f"\n✅ Saved to {output_dir / 'model_conservative_*.joblib'}")

    results['conservative'] = {
        'class_weights': class_weights_2,
        'params': params_2,
        'uncalibrated': {
            'train': train_metrics_2,
            'val': val_metrics_2,
            'test': test_metrics_2
        },
        'calibrated': {
            'train': train_metrics_2_cal,
            'val': val_metrics_2_cal,
            'test': test_metrics_2_cal
        }
    }

    # ==================================================================================
    # FINAL COMPARISON
    # ==================================================================================
    logger.info("\n" + "=" * 100)
    logger.info("FINAL COMPARISON - TEST SET RESULTS")
    logger.info("=" * 100)

    comparison_data = []
    for model_name in ['no_weights', 'conservative']:
        for cal_status in ['uncalibrated', 'calibrated']:
            metrics = results[model_name][cal_status]['test']
            comparison_data.append({
                'Model': model_name,
                'Calibration': cal_status,
                'Log_Loss': metrics['log_loss'],
                'Brier_Score': metrics['brier_score'],
                'Accuracy': metrics['accuracy'],
                'Draw_Accuracy': metrics['draw_accuracy'],
                'Away_Accuracy': metrics['away_accuracy'],
                'Home_Accuracy': metrics['home_accuracy'],
                'Draw_Pred_Pct': metrics['draw_pred_pct'],
                'Away_Pred_Pct': metrics['away_pred_pct'],
                'Home_Pred_Pct': metrics['home_pred_pct']
            })

    df_comparison = pd.DataFrame(comparison_data)

    # Print comparison table
    print("\n" + "=" * 140)
    print(f"{'Model':<20} {'Calibration':<15} {'Log Loss':<12} {'Brier':<10} {'Accuracy':<10} "
          f"{'Draw Acc':<10} {'Draw Pred%':<12}")
    print("-" * 140)
    for _, row in df_comparison.iterrows():
        print(f"{row['Model']:<20} {row['Calibration']:<15} {row['Log_Loss']:<12.4f} "
              f"{row['Brier_Score']:<10.4f} {row['Accuracy']:<10.4f} "
              f"{row['Draw_Accuracy']:<10.4f} {row['Draw_Pred_Pct']:<12.2%}")
    print("=" * 140)

    # Save comparison
    df_comparison.to_csv(output_dir / 'model_comparison.csv', index=False)

    # Save detailed results
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for model_key, model_data in results.items():
        results_serializable[model_key] = {
            'class_weights': model_data['class_weights'],
            'params': model_data['params'],
            'uncalibrated': {},
            'calibrated': {}
        }
        for cal_key in ['uncalibrated', 'calibrated']:
            for split_key in ['train', 'val', 'test']:
                metrics = model_data[cal_key][split_key].copy()
                if 'confusion_matrix' in metrics:
                    metrics['confusion_matrix'] = metrics['confusion_matrix'].tolist()
                results_serializable[model_key][cal_key][split_key] = metrics

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Save feature list
    with open(output_dir / 'feature_list.txt', 'w') as f:
        f.write('\n'.join(feature_cols))

    # Print recommendations
    logger.info("\n" + "=" * 100)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 100)

    best_logloss = df_comparison.loc[df_comparison['Log_Loss'].idxmin()]
    best_draw = df_comparison.loc[df_comparison['Draw_Accuracy'].idxmax()]

    logger.info(f"\n🏆 BEST LOG LOSS:")
    logger.info(f"   {best_logloss['Model']} ({best_logloss['Calibration']})")
    logger.info(f"   Log Loss: {best_logloss['Log_Loss']:.4f}")
    logger.info(f"   Draw Accuracy: {best_logloss['Draw_Accuracy']:.2%}")

    logger.info(f"\n🎯 BEST DRAW PREDICTION:")
    logger.info(f"   {best_draw['Model']} ({best_draw['Calibration']})")
    logger.info(f"   Draw Accuracy: {best_draw['Draw_Accuracy']:.2%}")
    logger.info(f"   Log Loss: {best_draw['Log_Loss']:.4f}")

    logger.info(f"\n💡 FOR BETTING:")
    logger.info(f"   Use: Conservative + Calibrated")
    logger.info(f"   File: {output_dir / 'model_conservative_calibrated.joblib'}")
    logger.info(f"   Reason: Best balance for real-world profit")

    logger.info(f"\n💡 FOR LOG LOSS COMPETITIONS:")
    logger.info(f"   Use: No Weights + Calibrated")
    logger.info(f"   File: {output_dir / 'model_no_weights_calibrated.joblib'}")
    logger.info(f"   Reason: Best pure prediction metric")

    logger.info("\n" + "=" * 100)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 100)
    logger.info(f"\nAll models saved to: {output_dir}")
    logger.info(f"Comparison saved to: {output_dir / 'model_comparison.csv'}")
    logger.info(f"Full results saved to: {output_dir / 'training_results.json'}")


if __name__ == '__main__':
    main()
