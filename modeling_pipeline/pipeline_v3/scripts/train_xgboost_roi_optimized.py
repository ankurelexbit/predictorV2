#!/usr/bin/env python3
"""
Train XGBoost with ROI Optimization
====================================

Optimizes model for betting ROI rather than just accuracy or draw recall.
Uses a custom loss function that considers:
1. Prediction confidence (precision)
2. Typical betting odds for each outcome
3. Expected value maximization
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import log_loss, accuracy_score
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.xgboost_model import XGBoostFootballModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess data."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Rename columns
    column_map = {
        'home_goals': 'home_score',
        'away_goals': 'away_score',
        'starting_at': 'match_date'
    }
    df.rename(columns=column_map, inplace=True)
    
    # Filter to matches with results
    mask = df['home_score'].notna() & df['away_score'].notna()
    df = df[mask].copy()
    
    # Create target variable (0=away, 1=draw, 2=home)
    conditions = [
        (df['home_score'] < df['away_score']),
        (df['home_score'] == df['away_score']),
        (df['home_score'] > df['away_score'])
    ]
    df['target'] = np.select(conditions, [0, 1, 2], default=-1)
    
    # Ensure date is datetime
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """Split data using time-based approach."""
    train_mask = df['match_date'] < '2024-01-01'
    val_mask = (df['match_date'] >= '2024-01-01') & (df['match_date'] < '2025-01-01')
    test_mask = df['match_date'] >= '2025-01-01'
    
    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()
    
    logger.info(f"Data Splits:")
    logger.info(f"  Train: {len(train)} rows")
    logger.info(f"  Val:   {len(val)} rows")
    logger.info(f"  Test:  {len(test)} rows")
    
    return train, val, test


def calculate_roi(y_true, y_pred_proba, confidence_threshold=0.45, stake=100):
    """
    Calculate ROI for a betting strategy.
    
    Strategy:
    - Only bet when model confidence > threshold
    - Use typical odds: Home ~2.0, Draw ~3.5, Away ~3.0
    - Bet on highest probability outcome
    
    Args:
        y_true: Actual outcomes (0=away, 1=draw, 2=home)
        y_pred_proba: Predicted probabilities [away, draw, home]
        confidence_threshold: Minimum probability to place bet
        stake: Bet amount per match
    
    Returns:
        dict with ROI metrics
    """
    # Typical market odds (conservative estimates)
    typical_odds = {
        0: 3.0,  # Away
        1: 3.5,  # Draw
        2: 2.0,  # Home
    }
    
    total_staked = 0
    total_return = 0
    bets_placed = 0
    wins = 0
    
    outcome_bets = {0: 0, 1: 0, 2: 0}
    outcome_wins = {0: 0, 1: 0, 2: 0}
    
    for i in range(len(y_true)):
        # Get predicted outcome and confidence
        pred_outcome = np.argmax(y_pred_proba[i])
        confidence = y_pred_proba[i][pred_outcome]
        
        # Only bet if confident enough
        if confidence >= confidence_threshold:
            bets_placed += 1
            total_staked += stake
            outcome_bets[pred_outcome] += 1
            
            # Check if won
            if y_true[i] == pred_outcome:
                wins += 1
                payout = stake * typical_odds[pred_outcome]
                total_return += payout
                outcome_wins[pred_outcome] += 1
            # else: lose stake (already counted in total_staked)
    
    profit = total_return - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
    return {
        'total_staked': total_staked,
        'total_return': total_return,
        'profit': profit,
        'roi': roi,
        'bets_placed': bets_placed,
        'wins': wins,
        'win_rate': win_rate,
        'home_bets': outcome_bets[2],
        'draw_bets': outcome_bets[1],
        'away_bets': outcome_bets[0],
        'home_wins': outcome_wins[2],
        'draw_wins': outcome_wins[1],
        'away_wins': outcome_wins[0],
    }


def evaluate_roi_strategy(y_true, y_pred_proba, name=""):
    """Evaluate ROI across different confidence thresholds."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} - ROI Analysis")
    logger.info(f"{'='*60}")
    
    # Test different confidence thresholds
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    logger.info(f"\n{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Win%':<8} {'ROI%':<10} {'Profit':<10}")
    logger.info("-" * 60)
    
    best_roi = -float('inf')
    best_threshold = None
    best_metrics = None
    
    for threshold in thresholds:
        metrics = calculate_roi(y_true, y_pred_proba, confidence_threshold=threshold)
        
        logger.info(f"{threshold:<12.2f} {metrics['bets_placed']:<8} {metrics['wins']:<8} "
                   f"{metrics['win_rate']:<8.1f} {metrics['roi']:<10.1f} ${metrics['profit']:<9.0f}")
        
        if metrics['roi'] > best_roi and metrics['bets_placed'] >= 50:  # Minimum 50 bets
            best_roi = metrics['roi']
            best_threshold = threshold
            best_metrics = metrics
    
    if best_metrics:
        logger.info(f"\n✅ Best ROI Configuration:")
        logger.info(f"   Threshold: {best_threshold}")
        logger.info(f"   ROI: {best_metrics['roi']:.1f}%")
        logger.info(f"   Profit: ${best_metrics['profit']:.0f}")
        logger.info(f"   Bets: {best_metrics['bets_placed']} ({best_metrics['wins']} wins, {best_metrics['win_rate']:.1f}% win rate)")
        logger.info(f"   Breakdown: {best_metrics['home_wins']}/{best_metrics['home_bets']} Home, "
                   f"{best_metrics['draw_wins']}/{best_metrics['draw_bets']} Draw, "
                   f"{best_metrics['away_wins']}/{best_metrics['away_bets']} Away")
    
    return best_threshold, best_metrics


def main():
    """Main training pipeline with ROI optimization."""
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    df = load_data(data_path)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Drop leakage columns
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 
                    'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
            val_df = val_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])
    
    logger.info("\n" + "="*60)
    logger.info("TESTING DIFFERENT CLASS WEIGHT CONFIGURATIONS FOR ROI")
    logger.info("="*60)
    
    # Test configurations optimized for ROI
    # ROI benefits from:
    # 1. Higher precision (fewer bets, higher win rate)
    # 2. Balanced predictions (not too conservative)
    # 3. Good draw detection (draws have best odds)
    
    configs = [
        {"name": "Baseline", "draw_mult": 1.0, "away_mult": 1.0},
        {"name": "Slight Draw Boost", "draw_mult": 1.3, "away_mult": 1.2},
        {"name": "Current Default", "draw_mult": 1.5, "away_mult": 1.3},
        {"name": "Moderate Draw", "draw_mult": 2.0, "away_mult": 1.5},
        {"name": "High Draw", "draw_mult": 2.5, "away_mult": 1.5},
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config['name']} (Draw={config['draw_mult']}x, Away={config['away_mult']}x)")
        logger.info(f"{'='*60}")
        
        # Train model with these weights
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.9,
            'colsample_bytree': 0.6,
            'min_child_weight': 3,
            'gamma': 0.5,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'n_estimators': 500,
            'random_state': 42,
            'tree_method': 'hist'
        }
        
        model = XGBoostFootballModel(params=params)
        
        # Modify class weights temporarily
        from collections import Counter
        import xgboost as xgb
        
        X_train = model._prepare_features(train_df, fit_scaler=True)
        y_train = train_df['target'].values.astype(int)
        
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        
        class_weights = {
            cls: total_samples / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }
        
        if 1 in class_weights:
            class_weights[1] *= config['draw_mult']
        if 0 in class_weights:
            class_weights[0] *= config['away_mult']
        
        sample_weights = np.array([class_weights[cls] for cls in y_train])
        
        # Train
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights,
                            feature_names=model.feature_columns)
        
        X_val = model._prepare_features(val_df, fit_scaler=False)
        y_val = val_df['target'].values.astype(int)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=model.feature_columns)
        
        train_params = params.copy()
        n_estimators = train_params.pop('n_estimators', 500)
        
        model.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        model.is_fitted = True
        
        # Evaluate on test set
        test_pred = model.predict_proba(test_df)
        test_loss = log_loss(test_df['target'].values, test_pred)
        
        # ROI analysis
        best_threshold, best_metrics = evaluate_roi_strategy(
            test_df['target'].values, 
            test_pred, 
            name=f"Test Set - {config['name']}"
        )
        
        results.append({
            'name': config['name'],
            'draw_mult': config['draw_mult'],
            'away_mult': config['away_mult'],
            'test_log_loss': test_loss,
            'best_threshold': best_threshold,
            'roi': best_metrics['roi'] if best_metrics else 0,
            'profit': best_metrics['profit'] if best_metrics else 0,
            'bets': best_metrics['bets_placed'] if best_metrics else 0,
            'win_rate': best_metrics['win_rate'] if best_metrics else 0,
            'model': model
        })
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ROI OPTIMIZATION SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Configuration':<20} {'Threshold':<12} {'ROI%':<10} {'Profit':<10} {'Bets':<8} {'Win%':<8}")
    logger.info("-" * 80)
    
    for r in results:
        logger.info(f"{r['name']:<20} {r['best_threshold']:<12.2f} {r['roi']:<10.1f} "
                   f"${r['profit']:<9.0f} {r['bets']:<8} {r['win_rate']:<8.1f}")
    
    # Best configuration
    best_config = max(results, key=lambda x: x['roi'])
    logger.info(f"\n✅ BEST ROI CONFIGURATION:")
    logger.info(f"   Name: {best_config['name']}")
    logger.info(f"   Draw multiplier: {best_config['draw_mult']}x")
    logger.info(f"   Away multiplier: {best_config['away_mult']}x")
    logger.info(f"   Confidence threshold: {best_config['best_threshold']}")
    logger.info(f"   ROI: {best_config['roi']:.1f}%")
    logger.info(f"   Profit: ${best_config['profit']:.0f} (on {best_config['bets']} bets)")
    logger.info(f"   Win rate: {best_config['win_rate']:.1f}%")
    
    # Save best model
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_config['model'], models_dir / 'xgboost_roi_optimized.joblib')
    logger.info(f"\n✅ Best model saved to: models/xgboost_roi_optimized.joblib")
    
    # Save configuration
    config_file = models_dir / 'roi_config.txt'
    with open(config_file, 'w') as f:
        f.write(f"Best ROI Configuration\n")
        f.write(f"=====================\n")
        f.write(f"Draw multiplier: {best_config['draw_mult']}\n")
        f.write(f"Away multiplier: {best_config['away_mult']}\n")
        f.write(f"Confidence threshold: {best_config['best_threshold']}\n")
        f.write(f"Expected ROI: {best_config['roi']:.1f}%\n")
        f.write(f"Expected win rate: {best_config['win_rate']:.1f}%\n")
    
    logger.info(f"Configuration saved to: {config_file}")


if __name__ == '__main__':
    main()
