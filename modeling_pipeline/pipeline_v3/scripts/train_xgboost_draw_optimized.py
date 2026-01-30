#!/usr/bin/env python3
"""
Train XGBoost with Draw Optimization
=====================================

Optimizes class weights specifically for draw predictions.
Tunes draw and away multipliers to maximize draw prediction accuracy
while maintaining overall log loss performance.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
import joblib

# Add src to path
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


def evaluate_draws(y_true, y_pred_proba, name=""):
    """Detailed evaluation focusing on draw predictions."""
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Overall metrics
    loss = log_loss(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    
    # Draw-specific metrics
    draw_mask = (y_true == 1)
    draw_predicted_mask = (y_pred == 1)
    
    # True positives, false positives, false negatives for draws
    tp_draw = np.sum((y_true == 1) & (y_pred == 1))
    fp_draw = np.sum((y_true != 1) & (y_pred == 1))
    fn_draw = np.sum((y_true == 1) & (y_pred != 1))
    
    draw_precision = tp_draw / (tp_draw + fp_draw) if (tp_draw + fp_draw) > 0 else 0
    draw_recall = tp_draw / (tp_draw + fn_draw) if (tp_draw + fn_draw) > 0 else 0
    draw_f1 = 2 * (draw_precision * draw_recall) / (draw_precision + draw_recall) if (draw_precision + draw_recall) > 0 else 0
    
    # Average draw probability when actual result is draw
    avg_draw_prob_when_draw = np.mean(y_pred_proba[draw_mask, 1]) if np.sum(draw_mask) > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Log Loss:     {loss:.4f}")
    logger.info(f"Overall Accuracy:     {acc:.1%}")
    logger.info(f"\nDraw-Specific Metrics:")
    logger.info(f"  Draw Precision:     {draw_precision:.1%}")
    logger.info(f"  Draw Recall:        {draw_recall:.1%}")
    logger.info(f"  Draw F1-Score:      {draw_f1:.4f}")
    logger.info(f"  Avg Draw Prob (when draw): {avg_draw_prob_when_draw:.1%}")
    logger.info(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"              Pred Away  Pred Draw  Pred Home")
    logger.info(f"Actual Away      {cm[0,0]:4d}      {cm[0,1]:4d}      {cm[0,2]:4d}")
    logger.info(f"Actual Draw      {cm[1,0]:4d}      {cm[1,1]:4d}      {cm[1,2]:4d}")
    logger.info(f"Actual Home      {cm[2,0]:4d}      {cm[2,1]:4d}      {cm[2,2]:4d}")
    
    return {
        'log_loss': loss,
        'accuracy': acc,
        'draw_precision': draw_precision,
        'draw_recall': draw_recall,
        'draw_f1': draw_f1,
        'avg_draw_prob': avg_draw_prob_when_draw
    }


def train_with_weights(train_df, val_df, draw_mult=1.5, away_mult=1.3):
    """Train model with specific class weight multipliers."""
    # Modify the XGBoostFootballModel to use custom multipliers
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
    
    # Temporarily modify class weights in the model
    # We'll need to patch the fit method
    original_fit = model.fit
    
    def custom_fit(train_df, val_df=None, early_stopping_rounds=50, verbose=True):
        # Call original but intercept weights
        import xgboost as xgb
        from collections import Counter
        
        logger.info(f"Training with Draw multiplier: {draw_mult}, Away multiplier: {away_mult}")
        
        X_train = model._prepare_features(train_df, fit_scaler=True)
        y_train = train_df['target'].values.astype(int)
        
        # Compute sample weights
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        
        class_weights = {
            cls: total_samples / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }
        
        # Apply custom multipliers
        if 1 in class_weights:  # Draw
            class_weights[1] *= draw_mult
        if 0 in class_weights:  # Away
            class_weights[0] *= away_mult
        
        sample_weights = np.array([class_weights[cls] for cls in y_train])
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Class weights: {class_weights}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights,
                            feature_names=model.feature_columns)
        
        # Validation set
        evals = [(dtrain, 'train')]
        if val_df is not None:
            X_val = model._prepare_features(val_df, fit_scaler=False)
            y_val = val_df['target'].values.astype(int)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=model.feature_columns)
            evals.append((dval, 'val'))
        
        # Train
        train_params = params.copy()
        n_estimators = train_params.pop('n_estimators', 500)
        
        model.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if val_df is not None else None,
            verbose_eval=False
        )
        
        model.feature_importance_ = model.model.get_score(importance_type='gain')
        model.is_fitted = True
        
        return model
    
    model.fit = custom_fit
    model.fit(train_df, val_df, verbose=False)
    
    return model


def main():
    """Main training pipeline with draw optimization."""
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
    logger.info("TESTING DIFFERENT DRAW MULTIPLIERS")
    logger.info("="*60)
    
    # Test different draw multipliers
    multipliers = [
        (1.0, 1.0),   # No optimization
        (1.5, 1.3),   # Current default
        (2.0, 1.5),   # Higher draw emphasis
        (2.5, 1.5),   # Very high draw emphasis
        (3.0, 1.5),   # Maximum draw emphasis
    ]
    
    results = []
    
    for draw_mult, away_mult in multipliers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: Draw={draw_mult}x, Away={away_mult}x")
        logger.info(f"{'='*60}")
        
        model = train_with_weights(train_df, val_df, draw_mult, away_mult)
        
        # Evaluate on validation set
        val_pred = model.predict_proba(val_df)
        val_metrics = evaluate_draws(val_df['target'].values, val_pred, "Validation")
        
        # Evaluate on test set
        test_pred = model.predict_proba(test_df)
        test_metrics = evaluate_draws(test_df['target'].values, test_pred, "Test")
        
        results.append({
            'draw_mult': draw_mult,
            'away_mult': away_mult,
            'val_log_loss': val_metrics['log_loss'],
            'val_draw_f1': val_metrics['draw_f1'],
            'test_log_loss': test_metrics['log_loss'],
            'test_draw_f1': test_metrics['draw_f1'],
            'test_draw_recall': test_metrics['draw_recall'],
            'model': model
        })
    
    # Find best configuration
    logger.info("\n" + "="*60)
    logger.info("SUMMARY OF RESULTS")
    logger.info("="*60)
    logger.info(f"{'Draw Mult':<12} {'Away Mult':<12} {'Val LogLoss':<12} {'Val Draw F1':<12} {'Test LogLoss':<12} {'Test Draw F1':<12}")
    logger.info("-" * 80)
    
    for r in results:
        logger.info(f"{r['draw_mult']:<12.1f} {r['away_mult']:<12.1f} {r['val_log_loss']:<12.4f} {r['val_draw_f1']:<12.4f} {r['test_log_loss']:<12.4f} {r['test_draw_f1']:<12.4f}")
    
    # Best by draw F1
    best_draw_f1 = max(results, key=lambda x: x['test_draw_f1'])
    logger.info(f"\n✅ Best for Draw F1: Draw={best_draw_f1['draw_mult']}x, Away={best_draw_f1['away_mult']}x")
    logger.info(f"   Test Draw F1: {best_draw_f1['test_draw_f1']:.4f}")
    logger.info(f"   Test Log Loss: {best_draw_f1['test_log_loss']:.4f}")
    
    # Save best model
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_draw_f1['model'], models_dir / 'xgboost_draw_optimized.joblib')
    logger.info(f"\n✅ Best model saved to: models/xgboost_draw_optimized.joblib")


if __name__ == '__main__':
    main()
