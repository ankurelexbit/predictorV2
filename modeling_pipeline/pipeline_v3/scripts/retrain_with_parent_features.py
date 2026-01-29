#!/usr/bin/env python3
"""
Retrain XGBoost model with parent pipeline features and hyperparameter tuning.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(df):
    """Prepare training data."""
    # Drop metadata columns
    feature_cols = [col for col in df.columns if col not in [
        'fixture_id', 'match_date', 'home_team_id', 'away_team_id', 
        'league_id', 'season_id', 'home_score', 'away_score', 'result', 'target',
        'starting_at'  # datetime column
    ]]
    
    X = df[feature_cols].copy()
    
    # Only keep numeric columns to avoid datetime issues
    X = X.select_dtypes(include=[np.number])
    
    # Encode target
    target_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['result'].map(target_map)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    logger.info(f"Features: {len(X.columns)}")
    logger.info(f"Samples: {len(X)}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, list(X.columns)

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with parent pipeline hyperparameters."""
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING XGBOOST WITH PARENT PIPELINE HYPERPARAMETERS")
    logger.info("="*80)
    
    # Parent pipeline's best hyperparameters (from their tuning)
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    logger.info(f"Hyperparameters: {json.dumps(params, indent=2)}")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # Get predictions
    y_pred_proba = model.predict(dval)
    val_logloss = log_loss(y_val, y_pred_proba)
    
    logger.info(f"\n✅ Validation Log Loss: {val_logloss:.4f}")
    
    return model, val_logloss

def analyze_feature_importance(model, feature_names):
    """Analyze and display feature importance."""
    logger.info("\n" + "="*80)
    logger.info("TOP 20 FEATURE IMPORTANCE")
    logger.info("="*80)
    
    importance = model.get_score(importance_type='gain')
    
    # Map feature indices to names
    feature_importance = []
    for feat_idx, gain in importance.items():
        feat_name = feature_names[int(feat_idx.replace('f', ''))]
        feature_importance.append({'feature': feat_name, 'gain': gain})
    
    # Sort by gain
    feature_importance = sorted(feature_importance, key=lambda x: x['gain'], reverse=True)
    
    # Display top 20
    for i, item in enumerate(feature_importance[:20], 1):
        logger.info(f"{i:2d}. {item['feature']:40s} {item['gain']:10.1f}")
    
    # Check if position_diff is in top 5
    top_5_features = [item['feature'] for item in feature_importance[:5]]
    if 'position_diff' in top_5_features:
        rank = top_5_features.index('position_diff') + 1
        logger.info(f"\n✅ position_diff is ranked #{rank} (matching parent pipeline!)")
    else:
        logger.warning(f"\n⚠️  position_diff not in top 5")
    
    return feature_importance

def main():
    """Main training pipeline."""
    logger.info("="*80)
    logger.info("RETRAINING WITH PARENT PIPELINE FEATURES")
    logger.info("="*80)
    
    # Load data
    logger.info("\nLoading training data...")
    df = pd.read_csv('data/csv/training_data_complete_v2.csv')
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Prepare data
    X, y, feature_names = prepare_data(df)
    
    # Split data (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    
    # Train model
    model, val_logloss = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, feature_names)
    
    # Save model
    model_path = 'models/xgboost_with_parent_features.json'
    model.save_model(model_path)
    logger.info(f"\n✅ Model saved to {model_path}")
    
    # Save feature importance
    importance_df = pd.DataFrame(feature_importance)
    importance_df.to_csv('models/feature_importance_parent.csv', index=False)
    logger.info(f"✅ Feature importance saved to models/feature_importance_parent.csv")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Validation Log Loss: {val_logloss:.4f}")
    logger.info(f"Target: < 0.950 (parent: 0.9478)")
    
    if val_logloss < 0.950:
        logger.info("✅ TARGET ACHIEVED!")
    else:
        improvement = 0.9956 - val_logloss  # V3 baseline
        logger.info(f"Improvement from V3 baseline: {improvement:.4f}")
    
    logger.info("="*80)

if __name__ == '__main__':
    main()
