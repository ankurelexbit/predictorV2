#!/usr/bin/env python3
"""
Fair Ensemble Comparison - Time-Based Split
============================================

Train ensemble model using the same time-based split as XGBoost
for a fair performance comparison.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.impute import SimpleImputer
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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


def split_data_time_based(df: pd.DataFrame) -> tuple:
    """Split data using time-based approach (same as XGBoost)."""
    train_mask = df['match_date'] < '2024-01-01'
    val_mask = (df['match_date'] >= '2024-01-01') & (df['match_date'] < '2025-01-01')
    test_mask = df['match_date'] >= '2025-01-01'
    
    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()
    
    logger.info(f"Time-Based Data Splits:")
    logger.info(f"  Train (< 2024):  {len(train)} rows")
    logger.info(f"  Val (2024):      {len(val)} rows")
    logger.info(f"  Test (2025+):    {len(test)} rows")
    
    return train, val, test


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features by dropping leakage columns."""
    # Drop zero-variance columns
    zero_var_cols = [col for col in df.columns 
                     if df[col].nunique() <= 1 
                     and col not in ['fixture_id', 'target', 'league_id', 'season_id']]
    if zero_var_cols:
        logger.info(f"Dropping {len(zero_var_cols)} constant columns")
        df = df.drop(columns=zero_var_cols)
    
    # Drop data leakage columns
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 
                    'result', 'home_score', 'away_score', 'match_date',
                    'fixture_id', 'home_team_id', 'away_team_id', 'league_id']
    to_drop = [c for c in leakage_cols if c in df.columns]
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} non-feature columns")
        df = df.drop(columns=to_drop)
    
    return df


class EnsembleModel:
    """Ensemble of XGBoost, LightGBM, and Neural Network."""
    
    def __init__(self, xgb_weight=0.6, lgb_weight=0.3, nn_weight=0.1):
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.nn_weight = nn_weight
        
        # Normalize weights
        total = xgb_weight + lgb_weight + nn_weight
        self.xgb_weight /= total
        self.lgb_weight /= total
        self.nn_weight /= total
        
        self.xgb_model = None
        self.lgb_model = None
        self.nn_model = None
        self.imputer = None
        
        logger.info(f"Ensemble weights: XGB={self.xgb_weight:.2f}, "
                   f"LGB={self.lgb_weight:.2f}, NN={self.nn_weight:.2f}")
    
    def fit(self, X_train, y_train):
        """Train all models in the ensemble."""
        logger.info("\n" + "="*60)
        logger.info("Training Ensemble Models")
        logger.info("="*60)
        
        # Train XGBoost
        logger.info("\n1. Training XGBoost...")
        self.xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Train LightGBM
        logger.info("2. Training LightGBM...")
        self.lgb_model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.lgb_model.fit(X_train, y_train)
        
        # Train Neural Network with imputation
        logger.info("3. Training Neural Network...")
        self.imputer = SimpleImputer(strategy='mean')
        X_train_imputed = self.imputer.fit_transform(X_train)
        
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            verbose=False
        )
        self.nn_model.fit(X_train_imputed, y_train)
        
        logger.info("✅ Ensemble training complete")
    
    def predict_proba(self, X):
        """Predict probabilities using weighted ensemble."""
        xgb_pred = self.xgb_model.predict_proba(X)
        lgb_pred = self.lgb_model.predict_proba(X)
        
        # Impute for NN
        X_imputed = self.imputer.transform(X)
        nn_pred = self.nn_model.predict_proba(X_imputed)
        
        # Weighted average
        ensemble_pred = (
            self.xgb_weight * xgb_pred +
            self.lgb_weight * lgb_pred +
            self.nn_weight * nn_pred
        )
        
        return ensemble_pred
    
    def evaluate(self, X, y, name=""):
        """Evaluate ensemble and individual models."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation - {name}")
        logger.info(f"{'='*60}")
        
        # Individual model predictions
        xgb_pred = self.xgb_model.predict_proba(X)
        lgb_pred = self.lgb_model.predict_proba(X)
        
        X_imputed = self.imputer.transform(X)
        nn_pred = self.nn_model.predict_proba(X_imputed)
        
        ensemble_pred = self.predict_proba(X)
        
        # Calculate metrics
        xgb_loss = log_loss(y, xgb_pred)
        lgb_loss = log_loss(y, lgb_pred)
        nn_loss = log_loss(y, nn_pred)
        ensemble_loss = log_loss(y, ensemble_pred)
        
        xgb_acc = accuracy_score(y, np.argmax(xgb_pred, axis=1))
        ensemble_acc = accuracy_score(y, np.argmax(ensemble_pred, axis=1))
        
        logger.info(f"XGBoost Log Loss:     {xgb_loss:.4f}")
        logger.info(f"LightGBM Log Loss:    {lgb_loss:.4f}")
        logger.info(f"Neural Net Log Loss:  {nn_loss:.4f}")
        logger.info(f"Ensemble Log Loss:    {ensemble_loss:.4f}")
        logger.info(f"Ensemble Accuracy:    {ensemble_acc:.1%}")
        
        best_individual = min(xgb_loss, lgb_loss, nn_loss)
        improvement = best_individual - ensemble_loss
        logger.info(f"\nEnsemble Improvement: {improvement:+.4f}")
        
        return ensemble_loss, ensemble_acc


def main():
    """Main training pipeline."""
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    df = load_data(data_path)
    logger.info(f"Loaded {len(df)} matches")
    
    # Split data (time-based)
    train_df, val_df, test_df = split_data_time_based(df)
    
    # Prepare features - do this BEFORE splitting to ensure consistency
    # Drop leakage columns from full dataset first
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 
                    'result', 'home_score', 'away_score', 'match_date',
                    'fixture_id', 'home_team_id', 'away_team_id', 'league_id']
    
    # Get feature columns (everything except target and leakage)
    feature_cols = [c for c in df.columns if c not in leakage_cols + ['target']]
    
    # Drop zero-variance columns based on TRAINING set only
    train_features = train_df[feature_cols]
    zero_var_cols = [col for col in feature_cols 
                     if train_features[col].nunique() <= 1]
    if zero_var_cols:
        logger.info(f"Dropping {len(zero_var_cols)} constant columns from training set")
        feature_cols = [c for c in feature_cols if c not in zero_var_cols]
    
    # Now extract features with consistent columns
    train_X = train_df[feature_cols]
    val_X = val_df[feature_cols]
    test_X = test_df[feature_cols]
    
    train_y = train_df['target'].values.astype(int)
    val_y = val_df['target'].values.astype(int)
    test_y = test_df['target'].values.astype(int)
    
    logger.info(f"\nFeature count: {len(feature_cols)}")
    
    # Train ensemble
    ensemble = EnsembleModel(xgb_weight=0.6, lgb_weight=0.3, nn_weight=0.1)
    ensemble.fit(train_X, train_y)
    
    # Evaluate
    train_loss, train_acc = ensemble.evaluate(train_X, train_y, "Train Set")
    val_loss, val_acc = ensemble.evaluate(val_X, val_y, "Validation Set (2024)")
    test_loss, test_acc = ensemble.evaluate(test_X, test_y, "Test Set (2025+)")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS (Time-Based Split)")
    logger.info("="*60)
    logger.info(f"Train Log Loss:      {train_loss:.4f}")
    logger.info(f"Validation Log Loss: {val_loss:.4f}")
    logger.info(f"Test Log Loss:       {test_loss:.4f}")
    logger.info(f"Test Accuracy:       {test_acc:.1%}")
    
    # Save model
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    joblib.dump(ensemble, models_dir / 'ensemble_time_based.pkl')
    logger.info(f"\nModel saved to {models_dir / 'ensemble_time_based.pkl'}")


if __name__ == '__main__':
    main()
