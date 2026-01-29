"""
Train Model with Feature Selection - Phase 7

Complete training pipeline with feature selection:
1. Load training data
2. Perform feature selection
3. Train XGBoost with selected features
4. Evaluate and compare to baseline

Usage:
    python train_with_feature_selection.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.feature_selector import FeatureSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> tuple:
    """Load and prepare training data."""
    logger.info(f"Loading training data from {data_path}")
    
    df = pd.read_csv(data_path)

    # 1. Map Target to Int (H=2, D=1, A=0)
    target_map = {'H': 2, 'D': 1, 'A': 0}
    if df['result'].dtype == 'object':
         df['target_encoded'] = df['result'].map(target_map)
    else:
         df['target_encoded'] = df['result'] # Assume already int if not object

    # 2. Drop Leakage & ID Columns
    target_col = 'target_encoded'
    leakage_cols = ['result', 'target_home_win', 'target_draw', 'target_away_win', 'home_goals', 'away_goals']
    id_cols = ['fixture_id', 'date', 'match_date', 'starting_at', 'home_team_id', 'away_team_id', 'league_id', 'season_id']
    
    exclude_cols = leakage_cols + id_cols + [target_col]
    
    # 3. Drop Zero-Variance Columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1 and c not in exclude_cols]
    if constant_cols:
        logger.info(f"Dropping {len(constant_cols)} constant columns")
        exclude_cols.extend(constant_cols)

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Ensure numeric
    X = X.select_dtypes(include=[np.number])
    
    logger.info(f"Loaded {len(df)} samples with {X.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, df


def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train baseline model with all features."""
    logger.info("\n" + "="*60)
    logger.info("Training Baseline Model (All Features)")
    logger.info("="*60)
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict_proba(X_train)
    test_pred = model.predict_proba(X_test)
    
    train_loss = log_loss(y_train, train_pred)
    test_loss = log_loss(y_test, test_pred)
    
    logger.info(f"Baseline - Train Log Loss: {train_loss:.4f}")
    logger.info(f"Baseline - Test Log Loss: {test_loss:.4f}")
    
    return model, test_loss


def train_with_feature_selection(X_train, y_train, X_test, y_test):
    """Train model with feature selection."""
    logger.info("\n" + "="*60)
    logger.info("Performing Feature Selection")
    logger.info("="*60)
    
    # Feature selection
    selector = FeatureSelector(
        correlation_threshold=0.95,
        importance_threshold=0.001,
        target_features=200
    )
    
    X_train_selected = selector.fit_transform(X_train, y_train, method='all')
    X_test_selected = selector.transform(X_test)
    
    logger.info(f"Selected {len(selector.selected_features)} features")
    
    # Train model with selected features
    logger.info("\n" + "="*60)
    logger.info("Training Model with Selected Features")
    logger.info("="*60)
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    model.fit(X_train_selected, y_train)
    
    # Evaluate
    train_pred = model.predict_proba(X_train_selected)
    test_pred = model.predict_proba(X_test_selected)
    
    train_loss = log_loss(y_train, train_pred)
    test_loss = log_loss(y_test, test_pred)
    
    logger.info(f"Selected - Train Log Loss: {train_loss:.4f}")
    logger.info(f"Selected - Test Log Loss: {test_loss:.4f}")
    
    return model, selector, test_loss


def main():
    """Main training pipeline."""
    # Paths
    data_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    X, y, df = load_training_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train baseline
    baseline_model, baseline_loss = train_baseline_model(
        X_train, y_train, X_test, y_test
    )
    
    # Train with feature selection
    selected_model, selector, selected_loss = train_with_feature_selection(
        X_train, y_train, X_test, y_test
    )
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("RESULTS COMPARISON")
    logger.info("="*60)
    logger.info(f"Baseline Model:")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Test Log Loss: {baseline_loss:.4f}")
    logger.info(f"\nWith Feature Selection:")
    logger.info(f"  Features: {len(selector.selected_features)}")
    logger.info(f"  Test Log Loss: {selected_loss:.4f}")
    logger.info(f"\nImprovement: {baseline_loss - selected_loss:.4f}")
    logger.info(f"Feature Reduction: {X_train.shape[1] - len(selector.selected_features)} features removed")
    
    # Save models and selector
    logger.info("\n" + "="*60)
    logger.info("Saving Models")
    logger.info("="*60)
    
    joblib.dump(baseline_model, models_dir / 'xgboost_baseline.pkl')
    logger.info(f"Saved baseline model")
    
    joblib.dump(selected_model, models_dir / 'xgboost_selected.pkl')
    logger.info(f"Saved selected model")
    
    selector.save(models_dir / 'feature_selector.pkl')
    logger.info(f"Saved feature selector")
    
    # Save feature importance
    importance = selector.get_feature_importance_report()
    importance.to_csv(models_dir / 'feature_importance.csv', index=False)
    logger.info(f"Saved feature importance")
    
    # Show top features
    logger.info("\n" + "="*60)
    logger.info("Top 20 Most Important Features")
    logger.info("="*60)
    for idx, row in importance.head(20).iterrows():
        logger.info(f"{row['feature']:50s} {row['importance']:.6f}")
    
    logger.info("\n✅ Training complete!")


if __name__ == '__main__':
    main()
