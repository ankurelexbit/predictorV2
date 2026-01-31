"""
Train V4 XGBoost Model with Draw-Focused Tuning.

Implements:
- Data loading and preprocessing
- Time-series split (Train/Val/Test)
- Hyperparameter tuning focused on draw prediction
- Model training and evaluation
- Feature importance analysis
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.xgboost_model import XGBoostFootballModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Surpress warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_path: str, columns_to_drop: list = None):
    """Load and preprocess training data."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Convert date
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    # Map target
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    
    # Drop rows with missing target (shouldn't happen for training data)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    
    # Sort by date
    df = df.sort_values('match_date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} samples")
    return df

def get_train_val_test_split(df: pd.DataFrame, train_split: float = 0.70, val_split: float = 0.15):
    """Split data chronologically."""
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Data Split:")
    logger.info(f"  Train:      {len(train_df)} ({len(train_df)/n:.1%}) - {train_df['match_date'].min().date()} to {train_df['match_date'].max().date()}")
    logger.info(f"  Validation: {len(val_df)} ({len(val_df)/n:.1%}) - {val_df['match_date'].min().date()} to {val_df['match_date'].max().date()}")
    logger.info(f"  Test:       {len(test_df)} ({len(test_df)/n:.1%}) - {test_df['match_date'].min().date()} to {test_df['match_date'].max().date()}")
    
    return train_df, val_df, test_df

def run_draw_focused_tuning(train_df, val_df, features, target_col='target', n_trials=20):
    """Run hyperparameter tuning focused on predicting draws."""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING DRAW-FOCUSED HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_val = val_df[features]
    y_val = val_df[target_col]
    
    # Conservative parameter grid (good for draws)
    param_grid = {
        'max_depth': [3, 4, 5],            # Lower depth = less overfitting
        'learning_rate': [0.01, 0.03],     # Slower learning
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7],
        'min_child_weight': [10, 15, 20],  # High weight = conservative
        'gamma': [1.0, 2.0, 3.0],          # Regularization
        'reg_alpha': [0.5, 1.0],
        'reg_lambda': [1.0, 2.0]
    }
    
    best_score = float('inf')
    best_params = None
    best_draw_count = 0
    
    for trial in range(n_trials):
        # Sample parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': 500,
            'random_state': 42 + trial,
            'n_jobs': -1,
            'verbosity': 0
        }
        for k, v in param_grid.items():
            params[k] = np.random.choice(v)
            
        try:
            # Train
            model = XGBoostFootballModel(params=params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Predict
            val_probs = model.predict_proba(X_val)
            val_preds = np.argmax(val_probs, axis=1)
            
            # Score
            score = log_loss(y_val, val_probs)
            draw_count = (val_preds == 1).sum()
            draw_pct = draw_count / len(val_preds) * 100
            
            # Update best (prioritize reasonable draw predictions + good loss)
            if draw_count > best_draw_count or (draw_count >= best_draw_count and score < best_score):
                logger.info(f"Trial {trial+1}/{n_trials}: LogLoss={score:.4f}, Draws={draw_count} ({draw_pct:.1f}%) -- NEW BEST")
                best_score = score
                best_draw_count = draw_count
                best_params = params
            else:
                logger.debug(f"Trial {trial+1}/{n_trials}: LogLoss={score:.4f}, Draws={draw_count} ({draw_pct:.1f}%)")
                
        except Exception as e:
            logger.error(f"Trial {trial+1} failed: {e}")
            
    return best_params

def main():
    parser = argparse.ArgumentParser(description='Train V4 XGBoost Model')
    parser.add_argument('--data', default='data/training_data.csv', help='Training data path')
    parser.add_argument('--output', default='models/v4_xgboost.joblib', help='Model output path')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    args = parser.parse_args()
    
    # 1. Load Data
    features_to_exclude = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id', 
        'match_date', 'home_score', 'away_score', 'result', 'target', 
        'home_team_name', 'away_team_name', 'state_id'
    ]
    
    df = load_and_preprocess_data(args.data)
    
    # Identify features
    feature_cols = [c for c in df.columns if c not in features_to_exclude]
    logger.info(f"Using {len(feature_cols)} features")
    
    # 2. Split Data
    train_df, val_df, test_df = get_train_val_test_split(df)
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    # 3. Tuning or Default Params
    if args.tune:
        best_params = run_draw_focused_tuning(train_df, val_df, feature_cols)
    else:
        logger.info("Using default draw-focused parameters")
        best_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'learning_rate': 0.03,
            'max_depth': 4,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'gamma': 1.0,
            'n_estimators': 1000,
            'random_state': 42,
            'n_jobs': -1
        }
    
    # 4. Train Final Model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MODEL")
    logger.info("=" * 80)
    
    final_model = XGBoostFootballModel(params=best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # Calibrate
    final_model.calibrate(X_val, y_val, method='isotonic')
    
    # 5. Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)
    
    # Probabilities
    test_probs = final_model.predict_proba(X_test, calibrated=True)
    test_preds = np.argmax(test_probs, axis=1)
    
    # Metrics
    loss = log_loss(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    
    logger.info(f"Log Loss: {loss:.4f}")
    logger.info(f"Accuracy: {acc:.1%}")
    
    # Draw Analysis
    draw_preds = (test_preds == 1).sum()
    draw_actual = (y_test == 1).sum()
    logger.info(f"Draws Predicted: {draw_preds} ({draw_preds/len(test_preds):.1%}) vs Actual: {draw_actual} ({draw_actual/len(test_preds):.1%})")
    
    # Distribution
    cm = confusion_matrix(y_test, test_preds)
    logger.info("\nConfusion Matrix (Pred vs Actual):")
    logger.info(f"         Away   Draw   Home")
    logger.info(f"Actual Away: {cm[0]}")
    logger.info(f"Actual Draw: {cm[1]}")
    logger.info(f"Actual Home: {cm[2]}")
    
    # Feature Importance
    importance = final_model.get_feature_importance().head(10)
    logger.info("\nTop 10 Features:")
    for idx, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(str(output_path))
    logger.info(f"\nModel saved to {output_path}")

if __name__ == '__main__':
    main()
