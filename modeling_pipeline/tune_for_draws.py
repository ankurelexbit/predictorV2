#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning - Draw-Focused

Custom tuning with aggressive conservative parameters to predict draws.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED
from utils import setup_logger, set_random_seed
from sklearn.metrics import log_loss

logger = setup_logger("xgboost_draw_tuning")
set_random_seed(RANDOM_SEED)

# Import XGBoost model
import importlib.util
spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
XGBoostFootballModel = mod.XGBoostFootballModel

def main():
    print("=" * 80)
    print("XGBOOST HYPERPARAMETER TUNING - DRAW-FOCUSED")
    print("=" * 80)
    
    # Load features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to matches with results
    mask = df['target'].notna()
    df = df[mask].copy()
    
    # Time-based split
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    y_val = val_df['target'].values.astype(int)
    y_test = test_df['target'].values.astype(int)
    
    # Draw-focused parameter grid
    # Focus on VERY conservative parameters
    param_grid = {
        'max_depth': [3, 4, 5, 6],  # Shallower trees
        'learning_rate': [0.01, 0.03, 0.05],  # Slower learning
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [7, 10, 15, 20],  # VERY conservative (key for draws)
        'gamma': [1.0, 2.0, 3.0, 5.0],  # Strong regularization (key for draws)
        'reg_alpha': [0.5, 1.0, 2.0],
        'reg_lambda': [1.0, 2.0, 5.0],
    }
    
    print("\n" + "=" * 80)
    print("DRAW-FOCUSED HYPERPARAMETER SEARCH")
    print("=" * 80)
    print("Key parameters for draw prediction:")
    print(f"  min_child_weight: {param_grid['min_child_weight']} (higher = more draws)")
    print(f"  gamma: {param_grid['gamma']} (higher = more regularization)")
    print(f"\nRunning 30 trials...")
    
    best_score = float('inf')
    best_params = None
    best_draw_count = 0
    results = []
    
    n_trials = 30
    
    for trial in range(n_trials):
        # Random sample parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': 500,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
        }
        
        for param, values in param_grid.items():
            params[param] = np.random.choice(values)
        
        # Train and evaluate
        try:
            model = XGBoostFootballModel(params=params)
            model.fit(train_df, val_df, verbose=False)
            
            # Get predictions
            val_probs = model.predict_proba(val_df, calibrated=False)
            val_preds = np.argmax(val_probs, axis=1)
            
            # Calculate metrics
            score = log_loss(y_val, val_probs)
            
            # Count draw predictions
            draw_count = (val_preds == 1).sum()
            draw_pct = draw_count / len(val_preds) * 100
            
            results.append({
                'trial': trial,
                'params': params.copy(),
                'log_loss': score,
                'draw_predictions': draw_count,
                'draw_pct': draw_pct
            })
            
            # Log if we get draws OR better log loss
            if draw_count > 0 or score < best_score:
                logger.info(f"Trial {trial}: log_loss={score:.4f}, draws={draw_count} ({draw_pct:.1f}%)")
                logger.info(f"  min_child_weight={params['min_child_weight']}, gamma={params['gamma']}")
            
            # Update best based on draw predictions AND log loss
            # Prioritize models that predict draws
            if draw_count > best_draw_count or (draw_count == best_draw_count and score < best_score):
                best_score = score
                best_params = params.copy()
                best_draw_count = draw_count
                logger.info(f"  *** New best! Draws: {draw_count}, Log Loss: {score:.4f}")
        
        except Exception as e:
            logger.warning(f"Trial {trial} failed: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("BEST PARAMETERS FOUND")
    print("=" * 80)
    print(f"Best log loss: {best_score:.4f}")
    print(f"Draw predictions: {best_draw_count} ({best_draw_count/len(val_df)*100:.1f}%)")
    print(f"\nParameters:")
    for key, value in best_params.items():
        if key not in ['objective', 'num_class', 'random_state', 'n_jobs', 'n_estimators']:
            print(f"  {key}: {value}")
    
    # Train final model with best params
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    final_model = XGBoostFootballModel(params=best_params)
    final_model.fit(train_df, val_df, verbose=True)
    
    # Calibrate
    final_model.calibrate(val_df, y_val, method='isotonic')
    
    # Evaluate on test set
    test_probs = final_model.predict_proba(test_df, calibrated=True)
    test_preds = np.argmax(test_probs, axis=1)
    test_loss = log_loss(y_test, test_probs)
    test_acc = (test_preds == y_test).mean()
    test_draws = (test_preds == 1).sum()
    
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print(f"Log Loss:         {test_loss:.4f}")
    print(f"Accuracy:         {test_acc:.1%}")
    print(f"Draw Predictions: {test_draws} ({test_draws/len(test_df)*100:.1f}%)")
    
    # Prediction distribution
    print("\nPrediction Distribution:")
    for outcome, label in [(0, 'Away'), (1, 'Draw'), (2, 'Home')]:
        count = (test_preds == outcome).sum()
        actual_count = (y_test == outcome).sum()
        print(f"  {label}: {count} predicted vs {actual_count} actual")
    
    # Save model
    model_path = MODELS_DIR / "xgboost_model_draw_tuned.joblib"
    final_model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if test_draws > 0:
        print(f"✅ SUCCESS! Model is now predicting {test_draws} draws ({test_draws/len(test_df)*100:.1f}%)")
    else:
        print(f"⚠️  Model still not predicting draws")
        print(f"   Consider even more aggressive parameters or class weight adjustment")
    
    print(f"\nTest Performance:")
    print(f"  Log Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.1%}")
    
    return {
        'best_params': best_params,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_draws': test_draws
    }

if __name__ == "__main__":
    results = main()
