"""
Comprehensive Model Optimization and Comparison.

This script:
1. Trains baseline XGBoost
2. Trains LightGBM, CatBoost
3. Hyperparameter optimization with Optuna
4. Stacking ensemble
5. Generates detailed comparison report
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import argparse
import json
from datetime import datetime
import warnings

# ML libraries
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
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

# Features to remove based on data quality analysis
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


class ModelComparison:
    """Comprehensive model comparison and optimization."""

    def __init__(self, data_path: str, n_trials: int = 50):
        """Initialize comparison."""
        self.data_path = data_path
        self.n_trials = n_trials
        self.results = {}
        self.models = {}
        self.best_params = {}

        # Load and prepare data
        self._load_data()
        self._prepare_data()

    def _load_data(self):
        """Load training data."""
        logger.info(f"\n{'=' * 80}")
        logger.info("LOADING DATA")
        logger.info(f"{'=' * 80}\n")

        df = pd.read_csv(self.data_path)
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

        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
        logger.info(f"Total columns: {len(df.columns)}")

        # Target distribution
        target_dist = df['target'].value_counts(normalize=True).sort_index()
        logger.info(f"\nTarget distribution:")
        logger.info(f"  Away (0): {target_dist[0]:.1%}")
        logger.info(f"  Draw (1): {target_dist[1]:.1%}")
        logger.info(f"  Home (2): {target_dist[2]:.1%}")

        self.df = df

    def _prepare_data(self):
        """Prepare train/val/test splits."""
        logger.info(f"\n{'=' * 80}")
        logger.info("DATA PREPARATION")
        logger.info(f"{'=' * 80}\n")

        # Get feature columns
        feature_cols = [col for col in self.df.columns
                       if col not in METADATA_COLS]

        # Remove bad features
        features_to_remove = [f for f in BAD_FEATURES if f in feature_cols]
        feature_cols = [f for f in feature_cols if f not in features_to_remove]

        logger.info(f"Features before cleanup: {len(self.df.columns) - len(METADATA_COLS)}")
        logger.info(f"Bad features removed: {len(features_to_remove)}")
        logger.info(f"Final feature count: {len(feature_cols)}")

        # Extract features and target
        X = self.df[feature_cols].copy()
        y = self.df['target'].values

        # Handle missing values (fill with median)
        X = X.fillna(X.median())

        # Chronological split (70/15/15)
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        self.X_train = X.iloc[:train_end]
        self.y_train = y[:train_end]
        self.X_val = X.iloc[train_end:val_end]
        self.y_val = y[train_end:val_end]
        self.X_test = X.iloc[val_end:]
        self.y_test = y[val_end:]

        logger.info(f"\nTrain set: {len(self.X_train):,} samples ({len(self.X_train)/n:.1%})")
        logger.info(f"Val set:   {len(self.X_val):,} samples ({len(self.X_val)/n:.1%})")
        logger.info(f"Test set:  {len(self.X_test):,} samples ({len(self.X_test)/n:.1%})")

        self.feature_cols = feature_cols

    def evaluate_model(self, model, X, y, dataset_name="Dataset"):
        """Evaluate model and return metrics."""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)

        acc = accuracy_score(y, y_pred)
        logloss = log_loss(y, y_pred_proba)

        # Per-class accuracy
        cm = confusion_matrix(y, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        return {
            'accuracy': acc,
            'log_loss': logloss,
            'away_accuracy': per_class_acc[0],
            'draw_accuracy': per_class_acc[1],
            'home_accuracy': per_class_acc[2],
            'confusion_matrix': cm,
            'predictions': y_pred_proba
        }

    def train_baseline_xgboost(self):
        """Train baseline XGBoost model."""
        logger.info(f"\n{'=' * 80}")
        logger.info("1. BASELINE XGBOOST")
        logger.info(f"{'=' * 80}\n")

        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        logger.info("Training with default parameters...")
        model = xgb.XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Evaluate
        train_metrics = self.evaluate_model(model, self.X_train, self.y_train, "Train")
        val_metrics = self.evaluate_model(model, self.X_val, self.y_val, "Val")
        test_metrics = self.evaluate_model(model, self.X_test, self.y_test, "Test")

        logger.info(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, Log Loss: {train_metrics['log_loss']:.4f}")
        logger.info(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, Log Loss: {test_metrics['log_loss']:.4f}")

        self.results['baseline_xgboost'] = {
            'params': params,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        self.models['baseline_xgboost'] = model

    def train_baseline_lightgbm(self):
        """Train baseline LightGBM model."""
        logger.info(f"\n{'=' * 80}")
        logger.info("2. BASELINE LIGHTGBM")
        logger.info(f"{'=' * 80}\n")

        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        logger.info("Training with default parameters...")
        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Evaluate
        train_metrics = self.evaluate_model(model, self.X_train, self.y_train, "Train")
        val_metrics = self.evaluate_model(model, self.X_val, self.y_val, "Val")
        test_metrics = self.evaluate_model(model, self.X_test, self.y_test, "Test")

        logger.info(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, Log Loss: {train_metrics['log_loss']:.4f}")
        logger.info(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, Log Loss: {test_metrics['log_loss']:.4f}")

        self.results['baseline_lightgbm'] = {
            'params': params,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        self.models['baseline_lightgbm'] = model

    def train_baseline_catboost(self):
        """Train baseline CatBoost model."""
        logger.info(f"\n{'=' * 80}")
        logger.info("3. BASELINE CATBOOST")
        logger.info(f"{'=' * 80}\n")

        params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'loss_function': 'MultiClass',
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }

        logger.info("Training with default parameters...")
        model = cb.CatBoostClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Evaluate
        train_metrics = self.evaluate_model(model, self.X_train, self.y_train, "Train")
        val_metrics = self.evaluate_model(model, self.X_val, self.y_val, "Val")
        test_metrics = self.evaluate_model(model, self.X_test, self.y_test, "Test")

        logger.info(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, Log Loss: {train_metrics['log_loss']:.4f}")
        logger.info(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, Log Loss: {test_metrics['log_loss']:.4f}")

        self.results['baseline_catboost'] = {
            'params': params,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        self.models['baseline_catboost'] = model

    def optimize_xgboost(self):
        """Hyperparameter optimization for XGBoost."""
        logger.info(f"\n{'=' * 80}")
        logger.info("4. XGBOOST HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'=' * 80}\n")

        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'n_jobs': -1
            }

            model = xgb.XGBClassifier(**params)
            model.fit(self.X_train, self.y_train)
            y_pred_proba = model.predict_proba(self.X_val)
            return log_loss(self.y_val, y_pred_proba)

        logger.info(f"Running Optuna optimization ({self.n_trials} trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"\nBest log loss: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        # Train final model with best params
        best_params = study.best_params.copy()
        best_params.update({'objective': 'multi:softprob', 'num_class': 3, 'random_state': 42, 'n_jobs': -1})

        model = xgb.XGBClassifier(**best_params)
        model.fit(self.X_train, self.y_train)

        # Evaluate
        train_metrics = self.evaluate_model(model, self.X_train, self.y_train, "Train")
        val_metrics = self.evaluate_model(model, self.X_val, self.y_val, "Val")
        test_metrics = self.evaluate_model(model, self.X_test, self.y_test, "Test")

        logger.info(f"\nFinal performance:")
        logger.info(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, Log Loss: {train_metrics['log_loss']:.4f}")
        logger.info(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, Log Loss: {test_metrics['log_loss']:.4f}")

        self.results['tuned_xgboost'] = {
            'params': best_params,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        self.models['tuned_xgboost'] = model
        self.best_params['xgboost'] = study.best_params

    def optimize_lightgbm(self):
        """Hyperparameter optimization for LightGBM."""
        logger.info(f"\n{'=' * 80}")
        logger.info("5. LIGHTGBM HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'=' * 80}\n")

        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(self.X_train, self.y_train)
            y_pred_proba = model.predict_proba(self.X_val)
            return log_loss(self.y_val, y_pred_proba)

        logger.info(f"Running Optuna optimization ({self.n_trials} trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"\nBest log loss: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        # Train final model with best params
        best_params = study.best_params.copy()
        best_params.update({'objective': 'multiclass', 'num_class': 3, 'random_state': 42, 'n_jobs': -1, 'verbose': -1})

        model = lgb.LGBMClassifier(**best_params)
        model.fit(self.X_train, self.y_train)

        # Evaluate
        train_metrics = self.evaluate_model(model, self.X_train, self.y_train, "Train")
        val_metrics = self.evaluate_model(model, self.X_val, self.y_val, "Val")
        test_metrics = self.evaluate_model(model, self.X_test, self.y_test, "Test")

        logger.info(f"\nFinal performance:")
        logger.info(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, Log Loss: {train_metrics['log_loss']:.4f}")
        logger.info(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, Log Loss: {test_metrics['log_loss']:.4f}")

        self.results['tuned_lightgbm'] = {
            'params': best_params,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        self.models['tuned_lightgbm'] = model
        self.best_params['lightgbm'] = study.best_params

    def optimize_catboost(self):
        """Hyperparameter optimization for CatBoost."""
        logger.info(f"\n{'=' * 80}")
        logger.info("6. CATBOOST HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'=' * 80}\n")

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'loss_function': 'MultiClass',
                'random_seed': 42,
                'verbose': False,
                'thread_count': -1
            }

            model = cb.CatBoostClassifier(**params)
            model.fit(self.X_train, self.y_train)
            y_pred_proba = model.predict_proba(self.X_val)
            return log_loss(self.y_val, y_pred_proba)

        logger.info(f"Running Optuna optimization ({self.n_trials} trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"\nBest log loss: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        # Train final model with best params
        best_params = study.best_params.copy()
        best_params.update({'loss_function': 'MultiClass', 'random_seed': 42, 'verbose': False, 'thread_count': -1})

        model = cb.CatBoostClassifier(**best_params)
        model.fit(self.X_train, self.y_train)

        # Evaluate
        train_metrics = self.evaluate_model(model, self.X_train, self.y_train, "Train")
        val_metrics = self.evaluate_model(model, self.X_val, self.y_val, "Val")
        test_metrics = self.evaluate_model(model, self.X_test, self.y_test, "Test")

        logger.info(f"\nFinal performance:")
        logger.info(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, Log Loss: {train_metrics['log_loss']:.4f}")
        logger.info(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, Log Loss: {test_metrics['log_loss']:.4f}")

        self.results['tuned_catboost'] = {
            'params': best_params,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        self.models['tuned_catboost'] = model
        self.best_params['catboost'] = study.best_params

    def train_stacking_ensemble(self):
        """Train stacking ensemble with best models."""
        logger.info(f"\n{'=' * 80}")
        logger.info("7. STACKING ENSEMBLE (XGBoost + LightGBM + CatBoost)")
        logger.info(f"{'=' * 80}\n")

        # Use tuned models if available, otherwise baseline
        xgb_model = self.models.get('tuned_xgboost', self.models.get('baseline_xgboost'))
        lgb_model = self.models.get('tuned_lightgbm', self.models.get('baseline_lightgbm'))
        cat_model = self.models.get('tuned_catboost', self.models.get('baseline_catboost'))

        logger.info("Building manual stacking ensemble...")
        # Get predictions from base models on validation set (for meta-features)
        xgb_val_pred = xgb_model.predict_proba(self.X_val)
        lgb_val_pred = lgb_model.predict_proba(self.X_val)
        cat_val_pred = cat_model.predict_proba(self.X_val)

        # Stack predictions for meta-learner
        meta_features_val = np.column_stack([xgb_val_pred, lgb_val_pred, cat_val_pred])

        # Train meta-learner (logistic regression)
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        meta_learner.fit(meta_features_val, self.y_val)

        # Create wrapper for predictions
        class StackingEnsemble:
            def __init__(self, base_models, meta_model):
                self.base_models = base_models
                self.meta_model = meta_model

            def predict_proba(self, X):
                # Get predictions from all base models
                preds = [model.predict_proba(X) for model in self.base_models]
                # Stack predictions
                meta_features = np.column_stack(preds)
                # Use meta-learner
                return self.meta_model.predict_proba(meta_features)

            def predict(self, X):
                proba = self.predict_proba(X)
                return np.argmax(proba, axis=1)

        stacking_model = StackingEnsemble([xgb_model, lgb_model, cat_model], meta_learner)

        logger.info("Stacking model trained.")

        # Evaluate
        train_metrics = self.evaluate_model(stacking_model, self.X_train, self.y_train, "Train")
        val_metrics = self.evaluate_model(stacking_model, self.X_val, self.y_val, "Val")
        test_metrics = self.evaluate_model(stacking_model, self.X_test, self.y_test, "Test")

        logger.info(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, Log Loss: {train_metrics['log_loss']:.4f}")
        logger.info(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, Log Loss: {test_metrics['log_loss']:.4f}")

        self.results['stacking_ensemble'] = {
            'params': {'base_models': ['tuned_xgb', 'tuned_lgb', 'tuned_cat']},
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        self.models['stacking_ensemble'] = stacking_model

    def generate_report(self, output_path: str = None):
        """Generate detailed comparison report."""
        logger.info(f"\n{'=' * 80}")
        logger.info("FINAL COMPARISON REPORT")
        logger.info(f"{'=' * 80}\n")

        # Create comparison DataFrame
        comparison = []
        for model_name, result in self.results.items():
            comparison.append({
                'Model': model_name,
                'Train_Accuracy': result['train']['accuracy'],
                'Train_LogLoss': result['train']['log_loss'],
                'Val_Accuracy': result['val']['accuracy'],
                'Val_LogLoss': result['val']['log_loss'],
                'Test_Accuracy': result['test']['accuracy'],
                'Test_LogLoss': result['test']['log_loss'],
                'Test_Away_Acc': result['test']['away_accuracy'],
                'Test_Draw_Acc': result['test']['draw_accuracy'],
                'Test_Home_Acc': result['test']['home_accuracy']
            })

        df_comparison = pd.DataFrame(comparison)

        # Sort by test log loss
        df_comparison = df_comparison.sort_values('Test_LogLoss')

        # Print table
        print("\n" + "=" * 120)
        print("MODEL COMPARISON TABLE")
        print("=" * 120)
        print(f"{'Model':<30} {'Test Acc':<12} {'Test LogLoss':<15} {'Val Acc':<12} {'Val LogLoss':<15}")
        print("-" * 120)

        for _, row in df_comparison.iterrows():
            print(f"{row['Model']:<30} {row['Test_Accuracy']:>10.4f}  {row['Test_LogLoss']:>13.4f}  "
                  f"{row['Val_Accuracy']:>10.4f}  {row['Val_LogLoss']:>13.4f}")

        print("=" * 120)

        # Best model
        best_model = df_comparison.iloc[0]
        logger.info(f"\n🏆 BEST MODEL: {best_model['Model']}")
        logger.info(f"   Test Accuracy: {best_model['Test_Accuracy']:.4f}")
        logger.info(f"   Test Log Loss: {best_model['Test_LogLoss']:.4f}")
        logger.info(f"   Per-class accuracy: Away={best_model['Test_Away_Acc']:.4f}, "
                   f"Draw={best_model['Test_Draw_Acc']:.4f}, Home={best_model['Test_Home_Acc']:.4f}")

        # Calculate improvements over baseline
        baseline = df_comparison[df_comparison['Model'] == 'baseline_xgboost'].iloc[0]
        best = df_comparison.iloc[0]

        acc_improvement = (best['Test_Accuracy'] - baseline['Test_Accuracy']) * 100
        logloss_improvement = baseline['Test_LogLoss'] - best['Test_LogLoss']

        logger.info(f"\n📈 IMPROVEMENT OVER BASELINE XGBOOST:")
        logger.info(f"   Accuracy: {acc_improvement:+.2f} percentage points")
        logger.info(f"   Log Loss: {logloss_improvement:+.4f} (lower is better)")

        # Save detailed report
        if output_path:
            report_data = {
                'comparison': df_comparison.to_dict('records'),
                'best_params': self.best_params,
                'summary': {
                    'best_model': best_model['Model'],
                    'test_accuracy': float(best_model['Test_Accuracy']),
                    'test_logloss': float(best_model['Test_LogLoss']),
                    'improvement_vs_baseline': {
                        'accuracy_pp': float(acc_improvement),
                        'logloss': float(logloss_improvement)
                    }
                }
            }

            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"\n💾 Detailed report saved to: {output_path}")

            # Save comparison CSV
            csv_path = output_path.replace('.json', '.csv')
            df_comparison.to_csv(csv_path, index=False)
            logger.info(f"💾 Comparison table saved to: {csv_path}")

        return df_comparison

    def save_best_model(self, output_path: str):
        """Save the best performing model."""
        # Find best model by test log loss
        best_name = min(self.results.items(), key=lambda x: x[1]['test']['log_loss'])[0]
        best_model = self.models[best_name]

        joblib.dump(best_model, output_path)
        logger.info(f"\n💾 Best model ({best_name}) saved to: {output_path}")

        return best_name


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Optimization')
    parser.add_argument('--data', type=str, default='data/training_data.csv',
                       help='Path to training data')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of Optuna trials for hyperparameter tuning')
    parser.add_argument('--output-report', type=str, default='results/model_comparison.json',
                       help='Path to save comparison report')
    parser.add_argument('--output-model', type=str, default='models/v4_optimized_model.joblib',
                       help='Path to save best model')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip hyperparameter tuning (faster)')

    args = parser.parse_args()

    # Create output directories
    Path('results').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)

    # Run comparison
    comparison = ModelComparison(args.data, n_trials=args.n_trials)

    # Train baseline models
    comparison.train_baseline_xgboost()
    comparison.train_baseline_lightgbm()
    comparison.train_baseline_catboost()

    # Hyperparameter optimization
    if not args.skip_tuning:
        comparison.optimize_xgboost()
        comparison.optimize_lightgbm()
        comparison.optimize_catboost()

    # Stacking ensemble
    comparison.train_stacking_ensemble()

    # Generate report
    comparison.generate_report(args.output_report)

    # Save best model
    best_name = comparison.save_best_model(args.output_model)

    logger.info(f"\n{'=' * 80}")
    logger.info("✅ OPTIMIZATION COMPLETE!")
    logger.info(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
