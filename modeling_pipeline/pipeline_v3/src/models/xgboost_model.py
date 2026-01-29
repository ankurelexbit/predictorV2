"""
XGBoost Football Model

Gradient boosting model using all 150+ features with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class XGBoostFootballModel:
    """
    XGBoost model for 1X2 football prediction.
    
    Features:
    - Multiclass classification (softmax)
    - Isotonic calibration
    - Feature importance analysis
    """
    
    def __init__(
        self,
        params: Dict = None,
        feature_columns: List[str] = None
    ):
        """
        Initialize XGBoost model.
        
        Args:
            params: XGBoost parameters
            feature_columns: List of feature column names to use
        """
        # Default parameters
        self.params = params or {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'min_child_weight': 7,
            'gamma': 1.0,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        
        self.feature_columns = feature_columns
        
        # Model components
        self.model = None
        self.scaler = RobustScaler()
        self.calibrators = {}
        
        # State
        self.is_fitted = False
        self.is_calibrated = False
        self.feature_importance_ = None
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Prepare feature matrix.
        
        Args:
            df: DataFrame with features
            fit_scaler: Whether to fit the scaler
        
        Returns:
            Feature matrix
        """
        # Auto-detect features if not specified
        if self.feature_columns is None:
            exclude_cols = ['fixture_id', 'match_date', 'home_team_id', 'away_team_id', 
                          'home_score', 'away_score', 'season_id', 'league_id', 
                          'starting_at', 'target']
            self.feature_columns = [c for c in df.columns if c not in exclude_cols]
            logger.info(f"Auto-detected {len(self.feature_columns)} features")
        
        # Select available features
        available_features = [f for f in self.feature_columns if f in df.columns]
        
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")
        
        X = df[available_features].copy()
        
        # Keep only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < len(available_features):
            non_numeric = set(available_features) - set(numeric_cols)
            logger.warning(f"Dropping {len(non_numeric)} non-numeric features: {list(non_numeric)[:5]}...")
            X = X[numeric_cols]
            self.feature_columns = numeric_cols
        
        # Handle missing values (fill with median)
        for col in X.columns:
            if X[col].isna().any():
                if fit_scaler:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                else:
                    X[col] = X[col].fillna(X[col].median())
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        Fit XGBoost model.
        
        Args:
            train_df: Training data with features and target
            val_df: Validation data for early stopping
            early_stopping_rounds: Early stopping patience
            verbose: Print progress
        """
        logger.info(f"Fitting XGBoost on {len(train_df)} training samples")
        
        # Prepare data
        X_train = self._prepare_features(train_df, fit_scaler=True)
        y_train = train_df['target'].values.astype(int)
        
        # Compute sample weights for class imbalance
        from collections import Counter
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        
        class_weights = {
            cls: total_samples / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }
        
        # Boost underrepresented classes
        if 1 in class_weights:  # Draw
            class_weights[1] *= 1.5
        if 0 in class_weights:  # Away
            class_weights[0] *= 1.3
        
        sample_weights = np.array([class_weights[cls] for cls in y_train])
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Class weights: {class_weights}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, 
                            feature_names=self.feature_columns)
        
        # Validation set
        evals = [(dtrain, 'train')]
        if val_df is not None:
            X_val = self._prepare_features(val_df, fit_scaler=False)
            y_val = val_df['target'].values.astype(int)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_columns)
            evals.append((dval, 'val'))
        
        # Train
        params = self.params.copy()
        n_estimators = params.pop('n_estimators', 500)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if val_df is not None else None,
            verbose_eval=50 if verbose else False
        )
        
        # Store feature importance
        self.feature_importance_ = self.model.get_score(importance_type='gain')
        
        self.is_fitted = True
        logger.info(f"Model fitted with {self.model.best_iteration} rounds")
        
        return self
    
    def predict_proba_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Get raw (uncalibrated) probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = self._prepare_features(df, fit_scaler=False)
        dtest = xgb.DMatrix(X, feature_names=self.feature_columns)
        
        probs = self.model.predict(dtest)
        
        return probs
    
    def calibrate(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ):
        """
        Fit calibration on validation data.
        
        Args:
            df: Validation features
            y_true: True labels
            method: 'isotonic' or 'sigmoid'
        """
        raw_probs = self.predict_proba_raw(df)
        
        for class_idx in range(3):
            class_probs = raw_probs[:, class_idx]
            class_labels = (y_true == class_idx).astype(int)
            
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
            
            calibrator.fit(class_probs.reshape(-1, 1), class_labels)
            self.calibrators[class_idx] = calibrator
        
        self.is_calibrated = True
        logger.info(f"Calibration fitted using {method} method")
    
    def predict_proba(
        self,
        df: pd.DataFrame,
        calibrated: bool = True
    ) -> np.ndarray:
        """
        Get probabilities.
        
        Args:
            df: Features DataFrame
            calibrated: Whether to apply calibration
        
        Returns:
            Array of shape (n_samples, 3)
        """
        probs = self.predict_proba_raw(df)
        
        if calibrated and self.is_calibrated:
            calibrated_probs = np.zeros_like(probs)
            
            for class_idx in range(3):
                class_probs = probs[:, class_idx]
                calibrator = self.calibrators[class_idx]
                
                if isinstance(calibrator, IsotonicRegression):
                    calibrated_probs[:, class_idx] = calibrator.predict(class_probs)
                else:
                    calibrated_probs[:, class_idx] = calibrator.predict_proba(
                        class_probs.reshape(-1, 1)
                    )[:, 1]
            
            # Renormalize
            calibrated_probs = np.clip(calibrated_probs, 0.001, 0.999)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            
            return calibrated_probs
        
        return probs
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        
        importance = []
        for feat in self.feature_columns:
            importance.append({
                'feature': feat,
                'importance': self.feature_importance_.get(feat, 0)
            })
        
        df = pd.DataFrame(importance)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def save(self, path: Path):
        """Save model to file."""
        # Save XGBoost model separately
        model_path = path.with_suffix('.xgb')
        self.model.save_model(str(model_path))
        
        # Save other components
        joblib.dump({
            'params': self.params,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'calibrators': self.calibrators,
            'is_fitted': self.is_fitted,
            'is_calibrated': self.is_calibrated,
            'feature_importance': self.feature_importance_,
            'model_path': str(model_path)
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        data = joblib.load(path)
        
        self.params = data['params']
        self.feature_columns = data['feature_columns']
        self.scaler = data['scaler']
        self.calibrators = data['calibrators']
        self.is_fitted = data['is_fitted']
        self.is_calibrated = data['is_calibrated']
        self.feature_importance_ = data['feature_importance']
        
        # Load XGBoost model
        model_path = data['model_path']
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        logger.info(f"Model loaded from {path}")


def tune_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 20
) -> Dict:
    """
    Hyperparameter tuning via random search.
    
    Args:
        train_df: Training data
        val_df: Validation data
        n_trials: Number of random configurations to try
    
    Returns:
        Best parameters and results
    """
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0, 0.1, 0.5, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    }
    
    y_val = val_df['target'].values.astype(int)
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for trial in range(n_trials):
        # Random sample parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': 500,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        for param, values in param_grid.items():
            params[param] = np.random.choice(values)
        
        # Train and evaluate
        model = XGBoostFootballModel(params=params)
        model.fit(train_df, val_df, verbose=False)
        
        val_probs = model.predict_proba(val_df, calibrated=False)
        score = log_loss(y_val, val_probs)
        
        results.append({
            'trial': trial,
            'params': params.copy(),
            'log_loss': score
        })
        
        if score < best_score:
            best_score = score
            best_params = params.copy()
            logger.info(f"Trial {trial}: New best log_loss = {score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }
