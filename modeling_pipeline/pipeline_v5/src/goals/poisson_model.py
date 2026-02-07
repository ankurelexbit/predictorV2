"""
Poisson Goals Model
====================

Predicts expected home/away goals using CatBoost + LightGBM regressors,
then derives advanced market probabilities (O/U, BTTS, Handicap, Correct Score)
from independent Poisson distributions.

One model → all markets.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# Maximum goals to consider in score matrix (0..MAX_GOALS)
MAX_GOALS = 10


class PoissonGoalsModel:
    """Poisson-based goal prediction model for advanced betting markets."""

    def __init__(self):
        self.home_cat = None
        self.home_lgb = None
        self.away_cat = None
        self.away_lgb = None
        self.feature_cols = None

    def train(self, X: pd.DataFrame, y_home: np.ndarray, y_away: np.ndarray,
              X_val: pd.DataFrame = None, y_home_val: np.ndarray = None, y_away_val: np.ndarray = None,
              cat_params: dict = None, lgb_params: dict = None):
        """Train CatBoost + LightGBM regressors for home and away goals.

        Args:
            X: Training features DataFrame.
            y_home: Home goals target.
            y_away: Away goals target.
            X_val: Optional validation features (for early stopping).
            y_home_val: Optional validation home goals.
            y_away_val: Optional validation away goals.
            cat_params: Optional CatBoost hyperparameters (overrides defaults).
            lgb_params: Optional LightGBM hyperparameters (overrides defaults).
        """
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor

        self.feature_cols = list(X.columns)
        X_clean = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        _cat_params = dict(
            iterations=500, depth=6, learning_rate=0.03,
            loss_function='Poisson', verbose=0,
        )
        if cat_params:
            _cat_params.update(cat_params)
        cat_params = _cat_params

        _lgb_params = dict(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            objective='poisson', verbose=-1, random_state=42,
        )
        if lgb_params:
            _lgb_params.update(lgb_params)
        lgb_params = _lgb_params

        # Prepare validation data if provided
        if X_val is not None:
            X_val_clean = X_val[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            cat_params['early_stopping_rounds'] = 50
            lgb_params['n_estimators'] = max(lgb_params.get('n_estimators', 500), 1000)

        logger.info("Training home goals regressors...")
        self.home_cat = CatBoostRegressor(**cat_params)
        if X_val is not None:
            self.home_cat.fit(X_clean, y_home, eval_set=(X_val_clean, y_home_val))
        else:
            self.home_cat.fit(X_clean, y_home)

        self.home_lgb = LGBMRegressor(**lgb_params)
        if X_val is not None:
            self.home_lgb.fit(X_clean, y_home, eval_set=[(X_val_clean, y_home_val)],
                              callbacks=[_lgb_early_stopping(50)])
        else:
            self.home_lgb.fit(X_clean, y_home)

        logger.info("Training away goals regressors...")
        self.away_cat = CatBoostRegressor(**cat_params)
        if X_val is not None:
            self.away_cat.fit(X_clean, y_away, eval_set=(X_val_clean, y_away_val))
        else:
            self.away_cat.fit(X_clean, y_away)

        self.away_lgb = LGBMRegressor(**lgb_params)
        if X_val is not None:
            self.away_lgb.fit(X_clean, y_away, eval_set=[(X_val_clean, y_away_val)],
                              callbacks=[_lgb_early_stopping(50)])
        else:
            self.away_lgb.fit(X_clean, y_away)

        # Log training stats
        home_pred = self._predict_raw(X_clean)
        logger.info(f"Training MAE — home: {np.mean(np.abs(home_pred[0] - y_home)):.3f}, "
                     f"away: {np.mean(np.abs(home_pred[1] - y_away)):.3f}")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict expected goals (lambda parameters).

        Args:
            X: Features DataFrame (single row or batch).

        Returns:
            (lambda_home, lambda_away) as numpy arrays.
        """
        X_clean = X[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        return self._predict_raw(X_clean)

    def _predict_raw(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Internal predict on already-cleaned data."""
        home_cat = np.maximum(self.home_cat.predict(X), 0.05)
        home_lgb = np.maximum(self.home_lgb.predict(X), 0.05)
        away_cat = np.maximum(self.away_cat.predict(X), 0.05)
        away_lgb = np.maximum(self.away_lgb.predict(X), 0.05)

        lambda_home = (home_cat + home_lgb) / 2
        lambda_away = (away_cat + away_lgb) / 2

        return lambda_home, lambda_away

    def predict_single(self, X: pd.DataFrame) -> Tuple[float, float]:
        """Predict for a single fixture. Returns scalar lambdas."""
        lh, la = self.predict(X)
        return float(lh[0]), float(la[0])

    @staticmethod
    def build_score_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
        """Build score probability matrix P(home=i, away=j).

        Assumes independent Poisson distributions for home and away goals.

        Returns:
            (MAX_GOALS+1) x (MAX_GOALS+1) matrix where entry [i][j] = P(home=i, away=j).
        """
        home_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_home)
        away_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_away)
        return np.outer(home_probs, away_probs)

    @staticmethod
    def derive_markets(lambda_home: float, lambda_away: float) -> Dict:
        """Derive all market probabilities from Poisson parameters.

        Args:
            lambda_home: Expected home goals.
            lambda_away: Expected away goals.

        Returns:
            Dict with probabilities for O/U, BTTS, handicap, and top scorelines.
        """
        matrix = PoissonGoalsModel.build_score_matrix(lambda_home, lambda_away)
        n = MAX_GOALS + 1

        # Over/Under
        total_probs = {}
        for i in range(n):
            for j in range(n):
                total = i + j
                total_probs[total] = total_probs.get(total, 0) + matrix[i][j]

        cumulative = 0.0
        under_probs = {}
        for total in sorted(total_probs.keys()):
            cumulative += total_probs[total]
            under_probs[total] = cumulative

        over_0_5 = 1 - under_probs.get(0, 0)
        over_1_5 = 1 - under_probs.get(1, 0)
        over_2_5 = 1 - under_probs.get(2, 0)
        over_3_5 = 1 - under_probs.get(3, 0)

        # BTTS: P(home>=1 AND away>=1) = 1 - P(home=0) - P(away=0) + P(both=0)
        p_home_zero = poisson.pmf(0, lambda_home)
        p_away_zero = poisson.pmf(0, lambda_away)
        btts = 1 - p_home_zero - p_away_zero + (p_home_zero * p_away_zero)

        # Asian Handicap (home perspective): P(home - away > line)
        handicaps = {}
        for line in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            prob = 0.0
            for i in range(n):
                for j in range(n):
                    if (i - j) > line:
                        prob += matrix[i][j]
            handicaps[line] = prob

        # Top scorelines
        scorelines = []
        for i in range(min(n, 6)):  # Cap at 5-x for practical purposes
            for j in range(min(n, 6)):
                scorelines.append((i, j, float(matrix[i][j])))
        scorelines.sort(key=lambda x: x[2], reverse=True)
        top_5 = [{'home': s[0], 'away': s[1], 'prob': round(s[2], 4)} for s in scorelines[:5]]

        return {
            'home_goals_lambda': round(float(lambda_home), 3),
            'away_goals_lambda': round(float(lambda_away), 3),
            'over_0_5_prob': round(float(over_0_5), 4),
            'over_1_5_prob': round(float(over_1_5), 4),
            'over_2_5_prob': round(float(over_2_5), 4),
            'over_3_5_prob': round(float(over_3_5), 4),
            'btts_prob': round(float(btts), 4),
            'handicap_minus_2_5_prob': round(float(handicaps[-2.5]), 4),
            'handicap_minus_1_5_prob': round(float(handicaps[-1.5]), 4),
            'handicap_minus_0_5_prob': round(float(handicaps[-0.5]), 4),
            'handicap_plus_0_5_prob': round(float(handicaps[0.5]), 4),
            'handicap_plus_1_5_prob': round(float(handicaps[1.5]), 4),
            'handicap_plus_2_5_prob': round(float(handicaps[2.5]), 4),
            'top_scorelines': top_5,
        }

    def save(self, path: str):
        """Save all 4 regressors + feature_cols to disk."""
        joblib.dump({
            'home_cat': self.home_cat,
            'home_lgb': self.home_lgb,
            'away_cat': self.away_cat,
            'away_lgb': self.away_lgb,
            'feature_cols': self.feature_cols,
        }, path)
        logger.info(f"Saved goals model to {path}")

    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.home_cat = data['home_cat']
        self.home_lgb = data['home_lgb']
        self.away_cat = data['away_cat']
        self.away_lgb = data['away_lgb']
        self.feature_cols = data['feature_cols']
        logger.info(f"Loaded goals model from {path} ({len(self.feature_cols)} features)")


def _lgb_early_stopping(stopping_rounds: int):
    """Create LightGBM early stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=stopping_rounds, verbose=False)
