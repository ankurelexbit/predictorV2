"""
Dixon-Coles Model

Poisson-based attack/defense strength model with time decay and Dixon-Coles correction.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DixonColesModel:
    """
    Dixon-Coles model for football score prediction.
    
    Models home and away goals as:
        home_goals ~ Poisson(attack_home * defense_away * home_advantage)
        away_goals ~ Poisson(attack_away * defense_home)
    
    With Dixon-Coles correction for low-scoring games.
    """
    
    def __init__(
        self,
        time_decay: float = 0.0018,  # Half-life ~385 days
        max_goals: int = 10
    ):
        """
        Initialize Dixon-Coles model.
        
        Args:
            time_decay: Time decay factor for weighting recent matches
            max_goals: Maximum goals to consider in score matrix
        """
        self.time_decay = time_decay
        self.max_goals = max_goals
        
        # Model parameters (fitted)
        self.attack = {}
        self.defense = {}
        self.home_adv = None
        self.rho = None
        
        # Team index mapping
        self.team_to_idx = {}
        self.idx_to_team = {}
        
        # Calibration
        self.calibrators = {}
        self.is_calibrated = False
        
        # Fitted flag
        self.is_fitted = False
    
    def _get_time_weights(
        self,
        dates: pd.Series,
        reference_date: pd.Timestamp = None
    ) -> np.ndarray:
        """
        Calculate time-decay weights.
        
        More recent matches get higher weights.
        """
        if reference_date is None:
            reference_date = dates.max()
        
        days_ago = (reference_date - dates).dt.days
        weights = np.exp(-self.time_decay * days_ago)
        
        return weights
    
    def _tau(
        self,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
        rho: float
    ) -> float:
        """
        Dixon-Coles tau correction factor.
        
        Adjusts for observed correlation in low-scoring games.
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - lambda_home * lambda_away * rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + lambda_home * rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + lambda_away * rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0
    
    def _log_likelihood(
        self,
        params: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Compute negative log-likelihood for optimization (VECTORIZED).
        
        params layout: [attack_0, ..., attack_n, defense_0, ..., defense_n, home_adv, rho]
        """
        n_teams = len(self.team_to_idx)
        
        # Extract parameters
        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1]
        
        # Calculate expected goals (vectorized)
        lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        lambda_away = np.exp(attack[away_idx] + defense[home_idx])
        
        # Clip to prevent numerical issues
        lambda_home = np.clip(lambda_home, 0.001, 10)
        lambda_away = np.clip(lambda_away, 0.001, 10)
        
        # Vectorized Poisson log-probability
        hg = home_goals.astype(int)
        ag = away_goals.astype(int)
        
        log_prob_home = hg * np.log(lambda_home) - lambda_home - gammaln(hg + 1)
        log_prob_away = ag * np.log(lambda_away) - lambda_away - gammaln(ag + 1)
        
        # Vectorized tau correction
        tau = np.ones(len(hg))
        
        # 0-0
        mask_00 = (hg == 0) & (ag == 0)
        tau[mask_00] = 1 - lambda_home[mask_00] * lambda_away[mask_00] * rho
        
        # 0-1
        mask_01 = (hg == 0) & (ag == 1)
        tau[mask_01] = 1 + lambda_home[mask_01] * rho
        
        # 1-0
        mask_10 = (hg == 1) & (ag == 0)
        tau[mask_10] = 1 + lambda_away[mask_10] * rho
        
        # 1-1
        mask_11 = (hg == 1) & (ag == 1)
        tau[mask_11] = 1 - rho
        
        # Ensure tau is positive
        tau = np.clip(tau, 1e-10, None)
        
        # Total log-likelihood (vectorized sum)
        log_lik = np.sum(weights * (log_prob_home + log_prob_away + np.log(tau)))
        
        # Add regularization
        reg_attack = 0.1 * np.sum(attack) ** 2
        reg_defense = 0.1 * np.sum(defense) ** 2
        
        # Return negative (for minimization)
        return -log_lik + reg_attack + reg_defense
    
    def _constraint_attack(self, params: np.ndarray) -> float:
        """Constraint: sum of attack parameters = 0."""
        n_teams = len(self.team_to_idx)
        return np.sum(params[:n_teams])
    
    def _constraint_defense(self, params: np.ndarray) -> float:
        """Constraint: sum of defense parameters = 0."""
        n_teams = len(self.team_to_idx)
        return np.sum(params[n_teams:2*n_teams])
    
    def fit(
        self,
        df: pd.DataFrame,
        reference_date: pd.Timestamp = None,
        verbose: bool = True
    ):
        """
        Fit Dixon-Coles model to historical data.
        
        Args:
            df: DataFrame with columns [match_date, home_team_id, away_team_id, home_score, away_score]
            reference_date: Date for time decay calculation
            verbose: Print progress
        """
        logger.info(f"Fitting Dixon-Coles model on {len(df)} matches")

        # Build team index
        teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())
        self.team_to_idx = {team: i for i, team in enumerate(sorted(teams))}
        self.idx_to_team = {i: team for team, i in self.team_to_idx.items()}
        n_teams = len(teams)

        logger.info(f"Number of teams: {n_teams}")

        # Prepare data
        df = df.copy()
        df['match_date'] = pd.to_datetime(df['match_date'])

        home_idx = df['home_team_id'].map(self.team_to_idx).values
        away_idx = df['away_team_id'].map(self.team_to_idx).values
        home_goals = df['home_score'].values
        away_goals = df['away_score'].values
        
        # Time weights
        weights = self._get_time_weights(df['match_date'], reference_date)
        
        # Initial parameters
        x0 = np.zeros(2 * n_teams + 2)
        x0[2*n_teams] = 0.15  # home advantage
        x0[2*n_teams + 1] = -0.1  # rho
        
        # Bounds
        bounds = [(None, None)] * (2 * n_teams)
        bounds.append((0.05, 0.30))  # home_adv
        bounds.append((-0.5, 0.5))  # rho
        
        # Optimize
        if verbose:
            logger.info("Optimizing parameters with L-BFGS-B...")

        result = minimize(
            self._log_likelihood,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals, weights),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 1000,
                'maxfun': 20000,
                'ftol': 1e-6,
                'gtol': 1e-5,
                'disp': verbose
            }
        )

        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")
        
        # Extract parameters
        params = result.x
        self.attack = {self.idx_to_team[i]: params[i] for i in range(n_teams)}
        self.defense = {self.idx_to_team[i]: params[n_teams + i] for i in range(n_teams)}
        self.home_adv = params[2*n_teams]
        self.rho = params[2*n_teams + 1]
        
        self.is_fitted = True
        
        logger.info(f"Home advantage: {self.home_adv:.3f}")
        logger.info(f"Rho correction: {self.rho:.3f}")
        logger.info(f"Final log-likelihood: {-result.fun:.2f}")
        
        return self
    
    def predict_score_probs(
        self,
        home_team_id: int,
        away_team_id: int
    ) -> np.ndarray:
        """
        Predict probability matrix for all scorelines.

        Returns:
            Matrix of shape (max_goals+1, max_goals+1) where [i,j] is P(home=i, away=j)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Get team strengths (default to 0 for unknown teams)
        att_home = self.attack.get(home_team_id, 0)
        def_home = self.defense.get(home_team_id, 0)
        att_away = self.attack.get(away_team_id, 0)
        def_away = self.defense.get(away_team_id, 0)
        
        # Expected goals
        lambda_home = np.exp(att_home + def_away + self.home_adv)
        lambda_away = np.exp(att_away + def_home)
        
        # Score matrix
        probs = np.zeros((self.max_goals+1, self.max_goals+1))
        
        for i in range(self.max_goals+1):
            for j in range(self.max_goals+1):
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                
                # Apply Dixon-Coles correction
                tau = self._tau(i, j, lambda_home, lambda_away, self.rho)
                probs[i, j] = prob * tau
        
        # Normalize
        probs = probs / probs.sum()
        
        return probs
    
    def predict_1x2(
        self,
        home_team_id: int,
        away_team_id: int
    ) -> Tuple[float, float, float]:
        """
        Predict 1X2 probabilities.

        Returns:
            (p_home, p_draw, p_away)
        """
        score_probs = self.predict_score_probs(home_team_id, away_team_id)
        
        # Home win: i > j
        p_home = np.sum(np.tril(score_probs, -1))
        
        # Draw: i == j
        p_draw = np.trace(score_probs)
        
        # Away win: i < j
        p_away = np.sum(np.triu(score_probs, 1))
        
        return p_home, p_draw, p_away
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict 1X2 probabilities for all matches in DataFrame.

        Args:
            df: DataFrame with home_team_id, away_team_id columns

        Returns:
            Array of shape (n_matches, 3) with [away, draw, home] probabilities
        """
        probs = []
        
        for _, row in df.iterrows():
            p_home, p_draw, p_away = self.predict_1x2(
                row['home_team_id'],
                row['away_team_id']
            )
            probs.append([p_away, p_draw, p_home])
        
        probs = np.array(probs)
        
        # Apply calibration if available
        if self.is_calibrated:
            probs = self._calibrate(probs)
        
        return probs
    
    def calibrate(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ):
        """Fit calibration on validation data."""
        raw_probs = self.predict_proba(df)
        
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
    
    def _calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        calibrated = np.zeros_like(probs)
        
        for class_idx in range(3):
            class_probs = probs[:, class_idx]
            calibrator = self.calibrators[class_idx]
            
            if isinstance(calibrator, IsotonicRegression):
                calibrated[:, class_idx] = calibrator.predict(class_probs)
            else:
                calibrated[:, class_idx] = calibrator.predict_proba(
                    class_probs.reshape(-1, 1)
                )[:, 1]
        
        # Renormalize
        calibrated = np.clip(calibrated, 0.001, 0.999)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return calibrated
    
    def get_team_strengths(self) -> pd.DataFrame:
        """Get team attack/defense strengths as DataFrame."""
        if not self.is_fitted:
            return pd.DataFrame()
        
        strengths = []
        for team_id in sorted(self.attack.keys()):
            strengths.append({
                'team_id': team_id,
                'attack': self.attack[team_id],
                'defense': self.defense[team_id]
            })
        
        return pd.DataFrame(strengths).sort_values('attack', ascending=False)
    
    def save(self, path: Path):
        """Save model to file."""
        joblib.dump({
            'time_decay': self.time_decay,
            'max_goals': self.max_goals,
            'attack': self.attack,
            'defense': self.defense,
            'home_adv': self.home_adv,
            'rho': self.rho,
            'team_to_idx': self.team_to_idx,
            'idx_to_team': self.idx_to_team,
            'calibrators': self.calibrators,
            'is_calibrated': self.is_calibrated,
            'is_fitted': self.is_fitted
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        data = joblib.load(path)
        self.time_decay = data['time_decay']
        self.max_goals = data['max_goals']
        self.attack = data['attack']
        self.defense = data['defense']
        self.home_adv = data['home_adv']
        self.rho = data['rho']
        self.team_to_idx = data['team_to_idx']
        self.idx_to_team = data['idx_to_team']
        self.calibrators = data['calibrators']
        self.is_calibrated = data['is_calibrated']
        self.is_fitted = data['is_fitted']
        logger.info(f"Model loaded from {path}")
