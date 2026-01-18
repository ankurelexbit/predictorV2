"""
05 - Dixon-Coles Model
======================

This notebook implements the Dixon-Coles model - the gold standard
for football score prediction.

The model:
1. Estimates attack/defense strength for each team
2. Models goals as Poisson distributed
3. Applies correction for low-scoring draws (the "rho" parameter)
4. Uses time-decay to weight recent matches more heavily

Paper: "Modelling Association Football Scores and Inefficiencies in the
       Football Betting Market" - Dixon & Coles (1997)

Usage:
    python 05_model_dixon_coles.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import matplotlib.pyplot as plt
import warnings
import joblib

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    TEST_SEASONS,
    RANDOM_SEED,
)
from utils import (
    setup_logger,
    set_random_seed,
    calculate_log_loss,
    calculate_brier_score,
    calculate_calibration_error,
    print_metrics_table,
    season_based_split,
)

# Setup
logger = setup_logger("dixon_coles")
set_random_seed(RANDOM_SEED)


# =============================================================================
# DIXON-COLES MODEL
# =============================================================================

class DixonColesModel:
    """
    Dixon-Coles model for football score prediction.
    
    Models home and away goals as:
        home_goals ~ Poisson(attack_home * defense_away * home_advantage)
        away_goals ~ Poisson(attack_away * defense_home)
    
    With a correction factor (tau) for scores 0-0, 1-0, 0-1, 1-1.
    
    Parameters:
        attack_i: Attack strength of team i
        defense_i: Defense strength of team i (lower = better)
        home_adv: Home advantage factor
        rho: Correction parameter for low-scoring games
    """
    
    def __init__(
        self,
        time_decay: float = 0.0018,  # Half-life ~385 days
        max_goals: int = 10
    ):
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
        # log(P(k; λ)) = k*log(λ) - λ - log(k!)
        hg = home_goals.astype(int)
        ag = away_goals.astype(int)
        
        from scipy.special import gammaln
        log_prob_home = hg * np.log(lambda_home) - lambda_home - gammaln(hg + 1)
        log_prob_away = ag * np.log(lambda_away) - lambda_away - gammaln(ag + 1)
        
        # Vectorized tau correction
        tau = np.ones(len(hg))
        
        # 0-0: tau = 1 - lh * la * rho
        mask_00 = (hg == 0) & (ag == 0)
        tau[mask_00] = 1 - lambda_home[mask_00] * lambda_away[mask_00] * rho
        
        # 0-1: tau = 1 + lh * rho
        mask_01 = (hg == 0) & (ag == 1)
        tau[mask_01] = 1 + lambda_home[mask_01] * rho
        
        # 1-0: tau = 1 + la * rho
        mask_10 = (hg == 1) & (ag == 0)
        tau[mask_10] = 1 + lambda_away[mask_10] * rho
        
        # 1-1: tau = 1 - rho
        mask_11 = (hg == 1) & (ag == 1)
        tau[mask_11] = 1 - rho
        
        # Ensure tau is positive
        tau = np.clip(tau, 1e-10, None)
        
        # Total log-likelihood (vectorized sum)
        log_lik = np.sum(weights * (log_prob_home + log_prob_away + np.log(tau)))
        
        # Add regularization to enforce sum-to-zero constraints (soft constraint)
        reg_attack = 0.1 * np.sum(attack) ** 2
        reg_defense = 0.1 * np.sum(defense) ** 2
        
        # Return negative (for minimization) + regularization
        return -log_lik + reg_attack + reg_defense
    
    def _constraint_attack(self, params: np.ndarray) -> float:
        """Constraint: sum of attack parameters = 0 (identifiability)."""
        n_teams = len(self.team_to_idx)
        return np.sum(params[:n_teams])
    
    def _constraint_defense(self, params: np.ndarray) -> float:
        """Constraint: sum of defense parameters = 0 (identifiability)."""
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
            df: DataFrame with columns [date, home_team, away_team, home_goals, away_goals]
            reference_date: Date for time decay calculation
            verbose: Print progress
        """
        logger.info(f"Fitting Dixon-Coles model on {len(df)} matches")
        
        # Build team index
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        self.team_to_idx = {team: i for i, team in enumerate(sorted(teams))}
        self.idx_to_team = {i: team for team, i in self.team_to_idx.items()}
        n_teams = len(teams)
        
        logger.info(f"Number of teams: {n_teams}")
        
        # Prepare data
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        home_idx = df['home_team'].map(self.team_to_idx).values
        away_idx = df['away_team'].map(self.team_to_idx).values
        home_goals = df['home_goals'].values
        away_goals = df['away_goals'].values
        
        # Time weights
        weights = self._get_time_weights(df['date'], reference_date)
        
        # Initial parameters
        # attack/defense ~ 0, home_adv ~ 0.25, rho ~ -0.1
        x0 = np.zeros(2 * n_teams + 2)
        x0[2*n_teams] = 0.25  # home advantage
        x0[2*n_teams + 1] = -0.1  # rho
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': self._constraint_attack},
            {'type': 'eq', 'fun': self._constraint_defense},
        ]
        
        # Bounds for rho: typically between -0.3 and 0
        bounds = [(None, None)] * (2 * n_teams)  # attack/defense unbounded
        bounds.append((0, 1))  # home_adv positive
        bounds.append((-0.5, 0.5))  # rho bounded
        
        # Optimize using L-BFGS-B (much faster than SLSQP)
        # Note: We use regularization instead of equality constraints for speed
        if verbose:
            logger.info("Optimizing parameters with L-BFGS-B...")
        
        result = minimize(
            self._log_likelihood,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals, weights),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'disp': verbose, 'maxfun': 5000}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
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
        home_team: str,
        away_team: str
    ) -> np.ndarray:
        """
        Predict probability matrix for all scorelines.
        
        Returns:
            Matrix of shape (max_goals+1, max_goals+1) where [i,j] is P(home=i, away=j)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Handle unseen teams
        attack_home = self.attack.get(home_team, 0.0)
        defense_home = self.defense.get(home_team, 0.0)
        attack_away = self.attack.get(away_team, 0.0)
        defense_away = self.defense.get(away_team, 0.0)
        
        # Expected goals
        lambda_home = np.exp(attack_home + defense_away + self.home_adv)
        lambda_away = np.exp(attack_away + defense_home)
        
        # Clip
        lambda_home = np.clip(lambda_home, 0.1, 8)
        lambda_away = np.clip(lambda_away, 0.1, 8)
        
        # Score probability matrix
        probs = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                prob_home = poisson.pmf(i, lambda_home)
                prob_away = poisson.pmf(j, lambda_away)
                tau = self._tau(i, j, lambda_home, lambda_away, self.rho)
                probs[i, j] = prob_home * prob_away * tau
        
        # Normalize
        probs = probs / probs.sum()
        
        return probs
    
    def predict_1x2(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float, float]:
        """
        Predict 1X2 probabilities.
        
        Returns:
            (p_home, p_draw, p_away)
        """
        score_probs = self.predict_score_probs(home_team, away_team)
        
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                if i > j:
                    p_home += score_probs[i, j]
                elif i == j:
                    p_draw += score_probs[i, j]
                else:
                    p_away += score_probs[i, j]
        
        # Normalize to ensure sum = 1
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict 1X2 probabilities for all matches in DataFrame.
        
        Args:
            df: DataFrame with home_team, away_team columns
        
        Returns:
            Array of shape (n_matches, 3)
        """
        probs = []
        
        for _, row in df.iterrows():
            p_h, p_d, p_a = self.predict_1x2(row['home_team'], row['away_team'])
            probs.append([p_h, p_d, p_a])
        
        return np.array(probs)
    
    def get_team_strengths(self) -> pd.DataFrame:
        """Get team attack/defense strengths as DataFrame."""
        if not self.is_fitted:
            return pd.DataFrame()
        
        data = []
        for team in self.attack:
            data.append({
                'team': team,
                'attack': self.attack[team],
                'defense': self.defense[team],
                'overall': self.attack[team] - self.defense[team]  # Higher = better
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('overall', ascending=False).reset_index(drop=True)
    
    def save(self, path: Path):
        """Save model to file."""
        joblib.dump({
            'attack': self.attack,
            'defense': self.defense,
            'home_adv': self.home_adv,
            'rho': self.rho,
            'team_to_idx': self.team_to_idx,
            'idx_to_team': self.idx_to_team,
            'time_decay': self.time_decay,
            'max_goals': self.max_goals,
            'is_fitted': self.is_fitted
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        data = joblib.load(path)
        self.attack = data['attack']
        self.defense = data['defense']
        self.home_adv = data['home_adv']
        self.rho = data['rho']
        self.team_to_idx = data['team_to_idx']
        self.idx_to_team = data['idx_to_team']
        self.time_decay = data['time_decay']
        self.max_goals = data['max_goals']
        self.is_fitted = data['is_fitted']
        logger.info(f"Model loaded from {path}")


# =============================================================================
# CALIBRATION
# =============================================================================

class CalibratedDixonColes:
    """Dixon-Coles with isotonic calibration."""
    
    def __init__(self, base_model: DixonColesModel):
        self.base_model = base_model
        self.calibrators = {}
        self.is_calibrated = False
    
    def calibrate(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ):
        """Fit calibration on validation data."""
        from sklearn.isotonic import IsotonicRegression
        
        raw_probs = self.base_model.predict_proba(df)
        
        for class_idx in range(3):
            class_probs = raw_probs[:, class_idx]
            class_labels = (y_true == class_idx).astype(int)
            
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(class_probs, class_labels)
            self.calibrators[class_idx] = calibrator
        
        self.is_calibrated = True
        logger.info("Calibration fitted")
    
    def predict_proba(
        self,
        df: pd.DataFrame,
        calibrated: bool = True
    ) -> np.ndarray:
        """Get calibrated probabilities."""
        probs = self.base_model.predict_proba(df)
        
        if calibrated and self.is_calibrated:
            calibrated_probs = np.zeros_like(probs)
            for class_idx in range(3):
                calibrated_probs[:, class_idx] = self.calibrators[class_idx].predict(
                    probs[:, class_idx]
                )
            
            # Renormalize
            calibrated_probs = np.clip(calibrated_probs, 0.001, 0.999)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            
            return calibrated_probs
        
        return probs
    
    def save(self, path: Path):
        """Save calibrated model."""
        joblib.dump({
            'base_model_data': {
                'attack': self.base_model.attack,
                'defense': self.base_model.defense,
                'home_adv': self.base_model.home_adv,
                'rho': self.base_model.rho,
                'team_to_idx': self.base_model.team_to_idx,
                'idx_to_team': self.base_model.idx_to_team,
                'time_decay': self.base_model.time_decay,
                'max_goals': self.base_model.max_goals,
                'is_fitted': self.base_model.is_fitted
            },
            'calibrators': self.calibrators,
            'is_calibrated': self.is_calibrated
        }, path)
        logger.info(f"Calibrated model saved to {path}")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model"
) -> Dict[str, float]:
    """Evaluate prediction quality."""
    metrics = {
        'log_loss': calculate_log_loss(y_true, y_pred),
        'brier_score': calculate_brier_score(y_true, y_pred),
    }
    
    cal_error = calculate_calibration_error(y_true, y_pred)
    metrics['calibration_error'] = cal_error['overall_ece']
    
    predicted_class = np.argmax(y_pred, axis=1)
    metrics['accuracy'] = np.mean(predicted_class == y_true)
    
    print_metrics_table(metrics, f"{name} Evaluation")
    
    return metrics


def plot_score_heatmap(model: DixonColesModel, home_team: str, away_team: str):
    """Plot heatmap of score probabilities."""
    probs = model.predict_score_probs(home_team, away_team)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(probs[:6, :6], cmap='YlOrRd')
    
    # Labels
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(range(6))
    ax.set_yticklabels(range(6))
    ax.set_xlabel(f'{away_team} Goals')
    ax.set_ylabel(f'{home_team} Goals')
    ax.set_title(f'Score Probability: {home_team} vs {away_team}')
    
    # Add probability values
    for i in range(6):
        for j in range(6):
            prob = probs[i, j]
            ax.text(j, i, f'{prob:.1%}', ha='center', va='center',
                   color='white' if prob > 0.1 else 'black', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Probability')
    plt.tight_layout()
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Train and evaluate Dixon-Coles model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dixon-Coles Model")
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROCESSED_DATA_DIR / "features_data_driven.csv"),
        help="Features CSV path"
    )
    parser.add_argument(
        "--time-decay",
        type=float,
        default=0.0018,
        help="Time decay parameter (default: 0.0018, ~1 year half-life)"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save evaluation plots"
    )
    
    args = parser.parse_args()
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    print(f"\nLoaded {len(features_df)} matches")
    
    # Filter to matches with results
    mask = features_df['result_numeric'].notna()
    df = features_df[mask].copy()
    print(f"Matches with results: {len(df)}")
    
    # Split by season
    train_df, val_df, test_df = season_based_split(
        df, 'season',
        TRAIN_SEASONS, VALIDATION_SEASONS, TEST_SEASONS
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} ({TRAIN_SEASONS})")
    print(f"  Validation: {len(val_df)} ({VALIDATION_SEASONS})")
    print(f"  Test: {len(test_df)} ({TEST_SEASONS})")
    
    # Initialize and fit model
    print("\n" + "=" * 60)
    print("FITTING DIXON-COLES MODEL")
    print("=" * 60)
    
    model = DixonColesModel(time_decay=args.time_decay)
    
    # Use training + validation for final model (time decay handles recency)
    train_val_df = pd.concat([train_df, val_df])
    
    model.fit(train_val_df, reference_date=val_df['date'].max())
    
    # Print team strengths
    strengths = model.get_team_strengths()
    print("\nTop 10 teams by overall strength:")
    print(strengths.head(10).to_string(index=False))
    
    print("\nBottom 10 teams:")
    print(strengths.tail(10).to_string(index=False))
    
    # Get predictions
    y_train = train_df['result_numeric'].values.astype(int)
    y_val = val_df['result_numeric'].values.astype(int)
    y_test = test_df['result_numeric'].values.astype(int)
    
    # Evaluate raw model on validation
    print("\n" + "=" * 60)
    print("UNCALIBRATED PREDICTIONS")
    print("=" * 60)
    
    val_probs_raw = model.predict_proba(val_df)
    evaluate_predictions(y_val, val_probs_raw, "Validation (Raw)")
    
    # Calibrate
    print("\n" + "=" * 60)
    print("CALIBRATING MODEL")
    print("=" * 60)
    
    calibrated_model = CalibratedDixonColes(model)
    calibrated_model.calibrate(val_df, y_val)
    
    val_probs_cal = calibrated_model.predict_proba(val_df)
    evaluate_predictions(y_val, val_probs_cal, "Validation (Calibrated)")
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    test_probs = calibrated_model.predict_proba(test_df)
    test_metrics = evaluate_predictions(y_test, test_probs, "Test Set")
    
    # Compare to market
    if 'market_prob_home' in test_df.columns:
        market_mask = test_df['market_prob_home'].notna()
        if market_mask.sum() > 0:
            market_probs = test_df.loc[market_mask, 
                ['market_prob_home', 'market_prob_draw', 'market_prob_away']].values
            test_probs_subset = test_probs[market_mask]
            y_test_subset = y_test[market_mask]
            
            print("\n" + "=" * 60)
            print("COMPARISON TO MARKET")
            print("=" * 60)
            
            model_metrics = evaluate_predictions(y_test_subset, test_probs_subset, "Dixon-Coles")
            market_metrics = evaluate_predictions(y_test_subset, market_probs, "Market")
            
            edge = model_metrics['log_loss'] - market_metrics['log_loss']
            print(f"\nLog Loss Edge: {edge:+.4f}")
    
    # Save model
    model_path = MODELS_DIR / "dixon_coles_model.joblib"
    calibrated_model.save(model_path)
    
    # Generate plots
    if args.save_plots:
        plots_dir = MODELS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Score heatmap for example match
        top_teams = strengths.head(4)['team'].tolist()
        if len(top_teams) >= 2:
            fig = plot_score_heatmap(model, top_teams[0], top_teams[1])
            fig.savefig(plots_dir / "dc_score_heatmap.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nPlots saved to {plots_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dixon-Coles Test Log Loss: {test_metrics['log_loss']:.4f}")
    print(f"Dixon-Coles Test Calibration Error: {test_metrics['calibration_error']:.4f}")
    print(f"\nModel saved to: {model_path}")
    print("\nNext: Run 06_model_xgboost.py for XGBoost model")


if __name__ == "__main__":
    main()
