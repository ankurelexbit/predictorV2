"""
Model Configuration for V4 Pipeline.

Centralized configuration for Elo, features, and model parameters.
"""


class EloConfig:
    """Elo rating system configuration."""

    # Elo parameters
    K_FACTOR = 32  # Update speed (higher = more responsive to recent results)
    HOME_ADVANTAGE = 35  # Home team Elo bonus (35 is calibrated for modern football)
    INITIAL_ELO = 1500  # Starting Elo for new teams

    @classmethod
    def get_params(cls):
        """Get Elo parameters as dict."""
        return {
            'k_factor': cls.K_FACTOR,
            'home_advantage': cls.HOME_ADVANTAGE,
            'initial_elo': cls.INITIAL_ELO
        }


class FeatureConfig:
    """Feature generation configuration."""

    # Form windows
    FORM_WINDOW_SHORT = 3  # Last 3 matches
    FORM_WINDOW_MEDIUM = 5  # Last 5 matches
    FORM_WINDOW_LONG = 10  # Last 10 matches

    # Player features
    KEY_PLAYERS_COUNT = 5  # Number of key players to track
    FORM_RATING_THRESHOLD = 7.0  # Rating threshold for "in form"

    # xG parameters
    XG_WINDOW = 5  # Matches to calculate average xG


class ModelConfig:
    """Model training configuration."""

    # Model type
    MODEL_TYPE = 'catboost'  # 'xgboost' or 'catboost'

    # Class weights (for draw-focused model)
    CLASS_WEIGHTS = {
        0: 1.2,  # Away
        1: 1.5,  # Draw
        2: 1.0   # Home
    }

    # Training split
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15


class PredictionConfig:
    """Live prediction configuration."""

    # Historical data
    HISTORY_DAYS = 365  # Days of history to load on startup

    # Betting thresholds
    THRESHOLD_HOME = 0.48  # Bet on home if prob > 48%
    THRESHOLD_DRAW = 0.35  # Bet on draw if prob > 35%
    THRESHOLD_AWAY = 0.45  # Bet on away if prob > 45%

    # Model path
    MODEL_PATH = 'models/with_draw_features/conservative_with_draw_features.joblib'


# Quick access
ELO_PARAMS = EloConfig.get_params()
