"""
Feature engineering configuration.
"""


class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Elo Rating Parameters
    ELO_K_FACTOR = 32
    ELO_HOME_ADVANTAGE = 35  # Reduced from 50 based on research
    ELO_INITIAL_RATING = 1500
    ELO_REGRESSION_FACTOR = 0.5  # Regression to mean between seasons
    
    # Derived xG Coefficients
    XG_INSIDE_BOX = 0.12
    XG_OUTSIDE_BOX = 0.03
    XG_BIG_CHANCE = 0.35
    XG_CORNER = 0.03
    XG_ACCURACY_MULTIPLIER_MAX = 1.3
    
    # Rolling Window Sizes
    FORM_WINDOWS = [3, 5, 10]
    XG_WINDOWS = [5, 10]
    PLAYER_FORM_WINDOW = 3
    
    # Momentum Calculation
    MOMENTUM_WINDOW = 10
    WEIGHTED_FORM_ALPHA = 0.3  # Exponential weighting
    
    # Player Features
    TOP_PLAYERS_COUNT = 5  # Number of key players to track
    PLAYER_FORM_THRESHOLD = 7.0  # Rating threshold for "in form"
    
    # Feature Thresholds
    MIN_MATCHES_FOR_FEATURES = 3  # Minimum matches needed for rolling features
    MIN_H2H_MATCHES = 3  # Minimum H2H matches for H2H features
    
    # Missing Data Handling
    FILL_MISSING_WITH_LEAGUE_AVG = True
    DEFAULT_ELO = 1500
    DEFAULT_XG = 1.0
    DEFAULT_XGA = 1.0
    
    # Feature Groups (for selective calculation)
    FEATURE_GROUPS = {
        'elo': True,
        'form': True,
        'xg': True,
        'h2h': True,
        'shots': True,
        'defensive': True,
        'player': True,
        'momentum': True,
        'context': True,
    }
    
    @classmethod
    def get_enabled_groups(cls):
        """Get list of enabled feature groups."""
        return [group for group, enabled in cls.FEATURE_GROUPS.items() if enabled]
