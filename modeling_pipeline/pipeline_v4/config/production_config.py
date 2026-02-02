"""
Production Configuration
========================

CRITICAL: This file contains the production model and threshold settings.
Any changes here will affect live predictions!

Last Updated: 2026-02-02
Model: Option 3 (Balanced) - H=1.2, D=1.4, A=1.1
Tested on: January 2026 (202 Top 5 League matches)
Performance: $41.81 profit, 28.3% ROI, 52.0% win rate
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Production Model Path
MODEL_PATH = "models/weight_experiments/option3_balanced.joblib"

# Model Metadata
MODEL_INFO = {
    'name': 'Option 3: Balanced',
    'version': 'v4.1',
    'class_weights': {'home': 1.2, 'draw': 1.4, 'away': 1.1},
    'trained_on': '2023-2024 seasons',
    'last_updated': '2026-02-02'
}

# ============================================================================
# BETTING THRESHOLDS
# ============================================================================

# Optimal thresholds for Option 3 model
# Based on January 2026 backtest (Top 5 Leagues)
THRESHOLDS = {
    'home': 0.65,  # 70.0% win rate, 1.2% ROI
    'draw': 0.30,  # 39.0% win rate, 37.8% ROI (primary profit source)
    'away': 0.42   # 66.7% win rate, 29.1% ROI
}

# Historical threshold performance (for reference)
THRESHOLD_HISTORY = {
    '2026-02-02': {
        'thresholds': {'home': 0.65, 'draw': 0.30, 'away': 0.42},
        'performance': {
            'total_bets': 148,
            'total_profit': 41.81,
            'roi': 28.3,
            'win_rate': 52.0,
            'draw_winrate': 39.0,
            'test_period': 'January 2026',
            'leagues': 'Top 5 only'
        }
    }
}

# ============================================================================
# LEAGUE FILTERING
# ============================================================================

# TOP 5 EUROPEAN LEAGUES (RECOMMENDED)
# Model performs significantly better on these leagues:
#   Top 5: 28.3% ROI, 39.0% draw win rate
#   All leagues: 9.0% ROI, 28.9% draw win rate
TOP_5_LEAGUES = [
    8,    # Premier League (England)
    82,   # Bundesliga (Germany)
    384,  # Serie A (Italy)
    564,  # Ligue 1 (France)
    301   # La Liga (Spain)
]

# Set to True to filter predictions to Top 5 leagues only
FILTER_TOP_5_ONLY = True

# Alternative: Specify custom league IDs
# CUSTOM_LEAGUES = [8, 82, 384, 564, 301, 301]  # Add more if needed
# FILTER_CUSTOM_LEAGUES = False

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================

# Minimum confidence threshold (predictions below this won't be stored)
MIN_CONFIDENCE_ANY_OUTCOME = 0.15  # At least one outcome must be >15%

# Only store predictions that cross betting threshold
STORE_ONLY_ACTIONABLE = False  # Set True to only store bets you'd actually place

# ============================================================================
# HISTORICAL DATA SETTINGS
# ============================================================================

# How much historical data to load for feature calculation
HISTORY_DAYS = 365  # 1 year of history

# Alternative: Specific date range
USE_DATE_RANGE = False
HISTORY_START_DATE = None  # "2025-01-01"
HISTORY_END_DATE = None    # "2026-01-31"

# ============================================================================
# DATABASE SETTINGS
# ============================================================================

# Table name for storing predictions
PREDICTIONS_TABLE = "predictions"

# Model version tag (for tracking different model versions)
MODEL_VERSION_TAG = "v4.1_option3_balanced"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []

    # Check model path exists
    from pathlib import Path
    if not Path(MODEL_PATH).exists():
        errors.append(f"Model path does not exist: {MODEL_PATH}")

    # Validate thresholds
    for outcome, thresh in THRESHOLDS.items():
        if not 0.0 <= thresh <= 1.0:
            errors.append(f"Invalid threshold for {outcome}: {thresh} (must be 0-1)")

    # Check league IDs
    if FILTER_TOP_5_ONLY:
        if not TOP_5_LEAGUES or not all(isinstance(x, int) for x in TOP_5_LEAGUES):
            errors.append("TOP_5_LEAGUES must be a list of integers")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


if __name__ == '__main__':
    """Print current configuration."""
    print("="*80)
    print("PRODUCTION CONFIGURATION")
    print("="*80)
    print()
    print(f"Model: {MODEL_INFO['name']}")
    print(f"Path: {MODEL_PATH}")
    print(f"Version: {MODEL_VERSION_TAG}")
    print()
    print("Thresholds:")
    for outcome, thresh in THRESHOLDS.items():
        print(f"  {outcome.capitalize()}: {thresh:.2f}")
    print()
    print(f"League Filter: {'Top 5 only' if FILTER_TOP_5_ONLY else 'All leagues'}")
    if FILTER_TOP_5_ONLY:
        print(f"  Leagues: {TOP_5_LEAGUES}")
    print()

    try:
        validate_config()
        print("✅ Configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration errors:")
        print(str(e))
