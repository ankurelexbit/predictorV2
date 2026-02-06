#!/usr/bin/env python3
"""
V5 Production Configuration
===========================

Optimized configuration validated on 3 years of data (2023-2025, 5322 predictions):
- Model: v1.0.1 (CatBoost + LightGBM ensemble)
- Strategy: Threshold + odds filter with recalibration every 8 weeks
- Default thresholds: H60/D35/A45 (3-year optimal, 6.7% ROI)
- Odds Filter: 1.5-3.5
- With recalibration (8wk/120d): 9.8% ROI, only 15% losing months
"""

import os
from pathlib import Path

# Load .env file early (before any env reads)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_DIR = Path(__file__).parent.parent / "models" / "production"

def get_latest_model_path() -> str:
    """Get path to latest production model."""
    latest_file = MODEL_DIR / "LATEST"
    if latest_file.exists():
        model_name = latest_file.read_text().strip()
        return str(MODEL_DIR / model_name)

    # Fallback: find highest version
    models = list(MODEL_DIR.glob("model_v*.joblib"))
    if models:
        return str(sorted(models)[-1])

    raise FileNotFoundError("No production model found")

# ============================================================================
# BETTING STRATEGY CONFIGURATION
# ============================================================================

# Validated on 3 years (2023-2025): 1757 bets, 46.0% WR, +6.7% ROI
# More away bets at lower threshold captures profitable value
THRESHOLDS = {
    'home': 0.60,
    'away': 0.45,
    'draw': 0.35
}

# Odds range filter (1.5-3.5 consistently optimal across all analyses)
ODDS_FILTER = {
    'min': 1.5,
    'max': 3.5,
    'enabled': True
}

# Strategy options (can be changed for different risk profiles)
STRATEGY_PROFILES = {
    'conservative': {
        'thresholds': {'home': 0.60, 'away': 0.45, 'draw': 0.35},
        'odds_filter': {'min': 1.5, 'max': 3.5, 'enabled': True},
        'description': '~50 bets/mo, 46% WR, +6.7% ROI (3-year validated)'
    },
    'selective': {
        'thresholds': {'home': 0.55, 'away': 0.55, 'draw': 0.40},
        'odds_filter': {'min': 1.5, 'max': 3.5, 'enabled': True},
        'description': '~30 bets/mo, 52% WR, +5.8% ROI (fewer high-confidence bets)'
    },
    'high_volume': {
        'thresholds': {'home': 0.45, 'away': 0.45, 'draw': 0.35},
        'odds_filter': {'min': 1.3, 'max': 3.5, 'enabled': True},
        'description': '~65 bets/mo, 49% WR, +3.6% ROI (max volume)'
    },
    'tight': {
        'thresholds': {'home': 0.60, 'away': 0.55, 'draw': 0.40},
        'odds_filter': {'min': 1.5, 'max': 3.5, 'enabled': True},
        'description': '~25 bets/mo, 52% WR, +5.8% ROI (high-confidence only)'
    }
}

# Active strategy
ACTIVE_STRATEGY = 'conservative'

# ============================================================================
# LEAGUE FILTERING
# ============================================================================

TOP_5_LEAGUES = [
    8,    # Premier League (England)
    82,   # Bundesliga (Germany)
    384,  # Serie A (Italy)
    564,  # Ligue 1 (France)
    301   # La Liga (Spain)
]

FILTER_TOP_5_ONLY = True

# ============================================================================
# ELO CONFIGURATION
# ============================================================================

class EloConfig:
    K_FACTOR = 32
    HOME_ADVANTAGE = 35
    INITIAL_ELO = 1500

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

META_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target'
]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    'test_size': 0.15,
    'val_size': 0.15,
    'random_state': 42,
    'catboost': {
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5,
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 50
    },
    'lightgbm': {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'reg_lambda': 5,
        'class_weight': 'balanced'
    }
}

# Hyperparameter search space for tuning
HYPERPARAM_SEARCH = {
    'catboost': {
        'iterations': [300, 500, 700],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'l2_leaf_reg': [3, 5, 7]
    },
    'lightgbm': {
        'n_estimators': [300, 500, 700],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'reg_lambda': [3, 5, 7]
    }
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

SPORTMONKS_API_KEY = os.environ.get('SPORTMONKS_API_KEY', '')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_active_strategy():
    """Get currently active betting strategy configuration."""
    return STRATEGY_PROFILES.get(ACTIVE_STRATEGY, STRATEGY_PROFILES['conservative'])

def validate_config():
    """Validate configuration settings."""
    errors = []

    strategy = get_active_strategy()
    for outcome, thresh in strategy['thresholds'].items():
        if not 0.0 <= thresh <= 1.0:
            errors.append(f"Invalid threshold for {outcome}: {thresh}")

    if strategy['odds_filter']['enabled']:
        if strategy['odds_filter']['min'] >= strategy['odds_filter']['max']:
            errors.append("odds_filter min must be < max")

    if errors:
        raise ValueError(f"Configuration errors: {errors}")

    return True


if __name__ == '__main__':
    print("V5 Production Configuration")
    print("=" * 60)
    print(f"\nActive Strategy: {ACTIVE_STRATEGY}")
    strategy = get_active_strategy()
    print(f"Description: {strategy['description']}")
    print(f"\nThresholds:")
    for k, v in strategy['thresholds'].items():
        print(f"  {k}: {v}")
    print(f"\nOdds Filter: {strategy['odds_filter']}")
    print(f"Top 5 Leagues Only: {FILTER_TOP_5_ONLY}")

    validate_config()
    print("\n✅ Configuration valid")
