"""
Configuration settings for Football Prediction Pipeline
========================================================

Setup Instructions:
1. Copy this file or edit directly
2. Add your API keys below
3. Adjust settings as needed

API Key Sources:
- Football-Data.org: https://www.football-data.org/client/register
- API-Football: https://www.api-football.com/ (via RapidAPI)
- The Odds API: https://the-odds-api.com/
"""

import os
from pathlib import Path
from datetime import datetime

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_PATH = DATA_DIR / "football.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# =============================================================================
# API KEYS - ADD YOUR KEYS HERE
# =============================================================================

# Football-Data.org (free tier: 10 requests/minute)
# Register at: https://www.football-data.org/client/register
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "YOUR_API_KEY_HERE")

# API-Football via RapidAPI (free tier: 100 requests/day)
# Register at: https://rapidapi.com/api-sports/api/api-football
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "YOUR_RAPIDAPI_KEY_HERE")
API_FOOTBALL_HOST = "api-football-v1.p.rapidapi.com"

# The Odds API (free tier: 500 requests/month)
# Register at: https://the-odds-api.com/
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "YOUR_ODDS_API_KEY_HERE")

# =============================================================================
# SUPPORTED LEAGUES
# =============================================================================

# Football-data.co.uk CSV codes
LEAGUES_CSV = {
    "E0": {"name": "Premier League", "country": "England"},
    "E1": {"name": "Championship", "country": "England"},
    "SP1": {"name": "La Liga", "country": "Spain"},
    "D1": {"name": "Bundesliga", "country": "Germany"},
    "I1": {"name": "Serie A", "country": "Italy"},
    "F1": {"name": "Ligue 1", "country": "France"},
}

# Football-Data.org API codes
LEAGUES_API = {
    "PL": {"name": "Premier League", "country": "England", "id": 2021},
    "ELC": {"name": "Championship", "country": "England", "id": 2016},
    "PD": {"name": "La Liga", "country": "Spain", "id": 2014},
    "BL1": {"name": "Bundesliga", "country": "Germany", "id": 2002},
    "SA": {"name": "Serie A", "country": "Italy", "id": 2019},
    "FL1": {"name": "Ligue 1", "country": "France", "id": 2015},
}

# API-Football league IDs
LEAGUES_API_FOOTBALL = {
    "Premier League": 39,
    "Championship": 40,
    "La Liga": 140,
    "Bundesliga": 78,
    "Serie A": 135,
    "Ligue 1": 61,
}

# =============================================================================
# DATA COLLECTION SETTINGS
# =============================================================================

# Historical data range
HISTORICAL_SEASONS = [
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

# football-data.co.uk season format (e.g., "2324" for 2023-2024)
def season_to_csv_format(season: str) -> str:
    """Convert '2023-2024' to '2324'"""
    years = season.split("-")
    return years[0][2:] + years[1][2:]

# Seasons for CSV download
CSV_SEASONS = [season_to_csv_format(s) for s in HISTORICAL_SEASONS]

# =============================================================================
# FEATURE ENGINEERING SETTINGS
# =============================================================================

# Elo settings
ELO_INITIAL_RATING = 1500
ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 100
ELO_SEASON_REGRESSION = 0.1  # Regress 10% toward mean between seasons

# Form calculation windows
FORM_WINDOWS = [3, 5, 10]  # Last N matches

# Rolling statistics windows
ROLLING_WINDOWS = [5, 10, 20]  # Matches

# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Train/validation/test split
TRAIN_SEASONS = ["2019-2020", "2020-2021", "2021-2022"]
VALIDATION_SEASONS = ["2022-2023"]
TEST_SEASONS = ["2023-2024"]

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# Ensemble weights (tune on validation set)
ENSEMBLE_WEIGHTS = {
    "elo": 0.2,
    "dixon_coles": 0.3,
    "xgboost": 0.5,
}

# Calibration settings
CALIBRATION_METHOD = "isotonic"  # or "sigmoid"
CALIBRATION_BINS = 10

# =============================================================================
# EDGE DETECTION SETTINGS
# =============================================================================

# Minimum edge to trigger alert
MIN_EDGE_THRESHOLD = 0.03  # 3%

# Minimum expected value
MIN_EV_THRESHOLD = 0.02  # 2%

# Stale data thresholds
ODDS_STALE_MINUTES = 30
MATCH_CUTOFF_MINUTES = 60  # Don't alert if match starts within this time

# =============================================================================
# API RATE LIMITS
# =============================================================================

RATE_LIMITS = {
    "football_data_org": {"calls": 10, "period": 60},  # 10 per minute
    "api_football": {"calls": 10, "period": 60},  # 10 per minute
    "odds_api": {"calls": 500, "period": 2592000},  # 500 per month
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# RANDOM SEED
# =============================================================================

RANDOM_SEED = 42
