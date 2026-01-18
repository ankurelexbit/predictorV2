"""
Predict Upcoming Matches
=========================

Make predictions for today's and tomorrow's matches using the trained ensemble model.

Usage:
    python predict_upcoming.py --date today
    python predict_upcoming.py --date tomorrow
    python predict_upcoming.py --date 2026-01-19
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR
from utils import setup_logger
import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup
logger = setup_logger("predict_upcoming")

# Sportmonks API configuration
API_KEY = "LmEYRdsf8CSmiblNVTn6JfV0y0s8tc4aQsEkJ6JuoBhOWa3Hd1FEanGcrijo"
BASE_URL = "https://api.sportmonks.com/v3/football"


def get_upcoming_fixtures(target_date: str) -> pd.DataFrame:
    """
    Fetch upcoming fixtures for a specific date.

    Args:
        target_date: Date in format YYYY-MM-DD

    Returns:
        DataFrame with fixture details
    """
    logger.info(f"Fetching fixtures for {target_date}")

    try:
        # Fetch fixtures by date using between endpoint
        url = f"{BASE_URL}/fixtures/between/{target_date}/{target_date}"
        params = {
            'api_token': API_KEY,
            'include': 'participants;league;venue'
        }

        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or 'data' not in data:
            logger.warning(f"No fixtures found for {target_date}")
            return pd.DataFrame()

        fixtures = []
        for fixture in data['data']:
            # Extract team info
            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            fixtures.append({
                'fixture_id': fixture['id'],
                'date': fixture.get('starting_at'),
                'league_name': fixture.get('league', {}).get('name', 'Unknown'),
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'venue': fixture.get('venue', {}).get('name', 'Unknown') if fixture.get('venue') else 'Unknown'
            })

        df = pd.DataFrame(fixtures)
        logger.info(f"Found {len(df)} fixtures")
        return df

    except Exception as e:
        logger.error(f"Error fetching fixtures: {e}")
        return pd.DataFrame()


def load_historical_features() -> pd.DataFrame:
    """Load historical features for team lookup."""
    features_path = Path("data/processed/sportmonks_features.csv")

    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return pd.DataFrame()

    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])

    # Get most recent features for each team
    df = df.sort_values('date', ascending=False)

    return df


def get_team_features(team_name: str, historical_df: pd.DataFrame, as_home: bool = True) -> Dict:
    """
    Get latest features for a team.

    Args:
        team_name: Name of the team
        historical_df: Historical features dataframe
        as_home: Whether team is playing at home

    Returns:
        Dictionary of features
    """
    # Find most recent match for this team
    if as_home:
        team_matches = historical_df[historical_df['home_team_name'] == team_name]
        prefix = 'home_'
    else:
        team_matches = historical_df[historical_df['away_team_name'] == team_name]
        prefix = 'away_'

    if len(team_matches) == 0:
        logger.warning(f"No historical data found for {team_name}")
        return {}

    # Get most recent match
    latest = team_matches.iloc[0]

    # Extract relevant features - ALL features the model expects
    features = {}

    # All team-specific features needed by the model
    team_features = [
        'elo', 'form_3', 'wins_3', 'form_5', 'wins_5',
        'goals_5', 'goals_conceded_5', 'xg_5', 'xg_conceded_5',
        'shots_total_5', 'shots_on_target_5', 'dangerous_attacks_5',
        'possession_pct_5', 'successful_passes_pct_5', 'tackles_5',
        'interceptions_5', 'big_chances_created_5', 'player_clearances_5',
        'player_rating_5', 'player_touches_5', 'player_duels_won_5',
        'goals_10', 'xg_10', 'position', 'points', 'injuries',
        'attack_strength_5', 'defense_strength_5'
    ]

    for feat in team_features:
        col_name = f"{prefix}{feat}"
        if col_name in latest.index:
            features[col_name] = latest[col_name]
        else:
            # Fill with 0 if not found
            features[col_name] = 0

    return features


def build_match_features(
    home_team: str,
    away_team: str,
    historical_df: pd.DataFrame,
    fixture_date: datetime
) -> Dict:
    """
    Build complete feature vector for a match.

    Args:
        home_team: Home team name
        away_team: Away team name
        historical_df: Historical features
        fixture_date: Date of the fixture

    Returns:
        Dictionary of all features needed for prediction
    """
    features = {}

    # Get team-specific features
    home_features = get_team_features(home_team, historical_df, as_home=True)
    away_features = get_team_features(away_team, historical_df, as_home=False)

    features.update(home_features)
    features.update(away_features)

    # Calculate all derived features that the model expects
    if 'home_elo' in features and 'away_elo' in features:
        features['elo_diff'] = features['home_elo'] - features['away_elo']

    if 'home_form_3' in features and 'away_form_3' in features:
        features['form_diff_3'] = features['home_form_3'] - features['away_form_3']

    if 'home_form_5' in features and 'away_form_5' in features:
        features['form_diff_5'] = features['home_form_5'] - features['away_form_5']

    if 'home_position' in features and 'away_position' in features:
        features['position_diff'] = features['away_position'] - features['home_position']

    if 'home_points' in features and 'away_points' in features:
        features['points_diff'] = features['home_points'] - features['away_points']

    if 'home_injuries' in features and 'away_injuries' in features:
        features['injury_diff'] = features['home_injuries'] - features['away_injuries']

    # Calculate H2H features from historical data
    h2h_matches = historical_df[
        ((historical_df['home_team_name'] == home_team) & (historical_df['away_team_name'] == away_team)) |
        ((historical_df['home_team_name'] == away_team) & (historical_df['away_team_name'] == home_team))
    ].head(10)  # Last 10 H2H matches

    if len(h2h_matches) > 0:
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = []
        away_goals = []

        for _, match in h2h_matches.iterrows():
            if match['home_team_name'] == home_team:
                home_goals.append(match['home_goals'])
                away_goals.append(match['away_goals'])
                if match['home_goals'] > match['away_goals']:
                    home_wins += 1
                elif match['home_goals'] < match['away_goals']:
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals.append(match['away_goals'])
                away_goals.append(match['home_goals'])
                if match['away_goals'] > match['home_goals']:
                    home_wins += 1
                elif match['away_goals'] < match['home_goals']:
                    away_wins += 1
                else:
                    draws += 1

        features['h2h_home_wins'] = home_wins
        features['h2h_draws'] = draws
        features['h2h_away_wins'] = away_wins
        features['h2h_home_goals_avg'] = np.mean(home_goals) if home_goals else 0
        features['h2h_away_goals_avg'] = np.mean(away_goals) if away_goals else 0
    else:
        # No H2H history
        features['h2h_home_wins'] = 0
        features['h2h_draws'] = 0
        features['h2h_away_wins'] = 0
        features['h2h_home_goals_avg'] = 0
        features['h2h_away_goals_avg'] = 0

    # Add contextual/temporal features
    # Estimate round_num and season_progress (approximate)
    features['round_num'] = 20  # Mid-season default
    features['season_progress'] = 0.5  # Mid-season default
    features['is_early_season'] = 0
    features['is_weekend'] = 1 if fixture_date.weekday() >= 5 else 0

    return features


def load_models():
    """Load all trained models."""
    models = {}

    # Load individual models
    elo_path = MODELS_DIR / "elo_model.joblib"
    if elo_path.exists():
        import importlib
        elo_module = importlib.import_module('04_model_baseline_elo')
        EloProbabilityModel = elo_module.EloProbabilityModel
        elo_model = EloProbabilityModel()
        elo_model.load(elo_path)
        models['elo'] = elo_model
        logger.info("Loaded Elo model")

    dixon_path = MODELS_DIR / "dixon_coles_model.joblib"
    if dixon_path.exists():
        try:
            dc_data = joblib.load(dixon_path)
            import importlib
            dc_module = importlib.import_module('05_model_dixon_coles')
            DixonColesModel = dc_module.DixonColesModel
            CalibratedDixonColes = dc_module.CalibratedDixonColes

            base_model = DixonColesModel()
            base_model.attack = dc_data['base_model_data']['attack']
            base_model.defense = dc_data['base_model_data']['defense']
            base_model.home_advantage = dc_data['base_model_data']['home_advantage']
            base_model.team_to_idx = dc_data['base_model_data']['team_to_idx']
            base_model.idx_to_team = dc_data['base_model_data']['idx_to_team']
            base_model.reference_date = dc_data['base_model_data']['reference_date']
            base_model.time_decay = dc_data['base_model_data']['time_decay']
            base_model.is_fitted = True

            dc_model = CalibratedDixonColes(base_model)
            dc_model.calibrators = dc_data['calibrators']
            dc_model.is_calibrated = dc_data['is_calibrated']

            models['dixon_coles'] = dc_model
            logger.info("Loaded Dixon-Coles model")
        except Exception as e:
            logger.warning(f"Could not load Dixon-Coles: {e}")

    xgb_path = MODELS_DIR / "xgboost_model.joblib"
    if xgb_path.exists():
        try:
            import importlib
            xgb_module = importlib.import_module('06_model_xgboost')
            XGBoostFootballModel = xgb_module.XGBoostFootballModel
            xgb_model = XGBoostFootballModel()
            xgb_model.load(xgb_path)
            models['xgboost'] = xgb_model
            logger.info("Loaded XGBoost model")
        except Exception as e:
            logger.warning(f"Could not load XGBoost: {e}")

    # Load stacking ensemble
    stacking_path = MODELS_DIR / "stacking_ensemble.joblib"
    if stacking_path.exists() and len(models) == 3:
        try:
            import importlib
            ensemble_module = importlib.import_module('07_model_ensemble')
            StackingEnsemble = ensemble_module.StackingEnsemble

            stacking = StackingEnsemble()
            for name, model in models.items():
                stacking.add_model(name, model)
            stacking.load(stacking_path)

            models['stacking'] = stacking
            logger.info("Loaded Stacking Ensemble")
        except Exception as e:
            logger.warning(f"Could not load Stacking: {e}")

    return models


def predict_match(
    fixture_row: pd.Series,
    historical_df: pd.DataFrame,
    models: Dict
) -> Dict:
    """
    Make prediction for a single match.

    Args:
        fixture_row: Row from fixtures dataframe
        historical_df: Historical features
        models: Dictionary of loaded models

    Returns:
        Dictionary with predictions
    """
    home_team = fixture_row['home_team_name']
    away_team = fixture_row['away_team_name']
    fixture_date = pd.to_datetime(fixture_row['date'])

    # Build features
    features = build_match_features(home_team, away_team, historical_df, fixture_date)

    if not features:
        return {
            'error': 'Could not build features',
            'home_win_prob': None,
            'draw_prob': None,
            'away_win_prob': None
        }

    # Convert to DataFrame for model input
    feature_df = pd.DataFrame([features])

    # Fill missing values with 0 (for any features not found)
    feature_df = feature_df.fillna(0)

    # Make prediction with best model (stacking if available)
    model_name = 'stacking' if 'stacking' in models else 'xgboost'
    model = models.get(model_name)

    if model is None:
        return {
            'error': 'No model available',
            'home_win_prob': None,
            'draw_prob': None,
            'away_win_prob': None
        }

    try:
        probs = model.predict_proba(feature_df)[0]

        # Format: [away, draw, home] -> convert to home/draw/away
        return {
            'model_used': model_name,
            'home_win_prob': float(probs[2]),
            'draw_prob': float(probs[1]),
            'away_win_prob': float(probs[0]),
            'predicted_outcome': ['Away Win', 'Draw', 'Home Win'][np.argmax(probs)]
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'error': str(e),
            'home_win_prob': None,
            'draw_prob': None,
            'away_win_prob': None
        }


def main():
    parser = argparse.ArgumentParser(description="Predict upcoming football matches")
    parser.add_argument(
        "--date",
        default="today",
        help="Date to predict (today, tomorrow, or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path (optional)"
    )

    args = parser.parse_args()

    # Parse date
    if args.date.lower() == "today":
        target_date = datetime.now().date()
    elif args.date.lower() == "tomorrow":
        target_date = (datetime.now() + timedelta(days=1)).date()
    else:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    date_str = target_date.strftime("%Y-%m-%d")

    print("=" * 80)
    print(f"MATCH PREDICTIONS FOR {date_str}")
    print("=" * 80)

    # Get fixtures
    fixtures_df = get_upcoming_fixtures(date_str)

    if fixtures_df.empty:
        print(f"\nNo fixtures found for {date_str}")
        return

    print(f"\nFound {len(fixtures_df)} fixtures")

    # Load historical data
    print("\nLoading historical data...")
    historical_df = load_historical_features()

    if historical_df.empty:
        print("ERROR: Could not load historical features")
        return

    # Load models
    print("\nLoading models...")
    models = load_models()

    if not models:
        print("ERROR: No models loaded")
        return

    print(f"Loaded {len(models)} model(s): {', '.join(models.keys())}")

    # Check which teams we have data for
    known_teams = set(historical_df['home_team_name'].unique()) | set(historical_df['away_team_name'].unique())

    # Filter to only fixtures where we have data for both teams
    predictable_fixtures = []
    skipped_fixtures = []

    for idx, fixture in fixtures_df.iterrows():
        home_team = fixture['home_team_name']
        away_team = fixture['away_team_name']

        if home_team in known_teams and away_team in known_teams:
            predictable_fixtures.append(fixture)
        else:
            skipped_fixtures.append(fixture)

    print(f"\nFound {len(predictable_fixtures)} predictable matches (have data for both teams)")
    print(f"Skipping {len(skipped_fixtures)} matches (missing team data)")

    # Make predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)

    predictions = []

    for fixture in predictable_fixtures:
        pred = predict_match(fixture, historical_df, models)

        # Combine fixture info with prediction
        result = {
            'date': fixture['date'],
            'league': fixture['league_name'],
            'home_team': fixture['home_team_name'],
            'away_team': fixture['away_team_name'],
            'venue': fixture['venue'],
            **pred
        }

        predictions.append(result)

        # Print prediction
        print(f"\n{fixture['league_name']}")
        print(f"{fixture['date']} | {fixture['venue']}")
        print(f"{fixture['home_team_name']} vs {fixture['away_team_name']}")

        if 'error' in pred:
            print(f"  ERROR: {pred['error']}")
        else:
            print(f"  Model: {pred['model_used']}")
            print(f"  Home Win: {pred['home_win_prob']:.1%}")
            print(f"  Draw:     {pred['draw_prob']:.1%}")
            print(f"  Away Win: {pred['away_win_prob']:.1%}")
            print(f"  → Prediction: {pred['predicted_outcome']}")

    # Save to CSV if requested
    if args.output:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")

    print("\n" + "=" * 80)
    print(f"Completed {len(predictions)} predictions")
    print("=" * 80)


if __name__ == "__main__":
    main()
