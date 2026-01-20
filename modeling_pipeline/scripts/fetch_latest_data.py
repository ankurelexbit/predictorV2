#!/usr/bin/env python3
"""
Fetch Latest Data from SportMonks API

Fetches the latest match data (last N days) including:
- Match results
- Odds
- Statistics
- Lineups (if available)

This script is run before weekly model retraining to ensure
the model is trained on the most recent data.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
import time
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from utils import setup_logger

logger = setup_logger("fetch_latest_data")

# SportMonks API configuration
API_KEY = os.getenv('SPORTMONKS_API_KEY', '')
BASE_URL = "https://api.sportmonks.com/v3/football"

def fetch_matches_by_date_range(start_date: str, end_date: str):
    """Fetch matches within a date range."""
    logger.info(f"Fetching matches from {start_date} to {end_date}")
    
    if not API_KEY:
        logger.error("SPORTMONKS_API_KEY not set!")
        return []
    
    url = f"{BASE_URL}/fixtures/between/{start_date}/{end_date}"
    params = {
        'api_token': API_KEY,
        'include': 'participants;scores;league;odds;statistics.details',
        'filters': 'fixtureLeagues:8,384,564,462,301'  # Top 5 leagues
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data:
            logger.warning("No data in response")
            return []
        
        logger.info(f"Fetched {len(data['data'])} matches")
        return data['data']
    
    except Exception as e:
        logger.error(f"Error fetching matches: {e}")
        return []

def process_match_data(matches):
    """Process raw match data into structured format."""
    processed = []
    
    for match in matches:
        try:
            # Extract participants
            participants = match.get('participants', [])
            if len(participants) < 2:
                continue
            
            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
            
            if not home_team or not away_team:
                continue
            
            # Extract scores
            scores = match.get('scores', [])
            home_score = None
            away_score = None
            
            for score in scores:
                if score.get('description') == 'CURRENT':
                    if score.get('participant_id') == home_team['id']:
                        home_score = score.get('score', {}).get('goals')
                    elif score.get('participant_id') == away_team['id']:
                        away_score = score.get('score', {}).get('goals')
            
            # Determine outcome
            target = None
            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    target = 2  # Home win
                elif home_score < away_score:
                    target = 0  # Away win
                else:
                    target = 1  # Draw
            
            # Extract odds
            odds_data = match.get('odds', [])
            odds_home = None
            odds_draw = None
            odds_away = None
            
            for odds in odds_data:
                if odds.get('name') == '3Way Result':
                    for value in odds.get('values', []):
                        if value.get('value') == 'Home':
                            odds_home = float(value.get('odd', 0))
                        elif value.get('value') == 'Draw':
                            odds_draw = float(value.get('odd', 0))
                        elif value.get('value') == 'Away':
                            odds_away = float(value.get('odd', 0))
            
            processed.append({
                'fixture_id': match['id'],
                'date': match.get('starting_at'),
                'season_id': match.get('season_id'),
                'league_id': match.get('league_id'),
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'home_goals': home_score,
                'away_goals': away_score,
                'target': target,
                'odds_home': odds_home,
                'odds_draw': odds_draw,
                'odds_away': odds_away,
                'state': match.get('state', {}).get('state')
            })
        
        except Exception as e:
            logger.warning(f"Error processing match {match.get('id')}: {e}")
            continue
    
    return processed

def update_dataset(new_data_df):
    """Update the existing dataset with new data."""
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Please run feature engineering first")
        return False
    
    # Load existing data
    logger.info(f"Loading existing dataset from {features_path}")
    existing_df = pd.read_csv(features_path)
    existing_df['date'] = pd.to_datetime(existing_df['date'])
    
    logger.info(f"Existing dataset: {len(existing_df)} matches")
    
    # Convert new data to DataFrame
    new_df = pd.DataFrame(new_data_df)
    new_df['date'] = pd.to_datetime(new_df['date'])
    
    # Filter for finished matches only
    finished = new_df[new_df['state'] == 'FT'].copy()
    logger.info(f"New finished matches: {len(finished)}")
    
    if len(finished) == 0:
        logger.info("No new finished matches to add")
        return True
    
    # Remove duplicates (matches already in dataset)
    existing_ids = set(existing_df['fixture_id'].values)
    new_matches = finished[~finished['fixture_id'].isin(existing_ids)]
    
    logger.info(f"New unique matches to add: {len(new_matches)}")
    
    if len(new_matches) == 0:
        logger.info("All matches already in dataset")
        return True
    
    # Save new matches to raw data
    raw_file = RAW_DATA_DIR / f'new_matches_{datetime.now().strftime("%Y%m%d")}.csv'
    new_matches.to_csv(raw_file, index=False)
    logger.info(f"Saved new matches to {raw_file}")
    
    logger.info(f"\n✅ Fetched {len(new_matches)} new matches")
    logger.info("   Run feature engineering to add them to the training dataset")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Fetch latest match data from SportMonks")
    parser.add_argument('--days', type=int, default=7, help='Number of days to fetch (default: 7)')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("FETCH LATEST DATA FROM SPORTMONKS")
    logger.info("=" * 80)
    logger.info(f"Fetching last {args.days} days of data")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_str} to {end_str}")
    
    # Fetch matches
    matches = fetch_matches_by_date_range(start_str, end_str)
    
    if not matches:
        logger.warning("No matches fetched")
        return 1
    
    # Process matches
    logger.info("Processing match data...")
    processed = process_match_data(matches)
    
    logger.info(f"Processed {len(processed)} matches")
    
    # Update dataset
    success = update_dataset(processed)
    
    if success:
        logger.info("\n✅ Data fetch complete!")
        return 0
    else:
        logger.error("\n❌ Data fetch failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
