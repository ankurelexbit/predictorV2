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
        'include': 'participants;scores;league;odds;statistics.details;lineups;events;sidelined',
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

def extract_lineups(matches):
    """Extract lineup data from matches."""
    lineups = []
    for match in matches:
        fixture_id = match['id']
        lineup_data = match.get('lineups', [])
        
        for lineup in lineup_data:
            lineups.append({
                'fixture_id': fixture_id,
                'player_id': lineup.get('player_id'),
                'team_id': lineup.get('participant_id'),
                'position_id': lineup.get('position_id'),
                'formation_position': lineup.get('formation_position'),
                'jersey_number': lineup.get('jersey_number')
            })
    
    return lineups

def extract_events(matches):
    """Extract event data from matches."""
    events = []
    for match in matches:
        fixture_id = match['id']
        event_data = match.get('events', [])
        
        for event in event_data:
            events.append({
                'fixture_id': fixture_id,
                'event_id': event.get('id'),
                'type_id': event.get('type_id'),
                'participant_id': event.get('participant_id'),
                'player_id': event.get('player_id'),
                'minute': event.get('minute'),
                'extra_minute': event.get('extra_minute'),
                'period_id': event.get('period_id')
            })
    
    return events

def extract_sidelined(matches):
    """Extract sidelined player data from matches."""
    sidelined = []
    for match in matches:
        fixture_id = match['id']
        sidelined_data = match.get('sidelined', [])
        
        for player in sidelined_data:
            sidelined.append({
                'fixture_id': fixture_id,
                'player_id': player.get('player_id'),
                'team_id': player.get('participant_id'),
                'category': player.get('category'),
                'start_date': player.get('start_date'),
                'end_date': player.get('end_date')
            })
    
    return sidelined

def save_to_raw_data(matches):
    """Save all data types to raw data directory."""
    from pathlib import Path
    
    raw_dir = RAW_DATA_DIR / 'sportmonks'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Extracting and saving data...")
    
    # 1. Process and save fixtures
    fixtures_data = process_match_data(matches)
    if fixtures_data:
        fixtures_df = pd.DataFrame(fixtures_data)
        
        # Append to existing fixtures.csv
        fixtures_file = raw_dir / 'fixtures.csv'
        if fixtures_file.exists():
            existing_fixtures = pd.read_csv(fixtures_file)
            # Remove duplicates
            existing_ids = set(existing_fixtures['fixture_id'].values)
            new_fixtures = fixtures_df[~fixtures_df['fixture_id'].isin(existing_ids)]
            if len(new_fixtures) > 0:
                combined = pd.concat([existing_fixtures, new_fixtures], ignore_index=True)
                combined.to_csv(fixtures_file, index=False)
                logger.info(f"  ✅ Added {len(new_fixtures)} new fixtures")
            else:
                logger.info(f"  ℹ️  No new fixtures to add")
        else:
            fixtures_df.to_csv(fixtures_file, index=False)
            logger.info(f"  ✅ Created fixtures.csv with {len(fixtures_df)} matches")
    
    # 2. Extract and save lineups
    lineups_data = extract_lineups(matches)
    if lineups_data:
        lineups_df = pd.DataFrame(lineups_data)
        lineups_file = raw_dir / 'lineups.csv'
        if lineups_file.exists():
            existing_lineups = pd.read_csv(lineups_file)
            # Append new lineups
            combined = pd.concat([existing_lineups, lineups_df], ignore_index=True)
            # Remove duplicates based on fixture_id and player_id
            combined = combined.drop_duplicates(subset=['fixture_id', 'player_id'], keep='last')
            combined.to_csv(lineups_file, index=False)
            logger.info(f"  ✅ Updated lineups.csv ({len(lineups_df)} new entries)")
        else:
            lineups_df.to_csv(lineups_file, index=False)
            logger.info(f"  ✅ Created lineups.csv with {len(lineups_df)} entries")
    
    # 3. Extract and save events
    events_data = extract_events(matches)
    if events_data:
        events_df = pd.DataFrame(events_data)
        events_file = raw_dir / 'events.csv'
        if events_file.exists():
            existing_events = pd.read_csv(events_file)
            combined = pd.concat([existing_events, events_df], ignore_index=True)
            # Remove duplicates based on event_id
            combined = combined.drop_duplicates(subset=['event_id'], keep='last')
            combined.to_csv(events_file, index=False)
            logger.info(f"  ✅ Updated events.csv ({len(events_data)} new entries)")
        else:
            events_df.to_csv(events_file, index=False)
            logger.info(f"  ✅ Created events.csv with {len(events_df)} entries")
    
    # 4. Extract and save sidelined
    sidelined_data = extract_sidelined(matches)
    if sidelined_data:
        sidelined_df = pd.DataFrame(sidelined_data)
        sidelined_file = raw_dir / 'sidelined.csv'
        if sidelined_file.exists():
            existing_sidelined = pd.read_csv(sidelined_file)
            combined = pd.concat([existing_sidelined, sidelined_df], ignore_index=True)
            # Remove duplicates
            combined = combined.drop_duplicates(subset=['fixture_id', 'player_id'], keep='last')
            combined.to_csv(sidelined_file, index=False)
            logger.info(f"  ✅ Updated sidelined.csv ({len(sidelined_data)} new entries)")
        else:
            sidelined_df.to_csv(sidelined_file, index=False)
            logger.info(f"  ✅ Created sidelined.csv with {len(sidelined_df)} entries")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Fetch latest match data from SportMonks")
    parser.add_argument('--days', type=int, default=7, help='Number of days to fetch (default: 7)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("FETCH LATEST DATA FROM SPORTMONKS")
    logger.info("=" * 80)
    
    # Calculate date range
    if args.start_date and args.end_date:
        start_str = args.start_date
        end_str = args.end_date
        logger.info(f"Using provided date range")
    else:
        logger.info(f"Fetching last {args.days} days of data")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_str} to {end_str}")
    
    # Fetch matches (now includes lineups, events, sidelined)
    matches = fetch_matches_by_date_range(start_str, end_str)
    
    if not matches:
        logger.warning("No matches fetched")
        return 1
    
    logger.info(f"Fetched {len(matches)} matches")
    
    # Save all data to raw directory
    success = save_to_raw_data(matches)
    
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✅ DATA FETCH COMPLETE!")
        logger.info("=" * 80)
        logger.info("Updated files in data/raw/sportmonks/:")
        logger.info("  - fixtures.csv (match info, scores, odds, league_id)")
        logger.info("  - lineups.csv (player lineups)")
        logger.info("  - events.csv (goals, cards, substitutions)")
        logger.info("  - sidelined.csv (injuries, suspensions)")
        logger.info("\nNext step: Run feature engineering")
        logger.info("  venv/bin/python 02_sportmonks_feature_engineering.py")
        return 0
    else:
        logger.error("\n❌ Data fetch failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

