#!/usr/bin/env python3
"""
Build In-Game Training Dataset
Reconstructs match state at different minutes from completed matches
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SPORTMONKS_API_KEY = os.getenv("SPORTMONKS_API_KEY", "DQQStChRaPnjIryuZH2SxqJI5ufoA57wWsmFIuPCH2rvlBtm0G7Ch3mJoyE4")
BASE_URL = "https://api.sportmonks.com/v3/football"
SAMPLE_MINUTES = [0, 15, 30, 45, 60, 75, 90]  # Minutes to sample

def get_match_events(fixture_id):
    """Fetch match events from SportMonks API"""
    try:
        url = f"{BASE_URL}/fixtures/{fixture_id}"
        params = {
            'api_token': SPORTMONKS_API_KEY,
            'include': 'events;scores;participants;state'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get('data')
        
    except Exception as e:
        logger.error(f"Error fetching fixture {fixture_id}: {e}")
        return None

def get_final_result(match_data):
    """Extract final result from match data"""
    scores = match_data.get('scores', [])
    
    home_goals = 0
    away_goals = 0
    
    for score in scores:
        if score.get('description') == 'CURRENT':
            if score['score']['participant'] == 'home':
                home_goals = score['score']['goals']
            else:
                away_goals = score['score']['goals']
    
    if home_goals > away_goals:
        return 'HOME', home_goals, away_goals
    elif away_goals > home_goals:
        return 'AWAY', home_goals, away_goals
    else:
        return 'DRAW', home_goals, away_goals

def reconstruct_timeline(match_data):
    """Reconstruct match state at each minute from events"""
    
    events = match_data.get('events', [])
    participants = match_data.get('participants', [])
    
    if len(participants) < 2:
        return None
    
    home_team_id = participants[0]['id']
    away_team_id = participants[1]['id']
    
    # Initialize timeline for 95 minutes (including injury time)
    timeline = []
    
    for minute in range(96):
        state = {
            'minute': minute,
            'home_goals': 0,
            'away_goals': 0,
            'home_red_cards': 0,
            'away_red_cards': 0,
            'home_yellow_cards': 0,
            'away_yellow_cards': 0,
            'total_goals': 0
        }
        
        # Count events up to this minute
        for event in events:
            event_minute = event.get('minute', 0)
            
            # Only count events that happened before or at this minute
            if event_minute <= minute:
                participant_id = event.get('participant_id')
                type_id = event.get('type_id')
                
                # Goals (type_id = 14)
                if type_id == 14:
                    if participant_id == home_team_id:
                        state['home_goals'] += 1
                    elif participant_id == away_team_id:
                        state['away_goals'] += 1
                    state['total_goals'] += 1
                
                # Red cards (type_id = 18)
                elif type_id == 18:
                    if participant_id == home_team_id:
                        state['home_red_cards'] += 1
                    elif participant_id == away_team_id:
                        state['away_red_cards'] += 1
                
                # Yellow cards (type_id = 17)
                elif type_id == 17:
                    if participant_id == home_team_id:
                        state['home_yellow_cards'] += 1
                    elif participant_id == away_team_id:
                        state['away_yellow_cards'] += 1
        
        timeline.append(state)
    
    return timeline

def create_training_samples(match_data, sample_minutes=SAMPLE_MINUTES):
    """Create training samples from a match"""
    
    # Get final result
    final_result, final_home, final_away = get_final_result(match_data)
    
    # Reconstruct timeline
    timeline = reconstruct_timeline(match_data)
    
    if timeline is None:
        return []
    
    # Get match info
    fixture_id = match_data.get('id')
    match_name = match_data.get('name', 'Unknown')
    
    # Sample at specific minutes
    samples = []
    
    for minute in sample_minutes:
        if minute < len(timeline):
            state = timeline[minute]
            
            sample = {
                # Match info
                'fixture_id': fixture_id,
                'match_name': match_name,
                'sample_minute': minute,
                
                # Current state at this minute
                'home_goals': state['home_goals'],
                'away_goals': state['away_goals'],
                'score_diff': state['home_goals'] - state['away_goals'],
                'total_goals': state['total_goals'],
                
                # Cards
                'home_red_cards': state['home_red_cards'],
                'away_red_cards': state['away_red_cards'],
                'home_yellow_cards': state['home_yellow_cards'],
                'away_yellow_cards': state['away_yellow_cards'],
                
                # Derived features
                'time_remaining': 90 - minute,
                'is_second_half': 1 if minute > 45 else 0,
                'is_late_game': 1 if minute > 75 else 0,
                'player_advantage': (11 - state['home_red_cards']) - (11 - state['away_red_cards']),
                
                # Match state
                'is_home_leading': 1 if state['home_goals'] > state['away_goals'] else 0,
                'is_away_leading': 1 if state['away_goals'] > state['home_goals'] else 0,
                'is_draw': 1 if state['home_goals'] == state['away_goals'] else 0,
                
                # Final result (target)
                'final_result': final_result,
                'final_home_goals': final_home,
                'final_away_goals': final_away,
                'target': 0 if final_result == 'AWAY' else (1 if final_result == 'DRAW' else 2)
            }
            
            samples.append(sample)
    
    return samples

def build_dataset(max_matches=1000, start_from=0):
    """Build complete in-game training dataset"""
    
    logger.info("="*80)
    logger.info("BUILDING IN-GAME TRAINING DATASET")
    logger.info("="*80)
    
    # Load completed fixtures
    fixtures_path = Path('data/raw/sportmonks/fixtures.csv')
    
    if not fixtures_path.exists():
        logger.error(f"Fixtures file not found: {fixtures_path}")
        return None
    
    logger.info(f"Loading fixtures from: {fixtures_path}")
    fixtures = pd.read_csv(fixtures_path)
    
    # Filter completed matches (state_id = 5 means Full Time)
    completed = fixtures[fixtures['state_id'] == 5].copy()
    logger.info(f"Found {len(completed)} completed matches")
    
    # Limit number of matches
    if max_matches:
        completed = completed.iloc[start_from:start_from + max_matches]
        logger.info(f"Processing {len(completed)} matches (from index {start_from})")
    
    # Build samples
    all_samples = []
    failed_count = 0
    
    logger.info(f"\nSampling at minutes: {SAMPLE_MINUTES}")
    logger.info(f"Expected samples per match: {len(SAMPLE_MINUTES)}")
    logger.info("")
    
    for idx, row in tqdm(completed.iterrows(), total=len(completed), desc="Processing matches"):
        fixture_id = row['fixture_id']
        
        try:
            # Fetch match data
            match_data = get_match_events(fixture_id)
            
            if match_data is None:
                failed_count += 1
                continue
            
            # Create samples
            samples = create_training_samples(match_data)
            
            if samples:
                all_samples.extend(samples)
            else:
                failed_count += 1
            
            # Rate limiting
            time.sleep(0.1)  # 10 requests per second
            
        except Exception as e:
            logger.error(f"Error processing fixture {fixture_id}: {e}")
            failed_count += 1
            continue
    
    # Convert to DataFrame
    if all_samples:
        df = pd.DataFrame(all_samples)
        
        logger.info("")
        logger.info("="*80)
        logger.info("DATASET SUMMARY")
        logger.info("="*80)
        logger.info(f"Total matches processed: {len(completed)}")
        logger.info(f"Failed matches: {failed_count}")
        logger.info(f"Successful matches: {len(completed) - failed_count}")
        logger.info(f"Total samples created: {len(df)}")
        logger.info(f"Samples per match: {len(df) / (len(completed) - failed_count):.1f}")
        logger.info("")
        logger.info("Target distribution:")
        logger.info(f"  HOME: {(df['target'] == 2).sum()} ({(df['target'] == 2).sum()/len(df)*100:.1f}%)")
        logger.info(f"  DRAW: {(df['target'] == 1).sum()} ({(df['target'] == 1).sum()/len(df)*100:.1f}%)")
        logger.info(f"  AWAY: {(df['target'] == 0).sum()} ({(df['target'] == 0).sum()/len(df)*100:.1f}%)")
        logger.info("")
        
        return df
    else:
        logger.error("No samples created!")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build in-game training dataset")
    parser.add_argument('--max-matches', type=int, default=1000, help='Maximum matches to process (default: 1000)')
    parser.add_argument('--start-from', type=int, default=0, help='Start from match index (default: 0)')
    parser.add_argument('--output', type=str, default='data/processed/in_game_training.csv', help='Output file path')
    
    args = parser.parse_args()
    
    # Build dataset
    df = build_dataset(max_matches=args.max_matches, start_from=args.start_from)
    
    if df is not None:
        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Dataset saved to: {output_path}")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")
        logger.info("")
        logger.info("="*80)
        logger.info("NEXT STEPS")
        logger.info("="*80)
        logger.info("1. Add pre-match features (Elo, form, etc.)")
        logger.info("2. Train in-game model")
        logger.info("3. Validate on held-out data")
        logger.info("="*80)
    else:
        logger.error("Failed to build dataset")
        sys.exit(1)

if __name__ == '__main__':
    main()
