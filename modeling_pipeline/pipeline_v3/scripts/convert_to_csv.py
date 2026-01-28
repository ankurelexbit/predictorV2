#!/usr/bin/env python3
"""
Convert JSON historical data to CSV format for fast feature engineering.

This script processes the downloaded JSON files and creates structured CSVs:
- fixtures.csv: Core match data
- statistics.csv: Team statistics per match
- lineups.csv: Player lineups per match
- events.csv: Match events (goals, cards, subs)
- sidelined.csv: Injuries/suspensions

Usage:
    python scripts/convert_to_csv.py
"""



import json
import ijson
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def append_to_csv(data: list, output_file: Path, first_batch: bool):
    """Append a list of dicts to CSV."""
    if not data:
        return
    
    df = pd.DataFrame(data)
    mode = 'w' if first_batch else 'a'
    header = first_batch
    df.to_csv(output_file, mode=mode, header=header, index=False)

def convert_fixtures_to_csv(data_dir: Path, output_dir: Path):
    """Convert fixture JSON files to CSV using streaming (ijson)."""
    logger.info("Converting fixtures to CSV (Streaming with ijson)...")
    
    fixtures_dir = data_dir / 'fixtures'
    fixture_files = sorted(fixtures_dir.glob('all_fixtures_*.json'))
    output_file = output_dir / 'fixtures.csv'
    
    count = 0
    first_batch = True
    
    for fixture_file in tqdm(fixture_files, desc="Processing fixtures"):
        try:
            # Use ijson for memory-efficient streaming
            with open(fixture_file, 'rb') as f:
                # 'item' iterates over items in the top-level list
                fixtures_stream = ijson.items(f, 'item')
                
                batch_data = []
                for fixture in fixtures_stream:
                    fixture_data = {
                        'fixture_id': fixture.get('id'),
                        'league_id': fixture.get('league_id'),
                        'season_id': fixture.get('season_id'),
                        'stage_id': fixture.get('stage_id'),
                        'round_id': fixture.get('round_id'),
                        'starting_at': fixture.get('starting_at'),
                        'starting_at_timestamp': fixture.get('starting_at_timestamp'),
                        'venue_id': fixture.get('venue_id'),
                        'state_id': fixture.get('state_id'),
                    }
                    
                    participants = fixture.get('participants', [])
                    if len(participants) >= 2:
                        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
                        if home_team:
                            fixture_data['home_team_id'] = home_team.get('id')
                            fixture_data['home_team_name'] = home_team.get('name')
                        if away_team:
                            fixture_data['away_team_id'] = away_team.get('id')
                            fixture_data['away_team_name'] = away_team.get('name')
                    
                    scores = fixture.get('scores', [])
                    for score in scores:
                        description = score.get('description', '').lower()
                        if 'current' in description or 'ft' in description:
                            participant_id = score.get('participant_id')
                            goals = score.get('score', {}).get('goals', 0)
                            if participant_id == fixture_data.get('home_team_id'):
                                fixture_data['home_score'] = goals
                            elif participant_id == fixture_data.get('away_team_id'):
                                fixture_data['away_score'] = goals
                    
                    if 'home_score' in fixture_data and 'away_score' in fixture_data:
                        home_score = fixture_data['home_score']
                        away_score = fixture_data['away_score']
                        if home_score > away_score:
                            fixture_data['result'] = 'H'
                        elif home_score < away_score:
                            fixture_data['result'] = 'A'
                        else:
                            fixture_data['result'] = 'D'
                    
                    state = fixture.get('state', {})
                    fixture_data['state'] = state.get('short_name', '')
                    
                    batch_data.append(fixture_data)
                    
                    # Write batch every 1000 items to keep memory low
                    if len(batch_data) >= 1000:
                        append_to_csv(batch_data, output_file, first_batch)
                        count += len(batch_data)
                        first_batch = False
                        batch_data = []
                
                # Write remaining
                if batch_data:
                    append_to_csv(batch_data, output_file, first_batch)
                    count += len(batch_data)
                    first_batch = False
                    
        except Exception as e:
            logger.error(f"Error processing {fixture_file.name}: {e}")
            continue
            
    logger.info(f"✅ Saved {count} fixtures to {output_file}")


def convert_statistics_to_csv(data_dir: Path, output_dir: Path):
    """Convert statistics from fixture JSON files to CSV using streaming (ijson)."""
    logger.info("Converting statistics to CSV (Streaming with ijson)...")
    
    fixtures_dir = data_dir / 'fixtures'
    fixture_files = sorted(fixtures_dir.glob('all_fixtures_*.json'))
    output_file = output_dir / 'statistics.csv'
    
    stat_type_names = {
        41: 'shots_total', 42: 'shots_on_target', 43: 'shots_off_target',
        44: 'shots_blocked', 45: 'shots_inside_box', 46: 'shots_outside_box',
        47: 'fouls', 48: 'corners', 49: 'offsides', 50: 'possession',
        51: 'yellow_cards', 52: 'red_cards', 53: 'saves',
        54: 'substitutions', 55: 'goal_kicks', 56: 'goal_attempts',
        57: 'free_kicks', 58: 'throw_ins', 59: 'ball_safe',
        60: 'goals', 61: 'penalties', 62: 'injuries',
        63: 'tackles', 64: 'attacks', 65: 'dangerous_attacks',
        80: 'passes_total', 81: 'passes_accurate', 82: 'passes_percentage',
        83: 'hit_woodwork', 84: 'goalkeeper_saves', 85: 'goal_line_clearances',
        86: 'interceptions', 87: 'clearances', 88: 'dispossessed',
        89: 'dribbles', 90: 'dribbles_attempts', 91: 'dribbles_success',
        92: 'duels', 93: 'duels_won', 94: 'aerials_won',
    }
    
    count = 0
    first_batch = True
    
    for fixture_file in tqdm(fixture_files, desc="Extracting stats"):
        try:
            with open(fixture_file, 'rb') as f:
                fixtures_stream = ijson.items(f, 'item')
                
                batch_data = []
                for fixture in fixtures_stream:
                    fixture_id = fixture.get('id')
                    stats_list = fixture.get('statistics', [])
                    if not stats_list: continue
                    
                    team_stats_dict = {}
                    for stat in stats_list:
                        participant_id = stat.get('participant_id')
                        type_id = stat.get('type_id')
                        value = stat.get('data', {}).get('value')
                        location = stat.get('location', '')
                        
                        if participant_id not in team_stats_dict:
                            team_stats_dict[participant_id] = {
                                'fixture_id': fixture_id,
                                'team_id': participant_id,
                                'is_home': location == 'home'
                            }
                        
                        stat_name = stat_type_names.get(type_id, f'stat_{type_id}')
                        try:
                            if value is not None:
                                clean_val = str(value).rstrip('%')
                                team_stats_dict[participant_id][stat_name] = float(clean_val) if '.' in clean_val else int(clean_val)
                        except:
                            team_stats_dict[participant_id][stat_name] = value

                    batch_data.extend(team_stats_dict.values())

                    if len(batch_data) >= 1000:
                        append_to_csv(batch_data, output_file, first_batch)
                        count += len(batch_data)
                        first_batch = False
                        batch_data = []
                
                if batch_data:
                    append_to_csv(batch_data, output_file, first_batch)
                    count += len(batch_data)
                    first_batch = False
                    
        except Exception as e:
            logger.error(f"Error {fixture_file.name}: {e}")
            continue

    logger.info(f"✅ Saved {count} statistics rows to {output_file}")


def convert_lineups_to_csv(data_dir: Path, output_dir: Path):
    """Convert lineups from fixture JSON files to CSV using streaming (ijson)."""
    logger.info("Converting lineups to CSV (Streaming with ijson)...")
    
    fixtures_dir = data_dir / 'fixtures'
    fixture_files = sorted(fixtures_dir.glob('all_fixtures_*.json'))
    output_file = output_dir / 'lineups.csv'
    
    detail_type_names = {
        88: 'rating', 89: 'minutes_played', 90: 'goals', 91: 'assists',
        92: 'yellow_cards', 93: 'red_cards', 94: 'shots_total',
        95: 'shots_on_target', 96: 'passes_total', 97: 'passes_accurate',
        98: 'tackles', 99: 'interceptions', 100: 'clearances',
        101: 'offsides', 102: 'saves', 103: 'duels_total',
        104: 'duels_won', 105: 'dribbles_attempts', 106: 'dribbles_success',
        107: 'fouls_drawn', 108: 'fouls_committed', 109: 'penalties_won',
        110: 'penalties_committed', 111: 'hit_woodwork',
    }
    
    count = 0
    first_batch = True
    
    for fixture_file in tqdm(fixture_files, desc="Extracting lineups"):
        try:
            with open(fixture_file, 'rb') as f:
                fixtures_stream = ijson.items(f, 'item')
                
                batch_data = []
                for fixture in fixtures_stream:
                    fixture_id = fixture.get('id')
                    lineups_list = fixture.get('lineups', [])
                    if not lineups_list: continue
                    
                    for player in lineups_list:
                        player_data = {
                            'fixture_id': fixture_id,
                            'team_id': player.get('team_id'),
                            'player_id': player.get('player_id'),
                            'player_name': player.get('player_name'),
                            'position_id': player.get('position_id'),
                            'is_starter': player.get('type_id') == 11
                        }
                        
                        details = player.get('details', [])
                        if details:
                            for detail in details:
                                type_id = detail.get('type_id')
                                value = detail.get('data', {}).get('value')
                                stat_name = detail_type_names.get(type_id, f'detail_{type_id}')
                                try:
                                    if value is not None:
                                        clean_val = str(value).rstrip('%')
                                        player_data[stat_name] = float(clean_val) if '.' in clean_val else int(clean_val)
                                except:
                                    player_data[stat_name] = value
                        
                        batch_data.append(player_data)

                    if len(batch_data) >= 1000:
                        append_to_csv(batch_data, output_file, first_batch)
                        count += len(batch_data)
                        first_batch = False
                        batch_data = []
                
                if batch_data:
                    append_to_csv(batch_data, output_file, first_batch)
                    count += len(batch_data)
                    first_batch = False
                    
        except Exception as e:
            logger.error(f"Error {fixture_file.name}: {e}")
            continue

    logger.info(f"✅ Saved {count} lineup rows to {output_file}")


def convert_sidelined_to_csv(data_dir: Path, output_dir: Path):
    """Convert sidelined from fixture JSON files to CSV using streaming (ijson)."""
    logger.info("Converting sidelined data to CSV (Streaming with ijson)...")
    
    fixtures_dir = data_dir / 'fixtures'
    fixture_files = sorted(fixtures_dir.glob('all_fixtures_*.json'))
    output_file = output_dir / 'sidelined.csv'
    
    count = 0
    first_batch = True
    
    for fixture_file in tqdm(fixture_files, desc="Extracting sidelined"):
        try:
            with open(fixture_file, 'rb') as f:
                fixtures_stream = ijson.items(f, 'item')
                
                batch_data = []
                for fixture in fixtures_stream:
                    fixture_id = fixture.get('id')
                    sidelined_list = fixture.get('sidelined', [])
                    if not sidelined_list: continue
                    
                    for sidelined in sidelined_list:
                        sidelined_data = {
                            'fixture_id': fixture_id,
                            'team_id': sidelined.get('team_id'),
                            'player_id': sidelined.get('player_id'),
                            'reason': sidelined.get('reason'),
                            'start_date': sidelined.get('start_date'),
                            'end_date': sidelined.get('end_date')
                        }
                        batch_data.append(sidelined_data)
                        
                    if len(batch_data) >= 1000:
                        append_to_csv(batch_data, output_file, first_batch)
                        count += len(batch_data)
                        first_batch = False
                        batch_data = []
                
                if batch_data:
                    append_to_csv(batch_data, output_file, first_batch)
                    count += len(batch_data)
                    first_batch = False
                    
        except Exception as e:
            logger.error(f"Error {fixture_file.name}: {e}")
            continue
            
    logger.info(f"✅ Saved {count} sidelined rows to {output_file}")


def main():
    logger.info("=" * 80)
    logger.info("JSON TO CSV STREAMING CONVERTER (IJSON)")
    logger.info("=" * 80)
    
    try:
        import ijson
    except ImportError:
        logger.error("❌ 'ijson' library not found!")
        logger.error("Please run: pip install ijson")
        return
        
    data_dir = Path('data/historical')
    output_dir = Path('data/csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old CSVs to ensure clean write
    for f in ['fixtures.csv', 'statistics.csv', 'lineups.csv', 'sidelined.csv']:
        path = output_dir / f
        if path.exists():
            path.unlink()
    
    convert_fixtures_to_csv(data_dir, output_dir)
    convert_statistics_to_csv(data_dir, output_dir)
    convert_lineups_to_csv(data_dir, output_dir)
    convert_sidelined_to_csv(data_dir, output_dir)
    
    logger.info("\nCONVERSION COMPLETE")

if __name__ == "__main__":
    main()
