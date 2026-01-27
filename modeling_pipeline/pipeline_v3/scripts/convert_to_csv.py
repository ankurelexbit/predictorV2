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
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_fixtures_to_csv(data_dir: Path, output_dir: Path):
    """Convert fixture JSON files to CSV."""
    logger.info("Converting fixtures to CSV...")
    
    fixtures_dir = data_dir / 'fixtures'
    fixture_files = sorted(fixtures_dir.glob('all_fixtures_*.json'))
    
    all_fixtures = []
    
    for fixture_file in tqdm(fixture_files, desc="Processing fixture files"):
        try:
            with open(fixture_file) as f:
                fixtures = json.load(f)
                
                for fixture in fixtures:
                    # Extract core fixture data
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
                    
                    # Extract participants (home/away teams)
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
                    
                    # Extract scores
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
                    
                    # Calculate result (H/D/A from home team perspective)
                    if 'home_score' in fixture_data and 'away_score' in fixture_data:
                        home_score = fixture_data['home_score']
                        away_score = fixture_data['away_score']
                        
                        if home_score > away_score:
                            fixture_data['result'] = 'H'
                        elif home_score < away_score:
                            fixture_data['result'] = 'A'
                        else:
                            fixture_data['result'] = 'D'
                    
                    # State
                    state = fixture.get('state', {})
                    fixture_data['state'] = state.get('short_name', '')
                    
                    all_fixtures.append(fixture_data)
        
        except Exception as e:
            logger.error(f"Error processing {fixture_file.name}: {e}")
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame(all_fixtures)
    output_file = output_dir / 'fixtures.csv'
    df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Saved {len(df)} fixtures to {output_file}")
    return df


def convert_statistics_to_csv(data_dir: Path, output_dir: Path):
    """Convert statistics JSON files to CSV."""
    logger.info("Converting statistics to CSV...")
    
    stats_dir = data_dir / 'statistics'
    stats_files = list(stats_dir.glob('fixture_*.json'))
    
    # Common stat type_id mappings (from SportMonks API)
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
    
    all_team_stats = []
    
    for stats_file in tqdm(stats_files, desc="Processing statistics files"):
        try:
            fixture_id = int(stats_file.stem.split('_')[1])
            
            with open(stats_file) as f:
                stats_list = json.load(f)
                
                # Group statistics by participant_id (team)
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
                            'location': location,
                        }
                    
                    # Map type_id to stat name
                    stat_name = stat_type_names.get(type_id, f'stat_{type_id}')
                    
                    # Convert value to numeric if possible
                    try:
                        if value is not None:
                            if isinstance(value, (int, float)):
                                team_stats_dict[participant_id][stat_name] = value
                            else:
                                team_stats_dict[participant_id][stat_name] = float(value) if '.' in str(value) else int(value)
                    except (ValueError, TypeError):
                        team_stats_dict[participant_id][stat_name] = value
                
                # Add all teams for this fixture
                all_team_stats.extend(team_stats_dict.values())
        
        except Exception as e:
            logger.error(f"Error processing {stats_file.name}: {e}")
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame(all_team_stats)
    
    # Add is_home flag based on location
    if 'location' in df.columns:
        df['is_home'] = df['location'] == 'home'
        df = df.drop('location', axis=1)
    
    output_file = output_dir / 'statistics.csv'
    df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Saved {len(df)} team statistics to {output_file}")
    return df


def convert_lineups_to_csv(data_dir: Path, output_dir: Path):
    """Convert lineups JSON files to CSV."""
    logger.info("Converting lineups to CSV...")
    
    lineups_dir = data_dir / 'lineups'
    lineup_files = list(lineups_dir.glob('fixture_*.json'))
    
    # Player detail type_id mappings
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
    
    all_lineups = []
    
    for lineup_file in tqdm(lineup_files, desc="Processing lineup files"):
        try:
            fixture_id = int(lineup_file.stem.split('_')[1])
            
            with open(lineup_file) as f:
                lineups_list = json.load(f)
                
                for player in lineups_list:
                    player_data = {
                        'fixture_id': fixture_id,
                        'team_id': player.get('team_id'),
                        'player_id': player.get('player_id'),
                        'player_name': player.get('player_name'),
                        'position_id': player.get('position_id'),
                        'formation_position': player.get('formation_position'),
                        'jersey_number': player.get('jersey_number'),
                        'type_id': player.get('type_id'),
                    }
                    
                    # Determine if starter (type_id 11 = starting, 12 = bench)
                    player_data['is_starter'] = player.get('type_id') == 11
                    
                    # Extract detailed stats from details array
                    details = player.get('details', [])
                    if details:
                        for detail in details:
                            type_id = detail.get('type_id')
                            value = detail.get('data', {}).get('value')
                            
                            # Map type_id to stat name
                            stat_name = detail_type_names.get(type_id, f'detail_{type_id}')
                            
                            # Convert value to numeric if possible
                            try:
                                if value is not None:
                                    if isinstance(value, (int, float)):
                                        player_data[stat_name] = value
                                    else:
                                        player_data[stat_name] = float(value) if '.' in str(value) else int(value)
                            except (ValueError, TypeError):
                                player_data[stat_name] = value
                    
                    all_lineups.append(player_data)
        
        except Exception as e:
            logger.error(f"Error processing {lineup_file.name}: {e}")
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame(all_lineups)
    output_file = output_dir / 'lineups.csv'
    df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Saved {len(df)} player lineups to {output_file}")
    return df


def convert_sidelined_to_csv(data_dir: Path, output_dir: Path):
    """Convert sidelined JSON files to CSV."""
    logger.info("Converting sidelined data to CSV...")
    
    sidelined_dir = data_dir / 'sidelined'
    sidelined_files = list(sidelined_dir.glob('team_*.json'))
    
    all_sidelined = []
    
    for sidelined_file in tqdm(sidelined_files, desc="Processing sidelined files"):
        try:
            team_id = int(sidelined_file.stem.split('_')[1])
            
            with open(sidelined_file) as f:
                sidelined_list = json.load(f)
                
                for sidelined in sidelined_list:
                    sidelined_data = {
                        'team_id': team_id,
                        'player_id': sidelined.get('player_id'),
                        'player_name': sidelined.get('player_name'),
                        'start_date': sidelined.get('start_date'),
                        'end_date': sidelined.get('end_date'),
                        'category': sidelined.get('category'),
                        'reason': sidelined.get('reason'),
                    }
                    
                    all_sidelined.append(sidelined_data)
        
        except Exception as e:
            logger.error(f"Error processing {sidelined_file.name}: {e}")
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame(all_sidelined)
    output_file = output_dir / 'sidelined.csv'
    df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Saved {len(df)} sidelined records to {output_file}")
    return df


def main():
    """Main conversion process."""
    logger.info("=" * 80)
    logger.info("JSON TO CSV CONVERSION")
    logger.info("=" * 80)
    
    # Paths
    data_dir = Path('data/historical')
    output_dir = Path('data/csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each data type
    fixtures_df = convert_fixtures_to_csv(data_dir, output_dir)
    statistics_df = convert_statistics_to_csv(data_dir, output_dir)
    lineups_df = convert_lineups_to_csv(data_dir, output_dir)
    sidelined_df = convert_sidelined_to_csv(data_dir, output_dir)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Fixtures:    {len(fixtures_df):,} rows")
    logger.info(f"Statistics:  {len(statistics_df):,} rows")
    logger.info(f"Lineups:     {len(lineups_df):,} rows")
    logger.info(f"Sidelined:   {len(sidelined_df):,} rows")
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
