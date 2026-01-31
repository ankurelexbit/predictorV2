"""
Convert JSON fixtures to comprehensive CSV with ALL statistics.

This extracts everything needed for feature generation so we never need to touch JSON again.
"""
import sys
from pathlib import Path
import logging
import pandas as pd
import json
from tqdm import tqdm
import ijson

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_lineup_data(fixture: dict) -> dict:
    """Extract lineup summary from fixture."""
    lineup_dict = {}
    
    # Get lineups array
    lineups = fixture.get('lineups', [])
    
    # Count players for each team
    home_lineup_count = 0
    away_lineup_count = 0
    home_avg_rating = None
    away_avg_rating = None
    
    for lineup in lineups:
        team_id = lineup.get('team_id')
        players = lineup.get('players', [])
        
        # Determine if home or away
        participants = fixture.get('participants', [])
        is_home = False
        for p in participants:
            if p.get('id') == team_id and p.get('meta', {}).get('location') == 'home':
                is_home = True
                break
        
        if is_home:
            home_lineup_count = len(players)
            # Could extract average rating if available
        else:
            away_lineup_count = len(players)
    
    lineup_dict['home_lineup_count'] = home_lineup_count
    lineup_dict['away_lineup_count'] = away_lineup_count
    lineup_dict['home_avg_rating'] = home_avg_rating
    lineup_dict['away_avg_rating'] = away_avg_rating
    
    return lineup_dict


def extract_statistics_from_fixture(fixture: dict) -> dict:
    """Extract all statistics from a fixture into flat dictionary."""
    stats_dict = {}
    
    # Get statistics array
    statistics = fixture.get('statistics', [])
    
    # Map of type_id to stat name (common SportMonks type IDs)
    stat_type_map = {
        52: 'shots_total',
        53: 'shots_on_target',
        54: 'shots_off_target',
        55: 'shots_blocked',
        56: 'shots_inside_box',
        57: 'shots_outside_box',
        84: 'fouls',
        80: 'corners',
        81: 'offsides',
        82: 'ball_possession',
        83: 'yellow_cards',
        86: 'red_cards',
        45: 'saves',
        78: 'tackles',
        79: 'interceptions',
        98: 'clearances',
        41: 'passes_total',
        42: 'passes_accurate',
        43: 'passes_percentage',
        44: 'attacks',
        45: 'dangerous_attacks',
    }
    
    # Initialize all stats to None
    for side in ['home', 'away']:
        for stat_name in stat_type_map.values():
            stats_dict[f'{side}_{stat_name}'] = None
    
    # Extract statistics
    for stat in statistics:
        type_id = stat.get('type_id')
        location = stat.get('location', '').lower()  # 'home' or 'away'
        value = stat.get('data', {}).get('value')
        
        if type_id in stat_type_map and location in ['home', 'away']:
            stat_name = stat_type_map[type_id]
            stats_dict[f'{location}_{stat_name}'] = value
    
    return stats_dict


def main():
    """Convert JSON fixtures to comprehensive CSV with statistics."""
    logger.info("\n" + "=" * 80)
    logger.info("CONVERTING JSON TO COMPREHENSIVE CSV")
    logger.info("=" * 80 + "\n")
    
    data_dir = Path('data/historical')
    fixtures_dir = data_dir / 'fixtures'
    
    # Find all fixture JSON files
    fixture_files = sorted(fixtures_dir.glob('*.json'))
    logger.info(f"Found {len(fixture_files)} JSON files")
    
    all_fixtures = []
    
    logger.info("\nExtracting fixtures and statistics...")
    for file_path in tqdm(fixture_files, desc="Processing files"):
        try:
            with open(file_path, 'rb') as f:
                # Stream parse the JSON array
                parser = ijson.items(f, 'item')
                
                for fixture in parser:
                    # Extract basic fixture info
                    participants = fixture.get('participants', [])
                    if len(participants) != 2:
                        continue
                    
                    home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                    away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Get scores
                    scores = fixture.get('scores', [])
                    home_score = None
                    away_score = None
                    
                    for score in scores:
                        if score.get('description') == 'CURRENT':
                            participant = score.get('score', {}).get('participant')
                            goals = score.get('score', {}).get('goals')
                            
                            if participant == 'home':
                                home_score = goals
                            elif participant == 'away':
                                away_score = goals
                    
                    # Determine result
                    result = None
                    if home_score is not None and away_score is not None:
                        if home_score > away_score:
                            result = 'H'
                        elif home_score < away_score:
                            result = 'A'
                        else:
                            result = 'D'
                    
                    # Extract statistics
                    stats = extract_statistics_from_fixture(fixture)
                    
                    # Extract lineup data
                    lineup_data = extract_lineup_data(fixture)
                    
                    # Combine all data
                    fixture_data = {
                        'id': fixture.get('id'),
                        'league_id': fixture.get('league_id'),
                        'season_id': fixture.get('season_id'),
                        'starting_at': fixture.get('starting_at'),
                        'home_team_id': home_team.get('id'),
                        'home_team_name': home_team.get('name'),
                        'away_team_id': away_team.get('id'),
                        'away_team_name': away_team.get('name'),
                        'home_score': home_score,
                        'away_score': away_score,
                        'result': result,
                        'state_id': fixture.get('state_id'),
                        **stats,  # Add all statistics
                        **lineup_data  # Add lineup data
                    }
                    
                    all_fixtures.append(fixture_data)
                    
        except Exception as e:
            logger.warning(f"Failed to parse {file_path.name}: {e}")
            continue
    
    logger.info(f"\n✅ Extracted {len(all_fixtures)} fixtures with statistics")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_fixtures)
    
    # Parse dates
    df['starting_at'] = pd.to_datetime(df['starting_at'])
    
    # Sort by date
    df = df.sort_values('starting_at').reset_index(drop=True)
    
    logger.info(f"   Date range: {df['starting_at'].min()} to {df['starting_at'].max()}")
    logger.info(f"   Total columns: {len(df.columns)}")
    
    # Save to CSV
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'fixtures_with_stats.csv'
    
    logger.info(f"\n💾 Saving to CSV: {csv_file}")
    df.to_csv(csv_file, index=False)
    
    # Show file size
    csv_size = csv_file.stat().st_size / (1024 * 1024)
    logger.info(f"   File size: {csv_size:.1f} MB")
    
    # Show sample statistics coverage
    stat_columns = [col for col in df.columns if col.startswith(('home_', 'away_')) and col not in ['home_team_id', 'home_team_name', 'home_score', 'away_team_id', 'away_team_name', 'away_score']]
    logger.info(f"\n📊 Statistics extracted: {len(stat_columns)} stat columns")
    logger.info(f"   Sample stats: {', '.join(stat_columns[:5])}...")
    
    # Show coverage
    for col in stat_columns[:5]:
        coverage = (df[col].notna().sum() / len(df)) * 100
        logger.info(f"   {col}: {coverage:.1f}% coverage")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ CONVERSION COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nNext: Update feature generation to use this CSV!")
    logger.info(f"CSV file: {csv_file}\n")


if __name__ == '__main__':
    main()
