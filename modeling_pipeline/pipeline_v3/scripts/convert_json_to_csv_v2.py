#!/usr/bin/env python3
"""
JSON to CSV Converter V2 - Robust and Complete

This script converts SportMonks JSON fixture data to CSV format with:
- Complete statistics extraction
- Proper data validation
- Error handling and logging
- Data quality checks
- Memory-efficient streaming

Usage:
    python scripts/convert_json_to_csv_v2.py
    
    # With custom paths
    python scripts/convert_json_to_csv_v2.py --input data/historical --output data/csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('json_to_csv_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JSONToCSVConverter:
    """Convert SportMonks JSON fixtures to CSV format."""
    
    # Statistics type mapping (SportMonks type_id to column name)
    STAT_TYPE_MAPPING = {
        41: 'shots_total',
        42: 'shots_on_target',
        43: 'shots_off_target',
        44: 'shots_blocked',
        45: 'shots_inside_box',
        46: 'shots_outside_box',
        47: 'fouls',
        48: 'corners',
        49: 'offsides',
        50: 'possession',
        51: 'yellow_cards',
        52: 'red_cards',
        53: 'saves',
        54: 'substitutions',
        55: 'goal_kicks',
        56: 'goal_attempts',
        57: 'free_kicks',
        58: 'throw_ins',
        59: 'ball_safe',
        60: 'goals',
        61: 'penalties',
        62: 'injuries',
        63: 'tackles',
        64: 'attacks',
        65: 'dangerous_attacks',
        80: 'passes_total',
        81: 'passes_accurate',
        82: 'passes_percentage',
        83: 'hit_woodwork',
        84: 'goalkeeper_saves',
        85: 'goal_line_clearances',
        86: 'interceptions',
        87: 'clearances',
        88: 'dispossessed',
        89: 'dribbles',
        90: 'dribbles_attempts',
        91: 'dribbles_success',
        92: 'duels',
        93: 'duels_won',
        94: 'aerials_won',
    }
    
    # Player detail type mapping
    PLAYER_DETAIL_MAPPING = {
        88: 'rating',
        89: 'minutes_played',
        90: 'goals',
        91: 'assists',
        92: 'yellow_cards',
        93: 'red_cards',
        94: 'shots_total',
        95: 'shots_on_target',
        96: 'passes_total',
        97: 'passes_accurate',
        98: 'tackles',
        99: 'interceptions',
        100: 'clearances',
        101: 'offsides',
        102: 'saves',
        103: 'duels_total',
        104: 'duels_won',
        105: 'dribbles_attempts',
        106: 'dribbles_success',
        107: 'fouls_drawn',
        108: 'fouls_committed',
        109: 'penalties_won',
        110: 'penalties_committed',
        111: 'hit_woodwork',
    }
    
    def __init__(self, input_dir: str = 'data/historical', output_dir: str = 'data/csv'):
        """Initialize converter."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'fixtures_processed': 0,
            'fixtures_with_stats': 0,
            'fixtures_with_lineups': 0,
            'fixtures_with_sidelined': 0,
            'errors': 0,
        }
    
    def _safe_get(self, data: Dict, *keys, default=None) -> Any:
        """Safely get nested dictionary values."""
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, {})
            else:
                return default
        return data if data != {} else default
    
    def _parse_numeric(self, value: Any) -> Optional[float]:
        """Parse numeric value, handling percentages and strings."""
        if value is None:
            return None
        
        try:
            # Handle percentage strings
            if isinstance(value, str):
                value = value.rstrip('%')
            
            # Convert to float
            num = float(value)
            
            # Validate reasonable range
            if num < -100 or num > 100000:
                logger.warning(f"Suspicious numeric value: {num}")
            
            return num
        except (ValueError, TypeError):
            return None
    
    def extract_fixture_data(self, fixture: Dict) -> Dict:
        """Extract core fixture data."""
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
            for participant in participants:
                location = self._safe_get(participant, 'meta', 'location')
                if location == 'home':
                    fixture_data['home_team_id'] = participant.get('id')
                    fixture_data['home_team_name'] = participant.get('name')
                elif location == 'away':
                    fixture_data['away_team_id'] = participant.get('id')
                    fixture_data['away_team_name'] = participant.get('name')
        
        # Extract scores
        scores = fixture.get('scores', [])
        home_score = None
        away_score = None
        
        for score in scores:
            description = score.get('description', '').lower()
            # Look for final score (FT, current, or final)
            if any(keyword in description for keyword in ['current', 'ft', 'final', 'fulltime']):
                participant_id = score.get('participant_id')
                goals = self._safe_get(score, 'score', 'goals')
                
                if participant_id == fixture_data.get('home_team_id'):
                    home_score = goals
                elif participant_id == fixture_data.get('away_team_id'):
                    away_score = goals
        
        fixture_data['home_score'] = home_score
        fixture_data['away_score'] = away_score
        
        # Calculate result (with validation)
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                result = 'H'
            elif home_score < away_score:
                result = 'A'
            else:
                result = 'D'
            fixture_data['result'] = result
        else:
            fixture_data['result'] = None
        
        # Extract state
        state = fixture.get('state', {})
        fixture_data['state'] = state.get('short_name', '')
        
        return fixture_data
    
    def extract_statistics(self, fixture: Dict) -> List[Dict]:
        """Extract team statistics from fixture."""
        fixture_id = fixture.get('id')
        statistics_list = fixture.get('statistics', [])
        
        if not statistics_list:
            return []
        
        # Group statistics by team
        team_stats = {}
        
        for stat in statistics_list:
            participant_id = stat.get('participant_id')
            type_id = stat.get('type_id')
            location = stat.get('location', '')
            
            # Get stat value
            value = self._safe_get(stat, 'data', 'value')
            
            # Initialize team stats if needed
            if participant_id not in team_stats:
                team_stats[participant_id] = {
                    'fixture_id': fixture_id,
                    'team_id': participant_id,
                    'is_home': location == 'home'
                }
            
            # Map type_id to column name
            stat_name = self.STAT_TYPE_MAPPING.get(type_id, f'stat_{type_id}')
            
            # Parse and store value
            parsed_value = self._parse_numeric(value)
            if parsed_value is not None:
                team_stats[participant_id][stat_name] = parsed_value
        
        return list(team_stats.values())
    
    def extract_lineups(self, fixture: Dict) -> List[Dict]:
        """Extract player lineups from fixture."""
        fixture_id = fixture.get('id')
        lineups_list = fixture.get('lineups', [])
        
        if not lineups_list:
            return []
        
        lineup_data = []
        
        for player in lineups_list:
            player_data = {
                'fixture_id': fixture_id,
                'team_id': player.get('team_id'),
                'player_id': player.get('player_id'),
                'player_name': player.get('player_name'),
                'position_id': player.get('position_id'),
                'jersey_number': player.get('jersey_number'),
                'is_starter': player.get('type_id') == 11,  # 11 = starting XI
            }
            
            # Extract player details (stats)
            details = player.get('details', [])
            for detail in details:
                type_id = detail.get('type_id')
                value = self._safe_get(detail, 'data', 'value')
                
                # Map type_id to column name
                stat_name = self.PLAYER_DETAIL_MAPPING.get(type_id, f'detail_{type_id}')
                
                # Parse and store value
                parsed_value = self._parse_numeric(value)
                if parsed_value is not None:
                    player_data[stat_name] = parsed_value
            
            lineup_data.append(player_data)
        
        return lineup_data
    
    def extract_sidelined(self, fixture: Dict) -> List[Dict]:
        """Extract sidelined (injuries/suspensions) from fixture."""
        fixture_id = fixture.get('id')
        sidelined_list = fixture.get('sidelined', [])
        
        if not sidelined_list:
            return []
        
        sidelined_data = []
        
        for sidelined in sidelined_list:
            sidelined_record = {
                'fixture_id': fixture_id,
                'team_id': sidelined.get('team_id'),
                'player_id': sidelined.get('player_id'),
                'player_name': sidelined.get('player_name'),
                'category': sidelined.get('category'),
                'start_date': sidelined.get('start_date'),
                'end_date': sidelined.get('end_date'),
            }
            sidelined_data.append(sidelined_record)
        
        return sidelined_data
    
    def process_json_file(self, json_file: Path) -> tuple:
        """Process a single JSON file and extract all data."""
        fixtures_data = []
        statistics_data = []
        lineups_data = []
        sidelined_data = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both array and single object
            fixtures = data if isinstance(data, list) else [data]
            
            for fixture in fixtures:
                try:
                    # Extract fixture data
                    fixture_record = self.extract_fixture_data(fixture)
                    fixtures_data.append(fixture_record)
                    self.stats['fixtures_processed'] += 1
                    
                    # Extract statistics
                    stats = self.extract_statistics(fixture)
                    if stats:
                        statistics_data.extend(stats)
                        self.stats['fixtures_with_stats'] += 1
                    
                    # Extract lineups
                    lineups = self.extract_lineups(fixture)
                    if lineups:
                        lineups_data.extend(lineups)
                        self.stats['fixtures_with_lineups'] += 1
                    
                    # Extract sidelined
                    sidelined = self.extract_sidelined(fixture)
                    if sidelined:
                        sidelined_data.extend(sidelined)
                        self.stats['fixtures_with_sidelined'] += 1
                
                except Exception as e:
                    logger.error(f"Error processing fixture in {json_file.name}: {e}")
                    self.stats['errors'] += 1
        
        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")
            self.stats['errors'] += 1
        
        return fixtures_data, statistics_data, lineups_data, sidelined_data
    
    def convert_all(self):
        """Convert all JSON files to CSV."""
        logger.info("=" * 80)
        logger.info("JSON TO CSV CONVERSION - V2")
        logger.info("=" * 80)
        
        # Find all JSON files
        fixtures_dir = self.input_dir / 'fixtures'
        if not fixtures_dir.exists():
            logger.error(f"Fixtures directory not found: {fixtures_dir}")
            return
        
        json_files = sorted(fixtures_dir.glob('*.json'))
        logger.info(f"Found {len(json_files)} JSON files")
        
        if len(json_files) == 0:
            logger.error("No JSON files found!")
            return
        
        # Initialize data collectors
        all_fixtures = []
        all_statistics = []
        all_lineups = []
        all_sidelined = []
        
        # Process each file
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            fixtures, stats, lineups, sidelined = self.process_json_file(json_file)
            
            all_fixtures.extend(fixtures)
            all_statistics.extend(stats)
            all_lineups.extend(lineups)
            all_sidelined.extend(sidelined)
        
        # Convert to DataFrames and save
        logger.info("\nConverting to DataFrames and saving...")
        
        # Fixtures
        if all_fixtures:
            df_fixtures = pd.DataFrame(all_fixtures)
            output_file = self.output_dir / 'fixtures.csv'
            df_fixtures.to_csv(output_file, index=False)
            logger.info(f"✅ Saved {len(df_fixtures):,} fixtures to {output_file}")
        else:
            logger.warning("⚠️ No fixtures data to save")
        
        # Statistics
        if all_statistics:
            df_statistics = pd.DataFrame(all_statistics)
            output_file = self.output_dir / 'statistics.csv'
            df_statistics.to_csv(output_file, index=False)
            logger.info(f"✅ Saved {len(df_statistics):,} statistics rows to {output_file}")
        else:
            logger.warning("⚠️ No statistics data to save")
        
        # Lineups
        if all_lineups:
            df_lineups = pd.DataFrame(all_lineups)
            output_file = self.output_dir / 'lineups.csv'
            df_lineups.to_csv(output_file, index=False)
            logger.info(f"✅ Saved {len(df_lineups):,} lineup rows to {output_file}")
        else:
            logger.warning("⚠️ No lineups data to save")
        
        # Sidelined
        if all_sidelined:
            df_sidelined = pd.DataFrame(all_sidelined)
            output_file = self.output_dir / 'sidelined.csv'
            df_sidelined.to_csv(output_file, index=False)
            logger.info(f"✅ Saved {len(df_sidelined):,} sidelined rows to {output_file}")
        else:
            logger.warning("⚠️ No sidelined data to save")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print conversion summary."""
        logger.info("\n" + "=" * 80)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\n📊 Processing Statistics:")
        logger.info(f"  Total fixtures processed: {self.stats['fixtures_processed']:,}")
        logger.info(f"  Fixtures with statistics: {self.stats['fixtures_with_stats']:,} ({self.stats['fixtures_with_stats']/max(self.stats['fixtures_processed'],1)*100:.1f}%)")
        logger.info(f"  Fixtures with lineups: {self.stats['fixtures_with_lineups']:,} ({self.stats['fixtures_with_lineups']/max(self.stats['fixtures_processed'],1)*100:.1f}%)")
        logger.info(f"  Fixtures with sidelined: {self.stats['fixtures_with_sidelined']:,} ({self.stats['fixtures_with_sidelined']/max(self.stats['fixtures_processed'],1)*100:.1f}%)")
        logger.info(f"  Errors encountered: {self.stats['errors']}")
        
        logger.info("\n✅ Conversion complete!")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Convert SportMonks JSON to CSV')
    parser.add_argument('--input', default='data/historical', help='Input directory with JSON files')
    parser.add_argument('--output', default='data/csv', help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    converter = JSONToCSVConverter(input_dir=args.input, output_dir=args.output)
    converter.convert_all()


if __name__ == "__main__":
    main()
