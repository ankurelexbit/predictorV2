"""
Load historical data from JSON files into Supabase database.

This script reads the downloaded JSON files and loads them into Supabase
for faster feature generation.

Usage:
    python scripts/load_data_to_database.py --data-dir data/historical
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import SupabaseDB


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_load.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseLoader:
    """Load historical data into Supabase database."""
    
    def __init__(self, data_dir: str):
        """
        Initialize database loader.
        
        Args:
            data_dir: Directory containing JSON files
        """
        self.data_dir = Path(data_dir)
        self.db = SupabaseDB()
        
        logger.info(f"Initialized database loader with data dir: {self.data_dir}")
    
    def load_fixtures(self) -> int:
        """
        Load fixtures from JSON into database.
        
        Returns:
            Number of fixtures loaded
        """
        logger.info("Loading fixtures into database...")
        
        fixtures_dir = self.data_dir / 'fixtures'
        
        # Load combined fixtures file
        combined_file = list(fixtures_dir.glob('all_fixtures_*.json'))
        
        if not combined_file:
            logger.error("No fixtures file found")
            return 0
        
        with open(combined_file[0], 'r') as f:
            fixtures = json.load(f)
        
        logger.info(f"Found {len(fixtures)} fixtures to load")
        
        # Prepare match records
        matches = []
        
        for fixture in tqdm(fixtures, desc="Preparing matches"):
            fixture_id = fixture.get('id')
            
            if not fixture_id:
                continue
            
            # Parse match data
            match_date = fixture.get('starting_at', '')
            league_id = fixture.get('league_id')
            season_id = fixture.get('season_id')
            
            participants = fixture.get('participants', [])
            if len(participants) != 2:
                continue
            
            home_team_id = participants[0].get('id')
            away_team_id = participants[1].get('id')
            
            # Get scores
            scores = fixture.get('scores', [])
            home_goals = None
            away_goals = None
            
            for score in scores:
                if score.get('description') == 'CURRENT':
                    participant_id = score.get('participant_id')
                    goals = score.get('score', {}).get('goals')
                    
                    if participant_id == home_team_id:
                        home_goals = goals
                    elif participant_id == away_team_id:
                        away_goals = goals
            
            # Determine result
            result = None
            if home_goals is not None and away_goals is not None:
                if home_goals > away_goals:
                    result = 'H'
                elif home_goals < away_goals:
                    result = 'A'
                else:
                    result = 'D'
            
            matches.append({
                'fixture_id': fixture_id,
                'league_id': league_id,
                'season_id': season_id,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'match_date': match_date,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'result': result,
            })
        
        # Insert in batches
        logger.info(f"Inserting {len(matches)} matches into database...")
        
        batch_size = 500
        for i in tqdm(range(0, len(matches), batch_size), desc="Inserting batches"):
            batch = matches[i:i + batch_size]
            try:
                self.db.insert_matches_batch(batch)
            except Exception as e:
                logger.error(f"Error inserting batch {i}: {e}")
                # Try individual inserts for this batch
                for match in batch:
                    try:
                        self.db.upsert_match(match)
                    except Exception as e2:
                        logger.error(f"Error inserting match {match['fixture_id']}: {e2}")
        
        logger.info(f"Loaded {len(matches)} fixtures")
        return len(matches)
    
    def load_statistics(self) -> int:
        """
        Load match statistics from JSON into database.
        
        Returns:
            Number of statistics records loaded
        """
        logger.info("Loading match statistics into database...")
        
        stats_dir = self.data_dir / 'statistics'
        stats_files = list(stats_dir.glob('fixture_*.json'))
        
        logger.info(f"Found {len(stats_files)} statistics files")
        
        all_stats = []
        
        for stats_file in tqdm(stats_files, desc="Preparing statistics"):
            fixture_id = int(stats_file.stem.split('_')[1])
            
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
            
            for stat_group in stats_data:
                team_id = stat_group.get('participant_id')
                
                if not team_id:
                    continue
                
                # Parse statistics
                stats_dict = {
                    'fixture_id': fixture_id,
                    'team_id': team_id,
                }
                
                details = stat_group.get('details', [])
                
                for detail in details:
                    type_name = detail.get('type', {}).get('name', '')
                    value = detail.get('value', {}).get('total', 0)
                    
                    # Map to database columns
                    if 'Shots Total' in type_name:
                        stats_dict['shots_total'] = value
                    elif 'Shots On Target' in type_name:
                        stats_dict['shots_on_target'] = value
                    elif 'Shots Insidebox' in type_name:
                        stats_dict['shots_insidebox'] = value
                    elif 'Shots Outsidebox' in type_name:
                        stats_dict['shots_outsidebox'] = value
                    elif 'Big Chances Created' in type_name:
                        stats_dict['big_chances_created'] = value
                    elif 'Corners' in type_name:
                        stats_dict['corners'] = value
                    elif 'Attacks' in type_name and 'Dangerous' not in type_name:
                        stats_dict['attacks'] = value
                    elif 'Dangerous Attacks' in type_name:
                        stats_dict['dangerous_attacks'] = value
                    elif 'Possession' in type_name:
                        stats_dict['possession'] = value
                    elif 'Passes' in type_name and 'Accurate' not in type_name:
                        stats_dict['passes'] = value
                    elif 'Accurate Passes' in type_name:
                        stats_dict['accurate_passes'] = value
                    elif 'Tackles' in type_name:
                        stats_dict['tackles'] = value
                    elif 'Interceptions' in type_name:
                        stats_dict['interceptions'] = value
                    elif 'Clearances' in type_name:
                        stats_dict['clearances'] = value
                
                all_stats.append(stats_dict)
        
        # Insert in batches
        logger.info(f"Inserting {len(all_stats)} statistics records...")
        
        batch_size = 500
        for i in tqdm(range(0, len(all_stats), batch_size), desc="Inserting batches"):
            batch = all_stats[i:i + batch_size]
            try:
                self.db.insert_statistics_batch(batch)
            except Exception as e:
                logger.error(f"Error inserting statistics batch {i}: {e}")
        
        logger.info(f"Loaded {len(all_stats)} statistics records")
        return len(all_stats)
    
    def run(self):
        """Run complete database loading process."""
        logger.info("=" * 80)
        logger.info("STARTING DATABASE LOAD")
        logger.info("=" * 80)
        
        # Test connection
        if not self.db.test_connection():
            logger.error("Database connection failed. Aborting.")
            return
        
        # Load fixtures
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Fixtures")
        logger.info("=" * 80)
        fixtures_count = self.load_fixtures()
        
        # Load statistics
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Loading Statistics")
        logger.info("=" * 80)
        stats_count = self.load_statistics()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE LOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Fixtures loaded: {fixtures_count}")
        logger.info(f"Statistics loaded: {stats_count}")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Load historical data into Supabase')
    parser.add_argument('--data-dir', default='data/historical', help='Directory with JSON files')
    
    args = parser.parse_args()
    
    # Run database load
    loader = DatabaseLoader(data_dir=args.data_dir)
    
    try:
        loader.run()
    except KeyboardInterrupt:
        logger.info("\nDatabase load interrupted by user")
    except Exception as e:
        logger.error(f"Database load failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
