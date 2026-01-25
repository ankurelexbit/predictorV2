"""
Historical Data Backfill Script.

Downloads historical fixtures, statistics, lineups, and player data
from SportMonks API for training data.

Usage:
    python scripts/backfill_historical_data.py --start-date 2022-08-01 --end-date 2025-05-31
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import logging
import json
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sportmonks_client import SportMonksClient
from config.api_config import SportMonksConfig
from config.database_config import DatabaseConfig


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backfill.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HistoricalDataBackfill:
    """Download and store historical football data."""
    
    def __init__(self, output_dir: str = 'data/historical'):
        """
        Initialize backfill process.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.client = SportMonksClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'fixtures').mkdir(exist_ok=True)
        (self.output_dir / 'statistics').mkdir(exist_ok=True)
        (self.output_dir / 'lineups').mkdir(exist_ok=True)
        (self.output_dir / 'sidelined').mkdir(exist_ok=True)
        
        logger.info(f"Initialized backfill with output directory: {self.output_dir}")
    
    def backfill_fixtures(
        self,
        start_date: str,
        end_date: str,
        league_ids: List[int] = None
    ) -> List[Dict]:
        """
        Download all fixtures between dates.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league_ids: Optional list of league IDs to filter
        
        Returns:
            List of fixtures
        """
        logger.info(f"Backfilling fixtures from {start_date} to {end_date}")
        
        # Use configured leagues if none specified
        if league_ids is None:
            league_ids = list(SportMonksConfig.LEAGUES.values())
        
        all_fixtures = []
        
        # Download fixtures for each league
        for league_id in tqdm(league_ids, desc="Leagues"):
            try:
                fixtures = self.client.get_fixtures_between(
                    start_date,
                    end_date,
                    league_id=league_id
                )
                
                logger.info(f"Downloaded {len(fixtures)} fixtures for league {league_id}")
                all_fixtures.extend(fixtures)
                
                # Save to file
                output_file = self.output_dir / 'fixtures' / f'league_{league_id}_{start_date}_{end_date}.json'
                with open(output_file, 'w') as f:
                    json.dump(fixtures, f, indent=2)
                
                # Small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error downloading fixtures for league {league_id}: {e}")
                continue
        
        logger.info(f"Total fixtures downloaded: {len(all_fixtures)}")
        
        # Save combined file
        combined_file = self.output_dir / 'fixtures' / f'all_fixtures_{start_date}_{end_date}.json'
        with open(combined_file, 'w') as f:
            json.dump(all_fixtures, f, indent=2)
        
        return all_fixtures
    
    
    def backfill_fixture_details(
        self, 
        fixtures: List[Dict],
        include_lineups: bool = True
    ) -> Dict[int, Dict]:
        """
        Download detailed fixture data (statistics + lineups) in single API call.
        
        This is MUCH more efficient than separate calls - reduces API calls by 66%!
        
        Args:
            fixtures: List of fixtures
            include_lineups: Include lineups in the request
        
        Returns:
            Dictionary mapping fixture_id to complete fixture data
        """
        includes = ['statistics']
        if include_lineups:
            includes.append('lineups')
        
        logger.info(f"Backfilling details for {len(fixtures)} fixtures (includes: {', '.join(includes)})")
        
        fixture_details = {}
        stats_count = 0
        lineups_count = 0
        
        for fixture in tqdm(fixtures, desc="Fixture Details"):
            fixture_id = fixture.get('id')
            
            if not fixture_id:
                continue
            
            try:
                # ONE API call gets everything!
                detailed_fixture = self.client.get_fixture_by_id(fixture_id, includes=includes)
                fixture_details[fixture_id] = detailed_fixture
                
                # Extract and save statistics
                stats = detailed_fixture.get('statistics', [])
                if stats:
                    stats_count += 1
                    output_file = self.output_dir / 'statistics' / f'fixture_{fixture_id}.json'
                    with open(output_file, 'w') as f:
                        json.dump(stats, f, indent=2)
                
                # Extract and save lineups
                if include_lineups:
                    lineups = detailed_fixture.get('lineups', [])
                    if lineups:
                        lineups_count += 1
                        output_file = self.output_dir / 'lineups' / f'fixture_{fixture_id}.json'
                        with open(output_file, 'w') as f:
                            json.dump(lineups, f, indent=2)
                
                # Small delay
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error downloading details for fixture {fixture_id}: {e}")
                continue
        
        logger.info(f"Downloaded statistics for {stats_count} fixtures")
        if include_lineups:
            logger.info(f"Downloaded lineups for {lineups_count} fixtures")
        
        return fixture_details
    
    
    def backfill_sidelined_players(self, team_ids: List[int]) -> Dict[int, List[Dict]]:
        """
        Download sidelined players (injuries/suspensions) for teams.
        
        Args:
            team_ids: List of team IDs
        
        Returns:
            Dictionary mapping team_id to sidelined players
        """
        logger.info(f"Backfilling sidelined players for {len(team_ids)} teams")
        
        sidelined_data = {}
        
        for team_id in tqdm(team_ids, desc="Sidelined Players"):
            try:
                sidelined = self.client.get_team_sidelined(team_id)
                sidelined_data[team_id] = sidelined
                
                # Save individual file
                output_file = self.output_dir / 'sidelined' / f'team_{team_id}.json'
                with open(output_file, 'w') as f:
                    json.dump(sidelined, f, indent=2)
                
                # Small delay
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error downloading sidelined for team {team_id}: {e}")
                continue
        
        logger.info(f"Downloaded sidelined data for {len(sidelined_data)} teams")
        
        return sidelined_data
    
    def extract_team_ids(self, fixtures: List[Dict]) -> List[int]:
        """
        Extract unique team IDs from fixtures.
        
        Args:
            fixtures: List of fixtures
        
        Returns:
            List of unique team IDs
        """
        team_ids = set()
        
        for fixture in fixtures:
            participants = fixture.get('participants', [])
            for participant in participants:
                team_id = participant.get('id')
                if team_id:
                    team_ids.add(team_id)
        
        return list(team_ids)
    
    def run_full_backfill(
        self,
        start_date: str,
        end_date: str,
        include_lineups: bool = True,
        include_sidelined: bool = True
    ):
        """
        Run complete backfill process.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_lineups: Download lineups
            include_sidelined: Download sidelined players
        """
        logger.info("=" * 80)
        logger.info("STARTING FULL HISTORICAL DATA BACKFILL")
        logger.info("=" * 80)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Include lineups: {include_lineups}")
        logger.info(f"Include sidelined: {include_sidelined}")
        
        # Step 1: Download fixtures
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Downloading Fixtures")
        logger.info("=" * 80)
        fixtures = self.backfill_fixtures(start_date, end_date)
        
        if not fixtures:
            logger.error("No fixtures downloaded. Aborting.")
            return
        
        # Step 2: Download fixture details (statistics + lineups in ONE call!)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Downloading Fixture Details (Statistics + Lineups)")
        logger.info("=" * 80)
        logger.info("Using OPTIMIZED single API call per fixture! 🚀")
        fixture_details = self.backfill_fixture_details(fixtures, include_lineups=include_lineups)
        
        # Count what we got
        stats_count = sum(1 for fid, data in fixture_details.items() if data.get('statistics'))
        lineups_count = sum(1 for fid, data in fixture_details.items() if data.get('lineups')) if include_lineups else 0
        
        # Step 3: Download sidelined players (optional)
        if include_sidelined:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Downloading Sidelined Players")
            logger.info("=" * 80)
            team_ids = self.extract_team_ids(fixtures)
            sidelined = self.backfill_sidelined_players(team_ids)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("BACKFILL COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Fixtures: {len(fixtures)}")
        logger.info(f"Statistics: {stats_count}")
        if include_lineups:
            logger.info(f"Lineups: {lineups_count}")
        if include_sidelined:
            logger.info(f"Sidelined data: {len(sidelined)} teams")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"\n💡 API Efficiency: Used ~{len(fixtures)} calls instead of ~{len(fixtures) * 3}!")
        logger.info("=" * 80)
    
    def close(self):
        """Close API client."""
        self.client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Backfill historical football data')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data/historical', help='Output directory')
    parser.add_argument('--skip-lineups', action='store_true', help='Skip downloading lineups')
    parser.add_argument('--skip-sidelined', action='store_true', help='Skip downloading sidelined players')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    # Run backfill
    backfill = HistoricalDataBackfill(output_dir=args.output_dir)
    
    try:
        backfill.run_full_backfill(
            start_date=args.start_date,
            end_date=args.end_date,
            include_lineups=not args.skip_lineups,
            include_sidelined=not args.skip_sidelined
        )
    except KeyboardInterrupt:
        logger.info("\nBackfill interrupted by user")
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
    finally:
        backfill.close()


if __name__ == '__main__':
    main()
