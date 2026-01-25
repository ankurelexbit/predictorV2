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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        
        SportMonks API has a ~90 day limit for /fixtures/between endpoint.
        This method automatically chunks large date ranges.
        
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
        
        # Calculate date range and chunk if needed
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end_dt - start_dt).days
        
        # SportMonks API limit is ~90 days, use 85 to be safe
        CHUNK_DAYS = 85
        
        if total_days > CHUNK_DAYS:
            logger.info(f"Date range is {total_days} days - chunking into {CHUNK_DAYS}-day periods")
            
            all_fixtures = []
            current_start = start_dt
            
            while current_start < end_dt:
                current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_dt)
                
                chunk_start_str = current_start.strftime('%Y-%m-%d')
                chunk_end_str = current_end.strftime('%Y-%m-%d')
                
                logger.info(f"Downloading chunk: {chunk_start_str} to {chunk_end_str}")
                
                # Download this chunk
                chunk_fixtures = self._download_fixtures_chunk(
                    chunk_start_str,
                    chunk_end_str,
                    league_ids
                )
                all_fixtures.extend(chunk_fixtures)
                
                current_start = current_end + timedelta(days=1)
            
            logger.info(f"Total fixtures downloaded across all chunks: {len(all_fixtures)}")
            
            # Save combined file
            combined_file = self.output_dir / 'fixtures' / f'all_fixtures_{start_date}_{end_date}.json'
            with open(combined_file, 'w') as f:
                json.dump(all_fixtures, f, indent=2)
            
            return all_fixtures
        else:
            # Single chunk - use existing logic
            return self._download_fixtures_chunk(start_date, end_date, league_ids)
    
    def _download_fixtures_chunk(
        self,
        start_date: str,
        end_date: str,
        league_ids: List[int]
    ) -> List[Dict]:
        """
        Download fixtures for a single date chunk (internal method).
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league_ids: List of league IDs
        
        Returns:
            List of fixtures
        """
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
        
        # Save combined file for this chunk
        combined_file = self.output_dir / 'fixtures' / f'all_fixtures_{start_date}_{end_date}.json'
        with open(combined_file, 'w') as f:
            json.dump(all_fixtures, f, indent=2)
        
        return all_fixtures
    
    
    def _download_single_fixture_details(
        self,
        fixture_id: int,
        includes: List[str],
        include_lineups: bool
    ) -> tuple:
        """
        Download details for a single fixture (for parallel processing).
        
        Args:
            fixture_id: Fixture ID
            includes: List of includes
            include_lineups: Whether lineups are included
        
        Returns:
            Tuple of (fixture_id, detailed_fixture, has_stats, has_lineups)
        """
        try:
            # ONE API call gets everything!
            detailed_fixture = self.client.get_fixture_by_id(fixture_id, includes=includes)
            
            # Extract and save statistics
            stats = detailed_fixture.get('statistics', [])
            has_stats = bool(stats)
            if has_stats:
                output_file = self.output_dir / 'statistics' / f'fixture_{fixture_id}.json'
                with open(output_file, 'w') as f:
                    json.dump(stats, f, indent=2)
            
            # Extract and save lineups
            has_lineups = False
            if include_lineups:
                lineups = detailed_fixture.get('lineups', [])
                has_lineups = bool(lineups)
                if has_lineups:
                    output_file = self.output_dir / 'lineups' / f'fixture_{fixture_id}.json'
                    with open(output_file, 'w') as f:
                        json.dump(lineups, f, indent=2)
            
            return (fixture_id, detailed_fixture, has_stats, has_lineups)
            
        except Exception as e:
            logger.error(f"Error downloading details for fixture {fixture_id}: {e}")
            return (fixture_id, None, False, False)
    
    def backfill_fixture_details(
        self, 
        fixtures: List[Dict],
        include_lineups: bool = True,
        max_workers: int = 10
    ) -> Dict[int, Dict]:
        """
        Download detailed fixture data (statistics + lineups) in parallel.
        
        Uses ThreadPoolExecutor for concurrent downloads - 10-20x faster!
        
        Args:
            fixtures: List of fixtures
            include_lineups: Include lineups in the request
            max_workers: Number of parallel workers (default: 10)
        
        Returns:
            Dictionary mapping fixture_id to complete fixture data
        """
        includes = ['statistics']
        if include_lineups:
            includes.append('lineups')
        
        logger.info(f"Backfilling details for {len(fixtures)} fixtures (includes: {', '.join(includes)})")
        logger.info(f"Using {max_workers} parallel workers for faster downloads! 🚀")
        
        fixture_details = {}
        stats_count = 0
        lineups_count = 0
        
        # Get fixture IDs
        fixture_ids = [f.get('id') for f in fixtures if f.get('id')]
        
        # Process in parallel with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(
                    self._download_single_fixture_details,
                    fid,
                    includes,
                    include_lineups
                ): fid for fid in fixture_ids
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(fixture_ids), desc="Fixture Details") as pbar:
                for future in as_completed(future_to_id):
                    fixture_id, detailed_fixture, has_stats, has_lineups_flag = future.result()
                    
                    if detailed_fixture:
                        fixture_details[fixture_id] = detailed_fixture
                        if has_stats:
                            stats_count += 1
                        if has_lineups_flag:
                            lineups_count += 1
                    
                    pbar.update(1)
        
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
        include_sidelined: bool = True,
        max_workers: int = 10
    ):
        """
        Run complete backfill process.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_lineups: Download lineups
            include_sidelined: Download sidelined players
            max_workers: Number of parallel workers for downloads
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
        fixture_details = self.backfill_fixture_details(
            fixtures, 
            include_lineups=include_lineups,
            max_workers=max_workers
        )
        
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
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10, max: 20)')
    
    args = parser.parse_args()
    
    # Validate workers
    if args.workers < 1 or args.workers > 20:
        logger.error("Workers must be between 1 and 20")
        sys.exit(1)
    
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
            include_sidelined=not args.skip_sidelined,
            max_workers=args.workers
        )
    except KeyboardInterrupt:
        logger.info("\nBackfill interrupted by user")
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
    finally:
        backfill.close()


if __name__ == '__main__':
    main()
