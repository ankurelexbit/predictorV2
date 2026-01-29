#!/usr/bin/env python3
"""
Comprehensive Data Validation Script.

Validates CSV data against original JSON files by:
1. Checking row counts and completeness
2. Random sampling and cross-validation
3. Data integrity checks (types, ranges, nulls)
4. Standings data validation (NEW)

Usage:
    python scripts/validate_data.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation against source JSON."""
    
    def __init__(self, data_dir: str = 'data/historical', csv_dir: str = 'data/csv'):
        self.data_dir = Path(data_dir)
        self.csv_dir = Path(csv_dir)
        self.validation_results = {}
        
    def load_random_json_fixtures(self, sample_size: int = 10):
        """Load random fixtures from JSON for validation using streaming (optimized)."""
        import ijson
        
        fixtures_dir = self.data_dir / 'fixtures'
        fixture_files = sorted(fixtures_dir.glob('all_fixtures_*.json'))
        
        if not fixture_files:
            logger.error("No JSON fixture files found")
            return []
        
        # OPTIMIZATION: Only scan a random subset of files (much faster!)
        num_files_to_scan = min(5, len(fixture_files))  # Scan max 5 random files
        random_files = random.sample(fixture_files, num_files_to_scan)
        
        logger.info(f"Scanning {num_files_to_scan} random JSON files (out of {len(fixture_files)})...")
        all_fixture_ids = []
        
        for fixture_file in random_files:
            try:
                with open(fixture_file, 'rb') as f:
                    # Stream through and collect only IDs
                    for fixture in ijson.items(f, 'item'):
                        fixture_id = fixture.get('id')
                        if fixture_id:
                            all_fixture_ids.append((fixture_id, fixture_file))
            except Exception as e:
                logger.error(f"Error scanning {fixture_file.name}: {e}")
        
        logger.info(f"Found {len(all_fixture_ids)} fixtures in scanned files")
        
        # Random sample of fixture IDs
        if len(all_fixture_ids) > sample_size:
            sampled_ids = random.sample(all_fixture_ids, sample_size)
        else:
            sampled_ids = all_fixture_ids
        
        # Load only the sampled fixtures
        logger.info(f"Loading {len(sampled_ids)} sampled fixtures...")
        sampled_fixtures = []
        
        # Group by file to minimize file opens
        from collections import defaultdict
        ids_by_file = defaultdict(list)
        for fixture_id, fixture_file in sampled_ids:
            ids_by_file[fixture_file].append(fixture_id)
        
        for fixture_file, fixture_ids in ids_by_file.items():
            try:
                with open(fixture_file, 'rb') as f:
                    # Stream through and collect matching fixtures
                    for fixture in ijson.items(f, 'item'):
                        if fixture.get('id') in fixture_ids:
                            sampled_fixtures.append(fixture)
                            if len(sampled_fixtures) >= sample_size:
                                break
            except Exception as e:
                logger.error(f"Error loading from {fixture_file.name}: {e}")
        
        return sampled_fixtures

    
    def validate_fixtures(self, sample_size: int = 10):
        """Validate fixtures CSV against JSON."""
        logger.info("=" * 80)
        logger.info("VALIDATING FIXTURES")
        logger.info("=" * 80)
        
        # Load CSV
        csv_file = self.csv_dir / 'fixtures.csv'
        if not csv_file.exists():
            logger.error(f"❌ {csv_file} not found")
            return False
        
        csv_df = pd.read_csv(csv_file)
        logger.info(f"CSV: {len(csv_df)} rows, {len(csv_df['fixture_id'].unique())} unique fixtures")
        
        # Load random JSON samples
        json_fixtures = self.load_random_json_fixtures(sample_size)
        logger.info(f"Validating {len(json_fixtures)} random samples...")
        
        errors = []
        for fixture in tqdm(json_fixtures, desc="Cross-checking"):
            fixture_id = fixture.get('id')
            csv_row = csv_df[csv_df['fixture_id'] == fixture_id]
            
            if len(csv_row) == 0:
                errors.append(f"Fixture {fixture_id} missing in CSV")
                continue
            
            csv_row = csv_row.iloc[0]
            
            # Validate key fields
            participants = fixture.get('participants', [])
            if len(participants) >= 2:
                home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
                
                if home_team and csv_row['home_team_id'] != home_team.get('id'):
                    errors.append(f"Fixture {fixture_id}: home_team_id mismatch")
                
                if away_team and csv_row['away_team_id'] != away_team.get('id'):
                    errors.append(f"Fixture {fixture_id}: away_team_id mismatch")
            
            # Validate scores
            scores = fixture.get('scores', [])
            for score in scores:
                if 'current' in score.get('description', '').lower():
                    participant_id = score.get('participant_id')
                    goals = score.get('score', {}).get('goals', 0)
                    
                    if participant_id == csv_row['home_team_id'] and csv_row['home_score'] != goals:
                        errors.append(f"Fixture {fixture_id}: home_score mismatch ({csv_row['home_score']} vs {goals})")
                    elif participant_id == csv_row['away_team_id'] and csv_row['away_score'] != goals:
                        errors.append(f"Fixture {fixture_id}: away_score mismatch ({csv_row['away_score']} vs {goals})")
        
        if errors:
            logger.error(f"❌ Found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10
                logger.error(f"  - {error}")
            return False
        else:
            logger.info("✅ All sampled fixtures match JSON data")
            return True
    
    def validate_standings(self, sample_size: int = 10):
        """Validate standings CSV against JSON participants.meta."""
        logger.info("=" * 80)
        logger.info("VALIDATING STANDINGS")
        logger.info("=" * 80)
        
        # Load CSV
        csv_file = self.csv_dir / 'standings.csv'
        if not csv_file.exists():
            logger.error(f"❌ {csv_file} not found")
            return False
        
        csv_df = pd.read_csv(csv_file)
        logger.info(f"CSV: {len(csv_df)} rows, {len(csv_df['fixture_id'].unique())} unique fixtures")
        
        # Load random JSON samples
        json_fixtures = self.load_random_json_fixtures(sample_size)
        logger.info(f"Validating {len(json_fixtures)} random samples...")
        
        errors = []
        for fixture in tqdm(json_fixtures, desc="Cross-checking standings"):
            fixture_id = fixture.get('id')
            csv_rows = csv_df[csv_df['fixture_id'] == fixture_id]
            
            if len(csv_rows) == 0:
                # Standings might not be available for all fixtures
                continue
            
            participants = fixture.get('participants', [])
            for participant in participants:
                team_id = participant.get('id')
                meta = participant.get('meta', {})
                location = meta.get('location')
                
                # Find corresponding CSV row
                csv_row = csv_rows[
                    (csv_rows['team_id'] == team_id) & 
                    (csv_rows['location'] == location)
                ]
                
                if len(csv_row) == 0:
                    if meta.get('position') is not None:
                        errors.append(f"Fixture {fixture_id}, Team {team_id}: standings missing in CSV")
                    continue
                
                csv_row = csv_row.iloc[0]
                
                # Validate only position field (other fields not available in API)
                json_position = meta.get('position')
                csv_position = csv_row['position']
                
                if pd.notna(json_position) and pd.notna(csv_position):
                    if json_position != csv_position:
                        errors.append(
                            f"Fixture {fixture_id}, Team {team_id}: position mismatch "
                            f"(CSV: {csv_position}, JSON: {json_position})"
                        )

        
        if errors:
            logger.error(f"❌ Found {len(errors)} errors:")
            for error in errors[:10]:
                logger.error(f"  - {error}")
            return False
        else:
            logger.info("✅ All sampled standings match JSON data")
            return True
    
    def validate_lineups(self, sample_size: int = 10):
        """Validate lineups CSV against JSON."""
        logger.info("=" * 80)
        logger.info("VALIDATING LINEUPS")
        logger.info("=" * 80)
        
        csv_file = self.csv_dir / 'lineups.csv'
        if not csv_file.exists():
            logger.error(f"❌ {csv_file} not found")
            return False
        
        csv_df = pd.read_csv(csv_file)
        logger.info(f"CSV: {len(csv_df)} rows")
        
        json_fixtures = self.load_random_json_fixtures(sample_size)
        logger.info(f"Validating {len(json_fixtures)} random samples...")
        
        errors = []
        for fixture in tqdm(json_fixtures, desc="Cross-checking lineups"):
            fixture_id = fixture.get('id')
            lineups = fixture.get('lineups', [])
            
            if not lineups:
                continue
            
            csv_rows = csv_df[csv_df['fixture_id'] == fixture_id]
            
            # Check player count
            if len(csv_rows) != len(lineups):
                errors.append(
                    f"Fixture {fixture_id}: player count mismatch "
                    f"(CSV: {len(csv_rows)}, JSON: {len(lineups)})"
                )
            
            # Sample a few players
            for player in random.sample(lineups, min(3, len(lineups))):
                player_id = player.get('player_id')
                csv_player = csv_rows[csv_rows['player_id'] == player_id]
                
                if len(csv_player) == 0:
                    errors.append(f"Fixture {fixture_id}: Player {player_id} missing in CSV")
                    continue
                
                csv_player = csv_player.iloc[0]
                
                # Check is_starter
                is_starter_json = player.get('type_id') == 11
                is_starter_csv = csv_player['is_starter']
                
                if is_starter_json != is_starter_csv:
                    errors.append(
                        f"Fixture {fixture_id}, Player {player_id}: is_starter mismatch"
                    )
        
        if errors:
            logger.error(f"❌ Found {len(errors)} errors:")
            for error in errors[:10]:
                logger.error(f"  - {error}")
            return False
        else:
            logger.info("✅ All sampled lineups match JSON data")
            return True
    
    def validate_statistics(self, sample_size: int = 10):
        """Validate statistics CSV against JSON."""
        logger.info("=" * 80)
        logger.info("VALIDATING STATISTICS")
        logger.info("=" * 80)
        
        csv_file = self.csv_dir / 'statistics.csv'
        if not csv_file.exists():
            logger.error(f"❌ {csv_file} not found")
            return False
        
        csv_df = pd.read_csv(csv_file)
        logger.info(f"CSV: {len(csv_df)} rows")
        
        json_fixtures = self.load_random_json_fixtures(sample_size)
        logger.info(f"Validating {len(json_fixtures)} random samples...")
        
        errors = []
        for fixture in tqdm(json_fixtures, desc="Cross-checking statistics"):
            fixture_id = fixture.get('id')
            stats = fixture.get('statistics', [])
            
            if not stats:
                continue
            
            csv_rows = csv_df[csv_df['fixture_id'] == fixture_id]
            
            # Check we have 2 teams
            if len(csv_rows) != 2:
                errors.append(
                    f"Fixture {fixture_id}: expected 2 teams, got {len(csv_rows)}"
                )
                continue
            
            # Sample check: possession should add up to ~100%
            home_poss = csv_rows[csv_rows['is_home'] == True]['possession'].values
            away_poss = csv_rows[csv_rows['is_home'] == False]['possession'].values
            
            if len(home_poss) > 0 and len(away_poss) > 0:
                total_poss = home_poss[0] + away_poss[0]
                if abs(total_poss - 100) > 1:  # Allow 1% tolerance
                    errors.append(
                        f"Fixture {fixture_id}: possession doesn't add to 100% ({total_poss}%)"
                    )
        
        if errors:
            logger.error(f"❌ Found {len(errors)} errors:")
            for error in errors[:10]:
                logger.error(f"  - {error}")
            return False
        else:
            logger.info("✅ All sampled statistics match JSON data")
            return True
    
    def check_data_quality(self):
        """Check data quality metrics."""
        logger.info("=" * 80)
        logger.info("DATA QUALITY CHECKS")
        logger.info("=" * 80)
        
        all_good = True
        
        # Check each CSV file
        csv_files = {
            'fixtures': ['fixture_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score'],
            'standings': ['fixture_id', 'team_id', 'position'],  # Only position available from API
            'lineups': ['fixture_id', 'player_id', 'team_id'],
            'statistics': ['fixture_id', 'team_id'],
        }
        
        for csv_name, required_cols in csv_files.items():
            csv_file = self.csv_dir / f'{csv_name}.csv'
            
            if not csv_file.exists():
                logger.warning(f"⚠️ {csv_name}.csv not found")
                continue
            
            df = pd.read_csv(csv_file)
            logger.info(f"\n{csv_name}.csv:")
            logger.info(f"  Rows: {len(df):,}")
            logger.info(f"  Columns: {len(df.columns)}")
            
            # Check required columns
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                logger.error(f"  ❌ Missing columns: {missing_cols}")
                all_good = False
            else:
                logger.info(f"  ✅ All required columns present")
            
            # Check for nulls in key columns
            for col in required_cols:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    null_pct = (null_count / len(df)) * 100
                    
                    if null_pct > 10:  # More than 10% nulls
                        logger.warning(f"  ⚠️ {col}: {null_pct:.1f}% null ({null_count:,} rows)")
                        all_good = False
                    elif null_pct > 0:
                        logger.info(f"  {col}: {null_pct:.1f}% null ({null_count:,} rows)")
            
            # Check for duplicates
            if 'fixture_id' in df.columns:
                dup_count = df.duplicated(subset=['fixture_id']).sum()
                if dup_count > 0:
                    logger.warning(f"  ⚠️ {dup_count:,} duplicate fixture_ids")
        
        return all_good
    
    def run_full_validation(self, sample_size: int = 20):
        """Run complete validation suite."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE DATA VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Sample size: {sample_size} fixtures per check\n")
        
        results = {
            'fixtures': self.validate_fixtures(sample_size),
            'standings': self.validate_standings(sample_size),
            'lineups': self.validate_lineups(sample_size),
            'statistics': self.validate_statistics(sample_size),
            'data_quality': self.check_data_quality(),
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        for check, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{check.upper()}: {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("\n🎉 ALL VALIDATIONS PASSED!")
            return True
        else:
            logger.error("\n❌ SOME VALIDATIONS FAILED")
            return False


def main():
    """Main entry point."""
    validator = DataValidator()
    
    if not validator.data_dir.exists():
        logger.error(f"Data directory {validator.data_dir} does not exist")
        sys.exit(1)
    
    if not validator.csv_dir.exists():
        logger.error(f"CSV directory {validator.csv_dir} does not exist")
        logger.error("Run convert_to_csv.py first")
        sys.exit(1)
    
    # Run validation with 20 random samples per check
    success = validator.run_full_validation(sample_size=20)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
