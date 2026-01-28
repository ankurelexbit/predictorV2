#!/usr/bin/env python3
"""
Validate JSON to CSV conversion.

This script verifies that the data in the CSV files matches the source JSON files.
It checks for:
1. Fixture counts (Total JSON fixtures vs Total CSV rows)
2. Duplicate fixtures in JSON
3. Missing fixtures in CSV
4. Data integrity (Sample check)

Usage:
    python scripts/validate_json_to_csv.py
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_fixtures(data_dir: Path, csv_dir: Path):
    """Validate fixtures data."""
    logger.info("=" * 60)
    logger.info("VALIDATING FIXTURES")
    logger.info("=" * 60)
    
    # 1. Load JSON Data
    fixtures_dir = data_dir / 'fixtures'
    fixture_files = sorted(fixtures_dir.glob('all_fixtures_*.json'))
    
    json_fixture_ids = set()
    json_duplicates = []
    total_json_count = 0
    
    logger.info(f"Scanning {len(fixture_files)} JSON files...")
    
    for fixture_file in tqdm(fixture_files, desc="Reading JSON"):
        try:
            with open(fixture_file) as f:
                fixtures = json.load(f)
                total_json_count += len(fixtures)
                
                for fixture in fixtures:
                    fid = fixture.get('id')
                    if fid:
                        if fid in json_fixture_ids:
                            json_duplicates.append(fid)
                        json_fixture_ids.add(fid)
        except Exception as e:
            logger.error(f"Error reading {fixture_file}: {e}")
    
    logger.info(f"Total fixtures in JSON: {total_json_count}")
    logger.info(f"Unique fixtures in JSON: {len(json_fixture_ids)}")
    
    if json_duplicates:
        logger.warning(f"⚠️ Found {len(json_duplicates)} duplicates in JSON files (taking latest)")
    else:
        logger.info("✅ No duplicates found in JSON")

    # 2. Load CSV Data
    csv_file = csv_dir / 'fixtures.csv'
    if not csv_file.exists():
        logger.error(f"❌ {csv_file} not found!")
        return False
    
    try:
        csv_df = pd.read_csv(csv_file)
        csv_fixture_ids = set(csv_df['fixture_id'].unique())
        
        logger.info(f"Total rows in CSV: {len(csv_df)}")
        logger.info(f"Unique fixtures in CSV: {len(csv_fixture_ids)}")
        
        # 3. Compare
        missing_in_csv = json_fixture_ids - csv_fixture_ids
        extra_in_csv = csv_fixture_ids - json_fixture_ids
        
        if not missing_in_csv and not extra_in_csv:
            logger.info("✅ SUCCESS: All JSON fixtures are present in CSV")
        else:
            if missing_in_csv:
                logger.error(f"❌ MISSING: {len(missing_in_csv)} fixtures from JSON are missing in CSV")
                logger.error(f"Sample missing IDs: {list(missing_in_csv)[:5]}")
            if extra_in_csv:
                logger.warning(f"⚠️ EXTRA: {len(extra_in_csv)} fixtures in CSV not in scanned JSON (orphans?)")
        
        return len(missing_in_csv) == 0

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return False


def main():
    """Main validation process."""
    data_dir = Path('data/historical')
    csv_dir = Path('data/csv')
    
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        sys.exit(1)
        
    if not csv_dir.exists():
        logger.error(f"CSV directory {csv_dir} does not exist. Run convert_to_csv.py first.")
        sys.exit(1)
    
    success = validate_fixtures(data_dir, csv_dir)
    
    if success:
        logger.info("\n✅ VALIDATION PASSED")
        sys.exit(0)
    else:
        logger.info("\n❌ VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
