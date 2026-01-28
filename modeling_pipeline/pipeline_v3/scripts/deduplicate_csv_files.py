#!/usr/bin/env python3
"""
CSV Deduplication Script

This script removes duplicate records from all CSV files in the pipeline.
It handles:
- Fixtures: deduplicate by fixture_id
- Statistics: deduplicate by (fixture_id, team_id)
- Lineups: deduplicate by (fixture_id, player_id)
- Sidelined: deduplicate by (fixture_id, player_id)

Usage:
    python scripts/deduplicate_csv_files.py
    
    # With custom directory
    python scripts/deduplicate_csv_files.py --data-dir data/csv
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def deduplicate_fixtures(data_dir: Path) -> dict:
    """Deduplicate fixtures CSV."""
    file_path = data_dir / 'fixtures.csv'
    
    if not file_path.exists():
        logger.warning(f"Fixtures file not found: {file_path}")
        return {'status': 'skipped', 'reason': 'file not found'}
    
    logger.info(f"\n📄 Processing: {file_path.name}")
    
    # Load data
    df = pd.read_csv(file_path)
    original_count = len(df)
    logger.info(f"   Original rows: {original_count:,}")
    
    # Deduplicate by fixture_id
    df_unique = df.drop_duplicates(subset=['fixture_id'], keep='first')
    final_count = len(df_unique)
    duplicates_removed = original_count - final_count
    
    logger.info(f"   Unique rows: {final_count:,}")
    logger.info(f"   Duplicates removed: {duplicates_removed:,}")
    
    # Save deduplicated data
    if duplicates_removed > 0:
        df_unique.to_csv(file_path, index=False)
        logger.info(f"   ✅ Saved deduplicated data")
    else:
        logger.info(f"   ✅ No duplicates found")
    
    return {
        'status': 'success',
        'original': original_count,
        'final': final_count,
        'removed': duplicates_removed
    }


def deduplicate_statistics(data_dir: Path) -> dict:
    """Deduplicate statistics CSV."""
    file_path = data_dir / 'statistics.csv'
    
    if not file_path.exists():
        logger.warning(f"Statistics file not found: {file_path}")
        return {'status': 'skipped', 'reason': 'file not found'}
    
    logger.info(f"\n📄 Processing: {file_path.name}")
    
    # Load data
    df = pd.read_csv(file_path, low_memory=False)
    original_count = len(df)
    logger.info(f"   Original rows: {original_count:,}")
    
    # Deduplicate by (fixture_id, team_id)
    # Each fixture should have 2 rows: one for home team, one for away team
    df_unique = df.drop_duplicates(subset=['fixture_id', 'team_id'], keep='first')
    final_count = len(df_unique)
    duplicates_removed = original_count - final_count
    
    logger.info(f"   Unique rows: {final_count:,}")
    logger.info(f"   Duplicates removed: {duplicates_removed:,}")
    
    # Verify we have ~2 rows per fixture
    rows_per_fixture = df_unique.groupby('fixture_id').size()
    logger.info(f"   Avg rows per fixture: {rows_per_fixture.mean():.1f}")
    
    # Save deduplicated data
    if duplicates_removed > 0:
        df_unique.to_csv(file_path, index=False)
        logger.info(f"   ✅ Saved deduplicated data")
    else:
        logger.info(f"   ✅ No duplicates found")
    
    return {
        'status': 'success',
        'original': original_count,
        'final': final_count,
        'removed': duplicates_removed
    }


def deduplicate_lineups(data_dir: Path) -> dict:
    """Deduplicate lineups CSV."""
    file_path = data_dir / 'lineups.csv'
    
    if not file_path.exists():
        logger.warning(f"Lineups file not found: {file_path}")
        return {'status': 'skipped', 'reason': 'file not found'}
    
    logger.info(f"\n📄 Processing: {file_path.name}")
    
    # Load data
    df = pd.read_csv(file_path, low_memory=False)
    original_count = len(df)
    logger.info(f"   Original rows: {original_count:,}")
    
    # Deduplicate by (fixture_id, player_id)
    # Each player should appear once per fixture
    df_unique = df.drop_duplicates(subset=['fixture_id', 'player_id'], keep='first')
    final_count = len(df_unique)
    duplicates_removed = original_count - final_count
    
    logger.info(f"   Unique rows: {final_count:,}")
    logger.info(f"   Duplicates removed: {duplicates_removed:,}")
    
    # Verify player counts
    players_per_fixture = df_unique.groupby('fixture_id').size()
    logger.info(f"   Avg players per fixture: {players_per_fixture.mean():.1f}")
    
    # Save deduplicated data
    if duplicates_removed > 0:
        df_unique.to_csv(file_path, index=False)
        logger.info(f"   ✅ Saved deduplicated data")
    else:
        logger.info(f"   ✅ No duplicates found")
    
    return {
        'status': 'success',
        'original': original_count,
        'final': final_count,
        'removed': duplicates_removed
    }


def deduplicate_sidelined(data_dir: Path) -> dict:
    """Deduplicate sidelined CSV."""
    file_path = data_dir / 'sidelined.csv'
    
    if not file_path.exists():
        logger.warning(f"Sidelined file not found: {file_path}")
        return {'status': 'skipped', 'reason': 'file not found'}
    
    logger.info(f"\n📄 Processing: {file_path.name}")
    
    # Load data
    df = pd.read_csv(file_path)
    original_count = len(df)
    logger.info(f"   Original rows: {original_count:,}")
    
    # Deduplicate by (fixture_id, player_id)
    # Same player can be sidelined for multiple fixtures, but not duplicated within same fixture
    df_unique = df.drop_duplicates(subset=['fixture_id', 'player_id'], keep='first')
    final_count = len(df_unique)
    duplicates_removed = original_count - final_count
    
    logger.info(f"   Unique rows: {final_count:,}")
    logger.info(f"   Duplicates removed: {duplicates_removed:,}")
    
    # Save deduplicated data
    if duplicates_removed > 0:
        df_unique.to_csv(file_path, index=False)
        logger.info(f"   ✅ Saved deduplicated data")
    else:
        logger.info(f"   ✅ No duplicates found")
    
    return {
        'status': 'success',
        'original': original_count,
        'final': final_count,
        'removed': duplicates_removed
    }


def print_summary(results: dict):
    """Print deduplication summary."""
    logger.info("\n" + "=" * 80)
    logger.info("DEDUPLICATION SUMMARY")
    logger.info("=" * 80)
    
    total_original = 0
    total_final = 0
    total_removed = 0
    
    for file_name, result in results.items():
        if result['status'] == 'success':
            total_original += result['original']
            total_final += result['final']
            total_removed += result['removed']
    
    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"   Total original rows: {total_original:,}")
    logger.info(f"   Total final rows: {total_final:,}")
    logger.info(f"   Total duplicates removed: {total_removed:,}")
    logger.info(f"   Reduction: {(total_removed/total_original)*100:.1f}%")
    
    logger.info(f"\n📋 Per-File Summary:")
    for file_name, result in results.items():
        if result['status'] == 'success':
            logger.info(f"   {file_name}:")
            logger.info(f"      {result['original']:,} → {result['final']:,} (-{result['removed']:,})")
        elif result['status'] == 'skipped':
            logger.info(f"   {file_name}: SKIPPED ({result['reason']})")
    
    logger.info("\n✅ Deduplication complete!")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Deduplicate CSV files')
    parser.add_argument('--data-dir', default='data/csv', help='Directory containing CSV files')
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    logger.info("=" * 80)
    logger.info("CSV DEDUPLICATION")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Deduplicate each file
    results = {}
    
    results['fixtures.csv'] = deduplicate_fixtures(data_dir)
    results['statistics.csv'] = deduplicate_statistics(data_dir)
    results['lineups.csv'] = deduplicate_lineups(data_dir)
    results['sidelined.csv'] = deduplicate_sidelined(data_dir)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
