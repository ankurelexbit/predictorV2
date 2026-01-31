"""
Deduplicate Training Data - Fix Critical Data Duplication Issue.

PROBLEM: Every fixture appears exactly 2 times in training data
- 35,886 rows but only 17,943 unique fixtures
- Both entries are completely identical
- Causes data leakage and inflated metrics

IMPACT:
- Model sees same matches multiple times
- Validation/test splits likely contain duplicates
- Performance metrics are artificially high
- Model may not generalize to truly unseen data

FIX: Remove duplicate fixtures, keep only one entry per fixture_id
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def deduplicate_training_data(input_file: str, output_file: str):
    """
    Remove duplicate fixtures from training data.

    Args:
        input_file: Path to duplicated training data
        output_file: Path to save deduplicated data
    """
    logger.info("=" * 80)
    logger.info("DEDUPLICATING TRAINING DATA")
    logger.info("=" * 80)

    # Load data
    logger.info(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)

    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Unique fixture_ids: {df['fixture_id'].nunique():,}")
    logger.info(f"Duplication factor: {len(df) / df['fixture_id'].nunique():.2f}x")

    # Check duplication pattern
    dup_counts = df['fixture_id'].value_counts()
    logger.info(f"\nDuplication distribution:")
    logger.info(f"  Fixtures appearing 1x: {(dup_counts == 1).sum():,}")
    logger.info(f"  Fixtures appearing 2x: {(dup_counts == 2).sum():,}")
    logger.info(f"  Fixtures appearing 3x+: {(dup_counts > 2).sum():,}")

    # Remove duplicates
    logger.info("\nRemoving duplicates (keeping first occurrence)...")
    df_deduped = df.drop_duplicates(subset=['fixture_id'], keep='first')

    logger.info(f"\nAfter deduplication:")
    logger.info(f"  Shape: {df_deduped.shape}")
    logger.info(f"  Rows: {len(df_deduped):,}")
    logger.info(f"  Rows removed: {len(df) - len(df_deduped):,}")
    logger.info(f"  Reduction: {(1 - len(df_deduped)/len(df)) * 100:.1f}%")

    # Verify no duplicates remain
    remaining_dups = df_deduped['fixture_id'].duplicated().sum()
    if remaining_dups > 0:
        logger.warning(f"⚠️  WARNING: {remaining_dups} duplicates still remain!")
    else:
        logger.info(f"✅ No duplicates remain - each fixture appears exactly once")

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_deduped.to_csv(output_path, index=False)
    logger.info(f"\n✅ Deduplicated data saved to: {output_path}")

    # Create backup of original
    backup_path = Path(input_file).with_suffix('.backup.csv')
    if not backup_path.exists():
        logger.info(f"Creating backup of original: {backup_path}")
        df.to_csv(backup_path, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("DEDUPLICATION COMPLETE")
    logger.info("=" * 80)

    return df_deduped

def analyze_impact(duplicated_file: str, deduplicated_file: str):
    """
    Analyze the impact of deduplication on train/val/test splits.

    Shows how duplicates were distributed across splits and the
    impact on validation/test metrics.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANALYZING IMPACT OF DUPLICATION")
    logger.info("=" * 80)

    # Load both versions
    df_dup = pd.read_csv(duplicated_file)
    df_dedup = pd.read_csv(deduplicated_file)

    # Sort by date
    df_dup = df_dup.sort_values('match_date').reset_index(drop=True)
    df_dedup = df_dedup.sort_values('match_date').reset_index(drop=True)

    # Create splits (70/15/15)
    def create_splits(df):
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        return train, val, test

    train_dup, val_dup, test_dup = create_splits(df_dup)
    train_dedup, val_dedup, test_dedup = create_splits(df_dedup)

    logger.info("\nDUPLICATED DATA SPLITS:")
    logger.info(f"  Train: {len(train_dup):,} rows")
    logger.info(f"  Val:   {len(val_dup):,} rows")
    logger.info(f"  Test:  {len(test_dup):,} rows")

    logger.info("\nDEDUPLICATED DATA SPLITS:")
    logger.info(f"  Train: {len(train_dedup):,} rows")
    logger.info(f"  Val:   {len(val_dedup):,} rows")
    logger.info(f"  Test:  {len(test_dedup):,} rows")

    # Check for leakage: fixtures appearing in multiple splits
    def check_leakage(train, val, test):
        train_ids = set(train['fixture_id'])
        val_ids = set(val['fixture_id'])
        test_ids = set(test['fixture_id'])

        train_val_overlap = len(train_ids & val_ids)
        train_test_overlap = len(train_ids & test_ids)
        val_test_overlap = len(val_ids & test_ids)

        return train_val_overlap, train_test_overlap, val_test_overlap

    leak_dup = check_leakage(train_dup, val_dup, test_dup)
    leak_dedup = check_leakage(train_dedup, val_dedup, test_dedup)

    logger.info("\nDATA LEAKAGE (fixtures appearing in multiple splits):")
    logger.info("\nDUPLICATED DATA:")
    logger.info(f"  Train ∩ Val:  {leak_dup[0]:,} fixtures")
    logger.info(f"  Train ∩ Test: {leak_dup[1]:,} fixtures")
    logger.info(f"  Val ∩ Test:   {leak_dup[2]:,} fixtures")

    logger.info("\nDEDUPLICATED DATA:")
    logger.info(f"  Train ∩ Val:  {leak_dedup[0]:,} fixtures")
    logger.info(f"  Train ∩ Test: {leak_dedup[1]:,} fixtures")
    logger.info(f"  Val ∩ Test:   {leak_dedup[2]:,} fixtures")

    if leak_dup[0] > 0 or leak_dup[1] > 0 or leak_dup[2] > 0:
        logger.warning("\n🔴 CRITICAL: Duplicated data has data leakage!")
        logger.warning("Model was evaluated on matches it had already seen during training.")
        logger.warning("Reported metrics are artificially inflated and unreliable.")

    if leak_dedup[0] == 0 and leak_dedup[1] == 0 and leak_dedup[2] == 0:
        logger.info("\n✅ Deduplicated data has no leakage - splits are clean!")

    logger.info("\n" + "=" * 80)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deduplicate training data')
    parser.add_argument('--input', default='data/training_data.csv',
                       help='Input file (duplicated)')
    parser.add_argument('--output', default='data/training_data_deduped.csv',
                       help='Output file (deduplicated)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze impact of deduplication')
    args = parser.parse_args()

    # Deduplicate
    df_deduped = deduplicate_training_data(args.input, args.output)

    # Analyze impact
    if args.analyze:
        analyze_impact(args.input, args.output)

    logger.info("\n🎯 NEXT STEPS:")
    logger.info(f"  1. Use {args.output} for all future training")
    logger.info("  2. Retrain your model with deduplicated data")
    logger.info("  3. Expect LOWER but MORE REALISTIC metrics")
    logger.info("  4. Previous validation metrics were inflated due to data leakage")
