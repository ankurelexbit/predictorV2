#!/usr/bin/env python3
"""
Generate training dataset from CSV files using COMPLETE 150-180 feature framework.

This script orchestrates the entire feature engineering pipeline:
1. Loads CSV data
2. Calculates Elo history
3. Generates ALL features for all fixtures (150-180 features per match)
4. Saves training dataset

Usage:
    python scripts/generate_training_data.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.comprehensive_csv_feature_engine import ComprehensiveCSVFeatureEngine
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate training dataset with complete 150-180 feature framework."""
    parser = argparse.ArgumentParser(description='Generate training dataset from CSV data')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)', default=None)
    parser.add_argument('--output', help='Output file', default='data/csv/training_data.csv')
    parser.add_argument('--n-jobs', type=int, default=-1, 
                       help='Number of parallel jobs (-1 = all CPUs, 1 = sequential)')
    
    args = parser.parse_args()
    
    # Initialize comprehensive feature engine
    logger.info("Initializing Comprehensive CSV Feature Engine...")
    logger.info("This implements the complete 150-180 feature framework from FEATURE_FRAMEWORK.md")
    engine = ComprehensiveCSVFeatureEngine()
    
    # Generate training dataset
    training_df = engine.generate_training_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output,
        n_jobs=args.n_jobs
    )
    
    # Show summary
    print("\n" + "=" * 80)
    print("TRAINING DATASET SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(training_df):,}")
    print(f"Total columns: {len(training_df.columns)}")
    
    # Count feature categories
    feature_cols = [col for col in training_df.columns 
                   if col not in ['fixture_id', 'home_team_id', 'away_team_id', 
                                 'starting_at', 'league_id', 'result', 
                                 'target_home_win', 'target_draw', 'target_away_win',
                                 'home_goals', 'away_goals']]
    print(f"Total features: {len(feature_cols)}")
    
    print(f"\nTarget distribution:")
    print(training_df['result'].value_counts())
    print(f"\nDate range: {training_df['starting_at'].min()} to {training_df['starting_at'].max()}")
    print(f"\nLeagues:")
    print(training_df['league_id'].value_counts())
    print(f"\nOutput file: {args.output}")
    print("=" * 80)
    
    # Show sample features
    print("\nSample features (first row):")
    sample_features = training_df.iloc[0][feature_cols].head(20)
    for feat, val in sample_features.items():
        print(f"  {feat}: {val}")
    print(f"  ... and {len(feature_cols) - 20} more features")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
