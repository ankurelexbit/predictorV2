#!/usr/bin/env python3
"""
Quick script to regenerate training data with updated parent pipeline features.
Uses the ComprehensiveCSVFeatureEngine which now has parent features.
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.comprehensive_csv_feature_engine import ComprehensiveCSVFeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Regenerate training data with parent pipeline features."""
    
    logger.info("=" * 80)
    logger.info("REGENERATING TRAINING DATA WITH PARENT PIPELINE FEATURES")
    logger.info("=" * 80)
    
    # Initialize feature engine (now has parent features!)
    logger.info("\nInitializing feature engine...")
    engine = ComprehensiveCSVFeatureEngine(data_dir='data/csv')
    
    # Load fixtures
    fixtures_df = engine.fixtures
    logger.info(f"Loaded {len(fixtures_df)} fixtures")
    
    # Filter completed fixtures
    completed = fixtures_df[fixtures_df['result'].notna()].copy()
    logger.info(f"Found {len(completed)} completed fixtures")
    
    # Generate features for each fixture
    logger.info("\nGenerating features...")
    all_features = []
    
    for idx, row in tqdm(completed.iterrows(), total=len(completed), desc="Processing"):
        fixture_id = row['fixture_id']
        as_of_date = row['starting_at']
        
        try:
            features = engine.generate_features_for_fixture(fixture_id, as_of_date)
            all_features.append(features)
        except Exception as e:
            logger.warning(f"Failed to generate features for fixture {fixture_id}: {e}")
            continue
    
    # Convert to DataFrame
    logger.info(f"\nGenerated features for {len(all_features)} fixtures")
    df = pd.DataFrame(all_features)
    
    # Save
    output_path = 'data/csv/training_data_complete_v2.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Saved {len(df)} rows with {len(df.columns)} features to {output_path}")
    
    # Show sample of parent features
    parent_features = [
        'home_position', 'away_position', 'position_diff',
        'home_xg_5', 'away_xg_5', 'home_xg_10', 'away_xg_10',
        'home_attack_strength_5', 'away_defense_strength_5',
        'home_wins_3', 'home_wins_5', 'home_form_5', 'form_diff_5'
    ]
    
    logger.info("\n" + "=" * 80)
    logger.info("PARENT PIPELINE FEATURES VERIFICATION")
    logger.info("=" * 80)
    
    for feat in parent_features:
        if feat in df.columns:
            logger.info(f"✅ {feat}: {df[feat].notna().sum()}/{len(df)} non-null")
        else:
            logger.warning(f"❌ {feat}: MISSING")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE!")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
