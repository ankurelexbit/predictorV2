"""
Generate Training Data for V4 Pipeline.

Creates complete training dataset with all 162 features from 3 pillars:
- Pillar 1: 50 fundamental features
- Pillar 2: 60 modern analytics features
- Pillar 3: 52 hidden edge features (40 original + 12 draw parity features)
"""
import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_orchestrator import FeatureOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate training data."""
    parser = argparse.ArgumentParser(description='Generate V4 training data')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--league-id', type=int, help='League ID filter')
    parser.add_argument('--max-fixtures', type=int, help='Max number of fixtures')
    parser.add_argument('--output', default='data/training_data.csv', help='Output file')
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 80)
    logger.info("V4 TRAINING DATA GENERATION")
    logger.info("=" * 80 + "\n")
    
    # Initialize orchestrator
    orchestrator = FeatureOrchestrator(data_dir='data/historical')
    
    # Show feature count
    feature_count = orchestrator.get_feature_count()
    logger.info("\n📊 Feature Breakdown:")
    logger.info(f"   Total columns: {feature_count['total']}")
    logger.info(f"   Feature columns: {feature_count['feature_columns']}")
    logger.info(f"   Metadata columns: {feature_count['metadata']}")
    logger.info(f"   Pillar 1 (approx): {feature_count['pillar1_approx']}")
    logger.info(f"   Pillar 2 (approx): {feature_count['pillar2_approx']}")
    logger.info(f"   Pillar 3 (approx): {feature_count['pillar3_approx']}\n")
    
    # Generate training data
    df = orchestrator.generate_training_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        league_id=args.league_id,
        output_file=args.output,
        max_fixtures=args.max_fixtures
    )
    
    logger.info("\n✅ Training data generation complete!")
    logger.info(f"   Output: {args.output}")
    logger.info(f"   Rows: {len(df)}")
    logger.info(f"   Columns: {len(df.columns)}")
    
    # Show sample
    logger.info("\n📋 Sample features (first 5 rows, first 10 feature columns):")
    feature_cols = [c for c in df.columns if c not in ['fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id', 'match_date', 'home_score', 'away_score', 'result']]
    print(df[feature_cols[:10]].head())
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
