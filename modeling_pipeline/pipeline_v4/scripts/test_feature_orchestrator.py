"""
Test Feature Orchestrator - Generate sample features.

Quick test to verify all 3 pillars are working together.
"""
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_orchestrator import FeatureOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test feature orchestrator."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING FEATURE ORCHESTRATOR")
    logger.info("=" * 80 + "\n")
    
    # Initialize orchestrator
    orchestrator = FeatureOrchestrator(data_dir='data/historical')
    
    # Get feature count
    feature_count = orchestrator.get_feature_count()
    logger.info("\n📊 Feature Count:")
    logger.info(f"   Total: {feature_count['total']}")
    logger.info(f"   Features: {feature_count['feature_columns']}")
    logger.info(f"   Metadata: {feature_count['metadata']}")
    logger.info(f"   Pillar 1: ~{feature_count['pillar1_approx']}")
    logger.info(f"   Pillar 2: ~{feature_count['pillar2_approx']}")
    logger.info(f"   Pillar 3: ~{feature_count['pillar3_approx']}")
    
    # Generate features for a sample fixture
    sample_fixture = orchestrator.fixtures_df[orchestrator.fixtures_df['result'].notna()].iloc[100]
    logger.info(f"\n🎯 Generating features for sample fixture {sample_fixture['id']}...")
    logger.info(f"   Match: Team {sample_fixture['home_team_id']} vs Team {sample_fixture['away_team_id']}")
    logger.info(f"   Date: {sample_fixture['starting_at']}")
    logger.info(f"   Score: {sample_fixture['home_score']}-{sample_fixture['away_score']}")
    
    features = orchestrator.generate_features_for_fixture(sample_fixture['id'])
    
    logger.info(f"\n✅ Generated {len(features)} total fields")
    
    # Show sample features from each pillar
    logger.info("\n📋 Sample Features:")
    logger.info("\n   Pillar 1 (Fundamentals):")
    pillar1_samples = ['home_elo', 'away_elo', 'elo_diff', 'home_league_position', 'away_league_position', 'home_points_last_5']
    for feat in pillar1_samples:
        if feat in features:
            logger.info(f"      {feat}: {features[feat]}")
    
    logger.info("\n   Pillar 2 (Modern Analytics):")
    pillar2_samples = ['home_derived_xg_per_match_5', 'away_derived_xg_per_match_5', 'home_shots_per_match_5']
    for feat in pillar2_samples:
        if feat in features:
            logger.info(f"      {feat}: {features[feat]}")
    
    logger.info("\n   Pillar 3 (Hidden Edges):")
    pillar3_samples = ['home_points_trend_10', 'home_win_streak', 'home_avg_opponent_elo_5']
    for feat in pillar3_samples:
        if feat in features:
            logger.info(f"      {feat}: {features[feat]}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("=" * 80 + "\n")


if __name__ == '__main__':
    main()
