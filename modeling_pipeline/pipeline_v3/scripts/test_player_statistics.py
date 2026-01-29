"""
Test Player Statistics Extraction

Validates that player statistics are correctly extracted from lineups.details.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.enhanced_csv_feature_engine import EnhancedCSVFeatureEngine
import pandas as pd

def test_player_statistics():
    """Test player statistics extraction."""
    
    print("="*60)
    print("Testing Player Statistics Extraction")
    print("="*60)
    
    # Initialize enhanced engine
    print("\n1. Initializing EnhancedCSVFeatureEngine...")
    engine = EnhancedCSVFeatureEngine(data_dir='data/csv')
    
    # Get a sample fixture
    print("\n2. Loading sample fixture...")
    fixtures = pd.read_csv('data/csv/fixtures.csv')
    
    # Get a recent fixture with lineups
    sample_fixture = fixtures.iloc[100]  # Arbitrary sample
    fixture_id = sample_fixture['fixture_id']
    as_of_date = sample_fixture['starting_at']
    
    print(f"   Fixture ID: {fixture_id}")
    print(f"   Date: {as_of_date}")
    print(f"   Home: {sample_fixture['home_team_id']}")
    print(f"   Away: {sample_fixture['away_team_id']}")
    
    # Generate features
    print("\n3. Generating features...")
    features = engine.generate_features_for_fixture(fixture_id, as_of_date)
    
    # Check player features
    print("\n4. Checking player features...")
    player_features = {k: v for k, v in features.items() if 'player' in k or 'rating' in k or 'touches' in k}
    
    print(f"   Total features: {len(features)}")
    print(f"   Player-related features: {len(player_features)}")
    
    # Display sample player features
    print("\n5. Sample player features:")
    sample_keys = [
        'home_rating_avg_5',
        'away_rating_avg_5',
        'home_touches_avg_5',
        'away_touches_avg_5',
        'home_player_rating_avg_5',
        'away_player_rating_avg_5',
        'player_rating_diff_5',
        'player_quality_gap_5',
    ]
    
    for key in sample_keys:
        if key in features:
            print(f"   {key}: {features[key]:.2f}")
        else:
            print(f"   {key}: NOT FOUND")
    
    # Validate feature ranges
    print("\n6. Validating feature ranges...")
    issues = []
    
    for key, value in player_features.items():
        if value is None:
            continue
        
        # Ratings should be 0-10
        if 'rating' in key:
            if not (0 <= value <= 10):
                issues.append(f"{key} = {value} (should be 0-10)")
        
        # Percentages should be 0-100
        if 'completion' in key or 'success' in key:
            if not (0 <= value <= 100):
                issues.append(f"{key} = {value} (should be 0-100)")
    
    if issues:
        print("   ⚠️  Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ All features within valid ranges")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total features generated: {len(features)}")
    print(f"Player features: {len(player_features)}")
    print(f"V3 baseline features: {len(features) - len(player_features)}")
    print(f"Validation issues: {len(issues)}")
    
    if len(player_features) > 0:
        print("\n✅ Player statistics successfully integrated!")
    else:
        print("\n❌ No player statistics found - check lineups.csv data")
    
    return features, player_features


if __name__ == '__main__':
    try:
        features, player_features = test_player_statistics()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
