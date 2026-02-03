"""
Test script for newly implemented features:
1. xG vs top/bottom half (4 features)
2. Players unavailable (2 features)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.data.json_loader import JSONDataLoader
from src.features.feature_orchestrator import FeatureOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_new_features():
    """Test the newly implemented features."""

    print("=" * 80)
    print("Testing New Feature Implementations")
    print("=" * 80)

    # Initialize feature orchestrator (which will load data)
    print("\n1. Initializing feature orchestrator...")
    try:
        orchestrator = FeatureOrchestrator(data_dir='data/historical')
        fixtures_df = orchestrator.fixtures_df
        print(f"✓ Loaded {len(fixtures_df)} fixtures")
        print("✓ Feature orchestrator initialized")
    except Exception as e:
        print(f"✗ Failed to initialize orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with a sample fixture
    print("\n2. Testing feature generation on sample fixtures...")

    # Get a sample of fixtures from different leagues/dates
    sample_fixtures = fixtures_df[
        (fixtures_df['league_id'] == 8) &  # Premier League
        (pd.to_datetime(fixtures_df['starting_at']).dt.year == 2023)
    ].head(5)

    if len(sample_fixtures) == 0:
        print("✗ No sample fixtures found")
        return

    print(f"\nTesting on {len(sample_fixtures)} fixtures...")

    features_to_check = [
        'home_xg_vs_top_half',
        'away_xg_vs_top_half',
        'home_xga_vs_bottom_half',
        'away_xga_vs_bottom_half',
        'home_players_unavailable',
        'away_players_unavailable',
    ]

    results = []

    for idx, fixture in sample_fixtures.iterrows():
        fixture_id = fixture['id']
        home_team = fixture['home_team_id']
        away_team = fixture['away_team_id']
        match_date = pd.to_datetime(fixture['starting_at'])

        print(f"\n  Fixture {fixture_id}: Team {home_team} vs {away_team} ({match_date.date()})")

        try:
            # Generate features (pass fixture_id, not the full fixture)
            features = orchestrator.generate_features_for_fixture(fixture_id)

            # Check our new features
            feature_values = {}
            for feature in features_to_check:
                value = features.get(feature, None)
                feature_values[feature] = value
                print(f"    {feature}: {value}")

            results.append(feature_values)

        except Exception as e:
            print(f"    ✗ Error generating features: {e}")
            import traceback
            traceback.print_exc()

    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    if len(results) == 0:
        print("✗ No features generated successfully")
        return

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)

    print("\nFeature Statistics:")
    print("-" * 80)
    for feature in features_to_check:
        if feature in results_df.columns:
            values = results_df[feature].dropna()
            if len(values) > 0:
                print(f"\n{feature}:")
                print(f"  Count: {len(values)}")
                print(f"  Min: {values.min():.3f}")
                print(f"  Max: {values.max():.3f}")
                print(f"  Mean: {values.mean():.3f}")
                print(f"  Std: {values.std():.3f}")

                # Check if constant
                unique_count = values.nunique()
                if unique_count == 1:
                    print(f"  ⚠️  WARNING: All values are constant ({values.iloc[0]})")
                else:
                    print(f"  ✓ Variable ({unique_count} unique values)")

    # Check for issues
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    issues_found = False

    for feature in features_to_check:
        if feature in results_df.columns:
            values = results_df[feature].dropna()

            # Check 1: Constant values
            if values.nunique() == 1:
                print(f"✗ {feature}: CONSTANT (all values = {values.iloc[0]})")
                issues_found = True

            # Check 2: NaN values
            nan_count = results_df[feature].isna().sum()
            if nan_count > 0:
                print(f"⚠️  {feature}: {nan_count} NaN values")

            # Check 3: Expected ranges
            if 'xg_vs_top' in feature or 'xga_vs_bottom' in feature:
                # xG should be in reasonable range (0-4)
                if values.min() < 0 or values.max() > 5:
                    print(f"⚠️  {feature}: Values outside expected range (0-5)")

            elif 'players_unavailable' in feature:
                # Players unavailable should be 0-8
                if values.min() < 0 or values.max() > 8:
                    print(f"⚠️  {feature}: Values outside expected range (0-8)")

    if not issues_found:
        print("✓ All features look good!")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    test_new_features()
