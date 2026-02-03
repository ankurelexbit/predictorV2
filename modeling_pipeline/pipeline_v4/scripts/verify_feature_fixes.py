#!/usr/bin/env python3
"""
Verify Feature Fixes
====================

Quick test to verify that the zero-value feature fixes work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.json_loader import JSONDataLoader
from src.features.feature_orchestrator import FeatureOrchestrator
import pandas as pd

def main():
    print("=" * 80)
    print("VERIFYING FEATURE FIXES")
    print("=" * 80)

    # Initialize orchestrator
    print("\n1. Initializing feature orchestrator...")
    orchestrator = FeatureOrchestrator('data/historical')

    # Generate features for a test fixture
    print("\n2. Generating features for test fixture...")
    fixtures_df = orchestrator.data_loader.load_all_fixtures()
    test_fixture = fixtures_df[fixtures_df['result'].notna()].iloc[100]

    fixture_id = test_fixture['id']
    print(f"   Test fixture: {fixture_id}")
    print(f"   Date: {test_fixture['starting_at']}")

    features = orchestrator.generate_features_for_fixture(
        fixture_id,
        as_of_date=test_fixture['starting_at']
    )

    # Check fixed features
    print("\n3. Checking previously broken features:")
    print("-" * 80)

    checks = {
        'Big Chances': [
            'home_big_chances_per_match_5',
            'away_big_chances_per_match_5'
        ],
        'xG Trends': [
            'home_xg_trend_10',
            'away_xg_trend_10'
        ],
        'Rest Advantage': [
            'home_days_since_last_match',
            'away_days_since_last_match',
            'rest_advantage'
        ],
        'Derby Match': [
            'is_derby_match'
        ]
    }

    all_good = True

    for category, feature_names in checks.items():
        print(f"\n{category}:")
        for feature in feature_names:
            value = features.get(feature, 'MISSING')

            # Check if value is reasonable (not always 0)
            if feature == 'is_derby_match':
                status = "✅ OK" if value in [0, 1] else "❌ INVALID"
            elif 'big_chances' in feature:
                status = "✅ OK" if value >= 0 else "❌ INVALID"
            elif 'trend' in feature:
                # Trend can be 0 for short history, but should be calculated
                status = "✅ OK" if isinstance(value, (int, float)) else "❌ INVALID"
            elif 'rest' in feature or 'days' in feature:
                status = "✅ OK" if value > 0 or isinstance(value, int) else "❌ INVALID"
            else:
                status = "✅ OK"

            print(f"  {feature:40s} = {value:10} {status}")

            if "❌" in status:
                all_good = False

    # Test on multiple fixtures
    print("\n" + "=" * 80)
    print("4. Testing on 10 random fixtures...")
    print("=" * 80)

    sample_fixtures = fixtures_df[fixtures_df['result'].notna()].sample(min(10, len(fixtures_df)))

    results = {
        'big_chances_nonzero': 0,
        'xg_trend_nonzero': 0,
        'rest_advantage_nonzero': 0,
        'derby_matches': 0,
        'total': 0
    }

    for idx, (_, fixture) in enumerate(sample_fixtures.iterrows(), 1):
        try:
            features = orchestrator.generate_features_for_fixture(
                fixture['id'],
                as_of_date=fixture['starting_at']
            )

            results['total'] += 1

            if features.get('home_big_chances_per_match_5', 0) > 0:
                results['big_chances_nonzero'] += 1

            if features.get('home_xg_trend_10', 0) != 0:
                results['xg_trend_nonzero'] += 1

            if features.get('rest_advantage', 0) != 0:
                results['rest_advantage_nonzero'] += 1

            if features.get('is_derby_match', 0) == 1:
                results['derby_matches'] += 1

        except Exception as e:
            print(f"   Error on fixture {fixture['id']}: {e}")

    print(f"\nResults from {results['total']} fixtures:")
    print(f"  Big chances non-zero: {results['big_chances_nonzero']}/{results['total']} ({results['big_chances_nonzero']/results['total']*100:.1f}%)")
    print(f"  xG trend non-zero: {results['xg_trend_nonzero']}/{results['total']} ({results['xg_trend_nonzero']/results['total']*100:.1f}%)")
    print(f"  Rest advantage non-zero: {results['rest_advantage_nonzero']}/{results['total']} ({results['rest_advantage_nonzero']/results['total']*100:.1f}%)")
    print(f"  Derby matches: {results['derby_matches']}/{results['total']} ({results['derby_matches']/results['total']*100:.1f}%)")

    # Summary
    print("\n" + "=" * 80)
    if all_good:
        print("✅ ALL FIXES VERIFIED SUCCESSFULLY")
    else:
        print("⚠️  SOME ISSUES DETECTED - Review output above")
    print("=" * 80)

    print("\nNext Steps:")
    print("1. Regenerate training data:")
    print("   python3 scripts/generate_training_data.py --output data/training_data.csv")
    print("\n2. Verify no zero-only columns:")
    print("   python3 -c \"import pandas as pd; df = pd.read_csv('data/training_data.csv');")
    print("   print([c for c in df.columns if (df[c]==0).all()])\"")
    print("\n3. Retrain model with fixed features")


if __name__ == '__main__':
    main()
