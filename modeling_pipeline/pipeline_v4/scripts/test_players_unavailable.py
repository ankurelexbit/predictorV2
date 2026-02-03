"""
Test players_unavailable feature across different months.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.features.feature_orchestrator import FeatureOrchestrator

print("Testing players_unavailable feature variation...")

# Initialize orchestrator
orchestrator = FeatureOrchestrator(data_dir='data/historical')
fixtures_df = orchestrator.fixtures_df

# Sample fixtures from different months
months_to_test = [1, 3, 5, 8, 10, 12]  # Different times of season
sample_fixtures = []

for month in months_to_test:
    month_fixtures = fixtures_df[
        (fixtures_df['league_id'] == 8) &  # Premier League
        (pd.to_datetime(fixtures_df['starting_at']).dt.year == 2023) &
        (pd.to_datetime(fixtures_df['starting_at']).dt.month == month)
    ]
    if len(month_fixtures) > 0:
        sample_fixtures.append(month_fixtures.iloc[0])

print(f"\nTesting {len(sample_fixtures)} fixtures from different months:\n")

results = []
for fixture in sample_fixtures:
    fixture_id = fixture['id']
    match_date = pd.to_datetime(fixture['starting_at'])

    features = orchestrator.generate_features_for_fixture(fixture_id)

    home_unavailable = features['home_players_unavailable']
    away_unavailable = features['away_players_unavailable']

    print(f"Month {match_date.month:2d} ({match_date.strftime('%B'):9s}): "
          f"home={home_unavailable}, away={away_unavailable}")

    results.append({
        'month': match_date.month,
        'home_unavailable': home_unavailable,
        'away_unavailable': away_unavailable
    })

# Analyze variation
print("\nVariation Analysis:")
df = pd.DataFrame(results)
print(f"  home_players_unavailable: min={df['home_unavailable'].min()}, "
      f"max={df['home_unavailable'].max()}, "
      f"unique={df['home_unavailable'].nunique()}")
print(f"  away_players_unavailable: min={df['away_unavailable'].min()}, "
      f"max={df['away_unavailable'].max()}, "
      f"unique={df['away_unavailable'].nunique()}")

if df['home_unavailable'].nunique() > 1 or df['away_unavailable'].nunique() > 1:
    print("\n✓ Features show variation across different months!")
else:
    print("\n⚠️ Features are still constant")
