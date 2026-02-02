#!/usr/bin/env python3
"""Debug prediction probabilities."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline
import pandas as pd

api_key = os.environ.get('SPORTMONKS_API_KEY')
pipeline = StandaloneLivePipeline(api_key)

print("Loading model...")
pipeline.load_model()

print("\nFetching a sample fixture...")

# Get a fixture from today
endpoint = "fixtures/between/2026-02-01/2026-02-02"
params = {
    'include': 'participants;league;state',
    'filters': 'fixtureStates:1,2,3',
    'page': 1
}

data = pipeline._api_call(endpoint, params)
fixtures_data = data['data'][:3]  # Get first 3 fixtures

print(f"\nTesting {len(fixtures_data)} fixtures:\n")

for fixture in fixtures_data:
    participants = fixture.get('participants', [])
    home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
    away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

    if not home_team or not away_team:
        continue

    fixture_dict = {
        'fixture_id': fixture['id'],
        'starting_at': fixture.get('starting_at'),
        'league_id': fixture.get('league_id'),
        'league_name': fixture.get('league', {}).get('name', 'Unknown'),
        'season_id': fixture.get('season_id'),
        'home_team_id': home_team['id'],
        'home_team_name': home_team['name'],
        'away_team_id': away_team['id'],
        'away_team_name': away_team['name'],
        'state_id': fixture.get('state_id')
    }

    print("=" * 80)
    print(f"Match: {home_team['name']} vs {away_team['name']}")
    print("=" * 80)

    # Generate features
    features = pipeline.generate_features(fixture_dict)

    if not features:
        print("⚠️  Failed to generate features")
        continue

    features_df = pd.DataFrame([features])

    # Get raw probabilities
    probas = pipeline.model.predict_proba(features_df)[0]

    # probas order: [away, draw, home]
    away_prob = probas[0]
    draw_prob = probas[1]
    home_prob = probas[2]

    print(f"\nRaw Probabilities:")
    print(f"  Home: {home_prob:.4f} ({home_prob*100:.2f}%)")
    print(f"  Draw: {draw_prob:.4f} ({draw_prob*100:.2f}%)")
    print(f"  Away: {away_prob:.4f} ({away_prob*100:.2f}%)")
    print(f"  Sum: {home_prob + draw_prob + away_prob:.4f}")

    # Check thresholds
    print(f"\nThreshold Check (Home>48%, Draw>35%, Away>45%):")
    print(f"  Home > 0.48? {home_prob > 0.48} {'✅' if home_prob > 0.48 else '❌'}")
    print(f"  Draw > 0.35? {draw_prob > 0.35} {'✅' if draw_prob > 0.35 else '❌'}")
    print(f"  Away > 0.45? {away_prob > 0.45} {'✅' if away_prob > 0.45 else '❌'}")

    # What would be recommended?
    candidates = []
    if home_prob > 0.48:
        candidates.append(('Home', home_prob))
    if draw_prob > 0.35:
        candidates.append(('Draw', draw_prob))
    if away_prob > 0.45:
        candidates.append(('Away', away_prob))

    if candidates:
        bet = max(candidates, key=lambda x: x[1])
        print(f"\n➡️  Recommended Bet: {bet[0]} ({bet[1]*100:.2f}%)")
    else:
        print(f"\n➡️  NO BET (no threshold exceeded)")

    print()
