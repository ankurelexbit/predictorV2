#!/usr/bin/env python3
"""
Validate Historical Data Availability
======================================

Check if teams have enough historical matches for feature calculation.
"""

import os
import sys
from pathlib import Path
import requests
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

def fetch_team_matches(api_key: str, team_id: int, end_date: str, days_back: int = 120):
    """Fetch team matches going back N days."""
    end_dt = datetime.strptime(end_date[:10], '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=days_back)
    start_date = start_dt.strftime('%Y-%m-%d')

    url = f"https://api.sportmonks.com/v3/football/fixtures/between/{start_date}/{end_date}/{team_id}"
    params = {
        'api_token': api_key,
        'include': 'participants;scores;state',
        'filters': 'fixtureStates:5'  # Only finished matches
    }

    try:
        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or 'data' not in data:
            return []

        matches = []
        for fixture in data['data']:
            if fixture.get('state_id') == 5:  # Finished
                matches.append({
                    'fixture_id': fixture['id'],
                    'starting_at': fixture.get('starting_at'),
                })

        return matches
    except Exception as e:
        print(f"Error fetching matches for team {team_id}: {e}")
        return []

def main():
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        print("❌ SPORTMONKS_API_KEY not set")
        sys.exit(1)

    # Read today's predictions
    predictions_file = 'predictions/today_20260201.csv'
    if not Path(predictions_file).exists():
        print(f"❌ Predictions file not found: {predictions_file}")
        sys.exit(1)

    df = pd.read_csv(predictions_file)

    print("=" * 80)
    print("VALIDATING HISTORICAL DATA AVAILABILITY")
    print("=" * 80)
    print(f"Checking {len(df)} fixtures...")
    print()

    results = []

    for idx, row in df.iterrows():
        home_team_id = None  # We need to get this from fixture data
        away_team_id = None

        # For simplicity, let's just check a few sample fixtures
        if idx >= 5:  # Check first 5 fixtures
            break

        print(f"Fixture {idx + 1}: {row['home_team_name']} vs {row['away_team_name']}")
        print(f"  Fixture ID: {row['fixture_id']}")
        print(f"  Match date: {row['match_date']}")

        # Fetch fixture details to get team IDs
        url = f"https://api.sportmonks.com/v3/football/fixtures/{row['fixture_id']}"
        params = {
            'api_token': api_key,
            'include': 'participants'
        }

        try:
            response = requests.get(url, params=params, verify=False, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data and 'data' in data:
                fixture = data['data']
                participants = fixture.get('participants', [])

                home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

                if home_team and away_team:
                    home_team_id = home_team['id']
                    away_team_id = away_team['id']

                    # Check matches for each team
                    for days_back in [60, 90, 120, 180]:
                        home_matches = fetch_team_matches(api_key, home_team_id, row['match_date'][:10], days_back)
                        away_matches = fetch_team_matches(api_key, away_team_id, row['match_date'][:10], days_back)

                        print(f"  {days_back} days back:")
                        print(f"    Home ({row['home_team_name']}): {len(home_matches)} matches")
                        print(f"    Away ({row['away_team_name']}): {len(away_matches)} matches")

                        results.append({
                            'fixture': f"{row['home_team_name']} vs {row['away_team_name']}",
                            'days_back': days_back,
                            'home_matches': len(home_matches),
                            'away_matches': len(away_matches),
                            'sufficient': len(home_matches) >= 10 and len(away_matches) >= 10
                        })

                    print()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()
            continue

    # Summary
    results_df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for days in [60, 90, 120, 180]:
        subset = results_df[results_df['days_back'] == days]
        sufficient = subset['sufficient'].sum()
        total = len(subset)

        print(f"{days} days back:")
        print(f"  Fixtures with ≥10 matches for both teams: {sufficient}/{total}")
        print(f"  Average matches - Home: {subset['home_matches'].mean():.1f}, Away: {subset['away_matches'].mean():.1f}")
        print()

if __name__ == '__main__':
    main()
