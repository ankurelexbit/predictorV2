#!/usr/bin/env python3
"""
Check Available Fixtures from SportMonks
=========================================

Quick diagnostic to see what fixtures are available for a date range.

Usage:
    export SPORTMONKS_API_KEY="your_key"
    python3 scripts/check_available_fixtures.py --start-date 2026-01-01 --end-date 2026-01-31
"""

import os
import sys
import argparse
import requests
from datetime import datetime

def check_fixtures(api_key, start_date, end_date):
    """Check what fixtures are available."""
    base_url = "https://api.sportmonks.com/v3/football"
    endpoint = f"fixtures/between/{start_date}/{end_date}"

    print("=" * 80)
    print("CHECKING AVAILABLE FIXTURES")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}")
    print(f"API endpoint: {base_url}/{endpoint}")
    print()

    # Test different state filters
    state_filters = {
        'Upcoming only (1,2,3)': 'fixtureStates:1,2,3',
        'All including finished (1,2,3,5)': 'fixtureStates:1,2,3,5',
        'No filter': None
    }

    for filter_name, state_filter in state_filters.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {filter_name}")
        print(f"{'=' * 80}")

        params = {
            'api_token': api_key,
            'include': 'participants;league;state'
        }

        if state_filter:
            params['filters'] = state_filter

        try:
            response = requests.get(
                f"{base_url}/{endpoint}",
                params=params,
                verify=False,
                timeout=30
            )

            print(f"Status code: {response.status_code}")

            data = response.json()

            if 'data' in data:
                fixtures = data['data']
                print(f"Fixtures found: {len(fixtures)}")

                if fixtures:
                    # Show sample
                    print(f"\nFirst 3 fixtures:")
                    for i, fixture in enumerate(fixtures[:3], 1):
                        state = fixture.get('state', {})
                        print(f"  {i}. ID: {fixture.get('id')}, "
                              f"Date: {fixture.get('starting_at')}, "
                              f"State: {state.get('name', 'Unknown')} (ID: {state.get('id')})")

                        # Show participants
                        participants = fixture.get('participants', [])
                        if len(participants) == 2:
                            home = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                            away = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
                            if home and away:
                                print(f"     {home.get('name')} vs {away.get('name')}")

                # Show state distribution
                if fixtures:
                    states = {}
                    for fixture in fixtures:
                        state_id = fixture.get('state_id')
                        state_name = fixture.get('state', {}).get('name', 'Unknown')
                        key = f"{state_name} (ID: {state_id})"
                        states[key] = states.get(key, 0) + 1

                    print(f"\nState distribution:")
                    for state, count in sorted(states.items()):
                        print(f"  {state}: {count} fixtures")

            elif 'message' in data:
                print(f"API message: {data['message']}")

            elif 'errors' in data:
                print(f"API errors: {data['errors']}")

            else:
                print(f"Unexpected response: {list(data.keys())}")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'=' * 80}")
    print("DONE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Check available fixtures')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        print("ERROR: SPORTMONKS_API_KEY not set")
        sys.exit(1)

    check_fixtures(api_key, args.start_date, args.end_date)


if __name__ == '__main__':
    main()
