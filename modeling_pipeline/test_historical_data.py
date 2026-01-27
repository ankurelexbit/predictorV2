#!/usr/bin/env python3
"""
Test Historical Data Collection Performance
"""

import requests
import time
from datetime import datetime, timedelta

API_KEY = "BstrildgLo31OEauPTHrcJzi03WHcZPanhQucdG4KxWThzNVjfM6aH9K9l0v"
BASE_URL = "https://api.sportmonks.com/v3/football"

def test_api_call(endpoint, params=None):
    """Make a test API call and measure performance"""
    if params is None:
        params = {}
    params['api_token'] = API_KEY
    
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
        elapsed = time.time() - start_time
        
        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'response_time': round(elapsed, 3),
            'remaining': response.headers.get('X-RateLimit-Remaining'),
            'data': response.json() if response.status_code == 200 else None,
            'error': response.text if response.status_code != 200 else None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'response_time': round(elapsed, 3)
        }

def main():
    print("=" * 80)
    print("HISTORICAL DATA COLLECTION TEST")
    print("=" * 80)
    print()
    
    # Test 1: Get fixtures for a specific date range
    print("📊 TEST 1: Historical Fixtures (Last 30 days)")
    print("-" * 80)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    result = test_api_call("fixtures", {
        'filters': f'fixtureLeagues:8;fixtureStartingAt:{start_date.strftime("%Y-%m-%d")},{end_date.strftime("%Y-%m-%d")}',
        'per_page': 100
    })
    
    if result['success']:
        fixtures = result['data'].get('data', [])
        pagination = result['data'].get('pagination', {})
        print(f"✅ Retrieved {len(fixtures)} fixtures")
        print(f"📄 Total available: {pagination.get('count', 'N/A')}")
        print(f"⏱️  Response Time: {result['response_time']}s")
        print(f"📈 Remaining: {result['remaining']}/3000")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown')}")
    print()
    
    # Test 2: Get fixture with all includes
    print("📊 TEST 2: Single Fixture with All Includes")
    print("-" * 80)
    
    if result['success'] and fixtures:
        fixture_id = fixtures[0]['id']
        
        includes = [
            'lineups',
            'statistics',
            'events',
            'participants',
            'scores',
            'state',
            'round',
            'venue'
        ]
        
        result = test_api_call(f"fixtures/{fixture_id}", {
            'include': ';'.join(includes)
        })
        
        if result['success']:
            data = result['data'].get('data', {})
            print(f"✅ Fixture ID: {fixture_id}")
            print(f"⏱️  Response Time: {result['response_time']}s")
            print(f"📦 Includes present:")
            for inc in includes:
                has_data = inc in data
                data_count = len(data.get(inc, [])) if isinstance(data.get(inc), list) else ('yes' if has_data else 'no')
                print(f"   - {inc}: {data_count}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown')}")
    print()
    
    # Test 3: Standings
    print("📊 TEST 3: League Standings")
    print("-" * 80)
    
    result = test_api_call("standings/seasons/23700")  # Current PL season
    
    if result['success']:
        standings = result['data'].get('data', [])
        print(f"✅ Retrieved standings")
        print(f"⏱️  Response Time: {result['response_time']}s")
        print(f"📈 Remaining: {result['remaining']}/3000")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown')}")
    print()
    
    # Test 4: Player statistics
    print("📊 TEST 4: Player Statistics")
    print("-" * 80)
    
    result = test_api_call("players", {
        'filters': 'playerIds:161635',  # Example player
        'include': 'statistics'
    })
    
    if result['success']:
        players = result['data'].get('data', [])
        print(f"✅ Retrieved player data")
        print(f"⏱️  Response Time: {result['response_time']}s")
        if players and 'statistics' in players[0]:
            print(f"📊 Statistics available: Yes")
        print(f"📈 Remaining: {result['remaining']}/3000")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown')}")
    print()
    
    # Test 5: Batch request simulation
    print("📊 TEST 5: Batch Request Simulation (5 fixtures with includes)")
    print("-" * 80)
    
    if fixtures:
        total_time = 0
        success_count = 0
        
        for i, fixture in enumerate(fixtures[:5]):
            result = test_api_call(f"fixtures/{fixture['id']}", {
                'include': 'lineups;statistics;events'
            })
            
            if result['success']:
                success_count += 1
                total_time += result['response_time']
                print(f"   Fixture {i+1}: ✅ {result['response_time']}s")
            else:
                print(f"   Fixture {i+1}: ❌ Failed")
            
            time.sleep(0.1)  # Small delay
        
        print()
        print(f"📊 Batch Results:")
        print(f"   Success Rate: {success_count}/5")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Avg Time per Request: {total_time/5:.3f}s")
        print(f"   Estimated time for 100 fixtures: {(total_time/5)*100:.1f}s ({(total_time/5)*100/60:.1f} min)")
    print()
    
    print("=" * 80)
    print("HISTORICAL DATA TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
