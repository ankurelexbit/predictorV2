#!/usr/bin/env python3
"""
Test SportMonks API Performance
Tests rate limits, response times, and available endpoints
"""

import requests
import time
from datetime import datetime
import json

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
        
        # Get rate limit headers
        rate_limit_info = {
            'remaining': response.headers.get('X-RateLimit-Remaining'),
            'limit': response.headers.get('X-RateLimit-Limit'),
            'reset': response.headers.get('X-RateLimit-Reset'),
        }
        
        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'response_time': round(elapsed, 3),
            'rate_limit': rate_limit_info,
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
    print("SPORTMONKS API PERFORMANCE TEST")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Key: {API_KEY[:20]}...")
    print()
    
    # Test 1: Check subscription tier
    print("📊 TEST 1: Subscription Information")
    print("-" * 80)
    result = test_api_call("my/resources")
    if result['success']:
        resources = result['data'].get('data', [])
        print(f"✅ Status: Active")
        print(f"⏱️  Response Time: {result['response_time']}s")
        print(f"📈 Rate Limit: {result['rate_limit']['remaining']}/{result['rate_limit']['limit']}")
        print(f"\n📦 Available Resources ({len(resources)}):")
        for resource in resources[:10]:  # Show first 10
            print(f"   - {resource}")
        if len(resources) > 10:
            print(f"   ... and {len(resources) - 10} more")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    print()
    
    # Test 2: Fixtures endpoint
    print("📊 TEST 2: Fixtures Endpoint")
    print("-" * 80)
    result = test_api_call("fixtures", {
        'filters': 'fixtureLeagues:8',  # Premier League
        'per_page': 5
    })
    if result['success']:
        fixtures = result['data'].get('data', [])
        print(f"✅ Retrieved {len(fixtures)} fixtures")
        print(f"⏱️  Response Time: {result['response_time']}s")
        print(f"📈 Rate Limit: {result['rate_limit']['remaining']}/{result['rate_limit']['limit']}")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    print()
    
    # Test 3: Rate limit stress test
    print("📊 TEST 3: Rate Limit Stress Test (10 rapid requests)")
    print("-" * 80)
    response_times = []
    success_count = 0
    
    for i in range(10):
        result = test_api_call("leagues", {'per_page': 1})
        response_times.append(result['response_time'])
        if result['success']:
            success_count += 1
            print(f"   Request {i+1}: ✅ {result['response_time']}s (Remaining: {result['rate_limit']['remaining']})")
        else:
            print(f"   Request {i+1}: ❌ {result.get('error', 'Failed')}")
        time.sleep(0.1)  # Small delay
    
    print()
    print(f"📊 Results:")
    print(f"   Success Rate: {success_count}/10 ({success_count*10}%)")
    print(f"   Avg Response Time: {sum(response_times)/len(response_times):.3f}s")
    print(f"   Min Response Time: {min(response_times):.3f}s")
    print(f"   Max Response Time: {max(response_times):.3f}s")
    print()
    
    # Test 4: Check specific includes
    print("📊 TEST 4: Testing Includes (lineups, statistics, etc.)")
    print("-" * 80)
    
    # Get a recent fixture first
    result = test_api_call("fixtures", {
        'filters': 'fixtureLeagues:8',
        'per_page': 1
    })
    
    if result['success'] and result['data'].get('data'):
        fixture_id = result['data']['data'][0]['id']
        print(f"Testing with fixture ID: {fixture_id}")
        
        includes_to_test = [
            'lineups',
            'statistics',
            'events',
            'participants',
            'scores'
        ]
        
        for include in includes_to_test:
            result = test_api_call(f"fixtures/{fixture_id}", {
                'include': include
            })
            if result['success']:
                has_data = include in result['data'].get('data', {})
                print(f"   {include}: ✅ {result['response_time']}s {'(has data)' if has_data else '(no data)'}")
            else:
                print(f"   {include}: ❌ Failed")
            time.sleep(0.2)
    else:
        print("❌ Could not get fixture for testing")
    print()
    
    # Test 5: Check current rate limit status
    print("📊 TEST 5: Final Rate Limit Status")
    print("-" * 80)
    result = test_api_call("leagues", {'per_page': 1})
    if result['success']:
        print(f"📈 Remaining Requests: {result['rate_limit']['remaining']}/{result['rate_limit']['limit']}")
        if result['rate_limit']['reset']:
            print(f"🔄 Resets at: {result['rate_limit']['reset']}")
    print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
