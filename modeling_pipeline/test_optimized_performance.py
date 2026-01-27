#!/usr/bin/env python3
"""
Test Data Collection Performance After Optimization
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'pipeline_v3'))

from src.data.sportmonks_client import SportMonksClient

def test_performance():
    print("=" * 80)
    print("DATA COLLECTION PERFORMANCE TEST (AFTER OPTIMIZATION)")
    print("=" * 80)
    print()
    
    client = SportMonksClient()
    
    # Test 1: Single fixture with full includes
    print("📊 TEST 1: Single Fixture with Full Includes")
    print("-" * 80)
    
    start = time.time()
    fixture = client.get_fixture_by_id(463, includes=['lineups', 'statistics', 'events', 'participants'])
    elapsed = time.time() - start
    
    print(f"✅ Retrieved fixture with all includes")
    print(f"⏱️  Time: {elapsed:.3f}s")
    print(f"📦 Data size: {len(str(fixture))} bytes")
    print()
    
    # Test 2: Batch of 10 fixtures
    print("📊 TEST 2: Batch of 10 Fixtures (Sequential)")
    print("-" * 80)
    
    # Get recent fixtures
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    fixtures_list = client.get_fixtures_between(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        league_id=8,
        include_details=False
    )
    
    if len(fixtures_list) >= 10:
        fixture_ids = [f['id'] for f in fixtures_list[:10]]
        
        start = time.time()
        for fid in fixture_ids:
            client.get_fixture_by_id(fid, includes=['lineups', 'statistics'])
        elapsed = time.time() - start
        
        print(f"✅ Retrieved 10 fixtures with includes")
        print(f"⏱️  Total Time: {elapsed:.3f}s")
        print(f"⏱️  Avg per Fixture: {elapsed/10:.3f}s")
        print(f"📈 Throughput: {10/elapsed:.2f} fixtures/sec")
        print()
        
        # Calculate projections
        print("📊 Performance Projections:")
        print(f"   100 fixtures: {elapsed*10:.1f}s ({elapsed*10/60:.1f} min)")
        print(f"   380 fixtures (1 season): {elapsed*38:.1f}s ({elapsed*38/60:.1f} min)")
        print(f"   1000 fixtures: {elapsed*100:.1f}s ({elapsed*100/60:.1f} min)")
    else:
        print(f"⚠️  Only {len(fixtures_list)} fixtures available")
    
    print()
    
    # Test 3: Rate limiter status
    print("📊 TEST 3: Rate Limiter Status")
    print("-" * 80)
    print(f"📈 Requests in last minute: {len(client.rate_limiter.request_times)}")
    print(f"📊 Rate limit: {client.rate_limiter.requests_per_minute} req/min")
    if client.rate_limiter.last_remaining:
        print(f"🔄 API remaining: {client.rate_limiter.last_remaining}")
    print()
    
    client.close()
    
    print("=" * 80)
    print("PERFORMANCE TEST COMPLETE")
    print("=" * 80)
    print()
    print("🎯 Expected Improvements:")
    print("   - No artificial 0.2s delays")
    print("   - Smart rate limiting (only waits when needed)")
    print("   - Response time naturally paces requests")
    print("   - 3-6x faster than before!")

if __name__ == "__main__":
    test_performance()
