#!/usr/bin/env python3
"""
Test if league filtering is working in API requests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'pipeline_v3'))

from src.data.sportmonks_client import SportMonksClient
import time

client = SportMonksClient()

print("Testing league filtering...")
print("=" * 80)

# Test 1: Get fixtures WITHOUT league filter
print("\nTest 1: Get ALL fixtures (no filter)")
start = time.time()
all_fixtures = client.get_fixtures_between('2024-01-01', '2024-01-31', league_id=None, include_details=False)
elapsed1 = time.time() - start
print(f"✅ Retrieved {len(all_fixtures)} fixtures in {elapsed1:.2f}s")

# Test 2: Get fixtures WITH league filter
print("\nTest 2: Get Premier League fixtures ONLY (league_id=8)")
start = time.time()
pl_fixtures = client.get_fixtures_between('2024-01-01', '2024-01-31', league_id=8, include_details=False)
elapsed2 = time.time() - start
print(f"✅ Retrieved {len(pl_fixtures)} fixtures in {elapsed2:.2f}s")

print("\n" + "=" * 80)
print("RESULTS:")
print(f"  All fixtures: {len(all_fixtures)} in {elapsed1:.2f}s")
print(f"  PL fixtures: {len(pl_fixtures)} in {elapsed2:.2f}s")
print(f"  Speedup: {elapsed1/elapsed2:.2f}x")

if elapsed2 < elapsed1 * 0.5:
    print("\n✅ FILTERING IS WORKING! API-level filter is faster.")
else:
    print("\n❌ FILTERING NOT WORKING! Both requests take similar time.")
    print("   This means API is returning all fixtures and filtering client-side.")

client.close()
