#!/usr/bin/env python3
"""
True Live Prediction System - Fetches from SportMonks API
==========================================================

Fetches upcoming fixtures from SportMonks API, generates features,
and provides betting recommendations with real odds.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import requests
from datetime import datetime, timedelta
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get API key from environment
API_KEY = os.getenv('SPORTMONKS_API_KEY')
BASE_URL = 'https://api.sportmonks.com/v3/football'


def fetch_upcoming_fixtures(start_date, end_date, league_ids=None):
    """
    Fetch upcoming fixtures from SportMonks API.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        league_ids: List of league IDs to filter (None = all)
    
    Returns:
        List of fixture dictionaries
    """
    logger.info(f"Fetching fixtures from {start_date} to {end_date}...")
    
    url = f"{BASE_URL}/fixtures/between/{start_date}/{end_date}"
    params = {
        'api_token': API_KEY,
        'include': 'participants;odds.bookmaker;odds.market',
        'per_page': 100
    }
    
    if league_ids:
        params['leagues'] = ','.join(map(str, league_ids))
    
    all_fixtures = []
    page = 1
    
    while True:
        params['page'] = page
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code}")
            break
        
        data = response.json()
        fixtures = data.get('data', [])
        
        if not fixtures:
            break
        
        all_fixtures.extend(fixtures)
        logger.info(f"  Page {page}: {len(fixtures)} fixtures")
        
        # Check if there are more pages
        pagination = data.get('pagination', {})
        if page >= pagination.get('total_pages', 1):
            break
        
        page += 1
    
    logger.info(f"✅ Fetched {len(all_fixtures)} fixtures")
    return all_fixtures


def extract_odds(fixture):
    """Extract 1X2 odds from fixture."""
    odds_data = fixture.get('odds', [])
    
    for bookmaker_odds in odds_data:
        # Prefer Bet365 (bookmaker_id=2)
        if bookmaker_odds.get('bookmaker_id') == 2:
            for market in bookmaker_odds.get('markets', []):
                if market.get('id') == 1:  # 1X2 market
                    selections = market.get('selections', [])
                    odds = {}
                    for sel in selections:
                        label = sel.get('label')
                        value = sel.get('odds')
                        if label == '1':
                            odds['home'] = float(value)
                        elif label == 'X':
                            odds['draw'] = float(value)
                        elif label == '2':
                            odds['away'] = float(value)
                    
                    if len(odds) == 3:
                        return odds
    
    return None


def prepare_fixture_for_prediction(fixture):
    """
    Prepare a fixture for prediction by extracting basic info.
    Note: Full feature engineering would require historical data.
    """
    participants = fixture.get('participants', [])
    home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), {})
    away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), {})
    
    odds = extract_odds(fixture)
    
    return {
        'fixture_id': fixture.get('id'),
        'starting_at': fixture.get('starting_at'),
        'league_id': fixture.get('league_id'),
        'home_team_id': home_team.get('id'),
        'home_team_name': home_team.get('name'),
        'away_team_id': away_team.get('id'),
        'away_team_name': away_team.get('name'),
        'odds_home': odds.get('home') if odds else None,
        'odds_draw': odds.get('draw') if odds else None,
        'odds_away': odds.get('away') if odds else None
    }


def main():
    """Main execution."""
    logger.info("="*70)
    logger.info("TRUE LIVE PREDICTION SYSTEM - JANUARY 2026")
    logger.info("="*70)
    
    if not API_KEY:
        logger.error("❌ SPORTMONKS_API_KEY not found in environment!")
        logger.error("Please set it in your .env file")
        return
    
    # Fetch January 2026 fixtures
    start_date = '2026-01-01'
    end_date = '2026-01-31'
    
    # Focus on major leagues
    league_ids = [
        8,    # Premier League
        564,  # Championship
        39,   # La Liga
        140,  # Serie A
        78,   # Bundesliga
        135   # Ligue 1
    ]
    
    fixtures = fetch_upcoming_fixtures(start_date, end_date, league_ids)
    
    if not fixtures:
        logger.warning("⚠️ No fixtures found for January 2026")
        logger.info("\nNote: This might be because:")
        logger.info("  1. Fixtures haven't been scheduled yet")
        logger.info("  2. API key doesn't have access")
        logger.info("  3. Date range is incorrect")
        return
    
    # Prepare fixtures
    logger.info("\n📋 Preparing fixtures for prediction...")
    prepared_fixtures = []
    
    for fixture in fixtures:
        try:
            prepared = prepare_fixture_for_prediction(fixture)
            prepared_fixtures.append(prepared)
        except Exception as e:
            logger.warning(f"Error preparing fixture {fixture.get('id')}: {e}")
    
    fixtures_df = pd.DataFrame(prepared_fixtures)
    logger.info(f"✅ Prepared {len(fixtures_df)} fixtures")
    
    # Show sample
    logger.info("\n📊 Sample Upcoming Fixtures:")
    logger.info("-" * 100)
    logger.info(f"{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Odds (H/D/A)':<20}")
    logger.info("-" * 100)
    
    for _, row in fixtures_df.head(20).iterrows():
        date_str = row['starting_at'][:10] if pd.notna(row['starting_at']) else 'Unknown'
        home = str(row['home_team_name'])[:24]
        away = str(row['away_team_name'])[:24]
        
        if pd.notna(row['odds_home']):
            odds_str = f"{row['odds_home']:.2f} / {row['odds_draw']:.2f} / {row['odds_away']:.2f}"
        else:
            odds_str = "No odds yet"
        
        logger.info(f"{date_str:<12} {home:<25} {away:<25} {odds_str:<20}")
    
    # Save fixtures
    output_dir = Path(__file__).parent.parent / 'predictions'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'january_2026_upcoming_fixtures.csv'
    fixtures_df.to_csv(output_file, index=False)
    
    logger.info(f"\n💾 Saved fixtures to: {output_file}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Total fixtures fetched: {len(fixtures_df)}")
    logger.info(f"Fixtures with odds: {fixtures_df['odds_home'].notna().sum()}")
    logger.info(f"Date range: {fixtures_df['starting_at'].min()} to {fixtures_df['starting_at'].max()}")
    
    logger.info("\n⚠️ NEXT STEPS:")
    logger.info("1. Generate features for these fixtures using historical data")
    logger.info("2. Run predictions with the V3 model")
    logger.info("3. Apply optimized thresholds")
    logger.info("4. Generate betting recommendations")
    
    logger.info("\n💡 To generate full predictions, run:")
    logger.info("   python3 scripts/generate_features_and_predict.py")


if __name__ == '__main__':
    main()
