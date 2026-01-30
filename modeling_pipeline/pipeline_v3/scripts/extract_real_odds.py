#!/usr/bin/env python3
"""
Extract Real Odds from Historical JSON Files
=============================================

Extracts 1X2 (Home/Draw/Away) odds from historical fixture JSON files
and populates the fixtures.csv with real bookmaker odds.
"""

import json
import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_1x2_odds(fixture):
    """
    Extract 1X2 odds from a fixture.
    
    Returns: (home_odds, draw_odds, away_odds) or (None, None, None)
    """
    odds_list = fixture.get('odds', [])
    
    if not odds_list:
        return None, None, None
    
    # Filter for 1X2 market (market_id=1) and a reliable bookmaker
    # Prefer Bet365 (bookmaker_id=2) or any major bookmaker
    preferred_bookmakers = [2, 3, 5, 10]  # Bet365, Pinnacle, William Hill, etc.
    
    home_odds = None
    draw_odds = None
    away_odds = None
    
    # First try preferred bookmakers
    for bookmaker_id in preferred_bookmakers:
        for odd in odds_list:
            if odd.get('market_id') == 1 and odd.get('bookmaker_id') == bookmaker_id:
                label = odd.get('label') or odd.get('name')
                value = odd.get('value')
                
                if value and label:
                    try:
                        odds_value = float(value)
                        if label == '1':  # Home
                            home_odds = odds_value
                        elif label == 'X':  # Draw
                            draw_odds = odds_value
                        elif label == '2':  # Away
                            away_odds = odds_value
                    except (ValueError, TypeError):
                        continue
        
        # If we found all three odds from this bookmaker, use them
        if home_odds and draw_odds and away_odds:
            return home_odds, draw_odds, away_odds
    
    # If preferred bookmakers didn't work, try any bookmaker
    for odd in odds_list:
        if odd.get('market_id') == 1:
            label = odd.get('label') or odd.get('name')
            value = odd.get('value')
            
            if value and label:
                try:
                    odds_value = float(value)
                    if label == '1' and not home_odds:
                        home_odds = odds_value
                    elif label == 'X' and not draw_odds:
                        draw_odds = odds_value
                    elif label == '2' and not away_odds:
                        away_odds = odds_value
                except (ValueError, TypeError):
                    continue
    
    return home_odds, draw_odds, away_odds


def main():
    """Extract odds from all historical JSON files."""
    logger.info("="*70)
    logger.info("EXTRACTING REAL ODDS FROM HISTORICAL JSON FILES")
    logger.info("="*70)
    
    # Path to historical fixtures
    fixtures_dir = Path('pipeline_v3/data/historical/fixtures')
    
    if not fixtures_dir.exists():
        logger.error(f"Directory not found: {fixtures_dir}")
        return
    
    # Collect all odds
    odds_data = {}
    total_fixtures = 0
    fixtures_with_odds = 0
    
    # Process all JSON files
    json_files = list(fixtures_dir.glob('*.json'))
    logger.info(f"\nProcessing {len(json_files)} JSON files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                fixtures = json.load(f)
            
            if not isinstance(fixtures, list):
                continue
            
            for fixture in fixtures:
                fixture_id = fixture.get('id')
                if not fixture_id:
                    continue
                
                total_fixtures += 1
                
                # Extract odds
                home_odds, draw_odds, away_odds = extract_1x2_odds(fixture)
                
                if home_odds and draw_odds and away_odds:
                    odds_data[fixture_id] = {
                        'odds_home': home_odds,
                        'odds_draw': draw_odds,
                        'odds_away': away_odds
                    }
                    fixtures_with_odds += 1
        
        except Exception as e:
            logger.warning(f"Error processing {json_file.name}: {e}")
            continue
    
    logger.info(f"\nProcessed {total_fixtures} fixtures")
    logger.info(f"Found odds for {fixtures_with_odds} fixtures ({fixtures_with_odds/total_fixtures*100:.1f}%)")
    
    # Load fixtures.csv
    fixtures_csv_path = Path('pipeline_v3/data/csv/fixtures.csv')
    
    if not fixtures_csv_path.exists():
        logger.error(f"fixtures.csv not found at {fixtures_csv_path}")
        return
    
    logger.info(f"\nLoading {fixtures_csv_path}...")
    fixtures_df = pd.read_csv(fixtures_csv_path)
    
    logger.info(f"Loaded {len(fixtures_df)} rows")
    
    # Update odds columns
    logger.info("\nUpdating odds columns...")
    
    fixtures_df['odds_home'] = fixtures_df['fixture_id'].map(lambda x: odds_data.get(x, {}).get('odds_home'))
    fixtures_df['odds_draw'] = fixtures_df['fixture_id'].map(lambda x: odds_data.get(x, {}).get('odds_draw'))
    fixtures_df['odds_away'] = fixtures_df['fixture_id'].map(lambda x: odds_data.get(x, {}).get('odds_away'))
    
    # Count how many were updated
    updated_count = fixtures_df['odds_home'].notna().sum()
    logger.info(f"Updated odds for {updated_count}/{len(fixtures_df)} rows ({updated_count/len(fixtures_df)*100:.1f}%)")
    
    # Save updated CSV
    logger.info(f"\nSaving updated fixtures.csv...")
    fixtures_df.to_csv(fixtures_csv_path, index=False)
    
    logger.info("✅ Done!")
    
    # Show sample
    logger.info("\nSample of updated data:")
    sample = fixtures_df[fixtures_df['odds_home'].notna()].head(5)
    logger.info(f"\n{sample[['fixture_id', 'home_team_name', 'away_team_name', 'odds_home', 'odds_draw', 'odds_away']]}")
    
    logger.info("\n" + "="*70)
    logger.info(f"SUMMARY: {updated_count} fixtures now have real odds!")
    logger.info("="*70)


if __name__ == '__main__':
    os.chdir('/Users/ankurgupta/code/predictorV2/modeling_pipeline')
    main()
