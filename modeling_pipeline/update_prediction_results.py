#!/usr/bin/env python3
"""
Update Predictions with Actual Results
Fetches match results from SportMonks API and updates the predictions table
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import requests
import logging
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from db_predictions import PredictionsDB
from utils import setup_logger

logger = setup_logger("update_results")

# SportMonks API configuration
SPORTMONKS_API_KEY = os.getenv("SPORTMONKS_API_KEY", "LmEYRdsf8CSmiblNVTn6JfV0y0s8tc4aQsEkJ6JuoBhOWa3Hd1FEanGcrijo")
BASE_URL = "https://api.sportmonks.com/v3/football"

def get_fixture_result(fixture_id: int) -> Optional[Dict]:
    """
    Fetch match result from SportMonks API.
    
    Args:
        fixture_id: SportMonks fixture ID
        
    Returns:
        Dictionary with match result or None if not found
    """
    try:
        url = f"{BASE_URL}/fixtures/{fixture_id}"
        params = {
            'api_token': SPORTMONKS_API_KEY,
            'include': 'scores;participants'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data:
            logger.warning(f"No data for fixture {fixture_id}")
            return None
        
        fixture = data['data']
        
        # Check if match is finished
        # State can be in state object or state_id
        state = None
        if 'state' in fixture and isinstance(fixture['state'], dict):
            state = fixture['state'].get('state')
        
        # Also check state_id (5 = Full Time)
        state_id = fixture.get('state_id')
        
        if state != 'FT' and state_id != 5:
            logger.debug(f"Fixture {fixture_id} not finished yet (state={state}, state_id={state_id})")
            return None
        
        # Extract scores
        scores = fixture.get('scores', [])
        home_score = None
        away_score = None
        
        # SportMonks v3 returns separate score objects for home and away
        for score in scores:
            if score.get('description') == 'CURRENT':
                score_data = score.get('score', {})
                participant = score_data.get('participant')
                goals = score_data.get('goals')
                
                if participant == 'home':
                    home_score = goals
                elif participant == 'away':
                    away_score = goals
        
        if home_score is None or away_score is None:
            logger.warning(f"Could not extract scores for fixture {fixture_id}")
            return None
        
        return {
            'fixture_id': fixture_id,
            'home_goals': int(home_score),
            'away_goals': int(away_score),
            'state': fixture.get('state', {}).get('state')
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API error for fixture {fixture_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing fixture {fixture_id}: {e}")
        return None

def update_predictions_with_results(days_back: int = 7, limit: int = 100):
    """
    Update predictions with actual results.
    
    Args:
        days_back: How many days back to check for results
        limit: Maximum number of predictions to update
    """
    logger.info("="*80)
    logger.info("UPDATING PREDICTIONS WITH ACTUAL RESULTS")
    logger.info("="*80)
    logger.info(f"Checking predictions from last {days_back} days")
    logger.info("")
    
    try:
        with PredictionsDB() as db:
            # Get predictions without results
            cursor = db.conn.cursor()
            
            sql = """
            SELECT id, fixture_id, home_team, away_team, match_date, recommended_bet, best_odds
            FROM predictions
            WHERE actual_result IS NULL
            AND match_date >= NOW() - INTERVAL '%s days'
            AND match_date < NOW()
            ORDER BY match_date DESC
            LIMIT %s;
            """
            
            cursor.execute(sql, (days_back, limit))
            predictions = cursor.fetchall()
            
            if not predictions:
                logger.info("✅ No predictions need updating")
                return
            
            logger.info(f"Found {len(predictions)} predictions to update")
            logger.info("")
            
            updated_count = 0
            failed_count = 0
            
            for pred in predictions:
                pred_id, fixture_id, home_team, away_team, match_date, recommended_bet, best_odds = pred
                
                logger.info(f"[{updated_count + failed_count + 1}/{len(predictions)}] {home_team} vs {away_team}")
                
                # Fetch result from API
                result = get_fixture_result(fixture_id)
                
                if not result:
                    logger.warning(f"  ⚠️  Could not fetch result for fixture {fixture_id}")
                    failed_count += 1
                    continue
                
                home_goals = result['home_goals']
                away_goals = result['away_goals']
                
                # Update database
                success = db.update_actual_result(fixture_id, home_goals, away_goals)
                
                if success:
                    # Determine actual result
                    if home_goals > away_goals:
                        actual = 'HOME'
                    elif home_goals < away_goals:
                        actual = 'AWAY'
                    else:
                        actual = 'DRAW'
                    
                    is_correct = (recommended_bet.upper() == actual)
                    
                    logger.info(f"  ✅ Updated: {home_goals}-{away_goals} (Actual: {actual}, Bet: {recommended_bet}, {'✓' if is_correct else '✗'})")
                    updated_count += 1
                else:
                    logger.warning(f"  ⚠️  Database update failed")
                    failed_count += 1
            
            logger.info("")
            logger.info("="*80)
            logger.info("SUMMARY")
            logger.info("="*80)
            logger.info(f"Total predictions checked: {len(predictions)}")
            logger.info(f"Successfully updated: {updated_count}")
            logger.info(f"Failed: {failed_count}")
            
            if updated_count > 0:
                # Get updated performance stats
                stats = db.get_performance_stats(days=days_back)
                
                if stats and stats.get('total_bets'):
                    logger.info("")
                    logger.info("PERFORMANCE (Last {} days):".format(days_back))
                    logger.info(f"  Total Bets: {stats.get('total_bets', 0)}")
                    logger.info(f"  Correct: {stats.get('correct_bets', 0)}")
                    logger.info(f"  Win Rate: {stats.get('win_rate_pct', 0):.1f}%")
                    logger.info(f"  Total Profit: ${stats.get('total_profit', 0):.2f}")
                    logger.info(f"  ROI: {stats.get('roi_pct', 0):.1f}%")
            
            logger.info("")
            logger.info("="*80)
            
    except Exception as e:
        logger.error(f"❌ Error updating predictions: {e}")
        raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Update predictions with actual results")
    parser.add_argument('--days', type=int, default=7, help='Days back to check (default: 7)')
    parser.add_argument('--limit', type=int, default=100, help='Max predictions to update (default: 100)')
    
    args = parser.parse_args()
    
    update_predictions_with_results(days_back=args.days, limit=args.limit)

if __name__ == '__main__':
    main()
