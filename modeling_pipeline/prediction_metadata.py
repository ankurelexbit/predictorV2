"""
Enhanced Metadata Collection for Live Predictions
==================================================

Helper functions to collect comprehensive metadata including:
- Lineups (11 players per team)
- Injuries (counts and player details)
- Multiple bookmaker odds
- Timing data
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_lineup_metadata(lineups_data: Optional[Dict], player_manager) -> Dict:
    """
    Extract detailed lineup information for storage.
    
    Returns:
        Dict with home_lineup, away_lineup, lineup_available, coverage stats
    """
    if not lineups_data or not player_manager.is_loaded:
        return {
            'home_lineup': None,
            'away_lineup': None,
            'lineup_available': False,
            'lineup_coverage_home': 0.0,
            'lineup_coverage_away': 0.0,
            'used_lineup_data': False
        }
    
    home_players = lineups_data.get('home', [])
    away_players = lineups_data.get('away', [])
    
    # Build detailed lineup data
    home_lineup_details = []
    away_lineup_details = []
    
    home_found = 0
    away_found = 0
    
    for player_id in home_players:
        player_stats = player_manager.get_player_stats(player_id)
        if player_stats:
            home_found += 1
            home_lineup_details.append({
                'id': player_id,
                'name': player_stats.get('player_name', 'Unknown'),
                'rating': round(player_stats.get('rating', 0), 2),
                'minutes': player_stats.get('minutes', 0),
                'appearances': player_stats.get('matches_played', 0)
            })
        else:
            home_lineup_details.append({
                'id': player_id,
                'name': 'Unknown',
                'rating': 0,
                'minutes': 0,
                'appearances': 0
            })
    
    for player_id in away_players:
        player_stats = player_manager.get_player_stats(player_id)
        if player_stats:
            away_found += 1
            away_lineup_details.append({
                'id': player_id,
                'name': player_stats.get('player_name', 'Unknown'),
                'rating': round(player_stats.get('rating', 0), 2),
                'minutes': player_stats.get('minutes', 0),
                'appearances': player_stats.get('matches_played', 0)
            })
        else:
            away_lineup_details.append({
                'id': player_id,
                'name': 'Unknown',
                'rating': 0,
                'minutes': 0,
                'appearances': 0
            })
    
    home_coverage = (home_found / len(home_players) * 100) if home_players else 0
    away_coverage = (away_found / len(away_players) * 100) if away_players else 0
    
    return {
        'home_lineup': {
            'players': home_lineup_details,
            'count': len(home_lineup_details),
            'avg_rating': round(sum(p['rating'] for p in home_lineup_details) / len(home_lineup_details), 2) if home_lineup_details else 0
        },
        'away_lineup': {
            'players': away_lineup_details,
            'count': len(away_lineup_details),
            'avg_rating': round(sum(p['rating'] for p in away_lineup_details) / len(away_lineup_details), 2) if away_lineup_details else 0
        },
        'lineup_available': len(home_players) > 0 and len(away_players) > 0,
        'lineup_coverage_home': round(home_coverage, 1),
        'lineup_coverage_away': round(away_coverage, 1),
        'used_lineup_data': home_coverage > 0 and away_coverage > 0
    }


def extract_injury_metadata(home_injuries: List[Dict], away_injuries: List[Dict]) -> Dict:
    """
    Extract detailed injury information for storage.
    
    Returns:
        Dict with injury counts and player details
    """
    return {
        'home_injuries_count': len(home_injuries),
        'away_injuries_count': len(away_injuries),
        'home_injured_players': {
            'players': [
                {
                    'id': inj.get('player_id'),
                    'name': inj.get('player_name', 'Unknown'),
                    'type': inj.get('type', 'injury'),
                    'reason': inj.get('reason', 'Unknown')
                }
                for inj in home_injuries
            ]
        } if home_injuries else None,
        'away_injured_players': {
            'players': [
                {
                    'id': inj.get('player_id'),
                    'name': inj.get('player_name', 'Unknown'),
                    'type': inj.get('type', 'injury'),
                    'reason': inj.get('reason', 'Unknown')
                }
                for inj in away_injuries
            ]
        } if away_injuries else None,
        'used_injury_data': len(home_injuries) > 0 or len(away_injuries) > 0
    }


def extract_bookmaker_odds(odds_data: Optional[Dict]) -> Dict:
    """
    Extract odds from multiple bookmakers.
    
    Returns:
        Dict with all bookmaker odds and best odds for each outcome
    """
    if not odds_data or 'bookmakers' not in odds_data:
        return {
            'bookmaker_odds': None,
            'best_odds_home': None,
            'best_odds_draw': None,
            'best_odds_away': None
        }
    
    bookmakers = []
    best_home = 0
    best_draw = 0
    best_away = 0
    best_home_bookie = None
    best_draw_bookie = None
    best_away_bookie = None
    
    for bookie_data in odds_data.get('bookmakers', []):
        bookie_name = bookie_data.get('name', 'Unknown')
        odds_home = bookie_data.get('odds_home', 0)
        odds_draw = bookie_data.get('odds_draw', 0)
        odds_away = bookie_data.get('odds_away', 0)
        
        bookmakers.append({
            'name': bookie_name,
            'odds_home': odds_home,
            'odds_draw': odds_draw,
            'odds_away': odds_away
        })
        
        # Track best odds
        if odds_home > best_home:
            best_home = odds_home
            best_home_bookie = bookie_name
        if odds_draw > best_draw:
            best_draw = odds_draw
            best_draw_bookie = bookie_name
        if odds_away > best_away:
            best_away = odds_away
            best_away_bookie = bookie_name
    
    return {
        'bookmaker_odds': {
            'bookmakers': bookmakers,
            'best': {
                'home': {'bookmaker': best_home_bookie, 'odds': best_home},
                'draw': {'bookmaker': best_draw_bookie, 'odds': best_draw},
                'away': {'bookmaker': best_away_bookie, 'odds': best_away}
            }
        } if bookmakers else None,
        'best_odds_home': best_home if best_home > 0 else None,
        'best_odds_draw': best_draw if best_draw > 0 else None,
        'best_odds_away': best_away if best_away > 0 else None
    }


def calculate_fair_odds(prob_home: float, prob_draw: float, prob_away: float) -> Dict:
    """
    Calculate fair odds from probabilities (no margin).
    
    Returns:
        Dict with our_odds_home, our_odds_draw, our_odds_away
    """
    return {
        'our_odds_home': round(1 / prob_home, 2) if prob_home > 0 else 999.0,
        'our_odds_draw': round(1 / prob_draw, 2) if prob_draw > 0 else 999.0,
        'our_odds_away': round(1 / prob_away, 2) if prob_away > 0 else 999.0
    }


def calculate_timing_data(kickoff_time: datetime, prediction_time: datetime) -> Dict:
    """
    Calculate timing-related metadata.
    
    Returns:
        Dict with kickoff_time, prediction_time, hours_before_kickoff
    """
    time_diff = kickoff_time - prediction_time
    hours_before = time_diff.total_seconds() / 3600
    
    return {
        'kickoff_time': kickoff_time,
        'prediction_time': prediction_time,
        'hours_before_kickoff': round(hours_before, 2)
    }


def calculate_data_quality_score(lineup_metadata: Dict, injury_metadata: Dict, features_count: int) -> float:
    """
    Calculate overall data quality score (0-1).
    
    Factors:
    - Lineup availability (0.4 weight)
    - Injury data availability (0.2 weight)
    - Feature completeness (0.4 weight)
    """
    lineup_score = 0.0
    if lineup_metadata.get('lineup_available'):
        coverage = (lineup_metadata.get('lineup_coverage_home', 0) + lineup_metadata.get('lineup_coverage_away', 0)) / 200
        lineup_score = coverage
    
    injury_score = 0.0
    if injury_metadata.get('used_injury_data'):
        injury_score = 1.0
    
    feature_score = min(features_count / 373, 1.0)  # 373 is max with all data
    
    total_score = (lineup_score * 0.4) + (injury_score * 0.2) + (feature_score * 0.4)
    
    return round(total_score, 2)
