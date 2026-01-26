"""
Odds Fetcher Module

Fetches real-time betting odds from SportMonks API for live predictions.
This module provides a simple interface to get market odds for fixtures.

Usage:
    from odds_fetcher import OddsFetcher
    
    fetcher = OddsFetcher()
    odds = fetcher.get_odds(fixture_id=12345)
    
    print(odds['odds_home'])  # e.g., 1.75
    print(odds['odds_draw'])  # e.g., 3.50
    print(odds['odds_away'])  # e.g., 4.20
"""

import os
import requests
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# SportMonks API configuration (same as predict_live.py)
SPORTMONKS_API_KEY = "DQQStChRaPnjIryuZH2SxqJI5ufoA57wWsmFIuPCH2rvlBtm0G7Ch3mJoyE4"

class OddsFetcher:
    """Fetch real-time betting odds from SportMonks API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize odds fetcher.
        
        Args:
            api_key: SportMonks API key (defaults to SPORTMONKS_API_KEY constant)
        """
        self.api_key = api_key or SPORTMONKS_API_KEY or os.getenv('SPORTMONKS_API_KEY', '')
        self.base_url = "https://api.sportmonks.com/v3/football"
        
        if not self.api_key:
            logger.warning("SPORTMONKS_API_KEY not set - odds fetching will use fallback")
    
    def get_odds(self, fixture_id: int) -> Dict[str, float]:
        """
        Fetch betting odds for a fixture.
        
        Args:
            fixture_id: SportMonks fixture ID
            
        Returns:
            Dictionary with odds_home, odds_draw, odds_away, and derived features
        """
        if not self.api_key:
            logger.warning(f"No API key - using neutral odds for fixture {fixture_id}")
            return self._get_neutral_odds()
        
        try:
            url = f"{self.base_url}/fixtures/{fixture_id}"
            params = {
                'api_token': self.api_key,
                'include': 'odds'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                logger.warning(f"No data in response for fixture {fixture_id}")
                return self._get_neutral_odds()
            
            fixture_data = data['data']
            odds_data = fixture_data.get('odds', [])
            
            # SportMonks v3 API structure: each odds entry is individual
            # market_id=1 is "Match Winner" (Home/Draw/Away)
            # Collect odds from ALL bookmakers
            
            # For features: use FIRST bookmaker (consistency with training)
            feature_odds_home = None
            feature_odds_draw = None
            feature_odds_away = None
            
            # For PnL: track BEST odds across all bookmakers
            best_odds_home = 0
            best_odds_draw = 0
            best_odds_away = 0
            
            for odds in odds_data:
                if odds.get('market_id') == 1:  # Match Winner market
                    label = odds.get('label', '')
                    value = odds.get('value')
                    
                    if value:
                        odds_value = float(value)
                        
                        if label == 'Home':
                            if feature_odds_home is None:
                                feature_odds_home = odds_value
                            best_odds_home = max(best_odds_home, odds_value)
                        elif label == 'Draw':
                            if feature_odds_draw is None:
                                feature_odds_draw = odds_value
                            best_odds_draw = max(best_odds_draw, odds_value)
                        elif label == 'Away':
                            if feature_odds_away is None:
                                feature_odds_away = odds_value
                            best_odds_away = max(best_odds_away, odds_value)
            
            if feature_odds_home and feature_odds_draw and feature_odds_away:
                logger.info(f"✅ Fetched odds for fixture {fixture_id}:")
                logger.info(f"   Feature odds: H={feature_odds_home:.2f}, D={feature_odds_draw:.2f}, A={feature_odds_away:.2f}")
                logger.info(f"   Best odds:    H={best_odds_home:.2f}, D={best_odds_draw:.2f}, A={best_odds_away:.2f}")
                
                # Return both feature odds and best odds
                result = self._calculate_odds_features(feature_odds_home, feature_odds_draw, feature_odds_away)
                result['best_odds_home'] = best_odds_home
                result['best_odds_draw'] = best_odds_draw
                result['best_odds_away'] = best_odds_away
                return result
            
            logger.warning(f"No complete Match Winner odds found for fixture {fixture_id}")
            return self._get_neutral_odds()
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"API error fetching odds for fixture {fixture_id}: {e}")
            return self._get_neutral_odds()
        except Exception as e:
            logger.error(f"Unexpected error fetching odds for fixture {fixture_id}: {e}")
            return self._get_neutral_odds()
    
    def _calculate_odds_features(self, odds_home: float, odds_draw: float, odds_away: float) -> Dict[str, float]:
        """
        Calculate odds features from raw odds.
        
        Args:
            odds_home: Home win odds
            odds_draw: Draw odds
            odds_away: Away win odds
            
        Returns:
            Dictionary with odds and derived features
        """
        return {
            'odds_home': odds_home,
            'odds_draw': odds_draw,
            'odds_away': odds_away,
            'odds_total': odds_home + odds_draw + odds_away,
            'odds_home_draw_ratio': odds_home / odds_draw if odds_draw > 0 else 1.0,
            'odds_home_away_ratio': odds_home / odds_away if odds_away > 0 else 1.0,
            'market_home_away_ratio': odds_home / odds_away if odds_away > 0 else 1.0,
        }
    
    def _get_neutral_odds(self) -> Dict[str, float]:
        """
        Return neutral odds as fallback.
        
        Returns:
            Dictionary with neutral odds (equal probabilities)
        """
        logger.debug("Using neutral odds (fallback)")
        result = self._calculate_odds_features(2.5, 3.3, 2.5)
        # Add best odds (same as feature odds for neutral)
        result['best_odds_home'] = 2.5
        result['best_odds_draw'] = 3.3
        result['best_odds_away'] = 2.5
        return result


# Convenience function for quick usage
def fetch_odds(fixture_id: int, api_key: Optional[str] = None) -> Dict[str, float]:
    """
    Convenience function to fetch odds for a fixture.
    
    Args:
        fixture_id: SportMonks fixture ID
        api_key: Optional API key (defaults to environment variable)
        
    Returns:
        Dictionary with odds and features
        
    Example:
        >>> odds = fetch_odds(12345)
        >>> print(f"Home odds: {odds['odds_home']}")
    """
    fetcher = OddsFetcher(api_key=api_key)
    return fetcher.get_odds(fixture_id)


if __name__ == "__main__":
    # Test the odds fetcher
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        fixture_id = int(sys.argv[1])
        print(f"\nFetching odds for fixture {fixture_id}...")
        
        odds = fetch_odds(fixture_id)
        
        print("\nOdds:")
        print(f"  Home: {odds['odds_home']:.2f}")
        print(f"  Draw: {odds['odds_draw']:.2f}")
        print(f"  Away: {odds['odds_away']:.2f}")
        print(f"\nDerived features:")
        print(f"  Total: {odds['odds_total']:.2f}")
        print(f"  Home/Draw ratio: {odds['odds_home_draw_ratio']:.2f}")
        print(f"  Home/Away ratio: {odds['odds_home_away_ratio']:.2f}")
    else:
        print("Usage: python odds_fetcher.py <fixture_id>")
        print("Example: python odds_fetcher.py 12345")
