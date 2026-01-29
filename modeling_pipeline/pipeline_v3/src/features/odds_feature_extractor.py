"""
Odds Feature Extractor - Phase 5

Extracts betting odds features from odds.csv:
- Market probabilities
- Bookmaker confidence
- Market efficiency

Expected impact: -0.002 to -0.005 log loss
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class OddsFeatureExtractor:
    """Extract betting odds features."""
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize extractor and load odds data."""
        self.data_dir = Path(data_dir)
        
        logger.info("Loading odds data...")
        try:
            self.odds = pd.read_csv(self.data_dir / 'odds.csv')
            logger.info(f"Loaded {len(self.odds)} odds records")
        except FileNotFoundError:
            logger.warning("odds.csv not found - will use defaults")
            self.odds = pd.DataFrame()
    
    def extract_odds_features(
        self,
        fixture_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """
        Extract odds features for a fixture.
        
        CRITICAL: Only uses odds from 24+ hours before match
        to prevent data leakage.
        
        Args:
            fixture_id: Fixture ID
            as_of_date: Match date (YYYY-MM-DD)
        
        Returns:
            Dictionary of odds features
        """
        if len(self.odds) == 0:
            return self._get_default_features()
        
        # Temporal cutoff: 24 hours before match
        as_of_date_dt = pd.to_datetime(as_of_date)
        cutoff_date = as_of_date_dt - timedelta(hours=24)
        
        # Get odds for this fixture before cutoff
        fixture_odds = self.odds[
            (self.odds['fixture_id'] == fixture_id) &
            (pd.to_datetime(self.odds['created_at']) <= cutoff_date)
        ]
        
        if len(fixture_odds) == 0:
            return self._get_default_features()
        
        # Get 1x2 odds (match result)
        # Assuming odds are stored with labels: 'Home', 'Draw', 'Away'
        home_odds = fixture_odds[fixture_odds['label'] == 'Home']['value'].values
        draw_odds = fixture_odds[fixture_odds['label'] == 'Draw']['value'].values
        away_odds = fixture_odds[fixture_odds['label'] == 'Away']['value'].values
        
        if len(home_odds) == 0 or len(draw_odds) == 0 or len(away_odds) == 0:
            return self._get_default_features()
        
        # Use most recent odds before cutoff
        home_odd = float(home_odds[-1])
        draw_odd = float(draw_odds[-1])
        away_odd = float(away_odds[-1])
        
        # Convert to probabilities
        probs = self._odds_to_probabilities(home_odd, draw_odd, away_odd)
        
        # Market confidence (difference between favorite and second choice)
        sorted_probs = sorted([probs['home_prob'], probs['draw_prob'], probs['away_prob']], reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1]
        
        # Favorite
        if probs['home_prob'] > max(probs['draw_prob'], probs['away_prob']):
            favorite = 1  # Home
        elif probs['away_prob'] > max(probs['home_prob'], probs['draw_prob']):
            favorite = -1  # Away
        else:
            favorite = 0  # Draw/Toss-up
        
        return {
            'bookmaker_home_win_prob': probs['home_prob'],
            'bookmaker_draw_prob': probs['draw_prob'],
            'bookmaker_away_win_prob': probs['away_prob'],
            'bookmaker_favorite': favorite,
            'bookmaker_confidence': confidence,
            'market_efficiency': probs['margin'],
        }
    
    def _odds_to_probabilities(
        self,
        home_odds: float,
        draw_odds: float,
        away_odds: float
    ) -> Dict[str, float]:
        """
        Convert odds to normalized probabilities.
        
        Args:
            home_odds: Home win odds
            draw_odds: Draw odds
            away_odds: Away win odds
        
        Returns:
            Dictionary with probabilities and margin
        """
        # Convert to implied probabilities
        home_prob = 1 / home_odds
        draw_prob = 1 / draw_odds
        away_prob = 1 / away_odds
        
        # Total (includes bookmaker margin)
        total = home_prob + draw_prob + away_prob
        margin = total - 1.0  # Bookmaker margin
        
        # Normalize (remove margin)
        home_prob_norm = home_prob / total
        draw_prob_norm = draw_prob / total
        away_prob_norm = away_prob / total
        
        return {
            'home_prob': home_prob_norm,
            'draw_prob': draw_prob_norm,
            'away_prob': away_prob_norm,
            'margin': margin
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default features when no odds available."""
        return {
            'bookmaker_home_win_prob': 0.33,
            'bookmaker_draw_prob': 0.33,
            'bookmaker_away_win_prob': 0.33,
            'bookmaker_favorite': 0,
            'bookmaker_confidence': 0.0,
            'market_efficiency': 0.05,
        }
