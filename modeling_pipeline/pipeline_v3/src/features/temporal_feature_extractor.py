"""
Temporal Feature Extractor - Phase 6

Extracts temporal and contextual features:
- Fixture congestion (fatigue)
- Season phase
- Competitive pressure

Expected impact: -0.001 to -0.003 log loss
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """Extract temporal and contextual features."""
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize extractor and load fixtures data."""
        self.data_dir = Path(data_dir)
        
        logger.info("Loading fixtures data for temporal features...")
        self.fixtures = pd.read_csv(self.data_dir / 'fixtures.csv')
        
        # Convert dates
        self.fixtures['starting_at'] = pd.to_datetime(self.fixtures['starting_at'])
        
        logger.info(f"Loaded {len(self.fixtures)} fixtures")
    
    def extract_temporal_features(
        self,
        team_id: int,
        fixture_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """
        Extract temporal features for a team.
        
        Args:
            team_id: Team ID
            fixture_id: Current fixture ID
            as_of_date: Match date
        
        Returns:
            Dictionary of temporal features
        """
        as_of_date_dt = pd.to_datetime(as_of_date)
        
        # Get team's recent matches
        team_matches = self.fixtures[
            ((self.fixtures['home_team_id'] == team_id) | 
             (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['starting_at'] < as_of_date_dt)
        ].sort_values('starting_at', ascending=False)
        
        if len(team_matches) == 0:
            return self._get_default_features()
        
        # Fixture congestion
        last_match_date = team_matches.iloc[0]['starting_at']
        days_since_last = (as_of_date_dt - last_match_date).days
        
        # Matches in last 7 days
        week_ago = as_of_date_dt - timedelta(days=7)
        matches_in_7_days = len(team_matches[team_matches['starting_at'] >= week_ago])
        
        # Congestion score (0 = rested, 1 = congested)
        congestion_score = min(matches_in_7_days / 3.0, 1.0)
        
        # Season phase (0 = start, 1 = end)
        # Assuming 38-game season over ~9 months
        season_start = team_matches.iloc[-1]['starting_at']  # First match
        season_progress = (as_of_date_dt - season_start).days / 270  # ~9 months
        season_phase = min(season_progress, 1.0)
        
        return {
            'days_since_last_match': min(days_since_last, 14),  # Cap at 2 weeks
            'matches_in_7_days': matches_in_7_days,
            'fixture_congestion_score': congestion_score,
            'season_phase': season_phase,
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default features when no data."""
        return {
            'days_since_last_match': 7.0,
            'matches_in_7_days': 1.0,
            'fixture_congestion_score': 0.33,
            'season_phase': 0.5,
        }
