"""
Injury Feature Extractor - Phase 4 (CRITICAL)

Extracts injury and availability features:
- Basic injury counts
- Key player identification
- Position-specific impact

Expected impact: -0.005 to -0.010 log loss
CRITICAL for handling missing key players
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class InjuryFeatureExtractor:
    """Extract injury and availability features."""
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize extractor and identify key players."""
        self.data_dir = Path(data_dir)
        
        logger.info("Loading sidelined and lineups data...")
        try:
            self.sidelined = pd.read_csv(self.data_dir / 'sidelined.csv')
            logger.info(f"Loaded {len(self.sidelined)} sidelined records")
        except FileNotFoundError:
            logger.warning("sidelined.csv not found - will use defaults")
            self.sidelined = pd.DataFrame()
        
        # Load lineups to identify key players
        try:
            lineups = pd.read_csv(self.data_dir / 'lineups.csv')
            self.key_players = self._identify_key_players(lineups)
            logger.info(f"Identified {len(self.key_players)} key players")
        except FileNotFoundError:
            logger.warning("lineups.csv not found - cannot identify key players")
            self.key_players = {}
    
    def _identify_key_players(self, lineups: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
        """
        Identify key players from historical performance.
        
        Args:
            lineups: Lineups dataframe
        
        Returns:
            Dict mapping team_id to list of (player_id, position_id) tuples
        """
        # Filter to starters only
        starters = lineups[lineups['is_starter'] == True].copy()
        
        # Calculate player stats
        player_stats = starters.groupby(['team_id', 'player_id', 'position_id']).agg({
            'detail_118': 'mean',      # avg rating
            'detail_119': 'mean',      # avg minutes
            'fixture_id': 'count'      # games played
        }).reset_index()
        
        player_stats.columns = ['team_id', 'player_id', 'position_id', 'avg_rating', 'avg_minutes', 'games_played']
        
        # Key players criteria:
        # - avg rating > 7.5 (top performers)
        # - games played >= 10 (regulars)
        # - avg minutes > 70 (starters)
        key_players_df = player_stats[
            (player_stats['avg_rating'] > 7.5) &
            (player_stats['games_played'] >= 10) &
            (player_stats['avg_minutes'] > 70)
        ]
        
        # Group by team
        key_players = {}
        for team_id in key_players_df['team_id'].unique():
            team_keys = key_players_df[key_players_df['team_id'] == team_id]
            key_players[team_id] = list(zip(
                team_keys['player_id'].tolist(),
                team_keys['position_id'].tolist()
            ))
        
        return key_players
    
    def extract_injury_features(
        self,
        team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """
        Extract injury features for a team at a specific date.
        
        Args:
            team_id: Team ID
            as_of_date: Date to check injuries (YYYY-MM-DD)
        
        Returns:
            Dictionary of injury features
        """
        if len(self.sidelined) == 0:
            return self._get_default_features()
        
        # Convert date
        as_of_date_dt = pd.to_datetime(as_of_date)
        
        # Get current injuries (active on this date)
        current_injuries = self.sidelined[
            (self.sidelined['team_id'] == team_id) &
            (pd.to_datetime(self.sidelined['start_date']) <= as_of_date_dt) &
            (
                (pd.to_datetime(self.sidelined['end_date']) >= as_of_date_dt) |
                (self.sidelined['end_date'].isna())
            )
        ]
        
        injured_player_ids = current_injuries['player_id'].tolist()
        
        # Basic injury metrics
        injuries_count = len(current_injuries)
        injuries_long_term = len(current_injuries[
            (pd.to_datetime(current_injuries['end_date']) - as_of_date_dt).dt.days > 28
        ])
        
        # Key player impact
        team_key_players = self.key_players.get(team_id, [])
        key_player_ids = [p[0] for p in team_key_players]
        
        key_players_missing = len(set(injured_player_ids) & set(key_player_ids))
        
        # Position-specific impact
        attack_impact = self._calculate_position_impact(
            injured_player_ids, team_key_players, position_id=27  # Forward
        )
        defense_impact = self._calculate_position_impact(
            injured_player_ids, team_key_players, position_id=25  # Defender
        )
        midfield_impact = self._calculate_position_impact(
            injured_player_ids, team_key_players, position_id=26  # Midfielder
        )
        
        # Lineup strength ratio
        total_key_players = len(team_key_players)
        lineup_strength_ratio = 1.0 - (key_players_missing / max(total_key_players, 1))
        
        return {
            'injuries_count': injuries_count,
            'injuries_long_term': injuries_long_term,
            'injury_crisis': 1 if injuries_count >= 3 else 0,
            'key_players_missing': key_players_missing,
            'lineup_strength_ratio': lineup_strength_ratio,
            'attack_strength_impact': attack_impact,
            'defense_strength_impact': defense_impact,
            'midfield_strength_impact': midfield_impact,
        }
    
    def _calculate_position_impact(
        self,
        injured_player_ids: List[int],
        team_key_players: List[Tuple[int, int]],
        position_id: int
    ) -> float:
        """Calculate impact of injuries on specific position."""
        # Get key players in this position
        position_key_players = [p[0] for p in team_key_players if p[1] == position_id]
        
        if len(position_key_players) == 0:
            return 0.0
        
        # Count missing key players in this position
        missing = len(set(injured_player_ids) & set(position_key_players))
        
        # Return impact ratio
        return missing / len(position_key_players)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default features when no data."""
        return {
            'injuries_count': 0.0,
            'injuries_long_term': 0.0,
            'injury_crisis': 0.0,
            'key_players_missing': 0.0,
            'lineup_strength_ratio': 1.0,
            'attack_strength_impact': 0.0,
            'defense_strength_impact': 0.0,
            'midfield_strength_impact': 0.0,
        }
