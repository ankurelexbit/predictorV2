"""
Player Statistics Extractor

Extracts player-level statistics from lineups.details and aggregates to team level.
This is the HIGHEST PRIORITY feature set (Priority 0) with expected impact of -0.020 to -0.030 log loss.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

from .type_id_mappings import PLAYER_STAT_TYPE_IDS_P0, LINEUP_TYPE_IDS

logger = logging.getLogger(__name__)


class PlayerStatisticsExtractor:
    """
    Extract player-level statistics from lineups.details.
    
    Focuses on Priority 0 features:
    - Player ratings
    - Touches
    - Passes
    - Duels
    - Shots
    - Key passes
    """
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize extractor and load lineup data."""
        self.data_dir = Path(data_dir)
        
        logger.info("Loading lineups data...")
        self.lineups = pd.read_csv(self.data_dir / 'lineups.csv')
        
        logger.info(f"Loaded {len(self.lineups)} lineup records")
        logger.info(f"Columns: {list(self.lineups.columns[:20])}...")  # Show first 20 columns
    
    def extract_team_player_stats(
        self,
        team_id: int,
        fixture_ids: List[int],
        window: int = 5
    ) -> Dict[str, float]:
        """
        Extract aggregated player stats for a team across multiple matches.
        
        Args:
            team_id: Team ID
            fixture_ids: List of fixture IDs (already filtered to last N matches)
            window: Window size (for naming)
        
        Returns:
            Dictionary of aggregated player statistics
        """
        all_match_stats = []
        
        for fixture_id in fixture_ids:
            match_stats = self._extract_match_player_stats(team_id, fixture_id)
            if match_stats:
                all_match_stats.append(match_stats)
        
        if not all_match_stats:
            logger.warning(f"No player stats found for team {team_id}")
            return self._get_default_stats()
        
        # Aggregate across all matches
        aggregated = self._aggregate_across_matches(all_match_stats)
        
        return aggregated
    
    def _extract_match_player_stats(
        self,
        team_id: int,
        fixture_id: int
    ) -> Optional[Dict[str, List[float]]]:
        """
        Extract player stats for a single match.
        
        Args:
            team_id: Team ID
            fixture_id: Fixture ID
        
        Returns:
            Dictionary mapping stat names to lists of values (one per starter)
        """
        # Get lineups for this team in this match
        team_lineups = self.lineups[
            (self.lineups['fixture_id'] == fixture_id) &
            (self.lineups['team_id'] == team_id)
        ]
        
        if len(team_lineups) == 0:
            return None
        
        # Filter to starters only (is_starter = True)
        starters = team_lineups[team_lineups['is_starter'] == True]
        
        if len(starters) == 0:
            logger.warning(f"No starters found for team {team_id} in fixture {fixture_id}")
            return None
        
        # Extract stats from detail_* columns
        match_stats = {}
        
        # Map detail columns to stat names
        detail_mapping = {
            'detail_118': 'rating',
            'detail_80': 'touches',
            'detail_119': 'minutes',
            'detail_1584': 'passes_total',
            'detail_120': 'pass_accuracy',
            'detail_122': 'passes_accurate',
            'detail_101': 'duels_won',
            'detail_27273': 'duels_total',
            'detail_56': 'shots',
            'detail_1491': 'shots_on_target',
            'detail_114': 'key_passes',
        }
        
        for detail_col, stat_name in detail_mapping.items():
            if detail_col in starters.columns:
                # Get non-null values
                values = starters[detail_col].dropna().tolist()
                
                if len(values) > 0:
                    match_stats[stat_name] = [float(v) for v in values]
        
        return match_stats if match_stats else None
    
    def _aggregate_across_matches(
        self,
        all_match_stats: List[Dict[str, List[float]]]
    ) -> Dict[str, float]:
        """
        Aggregate player stats across multiple matches.
        
        For each stat, we calculate:
        - avg: Average across all starters in all matches
        - max: Maximum value seen
        - min: Minimum value seen
        - std: Standard deviation (consistency)
        
        Args:
            all_match_stats: List of match stat dictionaries
        
        Returns:
            Aggregated statistics
        """
        aggregated = {}
        
        # Collect all values for each stat across all matches
        stat_values = {}
        for match_stats in all_match_stats:
            for stat_name, values in match_stats.items():
                if stat_name not in stat_values:
                    stat_values[stat_name] = []
                stat_values[stat_name].extend(values)
        
        # Calculate aggregations
        for stat_name, values in stat_values.items():
            if len(values) > 0:
                aggregated[f'{stat_name}_avg'] = np.mean(values)
                aggregated[f'{stat_name}_max'] = np.max(values)
                aggregated[f'{stat_name}_min'] = np.min(values)
                
                if len(values) > 1:
                    aggregated[f'{stat_name}_std'] = np.std(values)
                else:
                    aggregated[f'{stat_name}_std'] = 0.0
        
        # Add derived metrics
        if 'rating_avg' in aggregated:
            # Team quality score (weighted average of top players)
            ratings = stat_values.get('rating', [])
            if len(ratings) >= 3:
                top_3 = sorted(ratings, reverse=True)[:3]
                aggregated['top_3_rating_avg'] = np.mean(top_3)
        
        if 'passes_total_avg' in aggregated and 'passes_accurate_avg' in aggregated:
            # Team pass completion
            total_passes = stat_values.get('passes_total', [])
            accurate_passes = stat_values.get('passes_accurate', [])
            
            if sum(total_passes) > 0:
                aggregated['team_pass_completion'] = sum(accurate_passes) / sum(total_passes) * 100
        
        if 'duels_won_avg' in aggregated and 'duels_total_avg' in aggregated:
            # Team duel success rate
            duels_won = stat_values.get('duels_won', [])
            duels_total = stat_values.get('duels_total', [])
            
            if sum(duels_total) > 0:
                aggregated['team_duel_success'] = sum(duels_won) / sum(duels_total) * 100
        
        return aggregated
    
    def _get_default_stats(self) -> Dict[str, float]:
        """Return default stats when no data available."""
        default_stats = {}
        
        for stat_name in PLAYER_STAT_TYPE_IDS_P0.values():
            default_stats[f'{stat_name}_avg'] = 0.0
            default_stats[f'{stat_name}_max'] = 0.0
            default_stats[f'{stat_name}_min'] = 0.0
            default_stats[f'{stat_name}_std'] = 0.0
        
        default_stats['top_3_rating_avg'] = 0.0
        default_stats['team_pass_completion'] = 0.0
        default_stats['team_duel_success'] = 0.0
        
        return default_stats
    
    def get_team_matches_before_date(
        self,
        team_id: int,
        before_date: str,
        fixtures_df: pd.DataFrame,
        limit: int = 5
    ) -> List[int]:
        """
        Get fixture IDs for team's last N matches before a date.
        
        Args:
            team_id: Team ID
            before_date: Date cutoff (YYYY-MM-DD)
            fixtures_df: Fixtures dataframe
            limit: Number of matches to return
        
        Returns:
            List of fixture IDs
        """
        # Convert date
        before_date = pd.to_datetime(before_date)
        
        # Get team's matches
        team_matches = fixtures_df[
            ((fixtures_df['home_team_id'] == team_id) | 
             (fixtures_df['away_team_id'] == team_id)) &
            (pd.to_datetime(fixtures_df['starting_at']) < before_date)
        ].sort_values('starting_at', ascending=False)
        
        # Get last N matches
        last_n = team_matches.head(limit)
        
        return last_n['fixture_id'].tolist()
