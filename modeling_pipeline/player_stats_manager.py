"""
Player Statistics Manager
==========================

Provides efficient lookup and aggregation of player statistics from the player database.

Usage:
    from player_stats_manager import PlayerStatsManager

    # Initialize
    manager = PlayerStatsManager()

    # Get player stats
    player_stats = manager.get_player_stats(player_id=12345)

    # Aggregate team stats from lineup
    team_stats = manager.aggregate_team_stats_from_lineup(
        player_ids=[123, 456, 789, ...],
        stat_names=['rating', 'touches', 'duels_won']
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).parent
PLAYER_DB_DIR = BASE_DIR / "data" / "processed" / "player_database"
PLAYER_DB_FILE = PLAYER_DB_DIR / "player_stats_db.json"

# Logging
logger = logging.getLogger("player_stats_manager")


class PlayerStatsManager:
    """Manage player statistics database."""

    def __init__(self, db_file: Path = PLAYER_DB_FILE):
        """
        Initialize player stats manager.

        Args:
            db_file: Path to player database JSON file
        """
        self.db_file = db_file
        self.player_db = {}
        self.is_loaded = False

        # Attempt to load database
        self.load_database()

    def load_database(self) -> bool:
        """
        Load player database from file.

        Returns:
            True if successful, False otherwise
        """
        if not self.db_file.exists():
            logger.warning(f"Player database not found: {self.db_file}")
            logger.warning("Run: python build_player_database.py --full")
            return False

        try:
            with open(self.db_file, 'r') as f:
                self.player_db = json.load(f)

            # Convert string keys back to integers for faster lookup
            self.player_db = {int(k): v for k, v in self.player_db.items()}

            self.is_loaded = True
            logger.info(f"✅ Loaded player database with {len(self.player_db):,} players")
            return True

        except Exception as e:
            logger.error(f"Error loading player database: {e}")
            return False

    def get_player_stats(
        self,
        player_id: int,
        stat_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get statistics for a player.

        Args:
            player_id: Player ID
            stat_name: Optional specific stat name (e.g., 'rating', 'touches')

        Returns:
            Player stats dict or None if not found
        """
        if not self.is_loaded:
            return None

        player_data = self.player_db.get(player_id)

        if not player_data:
            return None

        if stat_name:
            # Return specific stat
            return player_data.get('avg_stats', {}).get(stat_name)
        else:
            # Return all data
            return player_data

    def get_player_stat_value(
        self,
        player_id: int,
        stat_name: str,
        metric: str = 'mean'
    ) -> float:
        """
        Get a specific stat value for a player.

        Args:
            player_id: Player ID
            stat_name: Stat name (e.g., 'rating', 'touches')
            metric: Which metric to return ('mean', 'median', 'max', 'min')

        Returns:
            Stat value or 0.0 if not found
        """
        player_stat = self.get_player_stats(player_id, stat_name)

        if not player_stat:
            return 0.0

        return player_stat.get(metric, 0.0)

    def aggregate_team_stats_from_lineup(
        self,
        player_ids: List[int],
        stat_names: Optional[List[str]] = None,
        aggregation: str = 'sum'
    ) -> Dict[str, float]:
        """
        Aggregate team statistics from a lineup of players.

        Args:
            player_ids: List of player IDs in the lineup
            stat_names: List of stat names to aggregate (None = all available)
            aggregation: How to aggregate ('sum', 'mean', 'median')

        Returns:
            Dictionary mapping stat names to aggregated values
        """
        if not self.is_loaded:
            logger.warning("Player database not loaded, returning empty stats")
            return {}

        # Determine which stats to aggregate
        if stat_names is None:
            # Find all available stats across these players
            all_stats = set()
            for player_id in player_ids:
                player_data = self.player_db.get(player_id)
                if player_data:
                    all_stats.update(player_data.get('avg_stats', {}).keys())
            stat_names = list(all_stats)

        # Aggregate each stat
        aggregated = {}
        for stat_name in stat_names:
            values = []

            for player_id in player_ids:
                value = self.get_player_stat_value(player_id, stat_name, metric='mean')
                if value > 0:  # Only include players with this stat
                    values.append(value)

            if len(values) > 0:
                if aggregation == 'sum':
                    aggregated[stat_name] = float(np.sum(values))
                elif aggregation == 'mean':
                    aggregated[stat_name] = float(np.mean(values))
                elif aggregation == 'median':
                    aggregated[stat_name] = float(np.median(values))
                else:
                    aggregated[stat_name] = 0.0
            else:
                aggregated[stat_name] = 0.0

        return aggregated

    def get_team_average_rating(self, player_ids: List[int]) -> float:
        """
        Get average player rating for a team lineup.

        Args:
            player_ids: List of player IDs

        Returns:
            Average rating (0-10 scale)
        """
        ratings = []
        for player_id in player_ids:
            rating = self.get_player_stat_value(player_id, 'rating', metric='mean')
            if rating > 0:
                ratings.append(rating)

        return float(np.mean(ratings)) if len(ratings) > 0 else 6.5  # Default to 6.5

    def get_lineup_quality_score(self, player_ids: List[int]) -> Dict[str, float]:
        """
        Calculate overall lineup quality metrics.

        Args:
            player_ids: List of player IDs in lineup

        Returns:
            Dictionary with quality metrics
        """
        if not player_ids:
            return {
                'avg_rating': 6.5,
                'total_touches': 0,
                'total_duels_won': 0,
                'total_tackles': 0,
                'total_interceptions': 0,
                'players_found': 0
            }

        # Key metrics for lineup quality
        key_stats = ['rating', 'touches', 'duels_won', 'tackles', 'interceptions']

        aggregated = self.aggregate_team_stats_from_lineup(
            player_ids,
            stat_names=key_stats,
            aggregation='sum'
        )

        # Calculate average rating
        avg_rating = self.get_team_average_rating(player_ids)

        # Count how many players we found in database
        players_found = sum(1 for pid in player_ids if self.player_db.get(pid))

        return {
            'avg_rating': avg_rating,
            'total_touches': aggregated.get('touches', 0),
            'total_duels_won': aggregated.get('duels_won', 0),
            'total_tackles': aggregated.get('tackles', 0),
            'total_interceptions': aggregated.get('interceptions', 0),
            'players_found': players_found,
            'coverage_pct': (players_found / len(player_ids) * 100) if player_ids else 0
        }

    def build_feature_dict_from_lineup(
        self,
        player_ids: List[int],
        prefix: str = 'home'
    ) -> Dict[str, float]:
        """
        Build a feature dictionary compatible with model input.

        This creates features like 'home_player_rating_3', 'home_player_touches_3'
        using actual player data instead of approximations.

        Args:
            player_ids: List of player IDs in lineup
            prefix: 'home' or 'away'

        Returns:
            Dictionary of features (e.g., {home_player_rating_3: 7.2, ...})
        """
        # Core stats to extract
        stat_mapping = {
            'rating': 'player_rating',
            'touches': 'player_touches',
            'duels_won': 'player_duels_won',
            'total_duels': 'player_total_duels',
            'clearances': 'player_clearances',
            'aerials_won': 'player_aerials_won',
            'accurate_passes': 'player_accurate_passes',
            'tackles_won': 'player_tackles_won',
            'interceptions': 'player_interceptions',
            'dribble_attempts': 'player_dribble_attempts',
            'successful_dribbles': 'player_successful_dribbles',
            'dispossessed': 'player_dispossessed',
            'possession_lost': 'player_possession_lost',
            'fouls_drawn': 'player_fouls_drawn',
            'blocked_shots': 'player_blocked_shots',
            'duels_lost': 'player_duels_lost',
            'aerials_lost': 'player_aerials_lost',
            'aerials_total': 'player_aerials_total',
            'saves': 'player_saves',
            'goals_conceded': 'player_goals_conceded',
            'saves_insidebox': 'player_saves_insidebox',
        }

        # Aggregate stats
        aggregated = self.aggregate_team_stats_from_lineup(
            player_ids,
            stat_names=list(stat_mapping.keys()),
            aggregation='sum'  # Sum for team totals
        )

        # Build feature dictionary for all time windows (3, 5, 10)
        features = {}
        for window in [3, 5, 10]:
            for db_stat_name, feature_name in stat_mapping.items():
                value = aggregated.get(db_stat_name, 0.0)
                feature_key = f"{prefix}_{feature_name}_{window}"
                features[feature_key] = value

        return features

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the player database.

        Returns:
            Dictionary with database stats
        """
        if not self.is_loaded:
            return {'loaded': False}

        total_players = len(self.player_db)

        # Count players with different data
        with_names = sum(1 for p in self.player_db.values() if p.get('player_name'))
        with_ratings = sum(1 for p in self.player_db.values()
                          if 'rating' in p.get('avg_stats', {}))

        # Average matches played
        matches_played = [p['matches_played'] for p in self.player_db.values()]
        avg_matches = np.mean(matches_played) if matches_played else 0

        return {
            'loaded': True,
            'total_players': total_players,
            'players_with_names': with_names,
            'players_with_ratings': with_ratings,
            'avg_matches_per_player': float(avg_matches)
        }


# Singleton instance for easy import
_manager_instance = None


def get_player_manager() -> PlayerStatsManager:
    """
    Get singleton player stats manager instance.

    Returns:
        PlayerStatsManager instance
    """
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = PlayerStatsManager()

    return _manager_instance


# Convenience functions
def get_lineup_stats(player_ids: List[int], prefix: str = 'home') -> Dict[str, float]:
    """
    Convenience function to get lineup stats.

    Args:
        player_ids: List of player IDs
        prefix: 'home' or 'away'

    Returns:
        Feature dictionary
    """
    manager = get_player_manager()
    return manager.build_feature_dict_from_lineup(player_ids, prefix)


def get_lineup_quality(player_ids: List[int]) -> Dict[str, float]:
    """
    Convenience function to get lineup quality metrics.

    Args:
        player_ids: List of player IDs

    Returns:
        Quality metrics dictionary
    """
    manager = get_player_manager()
    return manager.get_lineup_quality_score(player_ids)


if __name__ == '__main__':
    # Test the manager
    import logging
    logging.basicConfig(level=logging.INFO)

    manager = PlayerStatsManager()

    if manager.is_loaded:
        print("\n" + "="*80)
        print("PLAYER STATS MANAGER TEST")
        print("="*80)

        stats = manager.get_database_stats()
        print(f"\nDatabase Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Test with random player IDs
        sample_player_ids = list(manager.player_db.keys())[:11]  # First 11 players

        print(f"\nTesting with {len(sample_player_ids)} players...")

        quality = manager.get_lineup_quality_score(sample_player_ids)
        print(f"\nLineup Quality:")
        for key, value in quality.items():
            print(f"  {key}: {value:.2f}")

        features = manager.build_feature_dict_from_lineup(sample_player_ids, 'home')
        print(f"\nGenerated {len(features)} features")
        print("Sample features:")
        for key, value in list(features.items())[:5]:
            print(f"  {key}: {value:.2f}")
    else:
        print("❌ Player database not loaded. Run: python build_player_database.py --full")
