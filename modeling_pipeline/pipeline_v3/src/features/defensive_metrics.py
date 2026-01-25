"""
Defensive Metrics Calculator.

Calculates PPDA, defensive actions, and possession metrics.
"""
from typing import Dict, List
import numpy as np
import logging


logger = logging.getLogger(__name__)


class DefensiveMetrics:
    """Calculate defensive intensity and efficiency metrics."""
    
    def calculate_ppda(self, team_stats: Dict, opponent_stats: Dict) -> float:
        """
        Calculate Passes Per Defensive Action (PPDA).
        
        Lower PPDA = more aggressive pressing
        
        Args:
            team_stats: Team's defensive statistics
            opponent_stats: Opponent's passing statistics
        
        Returns:
            PPDA value
        """
        tackles = team_stats.get('tackles', 0)
        interceptions = team_stats.get('interceptions', 0)
        defensive_actions = tackles + interceptions
        
        opponent_passes = opponent_stats.get('passes', 0)
        
        if defensive_actions == 0:
            return 999.0  # Very passive
        
        ppda = opponent_passes / defensive_actions
        return round(ppda, 2)
    
    def calculate_defensive_features(
        self,
        matches: List[Dict],
        window: int = 5
    ) -> Dict[str, float]:
        """
        Calculate defensive features.
        
        Args:
            matches: List of match data
            window: Rolling window size
        
        Returns:
            Dictionary of defensive features
        """
        if not matches:
            return self._get_default_defensive_features()
        
        recent = matches[-window:]
        
        # Aggregate defensive statistics
        ppda_values = []
        tackles_list = []
        interceptions_list = []
        clearances_list = []
        possession_list = []
        
        for match in recent:
            team_stats = match.get('team_stats', {})
            opp_stats = match.get('opponent_stats', {})
            
            # PPDA
            ppda = self.calculate_ppda(team_stats, opp_stats)
            ppda_values.append(ppda)
            
            # Defensive actions
            tackles_list.append(team_stats.get('tackles', 0))
            interceptions_list.append(team_stats.get('interceptions', 0))
            clearances_list.append(team_stats.get('clearances', 0))
            
            # Possession
            possession_list.append(team_stats.get('possession', 50))
        
        # Calculate averages
        avg_ppda = np.mean(ppda_values) if ppda_values else 0
        avg_tackles = np.mean(tackles_list) if tackles_list else 0
        avg_interceptions = np.mean(interceptions_list) if interceptions_list else 0
        avg_clearances = np.mean(clearances_list) if clearances_list else 0
        avg_possession = np.mean(possession_list) if possession_list else 50
        
        # Defensive actions per 90
        avg_defensive_actions = avg_tackles + avg_interceptions + avg_clearances
        
        return {
            'ppda': round(avg_ppda, 2),
            'tackles_per_90': round(avg_tackles, 2),
            'interceptions_per_90': round(avg_interceptions, 2),
            'clearances_per_90': round(avg_clearances, 2),
            'defensive_actions_per_90': round(avg_defensive_actions, 2),
            'possession_pct': round(avg_possession, 2),
        }
    
    def _get_default_defensive_features(self) -> Dict[str, float]:
        """Get default defensive features."""
        return {
            'ppda': 0.0,
            'tackles_per_90': 0.0,
            'interceptions_per_90': 0.0,
            'clearances_per_90': 0.0,
            'defensive_actions_per_90': 0.0,
            'possession_pct': 50.0,
        }
