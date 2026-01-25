"""
Shot Analyzer - Shot patterns and efficiency metrics.
"""
from typing import Dict, List
import numpy as np
import logging


logger = logging.getLogger(__name__)


class ShotAnalyzer:
    """Analyze shot patterns and efficiency."""
    
    def calculate_shot_features(
        self,
        matches: List[Dict],
        window: int = 5
    ) -> Dict[str, float]:
        """
        Calculate shot-based features.
        
        Args:
            matches: List of match data with shot statistics
            window: Rolling window size
        
        Returns:
            Dictionary of shot features
        """
        if not matches:
            return self._get_default_shot_features()
        
        recent = matches[-window:]
        
        # Aggregate shot statistics
        shots_total = []
        shots_on_target = []
        shots_inside_box = []
        shots_outside_box = []
        goals_scored = []
        
        for match in recent:
            stats = match.get('team_stats', {})
            shots_total.append(stats.get('shots_total', 0))
            shots_on_target.append(stats.get('shots_on_target', 0))
            shots_inside_box.append(stats.get('shots_insidebox', 0))
            shots_outside_box.append(stats.get('shots_outsidebox', 0))
            goals_scored.append(match.get('goals_scored', 0))
        
        # Calculate averages
        avg_shots = np.mean(shots_total) if shots_total else 0
        avg_sot = np.mean(shots_on_target) if shots_on_target else 0
        avg_inside = np.mean(shots_inside_box) if shots_inside_box else 0
        avg_outside = np.mean(shots_outside_box) if shots_outside_box else 0
        avg_goals = np.mean(goals_scored) if goals_scored else 0
        
        # Calculate percentages
        total_shots_sum = sum(shots_total)
        inside_pct = sum(shots_inside_box) / total_shots_sum if total_shots_sum > 0 else 0
        shot_accuracy = sum(shots_on_target) / total_shots_sum if total_shots_sum > 0 else 0
        
        # Efficiency
        total_goals = sum(goals_scored)
        shots_per_goal = total_shots_sum / total_goals if total_goals > 0 else 0
        
        return {
            'shots_per_match': round(avg_shots, 2),
            'shots_on_target_per_match': round(avg_sot, 2),
            'inside_box_shot_pct': round(inside_pct, 3),
            'outside_box_shot_pct': round(1 - inside_pct, 3),
            'shot_accuracy': round(shot_accuracy, 3),
            'shots_per_goal': round(shots_per_goal, 2),
        }
    
    def _get_default_shot_features(self) -> Dict[str, float]:
        """Get default shot features."""
        return {
            'shots_per_match': 0.0,
            'shots_on_target_per_match': 0.0,
            'inside_box_shot_pct': 0.0,
            'outside_box_shot_pct': 0.0,
            'shot_accuracy': 0.0,
            'shots_per_goal': 0.0,
        }
