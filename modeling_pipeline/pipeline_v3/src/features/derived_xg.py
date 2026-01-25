"""
Derived Expected Goals (xG) Calculator.

Calculates xG from available match statistics without paid add-ons.
"""
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import logging

from config.feature_config import FeatureConfig


logger = logging.getLogger(__name__)


class DerivedXGCalculator:
    """
    Calculate expected goals from base match statistics.
    
    Uses research-based conversion rates:
    - Inside box shots: 12%
    - Outside box shots: 3%
    - Big chances: 35%
    - Corners: 3%
    """
    
    def __init__(
        self,
        inside_box_coef: float = None,
        outside_box_coef: float = None,
        big_chance_coef: float = None,
        corner_coef: float = None
    ):
        """
        Initialize xG calculator.
        
        Args:
            inside_box_coef: xG per inside box shot
            outside_box_coef: xG per outside box shot
            big_chance_coef: xG per big chance
            corner_coef: xG per corner
        """
        self.inside_box_coef = inside_box_coef or FeatureConfig.XG_INSIDE_BOX
        self.outside_box_coef = outside_box_coef or FeatureConfig.XG_OUTSIDE_BOX
        self.big_chance_coef = big_chance_coef or FeatureConfig.XG_BIG_CHANCE
        self.corner_coef = corner_coef or FeatureConfig.XG_CORNER
        self.max_accuracy_mult = FeatureConfig.XG_ACCURACY_MULTIPLIER_MAX
    
    def calculate_xg(self, match_stats: Dict) -> float:
        """
        Calculate expected goals from match statistics.
        
        Args:
            match_stats: Dictionary with keys:
                - shots_insidebox: int
                - shots_outsidebox: int
                - big_chances_created: int
                - corners: int
                - shots_total: int
                - shots_on_target: int
        
        Returns:
            Derived xG value
        """
        # Base xG from shot location
        xg_inside = match_stats.get('shots_insidebox', 0) * self.inside_box_coef
        xg_outside = match_stats.get('shots_outsidebox', 0) * self.outside_box_coef
        
        # Big chances
        xg_big_chances = match_stats.get('big_chances_created', 0) * self.big_chance_coef
        
        # Set pieces
        xg_corners = match_stats.get('corners', 0) * self.corner_coef
        
        # Calculate accuracy multiplier (1.0 to max)
        shots_total = match_stats.get('shots_total', 1)
        shots_on_target = match_stats.get('shots_on_target', 0)
        shot_accuracy = shots_on_target / max(shots_total, 1)
        
        # Linear scaling from 1.0 to max based on accuracy
        accuracy_multiplier = 1.0 + (shot_accuracy * (self.max_accuracy_mult - 1.0))
        
        # Combined xG
        base_xg = xg_inside + xg_outside + xg_big_chances + xg_corners
        derived_xg = base_xg * accuracy_multiplier
        
        return round(derived_xg, 2)
    
    def calculate_xga(
        self,
        team_stats: Dict,
        opponent_stats: Dict
    ) -> float:
        """
        Calculate expected goals against (defensive quality).
        
        Args:
            team_stats: Team's defensive statistics
            opponent_stats: Opponent's attacking statistics
        
        Returns:
            Derived xGA value
        """
        # Opponent's attacking threat
        opponent_xg = self.calculate_xg(opponent_stats)
        
        # Defensive pressure reduction
        tackles = team_stats.get('tackles', 0)
        interceptions = team_stats.get('interceptions', 0)
        clearances = team_stats.get('clearances', 0)
        defensive_actions = tackles + interceptions + clearances
        
        # More defensive actions = lower xGA (max 30% reduction)
        defensive_multiplier = max(0.7, 1 - (defensive_actions / 100))
        
        derived_xga = opponent_xg * defensive_multiplier
        
        return round(derived_xga, 2)
    
    def calculate_rolling_xg(
        self,
        match_stats_list: List[Dict],
        window: int = 5
    ) -> Dict[str, float]:
        """
        Calculate rolling average xG metrics.
        
        Args:
            match_stats_list: List of match statistics (most recent last)
            window: Rolling window size
        
        Returns:
            Dictionary with rolling xG metrics
        """
        if not match_stats_list:
            return {
                'xg_per_match': 0.0,
                'xga_per_match': 0.0,
                'xgd_per_match': 0.0,
            }
        
        # Take last N matches
        recent_matches = match_stats_list[-window:]
        
        # Calculate xG for each match
        xg_values = []
        for match in recent_matches:
            xg = self.calculate_xg(match.get('team_stats', {}))
            xg_values.append(xg)
        
        # Calculate averages
        xg_avg = np.mean(xg_values) if xg_values else 0.0
        
        # For xGA, we'd need opponent stats - simplified here
        xga_avg = 0.0  # Would calculate from opponent_stats
        
        return {
            'xg_per_match': round(xg_avg, 2),
            'xga_per_match': round(xga_avg, 2),
            'xgd_per_match': round(xg_avg - xga_avg, 2),
        }
    
    def calculate_xg_trend(
        self,
        match_stats_list: List[Dict],
        window: int = 10
    ) -> float:
        """
        Calculate xG trend (improving/declining).
        
        Args:
            match_stats_list: List of match statistics
            window: Window size for trend
        
        Returns:
            Trend slope (positive = improving)
        """
        if len(match_stats_list) < 3:
            return 0.0
        
        recent_matches = match_stats_list[-window:]
        xg_values = [
            self.calculate_xg(match.get('team_stats', {}))
            for match in recent_matches
        ]
        
        if len(xg_values) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(xg_values))
        slope = np.polyfit(x, xg_values, 1)[0]
        
        return round(slope, 3)
    
    def calculate_xg_features(
        self,
        team_matches: List[Dict],
        window: int = 5
    ) -> Dict[str, float]:
        """
        Calculate all xG-based features for a team.
        
        Args:
            team_matches: List of team's match data
            window: Rolling window size
        
        Returns:
            Dictionary of xG features
        """
        if not team_matches:
            return self._get_default_xg_features()
        
        recent_matches = team_matches[-window:]
        
        # Calculate xG for each match
        xg_values = []
        xga_values = []
        goals_values = []
        
        for match in recent_matches:
            team_stats = match.get('team_stats', {})
            opp_stats = match.get('opponent_stats', {})
            
            xg = self.calculate_xg(team_stats)
            xga = self.calculate_xga(team_stats, opp_stats)
            goals = match.get('goals_scored', 0)
            
            xg_values.append(xg)
            xga_values.append(xga)
            goals_values.append(goals)
        
        # Calculate metrics
        xg_avg = np.mean(xg_values) if xg_values else 0.0
        xga_avg = np.mean(xga_values) if xga_values else 0.0
        xgd_avg = xg_avg - xga_avg
        
        # Performance vs expectation
        goals_avg = np.mean(goals_values) if goals_values else 0.0
        goals_vs_xg = goals_avg - xg_avg
        
        # Shot quality
        total_shots = sum(m.get('team_stats', {}).get('shots_total', 0) for m in recent_matches)
        xg_per_shot = xg_avg / (total_shots / len(recent_matches)) if total_shots > 0 else 0.0
        
        # Trends
        xg_trend = self.calculate_xg_trend(team_matches, window=10)
        
        return {
            'derived_xg_per_match': round(xg_avg, 2),
            'derived_xga_per_match': round(xga_avg, 2),
            'derived_xgd': round(xgd_avg, 2),
            'goals_vs_xg': round(goals_vs_xg, 2),
            'xg_per_shot': round(xg_per_shot, 3),
            'xg_trend': xg_trend,
        }
    
    def _get_default_xg_features(self) -> Dict[str, float]:
        """Get default xG features when no data available."""
        return {
            'derived_xg_per_match': 1.0,
            'derived_xga_per_match': 1.0,
            'derived_xgd': 0.0,
            'goals_vs_xg': 0.0,
            'xg_per_shot': 0.0,
            'xg_trend': 0.0,
        }
