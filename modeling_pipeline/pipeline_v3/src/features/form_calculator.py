"""
Form Calculator - Recent performance metrics.

Calculates points, goals, and win/draw/loss counts over rolling windows.
"""
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import logging


logger = logging.getLogger(__name__)


class FormCalculator:
    """Calculate team form metrics over various time windows."""
    
    def calculate_points_from_result(self, result: str) -> int:
        """
        Convert match result to points.
        
        Args:
            result: 'W', 'D', or 'L'
        
        Returns:
            Points (3, 1, or 0)
        """
        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        else:
            return 0
    
    def calculate_form_features(
        self,
        matches: List[Dict],
        windows: List[int] = [3, 5, 10]
    ) -> Dict[str, float]:
        """
        Calculate form features over multiple windows.
        
        Args:
            matches: List of match data (most recent last)
                Each match should have: result, goals_scored, goals_conceded
            windows: List of window sizes
        
        Returns:
            Dictionary of form features
        """
        if not matches:
            return self._get_default_form_features()
        
        features = {}
        
        for window in windows:
            recent = matches[-window:]
            
            # Points
            points = sum(self.calculate_points_from_result(m.get('result', 'L')) for m in recent)
            features[f'points_last_{window}'] = points
            
            # Wins, draws, losses
            wins = sum(1 for m in recent if m.get('result') == 'W')
            draws = sum(1 for m in recent if m.get('result') == 'D')
            losses = sum(1 for m in recent if m.get('result') == 'L')
            
            features[f'wins_last_{window}'] = wins
            features[f'draws_last_{window}'] = draws
            features[f'losses_last_{window}'] = losses
            
            # Goals
            goals_scored = sum(m.get('goals_scored', 0) for m in recent)
            goals_conceded = sum(m.get('goals_conceded', 0) for m in recent)
            
            features[f'goals_scored_last_{window}'] = goals_scored
            features[f'goals_conceded_last_{window}'] = goals_conceded
            features[f'goal_diff_last_{window}'] = goals_scored - goals_conceded
        
        return features
    
    def calculate_weighted_form(
        self,
        matches: List[Dict],
        window: int = 5,
        alpha: float = 0.3
    ) -> float:
        """
        Calculate exponentially weighted form (recent matches weighted more).
        
        Args:
            matches: List of match data
            window: Number of matches to consider
            alpha: Weighting factor (higher = more weight on recent)
        
        Returns:
            Weighted form score
        """
        if not matches:
            return 0.0
        
        recent = matches[-window:]
        points = [self.calculate_points_from_result(m.get('result', 'L')) for m in recent]
        
        # Calculate exponential weights (most recent gets highest weight)
        weights = [alpha * (1 - alpha) ** i for i in range(len(points))]
        weights.reverse()  # Reverse so most recent gets highest weight
        
        # Weighted average
        weighted_points = sum(p * w for p, w in zip(points, weights))
        total_weight = sum(weights)
        
        return round(weighted_points / total_weight if total_weight > 0 else 0.0, 2)
    
    def calculate_streaks(self, matches: List[Dict]) -> Dict[str, int]:
        """
        Calculate current streaks (wins, unbeaten, clean sheets).
        
        Args:
            matches: List of match data (most recent last)
        
        Returns:
            Dictionary of streak features
        """
        if not matches:
            return {
                'win_streak': 0,
                'unbeaten_streak': 0,
                'clean_sheet_streak': 0,
            }
        
        # Win streak
        win_streak = 0
        for match in reversed(matches):
            if match.get('result') == 'W':
                win_streak += 1
            else:
                break
        
        # Unbeaten streak
        unbeaten_streak = 0
        for match in reversed(matches):
            if match.get('result') in ['W', 'D']:
                unbeaten_streak += 1
            else:
                break
        
        # Clean sheet streak
        clean_sheet_streak = 0
        for match in reversed(matches):
            if match.get('goals_conceded', 1) == 0:
                clean_sheet_streak += 1
            else:
                break
        
        return {
            'win_streak': win_streak,
            'unbeaten_streak': unbeaten_streak,
            'clean_sheet_streak': clean_sheet_streak,
        }
    
    def calculate_form_trend(
        self,
        matches: List[Dict],
        window: int = 10
    ) -> float:
        """
        Calculate form trend (improving/declining).
        
        Args:
            matches: List of match data
            window: Window size for trend
        
        Returns:
            Trend slope (positive = improving form)
        """
        if len(matches) < 3:
            return 0.0
        
        recent = matches[-window:]
        points = [self.calculate_points_from_result(m.get('result', 'L')) for m in recent]
        
        if len(points) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(points))
        slope = np.polyfit(x, points, 1)[0]
        
        return round(slope, 3)
    
    def _get_default_form_features(self) -> Dict[str, float]:
        """Get default form features when no data available."""
        features = {}
        for window in [3, 5, 10]:
            features[f'points_last_{window}'] = 0
            features[f'wins_last_{window}'] = 0
            features[f'draws_last_{window}'] = 0
            features[f'losses_last_{window}'] = 0
            features[f'goals_scored_last_{window}'] = 0
            features[f'goals_conceded_last_{window}'] = 0
            features[f'goal_diff_last_{window}'] = 0
        return features
