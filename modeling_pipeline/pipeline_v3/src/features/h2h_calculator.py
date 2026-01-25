"""
Head-to-Head (H2H) Calculator.

Analyzes historical matchups between two teams.
"""
from typing import Dict, List
import logging


logger = logging.getLogger(__name__)


class H2HCalculator:
    """Calculate head-to-head features between two teams."""
    
    def calculate_h2h_features(
        self,
        h2h_matches: List[Dict],
        home_team_id: int,
        away_team_id: int,
        window: int = 5
    ) -> Dict[str, float]:
        """
        Calculate H2H features from historical matchups.
        
        Args:
            h2h_matches: List of historical H2H matches
            home_team_id: Home team ID
            away_team_id: Away team ID
            window: Number of recent H2H matches to consider
        
        Returns:
            Dictionary of H2H features
        """
        if not h2h_matches:
            return self._get_default_h2h_features()
        
        # Take most recent N matches
        recent_h2h = h2h_matches[-window:]
        
        # Count results
        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals_total = 0
        away_goals_total = 0
        btts_count = 0
        over_2_5_count = 0
        
        for match in recent_h2h:
            home_id = match.get('home_team_id')
            home_goals = match.get('home_goals', 0)
            away_goals = match.get('away_goals', 0)
            
            # Determine winner from perspective of current home team
            if home_id == home_team_id:
                # Match was at current home team's venue
                if home_goals > away_goals:
                    home_wins += 1
                elif home_goals < away_goals:
                    away_wins += 1
                else:
                    draws += 1
                home_goals_total += home_goals
                away_goals_total += away_goals
            else:
                # Match was at current away team's venue
                if away_goals > home_goals:
                    home_wins += 1
                elif away_goals < home_goals:
                    away_wins += 1
                else:
                    draws += 1
                home_goals_total += away_goals
                away_goals_total += home_goals
            
            # Both teams to score
            if home_goals > 0 and away_goals > 0:
                btts_count += 1
            
            # Over 2.5 goals
            if (home_goals + away_goals) > 2.5:
                over_2_5_count += 1
        
        total_matches = len(recent_h2h)
        
        return {
            'h2h_home_wins_last_5': home_wins,
            'h2h_draws_last_5': draws,
            'h2h_away_wins_last_5': away_wins,
            'h2h_home_goals_avg': round(home_goals_total / total_matches, 2),
            'h2h_away_goals_avg': round(away_goals_total / total_matches, 2),
            'h2h_home_win_pct': round(home_wins / total_matches, 3),
            'h2h_btts_pct': round(btts_count / total_matches, 3),
            'h2h_over_2_5_pct': round(over_2_5_count / total_matches, 3),
        }
    
    def _get_default_h2h_features(self) -> Dict[str, float]:
        """Get default H2H features when no data available."""
        return {
            'h2h_home_wins_last_5': 0,
            'h2h_draws_last_5': 0,
            'h2h_away_wins_last_5': 0,
            'h2h_home_goals_avg': 0.0,
            'h2h_away_goals_avg': 0.0,
            'h2h_home_win_pct': 0.0,
            'h2h_btts_pct': 0.0,
            'h2h_over_2_5_pct': 0.0,
        }
