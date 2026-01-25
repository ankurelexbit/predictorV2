"""
Elo Rating Calculator for football teams.

Implements standard Elo rating system with home advantage.
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging

from config.feature_config import FeatureConfig


logger = logging.getLogger(__name__)


class EloCalculator:
    """
    Calculate and track Elo ratings for football teams.
    
    Uses standard Elo formula with home advantage adjustment.
    """
    
    def __init__(
        self,
        k_factor: int = None,
        home_advantage: int = None,
        initial_rating: int = None
    ):
        """
        Initialize Elo calculator.
        
        Args:
            k_factor: How quickly ratings change (default: 32)
            home_advantage: Home team bonus points (default: 35)
            initial_rating: Starting Elo for new teams (default: 1500)
        """
        self.k_factor = k_factor or FeatureConfig.ELO_K_FACTOR
        self.home_advantage = home_advantage or FeatureConfig.ELO_HOME_ADVANTAGE
        self.initial_rating = initial_rating or FeatureConfig.ELO_INITIAL_RATING
        
        # Track current Elo ratings
        self.ratings: Dict[int, float] = {}
        
        # Track Elo history
        self.history: Dict[int, List[Tuple[datetime, float]]] = {}
    
    def get_rating(self, team_id: int) -> float:
        """
        Get current Elo rating for a team.
        
        Args:
            team_id: Team ID
        
        Returns:
            Current Elo rating
        """
        if team_id not in self.ratings:
            self.ratings[team_id] = self.initial_rating
            self.history[team_id] = []
        return self.ratings[team_id]
    
    def calculate_expected_score(
        self,
        team_elo: float,
        opponent_elo: float,
        is_home: bool = False
    ) -> float:
        """
        Calculate expected score (0-1) for a team.
        
        Args:
            team_elo: Team's Elo rating
            opponent_elo: Opponent's Elo rating
            is_home: Whether team is playing at home
        
        Returns:
            Expected score (0 = certain loss, 1 = certain win)
        """
        # Adjust for home advantage
        if is_home:
            team_elo += self.home_advantage
        
        # Standard Elo expected score formula
        expected = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))
        return expected
    
    def update_ratings(
        self,
        home_team_id: int,
        away_team_id: int,
        home_goals: int,
        away_goals: int,
        match_date: datetime
    ) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_date: Date of the match
        
        Returns:
            Tuple of (new_home_elo, new_away_elo)
        """
        # Get current ratings
        home_elo = self.get_rating(home_team_id)
        away_elo = self.get_rating(away_team_id)
        
        # Calculate expected scores
        home_expected = self.calculate_expected_score(home_elo, away_elo, is_home=True)
        away_expected = 1 - home_expected
        
        # Determine actual result
        if home_goals > away_goals:
            home_result = 1.0
            away_result = 0.0
        elif home_goals < away_goals:
            home_result = 0.0
            away_result = 1.0
        else:
            home_result = 0.5
            away_result = 0.5
        
        # Update ratings
        home_new_elo = home_elo + self.k_factor * (home_result - home_expected)
        away_new_elo = away_elo + self.k_factor * (away_result - away_expected)
        
        # Store new ratings
        self.ratings[home_team_id] = home_new_elo
        self.ratings[away_team_id] = away_new_elo
        
        # Record history
        self.history[home_team_id].append((match_date, home_new_elo))
        self.history[away_team_id].append((match_date, away_new_elo))
        
        logger.debug(
            f"Updated Elo: Home {home_team_id}: {home_elo:.0f} -> {home_new_elo:.0f}, "
            f"Away {away_team_id}: {away_elo:.0f} -> {away_new_elo:.0f}"
        )
        
        return home_new_elo, away_new_elo
    
    def get_rating_at_date(self, team_id: int, target_date: datetime) -> float:
        """
        Get Elo rating at a specific date (for historical features).
        
        Args:
            team_id: Team ID
            target_date: Date to get rating for
        
        Returns:
            Elo rating at that date
        """
        if team_id not in self.history or not self.history[team_id]:
            return self.initial_rating
        
        # Find last rating before target_date
        team_history = self.history[team_id]
        
        # Binary search for efficiency
        for date, rating in reversed(team_history):
            if date <= target_date:
                return rating
        
        # If no rating before target_date, return initial
        return self.initial_rating
    
    def get_elo_change(
        self,
        team_id: int,
        target_date: datetime,
        n_matches: int = 5
    ) -> float:
        """
        Get Elo change over last N matches before a date.
        
        Args:
            team_id: Team ID
            target_date: Date to calculate from
            n_matches: Number of matches to look back
        
        Returns:
            Elo change (positive = improving, negative = declining)
        """
        if team_id not in self.history or not self.history[team_id]:
            return 0.0
        
        # Get ratings before target_date
        relevant_history = [
            (date, rating) for date, rating in self.history[team_id]
            if date <= target_date
        ]
        
        if len(relevant_history) < 2:
            return 0.0
        
        # Get last n_matches
        recent = relevant_history[-n_matches:]
        
        if len(recent) < 2:
            return 0.0
        
        # Calculate change
        elo_change = recent[-1][1] - recent[0][1]
        return elo_change
    
    def calculate_elo_features(
        self,
        home_team_id: int,
        away_team_id: int,
        match_date: datetime,
        league_avg_elo: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all Elo-based features for a match.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            match_date: Match date
            league_avg_elo: Average Elo in league (optional)
        
        Returns:
            Dictionary of Elo features
        """
        # Get current ratings at match date
        home_elo = self.get_rating_at_date(home_team_id, match_date)
        away_elo = self.get_rating_at_date(away_team_id, match_date)
        
        # Calculate Elo changes
        home_elo_change_5 = self.get_elo_change(home_team_id, match_date, n_matches=5)
        away_elo_change_5 = self.get_elo_change(away_team_id, match_date, n_matches=5)
        home_elo_change_10 = self.get_elo_change(home_team_id, match_date, n_matches=10)
        away_elo_change_10 = self.get_elo_change(away_team_id, match_date, n_matches=10)
        
        # Calculate differentials
        elo_diff = home_elo - away_elo
        elo_diff_with_ha = elo_diff + self.home_advantage
        
        # League context (if available)
        league_avg = league_avg_elo or self.initial_rating
        home_elo_vs_league = home_elo - league_avg
        away_elo_vs_league = away_elo - league_avg
        
        return {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'elo_diff_with_ha': elo_diff_with_ha,
            'home_elo_change_5': home_elo_change_5,
            'away_elo_change_5': away_elo_change_5,
            'home_elo_change_10': home_elo_change_10,
            'away_elo_change_10': away_elo_change_10,
            'home_elo_vs_league_avg': home_elo_vs_league,
            'away_elo_vs_league_avg': away_elo_vs_league,
        }
    
    def reset_ratings(self):
        """Reset all ratings (e.g., for new season)."""
        self.ratings = {}
        self.history = {}
    
    def apply_season_regression(self, regression_factor: float = None):
        """
        Apply regression to mean between seasons.
        
        Args:
            regression_factor: How much to regress (0-1, default 0.5)
        """
        factor = regression_factor or FeatureConfig.ELO_REGRESSION_FACTOR
        
        for team_id in self.ratings:
            current_elo = self.ratings[team_id]
            # Regress towards initial rating
            new_elo = current_elo * (1 - factor) + self.initial_rating * factor
            self.ratings[team_id] = new_elo
        
        logger.info(f"Applied season regression with factor {factor}")
