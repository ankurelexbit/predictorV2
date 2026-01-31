"""
Elo Rating Calculator for V4 Pipeline.

Calculates and tracks Elo ratings for teams over time.
"""
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EloCalculator:
    """
    Calculate Elo ratings for teams from fixture history.
    
    Uses standard Elo formula with home advantage.
    """
    
    def __init__(self, k_factor: int = 32, home_advantage: int = 35, initial_elo: int = 1500):
        """
        Initialize Elo calculator.
        
        Args:
            k_factor: Update speed (32 is standard)
            home_advantage: Home team bonus (35 calibrated for modern football)
            initial_elo: Starting Elo for new teams
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_elo = initial_elo
        
        # Track Elo history: {team_id: [(date, elo), ...]}
        self.elo_history = {}
        
        logger.info(f"Initialized EloCalculator (k={k_factor}, home_adv={home_advantage})")
    
    def calculate_elo_history(self, fixtures_df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate Elo ratings for all teams from fixture history.
        
        Args:
            fixtures_df: DataFrame with all fixtures (sorted by date)
            
        Returns:
            Dict mapping team_id -> current Elo
        """
        # Initialize Elo for all teams
        current_elo = {}
        self.elo_history = {}
        
        # Process fixtures chronologically
        for _, fixture in fixtures_df.iterrows():
            # Skip if no result
            if pd.isna(fixture.get('result')):
                continue
            
            home_id = fixture['home_team_id']
            away_id = fixture['away_team_id']
            home_score = fixture['home_score']
            away_score = fixture['away_score']
            match_date = fixture['starting_at']
            
            # Initialize teams if not seen
            if home_id not in current_elo:
                current_elo[home_id] = self.initial_elo
                self.elo_history[home_id] = [(match_date, self.initial_elo)]
            
            if away_id not in current_elo:
                current_elo[away_id] = self.initial_elo
                self.elo_history[away_id] = [(match_date, self.initial_elo)]
            
            # Get current Elos
            home_elo = current_elo[home_id]
            away_elo = current_elo[away_id]
            
            # Determine result (from home team perspective)
            if home_score > away_score:
                home_result = 1.0
                away_result = 0.0
            elif home_score < away_score:
                home_result = 0.0
                away_result = 1.0
            else:
                home_result = 0.5
                away_result = 0.5
            
            # Update Elos
            new_home_elo = self._update_elo(
                home_elo, away_elo, home_result, is_home=True
            )
            new_away_elo = self._update_elo(
                away_elo, home_elo, away_result, is_home=False
            )
            
            # Store new Elos
            current_elo[home_id] = new_home_elo
            current_elo[away_id] = new_away_elo
            
            # Record history
            self.elo_history[home_id].append((match_date, new_home_elo))
            self.elo_history[away_id].append((match_date, new_away_elo))
        
        logger.info(f"Calculated Elo history for {len(current_elo)} teams")
        return current_elo
    
    def get_elo_at_date(
        self,
        team_id: int,
        as_of_date: datetime
    ) -> Optional[float]:
        """
        Get team's Elo rating at a specific date.
        
        Args:
            team_id: Team ID
            as_of_date: Date to get Elo for
            
        Returns:
            Elo rating or None if team not found
        """
        if team_id not in self.elo_history:
            return None
        
        history = self.elo_history[team_id]
        
        # Find most recent Elo before as_of_date
        elo = None
        for date, rating in history:
            if date < as_of_date:
                elo = rating
            else:
                break
        
        return elo if elo is not None else self.initial_elo
    
    def get_elo_change(
        self,
        team_id: int,
        as_of_date: datetime,
        num_matches: int = 5
    ) -> Optional[float]:
        """
        Get Elo change over last N matches.
        
        Args:
            team_id: Team ID
            as_of_date: Date cutoff
            num_matches: Number of matches to look back
            
        Returns:
            Elo change or None if not enough history
        """
        if team_id not in self.elo_history:
            return None
        
        history = self.elo_history[team_id]
        
        # Get ratings before as_of_date
        ratings_before = [(date, rating) for date, rating in history if date < as_of_date]
        
        if len(ratings_before) < 2:
            return None
        
        # Get current and N matches ago
        current_elo = ratings_before[-1][1]
        
        # Find Elo N matches ago
        lookback_idx = max(0, len(ratings_before) - num_matches - 1)
        old_elo = ratings_before[lookback_idx][1]
        
        return current_elo - old_elo
    
    def _update_elo(
        self,
        team_elo: float,
        opponent_elo: float,
        result: float,
        is_home: bool
    ) -> float:
        """
        Update Elo rating based on match result.
        
        Args:
            team_elo: Team's current Elo
            opponent_elo: Opponent's Elo
            result: 1.0 (win), 0.5 (draw), 0.0 (loss)
            is_home: Whether team is playing at home
            
        Returns:
            New Elo rating
        """
        # Apply home advantage
        if is_home:
            adjusted_elo = team_elo + self.home_advantage
        else:
            adjusted_elo = team_elo
        
        # Calculate expected score
        expected = 1 / (1 + 10 ** ((opponent_elo - adjusted_elo) / 400))
        
        # Update Elo
        new_elo = team_elo + self.k_factor * (result - expected)
        
        return new_elo
