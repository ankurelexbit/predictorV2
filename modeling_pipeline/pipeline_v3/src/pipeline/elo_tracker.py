"""
Elo Rating Tracker

Tracks Elo ratings chronologically for all teams with point-in-time correctness.
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EloTracker:
    """Track Elo ratings chronologically for all teams."""
    
    def __init__(
        self, 
        k_factor: int = 32, 
        home_advantage: int = 35, 
        initial_elo: int = 1500,
        regression_factor: float = 0.5
    ):
        """
        Initialize Elo tracker.
        
        Args:
            k_factor: Elo K-factor (update speed)
            home_advantage: Home advantage in Elo points
            initial_elo: Initial Elo rating for new teams
            regression_factor: Regression to mean between seasons (0-1)
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_elo = initial_elo
        self.regression_factor = regression_factor
        
        # Store Elo history: {team_id: [(date, elo), ...]}
        self.elo_history = {}
        
        # Current Elo ratings: {team_id: elo}
        self.current_elo = {}
        
        # Track season changes for regression
        self.last_season = {}
        
        logger.info(f"Initialized EloTracker (K={k_factor}, HA={home_advantage}, Initial={initial_elo})")
    
    def process_all_fixtures(self, fixtures_df: pd.DataFrame):
        """
        Process all fixtures chronologically to build Elo history.
        
        CRITICAL: Fixtures must be sorted chronologically.
        
        Args:
            fixtures_df: DataFrame with fixtures (sorted by starting_at)
        """
        logger.info(f"Processing {len(fixtures_df)} fixtures to build Elo history...")
        
        # Ensure sorted
        fixtures_df = fixtures_df.sort_values('starting_at').reset_index(drop=True)
        
        processed = 0
        
        for idx, fixture in fixtures_df.iterrows():
            # Skip if not finished
            if fixture.get('state') != 'FT':
                continue
            
            # Skip if scores missing
            if pd.isna(fixture.get('home_score')) or pd.isna(fixture.get('away_score')):
                continue
            
            home_team_id = fixture['home_team_id']
            away_team_id = fixture['away_team_id']
            home_score = fixture['home_score']
            away_score = fixture['away_score']
            match_date = fixture['starting_at']
            season_id = fixture.get('season_id')
            
            # Check for season change (apply regression to mean)
            self._check_season_change(home_team_id, season_id)
            self._check_season_change(away_team_id, season_id)
            
            # Get current Elo ratings
            home_elo = self._get_current_elo(home_team_id)
            away_elo = self._get_current_elo(away_team_id)
            
            # Calculate result (from home team perspective)
            if home_score > away_score:
                result = 1.0  # Home win
            elif home_score == away_score:
                result = 0.5  # Draw
            else:
                result = 0.0  # Away win
            
            # Update Elo ratings
            new_home_elo, new_away_elo = self._update_elo_ratings(
                home_elo, away_elo, result
            )
            
            # Store in history
            self._add_to_history(home_team_id, match_date, new_home_elo)
            self._add_to_history(away_team_id, match_date, new_away_elo)
            
            # Update current ratings
            self.current_elo[home_team_id] = new_home_elo
            self.current_elo[away_team_id] = new_away_elo
            
            processed += 1
            
            if processed % 1000 == 0:
                logger.info(f"Processed {processed} fixtures...")
        
        logger.info(f"Elo history built for {len(self.elo_history)} teams from {processed} fixtures")
    
    def _get_current_elo(self, team_id: int) -> float:
        """Get current Elo rating for a team."""
        if team_id not in self.current_elo:
            self.current_elo[team_id] = self.initial_elo
        return self.current_elo[team_id]
    
    def _update_elo_ratings(
        self, 
        home_elo: float, 
        away_elo: float, 
        result: float
    ) -> tuple:
        """
        Update Elo ratings based on match result.
        
        Args:
            home_elo: Home team Elo
            away_elo: Away team Elo
            result: Match result (1.0=home win, 0.5=draw, 0.0=away win)
            
        Returns:
            Tuple of (new_home_elo, new_away_elo)
        """
        # Expected score for home team
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - self.home_advantage) / 400))
        
        # Expected score for away team
        expected_away = 1 - expected_home
        
        # Update Elo
        new_home_elo = home_elo + self.k_factor * (result - expected_home)
        new_away_elo = away_elo + self.k_factor * ((1 - result) - expected_away)
        
        return new_home_elo, new_away_elo
    
    def _add_to_history(self, team_id: int, date: datetime, elo: float):
        """Add Elo rating to team's history."""
        if team_id not in self.elo_history:
            self.elo_history[team_id] = []
        
        self.elo_history[team_id].append((date, elo))
    
    def _check_season_change(self, team_id: int, season_id: int):
        """
        Check if season changed and apply regression to mean.
        
        Args:
            team_id: Team ID
            season_id: Current season ID
        """
        if team_id not in self.last_season:
            self.last_season[team_id] = season_id
            return
        
        if season_id != self.last_season[team_id]:
            # Season changed - apply regression to mean
            current_elo = self._get_current_elo(team_id)
            regressed_elo = current_elo * (1 - self.regression_factor) + self.initial_elo * self.regression_factor
            self.current_elo[team_id] = regressed_elo
            self.last_season[team_id] = season_id
            
            logger.debug(f"Season change for team {team_id}: {current_elo:.1f} -> {regressed_elo:.1f}")
    
    def get_elo_at_date(self, team_id: int, as_of_date: str) -> float:
        """
        Get team's Elo rating at a specific date.
        
        Args:
            team_id: Team ID
            as_of_date: Date cutoff (ISO format)
            
        Returns:
            Elo rating (or initial_elo if no history)
        """
        if team_id not in self.elo_history:
            return self.initial_elo
        
        cutoff_date = pd.to_datetime(as_of_date)
        
        # Find most recent Elo before cutoff date
        history = self.elo_history[team_id]
        
        # Binary search or linear search (history is sorted)
        elo = self.initial_elo
        for date, rating in history:
            if date < cutoff_date:
                elo = rating
            else:
                break
        
        return elo
    
    def get_elo_change(
        self, 
        team_id: int, 
        as_of_date: str, 
        n_matches: int = 5
    ) -> float:
        """
        Get Elo change over last n matches.
        
        Args:
            team_id: Team ID
            as_of_date: Date cutoff
            n_matches: Number of matches to look back
            
        Returns:
            Elo change (current - n_matches_ago)
        """
        if team_id not in self.elo_history:
            return 0.0
        
        cutoff_date = pd.to_datetime(as_of_date)
        
        # Get history before cutoff
        history = [(d, e) for d, e in self.elo_history[team_id] if d < cutoff_date]
        
        if len(history) == 0:
            return 0.0
        
        current_elo = history[-1][1]
        
        if len(history) <= n_matches:
            # Not enough history
            past_elo = self.initial_elo
        else:
            past_elo = history[-(n_matches + 1)][1]
        
        return current_elo - past_elo
    
    def get_league_avg_elo(
        self, 
        league_id: int, 
        season_id: int,
        as_of_date: str,
        fixtures_df: pd.DataFrame
    ) -> float:
        """
        Get average Elo for a league at a specific date.
        
        Args:
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff
            fixtures_df: Fixtures DataFrame
            
        Returns:
            Average Elo rating
        """
        cutoff_date = pd.to_datetime(as_of_date)
        
        # Get all teams in this league/season
        mask = (
            (fixtures_df['league_id'] == league_id) &
            (fixtures_df['season_id'] == season_id) &
            (fixtures_df['starting_at'] < cutoff_date)
        )
        
        league_fixtures = fixtures_df[mask]
        
        if len(league_fixtures) == 0:
            return self.initial_elo
        
        teams = set(league_fixtures['home_team_id'].unique()) | set(league_fixtures['away_team_id'].unique())
        
        if len(teams) == 0:
            return self.initial_elo
        
        # Get Elo for all teams
        elos = [self.get_elo_at_date(team_id, as_of_date) for team_id in teams]
        
        return sum(elos) / len(elos)
    
    def get_elo_stats(self) -> Dict:
        """
        Get Elo statistics.
        
        Returns:
            Dict with stats
        """
        if len(self.current_elo) == 0:
            return {}
        
        elos = list(self.current_elo.values())
        
        return {
            'teams_tracked': len(self.current_elo),
            'min_elo': min(elos),
            'max_elo': max(elos),
            'avg_elo': sum(elos) / len(elos),
            'total_updates': sum(len(h) for h in self.elo_history.values())
        }
