"""
Pillar 1: Fundamentals Feature Engine

Generates 50 fundamental features:
- Elo Ratings (10 features)
- League Position & Points (12 features)
- Recent Form (15 features)
- Head-to-Head (8 features)
- Home Advantage (5 features)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Pillar1FundamentalsEngine:
    """Generate Pillar 1 fundamental features."""
    
    def __init__(self, data_loader, standings_calculator, elo_tracker):
        """
        Initialize Pillar 1 engine.
        
        Args:
            data_loader: HistoricalDataLoader instance
            standings_calculator: SeasonAwareStandingsCalculator instance
            elo_tracker: EloTracker instance
        """
        self.data_loader = data_loader
        self.standings_calc = standings_calculator
        self.elo_tracker = elo_tracker
        
        logger.info("Initialized Pillar1FundamentalsEngine")
    
    def generate_features(
        self,
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        season_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """
        Generate all Pillar 1 features for a fixture.
        
        Args:
            fixture_id: Fixture ID
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff (point-in-time)
            
        Returns:
            Dict with 50 features
        """
        features = {}
        
        # 1. Elo features (10)
        elo_features = self._get_elo_features(
            home_team_id, away_team_id, league_id, season_id, as_of_date
        )
        features.update(elo_features)
        
        # 2. League position & points features (12)
        position_features = self._get_position_features(
            home_team_id, away_team_id, league_id, season_id, as_of_date
        )
        features.update(position_features)
        
        # 3. Recent form features (15)
        form_features = self._get_form_features(
            home_team_id, away_team_id, as_of_date
        )
        features.update(form_features)
        
        # 4. Head-to-head features (8)
        h2h_features = self._get_h2h_features(
            home_team_id, away_team_id, as_of_date
        )
        features.update(h2h_features)
        
        # 5. Home advantage features (5)
        home_adv_features = self._get_home_advantage_features(
            home_team_id, away_team_id, as_of_date
        )
        features.update(home_adv_features)
        
        return features
    
    def _get_elo_features(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        season_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate Elo features (10 features)."""
        # Get current Elo
        home_elo = self.elo_tracker.get_elo_at_date(home_team_id, as_of_date)
        away_elo = self.elo_tracker.get_elo_at_date(away_team_id, as_of_date)
        
        # Elo differential
        elo_diff = home_elo - away_elo
        elo_diff_with_ha = elo_diff + self.elo_tracker.home_advantage
        
        # Elo momentum (change over last 5 and 10 matches)
        home_elo_change_5 = self.elo_tracker.get_elo_change(home_team_id, as_of_date, 5)
        away_elo_change_5 = self.elo_tracker.get_elo_change(away_team_id, as_of_date, 5)
        home_elo_change_10 = self.elo_tracker.get_elo_change(home_team_id, as_of_date, 10)
        away_elo_change_10 = self.elo_tracker.get_elo_change(away_team_id, as_of_date, 10)
        
        # Elo vs league average
        league_avg_elo = self.elo_tracker.get_league_avg_elo(
            league_id, season_id, as_of_date, self.data_loader.fixtures_df
        )
        home_elo_vs_league_avg = home_elo - league_avg_elo
        away_elo_vs_league_avg = away_elo - league_avg_elo
        
        return {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'elo_diff_with_ha': elo_diff_with_ha,
            'home_elo_change_5': home_elo_change_5,
            'away_elo_change_5': away_elo_change_5,
            'home_elo_change_10': home_elo_change_10,
            'away_elo_change_10': away_elo_change_10,
            'home_elo_vs_league_avg': home_elo_vs_league_avg,
            'away_elo_vs_league_avg': away_elo_vs_league_avg,
        }
    
    def _get_position_features(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        season_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate league position & points features (12 features)."""
        # Get standings
        home_stats = self.standings_calc.get_team_stats(
            home_team_id, league_id, season_id, as_of_date
        )
        away_stats = self.standings_calc.get_team_stats(
            away_team_id, league_id, season_id, as_of_date
        )
        
        # Default values if not found
        if home_stats is None:
            home_stats = {
                'position': 10, 'points': 0, 'matches_played': 0,
                'points_per_game': 0.0
            }
        if away_stats is None:
            away_stats = {
                'position': 10, 'points': 0, 'matches_played': 0,
                'points_per_game': 0.0
            }
        
        # Position features
        home_position = home_stats['position']
        away_position = away_stats['position']
        position_diff = home_position - away_position
        
        # Points features
        home_points = home_stats['points']
        away_points = away_stats['points']
        points_diff = home_points - away_points
        
        # Points per game
        home_ppg = home_stats['points_per_game']
        away_ppg = away_stats['points_per_game']
        
        # Top 6 / Bottom 3
        home_in_top_6 = 1.0 if home_position <= 6 else 0.0
        away_in_top_6 = 1.0 if away_position <= 6 else 0.0
        
        # Get total teams to determine bottom 3
        standings = self.standings_calc.calculate_standings_at_date(
            league_id, season_id, as_of_date
        )
        total_teams = len(standings) if len(standings) > 0 else 20
        
        home_in_bottom_3 = 1.0 if home_position >= (total_teams - 2) else 0.0
        away_in_bottom_3 = 1.0 if away_position >= (total_teams - 2) else 0.0
        
        return {
            'home_league_position': float(home_position),
            'away_league_position': float(away_position),
            'position_diff': float(position_diff),
            'home_points': float(home_points),
            'away_points': float(away_points),
            'points_diff': float(points_diff),
            'home_points_per_game': home_ppg,
            'away_points_per_game': away_ppg,
            'home_in_top_6': home_in_top_6,
            'away_in_top_6': away_in_top_6,
            'home_in_bottom_3': home_in_bottom_3,
            'away_in_bottom_3': away_in_bottom_3,
        }
    
    def _get_form_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate recent form features (15 features)."""
        # Get recent fixtures for both teams
        home_last_3 = self.data_loader.get_fixtures_before_date(home_team_id, as_of_date, n=3)
        home_last_5 = self.data_loader.get_fixtures_before_date(home_team_id, as_of_date, n=5)
        home_last_10 = self.data_loader.get_fixtures_before_date(home_team_id, as_of_date, n=10)
        
        away_last_3 = self.data_loader.get_fixtures_before_date(away_team_id, as_of_date, n=3)
        away_last_5 = self.data_loader.get_fixtures_before_date(away_team_id, as_of_date, n=5)
        away_last_10 = self.data_loader.get_fixtures_before_date(away_team_id, as_of_date, n=10)
        
        # Calculate form metrics
        home_form_3 = self._calculate_form_metrics(home_last_3, home_team_id)
        home_form_5 = self._calculate_form_metrics(home_last_5, home_team_id)
        home_form_10 = self._calculate_form_metrics(home_last_10, home_team_id)
        
        away_form_3 = self._calculate_form_metrics(away_last_3, away_team_id)
        away_form_5 = self._calculate_form_metrics(away_last_5, away_team_id)
        away_form_10 = self._calculate_form_metrics(away_last_10, away_team_id)
        
        return {
            'home_points_last_3': home_form_3['points'],
            'away_points_last_3': away_form_3['points'],
            'home_points_last_5': home_form_5['points'],
            'away_points_last_5': away_form_5['points'],
            'home_points_last_10': home_form_10['points'],
            'away_points_last_10': away_form_10['points'],
            'home_wins_last_5': home_form_5['wins'],
            'away_wins_last_5': away_form_5['wins'],
            'home_draws_last_5': home_form_5['draws'],
            'away_draws_last_5': away_form_5['draws'],
            'home_goals_scored_last_5': home_form_5['goals_for'],
            'away_goals_scored_last_5': away_form_5['goals_for'],
            'home_goals_conceded_last_5': home_form_5['goals_against'],
            'away_goals_conceded_last_5': away_form_5['goals_against'],
            'home_goal_diff_last_5': home_form_5['goal_diff'],
        }
    
    def _calculate_form_metrics(self, fixtures: pd.DataFrame, team_id: int) -> Dict:
        """Calculate form metrics from fixtures."""
        if len(fixtures) == 0:
            return {
                'points': 0.0, 'wins': 0.0, 'draws': 0.0, 'losses': 0.0,
                'goals_for': 0.0, 'goals_against': 0.0, 'goal_diff': 0.0
            }
        
        points = 0
        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0
        
        for _, match in fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            # Skip if scores missing
            if pd.isna(home_score) or pd.isna(away_score):
                continue
            
            if is_home:
                goals_for += home_score
                goals_against += away_score
                if home_score > away_score:
                    points += 3
                    wins += 1
                elif home_score == away_score:
                    points += 1
                    draws += 1
                else:
                    losses += 1
            else:
                goals_for += away_score
                goals_against += home_score
                if away_score > home_score:
                    points += 3
                    wins += 1
                elif away_score == home_score:
                    points += 1
                    draws += 1
                else:
                    losses += 1
        
        return {
            'points': float(points),
            'wins': float(wins),
            'draws': float(draws),
            'losses': float(losses),
            'goals_for': float(goals_for),
            'goals_against': float(goals_against),
            'goal_diff': float(goals_for - goals_against)
        }
    
    def _get_h2h_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate head-to-head features (8 features)."""
        # Get H2H fixtures
        h2h_last_5 = self.data_loader.get_h2h_fixtures(
            home_team_id, away_team_id, as_of_date, n=5
        )
        h2h_all = self.data_loader.get_h2h_fixtures(
            home_team_id, away_team_id, as_of_date
        )
        
        if len(h2h_last_5) == 0:
            return {
                'h2h_home_wins_last_5': 0.0,
                'h2h_draws_last_5': 0.0,
                'h2h_away_wins_last_5': 0.0,
                'h2h_home_goals_avg': 0.0,
                'h2h_away_goals_avg': 0.0,
                'h2h_home_win_pct': 0.0,
                'h2h_btts_pct': 0.0,
                'h2h_over_2_5_pct': 0.0,
            }
        
        # Calculate H2H stats
        home_wins_5 = 0
        draws_5 = 0
        away_wins_5 = 0
        
        for _, match in h2h_last_5.iterrows():
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            if pd.isna(home_score) or pd.isna(away_score):
                continue
            
            # From perspective of current home team
            if match['home_team_id'] == home_team_id:
                if home_score > away_score:
                    home_wins_5 += 1
                elif home_score == away_score:
                    draws_5 += 1
                else:
                    away_wins_5 += 1
            else:
                if away_score > home_score:
                    home_wins_5 += 1
                elif away_score == home_score:
                    draws_5 += 1
                else:
                    away_wins_5 += 1
        
        # All-time H2H stats
        total_h2h = len(h2h_all)
        home_goals_total = 0
        away_goals_total = 0
        btts_count = 0
        over_2_5_count = 0
        home_wins_all = 0
        
        for _, match in h2h_all.iterrows():
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            if pd.isna(home_score) or pd.isna(away_score):
                continue
            
            # Goals from perspective of current home team
            if match['home_team_id'] == home_team_id:
                home_goals_total += home_score
                away_goals_total += away_score
                if home_score > away_score:
                    home_wins_all += 1
            else:
                home_goals_total += away_score
                away_goals_total += home_score
                if away_score > home_score:
                    home_wins_all += 1
            
            # BTTS and Over 2.5
            if home_score > 0 and away_score > 0:
                btts_count += 1
            if (home_score + away_score) > 2.5:
                over_2_5_count += 1
        
        return {
            'h2h_home_wins_last_5': float(home_wins_5),
            'h2h_draws_last_5': float(draws_5),
            'h2h_away_wins_last_5': float(away_wins_5),
            'h2h_home_goals_avg': home_goals_total / total_h2h if total_h2h > 0 else 0.0,
            'h2h_away_goals_avg': away_goals_total / total_h2h if total_h2h > 0 else 0.0,
            'h2h_home_win_pct': home_wins_all / total_h2h if total_h2h > 0 else 0.0,
            'h2h_btts_pct': btts_count / total_h2h if total_h2h > 0 else 0.0,
            'h2h_over_2_5_pct': over_2_5_count / total_h2h if total_h2h > 0 else 0.0,
        }
    
    def _get_home_advantage_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate home advantage features (5 features)."""
        # Get home fixtures for home team
        home_at_home = self.data_loader.get_fixtures_before_date(
            home_team_id, as_of_date, n=None
        )
        home_at_home = home_at_home[home_at_home['home_team_id'] == home_team_id]
        
        # Get away fixtures for away team
        away_at_away = self.data_loader.get_fixtures_before_date(
            away_team_id, as_of_date, n=None
        )
        away_at_away = away_at_away[away_at_away['away_team_id'] == away_team_id]
        
        # Calculate home stats
        home_form = self._calculate_form_metrics(home_at_home, home_team_id)
        away_form = self._calculate_form_metrics(away_at_away, away_team_id)
        
        home_matches = len(home_at_home)
        away_matches = len(away_at_away)
        
        home_points_at_home = home_form['points']
        away_points_away = away_form['points']
        
        home_home_win_pct = home_form['wins'] / home_matches if home_matches > 0 else 0.0
        away_away_win_pct = away_form['wins'] / away_matches if away_matches > 0 else 0.0
        
        # Home advantage strength (compared to league average)
        # Simplified: just use PPG difference
        home_ppg_at_home = home_points_at_home / home_matches if home_matches > 0 else 0.0
        league_avg_home_ppg = 1.5  # Approximate league average
        home_advantage_strength = home_ppg_at_home - league_avg_home_ppg
        
        return {
            'home_points_at_home': home_points_at_home,
            'away_points_away': away_points_away,
            'home_home_win_pct': home_home_win_pct,
            'away_away_win_pct': away_away_win_pct,
            'home_advantage_strength': home_advantage_strength,
        }
