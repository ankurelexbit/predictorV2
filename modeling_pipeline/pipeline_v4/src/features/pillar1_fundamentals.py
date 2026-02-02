"""
Pillar 1: Fundamentals Feature Engine for V4 Pipeline.

Generates 50 fundamental features:
- Elo ratings (10 features)
- League position & points (12 features)
- Recent form (15 features)
- Head-to-head (8 features)
- Home advantage (5 features)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Pillar1FundamentalsEngine:
    """
    Generate fundamental features (Pillar 1).
    
    These are time-tested features that have always worked.
    """
    
    def __init__(self, data_loader, standings_calc, elo_calc):
        """
        Initialize Pillar 1 engine.
        
        Args:
            data_loader: JSONDataLoader instance
            standings_calc: StandingsCalculator instance
            elo_calc: EloCalculator instance
        """
        self.data_loader = data_loader
        self.standings_calc = standings_calc
        self.elo_calc = elo_calc
        
        logger.info("Initialized Pillar1FundamentalsEngine")
    
    def generate_features(
        self,
        home_team_id: int,
        away_team_id: int,
        season_id: int,
        league_id: int,
        as_of_date: datetime,
        fixtures_df: pd.DataFrame
    ) -> Dict:
        """
        Generate all 50 Pillar 1 features.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season_id: Season ID
            league_id: League ID
            as_of_date: Date to generate features for
            fixtures_df: All fixtures DataFrame
            
        Returns:
            Dict with 50 features
        """
        features = {}
        
        # 1. Elo features (10)
        features.update(self._get_elo_features(
            home_team_id, away_team_id, as_of_date, fixtures_df
        ))
        
        # 2. League position & points (12)
        features.update(self.standings_calc.get_standing_features(
            home_team_id, away_team_id, fixtures_df, season_id, league_id, as_of_date
        ))
        
        # 3. Recent form (15)
        features.update(self._get_form_features(
            home_team_id, away_team_id, as_of_date
        ))
        
        # 4. Head-to-head (8)
        features.update(self._get_h2h_features(
            home_team_id, away_team_id, as_of_date
        ))
        
        # 5. Home advantage (5)
        features.update(self._get_home_advantage_features(
            home_team_id, away_team_id, season_id, league_id, as_of_date
        ))
        
        return features
    
    def _get_elo_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: datetime,
        fixtures_df: pd.DataFrame
    ) -> Dict:
        """Generate 10 Elo features."""
        # Get current Elos
        home_elo = self.elo_calc.get_elo_at_date(home_team_id, as_of_date)
        away_elo = self.elo_calc.get_elo_at_date(away_team_id, as_of_date)
        
        # Calculate league average Elo
        league_avg_elo = 1500  # Standard starting Elo
        
        # Get Elo changes
        home_elo_change_5 = self.elo_calc.get_elo_change(home_team_id, as_of_date, 5) or 0
        away_elo_change_5 = self.elo_calc.get_elo_change(away_team_id, as_of_date, 5) or 0
        home_elo_change_10 = self.elo_calc.get_elo_change(home_team_id, as_of_date, 10) or 0
        away_elo_change_10 = self.elo_calc.get_elo_change(away_team_id, as_of_date, 10) or 0
        
        return {
            # Core Elo
            'home_elo': float(home_elo or 1500),
            'away_elo': float(away_elo or 1500),
            'elo_diff': float((home_elo or 1500) - (away_elo or 1500)),
            'elo_diff_with_home_advantage': float((home_elo or 1500) - (away_elo or 1500) + self.elo_calc.home_advantage),
            
            # Elo momentum
            'home_elo_change_5': float(home_elo_change_5),
            'away_elo_change_5': float(away_elo_change_5),
            'home_elo_change_10': float(home_elo_change_10),
            'away_elo_change_10': float(away_elo_change_10),
            
            # Elo context
            'home_elo_vs_league_avg': float((home_elo or 1500) - league_avg_elo),
            'away_elo_vs_league_avg': float((away_elo or 1500) - league_avg_elo),
        }
    
    def _get_form_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: datetime
    ) -> Dict:
        """Generate 15 recent form features."""
        # Get recent fixtures for both teams
        home_recent = self.data_loader.get_team_fixtures(home_team_id, as_of_date, limit=10)
        away_recent = self.data_loader.get_team_fixtures(away_team_id, as_of_date, limit=10)
        
        # Calculate form for home team
        home_form = self._calculate_team_form(home_team_id, home_recent)
        away_form = self._calculate_team_form(away_team_id, away_recent)
        
        return {
            # Points form
            'home_points_last_3': home_form['points_3'],
            'away_points_last_3': away_form['points_3'],
            'home_points_last_5': home_form['points_5'],
            'away_points_last_5': away_form['points_5'],
            'home_points_last_10': home_form['points_10'],
            'away_points_last_10': away_form['points_10'],
            
            # Win/Draw counts
            'home_wins_last_5': home_form['wins_5'],
            'away_wins_last_5': away_form['wins_5'],
            'home_draws_last_5': home_form['draws_5'],
            'away_draws_last_5': away_form['draws_5'],
            
            # Goal form
            'home_goals_scored_last_5': home_form['goals_scored_5'],
            'away_goals_scored_last_5': away_form['goals_scored_5'],
            'home_goals_conceded_last_5': home_form['goals_conceded_5'],
            'away_goals_conceded_last_5': away_form['goals_conceded_5'],
            'home_goal_diff_last_5': home_form['goal_diff_5'],
        }
    
    def _calculate_team_form(self, team_id: int, fixtures: pd.DataFrame) -> Dict:
        """Calculate form metrics for a team."""
        if len(fixtures) == 0:
            return {
                'points_3': 0, 'points_5': 0, 'points_10': 0,
                'wins_5': 0, 'draws_5': 0,
                'goals_scored_5': 0, 'goals_conceded_5': 0, 'goal_diff_5': 0
            }
        
        points = []
        wins = []
        draws = []
        goals_scored = []
        goals_conceded = []
        
        for _, match in fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            
            if is_home:
                team_score = match['home_score']
                opp_score = match['away_score']
            else:
                team_score = match['away_score']
                opp_score = match['home_score']
            
            # Points
            if team_score > opp_score:
                points.append(3)
                wins.append(1)
                draws.append(0)
            elif team_score == opp_score:
                points.append(1)
                wins.append(0)
                draws.append(1)
            else:
                points.append(0)
                wins.append(0)
                draws.append(0)
            
            goals_scored.append(team_score)
            goals_conceded.append(opp_score)
        
        return {
            'points_3': sum(points[:3]),
            'points_5': sum(points[:5]),
            'points_10': sum(points[:10]),
            'wins_5': sum(wins[:5]),
            'draws_5': sum(draws[:5]),
            'goals_scored_5': sum(goals_scored[:5]),
            'goals_conceded_5': sum(goals_conceded[:5]),
            'goal_diff_5': sum(goals_scored[:5]) - sum(goals_conceded[:5]),
        }
    
    def _get_h2h_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: datetime
    ) -> Dict:
        """Generate 8 head-to-head features."""
        # Get all fixtures before as_of_date
        all_fixtures = self.data_loader.get_fixtures_before(as_of_date)
        
        # Filter for H2H matches
        h2h_matches = all_fixtures[
            ((all_fixtures['home_team_id'] == home_team_id) & (all_fixtures['away_team_id'] == away_team_id)) |
            ((all_fixtures['home_team_id'] == away_team_id) & (all_fixtures['away_team_id'] == home_team_id))
        ]
        
        if len(h2h_matches) == 0:
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
        
        # Last 5 H2H
        last_5 = h2h_matches.head(5)
        home_wins_5 = 0
        draws_5 = 0
        away_wins_5 = 0
        
        for _, match in last_5.iterrows():
            if match['home_team_id'] == home_team_id:
                if match['home_score'] > match['away_score']:
                    home_wins_5 += 1
                elif match['home_score'] == match['away_score']:
                    draws_5 += 1
                else:
                    away_wins_5 += 1
            else:
                if match['away_score'] > match['home_score']:
                    home_wins_5 += 1
                elif match['away_score'] == match['home_score']:
                    draws_5 += 1
                else:
                    away_wins_5 += 1
        
        # All-time H2H stats
        home_goals = []
        away_goals = []
        home_wins_all = 0
        btts_count = 0
        over_2_5_count = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team_id'] == home_team_id:
                home_goals.append(match['home_score'])
                away_goals.append(match['away_score'])
                if match['home_score'] > match['away_score']:
                    home_wins_all += 1
            else:
                home_goals.append(match['away_score'])
                away_goals.append(match['home_score'])
                if match['away_score'] > match['home_score']:
                    home_wins_all += 1
            
            # BTTS and Over 2.5
            if match['home_score'] > 0 and match['away_score'] > 0:
                btts_count += 1
            if (match['home_score'] + match['away_score']) > 2.5:
                over_2_5_count += 1
        
        total_h2h = len(h2h_matches)
        
        return {
            'h2h_home_wins_last_5': home_wins_5,
            'h2h_draws_last_5': draws_5,
            'h2h_away_wins_last_5': away_wins_5,
            'h2h_home_goals_avg': float(np.mean(home_goals)) if home_goals else 0.0,
            'h2h_away_goals_avg': float(np.mean(away_goals)) if away_goals else 0.0,
            'h2h_home_win_pct': float(home_wins_all / total_h2h) if total_h2h > 0 else 0.0,
            'h2h_btts_pct': float(btts_count / total_h2h) if total_h2h > 0 else 0.0,
            'h2h_over_2_5_pct': float(over_2_5_count / total_h2h) if total_h2h > 0 else 0.0,
        }
    
    def _get_home_advantage_features(
        self,
        home_team_id: int,
        away_team_id: int,
        season_id: int,
        league_id: int,
        as_of_date: datetime
    ) -> Dict:
        """Generate 5 home advantage features."""
        # Get season fixtures for both teams
        season_fixtures = self.data_loader.get_fixtures_before(
            as_of_date, league_id=league_id, season_id=season_id
        )
        
        # Home team's home record
        home_at_home = season_fixtures[season_fixtures['home_team_id'] == home_team_id]
        home_points_at_home = 0
        home_home_wins = 0
        
        for _, match in home_at_home.iterrows():
            if match['home_score'] > match['away_score']:
                home_points_at_home += 3
                home_home_wins += 1
            elif match['home_score'] == match['away_score']:
                home_points_at_home += 1
        
        home_home_matches = len(home_at_home)
        home_home_win_pct = home_home_wins / home_home_matches if home_home_matches > 0 else 0.0
        
        # Away team's away record
        away_at_away = season_fixtures[season_fixtures['away_team_id'] == away_team_id]
        away_points_away = 0
        away_away_wins = 0
        
        for _, match in away_at_away.iterrows():
            if match['away_score'] > match['home_score']:
                away_points_away += 3
                away_away_wins += 1
            elif match['away_score'] == match['home_score']:
                away_points_away += 1
        
        away_away_matches = len(away_at_away)
        away_away_win_pct = away_away_wins / away_away_matches if away_away_matches > 0 else 0.0
        
        # League average home PPG (approximation)
        league_avg_home_ppg = 1.5  # Typical home advantage
        home_home_ppg = home_points_at_home / home_home_matches if home_home_matches > 0 else 0.0
        
        return {
            'home_points_at_home': home_points_at_home,
            'away_points_away': away_points_away,
            'home_home_win_pct': float(home_home_win_pct),
            'away_away_win_pct': float(away_away_win_pct),
            'home_advantage_strength': float(home_home_ppg - league_avg_home_ppg),
        }
