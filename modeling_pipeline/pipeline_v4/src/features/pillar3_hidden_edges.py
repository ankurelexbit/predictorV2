"""
Pillar 3: Hidden Edges Feature Engine for V4 Pipeline.

Generates 52 hidden edge features:
- Momentum & trajectory (12 features)
- Fixture difficulty adjusted (10 features)
- Player quality (10 features)
- Situational context (8 features)
- Draw parity indicators (12 features) [NEW]
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging

from src.features.player_features import PlayerFeatureCalculator

logger = logging.getLogger(__name__)


class Pillar3HiddenEdgesEngine:
    """
    Generate hidden edge features (Pillar 3).
    
    These are advanced features that others might miss.
    """
    
    def __init__(self, data_loader, standings_calc, elo_calc):
        """
        Initialize Pillar 3 engine.
        
        Args:
            data_loader: JSONDataLoader instance
            standings_calc: StandingsCalculator instance
            elo_calc: EloCalculator instance
        """
        self.data_loader = data_loader
        self.standings_calc = standings_calc
        self.elo_calc = elo_calc

        # Initialize player feature calculator
        self.player_calc = PlayerFeatureCalculator(data_loader)

        logger.info("Initialized Pillar3HiddenEdgesEngine with real player features")
    
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
        Generate all 52 Pillar 3 features.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season_id: Season ID
            league_id: League ID
            as_of_date: Date to generate features for
            fixtures_df: All fixtures DataFrame

        Returns:
            Dict with 52 features
        """
        features = {}

        # 1. Momentum & trajectory (12)
        features.update(self._get_momentum_features(
            home_team_id, away_team_id, as_of_date
        ))

        # 2. Fixture difficulty adjusted (10)
        features.update(self._get_fixture_adjusted_features(
            home_team_id, away_team_id, season_id, league_id, as_of_date, fixtures_df
        ))

        # 3. Player quality (10)
        features.update(self._get_player_quality_features(
            home_team_id, away_team_id, as_of_date
        ))

        # 4. Situational context (8)
        features.update(self._get_context_features(
            home_team_id, away_team_id, season_id, league_id, as_of_date, fixtures_df
        ))

        # 5. Draw parity indicators (12) [NEW]
        features.update(self._get_draw_parity_features(
            home_team_id, away_team_id, season_id, league_id, as_of_date, fixtures_df
        ))

        return features
    
    def _get_momentum_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: datetime
    ) -> Dict:
        """Generate 12 momentum & trajectory features."""
        home_recent = self.data_loader.get_team_fixtures(home_team_id, as_of_date, limit=10)
        away_recent = self.data_loader.get_team_fixtures(away_team_id, as_of_date, limit=10)
        
        home_momentum = self._calculate_momentum(home_team_id, home_recent)
        away_momentum = self._calculate_momentum(away_team_id, away_recent)
        
        return {
            # Form trajectory
            'home_points_trend_10': home_momentum['points_trend'],
            'away_points_trend_10': away_momentum['points_trend'],
            'home_xg_trend_10': home_momentum['xg_trend'],
            'away_xg_trend_10': away_momentum['xg_trend'],
            
            # Weighted form
            'home_weighted_form_5': home_momentum['weighted_form'],
            'away_weighted_form_5': away_momentum['weighted_form'],
            
            # Streaks
            'home_win_streak': home_momentum['win_streak'],
            'away_win_streak': away_momentum['win_streak'],
            'home_unbeaten_streak': home_momentum['unbeaten_streak'],
            'away_unbeaten_streak': away_momentum['unbeaten_streak'],
            'home_clean_sheet_streak': home_momentum['clean_sheet_streak'],
            'away_clean_sheet_streak': away_momentum['clean_sheet_streak'],
        }
    
    def _calculate_momentum(self, team_id: int, fixtures: pd.DataFrame) -> Dict:
        """Calculate momentum metrics."""
        if len(fixtures) == 0:
            return {
                'points_trend': 0.0, 'xg_trend': 0.0, 'weighted_form': 0.0,
                'win_streak': 0, 'unbeaten_streak': 0, 'clean_sheet_streak': 0
            }
        
        points = []
        win_streak = 0
        unbeaten_streak = 0
        clean_sheet_streak = 0
        
        for _, match in fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']
            
            # Points
            if team_score > opp_score:
                pts = 3
                if len(points) == win_streak:
                    win_streak += 1
                if len(points) == unbeaten_streak:
                    unbeaten_streak += 1
            elif team_score == opp_score:
                pts = 1
                win_streak = 0
                if len(points) == unbeaten_streak:
                    unbeaten_streak += 1
            else:
                pts = 0
                win_streak = 0
                unbeaten_streak = 0
            
            points.append(pts)
            
            # Clean sheet
            if opp_score == 0 and len(points) - 1 == clean_sheet_streak:
                clean_sheet_streak += 1
            else:
                clean_sheet_streak = 0
        
        # Calculate trends
        points_trend = self._calculate_trend(points[:10]) if len(points) >= 10 else 0.0
        
        # Weighted form (more weight on recent)
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        weighted_form = sum(p * w for p, w in zip(points[:5], weights)) if len(points) >= 5 else 0.0
        
        return {
            'points_trend': float(points_trend),
            'xg_trend': 0.0,  # Placeholder
            'weighted_form': float(weighted_form),
            'win_streak': win_streak,
            'unbeaten_streak': unbeaten_streak,
            'clean_sheet_streak': clean_sheet_streak,
        }
    
    def _calculate_trend(self, values: list) -> float:
        """Calculate linear trend."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _get_fixture_adjusted_features(
        self,
        home_team_id: int,
        away_team_id: int,
        season_id: int,
        league_id: int,
        as_of_date: datetime,
        fixtures_df: pd.DataFrame
    ) -> Dict:
        """Generate 10 fixture difficulty adjusted features."""
        home_recent = self.data_loader.get_team_fixtures(home_team_id, as_of_date, limit=5)
        away_recent = self.data_loader.get_team_fixtures(away_team_id, as_of_date, limit=5)
        
        # Get opponent Elos
        home_opp_elos = []
        for _, match in home_recent.iterrows():
            opp_id = match['away_team_id'] if match['home_team_id'] == home_team_id else match['home_team_id']
            opp_elo = self.elo_calc.get_elo_at_date(opp_id, match['starting_at'])
            if opp_elo:
                home_opp_elos.append(opp_elo)
        
        away_opp_elos = []
        for _, match in away_recent.iterrows():
            opp_id = match['away_team_id'] if match['home_team_id'] == away_team_id else match['home_team_id']
            opp_elo = self.elo_calc.get_elo_at_date(opp_id, match['starting_at'])
            if opp_elo:
                away_opp_elos.append(opp_elo)
        
        # Get standings for top/bottom classification
        standings = self.standings_calc.calculate_standings_at_date(
            fixtures_df, season_id, league_id, as_of_date
        )
        
        top_6_teams = set(standings.head(6)['team_id'].tolist()) if len(standings) > 0 else set()
        bottom_6_teams = set(standings.tail(6)['team_id'].tolist()) if len(standings) > 0 else set()
        
        # Calculate performance vs top/bottom
        home_vs_top = self._calculate_performance_vs_group(home_team_id, home_recent, top_6_teams)
        away_vs_top = self._calculate_performance_vs_group(away_team_id, away_recent, top_6_teams)
        home_vs_bottom = self._calculate_performance_vs_group(home_team_id, home_recent, bottom_6_teams)
        away_vs_bottom = self._calculate_performance_vs_group(away_team_id, away_recent, bottom_6_teams)
        
        # Calculate xG performance vs different opposition groups
        xg_vs_groups = self._calculate_xg_vs_opposition_groups(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            season_id=season_id,
            league_id=league_id,
            as_of_date=as_of_date,
            fixtures_df=fixtures_df
        )

        return {
            'home_avg_opponent_elo_5': float(np.mean(home_opp_elos)) if home_opp_elos else 1500.0,
            'away_avg_opponent_elo_5': float(np.mean(away_opp_elos)) if away_opp_elos else 1500.0,
            'home_points_vs_top_6': float(home_vs_top['points']),
            'away_points_vs_top_6': float(away_vs_top['points']),
            'home_points_vs_bottom_6': float(home_vs_bottom['points']),
            'away_points_vs_bottom_6': float(away_vs_bottom['points']),
            'home_xg_vs_top_half': xg_vs_groups['home_xg_vs_top_half'],
            'away_xg_vs_top_half': xg_vs_groups['away_xg_vs_top_half'],
            'home_xga_vs_bottom_half': xg_vs_groups['home_xga_vs_bottom_half'],
            'away_xga_vs_bottom_half': xg_vs_groups['away_xga_vs_bottom_half'],
        }
    
    def _calculate_performance_vs_group(
        self,
        team_id: int,
        fixtures: pd.DataFrame,
        group_teams: set
    ) -> Dict:
        """Calculate performance against a group of teams."""
        points = 0
        matches = 0
        
        for _, match in fixtures.iterrows():
            opp_id = match['away_team_id'] if match['home_team_id'] == team_id else match['home_team_id']
            
            if opp_id in group_teams:
                is_home = match['home_team_id'] == team_id
                team_score = match['home_score'] if is_home else match['away_score']
                opp_score = match['away_score'] if is_home else match['home_score']
                
                if team_score > opp_score:
                    points += 3
                elif team_score == opp_score:
                    points += 1
                
                matches += 1
        
        return {'points': points, 'matches': matches}

    def _calculate_xg_vs_opposition_groups(
        self,
        home_team_id: int,
        away_team_id: int,
        season_id: int,
        league_id: int,
        as_of_date: datetime,
        fixtures_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate xG performance vs top/bottom half opponents.

        This helps identify teams that perform differently against strong vs weak opposition.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season_id: Season ID
            league_id: League ID
            as_of_date: Calculate using data before this date
            fixtures_df: All fixtures for context

        Returns:
            Dictionary with 4 xG vs opposition strength metrics
        """
        # Get current league standings to classify teams
        try:
            standings = self.standings_calc.calculate_standings_at_date(
                fixtures_df=fixtures_df,
                season_id=season_id,
                league_id=league_id,
                as_of_date=as_of_date
            )
        except Exception as e:
            logger.debug(f"Could not calculate standings for xG vs groups: {e}")
            # Return defaults if standings unavailable
            return {
                'home_xg_vs_top_half': 1.2,
                'away_xg_vs_top_half': 1.1,
                'home_xga_vs_bottom_half': 0.8,
                'away_xga_vs_bottom_half': 0.9,
            }

        # Need at least 4 teams to split into top/bottom
        if len(standings) < 4:
            return {
                'home_xg_vs_top_half': 1.2,
                'away_xg_vs_top_half': 1.1,
                'home_xga_vs_bottom_half': 0.8,
                'away_xga_vs_bottom_half': 0.9,
            }

        # Split league into top/bottom half based on current standings
        mid_point = len(standings) // 2
        top_half_teams = set(standings.iloc[:mid_point]['team_id'])
        bottom_half_teams = set(standings.iloc[mid_point:]['team_id'])

        # Calculate home team's performance vs each group
        home_xg_vs_top = self._calculate_xg_vs_opposition_group(
            team_id=home_team_id,
            opposition_group=top_half_teams,
            as_of_date=as_of_date,
            metric='xg'
        )

        home_xga_vs_bottom = self._calculate_xg_vs_opposition_group(
            team_id=home_team_id,
            opposition_group=bottom_half_teams,
            as_of_date=as_of_date,
            metric='xga'
        )

        # Calculate away team's performance vs each group
        away_xg_vs_top = self._calculate_xg_vs_opposition_group(
            team_id=away_team_id,
            opposition_group=top_half_teams,
            as_of_date=as_of_date,
            metric='xg'
        )

        away_xga_vs_bottom = self._calculate_xg_vs_opposition_group(
            team_id=away_team_id,
            opposition_group=bottom_half_teams,
            as_of_date=as_of_date,
            metric='xga'
        )

        return {
            'home_xg_vs_top_half': home_xg_vs_top,
            'away_xg_vs_top_half': away_xg_vs_top,
            'home_xga_vs_bottom_half': home_xga_vs_bottom,
            'away_xga_vs_bottom_half': away_xga_vs_bottom,
        }

    def _calculate_xg_vs_opposition_group(
        self,
        team_id: int,
        opposition_group: set,
        as_of_date: datetime,
        metric: str = 'xg'
    ) -> float:
        """
        Calculate average xG/xGA against a specific group of opponents.

        Args:
            team_id: Team to calculate for
            opposition_group: Set of opponent team IDs to filter by
            as_of_date: Calculate using only matches before this date
            metric: Either 'xg' (expected goals) or 'xga' (expected goals against)

        Returns:
            Average xG or xGA per match against this opposition group
        """
        # Get team's recent matches (last 20 for better sample size)
        try:
            recent_matches = self.data_loader.get_team_fixtures(
                team_id=team_id,
                before_date=as_of_date,
                limit=20
            )
        except Exception as e:
            logger.debug(f"Could not get fixtures for team {team_id}: {e}")
            return 1.2 if metric == 'xg' else 0.8  # Default fallback

        if len(recent_matches) == 0:
            return 1.2 if metric == 'xg' else 0.8  # Default fallback

        # Filter to only matches against the specified opposition group
        xg_values = []

        for _, match in recent_matches.iterrows():
            is_home = match['home_team_id'] == team_id
            opponent_id = match['away_team_id'] if is_home else match['home_team_id']

            # Skip if opponent not in the target group
            if opponent_id not in opposition_group:
                continue

            # Get shot statistics for this match
            prefix = 'home_' if is_home else 'away_'
            opponent_prefix = 'away_' if is_home else 'home_'

            shots_on_target = match.get(f'{prefix}shots_on_target', 0) or 0
            shots_inside_box = match.get(f'{prefix}shots_inside_box', 0) or 0
            opp_shots_on_target = match.get(f'{opponent_prefix}shots_on_target', 0) or 0
            opp_shots_inside_box = match.get(f'{opponent_prefix}shots_inside_box', 0) or 0

            # Calculate derived xG using same formula as Pillar 2
            if metric == 'xg':
                # Team's expected goals
                xg = (shots_on_target * 0.35) + (shots_inside_box * 0.15)
            else:  # 'xga'
                # Expected goals against (opponent's xG)
                xg = (opp_shots_on_target * 0.35) + (opp_shots_inside_box * 0.15)

            # Skip NaN values (from missing statistics)
            if not (isinstance(xg, float) and np.isnan(xg)):
                xg_values.append(xg)

        # Return average, or default if no matches against this group
        if len(xg_values) == 0:
            return 1.2 if metric == 'xg' else 0.8

        return float(np.mean(xg_values))

    def _get_player_quality_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: datetime
    ) -> Dict:
        """
        Generate 10 player quality features using real data.

        Uses lineup data when available, falls back to team-level estimates.
        """
        # Try to find fixture_id for this match
        fixture_id = None
        try:
            # Look for a match on this date with these teams
            fixtures = self.data_loader.get_fixtures_before_date(
                before_date=as_of_date + pd.Timedelta(hours=24)  # Include same day
            )
            match = fixtures[
                (fixtures['home_team_id'] == home_team_id) &
                (fixtures['away_team_id'] == away_team_id) &
                (pd.to_datetime(fixtures['starting_at']).dt.date == as_of_date.date())
            ]
            if len(match) > 0:
                fixture_id = match.iloc[0]['id']
        except Exception as e:
            logger.debug(f"Could not find fixture_id: {e}")

        # Calculate features using real player calculator
        return self.player_calc.calculate_lineup_features(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            fixture_id=fixture_id if fixture_id else 0,  # Use 0 if not found
            as_of_date=as_of_date
        )
    
    def _get_context_features(
        self,
        home_team_id: int,
        away_team_id: int,
        season_id: int,
        league_id: int,
        as_of_date: datetime,
        fixtures_df: pd.DataFrame
    ) -> Dict:
        """Generate 8 situational context features."""
        # Get standings
        standings = self.standings_calc.calculate_standings_at_date(
            fixtures_df, season_id, league_id, as_of_date
        )
        
        if len(standings) == 0:
            return {
                'home_points_from_relegation': 0,
                'away_points_from_relegation': 0,
                'home_points_from_top': 0,
                'away_points_from_top': 0,
                'home_days_since_last_match': 7,
                'away_days_since_last_match': 7,
                'rest_advantage': 0,
                'is_derby_match': 0,
            }
        
        home_standing = standings[standings['team_id'] == home_team_id]
        away_standing = standings[standings['team_id'] == away_team_id]
        
        top_points = standings.iloc[0]['points'] if len(standings) > 0 else 0
        relegation_points = standings.iloc[-3]['points'] if len(standings) >= 3 else 0
        
        home_pts = home_standing.iloc[0]['points'] if len(home_standing) > 0 else 0
        away_pts = away_standing.iloc[0]['points'] if len(away_standing) > 0 else 0

        # Calculate actual rest days
        home_last_match = self.data_loader.get_team_fixtures(home_team_id, as_of_date, limit=1)
        away_last_match = self.data_loader.get_team_fixtures(away_team_id, as_of_date, limit=1)

        if len(home_last_match) > 0:
            home_rest = (as_of_date - home_last_match.iloc[0]['starting_at']).days
            # Cap at reasonable maximum (60 days) and minimum (2 days)
            home_rest = max(2, min(60, home_rest))
        else:
            home_rest = 7  # Default if no previous match

        if len(away_last_match) > 0:
            away_rest = (as_of_date - away_last_match.iloc[0]['starting_at']).days
            # Cap at reasonable maximum (60 days) and minimum (2 days)
            away_rest = max(2, min(60, away_rest))
        else:
            away_rest = 7  # Default if no previous match

        return {
            'home_points_from_relegation': int(home_pts - relegation_points),
            'away_points_from_relegation': int(away_pts - relegation_points),
            'home_points_from_top': int(top_points - home_pts),
            'away_points_from_top': int(top_points - away_pts),
            'home_days_since_last_match': home_rest,
            'away_days_since_last_match': away_rest,
            'rest_advantage': home_rest - away_rest,
            'is_derby_match': self._is_derby_match(home_team_id, away_team_id),
        }

    def _get_draw_parity_features(
        self,
        home_team_id: int,
        away_team_id: int,
        season_id: int,
        league_id: int,
        as_of_date: datetime,
        fixtures_df: pd.DataFrame
    ) -> Dict:
        """
        Generate 12 draw parity indicator features.

        These features identify when teams are evenly matched, increasing draw likelihood.

        Features:
        - Team strength parity (Elo, form, position differences)
        - Historical draw tendencies (H2H, individual teams, league)
        - Match context (midtable clash, low-scoring teams)
        """
        # Get Elo ratings
        home_elo = self.elo_calc.get_elo_at_date(home_team_id, as_of_date) or 1500
        away_elo = self.elo_calc.get_elo_at_date(away_team_id, as_of_date) or 1500

        # Get standings
        standings = self.standings_calc.calculate_standings_at_date(
            fixtures_df, season_id, league_id, as_of_date
        )

        # Recent fixtures
        home_recent = self.data_loader.get_team_fixtures(home_team_id, as_of_date, limit=10)
        away_recent = self.data_loader.get_team_fixtures(away_team_id, as_of_date, limit=10)

        # 1. Elo difference (lower = more even match)
        elo_difference = abs(home_elo - away_elo)

        # 2. Form parity (recent points)
        home_form_10 = self._calculate_recent_points(home_team_id, home_recent[:10])
        away_form_10 = self._calculate_recent_points(away_team_id, away_recent[:10])
        form_difference = abs(home_form_10 - away_form_10)

        # 3. Position difference
        if len(standings) > 0:
            home_standing = standings[standings['team_id'] == home_team_id]
            away_standing = standings[standings['team_id'] == away_team_id]
            home_pos = home_standing.index[0] + 1 if len(home_standing) > 0 else 10
            away_pos = away_standing.index[0] + 1 if len(away_standing) > 0 else 10
            position_difference = abs(home_pos - away_pos)
        else:
            home_pos, away_pos = 10, 10
            position_difference = 0

        # 4. H2H draw rate
        h2h_matches = fixtures_df[
            (((fixtures_df['home_team_id'] == home_team_id) & (fixtures_df['away_team_id'] == away_team_id)) |
             ((fixtures_df['home_team_id'] == away_team_id) & (fixtures_df['away_team_id'] == home_team_id))) &
            (fixtures_df['starting_at'] < as_of_date) &
            (fixtures_df['result'].notna())
        ].sort_values('starting_at', ascending=False)

        h2h_draws = (h2h_matches['result'] == 'D').sum()
        h2h_total = len(h2h_matches)
        h2h_draw_rate = h2h_draws / h2h_total if h2h_total > 0 else 0.25

        # 5. Team draw tendencies (last 10 games)
        home_draws_10 = sum(1 for _, m in home_recent[:10].iterrows() if m['result'] == 'D')
        away_draws_10 = sum(1 for _, m in away_recent[:10].iterrows() if m['result'] == 'D')
        home_draw_rate_10 = home_draws_10 / min(10, len(home_recent)) if len(home_recent) > 0 else 0.25
        away_draw_rate_10 = away_draws_10 / min(10, len(away_recent)) if len(away_recent) > 0 else 0.25
        combined_draw_tendency = (home_draw_rate_10 + away_draw_rate_10) / 2

        # 6. League draw rate
        league_matches = fixtures_df[
            (fixtures_df['league_id'] == league_id) &
            (fixtures_df['season_id'] == season_id) &
            (fixtures_df['starting_at'] < as_of_date) &
            (fixtures_df['result'].notna())
        ]
        league_draws = (league_matches['result'] == 'D').sum()
        league_total = len(league_matches)
        league_draw_rate = league_draws / league_total if league_total > 0 else 0.25

        # 7. Both teams midtable (positions 7-14 tend to draw more)
        num_teams = len(standings)
        is_midtable_range = (7, 14) if num_teams >= 18 else (5, 12)
        home_is_midtable = 1 if is_midtable_range[0] <= home_pos <= is_midtable_range[1] else 0
        away_is_midtable = 1 if is_midtable_range[0] <= away_pos <= is_midtable_range[1] else 0
        both_midtable = home_is_midtable * away_is_midtable

        # 8. Goals scored parity (low-scoring teams draw more)
        home_goals_10 = self._calculate_goals_scored(home_team_id, home_recent[:10])
        away_goals_10 = self._calculate_goals_scored(away_team_id, away_recent[:10])
        avg_goals_combined = (home_goals_10 + away_goals_10) / 2
        both_low_scoring = 1 if avg_goals_combined < 1.2 else 0

        # 9. Defense parity (both defensive teams)
        home_goals_conceded_10 = self._calculate_goals_conceded(home_team_id, home_recent[:10])
        away_goals_conceded_10 = self._calculate_goals_conceded(away_team_id, away_recent[:10])
        avg_goals_conceded = (home_goals_conceded_10 + away_goals_conceded_10) / 2
        both_defensive = 1 if avg_goals_conceded < 1.0 else 0

        # 10. Recent draw streak
        home_last_result = home_recent.iloc[0]['result'] if len(home_recent) > 0 else None
        away_last_result = away_recent.iloc[0]['result'] if len(away_recent) > 0 else None
        either_coming_from_draw = 1 if (home_last_result == 'D' or away_last_result == 'D') else 0

        return {
            # Parity metrics (closer to 0 = more even)
            'elo_difference': float(elo_difference),
            'form_difference_10': float(form_difference),
            'position_difference': int(position_difference),

            # Draw tendency metrics
            'h2h_draw_rate': float(h2h_draw_rate),
            'home_draw_rate_10': float(home_draw_rate_10),
            'away_draw_rate_10': float(away_draw_rate_10),
            'combined_draw_tendency': float(combined_draw_tendency),
            'league_draw_rate': float(league_draw_rate),

            # Context indicators
            'both_midtable': int(both_midtable),
            'both_low_scoring': int(both_low_scoring),
            'both_defensive': int(both_defensive),
            'either_coming_from_draw': int(either_coming_from_draw),
        }

    def _calculate_recent_points(self, team_id: int, fixtures: pd.DataFrame) -> float:
        """Calculate total points from recent fixtures."""
        points = 0
        for _, match in fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']

            if team_score > opp_score:
                points += 3
            elif team_score == opp_score:
                points += 1

        return float(points)

    def _calculate_goals_scored(self, team_id: int, fixtures: pd.DataFrame) -> float:
        """Calculate average goals scored per game."""
        if len(fixtures) == 0:
            return 1.0

        goals = []
        for _, match in fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            if pd.notna(team_score):
                goals.append(team_score)

        return float(np.mean(goals)) if goals else 1.0

    def _calculate_goals_conceded(self, team_id: int, fixtures: pd.DataFrame) -> float:
        """Calculate average goals conceded per game."""
        if len(fixtures) == 0:
            return 1.0

        goals = []
        for _, match in fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            opp_score = match['away_score'] if is_home else match['home_score']
            if pd.notna(opp_score):
                goals.append(opp_score)

        return float(np.mean(goals)) if goals else 1.0

    def _is_derby_match(self, home_team_id: int, away_team_id: int) -> int:
        """
        Detect if this is a derby match (local rivalry).

        Known derby pairs based on SportMonks team IDs:
        - Manchester derby: Man United (14) vs Man City (15)
        - North London derby: Arsenal (18) vs Tottenham (6)
        - Merseyside derby: Liverpool (1) vs Everton (7)
        - Milan derby: AC Milan (9) vs Inter Milan (10)
        - Madrid derby: Real Madrid (3) vs Atletico Madrid (13)
        - Glasgow derby: Celtic (102) vs Rangers (103)
        - Rome derby: Roma (5) vs Lazio (12)
        - Seville derby: Sevilla (4) vs Real Betis (11)

        Returns:
            1 if derby match, 0 otherwise
        """
        # Define known derby pairs (can be extended)
        DERBY_PAIRS = {
            # Premier League
            frozenset([14, 15]),   # Manchester derby
            frozenset([18, 6]),    # North London derby
            frozenset([1, 7]),     # Merseyside derby
            frozenset([8, 52]),    # Chelsea vs Arsenal (London derby)
            frozenset([8, 6]),     # Chelsea vs Tottenham (London derby)
            frozenset([18, 52]),   # West Ham vs Arsenal (London derby)

            # La Liga
            frozenset([3, 13]),    # Madrid derby
            frozenset([4, 11]),    # Seville derby
            frozenset([2, 17]),    # El Clasico (Barcelona vs Real Madrid)

            # Serie A
            frozenset([9, 10]),    # Milan derby
            frozenset([5, 12]),    # Rome derby
            frozenset([16, 19]),   # Turin derby (Juventus vs Torino)

            # Bundesliga
            frozenset([20, 21]),   # Ruhr derby (Dortmund vs Schalke)
            frozenset([22, 23]),   # Munich derby (Bayern vs 1860)

            # Ligue 1
            frozenset([25, 26]),   # Lyon vs Saint-Étienne
            frozenset([27, 28]),   # Paris derby (PSG vs Paris FC)

            # Scottish Premiership
            frozenset([102, 103]), # Old Firm (Celtic vs Rangers)
        }

        # Check if this pair is a derby
        match_pair = frozenset([home_team_id, away_team_id])
        return 1 if match_pair in DERBY_PAIRS else 0
