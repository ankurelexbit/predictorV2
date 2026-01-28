"""
Pillar 3: Hidden Edges Feature Engine

Generates 40 hidden edge features:
- Momentum & Trajectory (12 features)
- Fixture Difficulty Adjusted (10 features)
- Player Quality (10 features - simplified)
- Situational Context (8 features)
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats as scipy_stats
import logging

logger = logging.getLogger(__name__)


class Pillar3HiddenEdgesEngine:
    """Generate Pillar 3 hidden edge features."""
    
    def __init__(self, data_loader, elo_tracker):
        """
        Initialize Pillar 3 engine.
        
        Args:
            data_loader: HistoricalDataLoader instance
            elo_tracker: EloTracker instance
        """
        self.data_loader = data_loader
        self.elo_tracker = elo_tracker
        
        logger.info("Initialized Pillar3HiddenEdgesEngine")
    
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
        Generate all Pillar 3 features for a fixture.
        
        Args:
            fixture_id: Fixture ID
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff (point-in-time)
            
        Returns:
            Dict with 40 features
        """
        features = {}
        
        # 1. Momentum & trajectory features (12)
        momentum_features = self._get_momentum_features(
            home_team_id, away_team_id, as_of_date
        )
        features.update(momentum_features)
        
        # 2. Fixture difficulty adjusted features (10)
        fixture_adj_features = self._get_fixture_adjusted_features(
            home_team_id, away_team_id, as_of_date
        )
        features.update(fixture_adj_features)
        
        # 3. Player quality features (10 - simplified)
        player_features = self._get_player_features(
            home_team_id, away_team_id, as_of_date
        )
        features.update(player_features)
        
        # 4. Situational context features (8)
        context_features = self._get_context_features(
            home_team_id, away_team_id, league_id, season_id, as_of_date
        )
        features.update(context_features)
        
        return features
    
    def _get_momentum_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate momentum & trajectory features (12 features)."""
        home_momentum = self._calculate_team_momentum(home_team_id, as_of_date)
        away_momentum = self._calculate_team_momentum(away_team_id, as_of_date)
        
        return {
            # Points trend (linear regression slope)
            'home_points_trend_10': home_momentum['points_trend'],
            'away_points_trend_10': away_momentum['points_trend'],
            
            # Weighted form (exponential weighting)
            'home_weighted_form_5': home_momentum['weighted_form'],
            'away_weighted_form_5': away_momentum['weighted_form'],
            
            # Streaks
            'home_win_streak': home_momentum['win_streak'],
            'away_win_streak': away_momentum['win_streak'],
            'home_unbeaten_streak': home_momentum['unbeaten_streak'],
            'away_unbeaten_streak': away_momentum['unbeaten_streak'],
            'home_clean_sheet_streak': home_momentum['clean_sheet_streak'],
            'away_clean_sheet_streak': away_momentum['clean_sheet_streak'],
            
            # Goal scoring momentum
            'home_goals_trend_10': home_momentum['goals_trend'],
            'away_goals_trend_10': away_momentum['goals_trend'],
        }
    
    def _calculate_team_momentum(self, team_id: int, as_of_date: str) -> Dict:
        """Calculate momentum metrics for a team."""
        # Get last 10 matches for trends
        fixtures_10 = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=10)
        fixtures_5 = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=5)
        
        if len(fixtures_10) == 0:
            return {
                'points_trend': 0.0, 'weighted_form': 0.0,
                'win_streak': 0.0, 'unbeaten_streak': 0.0,
                'clean_sheet_streak': 0.0, 'goals_trend': 0.0
            }
        
        # Calculate points trend (linear regression)
        points_list = []
        goals_list = []
        
        for _, fixture in fixtures_10.iterrows():
            is_home = fixture['home_team_id'] == team_id
            home_score = fixture.get('home_score', 0) or 0
            away_score = fixture.get('away_score', 0) or 0
            
            if is_home:
                if home_score > away_score:
                    points = 3
                elif home_score == away_score:
                    points = 1
                else:
                    points = 0
                goals = home_score
            else:
                if away_score > home_score:
                    points = 3
                elif away_score == home_score:
                    points = 1
                else:
                    points = 0
                goals = away_score
            
            points_list.append(points)
            goals_list.append(goals)
        
        # Linear regression for trend
        if len(points_list) >= 3:
            x = np.arange(len(points_list))
            points_trend, _ = np.polyfit(x, points_list, 1)
            goals_trend, _ = np.polyfit(x, goals_list, 1)
        else:
            points_trend = 0.0
            goals_trend = 0.0
        
        # Weighted form (exponential weighting - more recent = more weight)
        if len(fixtures_5) > 0:
            weights = np.exp(np.linspace(0, 1, len(points_list[-5:])))
            weighted_points = np.array(points_list[-5:]) * weights
            weighted_form = np.sum(weighted_points) / np.sum(weights)
        else:
            weighted_form = 0.0
        
        # Calculate streaks (from most recent backwards)
        win_streak = 0
        unbeaten_streak = 0
        clean_sheet_streak = 0
        
        for _, fixture in fixtures_10.iloc[::-1].iterrows():  # Reverse order
            is_home = fixture['home_team_id'] == team_id
            home_score = fixture.get('home_score', 0) or 0
            away_score = fixture.get('away_score', 0) or 0
            
            if is_home:
                won = home_score > away_score
                not_lost = home_score >= away_score
                clean_sheet = away_score == 0
            else:
                won = away_score > home_score
                not_lost = away_score >= home_score
                clean_sheet = home_score == 0
            
            # Win streak
            if won:
                win_streak += 1
            else:
                break
        
        # Unbeaten streak
        for _, fixture in fixtures_10.iloc[::-1].iterrows():
            is_home = fixture['home_team_id'] == team_id
            home_score = fixture.get('home_score', 0) or 0
            away_score = fixture.get('away_score', 0) or 0
            
            if is_home:
                not_lost = home_score >= away_score
            else:
                not_lost = away_score >= home_score
            
            if not_lost:
                unbeaten_streak += 1
            else:
                break
        
        # Clean sheet streak
        for _, fixture in fixtures_10.iloc[::-1].iterrows():
            is_home = fixture['home_team_id'] == team_id
            home_score = fixture.get('home_score', 0) or 0
            away_score = fixture.get('away_score', 0) or 0
            
            if is_home:
                clean_sheet = away_score == 0
            else:
                clean_sheet = home_score == 0
            
            if clean_sheet:
                clean_sheet_streak += 1
            else:
                break
        
        return {
            'points_trend': float(points_trend),
            'weighted_form': float(weighted_form),
            'win_streak': float(win_streak),
            'unbeaten_streak': float(unbeaten_streak),
            'clean_sheet_streak': float(clean_sheet_streak),
            'goals_trend': float(goals_trend),
        }
    
    def _get_fixture_adjusted_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate fixture difficulty adjusted features (10 features)."""
        home_fixture_adj = self._calculate_fixture_adjusted_metrics(home_team_id, as_of_date)
        away_fixture_adj = self._calculate_fixture_adjusted_metrics(away_team_id, as_of_date)
        
        return {
            # Strength of schedule
            'home_avg_opponent_elo_5': home_fixture_adj['avg_opp_elo'],
            'away_avg_opponent_elo_5': away_fixture_adj['avg_opp_elo'],
            
            # Performance vs strength (simplified - use Elo as proxy)
            'home_points_vs_strong_5': home_fixture_adj['points_vs_strong'],
            'away_points_vs_strong_5': away_fixture_adj['points_vs_strong'],
            'home_points_vs_weak_5': home_fixture_adj['points_vs_weak'],
            'away_points_vs_weak_5': away_fixture_adj['points_vs_weak'],
            
            # Goals vs strength
            'home_goals_vs_strong_5': home_fixture_adj['goals_vs_strong'],
            'away_goals_vs_strong_5': away_fixture_adj['goals_vs_strong'],
            'home_goals_vs_weak_5': home_fixture_adj['goals_vs_weak'],
            'away_goals_vs_weak_5': away_fixture_adj['goals_vs_weak'],
        }
    
    def _calculate_fixture_adjusted_metrics(self, team_id: int, as_of_date: str) -> Dict:
        """Calculate fixture difficulty adjusted metrics."""
        fixtures = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=5)
        
        if len(fixtures) == 0:
            return {
                'avg_opp_elo': 1500.0,
                'points_vs_strong': 0.0,
                'points_vs_weak': 0.0,
                'goals_vs_strong': 0.0,
                'goals_vs_weak': 0.0,
            }
        
        opponent_elos = []
        points_vs_strong = 0
        points_vs_weak = 0
        goals_vs_strong = 0
        goals_vs_weak = 0
        strong_count = 0
        weak_count = 0
        
        for _, fixture in fixtures.iterrows():
            is_home = fixture['home_team_id'] == team_id
            opp_id = fixture['away_team_id'] if is_home else fixture['home_team_id']
            
            # Get opponent Elo at that time
            fixture_date = fixture['starting_at']
            opp_elo = self.elo_tracker.get_elo_at_date(opp_id, str(fixture_date))
            opponent_elos.append(opp_elo)
            
            # Calculate points
            home_score = fixture.get('home_score', 0) or 0
            away_score = fixture.get('away_score', 0) or 0
            
            if is_home:
                if home_score > away_score:
                    points = 3
                elif home_score == away_score:
                    points = 1
                else:
                    points = 0
                goals = home_score
            else:
                if away_score > home_score:
                    points = 3
                elif away_score == home_score:
                    points = 1
                else:
                    points = 0
                goals = away_score
            
            # Classify opponent (strong = Elo > 1550, weak = Elo < 1450)
            if opp_elo > 1550:
                points_vs_strong += points
                goals_vs_strong += goals
                strong_count += 1
            elif opp_elo < 1450:
                points_vs_weak += points
                goals_vs_weak += goals
                weak_count += 1
        
        avg_opp_elo = np.mean(opponent_elos) if len(opponent_elos) > 0 else 1500.0
        
        return {
            'avg_opp_elo': avg_opp_elo,
            'points_vs_strong': points_vs_strong / strong_count if strong_count > 0 else 0.0,
            'points_vs_weak': points_vs_weak / weak_count if weak_count > 0 else 0.0,
            'goals_vs_strong': goals_vs_strong / strong_count if strong_count > 0 else 0.0,
            'goals_vs_weak': goals_vs_weak / weak_count if weak_count > 0 else 0.0,
        }
    
    def _get_player_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate player quality features (10 features - simplified)."""
        # Note: Full player features require lineup data
        # This is a simplified version using available data
        
        home_player = self._calculate_player_metrics(home_team_id, as_of_date)
        away_player = self._calculate_player_metrics(away_team_id, as_of_date)
        
        return {
            'home_lineup_quality_proxy': home_player['quality_proxy'],
            'away_lineup_quality_proxy': away_player['quality_proxy'],
            'home_squad_depth_proxy': home_player['depth_proxy'],
            'away_squad_depth_proxy': away_player['depth_proxy'],
            'home_consistency_rating': home_player['consistency'],
            'away_consistency_rating': away_player['consistency'],
            'home_recent_performance': home_player['recent_perf'],
            'away_recent_performance': away_player['recent_perf'],
            'home_goal_threat': home_player['goal_threat'],
            'away_goal_threat': away_player['goal_threat'],
        }
    
    def _calculate_player_metrics(self, team_id: int, as_of_date: str) -> Dict:
        """Calculate simplified player metrics (proxy using team performance)."""
        fixtures = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=10)
        
        if len(fixtures) == 0:
            return {
                'quality_proxy': 0.0,
                'depth_proxy': 0.0,
                'consistency': 0.0,
                'recent_perf': 0.0,
                'goal_threat': 0.0,
            }
        
        # Use team Elo as quality proxy
        team_elo = self.elo_tracker.get_elo_at_date(team_id, as_of_date)
        quality_proxy = (team_elo - 1500) / 100  # Normalize
        
        # Calculate consistency (std dev of points)
        points_list = []
        goals_list = []
        
        for _, fixture in fixtures.iterrows():
            is_home = fixture['home_team_id'] == team_id
            home_score = fixture.get('home_score', 0) or 0
            away_score = fixture.get('away_score', 0) or 0
            
            if is_home:
                if home_score > away_score:
                    points = 3
                elif home_score == away_score:
                    points = 1
                else:
                    points = 0
                goals = home_score
            else:
                if away_score > home_score:
                    points = 3
                elif away_score == home_score:
                    points = 1
                else:
                    points = 0
                goals = away_score
            
            points_list.append(points)
            goals_list.append(goals)
        
        consistency = 1.0 / (1.0 + np.std(points_list)) if len(points_list) > 0 else 0.0
        recent_perf = np.mean(points_list[-5:]) if len(points_list) >= 5 else 0.0
        goal_threat = np.mean(goals_list) if len(goals_list) > 0 else 0.0
        
        # Depth proxy (based on consistency)
        depth_proxy = consistency
        
        return {
            'quality_proxy': quality_proxy,
            'depth_proxy': depth_proxy,
            'consistency': consistency,
            'recent_perf': recent_perf,
            'goal_threat': goal_threat,
        }
    
    def _get_context_features(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        season_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate situational context features (8 features)."""
        # Get days since last match
        home_days_rest = self._get_days_since_last_match(home_team_id, as_of_date)
        away_days_rest = self._get_days_since_last_match(away_team_id, as_of_date)
        
        # Rest advantage
        rest_advantage = home_days_rest - away_days_rest
        
        # Derby match (simplified - check if same league and close Elo)
        home_elo = self.elo_tracker.get_elo_at_date(home_team_id, as_of_date)
        away_elo = self.elo_tracker.get_elo_at_date(away_team_id, as_of_date)
        elo_diff = abs(home_elo - away_elo)
        is_derby = 1.0 if elo_diff < 100 else 0.0  # Simplified derby indicator
        
        return {
            'home_days_since_last_match': home_days_rest,
            'away_days_since_last_match': away_days_rest,
            'rest_advantage': rest_advantage,
            'is_derby_match': is_derby,
            'home_elo_pressure': 1.0 if home_elo > 1600 else 0.0,  # High expectations
            'away_elo_pressure': 1.0 if away_elo > 1600 else 0.0,
            'home_underdog': 1.0 if home_elo < away_elo - 100 else 0.0,
            'away_underdog': 1.0 if away_elo < home_elo - 100 else 0.0,
        }
    
    def _get_days_since_last_match(self, team_id: int, as_of_date: str) -> float:
        """Get days since team's last match."""
        fixtures = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=1)
        
        if len(fixtures) == 0:
            return 7.0  # Default
        
        last_match_date = fixtures.iloc[-1]['starting_at']
        current_date = pd.to_datetime(as_of_date)
        
        days_diff = (current_date - last_match_date).days
        
        return float(days_diff)
