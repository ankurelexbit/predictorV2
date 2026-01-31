"""
Comprehensive CSV-based Feature Engine - Complete 150-180 Feature Framework

This module implements the complete feature framework from FEATURE_FRAMEWORK.md:
- PILLAR 1: Fundamentals (50 features)
- PILLAR 2: Modern Analytics (60 features)  
- PILLAR 3: Hidden Edges (40 features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
from scipy import stats

from .csv_elo_calculator import CSVEloCalculator
from .csv_league_position import CSVLeaguePositionCalculator
from .csv_h2h import CSVH2HCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveCSVFeatureEngine:
    """
    Complete feature engineering engine implementing all 150-180 features.
    
    Features organized by 3 pillars:
    1. Fundamentals: Elo, position, form, H2H, home advantage
    2. Modern Analytics: xG, shots, defensive intensity, attack patterns
    3. Hidden Edges: Momentum, fixture-adjusted, player quality, context
    """
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize feature engine and load all data."""
        self.data_dir = Path(data_dir)
        
        logger.info("Loading CSV data...")
        self.fixtures = pd.read_csv(self.data_dir / 'fixtures.csv')
        self.statistics = pd.read_csv(self.data_dir / 'statistics.csv')
        self.lineups = pd.read_csv(self.data_dir / 'lineups.csv')
        
        # Load standings
        standings_path = self.data_dir / 'standings.csv'
        if standings_path.exists():
            self.standings = pd.read_csv(standings_path)
        else:
            logger.warning("standings.csv not found, creating empty DataFrame")
            self.standings = pd.DataFrame(columns=['fixture_id', 'team_id', 'position'])
        
        # Initialize calculators with correct parameters
        self.elo_calc = CSVEloCalculator(k_factor=32, initial_elo=1500, home_advantage=35)
        self.position_calc = CSVLeaguePositionCalculator(self.fixtures, self.standings)
        self.h2h_calc = CSVH2HCalculator(self.fixtures)
        
        # Pre-calculate expensive operations
        logger.info("Calculating Elo history...")
        self.elo_history = self.elo_calc.calculate_elo_history(self.fixtures)
        
        logger.info("Adding derived xG to statistics...")
        self.statistics = self._add_derived_xg(self.statistics)
        
        logger.info("✅ Feature engine initialized")
    
    def _add_derived_xg(self, statistics_df: pd.DataFrame) -> pd.DataFrame:
        """Add derived xG to statistics DataFrame."""
        df = statistics_df.copy()
        
        # Get shot columns (handle missing data)
        shots_on_target = df.get('shots_on_target', pd.Series([0] * len(df))).fillna(0)
        shots_off_target = df.get('shots_off_target', pd.Series([0] * len(df))).fillna(0)
        shots_blocked = df.get('shots_blocked', pd.Series([0] * len(df))).fillna(0)
        
        # Calculate xG using formula from framework
        df['xG'] = (
            shots_on_target * 0.32 +
            shots_off_target * 0.05 +
            shots_blocked * 0.03
        )
        
        return df
    
    def generate_features_for_fixture(
        self,
        fixture_id: int,
        as_of_date: str
    ) -> Dict:
        """
        Generate ALL 150-180 features for a single fixture.
        
        CRITICAL: Uses only data from BEFORE as_of_date (point-in-time correct).
        
        Args:
            fixture_id: Fixture ID
            as_of_date: Fixture date (use data BEFORE this)
        
        Returns:
            Dictionary with all features
        """
        # Get fixture details
        fixture = self.fixtures[self.fixtures['fixture_id'] == fixture_id].iloc[0]
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        league_id = fixture['league_id']
        
        features = {
            'fixture_id': fixture_id,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'starting_at': as_of_date,
            'league_id': league_id,
        }
        
        # ===== PILLAR 1: FUNDAMENTALS (50 features) =====
        
        # 1.1 Elo Features (10 features)
        features.update(self._get_elo_features(home_team_id, away_team_id, league_id, as_of_date))
        
        # 1.2 League Position (12 features) - SEASON-AWARE
        features.update(self._get_position_features(fixture_id, as_of_date))
        
        # 1.3 Recent Form (15 features)
        features.update(self._get_form_features(home_team_id, away_team_id, as_of_date))
        
        # 1.4 Head-to-Head (8 features)
        features.update(self._get_h2h_features(home_team_id, away_team_id, as_of_date))
        
        # 1.5 Home Advantage (5 features)
        features.update(self._get_home_advantage_features(home_team_id, away_team_id, as_of_date))
        
        # ===== PILLAR 2: MODERN ANALYTICS (60 features) =====
        
        # 2.1 Derived xG (25 features)
        features.update(self._get_xg_features(home_team_id, away_team_id, as_of_date))
        
        # 2.2 Shot Analysis (15 features)
        features.update(self._get_shot_features(home_team_id, away_team_id, as_of_date))
        
        # 2.3 Defensive Intensity (12 features)
        features.update(self._get_defensive_features(home_team_id, away_team_id, as_of_date))
        
        # 2.4 Attack Patterns (8 features)
        features.update(self._get_attack_features(home_team_id, away_team_id, as_of_date))
        
        # ===== PILLAR 3: HIDDEN EDGES (40 features) =====
        
        # 3.1 Momentum & Trajectory (12 features)
        features.update(self._get_momentum_features(home_team_id, away_team_id, as_of_date))
        
        # 3.2 Fixture Difficulty Adjusted (10 features)
        features.update(self._get_fixture_adjusted_features(home_team_id, away_team_id, as_of_date))
        
        # 3.3 Player Quality (10 features)
        features.update(self._get_player_features(home_team_id, away_team_id, as_of_date))
        
        # 3.3 Situational Context (4 features - simplified after season-aware fix)
        features.update(self._get_context_features(home_team_id, away_team_id, league_id, fixture_id, as_of_date))
        
        # ===== TARGET VARIABLE =====
        if 'result' in fixture and not pd.isna(fixture['result']):
            features['result'] = fixture['result']
            features['target_home_win'] = 1 if fixture['result'] == 'H' else 0
            features['target_draw'] = 1 if fixture['result'] == 'D' else 0
            features['target_away_win'] = 1 if fixture['result'] == 'A' else 0
            features['home_goals'] = fixture.get('home_score', 0)
            features['away_goals'] = fixture.get('away_score', 0)
        
        return features
    
    # ===== PILLAR 1: FUNDAMENTALS =====
    
    def _get_elo_features(self, home_team_id, away_team_id, league_id, as_of_date) -> Dict:
        """Get Elo features (10 features)."""
        home_elo = self.elo_calc.get_elo_at_date(self.elo_history, home_team_id, as_of_date)
        away_elo = self.elo_calc.get_elo_at_date(self.elo_history, away_team_id, as_of_date)
        league_avg_elo = self.elo_calc.get_league_average_elo(self.elo_history, league_id, as_of_date)
        
        return {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            'elo_diff_with_home_advantage': (home_elo - away_elo) + 50,
            'home_elo_change_5': self.elo_calc.get_elo_change(self.elo_history, home_team_id, as_of_date, 5),
            'away_elo_change_5': self.elo_calc.get_elo_change(self.elo_history, away_team_id, as_of_date, 5),
            'home_elo_change_10': self.elo_calc.get_elo_change(self.elo_history, home_team_id, as_of_date, 10),
            'away_elo_change_10': self.elo_calc.get_elo_change(self.elo_history, away_team_id, as_of_date, 10),
            'home_elo_vs_league_avg': home_elo - league_avg_elo,
            'away_elo_vs_league_avg': away_elo - league_avg_elo,
        }
    
    def _get_position_features(self, fixture_id: int, as_of_date: str) -> Dict:
        """Get league position features (season-aware)."""
        return self.position_calc.get_position_features(fixture_id, as_of_date)
    
    def _get_form_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get form features (15 features)."""
        features = {}
        
        # Multiple form windows (3, 5, 10 matches)
        for n in [3, 5, 10]:
            home_form = self._calculate_form(home_team_id, as_of_date, n)
            away_form = self._calculate_form(away_team_id, as_of_date, n)
            
            features[f'home_points_last_{n}'] = home_form['points']
            features[f'away_points_last_{n}'] = away_form['points']
            
            if n == 5:  # Only for 5 matches
                features['home_wins_last_5'] = home_form['wins']
                features['away_wins_last_5'] = away_form['wins']
                features['home_draws_last_5'] = home_form['draws']
                features['away_draws_last_5'] = away_form['draws']
                features['home_goals_scored_last_5'] = home_form['goals_scored']
                features['away_goals_scored_last_5'] = away_form['goals_scored']
                features['home_goals_conceded_last_5'] = home_form['goals_conceded']
                features['away_goals_conceded_last_5'] = away_form['goals_conceded']
                features['home_goal_diff_last_5'] = home_form['goal_difference']
        
        return features
    
    def _get_h2h_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get head-to-head features (8 features)."""
        return self.h2h_calc.get_h2h_features(
            home_team_id, away_team_id, as_of_date, n_matches=5
        )
    
    def _get_home_advantage_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get home advantage features (5 features)."""
        home_at_home = self._calculate_home_away_split(home_team_id, as_of_date, is_home=True)
        away_away = self._calculate_home_away_split(away_team_id, as_of_date, is_home=False)
        
        return {
            'home_points_at_home': home_at_home['points'],
            'away_points_away': away_away['points'],
            'home_home_win_pct': home_at_home['win_pct'],
            'away_away_win_pct': away_away['win_pct'],
            'home_advantage_strength': home_at_home['ppg'] - 1.5,  # vs league avg ~1.5
        }
    
    # ===== PILLAR 2: MODERN ANALYTICS =====
    
    def _get_xg_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get derived xG features (25 features)."""
        home_xg = self._calculate_xg_metrics(home_team_id, as_of_date, n_matches=5)
        away_xg = self._calculate_xg_metrics(away_team_id, as_of_date, n_matches=5)
        home_xg_10 = self._calculate_xg_metrics(home_team_id, as_of_date, n_matches=10)
        away_xg_10 = self._calculate_xg_metrics(away_team_id, as_of_date, n_matches=10)
        
        return {
            # Core xG (last 5)
            'home_derived_xg_per_match_5': home_xg['xg_per_match'],
            'away_derived_xg_per_match_5': away_xg['xg_per_match'],
            'home_derived_xga_per_match_5': home_xg['xga_per_match'],
            'away_derived_xga_per_match_5': away_xg['xga_per_match'],
            'home_derived_xgd_5': home_xg['xgd'],
            'away_derived_xgd_5': away_xg['xgd'],
            'derived_xgd_matchup': home_xg['xgd'] - away_xg['xgd'],
            
            # Performance vs expectation
            'home_goals_vs_xg_5': home_xg['goals_vs_xg'],
            'away_goals_vs_xg_5': away_xg['goals_vs_xg'],
            'home_ga_vs_xga_5': home_xg['ga_vs_xga'],
            'away_ga_vs_xga_5': away_xg['ga_vs_xga'],
            
            # Shot quality
            'home_xg_per_shot_5': home_xg['xg_per_shot'],
            'away_xg_per_shot_5': away_xg['xg_per_shot'],
            'home_inside_box_xg_ratio': home_xg['inside_box_xg_ratio'],
            'away_inside_box_xg_ratio': away_xg['inside_box_xg_ratio'],
            
            # Big chances
            'home_big_chances_per_match_5': home_xg['big_chances_per_match'],
            'away_big_chances_per_match_5': away_xg['big_chances_per_match'],
            'home_big_chance_conversion_5': home_xg['big_chance_conversion'],
            'away_big_chance_conversion_5': away_xg['big_chance_conversion'],
            
            # xG trends (10 matches)
            'home_xg_trend_10': self._calculate_trend(home_xg_10['xg_values']),
            'away_xg_trend_10': self._calculate_trend(away_xg_10['xg_values']),
            'home_xga_trend_10': self._calculate_trend(home_xg_10['xga_values']),
            'away_xga_trend_10': self._calculate_trend(away_xg_10['xga_values']),
            
            # Set pieces
            'home_xg_from_corners_5': home_xg['xg_from_corners'],
            'away_xg_from_corners_5': away_xg['xg_from_corners'],
        }
    
    def _get_shot_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get shot analysis features (15 features)."""
        home_shots = self._calculate_shot_metrics(home_team_id, as_of_date)
        away_shots = self._calculate_shot_metrics(away_team_id, as_of_date)
        
        return {
            'home_shots_per_match_5': home_shots['shots_per_match'],
            'away_shots_per_match_5': away_shots['shots_per_match'],
            'home_shots_on_target_per_match_5': home_shots['sot_per_match'],
            'away_shots_on_target_per_match_5': away_shots['sot_per_match'],
            'home_inside_box_shot_pct_5': home_shots['inside_box_pct'],
            'away_inside_box_shot_pct_5': away_shots['inside_box_pct'],
            'home_outside_box_shot_pct_5': home_shots['outside_box_pct'],
            'away_outside_box_shot_pct_5': away_shots['outside_box_pct'],
            'home_shot_accuracy_5': home_shots['shot_accuracy'],
            'away_shot_accuracy_5': away_shots['shot_accuracy'],
            'home_shots_per_goal_5': home_shots['shots_per_goal'],
            'away_shots_per_goal_5': away_shots['shots_per_goal'],
            'home_shots_conceded_per_match_5': home_shots['shots_conceded_per_match'],
            'away_shots_conceded_per_match_5': away_shots['shots_conceded_per_match'],
            'home_shots_on_target_conceded_5': home_shots['sot_conceded_per_match'],
        }
    
    def _get_defensive_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get defensive intensity features (12 features)."""
        home_def = self._calculate_defensive_metrics(home_team_id, as_of_date)
        away_def = self._calculate_defensive_metrics(away_team_id, as_of_date)
        
        return {
            'home_ppda_5': home_def['ppda'],
            'away_ppda_5': away_def['ppda'],
            'home_tackles_per_90': home_def['tackles_per_90'],
            'away_tackles_per_90': away_def['tackles_per_90'],
            'home_interceptions_per_90': home_def['interceptions_per_90'],
            'away_interceptions_per_90': away_def['interceptions_per_90'],
            'home_tackle_success_rate_5': home_def['tackle_success_rate'],
            'away_tackle_success_rate_5': away_def['tackle_success_rate'],
            'home_defensive_actions_per_90': home_def['defensive_actions_per_90'],
            'away_defensive_actions_per_90': away_def['defensive_actions_per_90'],
            'home_possession_pct_5': home_def['possession_pct'],
            'away_possession_pct_5': away_def['possession_pct'],
        }
    
    def _get_attack_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get attack pattern features (8 features)."""
        home_attack = self._calculate_attack_metrics(home_team_id, as_of_date)
        away_attack = self._calculate_attack_metrics(away_team_id, as_of_date)
        
        return {
            'home_attacks_per_match_5': home_attack['attacks_per_match'],
            'away_attacks_per_match_5': away_attack['attacks_per_match'],
            'home_dangerous_attacks_per_match_5': home_attack['dangerous_attacks_per_match'],
            'away_dangerous_attacks_per_match_5': away_attack['dangerous_attacks_per_match'],
            'home_dangerous_attack_ratio_5': home_attack['dangerous_attack_ratio'],
            'away_dangerous_attack_ratio_5': away_attack['dangerous_attack_ratio'],
            'home_shots_per_attack_5': home_attack['shots_per_attack'],
            'away_shots_per_attack_5': away_attack['shots_per_attack'],
        }
    
    # ===== PILLAR 3: HIDDEN EDGES =====
    
    def _get_momentum_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get momentum & trajectory features (12 features)."""
        home_momentum = self._calculate_momentum(home_team_id, as_of_date)
        away_momentum = self._calculate_momentum(away_team_id, as_of_date)
        
        return {
            'home_points_trend_10': home_momentum['points_trend'],
            'away_points_trend_10': away_momentum['points_trend'],
            'home_weighted_form_5': home_momentum['weighted_form'],
            'away_weighted_form_5': away_momentum['weighted_form'],
            'home_win_streak': home_momentum['win_streak'],
            'away_win_streak': away_momentum['win_streak'],
            'home_unbeaten_streak': home_momentum['unbeaten_streak'],
            'away_unbeaten_streak': away_momentum['unbeaten_streak'],
            'home_clean_sheet_streak': home_momentum['clean_sheet_streak'],
            'away_clean_sheet_streak': away_momentum['clean_sheet_streak'],
            # xG trends already in xG features
            'home_xg_momentum': home_momentum.get('xg_trend', 0.0),
            'away_xg_momentum': away_momentum.get('xg_trend', 0.0),
        }
    
    def _get_fixture_adjusted_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get fixture difficulty adjusted features (10 features)."""
        home_adj = self._calculate_fixture_adjusted(home_team_id, as_of_date)
        away_adj = self._calculate_fixture_adjusted(away_team_id, as_of_date)
        
        return {
            'home_avg_opponent_elo_5': home_adj['avg_opponent_elo'],
            'away_avg_opponent_elo_5': away_adj['avg_opponent_elo'],
            'home_points_vs_top_6': home_adj['points_vs_top_6'],
            'away_points_vs_top_6': away_adj['points_vs_top_6'],
            'home_points_vs_bottom_6': home_adj['points_vs_bottom_6'],
            'away_points_vs_bottom_6': away_adj['points_vs_bottom_6'],
            'home_xg_vs_top_half': home_adj['xg_vs_top_half'],
            'away_xg_vs_top_half': away_adj['xg_vs_top_half'],
            'home_xga_vs_bottom_half': home_adj['xga_vs_bottom_half'],
            'away_xga_vs_bottom_half': away_adj['xga_vs_bottom_half'],
        }
    
    def _get_player_features(self, home_team_id, away_team_id, as_of_date) -> Dict:
        """Get player quality features (10 features)."""
        home_players = self._calculate_player_quality(home_team_id, as_of_date)
        away_players = self._calculate_player_quality(away_team_id, as_of_date)
        
        return {
            'home_lineup_avg_rating_5': home_players['avg_rating'],
            'away_lineup_avg_rating_5': away_players['avg_rating'],
            'home_top_3_players_rating': home_players['top_3_rating'],
            'away_top_3_players_rating': away_players['top_3_rating'],
            'home_players_in_form': home_players['players_in_form_pct'],
            'away_players_in_form': away_players['players_in_form_pct'],
            'home_key_players_available': home_players['key_players_available'],
            'away_key_players_available': away_players['key_players_available'],
            'home_players_unavailable': home_players['players_unavailable'],
            'away_players_unavailable': away_players['players_unavailable'],
        }
    
    def _get_context_features(self, home_team_id, away_team_id, league_id, fixture_id, as_of_date) -> Dict:
        """Get situational context features (4 features - simplified)."""
        home_rest = self._get_days_since_last_match(home_team_id, as_of_date)
        away_rest = self._get_days_since_last_match(away_team_id, as_of_date)
        
        return {
            'home_days_since_last_match': home_rest,
            'away_days_since_last_match': away_rest,
            'rest_advantage': home_rest - away_rest,
            'is_derby_match': 0,  # Would need city/region data
        }
    
    # ===== HELPER METHODS =====
    
    def _calculate_form(self, team_id, as_of_date, n_matches=5) -> Dict:
        """Calculate form metrics for a team."""
        team_matches = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            ((self.fixtures['home_team_id'] == team_id) | (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['result'].notna())
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(team_matches) == 0:
            return {'points': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                    'goals_scored': 0, 'goals_conceded': 0, 'goal_difference': 0}
        
        points = wins = draws = losses = goals_scored = goals_conceded = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']
            
            goals_scored += team_score
            goals_conceded += opp_score
            
            if team_score > opp_score:
                points += 3
                wins += 1
            elif team_score == opp_score:
                points += 1
                draws += 1
            else:
                losses += 1
        
        return {
            'points': points,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goals_scored - goals_conceded,
        }
    
    def _calculate_home_away_split(self, team_id, as_of_date, is_home=True) -> Dict:
        """Calculate home or away specific form."""
        if is_home:
            matches = self.fixtures[
                (self.fixtures['starting_at'] < as_of_date) &
                (self.fixtures['home_team_id'] == team_id) &
                (self.fixtures['result'].notna())
            ]
        else:
            matches = self.fixtures[
                (self.fixtures['starting_at'] < as_of_date) &
                (self.fixtures['away_team_id'] == team_id) &
                (self.fixtures['result'].notna())
            ]
        
        if len(matches) == 0:
            return {'points': 0, 'ppg': 0.0, 'win_pct': 0.0}
        
        points = wins = 0
        for _, match in matches.iterrows():
            if is_home:
                if match['result'] == 'H':
                    points += 3
                    wins += 1
                elif match['result'] == 'D':
                    points += 1
            else:
                if match['result'] == 'A':
                    points += 3
                    wins += 1
                elif match['result'] == 'D':
                    points += 1
        
        return {
            'points': points,
            'ppg': points / len(matches),
            'win_pct': wins / len(matches),
        }
    
    def _calculate_xg_metrics(self, team_id, as_of_date, n_matches=5) -> Dict:
        """Calculate xG metrics for a team."""
        # Get team's recent matches
        team_fixtures = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            ((self.fixtures['home_team_id'] == team_id) | (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['result'].notna())
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(team_fixtures) == 0:
            return self._empty_xg_metrics()
        
        xg_values = []
        xga_values = []
        goals = []
        goals_against = []
        shots = []
        inside_box_xg = []
        total_xg = []
        big_chances = []
        big_chance_goals = []
        corners_xg = []
        
        for _, fixture in team_fixtures.iterrows():
            # Get team's statistics
            team_stats = self.statistics[
                (self.statistics['fixture_id'] == fixture['fixture_id']) &
                (self.statistics['team_id'] == team_id)
            ]
            
            # Get opponent's statistics
            opponent_id = fixture['away_team_id'] if fixture['home_team_id'] == team_id else fixture['home_team_id']
            opp_stats = self.statistics[
                (self.statistics['fixture_id'] == fixture['fixture_id']) &
                (self.statistics['team_id'] == opponent_id)
            ]
            
            if len(team_stats) > 0:
                team_stat = team_stats.iloc[0]
                xg_values.append(team_stat.get('xG', 0))
                shots.append(team_stat.get('shots_total', 0))
                inside_box_xg.append(team_stat.get('shots_inside_box', 0) * 0.32)
                total_xg.append(team_stat.get('xG', 0))
                
                # Big chances (shots on target from inside box)
                sot_inside = min(team_stat.get('shots_on_target', 0), team_stat.get('shots_inside_box', 0))
                big_chances.append(sot_inside)
                
                # Corners xG
                corners_xg.append(team_stat.get('corners', 0) * 0.03)  # Rough estimate
            
            if len(opp_stats) > 0:
                xga_values.append(opp_stats.iloc[0].get('xG', 0))
            
            # Actual goals
            is_home = fixture['home_team_id'] == team_id
            team_goals = fixture['home_score'] if is_home else fixture['away_score']
            opp_goals = fixture['away_score'] if is_home else fixture['home_score']
            goals.append(team_goals)
            goals_against.append(opp_goals)
            
            if len(big_chances) > 0 and big_chances[-1] > 0:
                big_chance_goals.append(team_goals)
        
        # Calculate metrics
        xg_per_match = np.mean(xg_values) if xg_values else 0
        xga_per_match = np.mean(xga_values) if xga_values else 0
        goals_per_match = np.mean(goals) if goals else 0
        ga_per_match = np.mean(goals_against) if goals_against else 0
        shots_per_match = np.mean(shots) if shots else 0
        
        return {
            'xg_per_match': xg_per_match,
            'xga_per_match': xga_per_match,
            'xgd': xg_per_match - xga_per_match,
            'goals_vs_xg': goals_per_match - xg_per_match,
            'ga_vs_xga': ga_per_match - xga_per_match,
            'xg_per_shot': xg_per_match / shots_per_match if shots_per_match > 0 else 0,
            'inside_box_xg_ratio': np.mean(inside_box_xg) / np.mean(total_xg) if np.mean(total_xg) > 0 else 0,
            'big_chances_per_match': np.mean(big_chances) if big_chances else 0,
            'big_chance_conversion': np.mean(big_chance_goals) / np.mean(big_chances) if np.mean(big_chances) > 0 else 0,
            'xg_from_corners': np.mean(corners_xg) if corners_xg else 0,
            'xg_values': xg_values,
            'xga_values': xga_values,
        }
    
    def _empty_xg_metrics(self) -> Dict:
        """Return empty xG metrics."""
        return {
            'xg_per_match': 0.0, 'xga_per_match': 0.0, 'xgd': 0.0,
            'goals_vs_xg': 0.0, 'ga_vs_xga': 0.0, 'xg_per_shot': 0.0,
            'inside_box_xg_ratio': 0.0, 'big_chances_per_match': 0.0,
            'big_chance_conversion': 0.0, 'xg_from_corners': 0.0,
            'xg_values': [], 'xga_values': [],
        }
    
    def _calculate_shot_metrics(self, team_id, as_of_date, n_matches=5) -> Dict:
        """Calculate shot metrics."""
        team_fixtures = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            ((self.fixtures['home_team_id'] == team_id) | (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['result'].notna())
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(team_fixtures) == 0:
            return self._empty_shot_metrics()
        
        shots = []
        sot = []
        inside_box = []
        outside_box = []
        goals = []
        shots_conceded = []
        sot_conceded = []
        
        for _, fixture in team_fixtures.iterrows():
            team_stats = self.statistics[
                (self.statistics['fixture_id'] == fixture['fixture_id']) &
                (self.statistics['team_id'] == team_id)
            ]
            
            opponent_id = fixture['away_team_id'] if fixture['home_team_id'] == team_id else fixture['home_team_id']
            opp_stats = self.statistics[
                (self.statistics['fixture_id'] == fixture['fixture_id']) &
                (self.statistics['team_id'] == opponent_id)
            ]
            
            if len(team_stats) > 0:
                stat = team_stats.iloc[0]
                shots.append(stat.get('shots_total', 0))
                sot.append(stat.get('shots_on_target', 0))
                inside_box.append(stat.get('shots_inside_box', 0))
                outside_box.append(stat.get('shots_outside_box', 0))
            
            if len(opp_stats) > 0:
                shots_conceded.append(opp_stats.iloc[0].get('shots_total', 0))
                sot_conceded.append(opp_stats.iloc[0].get('shots_on_target', 0))
            
            is_home = fixture['home_team_id'] == team_id
            goals.append(fixture['home_score'] if is_home else fixture['away_score'])
        
        shots_avg = np.mean(shots) if shots else 0
        sot_avg = np.mean(sot) if sot else 0
        inside_avg = np.mean(inside_box) if inside_box else 0
        outside_avg = np.mean(outside_box) if outside_box else 0
        total_shots = inside_avg + outside_avg
        
        return {
            'shots_per_match': shots_avg,
            'sot_per_match': sot_avg,
            'inside_box_pct': inside_avg / total_shots if total_shots > 0 else 0,
            'outside_box_pct': outside_avg / total_shots if total_shots > 0 else 0,
            'shot_accuracy': sot_avg / shots_avg if shots_avg > 0 else 0,
            'shots_per_goal': shots_avg / np.mean(goals) if np.mean(goals) > 0 else 0,
            'shots_conceded_per_match': np.mean(shots_conceded) if shots_conceded else 0,
            'sot_conceded_per_match': np.mean(sot_conceded) if sot_conceded else 0,
        }
    
    def _empty_shot_metrics(self) -> Dict:
        """Return empty shot metrics."""
        return {
            'shots_per_match': 0.0, 'sot_per_match': 0.0,
            'inside_box_pct': 0.0, 'outside_box_pct': 0.0,
            'shot_accuracy': 0.0, 'shots_per_goal': 0.0,
            'shots_conceded_per_match': 0.0, 'sot_conceded_per_match': 0.0,
        }
    
    def _calculate_defensive_metrics(self, team_id, as_of_date, n_matches=5) -> Dict:
        """Calculate defensive metrics including PPDA."""
        team_fixtures = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            ((self.fixtures['home_team_id'] == team_id) | (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['result'].notna())
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(team_fixtures) == 0:
            return self._empty_defensive_metrics()
        
        ppda_values = []
        tackles = []
        interceptions = []
        possession = []
        
        for _, fixture in team_fixtures.iterrows():
            team_stats = self.statistics[
                (self.statistics['fixture_id'] == fixture['fixture_id']) &
                (self.statistics['team_id'] == team_id)
            ]
            
            opponent_id = fixture['away_team_id'] if fixture['home_team_id'] == team_id else fixture['home_team_id']
            opp_stats = self.statistics[
                (self.statistics['fixture_id'] == fixture['fixture_id']) &
                (self.statistics['team_id'] == opponent_id)
            ]
            
            if len(team_stats) > 0 and len(opp_stats) > 0:
                stat = team_stats.iloc[0]
                opp_stat = opp_stats.iloc[0]
                
                # PPDA = opponent passes / (tackles + interceptions)
                team_tackles = stat.get('tackles', 0)
                team_int = stat.get('interceptions', 0)
                opp_passes = opp_stat.get('passes_total', 0)
                
                defensive_actions = team_tackles + team_int
                if defensive_actions > 0:
                    ppda_values.append(opp_passes / defensive_actions)
                
                tackles.append(team_tackles)
                interceptions.append(team_int)
                possession.append(stat.get('possession', 50))
        
        tackles_avg = np.mean(tackles) if tackles else 0
        int_avg = np.mean(interceptions) if interceptions else 0
        
        return {
            'ppda': np.mean(ppda_values) if ppda_values else 10.0,  # Default ~10
            'tackles_per_90': tackles_avg,
            'interceptions_per_90': int_avg,
            'tackle_success_rate': 0.7,  # Would need tackle success data
            'defensive_actions_per_90': tackles_avg + int_avg,
            'possession_pct': np.mean(possession) if possession else 50.0,
        }
    
    def _empty_defensive_metrics(self) -> Dict:
        """Return empty defensive metrics."""
        return {
            'ppda': 10.0, 'tackles_per_90': 0.0, 'interceptions_per_90': 0.0,
            'tackle_success_rate': 0.0, 'defensive_actions_per_90': 0.0,
            'possession_pct': 50.0,
        }
    
    def _calculate_attack_metrics(self, team_id, as_of_date, n_matches=5) -> Dict:
        """Calculate attack pattern metrics."""
        team_fixtures = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            ((self.fixtures['home_team_id'] == team_id) | (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['result'].notna())
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(team_fixtures) == 0:
            return self._empty_attack_metrics()
        
        attacks = []
        dangerous_attacks = []
        shots = []
        
        for _, fixture in team_fixtures.iterrows():
            team_stats = self.statistics[
                (self.statistics['fixture_id'] == fixture['fixture_id']) &
                (self.statistics['team_id'] == team_id)
            ]
            
            if len(team_stats) > 0:
                stat = team_stats.iloc[0]
                attacks.append(stat.get('attacks', 0))
                dangerous_attacks.append(stat.get('dangerous_attacks', 0))
                shots.append(stat.get('shots_total', 0))
        
        attacks_avg = np.mean(attacks) if attacks else 0
        dangerous_avg = np.mean(dangerous_attacks) if dangerous_attacks else 0
        shots_avg = np.mean(shots) if shots else 0
        
        return {
            'attacks_per_match': attacks_avg,
            'dangerous_attacks_per_match': dangerous_avg,
            'dangerous_attack_ratio': dangerous_avg / attacks_avg if attacks_avg > 0 else 0,
            'shots_per_attack': shots_avg / attacks_avg if attacks_avg > 0 else 0,
        }
    
    def _empty_attack_metrics(self) -> Dict:
        """Return empty attack metrics."""
        return {
            'attacks_per_match': 0.0, 'dangerous_attacks_per_match': 0.0,
            'dangerous_attack_ratio': 0.0, 'shots_per_attack': 0.0,
        }
    
    def _calculate_momentum(self, team_id, as_of_date) -> Dict:
        """Calculate momentum and trajectory features."""
        team_fixtures = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            ((self.fixtures['home_team_id'] == team_id) | (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['result'].notna())
        ].sort_values('starting_at', ascending=False).head(10)
        
        if len(team_fixtures) == 0:
            return self._empty_momentum()
        
        # Points trend (linear regression)
        points = []
        for _, match in team_fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']
            
            if team_score > opp_score:
                points.append(3)
            elif team_score == opp_score:
                points.append(1)
            else:
                points.append(0)
        
        points_trend = self._calculate_trend(points)
        
        # Weighted form (exponential decay)
        weighted_points = 0
        total_weight = 0
        decay = 0.9
        for i, p in enumerate(points[:5]):
            weight = decay ** i
            weighted_points += p * weight
            total_weight += weight
        weighted_form = weighted_points / total_weight if total_weight > 0 else 0
        
        # Streaks
        win_streak = unbeaten_streak = clean_sheet_streak = 0
        for _, match in team_fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']
            
            # Win streak
            if team_score > opp_score:
                win_streak += 1
            else:
                break
        
        for _, match in team_fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']
            
            # Unbeaten streak
            if team_score >= opp_score:
                unbeaten_streak += 1
            else:
                break
        
        for _, match in team_fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            opp_score = match['away_score'] if is_home else match['home_score']
            
            # Clean sheet streak
            if opp_score == 0:
                clean_sheet_streak += 1
            else:
                break
        
        return {
            'points_trend': points_trend,
            'weighted_form': weighted_form,
            'win_streak': win_streak,
            'unbeaten_streak': unbeaten_streak,
            'clean_sheet_streak': clean_sheet_streak,
        }
    
    def _empty_momentum(self) -> Dict:
        """Return empty momentum metrics."""
        return {
            'points_trend': 0.0, 'weighted_form': 0.0,
            'win_streak': 0, 'unbeaten_streak': 0, 'clean_sheet_streak': 0,
        }
    
    def _calculate_fixture_adjusted(self, team_id, as_of_date) -> Dict:
        """Calculate fixture difficulty adjusted metrics."""
        # Simplified version - would need full implementation
        return {
            'avg_opponent_elo': 1500.0,
            'points_vs_top_6': 0,
            'points_vs_bottom_6': 0,
            'xg_vs_top_half': 0.0,
            'xga_vs_bottom_half': 0.0,
        }
    
    def _calculate_player_quality(self, team_id, as_of_date) -> Dict:
        """Calculate player quality metrics from lineups."""
        team_fixtures = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            ((self.fixtures['home_team_id'] == team_id) | (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['result'].notna())
        ].sort_values('starting_at', ascending=False).head(5)
        
        if len(team_fixtures) == 0:
            return self._empty_player_quality()
        
        ratings = []
        top_ratings = []
        in_form_count = 0
        total_players = 0
        
        for _, fixture in team_fixtures.iterrows():
            # Get lineups for this fixture
            fixture_lineups = self.lineups[
                (self.lineups['fixture_id'] == fixture['fixture_id']) &
                (self.lineups['team_id'] == team_id) &
                (self.lineups['is_starter'] == True)
            ]
            
            if len(fixture_lineups) > 0:
                player_ratings = fixture_lineups['rating'].dropna()
                if len(player_ratings) > 0:
                    ratings.extend(player_ratings.tolist())
                    top_ratings.extend(player_ratings.nlargest(3).tolist())
                    in_form_count += len(player_ratings[player_ratings > 7.0])
                    total_players += len(player_ratings)
        
        return {
            'avg_rating': np.mean(ratings) if ratings else 6.5,
            'top_3_rating': np.mean(top_ratings) if top_ratings else 6.5,
            'players_in_form_pct': in_form_count / total_players if total_players > 0 else 0.0,
            'key_players_available': 5,  # Would need injury data
            'players_unavailable': 0,  # Would need injury data
        }
    
    def _empty_player_quality(self) -> Dict:
        """Return empty player quality metrics."""
        return {
            'avg_rating': 6.5, 'top_3_rating': 6.5,
            'players_in_form_pct': 0.0,
            'key_players_available': 5, 'players_unavailable': 0,
        }
    
    def _get_days_since_last_match(self, team_id, as_of_date) -> int:
        """Get days since team's last match."""
        last_match = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            ((self.fixtures['home_team_id'] == team_id) | (self.fixtures['away_team_id'] == team_id)) &
            (self.fixtures['result'].notna())
        ].sort_values('starting_at', ascending=False).head(1)
        
        if len(last_match) == 0:
            return 7  # Default
        
        last_date = pd.to_datetime(last_match.iloc[0]['starting_at'])
        current_date = pd.to_datetime(as_of_date)
        return (current_date - last_date).days
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear regression slope (trend)."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            return float(slope)
        except:
            return 0.0
    
    def generate_training_dataset(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_file: str = 'data/csv/training_data.csv',
        n_jobs: int = 1  # Keep for API compatibility but default to sequential
    ) -> pd.DataFrame:
        """
        Generate complete training dataset with all 150-180 features.
        
        NOTE: Parallelization is not used due to high overhead (each worker
        loads 6GB of data). Sequential processing is faster for this use case.
        
        Args:
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            output_file: Output CSV file path
            n_jobs: Ignored (kept for API compatibility)
        
        Returns:
            DataFrame with all features
        """
        import time
        
        logger.info("=" * 80)
        logger.info("GENERATING TRAINING DATASET - COMPLETE FEATURE FRAMEWORK")
        logger.info("=" * 80)
        
        # Filter fixtures
        fixtures_to_process = self.fixtures.copy()
        
        if start_date:
            fixtures_to_process = fixtures_to_process[
                fixtures_to_process['starting_at'] >= start_date
            ]
        if end_date:
            fixtures_to_process = fixtures_to_process[
                fixtures_to_process['starting_at'] <= end_date
            ]
        
        # Only completed fixtures with results
        fixtures_to_process = fixtures_to_process[
            fixtures_to_process['result'].notna()
        ]
        
        logger.info(f"Processing {len(fixtures_to_process)} fixtures...")
        logger.info("Using optimized sequential processing (parallelization overhead too high)")
        
        # Generate features with timing
        all_features = []
        start_time = time.time()
        
        for idx, (_, fixture) in enumerate(tqdm(fixtures_to_process.iterrows(), 
                                                 total=len(fixtures_to_process),
                                                 desc="Generating features")):
            try:
                features = self.generate_features_for_fixture(
                    fixture['fixture_id'],
                    fixture['starting_at']
                )
                all_features.append(features)
                
                # Show progress every 1000 fixtures
                if (idx + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    remaining = (len(fixtures_to_process) - idx - 1) / rate
                    logger.info(f"Processed {idx+1}/{len(fixtures_to_process)} | "
                              f"Rate: {rate:.1f} fixtures/sec | "
                              f"ETA: {remaining/60:.1f} min")
            except Exception as e:
                logger.error(f"Error processing fixture {fixture['fixture_id']}: {e}")
                continue
        
        # Create DataFrame
        training_df = pd.DataFrame(all_features)
        
        # Save
        training_df.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("TRAINING DATASET COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(training_df):,}")
        logger.info(f"Total features: {len(training_df.columns)}")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Average rate: {len(training_df)/elapsed:.1f} fixtures/sec")
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 80)
        
        return training_df


def main():
    """Generate training dataset with complete feature framework."""
    engine = ComprehensiveCSVFeatureEngine()
    
    # Generate full training dataset
    training_df = engine.generate_training_dataset()
    
    print("\nFeature count by category:")
    feature_cols = [col for col in training_df.columns 
                   if col not in ['fixture_id', 'home_team_id', 'away_team_id', 
                                 'starting_at', 'league_id', 'result', 
                                 'target_home_win', 'target_draw', 'target_away_win',
                                 'home_goals', 'away_goals']]
    print(f"Total features: {len(feature_cols)}")
    
    print("\nTarget distribution:")
    print(training_df['result'].value_counts())


if __name__ == "__main__":
    main()
