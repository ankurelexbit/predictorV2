"""
Pillar 2: Modern Analytics Feature Engine

Generates 60 modern analytics features:
- Derived xG (25 features)
- Shot Analysis (15 features)
- Defensive Intensity (12 features)
- Attack Patterns (8 features)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class Pillar2ModernAnalyticsEngine:
    """Generate Pillar 2 modern analytics features."""
    
    # Derived xG coefficients (from config)
    XG_INSIDE_BOX = 0.12
    XG_OUTSIDE_BOX = 0.03
    XG_BIG_CHANCE = 0.35
    XG_CORNER = 0.03
    XG_ACCURACY_MULTIPLIER_MAX = 1.3
    
    def __init__(self, data_loader):
        """
        Initialize Pillar 2 engine.
        
        Args:
            data_loader: HistoricalDataLoader instance
        """
        self.data_loader = data_loader
        
        logger.info("Initialized Pillar2ModernAnalyticsEngine")
    
    def generate_features(
        self,
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """
        Generate all Pillar 2 features for a fixture.
        
        Args:
            fixture_id: Fixture ID
            home_team_id: Home team ID
            away_team_id: Away team ID
            as_of_date: Date cutoff (point-in-time)
            
        Returns:
            Dict with 60 features
        """
        features = {}
        
        # 1. Derived xG features (25)
        xg_features = self._get_xg_features(home_team_id, away_team_id, as_of_date)
        features.update(xg_features)
        
        # 2. Shot analysis features (15)
        shot_features = self._get_shot_features(home_team_id, away_team_id, as_of_date)
        features.update(shot_features)
        
        # 3. Defensive intensity features (12)
        defensive_features = self._get_defensive_features(home_team_id, away_team_id, as_of_date)
        features.update(defensive_features)
        
        # 4. Attack pattern features (8)
        attack_features = self._get_attack_features(home_team_id, away_team_id, as_of_date)
        features.update(attack_features)
        
        return features
    
    def _calculate_derived_xg(self, stats: pd.Series) -> float:
        """
        Calculate derived xG from base statistics.
        
        Formula:
        xG = (shots_inside_box × 0.12 + shots_outside_box × 0.03 + 
              big_chances × 0.35 + corners × 0.03) × accuracy_multiplier
        """
        if len(stats) == 0:
            return 0.0
        
        # Get base stats (handle missing values)
        shots_inside = stats.get('shots_inside_box', 0) or 0
        shots_outside = stats.get('shots_outside_box', 0) or 0
        big_chances = stats.get('stat_580', 0) or 0  # Big chances created
        corners = stats.get('stat_34', 0) or 0
        shots_on_target = stats.get('shots_on_target', 0) or 0
        shots_total = stats.get('shots_total', 0) or 0
        
        # Calculate accuracy multiplier
        if shots_total > 0:
            accuracy = shots_on_target / shots_total
            accuracy_multiplier = 1.0 + (accuracy * 0.3)  # Max 1.3
            accuracy_multiplier = min(accuracy_multiplier, self.XG_ACCURACY_MULTIPLIER_MAX)
        else:
            accuracy_multiplier = 1.0
        
        # Calculate xG
        xg = (
            shots_inside * self.XG_INSIDE_BOX +
            shots_outside * self.XG_OUTSIDE_BOX +
            big_chances * self.XG_BIG_CHANCE +
            corners * self.XG_CORNER
        ) * accuracy_multiplier
        
        return xg
    
    def _get_team_xg_stats(self, team_id: int, as_of_date: str, n: int = 5) -> Dict:
        """Get xG statistics for a team over last n matches."""
        fixtures = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=n)
        
        if len(fixtures) == 0:
            return {
                'xg_per_match': 0.0,
                'xga_per_match': 0.0,
                'xgd': 0.0,
                'goals_vs_xg': 0.0,
                'ga_vs_xga': 0.0,
                'xg_per_shot': 0.0,
                'big_chances_per_match': 0.0,
                'big_chance_conversion': 0.0,
            }
        
        total_xg = 0.0
        total_xga = 0.0
        total_goals = 0.0
        total_goals_conceded = 0.0
        total_shots = 0.0
        total_big_chances = 0.0
        
        for _, fixture in fixtures.iterrows():
            fixture_id = fixture['fixture_id']
            is_home = fixture['home_team_id'] == team_id
            
            # Get statistics
            stats = self.data_loader.get_statistics_for_fixture(fixture_id)
            
            if is_home:
                team_stats = stats['home']
                opp_stats = stats['away']
                goals = fixture.get('home_score', 0) or 0
                goals_conceded = fixture.get('away_score', 0) or 0
            else:
                team_stats = stats['away']
                opp_stats = stats['home']
                goals = fixture.get('away_score', 0) or 0
                goals_conceded = fixture.get('home_score', 0) or 0
            
            # Calculate xG for this match
            if len(team_stats) > 0:
                xg = self._calculate_derived_xg(team_stats)
                total_xg += xg
                total_shots += team_stats.get('shots_total', 0) or 0
                total_big_chances += team_stats.get('stat_580', 0) or 0
            
            # Calculate xGA (opponent's xG)
            if len(opp_stats) > 0:
                xga = self._calculate_derived_xg(opp_stats)
                total_xga += xga
            
            total_goals += goals
            total_goals_conceded += goals_conceded
        
        n_matches = len(fixtures)
        
        xg_per_match = total_xg / n_matches
        xga_per_match = total_xga / n_matches
        xgd = xg_per_match - xga_per_match
        goals_vs_xg = (total_goals - total_xg) / n_matches
        ga_vs_xga = (total_goals_conceded - total_xga) / n_matches
        xg_per_shot = total_xg / total_shots if total_shots > 0 else 0.0
        big_chances_per_match = total_big_chances / n_matches
        big_chance_conversion = total_goals / total_big_chances if total_big_chances > 0 else 0.0
        
        return {
            'xg_per_match': xg_per_match,
            'xga_per_match': xga_per_match,
            'xgd': xgd,
            'goals_vs_xg': goals_vs_xg,
            'ga_vs_xga': ga_vs_xga,
            'xg_per_shot': xg_per_shot,
            'big_chances_per_match': big_chances_per_match,
            'big_chance_conversion': big_chance_conversion,
        }
    
    def _get_xg_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate derived xG features (25 features)."""
        # Get xG stats for both teams (last 5 and 10 matches)
        home_xg_5 = self._get_team_xg_stats(home_team_id, as_of_date, n=5)
        away_xg_5 = self._get_team_xg_stats(away_team_id, as_of_date, n=5)
        home_xg_10 = self._get_team_xg_stats(home_team_id, as_of_date, n=10)
        away_xg_10 = self._get_team_xg_stats(away_team_id, as_of_date, n=10)
        
        # Calculate xG trends (10 match vs 5 match)
        home_xg_trend = home_xg_5['xg_per_match'] - home_xg_10['xg_per_match']
        away_xg_trend = away_xg_5['xg_per_match'] - away_xg_10['xg_per_match']
        home_xga_trend = home_xg_5['xga_per_match'] - home_xg_10['xga_per_match']
        away_xga_trend = away_xg_5['xga_per_match'] - away_xg_10['xga_per_match']
        
        return {
            # Core xG (5 match window)
            'home_derived_xg_per_match_5': home_xg_5['xg_per_match'],
            'away_derived_xg_per_match_5': away_xg_5['xg_per_match'],
            'home_derived_xga_per_match_5': home_xg_5['xga_per_match'],
            'away_derived_xga_per_match_5': away_xg_5['xga_per_match'],
            'home_derived_xgd_5': home_xg_5['xgd'],
            'away_derived_xgd_5': away_xg_5['xgd'],
            'derived_xgd_matchup': home_xg_5['xgd'] - away_xg_5['xgd'],
            
            # Performance vs expectation
            'home_goals_vs_xg_5': home_xg_5['goals_vs_xg'],
            'away_goals_vs_xg_5': away_xg_5['goals_vs_xg'],
            'home_ga_vs_xga_5': home_xg_5['ga_vs_xga'],
            'away_ga_vs_xga_5': away_xg_5['ga_vs_xga'],
            
            # Shot quality
            'home_xg_per_shot_5': home_xg_5['xg_per_shot'],
            'away_xg_per_shot_5': away_xg_5['xg_per_shot'],
            
            # Big chances
            'home_big_chances_per_match_5': home_xg_5['big_chances_per_match'],
            'away_big_chances_per_match_5': away_xg_5['big_chances_per_match'],
            'home_big_chance_conversion_5': home_xg_5['big_chance_conversion'],
            'away_big_chance_conversion_5': away_xg_5['big_chance_conversion'],
            
            # xG trends
            'home_xg_trend_10': home_xg_trend,
            'away_xg_trend_10': away_xg_trend,
            'home_xga_trend_10': home_xga_trend,
            'away_xga_trend_10': away_xga_trend,
            
            # 10 match window (additional context)
            'home_derived_xg_per_match_10': home_xg_10['xg_per_match'],
            'away_derived_xg_per_match_10': away_xg_10['xg_per_match'],
            'home_derived_xga_per_match_10': home_xg_10['xga_per_match'],
            'away_derived_xga_per_match_10': away_xg_10['xga_per_match'],
        }
    
    def _get_shot_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate shot analysis features (15 features)."""
        home_shots = self._get_team_shot_stats(home_team_id, as_of_date, n=5)
        away_shots = self._get_team_shot_stats(away_team_id, as_of_date, n=5)
        
        return {
            'home_shots_per_match_5': home_shots['shots_per_match'],
            'away_shots_per_match_5': away_shots['shots_per_match'],
            'home_shots_on_target_per_match_5': home_shots['sot_per_match'],
            'away_shots_on_target_per_match_5': away_shots['sot_per_match'],
            'home_inside_box_shot_pct_5': home_shots['inside_box_pct'],
            'away_inside_box_shot_pct_5': away_shots['inside_box_pct'],
            'home_outside_box_shot_pct_5': home_shots['outside_box_pct'],
            'away_outside_box_shot_pct_5': away_shots['outside_box_pct'],
            'home_shot_accuracy_5': home_shots['accuracy'],
            'away_shot_accuracy_5': away_shots['accuracy'],
            'home_shots_per_goal_5': home_shots['shots_per_goal'],
            'away_shots_per_goal_5': away_shots['shots_per_goal'],
            'home_shots_conceded_per_match_5': home_shots['shots_conceded'],
            'away_shots_conceded_per_match_5': away_shots['shots_conceded'],
            'home_shots_on_target_conceded_5': home_shots['sot_conceded'],
        }
    
    def _get_team_shot_stats(self, team_id: int, as_of_date: str, n: int = 5) -> Dict:
        """Get shot statistics for a team."""
        fixtures = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=n)
        
        if len(fixtures) == 0:
            return {
                'shots_per_match': 0.0, 'sot_per_match': 0.0,
                'inside_box_pct': 0.0, 'outside_box_pct': 0.0,
                'accuracy': 0.0, 'shots_per_goal': 0.0,
                'shots_conceded': 0.0, 'sot_conceded': 0.0
            }
        
        total_shots = 0
        total_sot = 0
        total_inside = 0
        total_outside = 0
        total_goals = 0
        total_shots_conceded = 0
        total_sot_conceded = 0
        
        for _, fixture in fixtures.iterrows():
            fixture_id = fixture['fixture_id']
            is_home = fixture['home_team_id'] == team_id
            
            stats = self.data_loader.get_statistics_for_fixture(fixture_id)
            
            if is_home:
                team_stats = stats['home']
                opp_stats = stats['away']
                goals = fixture.get('home_score', 0) or 0
            else:
                team_stats = stats['away']
                opp_stats = stats['home']
                goals = fixture.get('away_score', 0) or 0
            
            if len(team_stats) > 0:
                total_shots += team_stats.get('shots_total', 0) or 0
                total_sot += team_stats.get('shots_on_target', 0) or 0
                total_inside += team_stats.get('shots_inside_box', 0) or 0
                total_outside += team_stats.get('shots_outside_box', 0) or 0
                total_goals += goals
            
            if len(opp_stats) > 0:
                total_shots_conceded += opp_stats.get('shots_total', 0) or 0
                total_sot_conceded += opp_stats.get('shots_on_target', 0) or 0
        
        n_matches = len(fixtures)
        
        return {
            'shots_per_match': total_shots / n_matches,
            'sot_per_match': total_sot / n_matches,
            'inside_box_pct': total_inside / total_shots if total_shots > 0 else 0.0,
            'outside_box_pct': total_outside / total_shots if total_shots > 0 else 0.0,
            'accuracy': total_sot / total_shots if total_shots > 0 else 0.0,
            'shots_per_goal': total_shots / total_goals if total_goals > 0 else 0.0,
            'shots_conceded': total_shots_conceded / n_matches,
            'sot_conceded': total_sot_conceded / n_matches,
        }
    
    def _get_defensive_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate defensive intensity features (12 features)."""
        home_def = self._get_team_defensive_stats(home_team_id, as_of_date, n=5)
        away_def = self._get_team_defensive_stats(away_team_id, as_of_date, n=5)
        
        return {
            'home_ppda_5': home_def['ppda'],
            'away_ppda_5': away_def['ppda'],
            'home_tackles_per_90': home_def['tackles_per_90'],
            'away_tackles_per_90': away_def['tackles_per_90'],
            'home_interceptions_per_90': home_def['interceptions_per_90'],
            'away_interceptions_per_90': away_def['interceptions_per_90'],
            'home_defensive_actions_per_90': home_def['def_actions_per_90'],
            'away_defensive_actions_per_90': away_def['def_actions_per_90'],
            'home_possession_pct_5': home_def['possession'],
            'away_possession_pct_5': away_def['possession'],
            'home_passes_per_match_5': home_def['passes_per_match'],
            'away_passes_per_match_5': away_def['passes_per_match'],
        }
    
    def _get_team_defensive_stats(self, team_id: int, as_of_date: str, n: int = 5) -> Dict:
        """Get defensive statistics for a team."""
        fixtures = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=n)
        
        if len(fixtures) == 0:
            return {
                'ppda': 0.0, 'tackles_per_90': 0.0, 'interceptions_per_90': 0.0,
                'def_actions_per_90': 0.0, 'possession': 0.0, 'passes_per_match': 0.0
            }
        
        total_opp_passes = 0
        total_tackles = 0
        total_interceptions = 0
        total_possession = 0
        total_passes = 0
        
        for _, fixture in fixtures.iterrows():
            fixture_id = fixture['fixture_id']
            is_home = fixture['home_team_id'] == team_id
            
            stats = self.data_loader.get_statistics_for_fixture(fixture_id)
            
            if is_home:
                team_stats = stats['home']
                opp_stats = stats['away']
            else:
                team_stats = stats['away']
                opp_stats = stats['home']
            
            if len(team_stats) > 0:
                total_tackles += team_stats.get('stat_109', 0) or 0  # Tackles
                total_interceptions += team_stats.get('interceptions', 0) or 0
                total_possession += team_stats.get('possession', 0) or 0
                total_passes += team_stats.get('passes_total', 0) or 0
            
            if len(opp_stats) > 0:
                total_opp_passes += opp_stats.get('passes_total', 0) or 0
        
        n_matches = len(fixtures)
        
        # PPDA = opponent passes / (tackles + interceptions)
        defensive_actions = total_tackles + total_interceptions
        ppda = total_opp_passes / defensive_actions if defensive_actions > 0 else 0.0
        
        return {
            'ppda': ppda,
            'tackles_per_90': total_tackles / n_matches,
            'interceptions_per_90': total_interceptions / n_matches,
            'def_actions_per_90': defensive_actions / n_matches,
            'possession': total_possession / n_matches,
            'passes_per_match': total_passes / n_matches,
        }
    
    def _get_attack_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str
    ) -> Dict[str, float]:
        """Generate attack pattern features (8 features)."""
        home_attack = self._get_team_attack_stats(home_team_id, as_of_date, n=5)
        away_attack = self._get_team_attack_stats(away_team_id, as_of_date, n=5)
        
        return {
            'home_attacks_per_match_5': home_attack['attacks_per_match'],
            'away_attacks_per_match_5': away_attack['attacks_per_match'],
            'home_dangerous_attacks_per_match_5': home_attack['dangerous_attacks_per_match'],
            'away_dangerous_attacks_per_match_5': away_attack['dangerous_attacks_per_match'],
            'home_dangerous_attack_ratio_5': home_attack['dangerous_ratio'],
            'away_dangerous_attack_ratio_5': away_attack['dangerous_ratio'],
            'home_shots_per_attack_5': home_attack['shots_per_attack'],
            'away_shots_per_attack_5': away_attack['shots_per_attack'],
        }
    
    def _get_team_attack_stats(self, team_id: int, as_of_date: str, n: int = 5) -> Dict:
        """Get attack statistics for a team."""
        fixtures = self.data_loader.get_fixtures_before_date(team_id, as_of_date, n=n)
        
        if len(fixtures) == 0:
            return {
                'attacks_per_match': 0.0, 'dangerous_attacks_per_match': 0.0,
                'dangerous_ratio': 0.0, 'shots_per_attack': 0.0
            }
        
        total_attacks = 0
        total_dangerous = 0
        total_shots = 0
        
        for _, fixture in fixtures.iterrows():
            fixture_id = fixture['fixture_id']
            is_home = fixture['home_team_id'] == team_id
            
            stats = self.data_loader.get_statistics_for_fixture(fixture_id)
            
            if is_home:
                team_stats = stats['home']
            else:
                team_stats = stats['away']
            
            if len(team_stats) > 0:
                total_attacks += team_stats.get('attacks', 0) or 0
                total_dangerous += team_stats.get('dangerous_attacks', 0) or 0
                total_shots += team_stats.get('shots_total', 0) or 0
        
        n_matches = len(fixtures)
        
        return {
            'attacks_per_match': total_attacks / n_matches,
            'dangerous_attacks_per_match': total_dangerous / n_matches,
            'dangerous_ratio': total_dangerous / total_attacks if total_attacks > 0 else 0.0,
            'shots_per_attack': total_shots / total_attacks if total_attacks > 0 else 0.0,
        }
