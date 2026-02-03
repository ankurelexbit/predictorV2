"""
Pillar 2: Modern Analytics Feature Engine for V4 Pipeline.

Generates 60 modern analytics features:
- Derived xG (25 features)
- Shot analysis (15 features)
- Defensive intensity (12 features)
- Attack patterns (8 features)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Pillar2ModernAnalyticsEngine:
    """
    Generate modern analytics features (Pillar 2).
    
    These are science-backed features from modern football analytics.
    """
    
    def __init__(self, data_loader):
        """
        Initialize Pillar 2 engine.
        
        Args:
            data_loader: JSONDataLoader instance
        """
        self.data_loader = data_loader
        logger.info("Initialized Pillar2ModernAnalyticsEngine")
    
    def generate_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: datetime
    ) -> Dict:
        """
        Generate all 60 Pillar 2 features.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            as_of_date: Date to generate features for
            
        Returns:
            Dict with 60 features
        """
        features = {}
        
        # Get recent fixtures with statistics
        home_recent = self.data_loader.get_team_fixtures(home_team_id, as_of_date, limit=10)
        away_recent = self.data_loader.get_team_fixtures(away_team_id, as_of_date, limit=10)
        
        # 1. Derived xG features (25)
        features.update(self._get_derived_xg_features(
            home_team_id, away_team_id, home_recent, away_recent
        ))
        
        # 2. Shot analysis (15)
        features.update(self._get_shot_features(
            home_team_id, away_team_id, home_recent, away_recent
        ))
        
        # 3. Defensive intensity (12)
        features.update(self._get_defensive_features(
            home_team_id, away_team_id, home_recent, away_recent
        ))
        
        # 4. Attack patterns (8)
        features.update(self._get_attack_features(
            home_team_id, away_team_id, home_recent, away_recent
        ))
        
        return features
    
    def _get_derived_xg_features(
        self,
        home_team_id: int,
        away_team_id: int,
        home_recent: pd.DataFrame,
        away_recent: pd.DataFrame
    ) -> Dict:
        """Generate 25 derived xG features."""
        home_stats = self._calculate_xg_stats(home_team_id, home_recent)
        away_stats = self._calculate_xg_stats(away_team_id, away_recent)
        
        return {
            # Core derived xG
            'home_derived_xg_per_match_5': home_stats['xg_per_match_5'],
            'away_derived_xg_per_match_5': away_stats['xg_per_match_5'],
            'home_derived_xga_per_match_5': home_stats['xga_per_match_5'],
            'away_derived_xga_per_match_5': away_stats['xga_per_match_5'],
            
            # xG differential
            'home_derived_xgd_5': home_stats['xgd_5'],
            'away_derived_xgd_5': away_stats['xgd_5'],
            'derived_xgd_matchup': home_stats['xgd_5'] - away_stats['xgd_5'],
            
            # Performance vs expectation
            'home_goals_vs_xg_5': home_stats['goals_vs_xg_5'],
            'away_goals_vs_xg_5': away_stats['goals_vs_xg_5'],
            'home_ga_vs_xga_5': home_stats['ga_vs_xga_5'],
            'away_ga_vs_xga_5': away_stats['ga_vs_xga_5'],
            
            # Shot quality
            'home_xg_per_shot_5': home_stats['xg_per_shot_5'],
            'away_xg_per_shot_5': away_stats['xg_per_shot_5'],
            'home_inside_box_xg_ratio': home_stats['inside_box_xg_ratio'],
            'away_inside_box_xg_ratio': away_stats['inside_box_xg_ratio'],
            
            # Big chances
            'home_big_chances_per_match_5': home_stats['big_chances_5'],
            'away_big_chances_per_match_5': away_stats['big_chances_5'],
            'home_big_chance_conversion_5': home_stats['big_chance_conversion_5'],
            'away_big_chance_conversion_5': away_stats['big_chance_conversion_5'],
            
            # Set pieces
            'home_xg_from_corners_5': home_stats['xg_corners_5'],
            'away_xg_from_corners_5': away_stats['xg_corners_5'],
            
            # xG trends (using up to 10 matches, minimum 2 for trend)
            'home_xg_trend_10': home_stats['xg_trend_10'],
            'away_xg_trend_10': away_stats['xg_trend_10'],
            'home_xga_trend_10': home_stats['xga_trend_10'],
            'away_xga_trend_10': away_stats['xga_trend_10'],
        }
    
    def _calculate_xg_stats(self, team_id: int, fixtures: pd.DataFrame) -> Dict:
        """Calculate xG statistics for a team."""
        if len(fixtures) == 0:
            return self._empty_xg_stats()
        
        xg_values = []
        xga_values = []
        goals = []
        goals_against = []
        shots = []
        big_chances = []
        
        for _, match in fixtures.iterrows():
            is_home = match['home_team_id'] == team_id
            
            # Get statistics from CSV columns
            team_stats = self._extract_team_stats(match, is_home)
            opp_stats = self._extract_team_stats(match, not is_home)
            
            # Derive xG from stats
            xg = self._derive_xg_from_stats(team_stats)
            xga = self._derive_xg_from_stats(opp_stats)
            
            xg_values.append(xg)
            xga_values.append(xga)
            goals.append(match['home_score'] if is_home else match['away_score'])
            goals_against.append(match['away_score'] if is_home else match['home_score'])
            shots.append(team_stats.get('shots_total', 0))
            big_chances.append(team_stats.get('big_chances_created', 0))
        
        # Calculate metrics
        xg_5 = xg_values[:5]
        xga_5 = xga_values[:5]
        goals_5 = goals[:5]
        ga_5 = goals_against[:5]
        shots_5 = shots[:5]
        big_chances_5 = big_chances[:5]
        
        return {
            'xg_per_match_5': float(np.mean(xg_5)) if xg_5 else 0.0,
            'xga_per_match_5': float(np.mean(xga_5)) if xga_5 else 0.0,
            'xgd_5': float(np.mean(xg_5) - np.mean(xga_5)) if xg_5 and xga_5 else 0.0,
            'goals_vs_xg_5': float(np.mean(goals_5) - np.mean(xg_5)) if goals_5 and xg_5 else 0.0,
            'ga_vs_xga_5': float(np.mean(ga_5) - np.mean(xga_5)) if ga_5 and xga_5 else 0.0,
            'xg_per_shot_5': float(np.mean(xg_5) / np.mean(shots_5)) if shots_5 and np.mean(shots_5) > 0 else 0.0,
            'inside_box_xg_ratio': 0.7,  # Approximation
            'big_chances_5': float(np.mean(big_chances_5)) if big_chances_5 else 0.0,
            'big_chance_conversion_5': 0.3,  # Approximation
            'xg_corners_5': 0.1,  # Approximation
            'xg_trend_10': self._calculate_trend(xg_values[:10]) if len(xg_values) >= 2 else 0.0,
            'xga_trend_10': self._calculate_trend(xga_values[:10]) if len(xga_values) >= 2 else 0.0,
        }
    
    def _derive_xg_from_stats(self, stats: Dict) -> float:
        """Derive xG from available statistics."""
        # Simplified xG formula based on shots
        shots_on_target = stats.get('shots_on_target', 0)
        shots_total = stats.get('shots_total', 0)
        shots_inside_box = stats.get('shots_inside_box', shots_total * 0.6)  # Estimate
        
        # xG formula: weighted combination
        xg = (shots_on_target * 0.35) + (shots_inside_box * 0.15)
        return float(xg)
    
    def _extract_team_stats(self, match: pd.Series, is_home: bool) -> Dict:
        """Extract team statistics from match row (CSV columns)."""
        prefix = 'home_' if is_home else 'away_'

        shots_total = match.get(f'{prefix}shots_total', 0) or 0
        shots_on_target = match.get(f'{prefix}shots_on_target', 0) or 0
        shots_inside_box = match.get(f'{prefix}shots_inside_box', 0) or 0

        # Calculate big chances (high-quality opportunities)
        # Big chance = shot inside box that's on target (high probability of scoring)
        # Formula: Estimate as shots inside box weighted by accuracy
        if shots_total > 0:
            accuracy = shots_on_target / shots_total
            big_chances = shots_inside_box * accuracy * 0.8  # ~80% of accurate inside-box shots are big chances
        else:
            big_chances = 0

        return {
            'shots_total': shots_total,
            'shots_on_target': shots_on_target,
            'shots_inside_box': shots_inside_box,
            'big_chances_created': big_chances,
        }
    
    def _empty_xg_stats(self) -> Dict:
        """Return empty xG stats."""
        return {
            'xg_per_match_5': 0.0, 'xga_per_match_5': 0.0, 'xgd_5': 0.0,
            'goals_vs_xg_5': 0.0, 'ga_vs_xga_5': 0.0, 'xg_per_shot_5': 0.0,
            'inside_box_xg_ratio': 0.0, 'big_chances_5': 0.0,
            'big_chance_conversion_5': 0.0, 'xg_corners_5': 0.0,
            'xg_trend_10': 0.0, 'xga_trend_10': 0.0,
        }
    
    def _calculate_trend(self, values: list) -> float:
        """Calculate linear trend (slope)."""
        # Filter out NaN values
        clean_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]

        if len(clean_values) < 2:
            return 0.0

        x = np.arange(len(clean_values))
        try:
            slope = np.polyfit(x, clean_values, 1)[0]
            return float(slope) if not np.isnan(slope) else 0.0
        except:
            return 0.0
    
    def _get_shot_features(
        self,
        home_team_id: int,
        away_team_id: int,
        home_recent: pd.DataFrame,
        away_recent: pd.DataFrame
    ) -> Dict:
        """Generate 15 shot analysis features."""
        home_stats = self._calculate_shot_stats(home_team_id, home_recent)
        away_stats = self._calculate_shot_stats(away_team_id, away_recent)
        
        return {
            'home_shots_per_match_5': home_stats['shots_per_match'],
            'away_shots_per_match_5': away_stats['shots_per_match'],
            'home_shots_on_target_per_match_5': home_stats['shots_on_target_per_match'],
            'away_shots_on_target_per_match_5': away_stats['shots_on_target_per_match'],
            'home_inside_box_shot_pct_5': home_stats['inside_box_pct'],
            'away_inside_box_shot_pct_5': away_stats['inside_box_pct'],
            'home_outside_box_shot_pct_5': home_stats['outside_box_pct'],
            'away_outside_box_shot_pct_5': away_stats['outside_box_pct'],
            'home_shot_accuracy_5': home_stats['shot_accuracy'],
            'away_shot_accuracy_5': away_stats['shot_accuracy'],
            'home_shots_per_goal_5': home_stats['shots_per_goal'],
            'away_shots_per_goal_5': away_stats['shots_per_goal'],
            'home_shots_conceded_per_match_5': home_stats['shots_conceded'],
            'away_shots_conceded_per_match_5': away_stats['shots_conceded'],
            'home_shots_on_target_conceded_5': home_stats['shots_on_target_conceded'],
        }
    
    def _calculate_shot_stats(self, team_id: int, fixtures: pd.DataFrame) -> Dict:
        """Calculate shot statistics from CSV."""
        if len(fixtures) == 0:
            return {'shots_per_match': 0.0, 'shots_on_target_per_match': 0.0, 
                    'inside_box_pct': 0.0, 'outside_box_pct': 0.0, 
                    'shot_accuracy': 0.0, 'shots_per_goal': 0.0,
                    'shots_conceded': 0.0, 'shots_on_target_conceded': 0.0}
        
        shots_list = []
        shots_on_target_list = []
        inside_box_list = []
        outside_box_list = []
        goals_list = []
        shots_conceded_list = []
        shots_on_target_conceded_list = []
        
        for _, match in fixtures[:5].iterrows():
            is_home = match['home_team_id'] == team_id
            prefix = 'home_' if is_home else 'away_'
            opp_prefix = 'away_' if is_home else 'home_'
            
            shots = match.get(f'{prefix}shots_total', 0) or 0
            shots_on_target = match.get(f'{prefix}shots_on_target', 0) or 0
            inside_box = match.get(f'{prefix}shots_inside_box', 0) or 0
            outside_box = match.get(f'{prefix}shots_outside_box', 0) or 0
            goals = match['home_score'] if is_home else match['away_score']
            shots_conceded = match.get(f'{opp_prefix}shots_total', 0) or 0
            shots_on_target_conceded = match.get(f'{opp_prefix}shots_on_target', 0) or 0
            
            shots_list.append(shots)
            shots_on_target_list.append(shots_on_target)
            inside_box_list.append(inside_box)
            outside_box_list.append(outside_box)
            goals_list.append(goals or 0)
            shots_conceded_list.append(shots_conceded)
            shots_on_target_conceded_list.append(shots_on_target_conceded)
        
        avg_shots = np.mean(shots_list) if shots_list else 0.0
        avg_shots_on_target = np.mean(shots_on_target_list) if shots_on_target_list else 0.0
        total_inside = sum(inside_box_list)
        total_outside = sum(outside_box_list)
        total_shots = sum(shots_list)
        total_goals = sum(goals_list)
        
        return {
            'shots_per_match': float(avg_shots),
            'shots_on_target_per_match': float(avg_shots_on_target),
            'inside_box_pct': float(total_inside / total_shots) if total_shots > 0 else 0.0,
            'outside_box_pct': float(total_outside / total_shots) if total_shots > 0 else 0.0,
            'shot_accuracy': float(sum(shots_on_target_list) / total_shots) if total_shots > 0 else 0.0,
            'shots_per_goal': float(total_shots / total_goals) if total_goals > 0 else 0.0,
            'shots_conceded': float(np.mean(shots_conceded_list)) if shots_conceded_list else 0.0,
            'shots_on_target_conceded': float(np.mean(shots_on_target_conceded_list)) if shots_on_target_conceded_list else 0.0,
        }
    
    def _get_defensive_features(
        self,
        home_team_id: int,
        away_team_id: int,
        home_recent: pd.DataFrame,
        away_recent: pd.DataFrame
    ) -> Dict:
        """Generate 12 defensive intensity features."""
        home_stats = self._calculate_defensive_stats(home_team_id, home_recent)
        away_stats = self._calculate_defensive_stats(away_team_id, away_recent)
        
        return {
            'home_ppda_5': home_stats['ppda'],
            'away_ppda_5': away_stats['ppda'],
            'home_tackles_per_90': home_stats['tackles'],
            'away_tackles_per_90': away_stats['tackles'],
            'home_interceptions_per_90': home_stats['interceptions'],
            'away_interceptions_per_90': away_stats['interceptions'],
            'home_tackle_success_rate_5': home_stats['tackle_success'],
            'away_tackle_success_rate_5': away_stats['tackle_success'],
            'home_defensive_actions_per_90': home_stats['defensive_actions'],
            'away_defensive_actions_per_90': away_stats['defensive_actions'],
            'home_possession_pct_5': home_stats['possession'],
            'away_possession_pct_5': away_stats['possession'],
        }
    
    def _calculate_defensive_stats(self, team_id: int, fixtures: pd.DataFrame) -> Dict:
        """Calculate defensive statistics from CSV."""
        if len(fixtures) == 0:
            return {'ppda': 0.0, 'tackles': 0.0, 'interceptions': 0.0,
                    'tackle_success': 0.0, 'defensive_actions': 0.0, 'possession': 0.0}
        
        tackles_list = []
        interceptions_list = []
        possession_list = []
        
        for _, match in fixtures[:5].iterrows():
            is_home = match['home_team_id'] == team_id
            prefix = 'home_' if is_home else 'away_'
            
            tackles = match.get(f'{prefix}tackles', 0) or 0
            interceptions = match.get(f'{prefix}interceptions', 0) or 0
            possession = match.get(f'{prefix}ball_possession', 0) or 0
            
            tackles_list.append(tackles)
            interceptions_list.append(interceptions)
            possession_list.append(possession)
        
        avg_tackles = np.mean(tackles_list) if tackles_list else 0.0
        avg_interceptions = np.mean(interceptions_list) if interceptions_list else 0.0
        avg_possession = np.mean(possession_list) if possession_list else 0.0
        
        return {
            'ppda': 10.5,  # Placeholder (requires passes allowed per defensive action)
            'tackles': float(avg_tackles),
            'interceptions': float(avg_interceptions),
            'tackle_success': 0.75,  # Placeholder
            'defensive_actions': float(avg_tackles + avg_interceptions),
            'possession': float(avg_possession),
        }
    
    def _get_attack_features(
        self,
        home_team_id: int,
        away_team_id: int,
        home_recent: pd.DataFrame,
        away_recent: pd.DataFrame
    ) -> Dict:
        """Generate 8 attack pattern features."""
        home_stats = self._calculate_attack_stats(home_team_id, home_recent)
        away_stats = self._calculate_attack_stats(away_team_id, away_recent)
        
        return {
            'home_attacks_per_match_5': home_stats['attacks'],
            'away_attacks_per_match_5': away_stats['attacks'],
            'home_dangerous_attacks_per_match_5': home_stats['dangerous_attacks'],
            'away_dangerous_attacks_per_match_5': away_stats['dangerous_attacks'],
            'home_dangerous_attack_ratio_5': home_stats['dangerous_ratio'],
            'away_dangerous_attack_ratio_5': away_stats['dangerous_ratio'],
            'home_shots_per_attack_5': home_stats['shots_per_attack'],
            'away_shots_per_attack_5': away_stats['shots_per_attack'],
        }
    
    def _calculate_attack_stats(self, team_id: int, fixtures: pd.DataFrame) -> Dict:
        """Calculate attack statistics from CSV."""
        if len(fixtures) == 0:
            return {'attacks': 0.0, 'dangerous_attacks': 0.0, 
                    'dangerous_ratio': 0.0, 'shots_per_attack': 0.0}
        
        attacks_list = []
        dangerous_attacks_list = []
        shots_list = []
        
        for _, match in fixtures[:5].iterrows():
            is_home = match['home_team_id'] == team_id
            prefix = 'home_' if is_home else 'away_'
            
            attacks = match.get(f'{prefix}attacks', 0) or 0
            dangerous_attacks = match.get(f'{prefix}dangerous_attacks', 0) or 0
            shots = match.get(f'{prefix}shots_total', 0) or 0
            
            attacks_list.append(attacks)
            dangerous_attacks_list.append(dangerous_attacks)
            shots_list.append(shots)
        
        avg_attacks = np.mean(attacks_list) if attacks_list else 0.0
        avg_dangerous = np.mean(dangerous_attacks_list) if dangerous_attacks_list else 0.0
        total_attacks = sum(attacks_list)
        total_shots = sum(shots_list)
        
        return {
            'attacks': float(avg_attacks),
            'dangerous_attacks': float(avg_dangerous),
            'dangerous_ratio': float(sum(dangerous_attacks_list) / total_attacks) if total_attacks > 0 else 0.0,
            'shots_per_attack': float(total_shots / total_attacks) if total_attacks > 0 else 0.0,
        }

