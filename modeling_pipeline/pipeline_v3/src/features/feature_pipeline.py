"""
Feature Pipeline - Orchestrates all feature calculators.

Combines all individual feature calculators to build complete feature vectors.
"""
from typing import Dict, List, Optional
from datetime import datetime
import logging

from src.features.elo_calculator import EloCalculator
from src.features.derived_xg import DerivedXGCalculator
from src.features.form_calculator import FormCalculator
from src.features.h2h_calculator import H2HCalculator
from src.features.shot_analyzer import ShotAnalyzer
from src.features.defensive_metrics import DefensiveMetrics
from src.features.standings_calculator import StandingsCalculator


logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orchestrate all feature calculators to build complete feature vectors.
    
    Combines:
    - Elo ratings
    - Derived xG
    - Form metrics
    - H2H history
    - Shot analysis
    - Defensive metrics
    - Player features (when available)
    """
    
    def __init__(self):
        """Initialize all feature calculators."""
        self.elo_calc = EloCalculator()
        self.xg_calc = DerivedXGCalculator()
        self.form_calc = FormCalculator()
        self.h2h_calc = H2HCalculator()
        self.shot_analyzer = ShotAnalyzer()
        self.defensive_calc = DefensiveMetrics()
        self.standings_calc = StandingsCalculator()
        
        logger.info("Feature pipeline initialized")
    
    def calculate_features_for_match(
        self,
        home_team_id: int,
        away_team_id: int,
        match_date: datetime,
        home_matches: List[Dict],
        away_matches: List[Dict],
        h2h_matches: List[Dict],
        league_avg_elo: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all features for a match.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            match_date: Match date
            home_matches: Home team's recent matches (before match_date)
            away_matches: Away team's recent matches (before match_date)
            h2h_matches: Historical H2H matches
            league_avg_elo: Average Elo in league (optional)
        
        Returns:
            Dictionary with all features
        """
        features = {}
        
        # 1. Elo features
        try:
            elo_features = self.elo_calc.calculate_elo_features(
                home_team_id,
                away_team_id,
                match_date,
                league_avg_elo
            )
            features.update({f'home_{k}' if not k.startswith('home_') and not k.startswith('away_') and not k.startswith('elo_') else k: v 
                           for k, v in elo_features.items()})
        except Exception as e:
            logger.error(f"Error calculating Elo features: {e}")
        
        # 2. Form features
        try:
            home_form = self.form_calc.calculate_form_features(home_matches)
            away_form = self.form_calc.calculate_form_features(away_matches)
            
            features.update({f'home_{k}': v for k, v in home_form.items()})
            features.update({f'away_{k}': v for k, v in away_form.items()})
            
            # Streaks
            home_streaks = self.form_calc.calculate_streaks(home_matches)
            away_streaks = self.form_calc.calculate_streaks(away_matches)
            
            features.update({f'home_{k}': v for k, v in home_streaks.items()})
            features.update({f'away_{k}': v for k, v in away_streaks.items()})
            
            # Weighted form
            features['home_weighted_form_5'] = self.form_calc.calculate_weighted_form(home_matches)
            features['away_weighted_form_5'] = self.form_calc.calculate_weighted_form(away_matches)
            
            # Form trends
            features['home_points_trend_10'] = self.form_calc.calculate_form_trend(home_matches)
            features['away_points_trend_10'] = self.form_calc.calculate_form_trend(away_matches)
            
        except Exception as e:
            logger.error(f"Error calculating form features: {e}")
        
        # 3. xG features
        try:
            home_xg = self.xg_calc.calculate_xg_features(home_matches)
            away_xg = self.xg_calc.calculate_xg_features(away_matches)
            
            features.update({f'home_{k}': v for k, v in home_xg.items()})
            features.update({f'away_{k}': v for k, v in away_xg.items()})
            
            # xG differential matchup
            features['derived_xgd_matchup'] = features.get('home_derived_xgd', 0) - features.get('away_derived_xgd', 0)
            
        except Exception as e:
            logger.error(f"Error calculating xG features: {e}")
        
        # 4. H2H features
        try:
            h2h_features = self.h2h_calc.calculate_h2h_features(
                h2h_matches,
                home_team_id,
                away_team_id
            )
            features.update(h2h_features)
        except Exception as e:
            logger.error(f"Error calculating H2H features: {e}")
        
        # 5. Shot features
        try:
            home_shots = self.shot_analyzer.calculate_shot_features(home_matches)
            away_shots = self.shot_analyzer.calculate_shot_features(away_matches)
            
            features.update({f'home_{k}': v for k, v in home_shots.items()})
            features.update({f'away_{k}': v for k, v in away_shots.items()})
        except Exception as e:
            logger.error(f"Error calculating shot features: {e}")
        
        # 6. Defensive features
        try:
            home_def = self.defensive_calc.calculate_defensive_features(home_matches)
            away_def = self.defensive_calc.calculate_defensive_features(away_matches)
            
            features.update({f'home_{k}': v for k, v in home_def.items()})
            features.update({f'away_{k}': v for k, v in away_def.items()})
        except Exception as e:
            logger.error(f"Error calculating defensive features: {e}")
        
        logger.info(f"Calculated {len(features)} features for match {home_team_id} vs {away_team_id}")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names.
        
        Returns:
            List of feature names
        """
        # This would return all possible feature names
        # For now, return a placeholder
        return list(self.calculate_features_for_match(
            1, 2, datetime.now(), [], [], []
        ).keys())
    
    def validate_features(self, features: Dict[str, float]) -> bool:
        """
        Validate feature vector.
        
        Args:
            features: Feature dictionary
        
        Returns:
            True if valid
        """
        # Check for NaN or inf values
        for key, value in features.items():
            if value is None or (isinstance(value, float) and (value != value or abs(value) == float('inf'))):
                logger.warning(f"Invalid value for feature {key}: {value}")
                return False
        
        return True
