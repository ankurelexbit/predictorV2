"""
Enhanced CSV Feature Engine - Complete Feature Engineering Pipeline

Integrates ALL feature extractors (Phases 0-6):
- Phase 0: V3 Baseline (173 features)
- Phase 1: Player Statistics (49 features)
- Phase 2: Match Events (16 features per team = 32)
- Phase 3: Formations (6 features per team = 12)
- Phase 4: Injuries (8 features per team = 16)
- Phase 5: Betting Odds (6 features)
- Phase 6: Temporal (4 features per team = 8)

Total: ~290 features before selection
Expected impact: -0.055 to -0.063 log loss improvement
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import logging

from .comprehensive_csv_feature_engine import ComprehensiveCSVFeatureEngine
from .player_statistics_extractor import PlayerStatisticsExtractor
from .event_feature_extractor import EventFeatureExtractor
from .formation_feature_extractor import FormationFeatureExtractor
from .injury_feature_extractor import InjuryFeatureExtractor
from .odds_feature_extractor import OddsFeatureExtractor
from .temporal_feature_extractor import TemporalFeatureExtractor
from .feature_validation import validate_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedCSVFeatureEngine(ComprehensiveCSVFeatureEngine):
    """
    Complete feature engineering pipeline with all 6 phases integrated.
    
    Features breakdown:
    - V3 Baseline: 173
    - Player Stats: 49
    - Events: 32
    - Formations: 12
    - Injuries: 16
    - Odds: 6
    - Temporal: 8
    Total: ~296 features
    """
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize complete feature engine with all extractors."""
        # Initialize parent
        super().__init__(data_dir)
        
        # Initialize all extractors
        logger.info("Initializing all feature extractors...")
        
        self.player_extractor = PlayerStatisticsExtractor(data_dir)
        self.event_extractor = EventFeatureExtractor(data_dir)
        self.formation_extractor = FormationFeatureExtractor(data_dir)
        self.injury_extractor = InjuryFeatureExtractor(data_dir)
        self.odds_extractor = OddsFeatureExtractor(data_dir)
        self.temporal_extractor = TemporalFeatureExtractor(data_dir)
        
        logger.info("✅ Complete feature engine initialized with all phases")
    
    def generate_features_for_fixture(
        self,
        fixture_id: int,
        as_of_date: str
    ) -> Dict:
        """
        Generate ALL features from all phases.
        
        Args:
            fixture_id: Fixture ID
            as_of_date: Fixture date (use data BEFORE this)
        
        Returns:
            Dictionary with ~296 features
        """
        # Get base V3 features (173)
        features = super().generate_features_for_fixture(fixture_id, as_of_date)
        
        # Get fixture details
        fixture = self.fixtures[self.fixtures['fixture_id'] == fixture_id].iloc[0]
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        
        # Phase 1: Player Statistics (49 features)
        features.update(self._get_player_statistics_features(fixture_id, as_of_date))
        
        # Phase 2: Match Events (32 features)
        features.update(self._get_event_features(home_team_id, away_team_id, as_of_date))
        
        # Phase 3: Formations (12 features)
        features.update(self._get_formation_features(home_team_id, away_team_id, as_of_date))
        
        # Phase 4: Injuries (16 features) - CRITICAL
        features.update(self._get_injury_features(home_team_id, away_team_id, as_of_date))
        
        # Phase 5: Betting Odds (6 features)
        features.update(self._get_odds_features(fixture_id, as_of_date))
        
        # Phase 6: Temporal (8 features)
        features.update(self._get_temporal_features(home_team_id, away_team_id, fixture_id, as_of_date))
        
        # Validate features for data leakage
        is_valid, messages = validate_features(features, strict=False)
        if not is_valid:
            logger.warning(f"Feature validation warnings for fixture {fixture_id}:")
            for msg in messages[:5]:  # Show first 5
                logger.warning(f"  {msg}")
        
        return features
    
    def _get_player_statistics_features(self, fixture_id: int, as_of_date: str) -> Dict:
        """Extract player statistics (Phase 1)."""
        fixture = self.fixtures[self.fixtures['fixture_id'] == fixture_id].iloc[0]
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        
        features = {}
        
        for team_id, prefix in [(home_team_id, 'home'), (away_team_id, 'away')]:
            for window in [5]:  # Use L5 only to reduce features
                fixture_ids = self.player_extractor.get_team_matches_before_date(
                    team_id=team_id,
                    before_date=as_of_date,
                    fixtures_df=self.fixtures,
                    limit=window
                )
                
                if len(fixture_ids) > 0:
                    stats = self.player_extractor.extract_team_player_stats(
                        team_id=team_id,
                        fixture_ids=fixture_ids,
                        window=window
                    )
                    
                    for stat_name, value in stats.items():
                        features[f'{prefix}_{stat_name}_{window}'] = value
        
        # Add relative features
        if 'home_rating_avg_5' in features and 'away_rating_avg_5' in features:
            features['player_rating_diff_5'] = features['home_rating_avg_5'] - features['away_rating_avg_5']
        
        return features
    
    def _get_event_features(self, home_team_id: int, away_team_id: int, as_of_date: str) -> Dict:
        """Extract event features (Phase 2)."""
        features = {}
        
        for team_id, prefix in [(home_team_id, 'home'), (away_team_id, 'away')]:
            fixture_ids = self.player_extractor.get_team_matches_before_date(
                team_id=team_id,
                before_date=as_of_date,
                fixtures_df=self.fixtures,
                limit=5
            )
            
            if len(fixture_ids) > 0:
                event_features = self.event_extractor.extract_event_features(
                    team_id=team_id,
                    fixture_ids=fixture_ids
                )
                
                for stat_name, value in event_features.items():
                    features[f'{prefix}_{stat_name}'] = value
        
        return features
    
    def _get_formation_features(self, home_team_id: int, away_team_id: int, as_of_date: str) -> Dict:
        """Extract formation features (Phase 3)."""
        features = {}
        
        for team_id, prefix in [(home_team_id, 'home'), (away_team_id, 'away')]:
            fixture_ids = self.player_extractor.get_team_matches_before_date(
                team_id=team_id,
                before_date=as_of_date,
                fixtures_df=self.fixtures,
                limit=5
            )
            
            if len(fixture_ids) > 0:
                formation_features = self.formation_extractor.extract_formation_features(
                    team_id=team_id,
                    fixture_ids=fixture_ids
                )
                
                for stat_name, value in formation_features.items():
                    features[f'{prefix}_{stat_name}'] = value
        
        return features
    
    def _get_injury_features(self, home_team_id: int, away_team_id: int, as_of_date: str) -> Dict:
        """Extract injury features (Phase 4) - CRITICAL."""
        features = {}
        
        for team_id, prefix in [(home_team_id, 'home'), (away_team_id, 'away')]:
            injury_features = self.injury_extractor.extract_injury_features(
                team_id=team_id,
                as_of_date=as_of_date
            )
            
            for stat_name, value in injury_features.items():
                features[f'{prefix}_{stat_name}'] = value
        
        return features
    
    def _get_odds_features(self, fixture_id: int, as_of_date: str) -> Dict:
        """Extract odds features (Phase 5)."""
        return self.odds_extractor.extract_odds_features(
            fixture_id=fixture_id,
            as_of_date=as_of_date
        )
    
    def _get_temporal_features(self, home_team_id: int, away_team_id: int, fixture_id: int, as_of_date: str) -> Dict:
        """Extract temporal features (Phase 6)."""
        features = {}
        
        for team_id, prefix in [(home_team_id, 'home'), (away_team_id, 'away')]:
            temporal_features = self.temporal_extractor.extract_temporal_features(
                team_id=team_id,
                fixture_id=fixture_id,
                as_of_date=as_of_date
            )
            
            for stat_name, value in temporal_features.items():
                features[f'{prefix}_{stat_name}'] = value
        
        return features
