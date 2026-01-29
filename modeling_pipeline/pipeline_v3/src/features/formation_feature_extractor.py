"""
Formation Feature Extractor - Phase 3

Extracts formation and tactical features from formations.csv:
- Formation stability
- Tactical style classification
- Formation adaptation

Expected impact: -0.003 to -0.008 log loss
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FormationFeatureExtractor:
    """Extract formation and tactical features."""
    
    # Formation classifications
    DEFENSIVE_FORMATIONS = ['5-4-1', '4-5-1', '5-3-2', '5-4-1-0']
    ATTACKING_FORMATIONS = ['4-3-3', '3-4-3', '4-2-4', '3-5-2']
    BALANCED_FORMATIONS = ['4-4-2', '4-2-3-1', '4-1-4-1', '4-4-1-1']
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize extractor and load formations data."""
        self.data_dir = Path(data_dir)
        
        logger.info("Loading formations data...")
        try:
            self.formations = pd.read_csv(self.data_dir / 'formations.csv')
            logger.info(f"Loaded {len(self.formations)} formation records")
        except FileNotFoundError:
            logger.warning("formations.csv not found - will use defaults")
            self.formations = pd.DataFrame()
    
    def extract_formation_features(
        self,
        team_id: int,
        fixture_ids: List[int]
    ) -> Dict[str, float]:
        """
        Extract formation features for a team.
        
        Args:
            team_id: Team ID
            fixture_ids: List of fixture IDs
        
        Returns:
            Dictionary of formation features
        """
        if len(self.formations) == 0:
            return self._get_default_features()
        
        # Get formations for this team in these matches
        team_formations = self.formations[
            (self.formations['fixture_id'].isin(fixture_ids)) &
            (self.formations['team_id'] == team_id)
        ]
        
        if len(team_formations) == 0:
            return self._get_default_features()
        
        formations_list = team_formations['formation'].tolist()
        n_matches = len(fixture_ids)
        
        # Formation stability
        unique_formations = len(set(formations_list))
        most_common = max(set(formations_list), key=formations_list.count) if formations_list else '4-4-2'
        consistency = formations_list.count(most_common) / len(formations_list) if formations_list else 0
        
        # Tactical style
        defensive_count = sum(1 for f in formations_list if f in self.DEFENSIVE_FORMATIONS)
        attacking_count = sum(1 for f in formations_list if f in self.ATTACKING_FORMATIONS)
        balanced_count = sum(1 for f in formations_list if f in self.BALANCED_FORMATIONS)
        
        total = len(formations_list) if formations_list else 1
        
        return {
            'formation_consistency': consistency,
            'formation_changes': unique_formations / n_matches,
            'primary_formation_encoded': self._encode_formation(most_common),
            'defensive_formation_rate': defensive_count / total,
            'attacking_formation_rate': attacking_count / total,
            'balanced_formation_rate': balanced_count / total,
        }
    
    def _encode_formation(self, formation: str) -> float:
        """Encode formation as numeric value."""
        # Simple encoding: count defenders
        try:
            parts = formation.split('-')
            defenders = int(parts[0]) if parts else 4
            return defenders / 5.0  # Normalize to 0-1
        except:
            return 0.8  # Default (4 defenders)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default features when no data."""
        return {
            'formation_consistency': 0.5,
            'formation_changes': 0.0,
            'primary_formation_encoded': 0.8,
            'defensive_formation_rate': 0.0,
            'attacking_formation_rate': 0.0,
            'balanced_formation_rate': 0.0,
        }
