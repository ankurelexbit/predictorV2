"""
Event Feature Extractor - Phase 2

Extracts match event features from events.csv:
- Goal timing patterns
- Card discipline
- Substitution patterns

Expected impact: -0.008 to -0.015 log loss
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EventFeatureExtractor:
    """Extract event-based features from match events."""
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize extractor and load events data."""
        self.data_dir = Path(data_dir)
        
        logger.info("Loading events data...")
        self.events = pd.read_csv(self.data_dir / 'events.csv')
        
        logger.info(f"Loaded {len(self.events)} event records")
    
    def extract_event_features(
        self,
        team_id: int,
        fixture_ids: List[int],
        window: int = 5
    ) -> Dict[str, float]:
        """
        Extract all event features for a team.
        
        Args:
            team_id: Team ID
            fixture_ids: List of fixture IDs (last N matches)
            window: Window size
        
        Returns:
            Dictionary of event features
        """
        features = {}
        
        # Goal timing features
        features.update(self._extract_goal_timing_features(team_id, fixture_ids))
        
        # Discipline features
        features.update(self._extract_discipline_features(team_id, fixture_ids))
        
        # Substitution features
        features.update(self._extract_substitution_features(team_id, fixture_ids))
        
        return features
    
    def _extract_goal_timing_features(
        self,
        team_id: int,
        fixture_ids: List[int]
    ) -> Dict[str, float]:
        """Extract goal timing patterns."""
        
        # Get all goals for this team in these matches
        team_goals = self.events[
            (self.events['fixture_id'].isin(fixture_ids)) &
            (self.events['team_id'] == team_id) &
            (self.events['type_id'] == 14)  # Goal
        ]
        
        # Get goals conceded (opponent goals)
        opponent_goals = self.events[
            (self.events['fixture_id'].isin(fixture_ids)) &
            (self.events['team_id'] != team_id) &
            (self.events['type_id'] == 14)
        ]
        
        n_matches = len(fixture_ids)
        
        if n_matches == 0:
            return self._get_default_goal_features()
        
        # Categorize goals by timing
        goals_0_15 = len(team_goals[team_goals['minute'] <= 15])
        goals_15_45 = len(team_goals[(team_goals['minute'] > 15) & (team_goals['minute'] <= 45)])
        goals_45_75 = len(team_goals[(team_goals['minute'] > 45) & (team_goals['minute'] <= 75)])
        goals_75_90 = len(team_goals[team_goals['minute'] > 75])
        
        # Conceded goals by timing
        conceded_75_90 = len(opponent_goals[opponent_goals['minute'] > 75])
        
        return {
            'goals_first_15min': goals_0_15 / n_matches,
            'goals_15_45min': goals_15_45 / n_matches,
            'goals_second_half': (goals_45_75 + goals_75_90) / n_matches,
            'goals_last_15min': goals_75_90 / n_matches,
            'late_goals_conceded': conceded_75_90 / n_matches,
            'early_pressure': 1 if goals_0_15 / max(len(team_goals), 1) > 0.3 else 0,
        }
    
    def _extract_discipline_features(
        self,
        team_id: int,
        fixture_ids: List[int]
    ) -> Dict[str, float]:
        """Extract card discipline features."""
        
        # Yellow cards (type_id = 15)
        yellow_cards = self.events[
            (self.events['fixture_id'].isin(fixture_ids)) &
            (self.events['team_id'] == team_id) &
            (self.events['type_id'] == 15)
        ]
        
        # Red cards (type_id = 16)
        red_cards = self.events[
            (self.events['fixture_id'].isin(fixture_ids)) &
            (self.events['team_id'] == team_id) &
            (self.events['type_id'] == 16)
        ]
        
        n_matches = len(fixture_ids)
        
        if n_matches == 0:
            return self._get_default_discipline_features()
        
        # Early cards (0-30 min)
        early_yellows = len(yellow_cards[yellow_cards['minute'] <= 30])
        
        return {
            'yellow_cards': len(yellow_cards) / n_matches,
            'red_cards': len(red_cards) / n_matches,
            'cards_per_match': (len(yellow_cards) + len(red_cards)) / n_matches,
            'early_cards': early_yellows / n_matches,
            'discipline_score': 1 - min((len(yellow_cards) + len(red_cards) * 3) / (n_matches * 5), 1),
            'has_discipline_issue': 1 if len(red_cards) > 0 or len(yellow_cards) / n_matches > 2.5 else 0,
        }
    
    def _extract_substitution_features(
        self,
        team_id: int,
        fixture_ids: List[int]
    ) -> Dict[str, float]:
        """Extract substitution pattern features."""
        
        # Substitutions (type_id = 18)
        subs = self.events[
            (self.events['fixture_id'].isin(fixture_ids)) &
            (self.events['team_id'] == team_id) &
            (self.events['type_id'] == 18)
        ]
        
        n_matches = len(fixture_ids)
        
        if n_matches == 0:
            return self._get_default_substitution_features()
        
        # Categorize by timing
        early_subs = len(subs[subs['minute'] < 60])  # Tactical/injury
        late_subs = len(subs[subs['minute'] >= 75])  # Time wasting/fresh legs
        
        return {
            'subs_before_60min': early_subs / n_matches,
            'subs_after_75min': late_subs / n_matches,
            'avg_subs_per_match': len(subs) / n_matches,
            'tactical_flexibility': 1 if early_subs / max(len(subs), 1) > 0.3 else 0,
        }
    
    def _get_default_goal_features(self) -> Dict[str, float]:
        """Default goal features when no data."""
        return {
            'goals_first_15min': 0.0,
            'goals_15_45min': 0.0,
            'goals_second_half': 0.0,
            'goals_last_15min': 0.0,
            'late_goals_conceded': 0.0,
            'early_pressure': 0.0,
        }
    
    def _get_default_discipline_features(self) -> Dict[str, float]:
        """Default discipline features when no data."""
        return {
            'yellow_cards': 0.0,
            'red_cards': 0.0,
            'cards_per_match': 0.0,
            'early_cards': 0.0,
            'discipline_score': 1.0,
            'has_discipline_issue': 0.0,
        }
    
    def _get_default_substitution_features(self) -> Dict[str, float]:
        """Default substitution features when no data."""
        return {
            'subs_before_60min': 0.0,
            'subs_after_75min': 0.0,
            'avg_subs_per_match': 0.0,
            'tactical_flexibility': 0.0,
        }
