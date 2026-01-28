"""
Feature generation pipeline package.

This package contains all components for generating training features
from raw historical data with point-in-time correctness.
"""

from .data_loader import HistoricalDataLoader
from .standings_calculator import SeasonAwareStandingsCalculator
from .elo_tracker import EloTracker
from .pillar1_fundamentals import Pillar1FundamentalsEngine
from .pillar2_modern_analytics import Pillar2ModernAnalyticsEngine
from .pillar3_hidden_edges import Pillar3HiddenEdgesEngine
from .feature_orchestrator import FeatureOrchestrator

__all__ = [
    'HistoricalDataLoader',
    'SeasonAwareStandingsCalculator',
    'EloTracker',
    'Pillar1FundamentalsEngine',
    'Pillar2ModernAnalyticsEngine',
    'Pillar3HiddenEdgesEngine',
    'FeatureOrchestrator',
]
