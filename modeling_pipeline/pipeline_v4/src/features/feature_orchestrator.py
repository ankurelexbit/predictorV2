"""
Feature Orchestrator for V4 Pipeline.

Coordinates all 3 feature pillars to generate complete 150-feature dataset.
"""
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
import logging
from pathlib import Path

from src.data.json_loader import JSONDataLoader
from src.features.standings_calculator import StandingsCalculator
from src.features.elo_calculator import EloCalculator
from src.features.pillar1_fundamentals import Pillar1FundamentalsEngine
from src.features.pillar2_modern_analytics import Pillar2ModernAnalyticsEngine
from src.features.pillar3_hidden_edges import Pillar3HiddenEdgesEngine

logger = logging.getLogger(__name__)


class FeatureOrchestrator:
    """
    Orchestrate all feature generation across 3 pillars.
    
    Generates 150 features total:
    - Pillar 1: Fundamentals (50 features)
    - Pillar 2: Modern Analytics (60 features)
    - Pillar 3: Hidden Edges (40 features)
    """
    
    def __init__(self, data_dir: str = 'data/historical'):
        """
        Initialize feature orchestrator.
        
        Args:
            data_dir: Directory containing historical data
        """
        logger.info("Initializing FeatureOrchestrator...")
        
        # Initialize data loader
        self.data_loader = JSONDataLoader(data_dir)
        
        # Load all fixtures
        logger.info("Loading all fixtures...")
        self.fixtures_df = self.data_loader.load_all_fixtures()
        logger.info(f"Loaded {len(self.fixtures_df)} fixtures")
        
        # Initialize calculators
        logger.info("Initializing calculators...")
        self.standings_calc = StandingsCalculator()
        self.elo_calc = EloCalculator()
        
        # Calculate Elo history
        logger.info("Calculating Elo history...")
        self.elo_calc.calculate_elo_history(self.fixtures_df)
        logger.info("Elo history calculated")
        
        # Initialize pillar engines
        logger.info("Initializing pillar engines...")
        self.pillar1 = Pillar1FundamentalsEngine(
            self.data_loader, self.standings_calc, self.elo_calc
        )
        self.pillar2 = Pillar2ModernAnalyticsEngine(self.data_loader)
        self.pillar3 = Pillar3HiddenEdgesEngine(
            self.data_loader, self.standings_calc, self.elo_calc
        )
        
        logger.info("✅ FeatureOrchestrator initialized successfully")
    
    def generate_features_for_fixture(
        self,
        fixture_id: int,
        as_of_date: Optional[datetime] = None
    ) -> Dict:
        """
        Generate all 150 features for a single fixture.
        
        Args:
            fixture_id: Fixture ID
            as_of_date: Optional date override (defaults to fixture date)
            
        Returns:
            Dict with all features
        """
        # Get fixture
        fixture = self.data_loader.get_fixture(fixture_id)
        
        # Use fixture date if as_of_date not provided
        if as_of_date is None:
            as_of_date = fixture['starting_at']
        
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        season_id = fixture['season_id']
        league_id = fixture['league_id']
        
        logger.debug(f"Generating features for fixture {fixture_id}")
        
        # Generate features from all 3 pillars
        features = {}
        
        # Pillar 1: Fundamentals (50 features)
        features.update(self.pillar1.generate_features(
            home_team_id, away_team_id, season_id, league_id, as_of_date, self.fixtures_df
        ))
        
        # Pillar 2: Modern Analytics (60 features)
        features.update(self.pillar2.generate_features(
            home_team_id, away_team_id, as_of_date
        ))
        
        # Pillar 3: Hidden Edges (40 features)
        features.update(self.pillar3.generate_features(
            home_team_id, away_team_id, season_id, league_id, as_of_date, self.fixtures_df
        ))
        
        # Add metadata
        features['fixture_id'] = fixture_id
        features['home_team_id'] = home_team_id
        features['away_team_id'] = away_team_id
        features['season_id'] = season_id
        features['league_id'] = league_id
        features['match_date'] = as_of_date
        features['home_score'] = fixture.get('home_score')
        features['away_score'] = fixture.get('away_score')
        features['result'] = fixture.get('result')
        
        return features
    
    def generate_training_dataset(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        league_id: Optional[int] = None,
        output_file: str = 'data/training_data.csv',
        max_fixtures: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate complete training dataset.
        
        Args:
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            league_id: Optional league filter
            output_file: Output CSV file path
            max_fixtures: Optional limit on number of fixtures
            
        Returns:
            DataFrame with all features
        """
        logger.info("=" * 80)
        logger.info("GENERATING TRAINING DATASET")
        logger.info("=" * 80)
        
        # Filter fixtures
        fixtures = self.fixtures_df.copy()
        
        # Only completed fixtures
        fixtures = fixtures[fixtures['result'].notna()]
        
        if start_date:
            fixtures = fixtures[fixtures['starting_at'] >= pd.to_datetime(start_date)]
            logger.info(f"Filtered by start_date: {start_date}")
        
        if end_date:
            fixtures = fixtures[fixtures['starting_at'] <= pd.to_datetime(end_date)]
            logger.info(f"Filtered by end_date: {end_date}")
        
        if league_id:
            fixtures = fixtures[fixtures['league_id'] == league_id]
            logger.info(f"Filtered by league_id: {league_id}")
        
        if max_fixtures:
            fixtures = fixtures.head(max_fixtures)
            logger.info(f"Limited to {max_fixtures} fixtures")
        
        logger.info(f"Processing {len(fixtures)} fixtures...")
        
        # Generate features for each fixture
        all_features = []
        
        for idx, (_, fixture) in enumerate(fixtures.iterrows(), 1):
            if idx % 100 == 0:
                logger.info(f"  Processed {idx}/{len(fixtures)} fixtures...")
            
            try:
                features = self.generate_features_for_fixture(
                    fixture['id'],
                    as_of_date=fixture['starting_at']
                )
                all_features.append(features)
            except Exception as e:
                logger.warning(f"  Failed to generate features for fixture {fixture['id']}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        logger.info(f"\n✅ Generated {len(df)} feature vectors")
        logger.info(f"   Total features: {len(df.columns)}")
        logger.info(f"   Feature columns: {len([c for c in df.columns if c not in ['fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id', 'match_date', 'home_score', 'away_score', 'result']])}")
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"   Saved to: {output_path}")
        
        logger.info("=" * 80)
        
        return df
    
    def get_feature_count(self) -> Dict[str, int]:
        """Get feature count breakdown by pillar."""
        # Generate features for a sample fixture to count
        sample_fixture = self.fixtures_df[self.fixtures_df['result'].notna()].iloc[0]
        features = self.generate_features_for_fixture(sample_fixture['id'])
        
        # Count features by pillar (approximate based on naming)
        pillar1_features = [f for f in features.keys() if any(x in f for x in ['elo', 'position', 'points', 'wins', 'draws', 'goals', 'h2h', 'home_'])]
        pillar2_features = [f for f in features.keys() if any(x in f for x in ['xg', 'shot', 'tackle', 'ppda', 'attack', 'possession'])]
        pillar3_features = [f for f in features.keys() if any(x in f for x in ['trend', 'streak', 'weighted', 'opponent', 'player', 'rest', 'derby'])]
        
        metadata_fields = ['fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id', 'match_date', 'home_score', 'away_score', 'result']
        
        return {
            'total': len(features),
            'pillar1_approx': len(pillar1_features),
            'pillar2_approx': len(pillar2_features),
            'pillar3_approx': len(pillar3_features),
            'metadata': len(metadata_fields),
            'feature_columns': len(features) - len(metadata_fields)
        }
