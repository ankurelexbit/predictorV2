"""
Feature Orchestrator

Coordinates all feature generation components to produce complete training data.
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

from .data_loader import HistoricalDataLoader
from .standings_calculator import SeasonAwareStandingsCalculator
from .elo_tracker import EloTracker
from .pillar1_fundamentals import Pillar1FundamentalsEngine
from .pillar2_modern_analytics import Pillar2ModernAnalyticsEngine
from .pillar3_hidden_edges import Pillar3HiddenEdgesEngine

logger = logging.getLogger(__name__)


class FeatureOrchestrator:
    """Orchestrate all feature generation."""
    
    def __init__(self, data_dir: str = 'data/csv'):
        """
        Initialize feature orchestrator.
        
        Args:
            data_dir: Directory containing CSV data files
        """
        logger.info("=" * 80)
        logger.info("INITIALIZING FEATURE ORCHESTRATOR")
        logger.info("=" * 80)
        
        # 1. Load data
        logger.info("\n1. Loading historical data...")
        self.data_loader = HistoricalDataLoader(data_dir)
        self.data_loader.load_all()
        
        # Validate data
        validation = self.data_loader.validate_data_completeness()
        logger.info(f"\nData validation: {validation}")
        
        # 2. Initialize standings calculator
        logger.info("\n2. Initializing standings calculator...")
        self.standings_calc = SeasonAwareStandingsCalculator(
            self.data_loader.fixtures_df
        )
        
        # 3. Initialize and build Elo tracker
        logger.info("\n3. Building Elo history...")
        self.elo_tracker = EloTracker(
            k_factor=32,
            home_advantage=35,
            initial_elo=1500,
            regression_factor=0.5
        )
        self.elo_tracker.process_all_fixtures(self.data_loader.fixtures_df)
        
        elo_stats = self.elo_tracker.get_elo_stats()
        logger.info(f"Elo stats: {elo_stats}")
        
        # 4. Initialize feature engines
        logger.info("\n4. Initializing feature engines...")
        self.pillar1_engine = Pillar1FundamentalsEngine(
            self.data_loader,
            self.standings_calc,
            self.elo_tracker
        )
        
        self.pillar2_engine = Pillar2ModernAnalyticsEngine(
            self.data_loader
        )
        
        self.pillar3_engine = Pillar3HiddenEdgesEngine(
            self.data_loader,
            self.elo_tracker
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE ORCHESTRATOR READY")
        logger.info("=" * 80 + "\n")
    
    def generate_features_for_fixture(
        self,
        fixture_id: int,
        as_of_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Generate all features for a single fixture.
        
        Args:
            fixture_id: Fixture ID
            as_of_date: Date cutoff (if None, uses fixture date - 1 hour)
            
        Returns:
            Dict with all features
        """
        # Get fixture info
        fixture = self.data_loader.get_fixture_info(fixture_id)
        
        if fixture is None:
            raise ValueError(f"Fixture {fixture_id} not found")
        
        # Extract fixture details
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        league_id = fixture['league_id']
        season_id = fixture['season_id']
        
        # Determine as_of_date (point-in-time cutoff)
        if as_of_date is None:
            fixture_date = fixture['starting_at']
            # Use 1 hour before match as cutoff
            cutoff_date = fixture_date - timedelta(hours=1)
            as_of_date = str(cutoff_date)
        
        # Generate features from all pillars
        features = {}
        
        # Pillar 1: Fundamentals (50 features)
        pillar1_features = self.pillar1_engine.generate_features(
            fixture_id, home_team_id, away_team_id,
            league_id, season_id, as_of_date
        )
        features.update(pillar1_features)
        
        # Pillar 2: Modern Analytics (60 features)
        pillar2_features = self.pillar2_engine.generate_features(
            fixture_id, home_team_id, away_team_id, as_of_date
        )
        features.update(pillar2_features)
        
        # Pillar 3: Hidden Edges (40 features)
        pillar3_features = self.pillar3_engine.generate_features(
            fixture_id, home_team_id, away_team_id,
            league_id, season_id, as_of_date
        )
        features.update(pillar3_features)
        
        return features
    
    def generate_all_training_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_matches_required: int = 5
    ) -> pd.DataFrame:
        """
        Generate training data for all fixtures.
        
        Args:
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            min_matches_required: Minimum matches each team needs before generating features
            
        Returns:
            DataFrame with all features and targets
        """
        logger.info("=" * 80)
        logger.info("GENERATING TRAINING DATA")
        logger.info("=" * 80)
        
        # Filter fixtures
        fixtures = self.data_loader.fixtures_df.copy()
        
        # Only finished matches
        fixtures = fixtures[fixtures['state'] == 'FT']
        
        # Date filters
        if start_date:
            fixtures = fixtures[fixtures['starting_at'] >= pd.to_datetime(start_date)]
        if end_date:
            fixtures = fixtures[fixtures['starting_at'] <= pd.to_datetime(end_date)]
        
        logger.info(f"\nProcessing {len(fixtures)} fixtures...")
        logger.info(f"Date range: {fixtures['starting_at'].min()} to {fixtures['starting_at'].max()}")
        
        training_data = []
        skipped = 0
        errors = 0
        
        for idx, fixture in tqdm(fixtures.iterrows(), total=len(fixtures), desc="Generating features"):
            try:
                fixture_id = fixture['fixture_id']
                home_team_id = fixture['home_team_id']
                away_team_id = fixture['away_team_id']
                
                # Check if teams have enough history
                as_of_date = str(fixture['starting_at'] - timedelta(hours=1))
                
                home_history = self.data_loader.get_fixtures_before_date(
                    home_team_id, as_of_date, n=None
                )
                away_history = self.data_loader.get_fixtures_before_date(
                    away_team_id, as_of_date, n=None
                )
                
                if len(home_history) < min_matches_required or len(away_history) < min_matches_required:
                    skipped += 1
                    continue
                
                # Generate features
                features = self.generate_features_for_fixture(fixture_id)
                
                # Add metadata
                features['fixture_id'] = fixture_id
                features['match_date'] = str(fixture['starting_at'])
                features['home_team_id'] = home_team_id
                features['away_team_id'] = away_team_id
                features['league_id'] = fixture['league_id']
                features['season_id'] = fixture['season_id']
                
                # Add target
                features['home_score'] = fixture.get('home_score', 0)
                features['away_score'] = fixture.get('away_score', 0)
                features['result'] = fixture.get('result', 'D')
                
                training_data.append(features)
                
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only log first 5 errors
                    logger.error(f"Error processing fixture {fixture_id}: {e}")
        
        logger.info(f"\n" + "=" * 80)
        logger.info(f"GENERATION COMPLETE")
        logger.info(f"=" * 80)
        logger.info(f"Total fixtures processed: {len(training_data)}")
        logger.info(f"Skipped (insufficient history): {skipped}")
        logger.info(f"Errors: {errors}")
        
        # Create DataFrame
        df = pd.DataFrame(training_data)
        
        if len(df) > 0:
            logger.info(f"\nTraining data shape: {df.shape}")
            logger.info(f"Features: {df.shape[1] - 9}")  # Subtract metadata columns
            logger.info(f"\nResult distribution:")
            logger.info(df['result'].value_counts())
            logger.info(f"\nMissing values: {df.isnull().sum().sum()}")
        
        return df
    
    def save_training_data(
        self,
        df: pd.DataFrame,
        output_path: str
    ):
        """
        Save training data to CSV.
        
        Args:
            df: Training data DataFrame
            output_path: Output file path
        """
        logger.info(f"\nSaving training data to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING DATA SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Output file: {output_path}")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Total columns: {df.shape[1]}")
        logger.info(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
        logger.info(f"Leagues: {df['league_id'].nunique()}")
        logger.info(f"Seasons: {df['season_id'].nunique()}")
        logger.info(f"Teams: {len(set(df['home_team_id'].unique()) | set(df['away_team_id'].unique()))}")
        logger.info("=" * 80 + "\n")
