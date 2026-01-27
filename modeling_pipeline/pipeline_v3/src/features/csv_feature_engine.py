"""
CSV-based Feature Engine - Main orchestrator for feature generation.

This module coordinates all feature calculators to generate point-in-time
correct features for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

from .csv_elo_calculator import CSVEloCalculator
from .csv_form_calculator import CSVFormCalculator
from .csv_derived_xg import CSVDerivedXGCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVFeatureEngine:
    """Main feature engineering engine using CSV data."""
    
    def __init__(self, data_dir: str = 'data/csv'):
        """
        Initialize feature engine.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        
        # Load data
        logger.info("Loading CSV data...")
        self.fixtures = pd.read_csv(self.data_dir / 'fixtures.csv')
        self.statistics = pd.read_csv(self.data_dir / 'statistics.csv')
        
        # Initialize calculators
        self.elo_calc = CSVEloCalculator()
        self.form_calc = CSVFormCalculator()
        self.xg_calc = CSVDerivedXGCalculator()
        
        # Calculate Elo history (one-time computation)
        logger.info("Calculating Elo history...")
        self.elo_history = self.elo_calc.calculate_elo_history(self.fixtures)
        
        # Add xG to statistics (one-time computation)
        logger.info("Calculating derived xG...")
        self.statistics = self.xg_calc.calculate_xg(self.statistics)
        
        logger.info("✅ Feature engine initialized")
    
    def generate_features_for_fixture(
        self,
        fixture_id: int,
        as_of_date: str
    ) -> Dict:
        """
        Generate all features for a single fixture.
        
        CRITICAL: Uses only data from BEFORE as_of_date to avoid data leakage.
        
        Args:
            fixture_id: Fixture ID to generate features for
            as_of_date: The fixture's date (use data BEFORE this)
        
        Returns:
            Dictionary of features
        """
        # Get fixture details
        fixture = self.fixtures[self.fixtures['fixture_id'] == fixture_id].iloc[0]
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        
        features = {
            'fixture_id': fixture_id,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'starting_at': as_of_date,
            'league_id': fixture['league_id'],
        }
        
        # === ELO FEATURES ===
        home_elo = self.elo_calc.get_elo_at_date(self.elo_history, home_team_id, as_of_date)
        away_elo = self.elo_calc.get_elo_at_date(self.elo_history, away_team_id, as_of_date)
        
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = home_elo - away_elo
        
        # === FORM FEATURES (Last 5 matches) ===
        home_form = self.form_calc.calculate_team_form(
            self.fixtures, home_team_id, as_of_date, n_matches=5
        )
        away_form = self.form_calc.calculate_team_form(
            self.fixtures, away_team_id, as_of_date, n_matches=5
        )
        
        # Add home form features
        for key, value in home_form.items():
            features[f'home_form_{key}'] = value
        
        # Add away form features
        for key, value in away_form.items():
            features[f'away_form_{key}'] = value
        
        # Form differentials
        features['form_points_diff'] = home_form['points'] - away_form['points']
        features['form_goals_diff'] = home_form['goals_scored'] - away_form['goals_scored']
        
        # === WEIGHTED FORM ===
        home_weighted_form = self.form_calc.calculate_weighted_form(
            self.fixtures, home_team_id, as_of_date, n_matches=10
        )
        away_weighted_form = self.form_calc.calculate_weighted_form(
            self.fixtures, away_team_id, as_of_date, n_matches=10
        )
        
        features['home_weighted_form'] = home_weighted_form
        features['away_weighted_form'] = away_weighted_form
        features['weighted_form_diff'] = home_weighted_form - away_weighted_form
        
        # === DERIVED XG FEATURES ===
        home_xg = self.xg_calc.calculate_rolling_xg(
            self.statistics, self.fixtures, home_team_id, as_of_date, n_matches=5
        )
        away_xg = self.xg_calc.calculate_rolling_xg(
            self.statistics, self.fixtures, away_team_id, as_of_date, n_matches=5
        )
        
        features['home_xG_avg'] = home_xg['xG_avg']
        features['home_xGA_avg'] = home_xg['xGA_avg']
        features['home_xG_diff_avg'] = home_xg['xG_diff_avg']
        
        features['away_xG_avg'] = away_xg['xG_avg']
        features['away_xGA_avg'] = away_xg['xGA_avg']
        features['away_xG_diff_avg'] = away_xg['xG_diff_avg']
        
        # === TARGET VARIABLE ===
        if 'result' in fixture and not pd.isna(fixture['result']):
            features['result'] = fixture['result']
            
            # Encode as numeric for modeling
            if fixture['result'] == 'H':
                features['target_home_win'] = 1
                features['target_draw'] = 0
                features['target_away_win'] = 0
            elif fixture['result'] == 'D':
                features['target_home_win'] = 0
                features['target_draw'] = 1
                features['target_away_win'] = 0
            else:  # 'A'
                features['target_home_win'] = 0
                features['target_draw'] = 0
                features['target_away_win'] = 1
        
        return features
    
    def generate_training_dataset(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_file: str = 'data/csv/training_data.csv'
    ) -> pd.DataFrame:
        """
        Generate complete training dataset with point-in-time correct features.
        
        Args:
            start_date: Start date for training data (optional)
            end_date: End date for training data (optional)
            output_file: Where to save the training data
        
        Returns:
            DataFrame with all features
        """
        logger.info("=" * 80)
        logger.info("GENERATING TRAINING DATASET")
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
        
        # Only include completed fixtures with results
        fixtures_to_process = fixtures_to_process[
            fixtures_to_process['result'].notna()
        ]
        
        logger.info(f"Processing {len(fixtures_to_process)} fixtures...")
        
        # Generate features for each fixture
        all_features = []
        
        for _, fixture in tqdm(fixtures_to_process.iterrows(), total=len(fixtures_to_process)):
            try:
                features = self.generate_features_for_fixture(
                    fixture['fixture_id'],
                    fixture['starting_at']
                )
                all_features.append(features)
            except Exception as e:
                logger.error(f"Error processing fixture {fixture['fixture_id']}: {e}")
                continue
        
        # Create DataFrame
        training_df = pd.DataFrame(all_features)
        
        # Save to CSV
        training_df.to_csv(output_file, index=False)
        
        logger.info("=" * 80)
        logger.info("TRAINING DATASET COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(training_df):,}")
        logger.info(f"Features: {len(training_df.columns)}")
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 80)
        
        return training_df


def main():
    """Generate training dataset."""
    engine = CSVFeatureEngine()
    
    # Generate full training dataset
    training_df = engine.generate_training_dataset()
    
    print("\nSample features:")
    print(training_df.head())
    
    print("\nFeature columns:")
    print(training_df.columns.tolist())
    
    print("\nTarget distribution:")
    print(training_df['result'].value_counts())


if __name__ == "__main__":
    main()
