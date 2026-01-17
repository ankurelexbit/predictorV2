#!/usr/bin/env python3
"""
Store Features to Database
==========================

This script loads the computed features from CSV and stores them in the Supabase database.

Usage:
    python store_features_to_db.py [--file features_cleaned.csv]
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATABASE_URL, PROCESSED_DATA_DIR
from utils import setup_logger
from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert

# Import the FeatureStore model
import importlib.util
spec = importlib.util.spec_from_file_location("data_storage", "02_data_storage.py")
data_storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_storage)
FeatureStore = data_storage.FeatureStore
Base = data_storage.Base

# Setup
logger = setup_logger("store_features")


class FeatureStorer:
    """Store features in database."""
    
    def __init__(self, db_url: str = DATABASE_URL):
        """Initialize with database connection."""
        self.engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.Session = sessionmaker(bind=self.engine)
    
    def clear_existing_features(self):
        """Clear existing features from the database."""
        session = self.Session()
        try:
            logger.info("Clearing existing features...")
            result = session.execute(delete(FeatureStore))
            session.commit()
            logger.info(f"Deleted {result.rowcount} existing feature rows")
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing features: {e}")
            raise
        finally:
            session.close()
    
    def store_features(self, features_df: pd.DataFrame, batch_size: int = 1000):
        """
        Store features in the database.
        
        Args:
            features_df: DataFrame with features
            batch_size: Number of rows to insert at once
        """
        session = self.Session()
        
        try:
            # Map DataFrame columns to database columns
            column_mapping = {
                'match_id': 'match_id',
                'home_elo': 'home_elo',
                'away_elo': 'away_elo',
                'elo_diff': 'elo_diff',
                'elo_prob_home': 'elo_prob_home',
                'elo_prob_draw': 'elo_prob_draw',
                'elo_prob_away': 'elo_prob_away',
                'home_form_3_points': 'home_form_3',
                'home_form_5_points': 'home_form_5',
                'home_form_10_points': 'home_form_10',
                'away_form_3_points': 'away_form_3',
                'away_form_5_points': 'away_form_5',
                'away_form_10_points': 'away_form_10',
                'home_form_5_gf': 'home_goals_scored_5',
                'home_form_5_ga': 'home_goals_conceded_5',
                'away_form_5_gf': 'away_goals_scored_5',
                'away_form_5_ga': 'away_goals_conceded_5',
                'home_rest_days': 'home_rest_days',
                'away_rest_days': 'away_rest_days',
                'h2h_home_wins': 'h2h_home_wins',
                'h2h_draws': 'h2h_draws',
                'h2h_away_wins': 'h2h_away_wins',
                'h2h_total': 'h2h_total_games',
                'home_position': 'home_position',
                'away_position': 'away_position',
                'position_diff': 'position_diff'
            }
            
            # Additional features to store separately
            extra_features_mapping = {
                'home_form_3_ppg': 'home_form_3_ppg',
                'home_form_5_ppg': 'home_form_5_ppg',
                'home_form_10_ppg': 'home_form_10_ppg',
                'away_form_3_ppg': 'away_form_3_ppg',
                'away_form_5_ppg': 'away_form_5_ppg',
                'away_form_10_ppg': 'away_form_10_ppg',
                'home_form_3_gf': 'home_form_3_gf',
                'home_form_3_ga': 'home_form_3_ga',
                'home_form_10_gf': 'home_form_10_gf',
                'home_form_10_ga': 'home_form_10_ga',
                'away_form_3_gf': 'away_form_3_gf',
                'away_form_3_ga': 'away_form_3_ga',
                'away_form_10_gf': 'away_form_10_gf',
                'away_form_10_ga': 'away_form_10_ga',
                'rest_diff': 'rest_diff',
                'h2h_home_win_rate': 'h2h_home_win_rate',
                'home_league_points': 'home_league_points',
                'away_league_points': 'away_league_points'
            }
            
            # Prepare data for insertion
            logger.info(f"Preparing {len(features_df)} features for insertion...")
            
            # Process in batches
            total_batches = (len(features_df) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(total_batches), desc="Storing features"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(features_df))
                batch_df = features_df.iloc[start_idx:end_idx]
                
                # Prepare batch data
                batch_data = []
                
                for _, row in batch_df.iterrows():
                    feature_dict = {'computed_at': datetime.now()}
                    
                    # Map basic features
                    for df_col, db_col in column_mapping.items():
                        if df_col in row.index:
                            value = row[df_col]
                            # Handle NaN values
                            if pd.isna(value):
                                feature_dict[db_col] = None
                            else:
                                feature_dict[db_col] = float(value) if db_col not in ['match_id', 'home_rest_days', 'away_rest_days', 'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_total_games', 'home_position', 'away_position', 'position_diff'] else int(value)
                    
                    batch_data.append(feature_dict)
                
                # Bulk insert
                if batch_data:
                    session.bulk_insert_mappings(FeatureStore, batch_data)
                
                # Commit every few batches to avoid memory issues
                if (batch_idx + 1) % 5 == 0:
                    session.commit()
            
            # Final commit
            session.commit()
            
            # Create extended features table for additional columns
            self._store_extended_features(session, features_df, extra_features_mapping, batch_size)
            
            logger.info(f"Successfully stored {len(features_df)} feature rows")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing features: {e}")
            raise
        finally:
            session.close()
    
    def _store_extended_features(self, session, features_df, mapping, batch_size):
        """Store extended features in a separate table (if needed)."""
        # For now, we'll just log that these features exist
        # In a production system, you might want to create an ExtendedFeatures table
        logger.info(f"Additional features available but not stored in main table: {list(mapping.keys())}")
    
    def verify_storage(self):
        """Verify features were stored correctly."""
        session = self.Session()
        
        try:
            # Count total features
            count = session.query(FeatureStore).count()
            logger.info(f"Total features in database: {count}")
            
            # Sample some features
            samples = session.query(FeatureStore).limit(5).all()
            
            print("\nSample features from database:")
            for sample in samples:
                print(f"  Match {sample.match_id}: Elo diff={sample.elo_diff:.1f}, "
                      f"Home form={sample.home_form_5}, Away form={sample.away_form_5}")
            
            # Check for missing critical features
            missing_elo = session.query(FeatureStore).filter(
                (FeatureStore.home_elo == None) | (FeatureStore.away_elo == None)
            ).count()
            
            if missing_elo > 0:
                logger.warning(f"Found {missing_elo} features with missing Elo ratings")
            
            return count
            
        finally:
            session.close()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Store features in database")
    parser.add_argument("--file", default="features_cleaned.csv", 
                       help="Features CSV file to load")
    parser.add_argument("--clear", action="store_true",
                       help="Clear existing features before storing")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for insertion")
    
    args = parser.parse_args()
    
    # Load features
    features_path = PROCESSED_DATA_DIR / args.file
    if not features_path.exists():
        # Try without _cleaned suffix
        features_path = PROCESSED_DATA_DIR / "features.csv"
    
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    
    print(f"\nLoaded {len(features_df)} features")
    print(f"Columns: {len(features_df.columns)}")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    
    # Initialize storer
    storer = FeatureStorer()
    
    # Clear existing if requested
    if args.clear:
        storer.clear_existing_features()
    
    # Store features
    print(f"\nStoring features to database...")
    storer.store_features(features_df, batch_size=args.batch_size)
    
    # Verify
    print("\nVerifying storage...")
    count = storer.verify_storage()
    
    if count == len(features_df):
        print(f"\n✅ Successfully stored all {count} features in database!")
    else:
        print(f"\n⚠️  Stored {count} features, expected {len(features_df)}")
    
    print("\nFeatures are now available in the database for:")
    print("  - Fast model inference")
    print("  - Real-time predictions")
    print("  - Historical analysis")


if __name__ == "__main__":
    main()