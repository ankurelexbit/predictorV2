#!/usr/bin/env python3
"""
Generate Complete Training Data

This script generates the complete 150-feature training dataset from raw historical data.
It implements proper point-in-time correctness and season-aware standings calculation.

Usage:
    python scripts/generate_complete_training_data.py \
        --data-dir data/csv \
        --output data/csv/training_data_complete.csv \
        --start-date 2020-01-01 \
        --end-date 2025-12-31
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.feature_orchestrator import FeatureOrchestrator


def setup_logging(log_file: str = 'feature_generation_complete.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate complete training data with all 150 features'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/csv',
        help='Directory containing CSV data files (default: data/csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/csv/training_data_complete.csv',
        help='Output CSV file path (default: data/csv/training_data_complete.csv)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date filter (ISO format: YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date filter (ISO format: YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--min-matches',
        type=int,
        default=5,
        help='Minimum matches required per team before generating features (default: 5)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='feature_generation_complete.log',
        help='Log file path (default: feature_generation_complete.log)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("COMPLETE FEATURE GENERATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Start date: {args.start_date or 'All'}")
    logger.info(f"End date: {args.end_date or 'All'}")
    logger.info(f"Min matches required: {args.min_matches}")
    logger.info(f"Log file: {args.log_file}")
    logger.info("=" * 80 + "\n")
    
    try:
        # Initialize orchestrator
        orchestrator = FeatureOrchestrator(data_dir=args.data_dir)
        
        # Generate training data
        training_df = orchestrator.generate_all_training_data(
            start_date=args.start_date,
            end_date=args.end_date,
            min_matches_required=args.min_matches
        )
        
        # Save to CSV
        orchestrator.save_training_data(training_df, args.output)
        
        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS!")
        logger.info("=" * 80)
        logger.info(f"Training data saved to: {args.output}")
        logger.info(f"Total samples: {len(training_df)}")
        logger.info(f"Total features: {training_df.shape[1] - 9}")  # Subtract metadata
        logger.info("=" * 80 + "\n")
        
        # Print feature summary
        logger.info("Feature Summary:")
        logger.info("-" * 80)
        
        # Count features by pillar
        pillar1_features = [col for col in training_df.columns if any(
            prefix in col for prefix in ['elo', 'position', 'points', 'wins', 'draws', 
                                         'goals_scored', 'goals_conceded', 'goal_diff',
                                         'h2h', 'home_advantage', 'in_top', 'in_bottom']
        )]
        
        pillar2_features = [col for col in training_df.columns if any(
            prefix in col for prefix in ['xg', 'xga', 'xgd', 'shots', 'ppda', 
                                         'tackles', 'interceptions', 'possession',
                                         'attacks', 'dangerous']
        )]
        
        pillar3_features = [col for col in training_df.columns if any(
            prefix in col for prefix in ['trend', 'weighted', 'streak', 'momentum',
                                         'opponent_elo', 'vs_strong', 'vs_weak',
                                         'lineup', 'player', 'days_since', 'rest',
                                         'derby', 'pressure', 'underdog']
        )]
        
        logger.info(f"Pillar 1 (Fundamentals): ~{len(pillar1_features)} features")
        logger.info(f"Pillar 2 (Modern Analytics): ~{len(pillar2_features)} features")
        logger.info(f"Pillar 3 (Hidden Edges): ~{len(pillar3_features)} features")
        logger.info(f"Total: {len(pillar1_features) + len(pillar2_features) + len(pillar3_features)} features")
        logger.info("-" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("ERROR!")
        logger.error("=" * 80)
        logger.error(f"An error occurred: {e}", exc_info=True)
        logger.error("=" * 80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
