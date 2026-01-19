"""
Generate predictions on historical data for threshold calibration.

This script loads historical matches and generates predictions using the trained
ensemble model, then saves them for threshold optimization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger("generate_predictions")

FEATURES_FILE = PROCESSED_DATA_DIR / "sportmonks_features.csv"


def load_ensemble_class():
    """Load the StackingEnsemble class from the module."""
    spec = importlib.util.spec_from_file_location(
        'ensemble',
        Path(__file__).parent / '07_model_ensemble.py'
    )
    ensemble_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ensemble_module)
    return ensemble_module.StackingEnsemble


def load_models():
    """Load all trained models."""
    logger.info("Loading trained models...")

    try:
        # Load the StackingEnsemble class
        StackingEnsemble = load_ensemble_class()

        # Create instance
        ensemble = StackingEnsemble()

        # Load base models
        logger.info("Loading base models...")
        elo_model = joblib.load(MODELS_DIR / 'elo_model.joblib')
        dc_model = joblib.load(MODELS_DIR / 'dixon_coles_model.joblib')
        xgb_model = joblib.load(MODELS_DIR / 'xgboost_model.joblib')

        # Add base models to ensemble
        ensemble.add_model('elo', elo_model)
        ensemble.add_model('dixon_coles', dc_model)
        ensemble.add_model('xgboost', xgb_model)

        # Load meta-model
        ensemble.load(MODELS_DIR / 'stacking_ensemble.joblib')

        logger.info("✅ Loaded stacking ensemble with all base models")
        return ensemble
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.exception(e)
        raise


def generate_predictions(df, model, feature_cols):
    """Generate predictions for dataframe."""
    logger.info(f"Generating predictions for {len(df)} matches...")

    # Generate probabilities - StackingEnsemble expects DataFrame
    probabilities = model.predict_proba(df)

    # Get predicted class (argmax of probabilities)
    predictions = np.argmax(probabilities, axis=1)

    # Create results dataframe
    results = pd.DataFrame({
        'date': df['date'],
        'home_team_name': df['home_team_name'],
        'away_team_name': df['away_team_name'],
        'target': df['target'],
        'predicted': predictions,
        'home_prob': probabilities[:, 0],
        'draw_prob': probabilities[:, 1],
        'away_prob': probabilities[:, 2]
    })

    return results


def main():
    """Generate predictions on last 180 days for calibration."""
    print("="*80)
    print("GENERATING HISTORICAL PREDICTIONS FOR CALIBRATION")
    print("="*80)

    # Load data
    logger.info(f"Loading data from {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE, parse_dates=['date'])
    logger.info(f"Loaded {len(df):,} total matches")

    # Filter to last 180 days
    cutoff_date = df['date'].max() - timedelta(days=180)
    df_recent = df[df['date'] >= cutoff_date].copy()
    logger.info(f"Filtered to last 180 days: {len(df_recent):,} matches")
    logger.info(f"Date range: {df_recent['date'].min().strftime('%Y-%m-%d')} to {df_recent['date'].max().strftime('%Y-%m-%d')}")

    # Load model
    model = load_models()

    # Get feature columns (exclude target and metadata)
    exclude_cols = ['fixture_id', 'date', 'season_id', 'season_name',
                    'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
                    'target', 'home_win', 'draw', 'away_win', 'home_goals', 'away_goals']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Generate predictions
    predictions_df = generate_predictions(df_recent, model, feature_cols)

    # Save results
    output_file = f'historical_predictions_180days_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    predictions_df.to_csv(output_file, index=False)
    logger.info(f"✅ Saved predictions to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"\nTotal Predictions: {len(predictions_df):,}")
    print(f"Date Range: {predictions_df['date'].min().strftime('%Y-%m-%d')} to {predictions_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Duration: {(predictions_df['date'].max() - predictions_df['date'].min()).days + 1} days")

    # Prediction distribution
    outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    predictions_df['predicted_outcome'] = predictions_df['predicted'].map(outcome_map)
    predictions_df['actual_outcome'] = predictions_df['target'].map(outcome_map)

    print(f"\n📊 PREDICTED DISTRIBUTION:")
    print(predictions_df['predicted_outcome'].value_counts())

    print(f"\n📊 ACTUAL DISTRIBUTION:")
    print(predictions_df['actual_outcome'].value_counts())

    # Accuracy
    accuracy = (predictions_df['predicted'] == predictions_df['target']).mean()
    print(f"\n🎯 Overall Accuracy: {accuracy:.1%}")

    print(f"\n✅ Ready for threshold optimization!")
    print(f"   Run: python optimize_betting_thresholds.py --data {output_file} --n-trials 500")

    return output_file


if __name__ == '__main__':
    output_file = main()
