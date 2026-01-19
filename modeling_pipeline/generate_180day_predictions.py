"""
Generate predictions for last 180 days using trained models.
Simplified version that uses the model classes directly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR, MODELS_DIR
from utils import setup_logger
import importlib.util

logger = setup_logger("gen_predictions")


def load_model_class(file_path, class_name):
    """Dynamically load a class from a Python file."""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


# Load model classes
EloProbabilityModel = load_model_class('04_model_baseline_elo.py', 'EloProbabilityModel')
DixonColesModel = load_model_class('05_model_dixon_coles.py', 'DixonColesModel')
XGBoostFootballModel = load_model_class('06_model_xgboost.py', 'XGBoostFootballModel')


def load_models():
    """Load trained model classes properly."""
    logger.info("Loading trained models...")

    # Load Elo model
    elo = EloProbabilityModel()
    elo_path = MODELS_DIR / 'elo_model.joblib'
    if elo_path.exists():
        elo.load(elo_path)
    logger.info("✅ Loaded Elo model")

    # Load Dixon-Coles model (saved as CalibratedDixonColes)
    CalibratedDixonColes = load_model_class('05_model_dixon_coles.py', 'CalibratedDixonColes')
    dc_path = MODELS_DIR / 'dixon_coles_model.joblib'
    if dc_path.exists():
        dc_data = joblib.load(dc_path)
        # Reconstruct the calibrated model
        base_dc = DixonColesModel()
        base_dc.attack = dc_data['base_model_data']['attack']
        base_dc.defense = dc_data['base_model_data']['defense']
        base_dc.home_adv = dc_data['base_model_data']['home_adv']
        base_dc.rho = dc_data['base_model_data']['rho']
        base_dc.team_to_idx = dc_data['base_model_data']['team_to_idx']
        base_dc.idx_to_team = dc_data['base_model_data']['idx_to_team']
        base_dc.time_decay = dc_data['base_model_data']['time_decay']
        base_dc.max_goals = dc_data['base_model_data']['max_goals']
        base_dc.is_fitted = dc_data['base_model_data']['is_fitted']

        dc = CalibratedDixonColes(base_dc)
        dc.calibrators = dc_data['calibrators']
        dc.is_calibrated = dc_data['is_calibrated']
    else:
        dc = DixonColesModel()
    logger.info("✅ Loaded Dixon-Coles model")

    # Load XGBoost model
    xgb = XGBoostFootballModel()
    xgb_path = MODELS_DIR / 'xgboost_model.joblib'
    if xgb_path.exists():
        xgb.load(xgb_path)
    logger.info("✅ Loaded XGBoost model")

    return {'elo': elo, 'dixon_coles': dc, 'xgboost': xgb}


def ensemble_predictions(models, df, weights={'elo': 0.2, 'dixon_coles': 0.3, 'xgboost': 0.5}):
    """Generate ensemble predictions."""
    logger.info(f"Generating predictions for {len(df)} matches...")

    # Get predictions from each model
    elo_probs = models['elo'].predict_proba(df)
    dc_probs = models['dixon_coles'].predict_proba(df)
    xgb_probs = models['xgboost'].predict_proba(df)

    # Weighted average
    ensemble_probs = (
        weights['elo'] * elo_probs +
        weights['dixon_coles'] * dc_probs +
        weights['xgboost'] * xgb_probs
    )

    # Normalize to sum to 1
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

    return ensemble_probs


def main():
    """Generate predictions for last 180 days."""
    print("="*80)
    print("GENERATING 180-DAY PREDICTIONS FOR CALIBRATION")
    print("="*80)

    # Load data
    data_file = PROCESSED_DATA_DIR / "sportmonks_features.csv"
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file, parse_dates=['date'])
    logger.info(f"Loaded {len(df):,} total matches")

    # Filter to last 180 days
    cutoff_date = df['date'].max() - timedelta(days=180)
    df_recent = df[df['date'] >= cutoff_date].copy()
    logger.info(f"Filtered to {len(df_recent):,} matches from {df_recent['date'].min().date()} to {df_recent['date'].max().date()}")

    # Load models
    models = load_models()

    # Generate predictions
    probabilities = ensemble_predictions(models, df_recent)
    predictions = np.argmax(probabilities, axis=1)

    # Create output dataframe
    results = pd.DataFrame({
        'date': df_recent['date'],
        'home_team_name': df_recent['home_team_name'],
        'away_team_name': df_recent['away_team_name'],
        'target': df_recent['target'],
        'predicted': predictions,
        'home_prob': probabilities[:, 0],
        'draw_prob': probabilities[:, 1],
        'away_prob': probabilities[:, 2]
    })

    # Save results
    output_file = f'historical_predictions_180days_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results.to_csv(output_file, index=False)
    logger.info(f"✅ Saved to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal Predictions: {len(results):,}")
    print(f"Date Range: {results['date'].min().date()} to {results['date'].max().date()}")
    print(f"Duration: {(results['date'].max() - results['date'].min()).days + 1} days")

    outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    results['predicted_outcome'] = results['predicted'].map(outcome_map)
    results['actual_outcome'] = results['target'].map(outcome_map)

    print(f"\n📊 Predicted Distribution:")
    print(results['predicted_outcome'].value_counts())

    print(f"\n📊 Actual Distribution:")
    print(results['actual_outcome'].value_counts())

    accuracy = (results['predicted'] == results['target']).mean()
    print(f"\n🎯 Accuracy: {accuracy:.1%}")

    print(f"\n✅ Ready for calibration!")
    print(f"   Run: python optimize_betting_thresholds.py --data {output_file} --n-trials 500")

    return output_file


if __name__ == '__main__':
    main()
