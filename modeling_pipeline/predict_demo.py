"""
Demo Predictions - Using Historical Test Data
==============================================

Make predictions for recent matches from the test dataset to demonstrate
model predictions without needing API access.

Usage:
    python predict_demo.py --n-matches 20
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR
from utils import setup_logger

# Setup
logger = setup_logger("predict_demo")


def load_models():
    """Load all trained models."""
    models = {}

    # Load individual models
    elo_path = MODELS_DIR / "elo_model.joblib"
    if elo_path.exists():
        import importlib
        elo_module = importlib.import_module('04_model_baseline_elo')
        EloProbabilityModel = elo_module.EloProbabilityModel
        elo_model = EloProbabilityModel()
        elo_model.load(elo_path)
        models['elo'] = elo_model
        logger.info("Loaded Elo model")

    dixon_path = MODELS_DIR / "dixon_coles_model.joblib"
    if dixon_path.exists():
        try:
            dc_data = joblib.load(dixon_path)
            import importlib
            dc_module = importlib.import_module('05_model_dixon_coles')
            DixonColesModel = dc_module.DixonColesModel
            CalibratedDixonColes = dc_module.CalibratedDixonColes

            base_model = DixonColesModel()
            base_model.attack = dc_data['base_model_data']['attack']
            base_model.defense = dc_data['base_model_data']['defense']
            base_model.home_advantage = dc_data['base_model_data']['home_advantage']
            base_model.team_to_idx = dc_data['base_model_data']['team_to_idx']
            base_model.idx_to_team = dc_data['base_model_data']['idx_to_team']
            base_model.reference_date = dc_data['base_model_data']['reference_date']
            base_model.time_decay = dc_data['base_model_data']['time_decay']
            base_model.is_fitted = True

            dc_model = CalibratedDixonColes(base_model)
            dc_model.calibrators = dc_data['calibrators']
            dc_model.is_calibrated = dc_data['is_calibrated']

            models['dixon_coles'] = dc_model
            logger.info("Loaded Dixon-Coles model")
        except Exception as e:
            logger.warning(f"Could not load Dixon-Coles: {e}")

    xgb_path = MODELS_DIR / "xgboost_model.joblib"
    if xgb_path.exists():
        try:
            import importlib
            xgb_module = importlib.import_module('06_model_xgboost')
            XGBoostFootballModel = xgb_module.XGBoostFootballModel
            xgb_model = XGBoostFootballModel()
            xgb_model.load(xgb_path)
            models['xgboost'] = xgb_model
            logger.info("Loaded XGBoost model")
        except Exception as e:
            logger.warning(f"Could not load XGBoost: {e}")

    # Load stacking ensemble
    stacking_path = MODELS_DIR / "stacking_ensemble.joblib"
    if stacking_path.exists() and len(models) == 3:
        try:
            import importlib
            ensemble_module = importlib.import_module('07_model_ensemble')
            StackingEnsemble = ensemble_module.StackingEnsemble

            stacking = StackingEnsemble()
            for name, model in models.items():
                stacking.add_model(name, model)
            stacking.load(stacking_path)

            models['stacking'] = stacking
            logger.info("Loaded Stacking Ensemble")
        except Exception as e:
            logger.warning(f"Could not load Stacking: {e}")

    return models


def main():
    parser = argparse.ArgumentParser(description="Demo predictions on test data")
    parser.add_argument(
        "--n-matches",
        type=int,
        default=20,
        help="Number of recent matches to predict (default: 20)"
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--season",
        default="2023/2024",
        help="Season to predict from (default: 2023/2024)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DEMO PREDICTIONS - Using Test Data")
    print("=" * 80)

    # Load test data
    print("\nLoading test data...")
    features_path = Path("data/processed/sportmonks_features.csv")

    if not features_path.exists():
        print(f"ERROR: Features file not found: {features_path}")
        return

    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])

    # Filter to test season
    test_df = df[df['season_name'] == args.season].copy()
    test_df = test_df.sort_values('date', ascending=False)

    print(f"Found {len(test_df)} matches in {args.season} season")

    # Take most recent n matches
    sample_df = test_df.head(args.n_matches)

    # Load models
    print("\nLoading models...")
    models = load_models()

    if not models:
        print("ERROR: No models loaded")
        return

    model_name = 'stacking' if 'stacking' in models else 'xgboost'
    model = models[model_name]
    print(f"Using model: {model_name}")

    # Make predictions
    print("\n" + "=" * 80)
    print(f"PREDICTIONS FOR {args.n_matches} RECENT MATCHES")
    print("=" * 80)

    predictions = []

    for idx, row in sample_df.iterrows():
        try:
            # Make prediction
            probs = model.predict_proba(pd.DataFrame([row]))[0]

            # Get actual result
            actual_outcome = int(row['target'])
            actual_label = ['Away Win', 'Draw', 'Home Win'][actual_outcome]
            predicted_outcome = np.argmax(probs)
            predicted_label = ['Away Win', 'Draw', 'Home Win'][predicted_outcome]

            correct = (predicted_outcome == actual_outcome)

            # Store prediction
            result = {
                'date': row['date'],
                'home_team': row['home_team_name'],
                'away_team': row['away_team_name'],
                'home_goals': int(row['home_goals']) if not pd.isna(row['home_goals']) else None,
                'away_goals': int(row['away_goals']) if not pd.isna(row['away_goals']) else None,
                'actual_result': actual_label,
                'home_win_prob': float(probs[2]),
                'draw_prob': float(probs[1]),
                'away_win_prob': float(probs[0]),
                'predicted_outcome': predicted_label,
                'correct': correct,
                'model_used': model_name
            }

            predictions.append(result)

            # Print prediction
            print(f"\n{row['date'].strftime('%Y-%m-%d %H:%M')}")
            print(f"{row['home_team_name']} vs {row['away_team_name']}")

            if result['home_goals'] is not None:
                print(f"Final Score: {result['home_goals']}-{result['away_goals']} ({actual_label})")

            print(f"Predictions:")
            print(f"  Home Win: {result['home_win_prob']:.1%}")
            print(f"  Draw:     {result['draw_prob']:.1%}")
            print(f"  Away Win: {result['away_win_prob']:.1%}")
            print(f"  → Predicted: {predicted_label}")
            print(f"  → Result: {'✅ CORRECT' if correct else '❌ WRONG'}")

        except Exception as e:
            logger.error(f"Error predicting match: {e}")
            continue

    # Calculate accuracy
    correct_predictions = sum(1 for p in predictions if p['correct'])
    accuracy = correct_predictions / len(predictions) if predictions else 0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Predictions: {len(predictions)}")
    print(f"Correct: {correct_predictions}")
    print(f"Wrong: {len(predictions) - correct_predictions}")
    print(f"Accuracy: {accuracy:.1%}")

    # Save to CSV if requested
    if args.output:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")

    print("=" * 80)


if __name__ == "__main__":
    main()
