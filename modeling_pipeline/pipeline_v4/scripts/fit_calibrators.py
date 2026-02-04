#!/usr/bin/env python3
"""
Fit Probability Calibrators
============================

Fits isotonic regression calibrators on resolved predictions from the database.
Calibrators correct CatBoost's raw probability output before thresholds are applied.

Run this monthly, or whenever you have 100+ new resolved predictions.

Usage:
    python3 scripts/fit_calibrators.py

    # Show calibration stats without saving
    python3 scripts/fit_calibrators.py --dry-run

    # Use minimum N resolved predictions (default: 100)
    python3 scripts/fit_calibrators.py --min-predictions 200
"""

import sys
import os
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib

from src.database import SupabaseClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fit isotonic probability calibrators')
    parser.add_argument('--dry-run', action='store_true', help='Show stats without saving')
    parser.add_argument('--min-predictions', type=int, default=100,
                        help='Minimum resolved predictions required (default: 100)')
    args = parser.parse_args()

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    db = SupabaseClient(database_url)

    # Pull all resolved predictions
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result
            FROM predictions
            WHERE actual_result IS NOT NULL
            ORDER BY match_date
        """)
        rows = cursor.fetchall()

    logger.info(f"Resolved predictions: {len(rows)}")

    if len(rows) < args.min_predictions:
        logger.error(f"Need >= {args.min_predictions} resolved predictions, have {len(rows)}")
        sys.exit(1)

    # Build arrays
    home_probs  = np.array([r[0] for r in rows])
    draw_probs  = np.array([r[1] for r in rows])
    away_probs  = np.array([r[2] for r in rows])
    home_actual = np.array([1 if r[3] == 'H' else 0 for r in rows])
    draw_actual = np.array([1 if r[3] == 'D' else 0 for r in rows])
    away_actual = np.array([1 if r[3] == 'A' else 0 for r in rows])

    # Fit isotonic regression per outcome
    iso_home = IsotonicRegression(out_of_bounds='clip').fit(home_probs, home_actual)
    iso_draw = IsotonicRegression(out_of_bounds='clip').fit(draw_probs, draw_actual)
    iso_away = IsotonicRegression(out_of_bounds='clip').fit(away_probs, away_actual)

    # Report calibration error before vs after
    logger.info("\nCalibration error (mean absolute, binned into 0.05 buckets):")
    bins = np.arange(0.20, 0.85, 0.05)

    for label, raw, calibrator, actual in [
        ("Home", home_probs, iso_home, home_actual),
        ("Draw", draw_probs, iso_draw, draw_actual),
        ("Away", away_probs, iso_away, away_actual),
    ]:
        cal = calibrator.predict(raw)
        raw_errors, cal_errors = [], []
        for i in range(len(bins) - 1):
            mask = (raw >= bins[i]) & (raw < bins[i+1])
            if mask.sum() < 5:
                continue
            act = actual[mask].mean()
            raw_errors.append(abs(raw[mask].mean() - act))
            cal_errors.append(abs(cal[mask].mean() - act))
        if raw_errors:
            logger.info(f"  {label}: raw MAE={np.mean(raw_errors)*100:.1f}%  →  cal MAE={np.mean(cal_errors)*100:.1f}%")

    if args.dry_run:
        logger.info("\n--dry-run: calibrators NOT saved")
        return

    # Save
    output_path = Path('models/calibrators.joblib')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'home': iso_home, 'draw': iso_draw, 'away': iso_away}, output_path)
    logger.info(f"\n✅ Calibrators saved to {output_path}")
    logger.info(f"   Fitted on {len(rows)} resolved predictions")


if __name__ == '__main__':
    main()
