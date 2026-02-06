#!/usr/bin/env python3
"""
Recalibrate Model - Fit Isotonic Regression Calibrators

Calibration corrects model overconfidence/underconfidence by mapping
raw probabilities to actual win rates using recent historical data.

WHEN TO RUN:
- After every model retrain (REQUIRED)
- Model and calibrators must be updated together
- Run standalone: python3 scripts/recalibrate_model.py --months-back 3

HOW IT WORKS:
1. Fetches recent resolved predictions from database (last N months)
2. Fits isotonic regression on (raw_prob, actual_outcome) pairs
3. Saves calibrators to models/calibrators.joblib

Example:
    Model says Home win prob = 0.60
    But historically, when model said 0.60, Home only won 48% of the time
    Calibrator learns to map: 0.60 → 0.48
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import psycopg2
import psycopg2.extras
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser(description='Recalibrate model on recent predictions')
    parser.add_argument('--months-back', type=int, default=3,
                       help='Number of months of predictions to use for calibration (default: 3)')
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum number of predictions required (default: 100)')
    parser.add_argument('--output', type=str, default='models/calibrators.joblib',
                       help='Output path for calibrators (default: models/calibrators.joblib)')

    args = parser.parse_args()

    print("="*80)
    print("MODEL RECALIBRATION")
    print("="*80)
    print(f"\nCalibration period: Last {args.months_back} months")
    print(f"Minimum samples required: {args.min_samples}")

    # Get DB credentials
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith('DATABASE_URL='):
                        DATABASE_URL = line.split('=', 1)[1].strip().strip('"')
                        break

    if not DATABASE_URL:
        print("\n❌ ERROR: DATABASE_URL not found in environment or .env file")
        sys.exit(1)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * args.months_back)

    print(f"\n[1/3] Fetching predictions from {start_date.date()} to {end_date.date()}...")

    # Fetch predictions
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result
        FROM predictions
        WHERE match_date >= %s AND match_date < %s
          AND actual_result IS NOT NULL
          AND pred_home_prob IS NOT NULL
        ORDER BY match_date
    """, (start_date.date(), end_date.date()))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"   Loaded {len(rows)} predictions with resolved results")

    if len(rows) < args.min_samples:
        print(f"\n❌ ERROR: Not enough data for calibration")
        print(f"   Found: {len(rows)} predictions")
        print(f"   Required: {args.min_samples} predictions")
        print(f"\nTry reducing --months-back or waiting for more results to resolve")
        sys.exit(1)

    # Fit calibrators
    print(f"\n[2/3] Fitting isotonic regression calibrators...")

    calibrators = {}

    for outcome, col in [('home', 'pred_home_prob'),
                         ('draw', 'pred_draw_prob'),
                         ('away', 'pred_away_prob')]:
        X = np.array([r[col] for r in rows])
        y = np.array([1.0 if r['actual_result'] == outcome[0].upper() else 0.0 for r in rows])

        # Fit isotonic regression
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(X, y)
        calibrators[outcome] = cal

        # Calculate metrics
        actual_rate = y.mean()
        raw_mean = X.mean()
        cal_mean = cal.predict(X).mean()
        mae_before = np.mean(np.abs(X - y))
        mae_after = np.mean(np.abs(cal.predict(X) - y))

        print(f"   {outcome.capitalize():5} — Actual: {actual_rate:.3f} | "
              f"Raw: {raw_mean:.3f} | Calibrated: {cal_mean:.3f} | "
              f"MAE: {mae_before:.3f}→{mae_after:.3f}")

    # Save calibrators
    print(f"\n[3/3] Saving calibrators to {args.output}...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrators, output_path)

    print(f"   ✅ Calibrators saved successfully")

    # Verification
    print("\n" + "="*80)
    print("CALIBRATION COMPLETE")
    print("="*80)
    print(f"\n✅ Calibrators trained on {len(rows)} predictions")
    print(f"✅ Saved to: {output_path}")
    print(f"\nNext steps:")
    print("  1. Calibrators are now in sync with your latest model")
    print("  2. Run predictions with: scripts/predict_production.py")
    print("  3. Thresholds in config/production_config.py apply to calibrated probs")

    # Test calibrators on sample inputs
    print("\n" + "="*80)
    print("SAMPLE CALIBRATION MAPPINGS")
    print("="*80)

    test_probs = [0.3, 0.4, 0.5, 0.6, 0.7]

    for outcome in ['home', 'draw', 'away']:
        print(f"\n{outcome.upper()}:")
        calibrated = calibrators[outcome].predict(test_probs)
        for raw, cal in zip(test_probs, calibrated):
            diff = cal - raw
            print(f"  Raw {raw:.2f} → Calibrated {cal:.3f} ({diff:+.3f})")

if __name__ == '__main__':
    main()
