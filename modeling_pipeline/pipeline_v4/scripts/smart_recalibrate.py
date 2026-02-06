#!/usr/bin/env python3
"""
Smart Recalibration - Only Update if Performance Improves

Problem: Blindly recalibrating can DEGRADE performance if new calibrators overfit.
Solution: Validate new calibrators before deploying them.

Process:
1. Fit new calibrators on training period (e.g., last 6 months)
2. Test OLD vs NEW calibrators on validation period (e.g., last 2 months)
3. Compare PnL, win rate, and other metrics
4. Only save new calibrators if they beat old ones
5. Otherwise, keep existing calibrators

Usage:
    python3 scripts/smart_recalibrate.py --train-months 6 --val-months 2
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

def fetch_predictions(start_date, end_date, cur):
    """Fetch predictions for date range."""
    cur.execute("""
        SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result,
               best_home_odds, best_draw_odds, best_away_odds
        FROM predictions
        WHERE match_date >= %s AND match_date < %s
          AND actual_result IS NOT NULL
          AND pred_home_prob IS NOT NULL
          AND best_home_odds IS NOT NULL
        ORDER BY match_date
    """, (start_date, end_date))
    return cur.fetchall()

def fit_calibrators(rows):
    """Fit isotonic calibrators on data."""
    calibrators = {}

    for outcome, col in [('home', 'pred_home_prob'),
                         ('draw', 'pred_draw_prob'),
                         ('away', 'pred_away_prob')]:
        X = np.array([r[col] for r in rows])
        y = np.array([1.0 if r['actual_result'] == outcome[0].upper() else 0.0 for r in rows])

        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(X, y)
        calibrators[outcome] = cal

    return calibrators

def apply_calibrators(rows, calibrators, prefix='cal'):
    """Apply calibrators to raw predictions."""
    for row in rows:
        row[f'{prefix}_home'] = float(calibrators['home'].predict([row['pred_home_prob']])[0])
        row[f'{prefix}_draw'] = float(calibrators['draw'].predict([row['pred_draw_prob']])[0])
        row[f'{prefix}_away'] = float(calibrators['away'].predict([row['pred_away_prob']])[0])

def test_strategy(rows, thresholds, cal_prefix):
    """Test pure threshold strategy with given calibrators."""
    bets = []

    for row in rows:
        candidates = []

        home_prob = row[f'{cal_prefix}_home']
        draw_prob = row[f'{cal_prefix}_draw']
        away_prob = row[f'{cal_prefix}_away']

        if home_prob > thresholds['home'] and row['best_home_odds']:
            candidates.append(('H', home_prob, row['best_home_odds']))
        if draw_prob > thresholds['draw'] and row['best_draw_odds']:
            candidates.append(('D', draw_prob, row['best_draw_odds']))
        if away_prob > thresholds['away'] and row['best_away_odds']:
            candidates.append(('A', away_prob, row['best_away_odds']))

        if candidates:
            outcome, _, odds = max(candidates, key=lambda x: x[1])
            won = (outcome == row['actual_result'])
            bets.append({'outcome': outcome, 'odds': odds, 'won': won})

    if not bets:
        return None

    n = len(bets)
    wins = sum(1 for b in bets if b['won'])
    pnl = sum((b['odds']-1) if b['won'] else -1 for b in bets)

    return {
        'total_bets': n,
        'total_wins': wins,
        'win_rate': wins/n*100,
        'total_pnl': pnl,
        'roi': pnl/n*100
    }

def print_calibrator_info(calibrators, name):
    """Print calibrator statistics."""
    print(f"\n{name}:")
    for outcome in ['home', 'draw', 'away']:
        cal = calibrators[outcome]
        if hasattr(cal, 'X_thresholds_'):
            n_points = len(cal.X_thresholds_)
            x_range = f"{cal.X_thresholds_[0]:.3f}-{cal.X_thresholds_[-1]:.3f}"
            y_range = f"{cal.y_thresholds_[0]:.3f}-{cal.y_thresholds_[-1]:.3f}"
            print(f"  {outcome.capitalize():5} — {n_points:3d} points | X range: {x_range} | Y range: {y_range}")

def main():
    parser = argparse.ArgumentParser(description='Smart recalibration with validation')
    parser.add_argument('--train-months', type=int, default=6,
                       help='Months of data for training calibrators (default: 6)')
    parser.add_argument('--val-months', type=int, default=2,
                       help='Months of data for validation (default: 2)')
    parser.add_argument('--min-train-samples', type=int, default=200,
                       help='Minimum training samples required (default: 200)')
    parser.add_argument('--min-val-samples', type=int, default=100,
                       help='Minimum validation samples required (default: 100)')
    parser.add_argument('--force', action='store_true',
                       help='Force update even if new calibrators are worse')

    args = parser.parse_args()

    print("="*80)
    print("SMART RECALIBRATION WITH VALIDATION")
    print("="*80)

    # Load production config for thresholds
    sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
    from production_config import THRESHOLDS

    print(f"\nProduction thresholds: H={THRESHOLDS['home']:.2f}, D={THRESHOLDS['draw']:.2f}, A={THRESHOLDS['away']:.2f}")
    print(f"Training period: Last {args.train_months} months")
    print(f"Validation period: Last {args.val_months} months")

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
        print("\n❌ ERROR: DATABASE_URL not found")
        sys.exit(1)

    # Calculate date ranges
    now = datetime.now()
    val_end = now
    val_start = now - timedelta(days=30 * args.val_months)
    train_end = val_start
    train_start = train_end - timedelta(days=30 * args.train_months)

    print(f"\nTraining:   {train_start.date()} to {train_end.date()}")
    print(f"Validation: {val_start.date()} to {val_end.date()}")

    # Connect to database
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Fetch data
    print("\n[1/5] Fetching training data...")
    train_rows = fetch_predictions(train_start.date(), train_end.date(), cur)
    print(f"   Loaded {len(train_rows)} training predictions")

    if len(train_rows) < args.min_train_samples:
        print(f"\n❌ ERROR: Not enough training data ({len(train_rows)} < {args.min_train_samples})")
        print("   Try reducing --train-months or waiting for more results")
        cur.close()
        conn.close()
        sys.exit(1)

    print("\n[2/5] Fetching validation data...")
    val_rows = fetch_predictions(val_start.date(), val_end.date(), cur)
    print(f"   Loaded {len(val_rows)} validation predictions")

    if len(val_rows) < args.min_val_samples:
        print(f"\n❌ ERROR: Not enough validation data ({len(val_rows)} < {args.min_val_samples})")
        print("   Try reducing --val-months or waiting for more results")
        cur.close()
        conn.close()
        sys.exit(1)

    cur.close()
    conn.close()

    # Load existing calibrators via LATEST_CALIBRATORS pointer
    print("\n[3/5] Loading existing calibrators...")
    production_dir = Path('models/production')
    latest_file = production_dir / 'LATEST_CALIBRATORS'

    existing_cal_path = None
    if latest_file.exists():
        cal_filename = latest_file.read_text().strip()
        candidate = production_dir / cal_filename
        if candidate.exists():
            existing_cal_path = candidate
        else:
            print(f"   ⚠️  LATEST_CALIBRATORS points to {cal_filename} but file missing")
    # Fallback: legacy unversioned path
    if existing_cal_path is None:
        legacy = Path('models/calibrators.joblib')
        if legacy.exists():
            existing_cal_path = legacy
            print("   ⚠️  Using legacy calibrators path (no LATEST_CALIBRATORS)")

    if existing_cal_path is None or not existing_cal_path.exists():
        print("   ⚠️  No existing calibrators found - will save new ones")
        existing_calibrators = None
    else:
        existing_calibrators = joblib.load(existing_cal_path)
        print(f"   ✓ Loaded existing calibrators from {existing_cal_path}")
        print_calibrator_info(existing_calibrators, "   Existing calibrators")

    # Fit new calibrators
    print("\n[4/5] Fitting new calibrators on training data...")
    new_calibrators = fit_calibrators(train_rows)
    print("   ✓ New calibrators fitted")
    print_calibrator_info(new_calibrators, "   New calibrators")

    # Validate both on validation set
    print("\n[5/5] Comparing performance on validation data...")
    print("-"*80)

    # Apply both calibrators to validation set
    if existing_calibrators:
        apply_calibrators(val_rows, existing_calibrators, prefix='old')
    apply_calibrators(val_rows, new_calibrators, prefix='new')

    # Test strategies
    if existing_calibrators:
        old_result = test_strategy(val_rows, THRESHOLDS, 'old')
        print(f"EXISTING calibrators: {old_result['total_bets']} bets | "
              f"{old_result['win_rate']:.1f}% WR | {old_result['total_pnl']:+.2f} PnL | "
              f"{old_result['roi']:+.1f}% ROI")
    else:
        old_result = None
        print("EXISTING calibrators: None")

    new_result = test_strategy(val_rows, THRESHOLDS, 'new')
    print(f"NEW calibrators:      {new_result['total_bets']} bets | "
          f"{new_result['win_rate']:.1f}% WR | {new_result['total_pnl']:+.2f} PnL | "
          f"{new_result['roi']:+.1f}% ROI")

    # Decision
    print("\n" + "="*80)
    print("DECISION")
    print("="*80)

    if existing_calibrators is None:
        # No existing calibrators - save new ones
        print("\n✅ No existing calibrators - saving new ones")
        should_update = True
    elif args.force:
        # Force update
        print("\n⚠️  --force flag set - updating calibrators regardless of performance")
        should_update = True
    else:
        # Compare performance
        pnl_diff = new_result['total_pnl'] - old_result['total_pnl']
        wr_diff = new_result['win_rate'] - old_result['win_rate']

        print(f"\nPerformance difference (NEW - EXISTING):")
        print(f"  PnL:      {pnl_diff:+.2f}")
        print(f"  Win Rate: {wr_diff:+.1f}%")
        print(f"  ROI:      {new_result['roi'] - old_result['roi']:+.1f}%")

        # Decision criteria: New must be better on PnL
        if new_result['total_pnl'] > old_result['total_pnl']:
            print(f"\n✅ NEW calibrators are BETTER (+{pnl_diff:.2f} PnL improvement)")
            print("   Updating calibrators...")
            should_update = True
        else:
            print(f"\n❌ NEW calibrators are WORSE ({pnl_diff:.2f} PnL degradation)")
            print("   KEEPING existing calibrators")
            print("\n   Why this happens:")
            print("   • Training period may not be representative of validation period")
            print("   • New calibrators may be overfitting to training data")
            print("   • Existing calibrators may have better generalization")
            print("\n   Suggestions:")
            print("   • Try longer training period: --train-months 9")
            print("   • Wait for more data to accumulate")
            print("   • Keep existing calibrators - they're working well!")
            should_update = False

    # Save if needed
    if should_update:
        import json, re
        production_dir = Path('models/production')
        production_dir.mkdir(parents=True, exist_ok=True)

        # Determine next version by scanning existing calibrator files
        existing_versions = []
        for f in production_dir.glob('calibrators_v*.joblib'):
            m = re.search(r'calibrators_v(\d+)\.(\d+)\.(\d+)\.joblib', f.name)
            if m:
                existing_versions.append(tuple(map(int, m.groups())))
        if existing_versions:
            existing_versions.sort(reverse=True)
            major, minor, patch = existing_versions[0]
            new_version = f"{major}.{minor+1}.{patch}"
        else:
            new_version = "1.0.0"

        versioned_path = production_dir / f'calibrators_v{new_version}.joblib'
        metadata_path = production_dir / f'calibrators_v{new_version}_metadata.json'
        latest_file = production_dir / 'LATEST_CALIBRATORS'

        # Save versioned calibrator
        joblib.dump(new_calibrators, versioned_path)
        print(f"\n   ✅ Saved new calibrators to: {versioned_path}")

        # Write metadata
        metadata = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "calibrator_type": "IsotonicRegression",
            "paired_model": "model_v2.0.0",
            "training_data": {
                "source": "predictions table (resolved, with odds)",
                "n_samples": len(train_rows),
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "val_start": str(val_start.date()),
                "val_end": str(val_end.date())
            },
            "step_function": {},
            "validation_performance": {
                "thresholds_used": THRESHOLDS,
                "new": new_result,
                "existing": old_result if old_result else None,
                "pnl_improvement": round(new_result['total_pnl'] - (old_result['total_pnl'] if old_result else 0), 2)
            },
            "replaces": existing_cal_path.name if existing_cal_path else None
        }
        # Capture step function details
        for outcome in ['home', 'draw', 'away']:
            c = new_calibrators[outcome]
            metadata["step_function"][outcome] = {
                "n_points": int(len(c.X_thresholds_)),
                "x_range": [round(float(c.X_thresholds_[0]), 4), round(float(c.X_thresholds_[-1]), 4)],
                "y_range": [round(float(c.y_thresholds_[0]), 4), round(float(c.y_thresholds_[-1]), 4)]
            }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadata saved to: {metadata_path}")

        # Update LATEST_CALIBRATORS pointer
        latest_file.write_text(versioned_path.name + '\n')
        print(f"   ✅ LATEST_CALIBRATORS updated → {versioned_path.name}")

        if old_result:
            print(f"\n   Performance improvement on validation: {pnl_diff:+.2f} PnL")
    else:
        print("\n   ✅ Kept existing calibrators (no changes made)")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
