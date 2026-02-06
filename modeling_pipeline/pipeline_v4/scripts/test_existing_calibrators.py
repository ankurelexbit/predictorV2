#!/usr/bin/env python3
"""
Test betting strategies using EXISTING calibrators.joblib
Compare with freshly fitted calibrators to explain Jan breakeven.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import psycopg2.extras
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression

# Get DB credentials
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    with open('.env') as f:
        for line in f:
            if line.startswith('DATABASE_URL='):
                DATABASE_URL = line.split('=', 1)[1].strip().strip('"')
                break

print("="*80)
print("TESTING EXISTING CALIBRATORS vs FRESH CALIBRATORS")
print("="*80)

# ============================================================================
# Load EXISTING calibrators
# ============================================================================
print("\n[1] Loading EXISTING calibrators from models/calibrators.joblib...")
existing_calibrators = joblib.load('models/calibrators.joblib')
print("   ✓ Loaded existing calibrators")

# ============================================================================
# Fit FRESH calibrators on Oct-Dec
# ============================================================================
print("\n[2] Fitting FRESH calibrators on Oct-Dec 2025...")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

cur.execute("""
    SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result
    FROM predictions
    WHERE match_date >= '2025-10-01' AND match_date < '2026-01-01'
      AND actual_result IS NOT NULL
      AND pred_home_prob IS NOT NULL
    ORDER BY match_date
""")
cal_rows = cur.fetchall()
print(f"   Loaded {len(cal_rows)} predictions for calibration")

fresh_calibrators = {}
for outcome, col in [('home', 'pred_home_prob'), ('draw', 'pred_draw_prob'), ('away', 'pred_away_prob')]:
    X = np.array([r[col] for r in cal_rows])
    y = np.array([1.0 if r['actual_result'] == outcome[0].upper() else 0.0 for r in cal_rows])

    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(X, y)
    fresh_calibrators[outcome] = cal

print("   ✓ Fresh calibrators fitted")

# ============================================================================
# Load Jan 2026 test data
# ============================================================================
print("\n[3] Loading Jan 2026 test data...")

cur.execute("""
    SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result,
           best_home_odds, best_draw_odds, best_away_odds
    FROM predictions
    WHERE match_date >= '2026-01-01' AND match_date < '2026-02-01'
      AND actual_result IS NOT NULL
      AND pred_home_prob IS NOT NULL
      AND best_home_odds IS NOT NULL
    ORDER BY match_date
""")
test_rows = cur.fetchall()
cur.close()
conn.close()

print(f"   Loaded {len(test_rows)} test predictions")

# Apply both calibrators
for row in test_rows:
    # Existing calibrators
    row['exist_home'] = float(existing_calibrators['home'].predict([row['pred_home_prob']])[0])
    row['exist_draw'] = float(existing_calibrators['draw'].predict([row['pred_draw_prob']])[0])
    row['exist_away'] = float(existing_calibrators['away'].predict([row['pred_away_prob']])[0])

    # Fresh calibrators
    row['fresh_home'] = float(fresh_calibrators['home'].predict([row['pred_home_prob']])[0])
    row['fresh_draw'] = float(fresh_calibrators['draw'].predict([row['pred_draw_prob']])[0])
    row['fresh_away'] = float(fresh_calibrators['away'].predict([row['pred_away_prob']])[0])

# ============================================================================
# Test production strategy with both calibrators
# ============================================================================
print("\n[4] Testing production strategy: H=0.36, D=0.28, A=0.40")
print("="*80)

THRESHOLDS = {'home': 0.36, 'draw': 0.28, 'away': 0.40}

def run_strategy(test_data, cal_prefix):
    """Run pure threshold strategy with specified calibrators."""
    bets = []

    for row in test_data:
        candidates = []

        home_prob = row[f'{cal_prefix}_home']
        draw_prob = row[f'{cal_prefix}_draw']
        away_prob = row[f'{cal_prefix}_away']

        if home_prob > THRESHOLDS['home'] and row['best_home_odds']:
            candidates.append(('H', home_prob, row['best_home_odds']))
        if draw_prob > THRESHOLDS['draw'] and row['best_draw_odds']:
            candidates.append(('D', draw_prob, row['best_draw_odds']))
        if away_prob > THRESHOLDS['away'] and row['best_away_odds']:
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

    # Per-outcome breakdown
    outcome_data = {}
    for o in ['H', 'D', 'A']:
        subset = [b for b in bets if b['outcome'] == o]
        if subset:
            ns = len(subset)
            ws = sum(1 for b in subset if b['won'])
            ps = sum((b['odds']-1) if b['won'] else -1 for b in subset)
            outcome_data[o] = {'bets': ns, 'wins': ws, 'pnl': ps}
        else:
            outcome_data[o] = {'bets': 0, 'wins': 0, 'pnl': 0.0}

    return {
        'total_bets': n,
        'total_wins': wins,
        'win_rate': wins/n*100,
        'total_pnl': pnl,
        'roi': pnl/n*100,
        'home': outcome_data['H'],
        'draw': outcome_data['D'],
        'away': outcome_data['A']
    }

# Test with existing calibrators
exist_result = run_strategy(test_rows, 'exist')
print("\nUsing EXISTING calibrators (models/calibrators.joblib):")
print(f"  Total: {exist_result['total_bets']} bets, {exist_result['total_wins']} wins, "
      f"{exist_result['win_rate']:.1f}% WR")
print(f"  PnL: {exist_result['total_pnl']:+.2f} ({exist_result['roi']:+.1f}% ROI)")
print(f"  Home: {exist_result['home']['bets']} bets, {exist_result['home']['wins']} wins, "
      f"{exist_result['home']['pnl']:+.2f} PnL")
print(f"  Draw: {exist_result['draw']['bets']} bets, {exist_result['draw']['wins']} wins, "
      f"{exist_result['draw']['pnl']:+.2f} PnL")
print(f"  Away: {exist_result['away']['bets']} bets, {exist_result['away']['wins']} wins, "
      f"{exist_result['away']['pnl']:+.2f} PnL")

# Test with fresh calibrators
fresh_result = run_strategy(test_rows, 'fresh')
print("\nUsing FRESH calibrators (fitted on Oct-Dec 2025):")
print(f"  Total: {fresh_result['total_bets']} bets, {fresh_result['total_wins']} wins, "
      f"{fresh_result['win_rate']:.1f}% WR")
print(f"  PnL: {fresh_result['total_pnl']:+.2f} ({fresh_result['roi']:+.1f}% ROI)")
print(f"  Home: {fresh_result['home']['bets']} bets, {fresh_result['home']['wins']} wins, "
      f"{fresh_result['home']['pnl']:+.2f} PnL")
print(f"  Draw: {fresh_result['draw']['bets']} bets, {fresh_result['draw']['wins']} wins, "
      f"{fresh_result['draw']['pnl']:+.2f} PnL")
print(f"  Away: {fresh_result['away']['bets']} bets, {fresh_result['away']['wins']} wins, "
      f"{fresh_result['away']['pnl']:+.2f} PnL")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

diff_pnl = exist_result['total_pnl'] - fresh_result['total_pnl']
diff_bets = exist_result['total_bets'] - fresh_result['total_bets']

print(f"\nPnL Difference: {diff_pnl:+.2f} (existing vs fresh)")
print(f"Bet Count Difference: {diff_bets:+d} bets")

if abs(exist_result['total_pnl']) < 2.0:
    print(f"\n✅ Existing calibrators show BREAKEVEN on Jan ({exist_result['total_pnl']:+.2f})")
else:
    print(f"\n⚠️  Existing calibrators show {exist_result['total_pnl']:+.2f} PnL (not breakeven)")

print("\nKey differences in calibration:")
print(f"  Draw volume: {exist_result['draw']['bets']} (existing) vs "
      f"{fresh_result['draw']['bets']} (fresh)")
print(f"  Home volume: {exist_result['home']['bets']} (existing) vs "
      f"{fresh_result['home']['bets']} (fresh)")
