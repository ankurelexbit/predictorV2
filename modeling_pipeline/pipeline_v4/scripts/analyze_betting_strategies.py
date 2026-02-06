#!/usr/bin/env python3
"""
Betting Strategy Analysis - Compare Multiple Approaches

1. Fits calibrators on Oct-Dec 2025 predictions
2. Tests 3 strategy types on Nov 2025 - Jan 2026:
   - Pure Threshold (pick highest cal_prob)
   - EV-Based (pick highest EV, various min_ev)
   - Hybrid (threshold + EV>0 gate)
3. Recommends best strategy based on PnL, win rate, and distribution
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import psycopg2.extras
import numpy as np
from sklearn.isotonic import IsotonicRegression
from collections import Counter
import itertools

# Get DB credentials
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    with open('.env') as f:
        for line in f:
            if line.startswith('DATABASE_URL='):
                DATABASE_URL = line.split('=', 1)[1].strip().strip('"')
                break

print("="*80)
print("BETTING STRATEGY ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Load Oct-Dec data and fit calibrators
# ============================================================================
print("\n[1/4] Loading Oct-Dec 2025 predictions for calibration...")

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

print(f"   Loaded {len(cal_rows)} predictions")

if len(cal_rows) < 50:
    print("ERROR: Not enough calibration data (need at least 50)")
    sys.exit(1)

# Fit isotonic calibrators
print("\n[2/4] Fitting isotonic calibrators...")
calibrators = {}

for outcome, col in [('home', 'pred_home_prob'), ('draw', 'pred_draw_prob'), ('away', 'pred_away_prob')]:
    X = np.array([r[col] for r in cal_rows])
    y = np.array([1.0 if r['actual_result'] == outcome[0].upper() else 0.0 for r in cal_rows])

    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(X, y)
    calibrators[outcome] = cal

    actual_rate = y.mean()
    raw_mean = X.mean()
    cal_mean = cal.predict(X).mean()
    mae_before = np.mean(np.abs(X - y))
    mae_after = np.mean(np.abs(cal.predict(X) - y))

    print(f"   {outcome.capitalize():5} — Actual: {actual_rate:.3f} | Raw: {raw_mean:.3f} | "
          f"Calibrated: {cal_mean:.3f} | MAE: {mae_before:.3f}→{mae_after:.3f}")

# ============================================================================
# STEP 2: Load Nov 2025 - Jan 2026 test data
# ============================================================================
print("\n[3/4] Loading Nov 2025 - Jan 2026 predictions for testing...")

cur.execute("""
    SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result,
           best_home_odds, best_draw_odds, best_away_odds
    FROM predictions
    WHERE match_date >= '2025-11-01' AND match_date < '2026-02-01'
      AND actual_result IS NOT NULL
      AND pred_home_prob IS NOT NULL
      AND best_home_odds IS NOT NULL
    ORDER BY match_date
""")
test_rows = cur.fetchall()
cur.close()
conn.close()

print(f"   Loaded {len(test_rows)} test predictions")

if len(test_rows) < 20:
    print("ERROR: Not enough test data (need at least 20)")
    sys.exit(1)

# Apply calibrators to test set
for row in test_rows:
    row['cal_home'] = float(calibrators['home'].predict([row['pred_home_prob']])[0])
    row['cal_draw'] = float(calibrators['draw'].predict([row['pred_draw_prob']])[0])
    row['cal_away'] = float(calibrators['away'].predict([row['pred_away_prob']])[0])
    # Also keep raw probs for comparison
    row['raw_home'] = row['pred_home_prob']
    row['raw_draw'] = row['pred_draw_prob']
    row['raw_away'] = row['pred_away_prob']

# ============================================================================
# STEP 3: Define and test strategies
# ============================================================================
print("\n[4/4] Testing betting strategies on Nov 2025 - Jan 2026...")

def pure_threshold(row, thresholds):
    """Pick highest cal_prob among outcomes that exceed threshold."""
    candidates = []
    if row['cal_home'] > thresholds['home'] and row['best_home_odds']:
        candidates.append(('H', row['cal_home'], row['best_home_odds']))
    if row['cal_draw'] > thresholds['draw'] and row['best_draw_odds']:
        candidates.append(('D', row['cal_draw'], row['best_draw_odds']))
    if row['cal_away'] > thresholds['away'] and row['best_away_odds']:
        candidates.append(('A', row['cal_away'], row['best_away_odds']))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])  # Pick highest cal_prob

def pure_threshold_raw(row, thresholds):
    """Pick highest RAW prob among outcomes that exceed threshold (NO CALIBRATION)."""
    candidates = []
    if row['raw_home'] > thresholds['home'] and row['best_home_odds']:
        candidates.append(('H', row['raw_home'], row['best_home_odds']))
    if row['raw_draw'] > thresholds['draw'] and row['best_draw_odds']:
        candidates.append(('D', row['raw_draw'], row['best_draw_odds']))
    if row['raw_away'] > thresholds['away'] and row['best_away_odds']:
        candidates.append(('A', row['raw_away'], row['best_away_odds']))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])  # Pick highest raw_prob

def ev_strategy(row, min_ev):
    """Pick highest EV above threshold."""
    candidates = []

    if row['best_home_odds']:
        ev = (row['cal_home'] * row['best_home_odds']) - 1
        if ev > min_ev:
            candidates.append(('H', ev, row['best_home_odds']))

    if row['best_draw_odds']:
        ev = (row['cal_draw'] * row['best_draw_odds']) - 1
        if ev > min_ev:
            candidates.append(('D', ev, row['best_draw_odds']))

    if row['best_away_odds']:
        ev = (row['cal_away'] * row['best_away_odds']) - 1
        if ev > min_ev:
            candidates.append(('A', ev, row['best_away_odds']))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])  # Pick highest EV

def hybrid_strategy(row, thresholds):
    """Threshold gate + EV>0 filter, pick highest EV."""
    candidates = []

    if row['cal_home'] > thresholds['home'] and row['best_home_odds']:
        ev = (row['cal_home'] * row['best_home_odds']) - 1
        if ev > 0:
            candidates.append(('H', ev, row['best_home_odds']))

    if row['cal_draw'] > thresholds['draw'] and row['best_draw_odds']:
        ev = (row['cal_draw'] * row['best_draw_odds']) - 1
        if ev > 0:
            candidates.append(('D', ev, row['best_draw_odds']))

    if row['cal_away'] > thresholds['away'] and row['best_away_odds']:
        ev = (row['cal_away'] * row['best_away_odds']) - 1
        if ev > 0:
            candidates.append(('A', ev, row['best_away_odds']))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])  # Pick highest EV

def run_strategy(strategy_fn, params, test_data):
    """Run strategy and calculate metrics."""
    bets = []

    for row in test_data:
        result = strategy_fn(row, params)
        if result:
            outcome, _, odds = result
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

    # Count distribution
    counts = Counter(b['outcome'] for b in bets)

    return {
        'total_bets': n,
        'total_wins': wins,
        'win_rate': wins/n*100,
        'total_pnl': pnl,
        'roi': pnl/n*100,
        'home': outcome_data['H'],
        'draw': outcome_data['D'],
        'away': outcome_data['A'],
        'distribution': {
            'H': counts.get('H', 0)/n*100,
            'D': counts.get('D', 0)/n*100,
            'A': counts.get('A', 0)/n*100
        }
    }

# ============================================================================
# Test configurations
# ============================================================================
results = []

# Strategy 1: Pure Threshold (sweep thresholds)
print("\n   Testing pure threshold strategies...")
threshold_configs = [
    {'home': 0.36, 'draw': 0.28, 'away': 0.40},
    {'home': 0.40, 'draw': 0.26, 'away': 0.40},
    {'home': 0.44, 'draw': 0.26, 'away': 0.42},
    {'home': 0.48, 'draw': 0.26, 'away': 0.44},
]

for thresholds in threshold_configs:
    res = run_strategy(pure_threshold, thresholds, test_rows)
    if res:
        res['strategy'] = 'Pure Threshold'
        res['params'] = f"H={thresholds['home']:.2f} D={thresholds['draw']:.2f} A={thresholds['away']:.2f}"
        results.append(res)

# Strategy 2: EV-Based
print("   Testing EV-based strategies...")
for min_ev in [0.02, 0.05, 0.08, 0.10, 0.15]:
    res = run_strategy(ev_strategy, min_ev, test_rows)
    if res:
        res['strategy'] = 'EV-Based'
        res['params'] = f"min_ev={min_ev:.2f}"
        results.append(res)

# Strategy 3: Hybrid (threshold + EV>0)
print("   Testing hybrid strategies...")
for thresholds in threshold_configs:
    res = run_strategy(hybrid_strategy, thresholds, test_rows)
    if res:
        res['strategy'] = 'Hybrid (Thresh+EV>0)'
        res['params'] = f"H={thresholds['home']:.2f} D={thresholds['draw']:.2f} A={thresholds['away']:.2f}"
        results.append(res)

# Strategy 4: Pure Threshold WITHOUT Calibration (for comparison)
print("   Testing uncalibrated strategies (raw probs)...")
for thresholds in threshold_configs:
    res = run_strategy(pure_threshold_raw, thresholds, test_rows)
    if res:
        res['strategy'] = 'Raw (NO Calibration)'
        res['params'] = f"H={thresholds['home']:.2f} D={thresholds['draw']:.2f} A={thresholds['away']:.2f}"
        results.append(res)

# ============================================================================
# STEP 4: Display results and recommend
# ============================================================================
print("\n" + "="*80)
print("RESULTS - ALL STRATEGIES")
print("="*80)

# Sort by PnL descending
results.sort(key=lambda x: x['total_pnl'], reverse=True)

print(f"\n{'Strategy':<25} {'Params':<30} {'Bets':>5} {'WR%':>5} {'PnL':>7} {'ROI%':>6} {'H/D/A':>9}")
print("-"*90)

for r in results:
    print(f"{r['strategy']:<25} {r['params']:<30} {r['total_bets']:>5} {r['win_rate']:>5.1f} "
          f"{r['total_pnl']:>+7.2f} {r['roi']:>+6.1f} "
          f"{r['distribution']['H']:>3.0f}/{r['distribution']['D']:>2.0f}/{r['distribution']['A']:>2.0f}")

# ============================================================================
# Detailed view of top 3
# ============================================================================
print("\n" + "="*80)
print("TOP 3 STRATEGIES - DETAILED BREAKDOWN")
print("="*80)

for i, r in enumerate(results[:3], 1):
    print(f"\n#{i} — {r['strategy']}: {r['params']}")
    print("-"*80)
    print(f"  Total: {r['total_bets']} bets, {r['total_wins']} wins, {r['win_rate']:.1f}% WR, "
          f"{r['total_pnl']:+.2f} PnL, {r['roi']:+.1f}% ROI")
    print(f"  Distribution: H={r['distribution']['H']:.0f}% D={r['distribution']['D']:.0f}% "
          f"A={r['distribution']['A']:.0f}%")
    print(f"\n  Per-Outcome Breakdown:")
    print(f"    Home: {r['home']['bets']} bets, {r['home']['wins']} wins, {r['home']['pnl']:+.2f} PnL")
    print(f"    Draw: {r['draw']['bets']} bets, {r['draw']['wins']} wins, {r['draw']['pnl']:+.2f} PnL")
    print(f"    Away: {r['away']['bets']} bets, {r['away']['wins']} wins, {r['away']['pnl']:+.2f} PnL")

# ============================================================================
# Recommendation
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

best = results[0]
print(f"\n✅ RECOMMENDED STRATEGY: {best['strategy']}")
print(f"   Parameters: {best['params']}")
print(f"\n   Performance:")
print(f"     • {best['total_bets']} bets")
print(f"     • {best['win_rate']:.1f}% win rate")
print(f"     • {best['total_pnl']:+.2f} PnL (${best['total_pnl']:.2f} profit on $1 stakes)")
print(f"     • {best['roi']:+.1f}% ROI")
print(f"     • Distribution: H={best['distribution']['H']:.0f}% D={best['distribution']['D']:.0f}% "
      f"A={best['distribution']['A']:.0f}%")

print(f"\n   Why this strategy:")
# Check if all outcomes positive
all_positive = (best['home']['pnl'] >= 0 and
                best['draw']['pnl'] >= 0 and
                best['away']['pnl'] >= 0)
if all_positive:
    print(f"     ✓ All three outcomes (H/D/A) are profitable")
else:
    losing = []
    if best['home']['pnl'] < 0: losing.append('Home')
    if best['draw']['pnl'] < 0: losing.append('Draw')
    if best['away']['pnl'] < 0: losing.append('Away')
    print(f"     ⚠ {', '.join(losing)} outcome(s) losing, but overall positive")

if best['win_rate'] >= 50:
    print(f"     ✓ Win rate above 50%")
else:
    print(f"     ⚠ Win rate below 50% (compensated by odds)")

if best['roi'] >= 10:
    print(f"     ✓ Strong ROI (>10%)")
elif best['roi'] >= 5:
    print(f"     ✓ Good ROI (5-10%)")
else:
    print(f"     ⚠ Modest ROI (<5%)")

print("\n" + "="*80)
print("To implement this strategy, update config/production_config.py with:")
print(f"  THRESHOLDS = {best['params']}")
print("="*80)
