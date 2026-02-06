#!/usr/bin/env python3
"""
Find optimal betting strategy with constraints:
1. Win rate > 50%
2. Maximize PnL
3. Min 10% distribution for each H/D/A
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
print("CONSTRAINED OPTIMIZATION")
print("Constraints: WR>50%, PnL maximized, each outcome ≥10% distribution")
print("="*80)

# ============================================================================
# Load calibrators and data
# ============================================================================
print("\n[1/4] Loading calibrators and data...")

# Test both existing and fresh calibrators
existing_calibrators = joblib.load('models/calibrators.joblib')

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Fit fresh calibrators on Oct-Dec
cur.execute("""
    SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result
    FROM predictions
    WHERE match_date >= '2025-10-01' AND match_date < '2026-01-01'
      AND actual_result IS NOT NULL
      AND pred_home_prob IS NOT NULL
    ORDER BY match_date
""")
cal_rows = cur.fetchall()

fresh_calibrators = {}
for outcome, col in [('home', 'pred_home_prob'), ('draw', 'pred_draw_prob'), ('away', 'pred_away_prob')]:
    X = np.array([r[col] for r in cal_rows])
    y = np.array([1.0 if r['actual_result'] == outcome[0].upper() else 0.0 for r in cal_rows])
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(X, y)
    fresh_calibrators[outcome] = cal

# Load Nov-Jan test data
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

# Apply both calibrators
for row in test_rows:
    row['exist_home'] = float(existing_calibrators['home'].predict([row['pred_home_prob']])[0])
    row['exist_draw'] = float(existing_calibrators['draw'].predict([row['pred_draw_prob']])[0])
    row['exist_away'] = float(existing_calibrators['away'].predict([row['pred_away_prob']])[0])

    row['fresh_home'] = float(fresh_calibrators['home'].predict([row['pred_home_prob']])[0])
    row['fresh_draw'] = float(fresh_calibrators['draw'].predict([row['pred_draw_prob']])[0])
    row['fresh_away'] = float(fresh_calibrators['away'].predict([row['pred_away_prob']])[0])

print(f"   Loaded {len(test_rows)} test predictions")

# ============================================================================
# Define strategies
# ============================================================================

def pure_threshold_strategy(test_data, thresholds, cal_prefix):
    """Standard pure threshold strategy."""
    bets = []
    for row in test_data:
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

    return bets

def quota_enforced_strategy(test_data, thresholds, cal_prefix, min_pct=10.0):
    """
    Enforce minimum quota for each outcome.
    First pass: collect all candidate bets per outcome
    Second pass: select bets to meet quotas while maximizing PnL
    """
    # Collect all candidates by outcome
    candidates_by_outcome = {'H': [], 'D': [], 'A': []}

    for i, row in enumerate(test_data):
        home_prob = row[f'{cal_prefix}_home']
        draw_prob = row[f'{cal_prefix}_draw']
        away_prob = row[f'{cal_prefix}_away']

        if home_prob > thresholds['home'] and row['best_home_odds']:
            ev = (home_prob * row['best_home_odds']) - 1
            candidates_by_outcome['H'].append({
                'idx': i, 'prob': home_prob, 'odds': row['best_home_odds'],
                'ev': ev, 'won': (row['actual_result'] == 'H')
            })
        if draw_prob > thresholds['draw'] and row['best_draw_odds']:
            ev = (draw_prob * row['best_draw_odds']) - 1
            candidates_by_outcome['D'].append({
                'idx': i, 'prob': draw_prob, 'odds': row['best_draw_odds'],
                'ev': ev, 'won': (row['actual_result'] == 'D')
            })
        if away_prob > thresholds['away'] and row['best_away_odds']:
            ev = (away_prob * row['best_away_odds']) - 1
            candidates_by_outcome['A'].append({
                'idx': i, 'prob': away_prob, 'odds': row['best_away_odds'],
                'ev': ev, 'won': (row['actual_result'] == 'A')
            })

    # Sort each outcome by EV (highest first)
    for outcome in ['H', 'D', 'A']:
        candidates_by_outcome[outcome].sort(key=lambda x: x['ev'], reverse=True)

    # Calculate minimum bets per outcome
    total_candidates = sum(len(candidates_by_outcome[o]) for o in ['H', 'D', 'A'])
    if total_candidates == 0:
        return []

    min_bets_per_outcome = int(total_candidates * min_pct / 100)

    # Select top bets from each outcome to meet quota
    selected = []
    for outcome in ['H', 'D', 'A']:
        available = candidates_by_outcome[outcome]
        if len(available) < min_bets_per_outcome:
            # Can't meet quota - take all available
            selected.extend(available)
        else:
            # Take top N to meet quota
            selected.extend(available[:min_bets_per_outcome])

    # Convert to bet format
    bets = []
    for bet in selected:
        bets.append({
            'outcome': next(o for o in ['H', 'D', 'A']
                          if bet in candidates_by_outcome[o]),
            'odds': bet['odds'],
            'won': bet['won']
        })

    return bets

def calculate_metrics(bets):
    """Calculate performance metrics from bets."""
    if not bets:
        return None

    n = len(bets)
    wins = sum(1 for b in bets if b['won'])
    pnl = sum((b['odds']-1) if b['won'] else -1 for b in bets)

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
        'away': outcome_data['A'],
        'distribution': {
            'H': outcome_data['H']['bets']/n*100,
            'D': outcome_data['D']['bets']/n*100,
            'A': outcome_data['A']['bets']/n*100
        }
    }

# ============================================================================
# Test configurations
# ============================================================================
print("\n[2/4] Testing threshold configurations...")

results = []

# Grid search over thresholds
home_thresholds = [0.32, 0.36, 0.40, 0.44, 0.48]
draw_thresholds = [0.20, 0.22, 0.24, 0.26, 0.28]
away_thresholds = [0.32, 0.36, 0.40, 0.44]

total_configs = len(home_thresholds) * len(draw_thresholds) * len(away_thresholds) * 2 * 2
print(f"   Testing {total_configs} configurations...")

count = 0
for h_thresh in home_thresholds:
    for d_thresh in draw_thresholds:
        for a_thresh in away_thresholds:
            thresholds = {'home': h_thresh, 'draw': d_thresh, 'away': a_thresh}

            # Test with both calibrators
            for cal_name, cal_prefix in [('Fresh', 'fresh'), ('Existing', 'exist')]:
                # Strategy 1: Pure threshold
                bets = pure_threshold_strategy(test_rows, thresholds, cal_prefix)
                metrics = calculate_metrics(bets)

                if metrics:
                    metrics['strategy'] = f'Pure Threshold ({cal_name} Cal)'
                    metrics['thresholds'] = thresholds
                    results.append(metrics)

                # Strategy 2: Quota enforced (min 10% each)
                bets = quota_enforced_strategy(test_rows, thresholds, cal_prefix, min_pct=10.0)
                metrics = calculate_metrics(bets)

                if metrics:
                    metrics['strategy'] = f'Quota 10% ({cal_name} Cal)'
                    metrics['thresholds'] = thresholds
                    results.append(metrics)

            count += 1
            if count % 20 == 0:
                print(f"      Progress: {count}/{len(home_thresholds) * len(draw_thresholds) * len(away_thresholds)} threshold combos")

# ============================================================================
# Filter by constraints
# ============================================================================
print("\n[3/4] Filtering by constraints...")

# Constraint 1: Win rate > 50%
results = [r for r in results if r['win_rate'] > 50.0]
print(f"   After WR>50% filter: {len(results)} strategies")

# Constraint 2: Min 10% distribution for each outcome
results = [r for r in results if
           r['distribution']['H'] >= 10.0 and
           r['distribution']['D'] >= 10.0 and
           r['distribution']['A'] >= 10.0]
print(f"   After distribution≥10% filter: {len(results)} strategies")

if not results:
    print("\n❌ NO STRATEGIES meet all constraints!")
    print("\nRelaxing constraints to find closest matches...")

    # Reload all results
    results = []
    for h_thresh in home_thresholds:
        for d_thresh in draw_thresholds:
            for a_thresh in away_thresholds:
                thresholds = {'home': h_thresh, 'draw': d_thresh, 'away': a_thresh}
                for cal_name, cal_prefix in [('Fresh', 'fresh'), ('Existing', 'exist')]:
                    bets = pure_threshold_strategy(test_rows, thresholds, cal_prefix)
                    metrics = calculate_metrics(bets)
                    if metrics:
                        metrics['strategy'] = f'Pure Threshold ({cal_name} Cal)'
                        metrics['thresholds'] = thresholds
                        results.append(metrics)

                    bets = quota_enforced_strategy(test_rows, thresholds, cal_prefix, min_pct=10.0)
                    metrics = calculate_metrics(bets)
                    if metrics:
                        metrics['strategy'] = f'Quota 10% ({cal_name} Cal)'
                        metrics['thresholds'] = thresholds
                        results.append(metrics)

    # Show strategies closest to meeting constraints
    print("\n   Top strategies by PnL (may not meet all constraints):")
    results.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"\n{'Strategy':<30} {'H/D/A Thresh':<20} {'Bets':>5} {'WR%':>5} {'PnL':>7} "
          f"{'ROI%':>6} {'H%':>4} {'D%':>4} {'A%':>4}")
    print("-"*95)

    for r in results[:15]:
        thresh = r['thresholds']
        meets_wr = "✓" if r['win_rate'] > 50 else "✗"
        meets_dist = "✓" if (r['distribution']['H'] >= 10 and
                             r['distribution']['D'] >= 10 and
                             r['distribution']['A'] >= 10) else "✗"

        print(f"{r['strategy']:<30} "
              f"{thresh['home']:.2f}/{thresh['draw']:.2f}/{thresh['away']:.2f}        "
              f"{r['total_bets']:>5} {r['win_rate']:>5.1f} {r['total_pnl']:>+7.2f} "
              f"{r['roi']:>+6.1f} {r['distribution']['H']:>4.0f} {r['distribution']['D']:>4.0f} "
              f"{r['distribution']['A']:>4.0f}  {meets_wr}{meets_dist}")

    sys.exit(0)

# ============================================================================
# Show results sorted by PnL
# ============================================================================
print("\n[4/4] Results meeting ALL constraints:")
print("="*80)

results.sort(key=lambda x: x['total_pnl'], reverse=True)

print(f"\n{'Strategy':<30} {'H/D/A Thresh':<20} {'Bets':>5} {'WR%':>5} {'PnL':>7} "
      f"{'ROI%':>6} {'H%':>4} {'D%':>4} {'A%':>4}")
print("-"*95)

for r in results[:10]:
    thresh = r['thresholds']
    print(f"{r['strategy']:<30} "
          f"{thresh['home']:.2f}/{thresh['draw']:.2f}/{thresh['away']:.2f}        "
          f"{r['total_bets']:>5} {r['win_rate']:>5.1f} {r['total_pnl']:>+7.2f} "
          f"{r['roi']:>+6.1f} {r['distribution']['H']:>4.0f} {r['distribution']['D']:>4.0f} "
          f"{r['distribution']['A']:>4.0f}")

# ============================================================================
# Detailed view of top 3
# ============================================================================
print("\n" + "="*80)
print("TOP 3 STRATEGIES - DETAILED BREAKDOWN")
print("="*80)

for i, r in enumerate(results[:3], 1):
    thresh = r['thresholds']
    print(f"\n#{i} — {r['strategy']}: H={thresh['home']:.2f} D={thresh['draw']:.2f} A={thresh['away']:.2f}")
    print("-"*80)
    print(f"  Total: {r['total_bets']} bets, {r['total_wins']} wins, {r['win_rate']:.1f}% WR, "
          f"{r['total_pnl']:+.2f} PnL, {r['roi']:+.1f}% ROI")
    print(f"  Distribution: H={r['distribution']['H']:.1f}% D={r['distribution']['D']:.1f}% "
          f"A={r['distribution']['A']:.1f}%")
    print(f"\n  Per-Outcome Breakdown:")
    print(f"    Home: {r['home']['bets']} bets, {r['home']['wins']} wins, {r['home']['pnl']:+.2f} PnL")
    print(f"    Draw: {r['draw']['bets']} bets, {r['draw']['wins']} wins, {r['draw']['pnl']:+.2f} PnL")
    print(f"    Away: {r['away']['bets']} bets, {r['away']['wins']} wins, {r['away']['pnl']:+.2f} PnL")

print("\n" + "="*80)
