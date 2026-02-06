#!/usr/bin/env python3
"""
Generate fresh predictions for EV-based strategy backtest.

1. Loads unweighted model v2.0.0
2. Generates predictions for Oct-Dec 2025 (calibration set)
3. Generates predictions for Jan 2026 (test set)
4. Stores all to predictions_ev_backtest table
5. Fits calibrators on Oct-Dec results
6. Tests EV strategy on Jan with multiple min_ev thresholds
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import psycopg2.extras
import requests
import json
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scripts.predict_live_with_history import ProductionLivePipeline

# Get credentials
API_KEY = os.environ.get('SPORTMONKS_API_KEY')
DATABASE_URL = os.environ.get('DATABASE_URL')

if not API_KEY:
    with open('.env') as f:
        for line in f:
            if line.startswith('SPORTMONKS_API_KEY='):
                API_KEY = line.split('=', 1)[1].strip().strip('"')
                break

if not DATABASE_URL:
    with open('.env') as f:
        for line in f:
            if line.startswith('DATABASE_URL='):
                DATABASE_URL = line.split('=', 1)[1].strip().strip('"')
                break

if not API_KEY or not DATABASE_URL:
    print("ERROR: Missing SPORTMONKS_API_KEY or DATABASE_URL")
    sys.exit(1)

print("="*80)
print("EV BACKTEST - FRESH PREDICTIONS WITH MODEL v2.0.0 (UNWEIGHTED)")
print("="*80)

# ============================================================================
# STEP 1: Initialize pipeline and load model
# ============================================================================
print("\n[1/6] Initializing pipeline with historical data...")
pipeline = ProductionLivePipeline(API_KEY, load_history_days=365)

model_path = 'models/production/model_v2.0.0.joblib'
print(f"\n[2/6] Loading model: {model_path}")
pipeline.load_model(model_path)

# ============================================================================
# STEP 2: Fetch fixtures from API
# ============================================================================
def fetch_fixtures(start_date, end_date):
    """Fetch finished fixtures for date range."""
    print(f"   Fetching {start_date} to {end_date}...")

    base_url = "https://api.sportmonks.com/v3/football"
    endpoint = f"fixtures/between/{start_date}/{end_date}"

    params = {
        'api_token': API_KEY,
        'include': 'participants;league;state;odds',
        'filters': 'fixtureStates:5',  # Only finished matches
        'per_page': 100
    }

    all_fixtures = []
    page = 1

    while True:
        params['page'] = page
        response = requests.get(f"{base_url}/{endpoint}", params=params, verify=False, timeout=30)
        data = response.json()

        if page == 1:
            print(f"      API status: {response.status_code}")

        if not data or 'data' not in data or not data['data']:
            break

        fixtures = data['data']
        all_fixtures.extend(fixtures)

        pagination = data.get('pagination', {})
        if not pagination.get('has_more', False):
            break

        page += 1
        if page > 50:
            break

    print(f"      Found {len(all_fixtures)} fixtures")
    return all_fixtures

print("\n[3/6] Fetching fixtures from SportMonks API...")
oct_fixtures = fetch_fixtures('2025-10-01', '2025-10-31')
nov_fixtures = fetch_fixtures('2025-11-01', '2025-11-30')
dec_fixtures = fetch_fixtures('2025-12-01', '2025-12-31')
jan_fixtures = fetch_fixtures('2026-01-01', '2026-01-31')

calibration_fixtures = oct_fixtures + nov_fixtures + dec_fixtures
test_fixtures = jan_fixtures

# Filter to Top 5 leagues (same as production config)
TOP_5_LEAGUES = [8, 82, 384, 564, 301]  # Premier League, Bundesliga, Serie A, Ligue 1, La Liga

print(f"\n   Before filtering: {len(calibration_fixtures)} calibration, {len(test_fixtures)} test")

calibration_fixtures = [f for f in calibration_fixtures if f.get('league_id') in TOP_5_LEAGUES]
test_fixtures = [f for f in test_fixtures if f.get('league_id') in TOP_5_LEAGUES]

print(f"   After Top 5 filter: {len(calibration_fixtures)} calibration, {len(test_fixtures)} test")

print(f"   (Leagues: Premier League, Bundesliga, Serie A, Ligue 1, La Liga)")

# ============================================================================
# STEP 3: Generate predictions and store
# ============================================================================
def extract_odds(fixture):
    """Extract best and avg odds from fixture."""
    odds_data = fixture.get('odds', [])
    home_odds_list, draw_odds_list, away_odds_list = [], [], []

    for odds_item in odds_data:
        if odds_item.get('market_id') != 1:
            continue

        label = odds_item.get('label', '').lower()
        value = odds_item.get('value')

        if value is None:
            continue

        try:
            value = float(value)
            if label in ['1', 'home', 'home win']:
                home_odds_list.append(value)
            elif label in ['x', 'draw']:
                draw_odds_list.append(value)
            elif label in ['2', 'away', 'away win']:
                away_odds_list.append(value)
        except:
            continue

    return {
        'best_home_odds': max(home_odds_list) if home_odds_list else None,
        'best_draw_odds': max(draw_odds_list) if draw_odds_list else None,
        'best_away_odds': max(away_odds_list) if away_odds_list else None,
        'avg_home_odds': sum(home_odds_list)/len(home_odds_list) if home_odds_list else None,
        'avg_draw_odds': sum(draw_odds_list)/len(draw_odds_list) if draw_odds_list else None,
        'avg_away_odds': sum(away_odds_list)/len(away_odds_list) if away_odds_list else None,
        'odds_count': len(home_odds_list)
    }

def get_actual_result(fixture):
    """Extract actual result from fixture."""
    participants = fixture.get('participants', [])
    home_score = away_score = None

    for p in participants:
        meta = p.get('meta', {})
        if meta.get('location') == 'home':
            home_score = meta.get('score')
        elif meta.get('location') == 'away':
            away_score = meta.get('score')

    if home_score is not None and away_score is not None:
        if home_score > away_score:
            return 'H'
        elif away_score > home_score:
            return 'A'
        else:
            return 'D'
    return None

def generate_and_store(fixtures, phase_name):
    """Generate predictions and store to DB."""
    import time
    print(f"\n   Generating predictions for {phase_name}...")
    predictions = []
    start_time = time.time()
    last_print_time = start_time

    for i, fixture in enumerate(fixtures, 1):
        current_time = time.time()

        # Print progress every 10 fixtures or every 30 seconds
        if i % 10 == 1 or (current_time - last_print_time) > 30:
            elapsed = current_time - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(fixtures) - i) / rate if rate > 0 else 0

            print(f"      {i}/{len(fixtures)} ({i/len(fixtures)*100:.1f}%) | "
                  f"Rate: {rate:.1f} fix/sec | "
                  f"ETA: {remaining/60:.1f} min | "
                  f"Generated: {len(predictions)} predictions")
            last_print_time = current_time

        try:
            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            # Run prediction
            result = pipeline.predict(fixture)
            if not result:
                continue

            # Extract odds and actual result
            odds = extract_odds(fixture)
            actual = get_actual_result(fixture)

            # Skip if no odds available
            if not odds['best_home_odds']:
                continue

            # Determine predicted outcome
            max_prob = max(result['home_prob'], result['draw_prob'], result['away_prob'])
            if max_prob == result['home_prob']:
                predicted = 'H'
            elif max_prob == result['away_prob']:
                predicted = 'A'
            else:
                predicted = 'D'

            pred = {
                'fixture_id': fixture['id'],
                'match_date': fixture.get('starting_at'),
                'league_id': fixture.get('league_id'),
                'league_name': fixture.get('league', {}).get('name'),
                'season_id': fixture.get('season_id'),
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'pred_home_prob': result['home_prob'],
                'pred_draw_prob': result['draw_prob'],
                'pred_away_prob': result['away_prob'],
                'predicted_outcome': predicted,
                'model_version': 'v2.0.0_unweighted',
                **odds,
                'features': result.get('features', {}),
                'actual_result': actual
            }

            predictions.append(pred)

        except Exception as e:
            if i <= 3:
                print(f"      Error on fixture {fixture.get('id')}: {e}")
            continue

    # Store to DB
    if predictions:
        print(f"      Storing {len(predictions)} predictions to database...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor()

        for pred in predictions:
            cur.execute("""
                INSERT INTO predictions_ev_backtest (
                    fixture_id, match_date, league_id, league_name, season_id,
                    home_team_id, home_team_name, away_team_id, away_team_name,
                    pred_home_prob, pred_draw_prob, pred_away_prob, predicted_outcome,
                    model_version, best_home_odds, best_draw_odds, best_away_odds,
                    avg_home_odds, avg_draw_odds, avg_away_odds, odds_count,
                    features, actual_result
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                pred['fixture_id'], pred['match_date'], pred['league_id'], pred['league_name'],
                pred['season_id'], pred['home_team_id'], pred['home_team_name'],
                pred['away_team_id'], pred['away_team_name'], pred['pred_home_prob'],
                pred['pred_draw_prob'], pred['pred_away_prob'], pred['predicted_outcome'],
                pred['model_version'], pred['best_home_odds'], pred['best_draw_odds'],
                pred['best_away_odds'], pred['avg_home_odds'], pred['avg_draw_odds'],
                pred['avg_away_odds'], pred['odds_count'], json.dumps(pred['features']),
                pred['actual_result']
            ))

        cur.close()
        conn.close()
        print(f"      ✅ Stored successfully")

    return len(predictions)

print("\n[4/6] Generating predictions...")
cal_count = generate_and_store(calibration_fixtures, "Oct-Dec 2025 (calibration)")
test_count = generate_and_store(test_fixtures, "Jan 2026 (test)")

print(f"\n   Calibration predictions: {cal_count}")
print(f"   Test predictions: {test_count}")

# ============================================================================
# STEP 4: Fit calibrators on Oct-Dec
# ============================================================================
print("\n[5/6] Fitting isotonic calibrators on Oct-Dec predictions...")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

cur.execute("""
    SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result
    FROM predictions_ev_backtest
    WHERE match_date >= '2025-10-01' AND match_date < '2026-01-01'
      AND actual_result IS NOT NULL
    ORDER BY match_date
""")
cal_rows = cur.fetchall()

print(f"   Loaded {len(cal_rows)} calibration predictions")

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

    print(f"   {outcome.capitalize():5} — Actual: {actual_rate:.3f} | Raw: {raw_mean:.3f} | Calibrated: {cal_mean:.3f} | MAE: {mae_before:.3f}→{mae_after:.3f}")

# ============================================================================
# STEP 5: Test EV strategy on Jan
# ============================================================================
print("\n[6/6] Testing EV-based strategy on Jan 2026...")

cur.execute("""
    SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result,
           best_home_odds, best_draw_odds, best_away_odds
    FROM predictions_ev_backtest
    WHERE match_date >= '2026-01-01'
      AND actual_result IS NOT NULL
      AND best_home_odds IS NOT NULL
    ORDER BY match_date
""")
test_rows = cur.fetchall()
cur.close()
conn.close()

print(f"   Loaded {len(test_rows)} test predictions\n")

def ev_strategy(cal_h, cal_d, cal_a, h_odds, d_odds, a_odds, min_ev=0.05):
    """EV-based selection: pick highest EV above threshold."""
    candidates = []

    ev_h = (cal_h * h_odds) - 1
    ev_d = (cal_d * d_odds) - 1
    ev_a = (cal_a * a_odds) - 1

    if ev_h > min_ev: candidates.append(('H', ev_h, cal_h, h_odds))
    if ev_d > min_ev: candidates.append(('D', ev_d, cal_d, d_odds))
    if ev_a > min_ev: candidates.append(('A', ev_a, cal_a, a_odds))

    return max(candidates, key=lambda x: x[1]) if candidates else None

print("="*80)
print("EV STRATEGY RESULTS (Jan 2026 Test Set)")
print("="*80)

ev_thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

for min_ev in ev_thresholds:
    bets = []

    for r in test_rows:
        cal_h = float(calibrators['home'].predict([r['pred_home_prob']])[0])
        cal_d = float(calibrators['draw'].predict([r['pred_draw_prob']])[0])
        cal_a = float(calibrators['away'].predict([r['pred_away_prob']])[0])

        result = ev_strategy(cal_h, cal_d, cal_a,
                            r['best_home_odds'], r['best_draw_odds'], r['best_away_odds'],
                            min_ev=min_ev)

        if result:
            outcome, ev, prob, odds = result
            won = (outcome == r['actual_result'])
            bets.append({'o': outcome, 'ev': ev, 'odds': odds, 'won': won})

    if not bets:
        print(f"\nmin_ev = {min_ev:.2f}: No bets")
        continue

    n = len(bets)
    wins = sum(1 for b in bets if b['won'])
    pnl = sum((b['odds']-1) if b['won'] else -1 for b in bets)

    outcome_stats = {}
    for o in ['H', 'D', 'A']:
        subset = [b for b in bets if b['o'] == o]
        if subset:
            ns = len(subset)
            ws = sum(1 for b in subset if b['won'])
            ps = sum((b['odds']-1) if b['won'] else -1 for b in subset)
            outcome_stats[o] = {
                'n': ns, 'w': ws, 'pnl': ps,
                'avg_odds': sum(b['odds'] for b in subset)/ns,
                'avg_ev': sum(b['ev'] for b in subset)/ns*100
            }

    print(f"\n{'─'*80}")
    print(f"min_ev = {min_ev:.2f} ({min_ev*100:.0f}% minimum EV) — {n} bets, {wins} wins, {wins/n*100:.1f}% WR")
    print(f"{'─'*80}")
    print(f"  Out  Bets Wins   PnL   AvgOdds  AvgEV%  ROI%   Share")

    for o in ['H', 'D', 'A']:
        s = outcome_stats.get(o)
        if s:
            print(f"  {o}   {s['n']:>4} {s['w']:>4} {s['pnl']:>+6.2f} {s['avg_odds']:>7.2f}x {s['avg_ev']:>6.1f}% {s['pnl']/s['n']*100:>+5.1f}% {s['n']/n*100:>5.1f}%")
        else:
            print(f"  {o}      0    0     —      —       —      —    0.0%")

    print(f"  {'─'*78}")
    print(f"  TOT {n:>4} {wins:>4} {pnl:>+6.2f}                      {pnl/n*100:>+5.1f}%")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"All predictions stored in table: predictions_ev_backtest")
print(f"Calibration set: {cal_count} predictions (Oct-Dec 2025)")
print(f"Test set: {test_count} predictions (Jan 2026)")
