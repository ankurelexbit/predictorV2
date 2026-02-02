#!/usr/bin/env python3
"""
Backtest V2 Model on January 2026
==================================

Tests V2's XGBoost model with calibration on the same January 2026 data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import requests

sys.path.insert(0, str(Path(__file__).parent))

from predict_live import LiveFeatureCalculator, load_models
from production_thresholds import get_production_thresholds

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "PeZeQDLtEN57cNh6q0e97drjLgCygFYV8BQRSuyg91fa8krbpmlX658H73r8"
BASE_URL = "https://api.sportmonks.com/v3/football"


def fetch_completed_fixtures(start_date, end_date):
    """Fetch completed fixtures with pagination."""
    print(f"\n📅 Fetching completed fixtures: {start_date} to {end_date}")

    endpoint = f"{BASE_URL}/fixtures/between/{start_date}/{end_date}"
    all_fixtures = []
    page = 1

    while True:
        params = {
            'api_token': API_KEY,
            'include': 'participants;scores;league;state',
            'filters': 'fixtureStates:5',
            'page': page
        }

        response = requests.get(endpoint, params=params, verify=False, timeout=30)
        data = response.json()

        if not data or 'data' not in data:
            break

        fixtures_data = data['data']
        if not fixtures_data:
            break

        if page == 1:
            print(f"📄 Fetching fixtures (found {len(fixtures_data)} on first page)")

        for fixture in fixtures_data:
            if fixture.get('state_id') != 5:
                continue

            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            scores = fixture.get('scores', [])
            home_score = None
            away_score = None

            for score in scores:
                if score.get('description') == 'CURRENT':
                    score_data = score.get('score', {})
                    if score_data.get('participant') == 'home':
                        home_score = score_data.get('goals')
                    elif score_data.get('participant') == 'away':
                        away_score = score_data.get('goals')

            if home_score is None or away_score is None:
                continue

            if home_score > away_score:
                result = 'H'
            elif away_score > home_score:
                result = 'A'
            else:
                result = 'D'

            all_fixtures.append({
                'fixture_id': fixture['id'],
                'starting_at': fixture.get('starting_at'),
                'league_id': fixture.get('league_id'),
                'league_name': fixture.get('league', {}).get('name', 'Unknown'),
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'home_score': home_score,
                'away_score': away_score,
                'actual_result': result
            })

        pagination = data.get('pagination', {})
        if not pagination.get('has_more', False):
            break

        page += 1
        print(f"   Fetching page {page}...")

        if page > 1000:
            break

    print(f"✅ Found {len(all_fixtures)} completed fixtures")
    return all_fixtures


def main():
    print("=" * 80)
    print("V2 MODEL BACKTEST - JANUARY 2026")
    print("=" * 80)

    # Load V2 model and thresholds
    print("\nLoading V2 model...")
    models = load_models('xgboost')
    if 'xgboost' not in models:
        print("❌ Failed to load model")
        return

    model = models['xgboost']
    thresholds = get_production_thresholds()

    print(f"Thresholds: Home={thresholds['home']:.2f}, Draw={thresholds['draw']:.2f}, Away={thresholds['away']:.2f}")
    print(f"Model features: 271 (V2 pipeline)")

    # Fetch fixtures
    fixtures = fetch_completed_fixtures('2026-01-01', '2026-01-31')

    if len(fixtures) == 0:
        print("❌ No fixtures found")
        return

    # Limit to first 50 fixtures for quick sample
    print(f"⚡ Limiting to first 50 fixtures for quick comparison")
    fixtures = fixtures[:50]

    # Generate predictions
    print(f"\n🤖 Generating predictions for {len(fixtures)} fixtures...")
    calculator = LiveFeatureCalculator()

    results = []
    errors = 0

    for fixture in tqdm(fixtures, desc="Predicting"):
        try:
            result = calculator.build_features_for_match(
                home_team_id=int(fixture['home_team_id']),
                away_team_id=int(fixture['away_team_id']),
                fixture_date=pd.to_datetime(fixture['starting_at']),
                home_team_name=fixture['home_team_name'],
                away_team_name=fixture['away_team_name'],
                league_name=fixture['league_name'],
                fixture_id=fixture['fixture_id']
            )

            if not result:
                errors += 1
                continue

            features, metadata = result
            features_df = pd.DataFrame([features])

            # Use CALIBRATED probabilities (V2's key feature)
            probs = model.predict_proba(features_df, calibrated=True)[0]
            p_away, p_draw, p_home = probs

            results.append({
                'fixture_id': fixture['fixture_id'],
                'home_team': fixture['home_team_name'],
                'away_team': fixture['away_team_name'],
                'actual_result': fixture['actual_result'],
                'p_home': p_home,
                'p_draw': p_draw,
                'p_away': p_away
            })

        except Exception as e:
            errors += 1
            continue

    print(f"✅ Generated {len(results)} predictions ({errors} errors)")

    # Apply V2 thresholds
    print(f"\n💰 Applying V2 Strategy (Home>{thresholds['home']:.0%}, Draw>{thresholds['draw']:.0%}, Away>{thresholds['away']:.0%})")
    print("=" * 80)

    total_bets = 0
    wins = 0
    total_profit = 0.0

    bets_by_outcome = {
        'Home': {'bets': 0, 'wins': 0, 'profit': 0.0},
        'Draw': {'bets': 0, 'wins': 0, 'profit': 0.0},
        'Away': {'bets': 0, 'wins': 0, 'profit': 0.0}
    }

    for row in results:
        p_home = row['p_home']
        p_draw = row['p_draw']
        p_away = row['p_away']
        actual = row['actual_result']

        # Calculate odds (with 5% margin)
        home_odds = 1 / p_home * 0.95 if p_home > 0.01 else 100
        draw_odds = 1 / p_draw * 0.95 if p_draw > 0.01 else 100
        away_odds = 1 / p_away * 0.95 if p_away > 0.01 else 100

        # Find which outcomes exceed thresholds
        candidates = []
        if p_home > thresholds['home']:
            candidates.append(('Home', p_home, home_odds, 'H'))
        if p_draw > thresholds['draw']:
            candidates.append(('Draw', p_draw, draw_odds, 'D'))
        if p_away > thresholds['away']:
            candidates.append(('Away', p_away, away_odds, 'A'))

        # Pick highest probability if multiple candidates
        if candidates:
            bet_outcome, bet_prob, bet_odds, bet_code = max(candidates, key=lambda x: x[1])

            total_bets += 1
            bets_by_outcome[bet_outcome]['bets'] += 1

            won = (bet_code == actual)
            if won:
                profit = (bet_odds - 1) * 1.0
                wins += 1
                bets_by_outcome[bet_outcome]['wins'] += 1
            else:
                profit = -1.0

            total_profit += profit
            bets_by_outcome[bet_outcome]['profit'] += profit

    # Display results
    win_rate = wins / total_bets if total_bets > 0 else 0
    roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

    print(f"\n📊 V2 Model Performance:")
    print(f"   Total Matches: {len(results)}")
    print(f"   Bets Placed: {total_bets} ({total_bets/len(results)*100:.1f}% of matches)")
    print(f"   Wins: {wins}")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Total Stake: ${total_bets:.2f}")
    print(f"   Net Profit: ${total_profit:+.2f}")
    print(f"   ROI: {roi:+.1f}%")

    print(f"\n📊 Performance by Bet Type:")
    print("-" * 80)
    for outcome in ['Home', 'Draw', 'Away']:
        stats = bets_by_outcome[outcome]
        if stats['bets'] > 0:
            wr = stats['wins'] / stats['bets']
            outcome_roi = (stats['profit'] / stats['bets'] * 100)
            print(f"   {outcome}:")
            print(f"      Bets: {stats['bets']}, Wins: {stats['wins']} ({wr:.1%})")
            print(f"      Profit: ${stats['profit']:+.2f}, ROI: {outcome_roi:+.1f}%")

    print("\n" + "=" * 80)
    print("V2 BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
