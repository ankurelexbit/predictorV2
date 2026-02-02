#!/usr/bin/env python3
"""
Backtest Threshold-Based Strategy
==================================

Strategy:
- HOME Bet: Probability > 48%
- DRAW Bet: Probability > 35%
- AWAY Bet: Probability > 45%

If multiple outcomes exceed thresholds, pick the highest probability.
If none exceed, no bet.

Usage:
    python3 scripts/backtest_threshold_strategy.py \
        --home-threshold 0.48 \
        --draw-threshold 0.35 \
        --away-threshold 0.45
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ThresholdStrategyBacktest:
    """Backtest threshold-based betting strategy."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pipeline = StandaloneLivePipeline(api_key)
        self.pipeline.load_model()

    def fetch_completed_fixtures(self, start_date: str, end_date: str) -> list:
        """Fetch completed fixtures with pagination."""
        print(f"\n📅 Fetching completed fixtures: {start_date} to {end_date}")

        endpoint = f"fixtures/between/{start_date}/{end_date}"
        base_params = {
            'include': 'participants;scores;league;state',
            'filters': 'fixtureStates:5'
        }

        all_fixtures = []
        page = 1

        while True:
            params = base_params.copy()
            params['page'] = page

            data = self.pipeline._api_call(endpoint, params)

            if not data or 'data' not in data:
                break

            fixtures_data = data['data']
            if not fixtures_data:
                break

            if page == 1:
                print(f"📄 Fetching fixtures (starting with page {page})")

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
                    result_label = 'Home Win'
                elif away_score > home_score:
                    result = 'A'
                    result_label = 'Away Win'
                else:
                    result = 'D'
                    result_label = 'Draw'

                all_fixtures.append({
                    'fixture_id': fixture['id'],
                    'starting_at': fixture.get('starting_at'),
                    'league_id': fixture.get('league_id'),
                    'league_name': fixture.get('league', {}).get('name', 'Unknown'),
                    'season_id': fixture.get('season_id'),
                    'home_team_id': home_team['id'],
                    'home_team_name': home_team['name'],
                    'away_team_id': away_team['id'],
                    'away_team_name': away_team['name'],
                    'actual_home_score': home_score,
                    'actual_away_score': away_score,
                    'actual_result': result,
                    'actual_result_label': result_label
                })

            pagination = data.get('pagination', {})
            if not pagination.get('has_more', False):
                break

            page += 1
            print(f"   Fetching page {page}...")

            if page > 1000:
                print(f"⚠️  Reached page limit")
                break

        print(f"✅ Found {len(all_fixtures)} completed fixtures across {page} page(s)")
        return all_fixtures

    def make_predictions(self, fixtures: list) -> pd.DataFrame:
        """Generate predictions."""
        print(f"\n🤖 Generating predictions for {len(fixtures)} fixtures...")

        results = []

        for fixture in tqdm(fixtures, desc="Predicting"):
            features = self.pipeline.generate_features(fixture)

            if not features:
                continue

            features_df = pd.DataFrame([features])
            # Use uncalibrated probabilities (V4 doesn't have calibration wrapper)
            probas = self.pipeline.model.predict_proba(features_df)[0]

            # probas = [away, draw, home]
            away_prob = probas[0]
            draw_prob = probas[1]
            home_prob = probas[2]

            results.append({
                'fixture_id': fixture['fixture_id'],
                'match_date': fixture['starting_at'],
                'league': fixture['league_name'],
                'home_team': fixture['home_team_name'],
                'away_team': fixture['away_team_name'],
                'home_score': fixture['actual_home_score'],
                'away_score': fixture['actual_away_score'],
                'actual_result': fixture['actual_result'],
                'actual_result_label': fixture['actual_result_label'],
                'pred_home_prob': home_prob,
                'pred_draw_prob': draw_prob,
                'pred_away_prob': away_prob,
            })

        df = pd.DataFrame(results)
        print(f"✅ Generated {len(df)} predictions")
        return df

    def apply_threshold_strategy(
        self,
        df: pd.DataFrame,
        home_threshold: float,
        draw_threshold: float,
        away_threshold: float
    ) -> dict:
        """
        Apply threshold-based betting strategy.

        Logic:
        1. Check which outcomes exceed their thresholds
        2. If multiple exceed, pick highest probability
        3. If none exceed, no bet
        """
        print(f"\n💰 Applying Threshold Strategy")
        print("=" * 80)
        print(f"Thresholds: Home>{home_threshold:.0%}, Draw>{draw_threshold:.0%}, Away>{away_threshold:.0%}")
        print("=" * 80)

        total_matches = len(df)
        bets_placed = 0
        no_bet_count = 0
        wins = 0
        total_profit = 0.0
        total_stake = 0.0

        bets_by_outcome = {
            'Home Win': {'bets': 0, 'wins': 0, 'profit': 0.0},
            'Draw': {'bets': 0, 'wins': 0, 'profit': 0.0},
            'Away Win': {'bets': 0, 'wins': 0, 'profit': 0.0}
        }

        bet_details = []

        for _, row in df.iterrows():
            home_prob = row['pred_home_prob']
            draw_prob = row['pred_draw_prob']
            away_prob = row['pred_away_prob']
            actual = row['actual_result']

            # Calculate implied odds (with 5% bookmaker margin)
            home_odds = 1 / home_prob * 0.95 if home_prob > 0.01 else 100
            draw_odds = 1 / draw_prob * 0.95 if draw_prob > 0.01 else 100
            away_odds = 1 / away_prob * 0.95 if away_prob > 0.01 else 100

            # Check which outcomes exceed thresholds
            candidates = []

            if home_prob > home_threshold:
                candidates.append(('Home Win', home_prob, home_odds, 'H'))

            if draw_prob > draw_threshold:
                candidates.append(('Draw', draw_prob, draw_odds, 'D'))

            if away_prob > away_threshold:
                candidates.append(('Away Win', away_prob, away_odds, 'A'))

            # Decision logic
            if len(candidates) == 0:
                # No bet
                no_bet_count += 1
                bet_details.append({
                    'match': f"{row['home_team']} vs {row['away_team']}",
                    'bet_outcome': 'NO_BET',
                    'probability': 0,
                    'odds': 0,
                    'actual_result': row['actual_result_label'],
                    'won': None,
                    'profit': 0
                })
            else:
                # Pick highest probability among candidates
                bet_outcome, bet_prob, bet_odds, bet_code = max(candidates, key=lambda x: x[1])

                bets_placed += 1
                total_stake += 1.0
                bets_by_outcome[bet_outcome]['bets'] += 1

                # Check if bet won
                won = (bet_code == actual)

                if won:
                    profit = (bet_odds - 1) * 1.0
                    wins += 1
                    bets_by_outcome[bet_outcome]['wins'] += 1
                else:
                    profit = -1.0

                total_profit += profit
                bets_by_outcome[bet_outcome]['profit'] += profit

                bet_details.append({
                    'match': f"{row['home_team']} vs {row['away_team']}",
                    'bet_outcome': bet_outcome,
                    'probability': bet_prob,
                    'odds': bet_odds,
                    'actual_result': row['actual_result_label'],
                    'won': won,
                    'profit': profit
                })

        # Calculate metrics
        win_rate = wins / bets_placed if bets_placed > 0 else 0
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
        bet_frequency = bets_placed / total_matches * 100

        # Display results
        print(f"\n📊 Overall Performance:")
        print(f"   Total Matches: {total_matches}")
        print(f"   Bets Placed: {bets_placed} ({bet_frequency:.1f}% of matches)")
        print(f"   No Bets: {no_bet_count} ({no_bet_count/total_matches*100:.1f}% of matches)")
        print(f"   Wins: {wins}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Total Stake: ${total_stake:.2f}")
        print(f"   Total Return: ${total_stake + total_profit:.2f}")
        print(f"   Net Profit: ${total_profit:+.2f}")
        print(f"   ROI: {roi:+.1f}%")

        print(f"\n📊 Performance by Bet Type:")
        print("-" * 80)
        for outcome in ['Home Win', 'Draw', 'Away Win']:
            stats = bets_by_outcome[outcome]
            if stats['bets'] > 0:
                win_rate_outcome = stats['wins'] / stats['bets']
                roi_outcome = (stats['profit'] / stats['bets'] * 100)
                print(f"   {outcome}:")
                print(f"      Bets: {stats['bets']}")
                print(f"      Wins: {stats['wins']} ({win_rate_outcome:.1%})")
                print(f"      Profit: ${stats['profit']:+.2f}")
                print(f"      ROI: {roi_outcome:+.1f}%")

        return {
            'total_matches': total_matches,
            'bets_placed': bets_placed,
            'no_bet_count': no_bet_count,
            'wins': wins,
            'win_rate': win_rate,
            'total_stake': total_stake,
            'total_profit': total_profit,
            'roi': roi,
            'bet_frequency': bet_frequency,
            'bets_by_outcome': bets_by_outcome,
            'bet_details': bet_details
        }


def main():
    parser = argparse.ArgumentParser(description='Threshold strategy backtest')
    parser.add_argument('--start-date', default='2026-01-01')
    parser.add_argument('--end-date', default='2026-01-31')
    parser.add_argument('--home-threshold', type=float, default=0.48, help='Home win threshold (default: 0.48)')
    parser.add_argument('--draw-threshold', type=float, default=0.35, help='Draw threshold (default: 0.35)')
    parser.add_argument('--away-threshold', type=float, default=0.45, help='Away win threshold (default: 0.45)')
    parser.add_argument('--output', default='results/backtest_threshold_strategy.csv')

    args = parser.parse_args()

    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        print("❌ SPORTMONKS_API_KEY not set")
        sys.exit(1)

    print("=" * 80)
    print("THRESHOLD-BASED STRATEGY BACKTEST")
    print("=" * 80)
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"\nStrategy:")
    print(f"  HOME Bet:  Probability > {args.home_threshold:.0%}")
    print(f"  DRAW Bet:  Probability > {args.draw_threshold:.0%}")
    print(f"  AWAY Bet:  Probability > {args.away_threshold:.0%}")
    print(f"\nConflict Resolution: Pick highest probability")
    print(f"No Threshold Met: NO_BET")
    print("=" * 80)

    # Initialize
    engine = ThresholdStrategyBacktest(api_key)

    # Fetch fixtures
    fixtures = engine.fetch_completed_fixtures(args.start_date, args.end_date)

    if len(fixtures) == 0:
        print("❌ No fixtures found")
        sys.exit(1)

    # Limit to first 50 fixtures for quick comparison
    print(f"⚡ Limiting to first 50 fixtures for quick comparison")
    fixtures = fixtures[:50]

    # Generate predictions
    predictions_df = engine.make_predictions(fixtures)

    # Apply strategy
    results = engine.apply_threshold_strategy(
        predictions_df,
        args.home_threshold,
        args.draw_threshold,
        args.away_threshold
    )

    # Save predictions with bet decisions
    bet_details_df = pd.DataFrame(results['bet_details'])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bet_details_df.to_csv(output_path, index=False)

    print(f"\n✅ Bet details saved to: {args.output}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
