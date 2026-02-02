#!/usr/bin/env python3
"""
Backtest January 2026 Predictions - Multi-Outcome Strategy
============================================================

Tests independent thresholds for Home/Draw/Away predictions.
Each outcome is evaluated separately (not just highest probability).

Usage:
    python3 scripts/backtest_multioutcome_january_2026.py \
        --start-date 2026-01-01 \
        --end-date 2026-01-31
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
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MultiOutcomeBacktestEngine:
    """Backtest with independent thresholds for each outcome type."""

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
            'filters': 'fixtureStates:5'  # Only finished matches
        }

        all_fixtures = []
        page = 1

        # Paginate through all results
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
                pagination = data.get('pagination', {})
                print(f"📄 Fetching fixtures (starting with page {page}, found {len(fixtures_data)} on this page)")

            # Process fixtures from this page
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

                # Get scores
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

                # Determine result
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

            # Check if there are more pages
            pagination = data.get('pagination', {})
            if not pagination.get('has_more', False):
                break

            page += 1
            print(f"   Fetching page {page}...")

            if page > 1000:
                print(f"⚠️  Reached page limit of 1000")
                break

        print(f"✅ Found {len(all_fixtures)} completed fixtures across {page} page(s)")
        return all_fixtures

    def make_predictions(self, fixtures: list) -> pd.DataFrame:
        """Generate predictions for all fixtures."""
        print(f"\n🤖 Generating predictions for {len(fixtures)} fixtures...")

        results = []

        for fixture in tqdm(fixtures, desc="Predicting"):
            features = self.pipeline.generate_features(fixture)

            if not features:
                continue

            # Make prediction
            features_df = pd.DataFrame([features])
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

    def evaluate_multioutcome_strategy(
        self,
        df: pd.DataFrame,
        home_threshold: float,
        draw_threshold: float,
        away_threshold: float,
        draw_closeness_threshold: float = 0.05
    ) -> dict:
        """
        Evaluate betting strategy with independent thresholds.

        Args:
            home_threshold: Minimum probability to bet on home win
            draw_threshold: Minimum probability to bet on draw
            away_threshold: Minimum probability to bet on away win
            draw_closeness_threshold: Max |home_prob - away_prob| for draw bet

        Returns:
            Dict with performance metrics
        """
        total_bets = 0
        total_profit = 0.0
        total_stake = 0.0
        wins = 0

        bets_by_outcome = {
            'Home Win': {'bets': 0, 'wins': 0, 'profit': 0.0, 'stake': 0.0},
            'Draw': {'bets': 0, 'wins': 0, 'profit': 0.0, 'stake': 0.0},
            'Away Win': {'bets': 0, 'wins': 0, 'profit': 0.0, 'stake': 0.0}
        }

        for _, row in df.iterrows():
            home_prob = row['pred_home_prob']
            draw_prob = row['pred_draw_prob']
            away_prob = row['pred_away_prob']
            actual = row['actual_result']

            # Calculate implied odds (with 5% bookmaker margin)
            home_odds = 1 / home_prob * 0.95 if home_prob > 0.01 else 100
            draw_odds = 1 / draw_prob * 0.95 if draw_prob > 0.01 else 100
            away_odds = 1 / away_prob * 0.95 if away_prob > 0.01 else 100

            # Evaluate each outcome independently

            # 1. Home Win bet
            if home_prob >= home_threshold:
                total_bets += 1
                total_stake += 1.0
                bets_by_outcome['Home Win']['bets'] += 1
                bets_by_outcome['Home Win']['stake'] += 1.0

                if actual == 'H':
                    profit = (home_odds - 1) * 1.0
                    wins += 1
                    bets_by_outcome['Home Win']['wins'] += 1
                else:
                    profit = -1.0

                total_profit += profit
                bets_by_outcome['Home Win']['profit'] += profit

            # 2. Draw bet (only if teams are closely matched)
            prob_diff = abs(home_prob - away_prob)
            if draw_prob >= draw_threshold and prob_diff < draw_closeness_threshold:
                total_bets += 1
                total_stake += 1.0
                bets_by_outcome['Draw']['bets'] += 1
                bets_by_outcome['Draw']['stake'] += 1.0

                if actual == 'D':
                    profit = (draw_odds - 1) * 1.0
                    wins += 1
                    bets_by_outcome['Draw']['wins'] += 1
                else:
                    profit = -1.0

                total_profit += profit
                bets_by_outcome['Draw']['profit'] += profit

            # 3. Away Win bet
            if away_prob >= away_threshold:
                total_bets += 1
                total_stake += 1.0
                bets_by_outcome['Away Win']['bets'] += 1
                bets_by_outcome['Away Win']['stake'] += 1.0

                if actual == 'A':
                    profit = (away_odds - 1) * 1.0
                    wins += 1
                    bets_by_outcome['Away Win']['wins'] += 1
                else:
                    profit = -1.0

                total_profit += profit
                bets_by_outcome['Away Win']['profit'] += profit

        # Calculate metrics
        win_rate = wins / total_bets if total_bets > 0 else 0
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

        return {
            'home_threshold': home_threshold,
            'draw_threshold': draw_threshold,
            'away_threshold': away_threshold,
            'draw_closeness_threshold': draw_closeness_threshold,
            'total_bets': total_bets,
            'wins': wins,
            'win_rate': win_rate,
            'total_stake': total_stake,
            'total_profit': total_profit,
            'roi': roi,
            'bets_by_outcome': bets_by_outcome
        }

    def optimize_multioutcome_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find optimal threshold combinations for multi-outcome betting.
        """
        print(f"\n💰 Optimizing Multi-Outcome Thresholds")
        print("=" * 80)

        # Test different threshold combinations
        home_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
        draw_thresholds = [0.25, 0.30, 0.35, 0.40]
        away_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
        draw_closeness = [0.03, 0.05, 0.07, 0.10]

        results = []
        total_combinations = len(home_thresholds) * len(draw_thresholds) * len(away_thresholds) * len(draw_closeness)

        print(f"Testing {total_combinations} threshold combinations...")
        print()

        for h_thresh, d_thresh, a_thresh, d_close in tqdm(
            product(home_thresholds, draw_thresholds, away_thresholds, draw_closeness),
            total=total_combinations,
            desc="Optimizing"
        ):
            result = self.evaluate_multioutcome_strategy(
                df, h_thresh, d_thresh, a_thresh, d_close
            )

            if result['total_bets'] > 0:  # Only include if there were bets
                results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            print("⚠️  No valid threshold combinations found")
            return results_df

        # Sort by ROI
        results_df = results_df.sort_values('roi', ascending=False)

        # Display top 10 strategies
        print(f"\n🎯 Top 10 Multi-Outcome Strategies (by ROI):")
        print("-" * 120)
        print(f"{'H_Thresh':<10} {'D_Thresh':<10} {'A_Thresh':<10} {'Close':<8} {'Bets':<6} {'Win%':<8} {'ROI%':<8} {'Profit':<10}")
        print("-" * 120)

        for _, row in results_df.head(10).iterrows():
            print(f"{row['home_threshold']:<10.2f} {row['draw_threshold']:<10.2f} {row['away_threshold']:<10.2f} "
                  f"{row['draw_closeness_threshold']:<8.2f} {int(row['total_bets']):<6} "
                  f"{row['win_rate']:<8.1%} {row['roi']:<8.1f}% ${row['total_profit']:<9.2f}")

        # Show best strategy details
        best = results_df.iloc[0]
        print(f"\n{'='*80}")
        print(f"🏆 OPTIMAL MULTI-OUTCOME STRATEGY")
        print(f"{'='*80}")
        print(f"Thresholds:")
        print(f"  Home Win: ≥{best['home_threshold']:.2f}")
        print(f"  Draw: ≥{best['draw_threshold']:.2f} (when |home-away| < {best['draw_closeness_threshold']:.2f})")
        print(f"  Away Win: ≥{best['away_threshold']:.2f}")
        print(f"\nOverall Performance:")
        print(f"  Total Bets: {int(best['total_bets'])}")
        print(f"  Win Rate: {best['win_rate']:.1%}")
        print(f"  Net Profit: ${best['total_profit']:.2f}")
        print(f"  ROI: {best['roi']:.1f}%")

        print(f"\nBreakdown by Outcome:")
        for outcome in ['Home Win', 'Draw', 'Away Win']:
            stats = best['bets_by_outcome'][outcome]
            if stats['bets'] > 0:
                win_rate = stats['wins'] / stats['bets']
                roi = (stats['profit'] / stats['stake'] * 100) if stats['stake'] > 0 else 0
                print(f"  {outcome}:")
                print(f"    Bets: {stats['bets']}, Wins: {stats['wins']} ({win_rate:.1%})")
                print(f"    Profit: ${stats['profit']:+.2f}, ROI: {roi:+.1f}%")

        return results_df


def main():
    parser = argparse.ArgumentParser(description='Multi-outcome backtest for January 2026')
    parser.add_argument('--start-date', default='2026-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2026-01-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='results/backtest_multioutcome_january_2026.csv', help='Output CSV')

    args = parser.parse_args()

    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        print("❌ SPORTMONKS_API_KEY not set")
        sys.exit(1)

    print("=" * 80)
    print("MULTI-OUTCOME STRATEGY BACKTEST - JANUARY 2026")
    print("=" * 80)
    print(f"Period: {args.start_date} to {args.end_date}")
    print("\nStrategy: Independent thresholds for Home/Draw/Away outcomes")
    print("=" * 80)

    # Initialize engine
    engine = MultiOutcomeBacktestEngine(api_key)

    # Fetch completed fixtures
    fixtures = engine.fetch_completed_fixtures(args.start_date, args.end_date)

    if len(fixtures) == 0:
        print("❌ No completed fixtures found")
        sys.exit(1)

    # Make predictions
    predictions_df = engine.make_predictions(fixtures)

    # Optimize thresholds
    optimization_results = engine.optimize_multioutcome_thresholds(predictions_df)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {args.output}")

    optimization_output = output_path.parent / 'multioutcome_threshold_optimization.csv'
    optimization_results.to_csv(optimization_output, index=False)
    print(f"✅ Optimization results saved to: {optimization_output}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
