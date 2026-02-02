#!/usr/bin/env python3
"""
Backtest January 2026 Predictions
==================================

Runs predictions for all January 2026 fixtures and compares with actual results.
Performs threshold optimization to find profitable betting strategies.

Usage:
    python3 scripts/backtest_january_2026.py --start-date 2026-01-01 --end-date 2026-01-31
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtest predictions against actual results."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pipeline = StandaloneLivePipeline(api_key)
        self.pipeline.load_model()

    def fetch_completed_fixtures(self, start_date: str, end_date: str) -> list:
        """Fetch completed fixtures for date range with pagination."""
        print(f"\n📅 Fetching completed fixtures: {start_date} to {end_date}")

        # Use API to fetch fixtures for date range
        endpoint = f"fixtures/between/{start_date}/{end_date}"
        base_params = {
            'include': 'participants;scores;league;state',
            'filters': 'fixtureStates:5'  # Only finished matches
        }

        all_fixtures = []
        page = 1

        # Paginate through all results (same approach as SportMonksClient)
        while True:
            params = base_params.copy()
            params['page'] = page

            data = self.pipeline._api_call(endpoint, params)

            if not data or 'data' not in data:
                print(f"⚠️  No fixtures found")
                break

            # Get fixtures from this page
            fixtures_data = data['data']
            if not fixtures_data:
                break

            if page == 1:
                # Show total count from first page
                pagination = data.get('pagination', {})
                total_count = pagination.get('count', len(fixtures_data))
                print(f"📄 Fetching fixtures (starting with page {page}, found {len(fixtures_data)} on this page)")

            # Process fixtures from this page
            for fixture in data['data']:
                if fixture.get('state_id') != 5:  # Double-check finished
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

            # Check if there are more pages (same as SportMonksClient)
            pagination = data.get('pagination', {})
            if not pagination.get('has_more', False):
                break

            page += 1
            print(f"   Fetching page {page}...")

            # Safety limit to avoid infinite loops
            if page > 1000:
                print(f"⚠️  Reached page limit of 1000")
                break

        print(f"✅ Found {len(all_fixtures)} completed fixtures across {page} page(s)")
        return all_fixtures

    def _fetch_fixture_result(self, fixture_id: int) -> dict:
        """Fetch actual result for a fixture."""
        endpoint = f"fixtures/{fixture_id}"
        params = {'include': 'scores;state'}

        data = self.pipeline._api_call(endpoint, params)

        if not data or 'data' not in data:
            return None

        fixture = data['data']

        # Check if finished
        if fixture.get('state_id') != 5:  # 5 = Finished
            return None

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
            return None

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

        return {
            'actual_home_score': home_score,
            'actual_away_score': away_score,
            'actual_result': result,
            'actual_result_label': result_label
        }

    def make_predictions(self, fixtures: list) -> pd.DataFrame:
        """Generate predictions for all fixtures."""
        print(f"\n🤖 Generating predictions for {len(fixtures)} fixtures...")

        results = []

        for fixture in tqdm(fixtures, desc="Predicting"):
            # Generate features and predict
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

            predicted_class = np.argmax(probas)
            predicted_label = ['Away Win', 'Draw', 'Home Win'][predicted_class]
            confidence = probas[predicted_class]

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
                'predicted_outcome': predicted_label,
                'confidence': confidence
            })

        df = pd.DataFrame(results)
        print(f"✅ Generated {len(df)} predictions")
        return df

    def evaluate_predictions(self, df: pd.DataFrame) -> dict:
        """Evaluate prediction accuracy."""
        print(f"\n📊 Evaluating Predictions")
        print("=" * 80)

        # Map predictions to actual results
        pred_map = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
        df['pred_result'] = df['predicted_outcome'].map(pred_map)

        # Overall accuracy
        correct = (df['pred_result'] == df['actual_result']).sum()
        total = len(df)
        accuracy = correct / total

        print(f"\n📈 Overall Performance:")
        print(f"   Accuracy: {accuracy:.2%} ({correct}/{total})")

        # Accuracy by outcome
        print(f"\n📊 Accuracy by Predicted Outcome:")
        for outcome in ['Home Win', 'Draw', 'Away Win']:
            subset = df[df['predicted_outcome'] == outcome]
            if len(subset) > 0:
                correct_subset = (subset['pred_result'] == subset['actual_result']).sum()
                acc = correct_subset / len(subset)
                print(f"   {outcome}: {acc:.2%} ({correct_subset}/{len(subset)})")

        # Distribution of actual results
        print(f"\n📊 Actual Results Distribution:")
        result_counts = df['actual_result_label'].value_counts()
        for result, count in result_counts.items():
            pct = count / total
            print(f"   {result}: {count} ({pct:.1%})")

        # Log loss
        y_true_encoded = df['actual_result'].map({'A': 0, 'D': 1, 'H': 2})
        y_pred_proba = df[['pred_away_prob', 'pred_draw_prob', 'pred_home_prob']].values

        from sklearn.metrics import log_loss
        logloss = log_loss(y_true_encoded, y_pred_proba)
        print(f"\n📉 Log Loss: {logloss:.4f}")

        return {
            'accuracy': accuracy,
            'log_loss': logloss,
            'total_fixtures': total,
            'correct_predictions': correct
        }

    def optimize_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find optimal probability thresholds for betting.

        Assumes standard 3-way betting odds based on probabilities.
        """
        print(f"\n💰 Threshold Optimization for Betting Strategy")
        print("=" * 80)

        results = []

        # Test different confidence thresholds
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

        for threshold in thresholds:
            # Filter predictions above threshold
            confident = df[df['confidence'] >= threshold].copy()

            if len(confident) == 0:
                continue

            # Calculate implied odds from probabilities
            # Fair odds = 1 / probability
            # Bookmaker odds (with margin) = Fair odds * 0.95 (5% margin)
            confident['implied_odds_home'] = 1 / confident['pred_home_prob'] * 0.95
            confident['implied_odds_draw'] = 1 / confident['pred_draw_prob'] * 0.95
            confident['implied_odds_away'] = 1 / confident['pred_away_prob'] * 0.95

            # For each prediction, bet on the highest probability outcome
            total_bets = len(confident)
            total_stake = total_bets * 1.0  # $1 per bet

            # Calculate profit/loss
            profits = []

            for _, row in confident.iterrows():
                # Determine which outcome we bet on
                if row['predicted_outcome'] == 'Home Win':
                    odds = row['implied_odds_home']
                    won = row['actual_result'] == 'H'
                elif row['predicted_outcome'] == 'Draw':
                    odds = row['implied_odds_draw']
                    won = row['actual_result'] == 'D'
                else:  # Away Win
                    odds = row['implied_odds_away']
                    won = row['actual_result'] == 'A'

                if won:
                    profit = (odds - 1) * 1.0  # Win: get odds * stake - stake
                else:
                    profit = -1.0  # Lose: lose stake

                profits.append(profit)

            confident['profit'] = profits

            # Calculate metrics
            wins = sum(1 for p in profits if p > 0)
            win_rate = wins / total_bets if total_bets > 0 else 0
            total_profit = sum(profits)
            roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0

            results.append({
                'threshold': threshold,
                'num_bets': total_bets,
                'wins': wins,
                'losses': total_bets - wins,
                'win_rate': win_rate,
                'total_stake': total_stake,
                'total_return': total_stake + total_profit,
                'net_profit': total_profit,
                'roi': roi,
                'avg_odds': confident['confidence'].apply(lambda x: 1/x * 0.95).mean()
            })

        results_df = pd.DataFrame(results)

        # Display results
        print(f"\n{'Threshold':<10} {'Bets':<6} {'Wins':<6} {'Win%':<8} {'Stake':<8} {'Return':<10} {'Profit':<10} {'ROI%':<8}")
        print("-" * 80)

        for _, row in results_df.iterrows():
            print(f"{row['threshold']:<10.2f} {int(row['num_bets']):<6} {int(row['wins']):<6} "
                  f"{row['win_rate']:<8.1%} ${row['total_stake']:<7.0f} ${row['total_return']:<9.2f} "
                  f"${row['net_profit']:<9.2f} {row['roi']:<8.1f}%")

        # Find best threshold
        if len(results_df) > 0:
            best = results_df.loc[results_df['net_profit'].idxmax()]
            print(f"\n🎯 Optimal Threshold: {best['threshold']:.2f}")
            print(f"   Bets: {int(best['num_bets'])}")
            print(f"   Win Rate: {best['win_rate']:.1%}")
            print(f"   Net Profit: ${best['net_profit']:.2f}")
            print(f"   ROI: {best['roi']:.1f}%")

        return results_df

    def analyze_by_outcome_type(self, df: pd.DataFrame):
        """Analyze profitability by outcome type (Home/Draw/Away)."""
        print(f"\n📊 Profitability by Outcome Type")
        print("=" * 80)

        for outcome in ['Home Win', 'Draw', 'Away Win']:
            print(f"\n{outcome}:")

            subset = df[df['predicted_outcome'] == outcome].copy()

            if len(subset) == 0:
                print("   No predictions")
                continue

            # Test thresholds for this outcome
            best_threshold = None
            best_profit = -float('inf')

            for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
                confident = subset[subset['confidence'] >= threshold]

                if len(confident) == 0:
                    continue

                # Map to result code
                result_map = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
                wins = (confident['actual_result'] == result_map[outcome]).sum()
                total = len(confident)
                win_rate = wins / total

                # Calculate average odds and profit
                avg_prob = confident['confidence'].mean()
                avg_odds = 1 / avg_prob * 0.95

                profit = wins * (avg_odds - 1) - (total - wins)
                roi = (profit / total) * 100

                if profit > best_profit:
                    best_profit = profit
                    best_threshold = threshold

                print(f"   Threshold {threshold:.2f}: {total:3} bets, {wins:3} wins "
                      f"({win_rate:.1%}), Profit: ${profit:+7.2f}, ROI: {roi:+6.1f}%")

            if best_threshold:
                print(f"   → Best: {best_threshold:.2f} threshold (Profit: ${best_profit:+.2f})")


def main():
    parser = argparse.ArgumentParser(description='Backtest January 2026 predictions')
    parser.add_argument('--start-date', default='2026-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2026-01-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='results/backtest_january_2026.csv', help='Output CSV file')

    args = parser.parse_args()

    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        print("❌ SPORTMONKS_API_KEY not set")
        sys.exit(1)

    print("=" * 80)
    print("JANUARY 2026 BACKTEST")
    print("=" * 80)
    print(f"Period: {args.start_date} to {args.end_date}")

    # Initialize engine
    engine = BacktestEngine(api_key)

    # Fetch completed fixtures
    fixtures = engine.fetch_completed_fixtures(args.start_date, args.end_date)

    if len(fixtures) == 0:
        print("❌ No completed fixtures found")
        sys.exit(1)

    # Make predictions
    predictions_df = engine.make_predictions(fixtures)

    # Evaluate
    metrics = engine.evaluate_predictions(predictions_df)

    # Optimize thresholds
    threshold_results = engine.optimize_thresholds(predictions_df)

    # Analyze by outcome type
    engine.analyze_by_outcome_type(predictions_df)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    print(f"\n✅ Results saved to: {args.output}")

    # Save threshold analysis
    threshold_output = output_path.parent / 'threshold_optimization.csv'
    threshold_results.to_csv(threshold_output, index=False)
    print(f"✅ Threshold analysis saved to: {threshold_output}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
