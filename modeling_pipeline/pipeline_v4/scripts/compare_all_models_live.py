#!/usr/bin/env python3
"""
Compare All 4 Models on LIVE Predictions
=========================================

Runs REAL predictions on UPCOMING/RECENT fixtures with:
- Real fixtures from SportMonks API
- Real market odds
- All 4 models in parallel
- Stores predictions in database with model tags
- Fetches actual results
- Optimizes thresholds for each model
- Compares profitability

Usage:
    # Predict next 7 days
    python3 scripts/compare_all_models_live.py --days-ahead 7

    # Backtest on specific date range
    python3 scripts/compare_all_models_live.py \
        --start-date 2026-01-01 \
        --end-date 2026-01-31 \
        --include-finished

    # Custom models
    python3 scripts/compare_all_models_live.py \
        --days-ahead 7 \
        --models models/custom1.joblib models/custom2.joblib
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sportmonks_client import SportMonksClient
from scripts.predict_live_with_history import ProductionLivePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')
TOP_5_LEAGUES = [8, 82, 384, 564]  # Premier League, Bundesliga, Serie A, Ligue 1

class MultiModelPredictor:
    """Runs predictions with multiple models and compares them."""

    def __init__(self, models_info, api_key, database_url):
        self.models_info = models_info
        self.api_key = api_key
        self.database_url = database_url
        self.client = SportMonksClient(api_key)

    def create_predictions_table(self):
        """Create predictions table with model_tag column."""
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions_comparison (
                id SERIAL PRIMARY KEY,
                model_tag VARCHAR(50) NOT NULL,
                fixture_id INTEGER NOT NULL,
                match_date TIMESTAMP NOT NULL,
                league_id INTEGER,
                league_name VARCHAR(255),
                home_team_name VARCHAR(255) NOT NULL,
                away_team_name VARCHAR(255) NOT NULL,
                pred_home_prob DOUBLE PRECISION NOT NULL,
                pred_draw_prob DOUBLE PRECISION NOT NULL,
                pred_away_prob DOUBLE PRECISION NOT NULL,
                predicted_outcome VARCHAR(10) NOT NULL,
                best_home_odds DOUBLE PRECISION,
                best_draw_odds DOUBLE PRECISION,
                best_away_odds DOUBLE PRECISION,
                actual_home_score INTEGER,
                actual_away_score INTEGER,
                actual_result VARCHAR(10),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(model_tag, fixture_id)
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_comparison_model_fixture
                ON predictions_comparison(model_tag, fixture_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_comparison_match_date
                ON predictions_comparison(match_date);
        """)

        conn.commit()
        conn.close()

        logger.info("✅ predictions_comparison table created/verified")

    def predict_with_model(self, model_info, start_date, end_date, include_finished=False):
        """Run predictions with a single model."""
        logger.info(f"[{model_info['tag']}] Starting predictions...")

        try:
            # Initialize pipeline (loads model internally)
            pipeline = ProductionLivePipeline(
                api_key=self.api_key,
                history_start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                history_end_date=datetime.now().strftime('%Y-%m-%d')
            )

            # Override model path
            import joblib
            pipeline.model = joblib.load(model_info['path'])
            logger.info(f"[{model_info['tag']}] Model loaded: {model_info['name']}")

            # Get fixtures
            logger.info(f"[{model_info['tag']}] Fetching fixtures {start_date} to {end_date}...")
            fixtures = self.client.get_fixtures_between(start_date, end_date)

            if not include_finished:
                # Filter to only upcoming fixtures
                fixtures = [f for f in fixtures if f.get('state', {}).get('state') != 'FT']

            logger.info(f"[{model_info['tag']}] Found {len(fixtures)} fixtures")

            # Generate predictions
            predictions = []
            for i, fixture in enumerate(fixtures, 1):
                if i % 50 == 0:
                    logger.info(f"[{model_info['tag']}] Processing {i}/{len(fixtures)}...")

                try:
                    pred_probs = pipeline.predict(fixture)
                    if pred_probs:
                        # Extract fixture metadata
                        participants = fixture.get('participants', [])
                        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

                        # Extract odds
                        odds = fixture.get('odds', [])
                        best_home_odds = None
                        best_draw_odds = None
                        best_away_odds = None

                        for bookmaker in odds:
                            if bookmaker.get('bookmaker_name') in ['bet365', 'Bet365']:
                                latest_odds = bookmaker.get('latest', {}).get('data', {})
                                best_home_odds = latest_odds.get('2')  # Home
                                best_draw_odds = latest_odds.get('X')  # Draw
                                best_away_odds = latest_odds.get('1')  # Away
                                break

                        # Build prediction dict
                        pred = {
                            'model_tag': model_info['tag'],
                            'fixture_id': fixture.get('id'),
                            'match_date': fixture.get('starting_at'),
                            'league_id': fixture.get('league_id'),
                            'league_name': fixture.get('league', {}).get('name'),
                            'home_team_name': home_team.get('name') if home_team else None,
                            'away_team_name': away_team.get('name') if away_team else None,
                            'pred_home_prob': pred_probs['home_prob'],
                            'pred_draw_prob': pred_probs['draw_prob'],
                            'pred_away_prob': pred_probs['away_prob'],
                            'predicted_outcome': max([('H', pred_probs['home_prob']),
                                                     ('D', pred_probs['draw_prob']),
                                                     ('A', pred_probs['away_prob'])],
                                                    key=lambda x: x[1])[0],
                            'best_home_odds': best_home_odds,
                            'best_draw_odds': best_draw_odds,
                            'best_away_odds': best_away_odds
                        }
                        predictions.append(pred)
                except Exception as e:
                    logger.warning(f"[{model_info['tag']}] Error on fixture {fixture.get('id')}: {e}")

            logger.info(f"[{model_info['tag']}] Generated {len(predictions)} predictions")

            # Store in database
            self._store_predictions(predictions, model_info['tag'])

            return {
                'model': model_info['name'],
                'tag': model_info['tag'],
                'predictions': len(predictions),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"[{model_info['tag']}] Failed: {e}")
            return {
                'model': model_info['name'],
                'tag': model_info['tag'],
                'predictions': 0,
                'status': 'failed',
                'error': str(e)
            }

    def _store_predictions(self, predictions, model_tag):
        """Store predictions in database."""
        if not predictions:
            return

        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor()

        for pred in predictions:
            try:
                cursor.execute("""
                    INSERT INTO predictions_comparison (
                        model_tag, fixture_id, match_date, league_id, league_name,
                        home_team_name, away_team_name,
                        pred_home_prob, pred_draw_prob, pred_away_prob,
                        predicted_outcome,
                        best_home_odds, best_draw_odds, best_away_odds
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_tag, fixture_id) DO UPDATE SET
                        pred_home_prob = EXCLUDED.pred_home_prob,
                        pred_draw_prob = EXCLUDED.pred_draw_prob,
                        pred_away_prob = EXCLUDED.pred_away_prob,
                        predicted_outcome = EXCLUDED.predicted_outcome,
                        best_home_odds = EXCLUDED.best_home_odds,
                        best_draw_odds = EXCLUDED.best_draw_odds,
                        best_away_odds = EXCLUDED.best_away_odds
                """, (
                    model_tag,
                    pred['fixture_id'],
                    pred['match_date'],
                    pred.get('league_id'),
                    pred.get('league_name'),
                    pred['home_team_name'],
                    pred['away_team_name'],
                    pred['pred_home_prob'],
                    pred['pred_draw_prob'],
                    pred['pred_away_prob'],
                    pred['predicted_outcome'],
                    pred.get('best_home_odds'),
                    pred.get('best_draw_odds'),
                    pred.get('best_away_odds')
                ))
            except Exception as e:
                logger.warning(f"Error storing prediction for fixture {pred['fixture_id']}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"[{model_tag}] ✅ Stored {len(predictions)} predictions")

    def update_actual_results(self):
        """Fetch actual results for all predictions."""
        logger.info("\n" + "="*80)
        logger.info("FETCHING ACTUAL RESULTS")
        logger.info("="*80)

        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor()

        # Get distinct fixture IDs and date range
        cursor.execute("""
            SELECT MIN(match_date), MAX(match_date)
            FROM predictions_comparison
            WHERE actual_result IS NULL
        """)
        min_date, max_date = cursor.fetchone()

        if not min_date:
            logger.info("No predictions need results")
            conn.close()
            return

        logger.info(f"Fetching results for {min_date} to {max_date}...")

        # Get fixtures with results
        fixtures = self.client.get_fixtures_between(
            min_date.strftime('%Y-%m-%d'),
            max_date.strftime('%Y-%m-%d')
        )

        # Extract results
        results_map = {}
        for fixture in fixtures:
            if fixture.get('state', {}).get('state') == 'FT':
                scores = fixture.get('scores', [])
                for score in scores:
                    if score.get('description') == 'CURRENT':
                        home_score = score.get('score', {}).get('goals')
                        away_score = score.get('score', {}).get('goals')

                        if home_score is not None and away_score is not None:
                            if home_score > away_score:
                                result = 'H'
                            elif away_score > home_score:
                                result = 'A'
                            else:
                                result = 'D'

                            results_map[fixture['id']] = {
                                'home_score': home_score,
                                'away_score': away_score,
                                'result': result
                            }
                        break

        logger.info(f"✅ Found results for {len(results_map)} fixtures")

        # Update database
        for fixture_id, result in results_map.items():
            cursor.execute("""
                UPDATE predictions_comparison
                SET actual_home_score = %s,
                    actual_away_score = %s,
                    actual_result = %s
                WHERE fixture_id = %s
            """, (result['home_score'], result['away_score'], result['result'], fixture_id))

        conn.commit()
        conn.close()

        logger.info(f"✅ Updated {len(results_map)} fixtures with results")

    def compare_models(self, top_5_only=True):
        """Compare all models with threshold optimization."""
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON & THRESHOLD OPTIMIZATION")
        logger.info("="*80)

        conn = psycopg2.connect(self.database_url)

        all_results = []

        for model_info in self.models_info:
            tag = model_info['tag']

            # Load predictions for this model
            query = """
                SELECT
                    pred_home_prob, pred_draw_prob, pred_away_prob,
                    best_home_odds, best_draw_odds, best_away_odds,
                    actual_result, league_id
                FROM predictions_comparison
                WHERE model_tag = %s
                  AND actual_result IS NOT NULL
                  AND best_home_odds IS NOT NULL
            """

            if top_5_only:
                query += " AND league_id = ANY(%s)"
                df = pd.read_sql_query(query, conn, params=(tag, TOP_5_LEAGUES))
            else:
                df = pd.read_sql_query(query, conn, params=(tag,))

            if len(df) == 0:
                logger.warning(f"[{tag}] No predictions with results yet")
                continue

            logger.info(f"\n[{tag}] Optimizing on {len(df)} matches...")

            # Optimize thresholds
            best_thresh, best_metrics = self._optimize_thresholds(df, tag)

            if best_metrics:
                all_results.append({
                    'model': model_info['name'],
                    'tag': tag,
                    'thresholds': best_thresh,
                    'metrics': best_metrics
                })

        conn.close()

        # Print comparison
        if all_results:
            self._print_comparison(all_results)

        return all_results

    def _optimize_thresholds(self, df, model_tag):
        """Optimize thresholds for a model."""
        home_thresholds = np.arange(0.30, 0.85, 0.01)
        draw_thresholds = np.arange(0.20, 0.55, 0.01)
        away_thresholds = np.arange(0.25, 0.55, 0.01)

        best_profit = -float('inf')
        best_thresholds = None
        best_metrics = None

        for home_thresh in home_thresholds:
            for draw_thresh in draw_thresholds:
                for away_thresh in away_thresholds:
                    metrics = self._calculate_metrics(df, home_thresh, draw_thresh, away_thresh)

                    if metrics['total_bets'] >= 30 and metrics['total_profit'] > best_profit:
                        best_profit = metrics['total_profit']
                        best_thresholds = (home_thresh, draw_thresh, away_thresh)
                        best_metrics = metrics

        if best_thresholds:
            logger.info(f"[{model_tag}] Optimal: H={best_thresholds[0]:.2f}, D={best_thresholds[1]:.2f}, A={best_thresholds[2]:.2f}")
            logger.info(f"[{model_tag}] Result: {best_metrics['total_bets']} bets, ${best_metrics['total_profit']:.2f}, {best_metrics['roi']:.1f}% ROI")

        return best_thresholds, best_metrics

    def _calculate_metrics(self, df, home_thresh, draw_thresh, away_thresh):
        """Calculate betting metrics for given thresholds."""
        home_bets = draw_bets = away_bets = 0
        home_wins = draw_wins = away_wins = 0
        home_profit = draw_profit = away_profit = 0

        for _, row in df.iterrows():
            home_prob = row['pred_home_prob']
            draw_prob = row['pred_draw_prob']
            away_prob = row['pred_away_prob']
            actual = row['actual_result']

            # NEW BETTING LOGIC: Check which cross threshold
            candidates = []
            if home_prob >= home_thresh:
                candidates.append(('H', home_prob, row['best_home_odds']))
            if draw_prob >= draw_thresh:
                candidates.append(('D', draw_prob, row['best_draw_odds']))
            if away_prob >= away_thresh:
                candidates.append(('A', away_prob, row['best_away_odds']))

            # If 2+ cross, pick max probability
            bet_outcome = None
            if len(candidates) >= 2:
                bet_outcome, _, bet_odds = max(candidates, key=lambda x: x[1])
            elif len(candidates) == 1:
                bet_outcome, _, bet_odds = candidates[0]

            # Place bet
            if bet_outcome == 'H':
                home_bets += 1
                if actual == 'H':
                    home_wins += 1
                    home_profit += (row['best_home_odds'] - 1)
                else:
                    home_profit -= 1
            elif bet_outcome == 'D':
                draw_bets += 1
                if actual == 'D':
                    draw_wins += 1
                    draw_profit += (row['best_draw_odds'] - 1)
                else:
                    draw_profit -= 1
            elif bet_outcome == 'A':
                away_bets += 1
                if actual == 'A':
                    away_wins += 1
                    away_profit += (row['best_away_odds'] - 1)
                else:
                    away_profit -= 1

        total_bets = home_bets + draw_bets + away_bets
        total_profit = home_profit + draw_profit + away_profit
        roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

        return {
            'total_bets': total_bets,
            'total_profit': total_profit,
            'roi': roi,
            'home_bets': home_bets,
            'draw_bets': draw_bets,
            'away_bets': away_bets,
            'home_wins': home_wins,
            'draw_wins': draw_wins,
            'away_wins': away_wins,
            'home_profit': home_profit,
            'draw_profit': draw_profit,
            'away_profit': away_profit
        }

    def _print_comparison(self, results):
        """Print final comparison."""
        logger.info("\n" + "="*80)
        logger.info("FINAL COMPARISON")
        logger.info("="*80)
        print()

        # Sort by profit
        results.sort(key=lambda x: x['metrics']['total_profit'], reverse=True)

        print(f"{'Rank':<6} {'Model':<30} {'Profit':<12} {'ROI':<10} {'Bets':<8} {'H/D/A':<15}")
        print("-"*90)

        for i, r in enumerate(results, 1):
            m = r['metrics']
            hda = f"{m['home_bets']}/{m['draw_bets']}/{m['away_bets']}"
            print(f"{i:<6} {r['model']:<30} ${m['total_profit']:<11.2f} {m['roi']:<9.1f}% {m['total_bets']:<8} {hda:<15}")

        print()
        print("="*80)
        print("VERDICT")
        print("="*80)

        best = results[0]
        current = next((r for r in results if 'Current' in r['model']), None)

        if current and best['tag'] == current['tag']:
            print("✅ Current production model is OPTIMAL!")
        elif current:
            diff = best['metrics']['total_profit'] - current['metrics']['total_profit']
            print(f"⚠️  {best['model']} outperforms Current by ${diff:.2f}")
            print(f"   Draw bets: {best['metrics']['draw_bets']} vs {current['metrics']['draw_bets']}")


def main():
    parser = argparse.ArgumentParser(description='Compare all models on live predictions')
    parser.add_argument('--days-ahead', type=int, help='Days ahead to predict')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--include-finished', action='store_true', help='Include finished fixtures (backtest)')
    parser.add_argument('--models', nargs='+', help='Custom model paths')
    parser.add_argument('--top-5-only', action='store_true', default=True, help='Compare on top 5 leagues only')

    args = parser.parse_args()

    # Determine date range
    if args.days_ahead:
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=args.days_ahead)).strftime('%Y-%m-%d')
    elif args.start_date:
        start_date = args.start_date
        end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    else:
        print("Error: Must specify either --days-ahead or --start-date")
        return 1

    # Define models
    if args.models:
        models_info = [
            {'name': f'Custom {i+1}', 'tag': f'custom{i+1}', 'path': Path(p)}
            for i, p in enumerate(args.models)
        ]
    else:
        models_info = [
            {
                'name': 'Current Production',
                'tag': 'current',
                'path': Path('models/with_draw_features/conservative_with_draw_features.joblib')
            },
            {
                'name': 'Option 1: Conservative',
                'tag': 'option1',
                'path': Path('models/weight_experiments/option1_conservative.joblib')
            },
            {
                'name': 'Option 2: Aggressive',
                'tag': 'option2',
                'path': Path('models/weight_experiments/option2_aggressive.joblib')
            },
            {
                'name': 'Option 3: Balanced',
                'tag': 'option3',
                'path': Path('models/weight_experiments/option3_balanced.joblib')
            }
        ]

    # Filter to existing models
    models_info = [m for m in models_info if m['path'].exists()]

    if not models_info:
        print("Error: No models found!")
        return 1

    # Check environment
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    database_url = os.environ.get('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')

    if not api_key:
        print("Error: SPORTMONKS_API_KEY not set")
        print("Set it with: export SPORTMONKS_API_KEY='your_key_here'")
        return 1

    logger.info("="*80)
    logger.info("MULTI-MODEL LIVE PREDICTION COMPARISON")
    logger.info("="*80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Models: {len(models_info)}")
    logger.info(f"Include finished: {args.include_finished}")
    logger.info("="*80)

    # Initialize predictor
    predictor = MultiModelPredictor(models_info, api_key, database_url)

    # Create table
    predictor.create_predictions_table()

    # Run predictions in parallel
    logger.info("\n🚀 Running predictions for all models in parallel...")

    with ThreadPoolExecutor(max_workers=len(models_info)) as executor:
        futures = {
            executor.submit(
                predictor.predict_with_model,
                model_info,
                start_date,
                end_date,
                args.include_finished
            ): model_info
            for model_info in models_info
        }

        for future in as_completed(futures):
            result = future.result()
            if result['status'] == 'success':
                logger.info(f"✅ [{result['tag']}] {result['predictions']} predictions")
            else:
                logger.error(f"❌ [{result['tag']}] Failed: {result.get('error')}")

    # Update actual results
    if args.include_finished:
        predictor.update_actual_results()

        # Compare models
        predictor.compare_models(top_5_only=args.top_5_only)

    logger.info("\n✅ COMPLETE!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
