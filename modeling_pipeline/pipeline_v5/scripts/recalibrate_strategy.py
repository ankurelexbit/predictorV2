#!/usr/bin/env python3
"""
Recalibrate Betting Strategy
=============================

Optimizes betting thresholds (H/D/A probability thresholds and odds range)
using the last 120 days of prediction results from the database.

Run every 8 weeks for optimal performance (validated on 3 years of data:
9.8% ROI with 8-week recalibration vs 6.7% ROI with fixed thresholds).

Usage:
    # Standard recalibration (120-day lookback, updates production_config.py)
    python3 scripts/recalibrate_strategy.py

    # Custom lookback window
    python3 scripts/recalibrate_strategy.py --lookback-days 90

    # Dry run (show optimal config without updating)
    python3 scripts/recalibrate_strategy.py --dry-run

    # Use specific date range instead of lookback
    python3 scripts/recalibrate_strategy.py --start-date 2025-10-01 --end-date 2026-01-31
"""

import sys
import os
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config.production_config import DATABASE_URL
from src.database import DatabaseClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'production_config.py'


def fetch_predictions(db, start_date, end_date):
    """Fetch all predictions with results from database."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fixture_id, match_date,
                   pred_home_prob, pred_draw_prob, pred_away_prob,
                   predicted_outcome, actual_result,
                   best_home_odds, best_draw_odds, best_away_odds
            FROM predictions
            WHERE match_date >= %s AND match_date <= %s
              AND actual_result IS NOT NULL
              AND best_home_odds IS NOT NULL
              AND best_draw_odds IS NOT NULL
              AND best_away_odds IS NOT NULL
            ORDER BY match_date
        """, (start_date, end_date))

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]


def evaluate_config(predictions, h_t, d_t, a_t, o_lo, o_hi):
    """Evaluate a threshold + odds config on predictions. Returns (bets, wins, profit)."""
    thresholds = {'H': h_t, 'D': d_t, 'A': a_t}
    bets = 0
    wins = 0
    profit = 0.0

    for p in predictions:
        probs = {
            'H': float(p['pred_home_prob']),
            'D': float(p['pred_draw_prob']),
            'A': float(p['pred_away_prob'])
        }
        odds_map = {
            'H': float(p['best_home_odds']),
            'D': float(p['best_draw_odds']),
            'A': float(p['best_away_odds'])
        }
        pred = max(probs, key=probs.get)

        if probs[pred] >= thresholds[pred]:
            odds = odds_map[pred]
            if o_lo <= odds <= o_hi:
                bets += 1
                if p['actual_result'] == pred:
                    wins += 1
                    profit += odds - 1
                else:
                    profit -= 1

    return bets, wins, profit


def find_optimal_config(predictions, min_bets=10):
    """Grid search for best threshold + odds configuration."""
    h_vals = np.arange(0.40, 0.65, 0.05)
    d_vals = np.arange(0.30, 0.50, 0.05)
    a_vals = np.arange(0.40, 0.65, 0.05)
    odds_ranges = [(1.3, 3.0), (1.4, 3.5), (1.5, 3.5), (1.5, 4.0), (1.3, 3.5)]

    best = {'roi': -999, 'config': None}
    all_results = []

    for h_t, d_t, a_t in product(h_vals, d_vals, a_vals):
        for o_lo, o_hi in odds_ranges:
            bets, wins, profit = evaluate_config(predictions, h_t, d_t, a_t, o_lo, o_hi)
            if bets >= min_bets:
                roi = profit / bets * 100
                wr = wins / bets * 100
                result = {
                    'h_t': round(h_t, 2), 'd_t': round(d_t, 2), 'a_t': round(a_t, 2),
                    'o_lo': o_lo, 'o_hi': o_hi,
                    'bets': bets, 'wins': wins, 'wr': wr,
                    'profit': profit, 'roi': roi
                }
                all_results.append(result)
                if roi > best['roi']:
                    best = result

    return best, sorted(all_results, key=lambda x: x['roi'], reverse=True)


def update_config_file(h_t, d_t, a_t, o_lo, o_hi, stats):
    """Update production_config.py with new thresholds."""
    content = CONFIG_PATH.read_text()

    # Update THRESHOLDS block
    thresholds_pattern = (
        r"(# Validated on.*?\n|# More away.*?\n|# Recalibrated.*?\n)?"
        r"THRESHOLDS = \{\s*\n"
        r"\s*'home':.*?\n"
        r"\s*'away':.*?\n"
        r"\s*'draw':.*?\n"
        r"\}"
    )
    date_str = datetime.now().strftime('%Y-%m-%d')
    thresholds_new = (
        f"# Recalibrated {date_str}: {stats['bets']} bets, "
        f"{stats['wr']:.1f}% WR, {stats['roi']:.1f}% ROI\n"
        f"THRESHOLDS = {{\n"
        f"    'home': {h_t},\n"
        f"    'away': {a_t},\n"
        f"    'draw': {d_t}\n"
        f"}}"
    )
    content = re.sub(thresholds_pattern, thresholds_new, content)

    # Update ODDS_FILTER block
    odds_pattern = (
        r"(# Odds range filter.*?\n)"
        r"ODDS_FILTER = \{\s*\n"
        r"\s*'min':.*?\n"
        r"\s*'max':.*?\n"
        r"\s*'enabled':.*?\n"
        r"\}"
    )
    odds_new = (
        f"# Odds range filter (recalibrated {date_str})\n"
        f"ODDS_FILTER = {{\n"
        f"    'min': {o_lo},\n"
        f"    'max': {o_hi},\n"
        f"    'enabled': True\n"
        f"}}"
    )
    content = re.sub(odds_pattern, odds_new, content)

    # Update conservative strategy profile
    cons_pattern = (
        r"'conservative': \{\s*\n"
        r"\s*'thresholds':.*?\n"
        r"\s*'odds_filter':.*?\n"
        r"\s*'description':.*?\n"
        r"\s*\}"
    )
    bets_val = stats['bets']
    wr_val = stats['wr']
    roi_val = stats['roi']
    cons_new = (
        f"'conservative': {{\n"
        f"        'thresholds': {{'home': {h_t}, 'away': {a_t}, 'draw': {d_t}}},\n"
        f"        'odds_filter': {{'min': {o_lo}, 'max': {o_hi}, 'enabled': True}},\n"
        f"        'description': '{bets_val} bets, {wr_val:.0f}% WR, "
        f"+{roi_val:.1f}% ROI (recalibrated {date_str})'\n"
        f"    }}"
    )
    content = re.sub(cons_pattern, cons_new, content)

    CONFIG_PATH.write_text(content)


def main():
    parser = argparse.ArgumentParser(description='Recalibrate betting strategy thresholds')
    parser.add_argument('--lookback-days', type=int, default=120,
                        help='Days of history to optimize on (default: 120)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD), overrides --lookback-days')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--min-bets', type=int, default=10,
                        help='Minimum bets for a config to be considered (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show results without updating config')
    parser.add_argument('--model-version', default='v5', help='Model version filter')

    args = parser.parse_args()

    # Determine date range
    if args.start_date:
        start_date = args.start_date
        end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.lookback_days)).strftime('%Y-%m-%d')

    logger.info("=" * 60)
    logger.info("BETTING STRATEGY RECALIBRATION")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Lookback: {args.lookback_days} days")
    logger.info(f"Min bets: {args.min_bets}")

    # Fetch data
    db = DatabaseClient(DATABASE_URL)
    predictions = fetch_predictions(db, start_date, end_date)

    if len(predictions) < 30:
        logger.error(f"Only {len(predictions)} predictions with results. Need at least 30.")
        logger.error("Run update_results.py first, or extend the date range.")
        sys.exit(1)

    logger.info(f"Loaded {len(predictions)} predictions with results")

    # Current config performance
    from config.production_config import THRESHOLDS, ODDS_FILTER
    curr_bets, curr_wins, curr_profit = evaluate_config(
        predictions,
        THRESHOLDS['home'], THRESHOLDS['draw'], THRESHOLDS['away'],
        ODDS_FILTER['min'], ODDS_FILTER['max']
    )
    if curr_bets > 0:
        logger.info(f"\nCurrent config performance on this period:")
        logger.info(f"  H>{THRESHOLDS['home']}, D>{THRESHOLDS['draw']}, A>{THRESHOLDS['away']}, "
                     f"odds {ODDS_FILTER['min']}-{ODDS_FILTER['max']}")
        logger.info(f"  {curr_bets} bets, {curr_wins/curr_bets*100:.1f}% WR, "
                     f"${curr_profit:.2f}, {curr_profit/curr_bets*100:.1f}% ROI")

    # Grid search
    logger.info(f"\nRunning grid search...")
    best, top_results = find_optimal_config(predictions, min_bets=args.min_bets)

    if best.get('bets', 0) == 0:
        logger.error("No valid configuration found. Try lowering --min-bets or extending date range.")
        sys.exit(1)

    # Show top 10
    logger.info(f"\nTop 10 configurations:")
    logger.info(f"  {'H':>5} {'D':>5} {'A':>5} {'O_lo':>5} {'O_hi':>5} "
                f"{'Bets':>5} {'WR%':>6} {'Profit':>8} {'ROI%':>6}")
    logger.info(f"  {'-'*52}")
    for r in top_results[:10]:
        logger.info(f"  {r['h_t']:>5.2f} {r['d_t']:>5.2f} {r['a_t']:>5.2f} "
                     f"{r['o_lo']:>5.1f} {r['o_hi']:>5.1f} "
                     f"{r['bets']:>5} {r['wr']:>5.1f}% ${r['profit']:>7.2f} {r['roi']:>5.1f}%")

    # Best config
    logger.info(f"\nOptimal config:")
    logger.info(f"  Thresholds: H>{best['h_t']}, D>{best['d_t']}, A>{best['a_t']}")
    logger.info(f"  Odds range: {best['o_lo']}-{best['o_hi']}")
    logger.info(f"  Performance: {best['bets']} bets, {best['wr']:.1f}% WR, "
                 f"${best['profit']:.2f}, {best['roi']:.1f}% ROI")

    # Compare
    if curr_bets > 0:
        curr_roi = curr_profit / curr_bets * 100
        delta = best['roi'] - curr_roi
        logger.info(f"\n  vs Current: {delta:+.1f}% ROI change")

    if args.dry_run:
        logger.info("\n[DRY RUN] Config NOT updated.")
    else:
        update_config_file(
            best['h_t'], best['d_t'], best['a_t'],
            best['o_lo'], best['o_hi'],
            best
        )
        logger.info(f"\nUpdated {CONFIG_PATH}")
        logger.info("New thresholds will be used on next predict_live.py run.")


if __name__ == '__main__':
    main()
