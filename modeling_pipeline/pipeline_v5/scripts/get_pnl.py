#!/usr/bin/env python3
"""
Get PnL (Profit and Loss) Report
=================================

Shows detailed betting performance from PostgreSQL database.

Usage:
    # All-time PnL (all strategies combined)
    python3 scripts/get_pnl.py

    # PnL for specific strategy
    python3 scripts/get_pnl.py --strategy hybrid
    python3 scripts/get_pnl.py --strategy threshold
    python3 scripts/get_pnl.py --strategy selector

    # Last 30 days
    python3 scripts/get_pnl.py --days 30

    # Last 30 days for a specific strategy
    python3 scripts/get_pnl.py --days 30 --strategy hybrid

    # Specific model version
    python3 scripts/get_pnl.py --model-version v5
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
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


def display_pnl_report(pnl_data: dict, model_version: str, strategy: str = None):
    """Display formatted PnL report."""
    summary = pnl_data['summary']
    by_outcome = pnl_data['by_outcome']
    monthly = pnl_data['monthly']

    print("\n" + "=" * 70)
    print("BETTING PERFORMANCE REPORT")
    print("=" * 70)

    # Period info
    period = pnl_data['period']
    if period['start_date'] and period['end_date']:
        print(f"Period: {period['start_date']} to {period['end_date']}")
    elif period['start_date']:
        print(f"Period: From {period['start_date']}")
    elif period['end_date']:
        print(f"Period: Until {period['end_date']}")
    else:
        print("Period: All-time")

    print(f"Model: {model_version}")
    if strategy:
        print(f"Strategy: {strategy}")
    else:
        print("Strategy: ALL (combined)")

    print("\n" + "-" * 70)
    print("OVERALL SUMMARY")
    print("-" * 70)

    print(f"Total Bets:        {summary['total_bets']}")
    print(f"Wins:              {summary['wins']}")
    print(f"Losses:            {summary['losses']}")
    print(f"Win Rate:          {summary['win_rate']:.1%}")
    print(f"Total Profit/Loss: ${summary['total_profit']:.2f}")
    print(f"ROI:               {summary['roi']:.2f}%")
    print(f"Avg Confidence:    {summary['avg_confidence']:.1%}")
    print(f"Avg Odds:          {summary['avg_odds']:.2f}")

    print("\n" + "-" * 70)
    print("BY BET TYPE")
    print("-" * 70)

    print(f"{'Outcome':<12} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'Profit':<12}")
    print("-" * 52)

    for bet_type in ['home', 'draw', 'away']:
        stats = by_outcome[bet_type]
        if stats['bets'] > 0:
            print(f"{bet_type.upper():<12} "
                  f"{stats['bets']:<8} "
                  f"{stats['wins']:<8} "
                  f"{stats['win_rate']:.1%}       "
                  f"${stats['profit']:.2f}")

    if monthly:
        print("\n" + "-" * 70)
        print("MONTHLY BREAKDOWN")
        print("-" * 70)
        print(f"{'Month':<12} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'Profit/Loss':<15}")
        print("-" * 55)

        for month_stats in monthly:
            print(f"{month_stats['month']:<12} "
                  f"{month_stats['bets']:<8} "
                  f"{month_stats['wins']:<8} "
                  f"{month_stats['win_rate']:.1%}       "
                  f"${month_stats['profit']:<14.2f}")

    # Quick analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    if summary['total_bets'] > 0:
        # Best performing outcome
        best_outcome = max(by_outcome.items(),
                          key=lambda x: x[1]['win_rate'] if x[1]['bets'] >= 5 else 0)
        if best_outcome[1]['bets'] >= 5:
            print(f"Best outcome:  {best_outcome[0].upper()} "
                  f"({best_outcome[1]['win_rate']:.1%} WR, ${best_outcome[1]['profit']:.2f} profit)")

        # Most profitable outcome
        most_profit = max(by_outcome.items(), key=lambda x: x[1]['profit'])
        if most_profit[1]['profit'] > 0:
            print(f"Most profit:   {most_profit[0].upper()} (${most_profit[1]['profit']:.2f})")

        # Verdict
        if summary['total_profit'] > 0:
            print(f"\nVerdict: PROFITABLE (+${summary['total_profit']:.2f}, {summary['roi']:.1f}% ROI)")
        else:
            print(f"\nVerdict: LOSING (${summary['total_profit']:.2f}, {summary['roi']:.1f}% ROI)")

    print("\n" + "=" * 70)


def display_strategy_comparison(db: DatabaseClient, start_date: str, end_date: str,
                                model_version: str):
    """Show side-by-side comparison of all strategies."""
    strategies = ['threshold', 'hybrid', 'selector']

    print("\n" + "=" * 70)
    print("PER-STRATEGY COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<12} {'Bets':<8} {'Wins':<8} {'WR':<10} {'Profit':<12} {'ROI':<10}")
    print("-" * 60)

    for strat in strategies:
        perf = db.get_betting_performance(
            days=9999, model_version=model_version, strategy=strat
        )
        # Re-query with date filter for accuracy
        pnl = db.get_detailed_pnl(
            start_date=start_date, end_date=end_date,
            model_version=model_version, strategy=strat
        )
        s = pnl['summary']
        if s['total_bets'] > 0:
            print(f"{strat:<12} {s['total_bets']:<8} {s['wins']:<8} "
                  f"{s['win_rate']:.1%}     ${s['total_profit']:<10.2f} {s['roi']:.1f}%")
        else:
            print(f"{strat:<12} {'0':<8} {'-':<8} {'-':<10} {'-':<12} {'-':<10}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Get betting PnL report')

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument('--days', type=int, help='Last N days')
    date_group.add_argument('--start-date', help='Start date (YYYY-MM-DD)')

    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', help='Filter by strategy (threshold, hybrid, selector)')
    parser.add_argument('--model-version', default='v5', help='Model version')
    parser.add_argument('--raw', action='store_true', help='Show raw data (JSON format)')

    args = parser.parse_args()

    # Determine date range
    start_date = None
    end_date = args.end_date

    if args.days:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
    elif args.start_date:
        start_date = args.start_date

    # Get PnL data
    logger.info("Fetching PnL data from database...")

    db = DatabaseClient(DATABASE_URL)
    pnl_data = db.get_detailed_pnl(
        start_date=start_date,
        end_date=end_date,
        model_version=args.model_version,
        strategy=args.strategy
    )

    if args.raw:
        import json
        print(json.dumps(pnl_data, indent=2, default=str))
    else:
        display_pnl_report(pnl_data, args.model_version, strategy=args.strategy)

        # If no specific strategy requested, show per-strategy comparison
        if not args.strategy:
            display_strategy_comparison(db, start_date, end_date, args.model_version)


if __name__ == '__main__':
    main()
