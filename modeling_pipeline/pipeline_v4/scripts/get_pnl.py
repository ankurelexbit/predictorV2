#!/usr/bin/env python3
"""
Get PnL (Profit and Loss) Report
=================================

Shows detailed betting performance with real market odds.

Usage:
    export DATABASE_URL="postgresql://..."

    # All-time PnL
    python3 scripts/get_pnl.py

    # PnL for specific period
    python3 scripts/get_pnl.py --start-date 2026-01-01 --end-date 2026-01-31

    # Last 30 days
    python3 scripts/get_pnl.py --days 30
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if variables already exported

from src.database import SupabaseClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_pnl_report(pnl_data: dict):
    """Display formatted PnL report."""
    summary = pnl_data['summary']
    by_outcome = pnl_data['by_outcome']
    monthly = pnl_data['monthly']

    print("\n" + "=" * 80)
    print("BETTING PERFORMANCE REPORT")
    print("=" * 80)

    # Period
    period = pnl_data['period']
    if period['start_date'] and period['end_date']:
        print(f"Period: {period['start_date']} to {period['end_date']}")
    elif period['start_date']:
        print(f"Period: From {period['start_date']}")
    elif period['end_date']:
        print(f"Period: Until {period['end_date']}")
    else:
        print("Period: All-time")

    print("\n" + "-" * 80)
    print("OVERALL SUMMARY")
    print("-" * 80)

    print(f"Total Bets: {summary['total_bets']}")
    print(f"Wins: {summary['wins']}")
    print(f"Losses: {summary['losses']}")
    print(f"Win Rate: {summary['win_rate']:.1%}")
    print(f"Total Profit/Loss: ${summary['total_profit']:.2f}")
    print(f"ROI: {summary['roi']:.2f}%")
    print(f"Average Confidence: {summary['avg_confidence']:.1%}")
    print(f"Average Odds: {summary['avg_odds']:.2f}")

    print("\n" + "-" * 80)
    print("BY BET TYPE")
    print("-" * 80)

    for bet_type, stats in by_outcome.items():
        if stats['bets'] > 0:
            print(f"\n{bet_type.upper()} WINS:")
            print(f"  Bets: {stats['bets']}")
            print(f"  Wins: {stats['wins']}")
            print(f"  Win Rate: {stats['win_rate']:.1%}")

    if monthly:
        print("\n" + "-" * 80)
        print("MONTHLY BREAKDOWN (Last 12 Months)")
        print("-" * 80)
        print(f"{'Month':<12} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'Profit/Loss':<15}")
        print("-" * 80)

        for month_stats in monthly:
            print(f"{month_stats['month']:<12} "
                  f"{month_stats['bets']:<8} "
                  f"{month_stats['wins']:<8} "
                  f"{month_stats['win_rate']:<12.1%} "
                  f"${month_stats['profit']:<14.2f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Get betting PnL report')

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument('--days', type=int, help='Last N days')
    date_group.add_argument('--start-date', help='Start date (YYYY-MM-DD)')

    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-version', default='v4_conservative', help='Model version (default: v4_conservative)')

    args = parser.parse_args()

    # Get database URL
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("❌ DATABASE_URL not set")
        sys.exit(1)

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

    db = SupabaseClient(database_url)
    pnl_data = db.get_detailed_pnl(
        start_date=start_date,
        end_date=end_date,
        model_version=args.model_version
    )

    # Display report
    display_pnl_report(pnl_data)


if __name__ == '__main__':
    main()
