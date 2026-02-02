#!/usr/bin/env python3
"""
Query Predictions from Database
===============================

Simple script to query and display predictions.

Usage:
    export DATABASE_URL="postgresql://..."
    python3 scripts/query_predictions.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SupabaseClient


def main():
    database_url = os.environ.get('DATABASE_URL')

    if not database_url:
        print("❌ DATABASE_URL not set")
        sys.exit(1)

    db = SupabaseClient(database_url)

    print("=" * 80)
    print("PREDICTIONS QUERY")
    print("=" * 80)

    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Total stats
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN should_bet THEN 1 ELSE 0 END) as bets,
                   MIN(match_date::date) as first_date,
                   MAX(match_date::date) as last_date
            FROM predictions
            WHERE model_version = 'v4'
        """)

        total, bets, first_date, last_date = cursor.fetchone()

        print(f"\nTotal Predictions: {total}")
        print(f"Recommended Bets: {bets} ({bets/total*100:.1f}%)")
        print(f"Date Range: {first_date} to {last_date}")

        # Top bets
        print(f"\n📊 Top 10 Bets by Confidence:")
        print("-" * 80)

        cursor.execute("""
            SELECT home_team_name, away_team_name, bet_outcome,
                   bet_probability, bet_odds, match_date
            FROM predictions
            WHERE should_bet = TRUE AND model_version = 'v4'
            ORDER BY bet_probability DESC
            LIMIT 10
        """)

        for i, (home, away, bet, prob, odds, date) in enumerate(cursor.fetchall(), 1):
            print(f"{i}. {home} vs {away}")
            print(f"   Bet: {bet} @ {odds:.2f} (confidence: {prob:.1%})")
            print(f"   Date: {date}")
            print()

        # Bet type breakdown
        print("📊 Bet Type Breakdown:")
        print("-" * 80)

        cursor.execute("""
            SELECT bet_outcome, COUNT(*) as count, AVG(bet_probability) as avg_conf
            FROM predictions
            WHERE should_bet = TRUE AND model_version = 'v4'
            GROUP BY bet_outcome
            ORDER BY count DESC
        """)

        for bet_type, count, avg_conf in cursor.fetchall():
            print(f"{bet_type}: {count} bets (avg confidence: {avg_conf:.1%})")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
