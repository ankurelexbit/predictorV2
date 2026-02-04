#!/usr/bin/env python3
"""
View Prediction History
=======================

Demonstration script showing how to query and analyze prediction history.

Usage:
    # View latest prediction for a fixture
    python3 scripts/view_prediction_history.py --fixture-id 12345

    # View complete prediction timeline
    python3 scripts/view_prediction_history.py --fixture-id 12345 --timeline

    # Analyze prediction changes
    python3 scripts/view_prediction_history.py --fixture-id 12345 --analyze

    # View fixtures with most prediction updates
    python3 scripts/view_prediction_history.py --top-tracked
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if variables already exported

from src.database.supabase_client import SupabaseClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_timestamp(ts):
    """Format timestamp for display."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts.strftime('%Y-%m-%d %H:%M:%S')


def view_latest_prediction(client: SupabaseClient, fixture_id: int):
    """View the latest prediction for a fixture."""
    prediction = client.get_latest_prediction(fixture_id)

    if not prediction:
        print(f"\n❌ No predictions found for fixture {fixture_id}")
        return

    print("\n" + "="*80)
    print(f"LATEST PREDICTION - Fixture {fixture_id}")
    print("="*80)
    print(f"\nMatch: {prediction['home_team_name']} vs {prediction['away_team_name']}")
    print(f"Date: {prediction['match_date']}")
    print(f"Predicted at: {format_timestamp(prediction['prediction_timestamp'])}")
    print(f"\nProbabilities:")
    print(f"  Home Win: {prediction['pred_home_prob']:.1%}")
    print(f"  Draw:     {prediction['pred_draw_prob']:.1%}")
    print(f"  Away Win: {prediction['pred_away_prob']:.1%}")
    print(f"\nPredicted Outcome: {prediction['predicted_outcome']}")

    if prediction['should_bet']:
        print(f"\n✅ BETTING RECOMMENDATION")
        print(f"   Bet on: {prediction['bet_outcome']}")
        print(f"   Confidence: {prediction['bet_probability']:.1%}")

    print("="*80 + "\n")


def view_timeline(client: SupabaseClient, fixture_id: int):
    """View complete prediction timeline for a fixture."""
    timeline = client.get_prediction_timeline(fixture_id)

    if not timeline:
        print(f"\n❌ No predictions found for fixture {fixture_id}")
        return

    first = timeline[0]
    print("\n" + "="*80)
    print(f"PREDICTION TIMELINE - Fixture {fixture_id}")
    print("="*80)
    print(f"Match: {first['home_team_name']} vs {first['away_team_name']}")
    print(f"Match Date: {first['match_date']}")
    print(f"Total Predictions: {len(timeline)}")
    print("="*80)

    print(f"\n{'#':<4} {'Date/Time':<20} {'Home':<8} {'Draw':<8} {'Away':<8} {'Predicted':<10} {'Bet?':<6}")
    print("-"*80)

    for i, pred in enumerate(timeline, 1):
        timestamp = format_timestamp(pred['prediction_timestamp'])
        home = f"{pred['pred_home_prob']:.1%}"
        draw = f"{pred['pred_draw_prob']:.1%}"
        away = f"{pred['pred_away_prob']:.1%}"
        outcome = pred['predicted_outcome']
        bet = "✓" if pred['should_bet'] else ""

        print(f"{i:<4} {timestamp:<20} {home:<8} {draw:<8} {away:<8} {outcome:<10} {bet:<6}")

    if timeline[-1].get('actual_result'):
        print("-"*80)
        print(f"Actual Result: {timeline[-1]['actual_result']}")
        if timeline[-1].get('bet_won') is not None:
            print(f"Bet Result: {'WON ✅' if timeline[-1]['bet_won'] else 'LOST ❌'}")
            if timeline[-1].get('bet_profit'):
                print(f"Profit/Loss: ${timeline[-1]['bet_profit']:.2f}")

    print("="*80 + "\n")


def analyze_changes(client: SupabaseClient, fixture_id: int):
    """Analyze how predictions changed over time."""
    analysis = client.get_prediction_changes(fixture_id)

    if 'error' in analysis:
        print(f"\n❌ {analysis['error']}")
        return

    if 'message' in analysis:
        print(f"\n ℹ️  {analysis['message']}")
        return

    print("\n" + "="*80)
    print(f"PREDICTION CHANGE ANALYSIS - Fixture {analysis['fixture_id']}")
    print("="*80)

    print(f"\nTracking Period:")
    print(f"  First Prediction: {format_timestamp(analysis['first_prediction'])}")
    print(f"  Latest Prediction: {format_timestamp(analysis['latest_prediction'])}")
    print(f"  Total Updates: {analysis['total_predictions']}")

    print(f"\n📊 PROBABILITY CHANGES:")
    print("-"*80)
    print(f"{'Outcome':<10} {'Initial':<10} {'Latest':<10} {'Change':<12} {'Volatility':<12}")
    print("-"*80)

    for outcome_name, outcome_key in [('Home Win', 'home'), ('Draw', 'draw'), ('Away Win', 'away')]:
        data = analysis['probability_changes'][outcome_key]
        initial = f"{data['initial']:.1%}"
        latest = f"{data['latest']:.1%}"
        change = f"{data['change_pct']:+.1f}%"
        volatility = f"{data['volatility']:.1%}"
        print(f"{outcome_name:<10} {initial:<10} {latest:<10} {change:<12} {volatility:<12}")

    print()

    if analysis['outcome_change']['changed']:
        print("⚠️  PREDICTED OUTCOME CHANGED")
        print(f"   {analysis['outcome_change']['initial_prediction']} → {analysis['outcome_change']['latest_prediction']}")
    else:
        print("✅ Predicted outcome remained stable")

    print()

    if analysis['betting_change']['initial_should_bet'] != analysis['betting_change']['latest_should_bet']:
        print("⚠️  BETTING RECOMMENDATION CHANGED")
        print(f"   Initial: {'Bet' if analysis['betting_change']['initial_should_bet'] else 'No bet'}")
        print(f"   Latest: {'Bet' if analysis['betting_change']['latest_should_bet'] else 'No bet'}")

    if analysis.get('actual_result'):
        print(f"\n🏆 Actual Result: {analysis['actual_result']}")

    print("="*80 + "\n")


def view_top_tracked(client: SupabaseClient, limit: int = 10):
    """View fixtures with most prediction updates."""
    with client.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT fixture_id,
                   MAX(home_team_name) as home_team,
                   MAX(away_team_name) as away_team,
                   MAX(match_date) as match_date,
                   COUNT(*) as prediction_count,
                   MIN(prediction_timestamp) as first_prediction,
                   MAX(prediction_timestamp) as latest_prediction
            FROM predictions
            GROUP BY fixture_id
            HAVING COUNT(*) > 1
            ORDER BY prediction_count DESC, latest_prediction DESC
            LIMIT %s
        """, (limit,))

        results = cursor.fetchall()

        if not results:
            print("\n❌ No fixtures with multiple predictions found yet")
            print("   Run predictions daily to build up prediction history!")
            return

        print("\n" + "="*80)
        print("TOP TRACKED FIXTURES (Most Prediction Updates)")
        print("="*80)
        print(f"\n{'ID':<10} {'Match':<40} {'Updates':<10} {'Tracking Period':<25}")
        print("-"*80)

        for row in results:
            fixture_id, home, away, match_date, count, first, latest = row
            match = f"{home[:15]} vs {away[:15]}"
            first_str = format_timestamp(first)
            latest_str = format_timestamp(latest)
            period = f"{first_str[:10]} to {latest_str[:10]}"

            print(f"{fixture_id:<10} {match:<40} {count:<10} {period:<25}")

        print("="*80)
        print(f"\nTotal fixtures with prediction history: {len(results)}")
        print("\nTip: Use --fixture-id <ID> --timeline to see detailed changes")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='View and analyze prediction history',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--fixture-id', type=int, help='Fixture ID to analyze')
    parser.add_argument('--timeline', action='store_true', help='Show complete prediction timeline')
    parser.add_argument('--analyze', action='store_true', help='Analyze prediction changes')
    parser.add_argument('--top-tracked', action='store_true', help='Show fixtures with most updates')
    parser.add_argument('--limit', type=int, default=10, help='Limit for top tracked (default: 10)')

    args = parser.parse_args()

    # Get database URL
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("❌ DATABASE_URL environment variable not set")
        sys.exit(1)

    # Initialize client
    client = SupabaseClient(database_url)

    # Execute requested operation
    if args.top_tracked:
        view_top_tracked(client, args.limit)

    elif args.fixture_id:
        if args.analyze:
            analyze_changes(client, args.fixture_id)
        elif args.timeline:
            view_timeline(client, args.fixture_id)
        else:
            view_latest_prediction(client, args.fixture_id)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 scripts/view_prediction_history.py --top-tracked")
        print("  python3 scripts/view_prediction_history.py --fixture-id 12345")
        print("  python3 scripts/view_prediction_history.py --fixture-id 12345 --timeline")
        print("  python3 scripts/view_prediction_history.py --fixture-id 12345 --analyze")


if __name__ == '__main__':
    main()
