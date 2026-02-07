#!/usr/bin/env python3
"""
Market Predictions Accuracy Report
====================================

Shows calibration and accuracy for advanced market predictions
(Over/Under, BTTS, Handicap) from the Poisson goals model.

Usage:
    python3 scripts/get_market_pnl.py --days 30
    python3 scripts/get_market_pnl.py --start-date 2026-01-01 --end-date 2026-01-31
"""

import sys
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.production_config import DATABASE_URL
from src.database import DatabaseClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_market_report(db: DatabaseClient, days: int = None,
                          start_date: str = None, end_date: str = None):
    """Display market prediction accuracy report."""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Build date filter
        where = "actual_home_score IS NOT NULL"
        params = []
        if days:
            where += " AND match_date >= NOW() - INTERVAL '%s days'"
            params.append(days)
        if start_date:
            where += " AND match_date >= %s"
            params.append(start_date)
        if end_date:
            where += " AND match_date <= %s"
            params.append(end_date)

        cursor.execute(f"""
            SELECT
                fixture_id, match_date, home_team_name, away_team_name,
                home_goals_lambda, away_goals_lambda,
                over_0_5_prob, over_1_5_prob, over_2_5_prob, over_3_5_prob,
                btts_prob,
                handicap_minus_2_5_prob, handicap_minus_1_5_prob,
                handicap_minus_0_5_prob, handicap_plus_0_5_prob,
                handicap_plus_1_5_prob, handicap_plus_2_5_prob,
                top_scorelines,
                actual_home_score, actual_away_score
            FROM market_predictions
            WHERE {where}
            ORDER BY match_date
        """, tuple(params))

        rows = cursor.fetchall()

    if not rows:
        logger.info("No market predictions with results found.")
        return

    # Calculate accuracy for each market
    n = len(rows)
    ou_stats = {0.5: [0, 0], 1.5: [0, 0], 2.5: [0, 0], 3.5: [0, 0]}  # [correct, total]
    btts_stats = [0, 0]
    hcap_stats = {}  # line -> [correct, total]
    exact_score_hits = 0
    top3_score_hits = 0
    lambda_errors_h, lambda_errors_a = [], []

    for row in rows:
        (fid, match_date, home_name, away_name,
         lh, la, o05, o15, o25, o35, btts,
         hm25, hm15, hm05, hp05, hp15, hp25,
         scorelines, actual_h, actual_a) = row

        actual_total = actual_h + actual_a
        actual_btts = actual_h >= 1 and actual_a >= 1
        actual_diff = actual_h - actual_a

        # Lambda accuracy
        lambda_errors_h.append(abs(lh - actual_h))
        lambda_errors_a.append(abs(la - actual_a))

        # Over/Under
        for line, prob in [(0.5, o05), (1.5, o15), (2.5, o25), (3.5, o35)]:
            if prob is not None:
                ou_stats[line][1] += 1
                if (prob > 0.5) == (actual_total > line):
                    ou_stats[line][0] += 1

        # BTTS
        if btts is not None:
            btts_stats[1] += 1
            if (btts > 0.5) == actual_btts:
                btts_stats[0] += 1

        # Handicap
        for line, prob in [(-2.5, hm25), (-1.5, hm15), (-0.5, hm05),
                           (0.5, hp05), (1.5, hp15), (2.5, hp25)]:
            if prob is not None:
                hcap_stats.setdefault(line, [0, 0])
                hcap_stats[line][1] += 1
                if (prob > 0.5) == (actual_diff > line):
                    hcap_stats[line][0] += 1

        # Correct score
        if scorelines:
            import json
            sl = scorelines if isinstance(scorelines, list) else json.loads(scorelines)
            if sl and sl[0]['home'] == actual_h and sl[0]['away'] == actual_a:
                exact_score_hits += 1
            if any(s['home'] == actual_h and s['away'] == actual_a for s in sl[:3]):
                top3_score_hits += 1

    # Print report
    import numpy as np
    print("=" * 70)
    print("MARKET PREDICTIONS ACCURACY REPORT")
    print("=" * 70)
    period = f"{start_date or ''} to {end_date or 'now'}" if start_date or end_date else f"Last {days} days"
    print(f"Period: {period}")
    print(f"Fixtures with results: {n}")
    print()

    # Goal prediction accuracy
    print("GOAL PREDICTION")
    print("-" * 40)
    print(f"  Home goals MAE: {np.mean(lambda_errors_h):.3f}")
    print(f"  Away goals MAE: {np.mean(lambda_errors_a):.3f}")
    print()

    # Over/Under
    print("OVER/UNDER")
    print("-" * 40)
    print(f"  {'Line':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    for line in [0.5, 1.5, 2.5, 3.5]:
        correct, total = ou_stats[line]
        acc = correct / total * 100 if total > 0 else 0
        print(f"  O/U {line:<5} {correct:>8} {total:>8} {acc:>9.1f}%")
    print()

    # BTTS
    print("BTTS")
    print("-" * 40)
    correct, total = btts_stats
    acc = correct / total * 100 if total > 0 else 0
    print(f"  Accuracy: {acc:.1f}% ({correct}/{total})")
    print()

    # Handicap
    print("ASIAN HANDICAP (Home Perspective)")
    print("-" * 40)
    print(f"  {'Line':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    for line in sorted(hcap_stats.keys()):
        correct, total = hcap_stats[line]
        acc = correct / total * 100 if total > 0 else 0
        sign = '+' if line > 0 else ''
        print(f"  AH {sign}{line:<6} {correct:>8} {total:>8} {acc:>9.1f}%")
    print()

    # Correct Score
    print("CORRECT SCORE")
    print("-" * 40)
    print(f"  Top-1 hit rate: {exact_score_hits}/{n} ({exact_score_hits/n*100:.1f}%)")
    print(f"  Top-3 hit rate: {top3_score_hits}/{n} ({top3_score_hits/n*100:.1f}%)")
    print()
    print("=" * 70)


def display_market_pnl(db: DatabaseClient, days: int = None,
                       start_date: str = None, end_date: str = None):
    """Display PnL report for goals market bets (O/U 2.5, BTTS)."""
    pnl = db.get_market_pnl_report(days=days, start_date=start_date, end_date=end_date)

    ou = pnl['ou_2_5']
    btts = pnl['btts']
    total_bets = ou['bets'] + btts['bets']

    if total_bets == 0:
        print("\nNo goals market bets found.")
        return

    print()
    print("=" * 70)
    print("GOALS MARKET BETTING PnL")
    print("=" * 70)
    period = f"{start_date or ''} to {end_date or 'now'}" if start_date or end_date else f"Last {days} days"
    print(f"Period: {period}")
    print()

    print(f"  {'Market':<12} {'Bets':>6} {'Wins':>6} {'WR':>8} {'Profit':>10} {'ROI':>8} {'Avg Edge':>10} {'Avg Odds':>10}")
    print("  " + "-" * 72)

    for label, stats in [('O/U 2.5', ou), ('BTTS', btts)]:
        if stats['bets'] > 0:
            print(f"  {label:<12} {stats['bets']:>6} {stats['wins']:>6} "
                  f"{stats['win_rate']:>7.1%} ${stats['profit']:>9.2f} "
                  f"{stats['roi']:>7.1f}% {stats['avg_edge']:>9.1%} "
                  f"{stats['avg_odds']:>9.2f}")

    total_profit = ou['profit'] + btts['profit']
    total_roi = total_profit / total_bets * 100 if total_bets > 0 else 0
    print("  " + "-" * 72)
    print(f"  {'TOTAL':<12} {total_bets:>6} {ou['wins'] + btts['wins']:>6} "
          f"{'':>8} ${total_profit:>9.2f} {total_roi:>7.1f}%")

    # Monthly breakdown
    monthly = pnl.get('monthly', [])
    if monthly:
        print()
        print("  MONTHLY BREAKDOWN")
        print(f"  {'Month':<10} {'O/U Bets':>10} {'O/U P&L':>10} {'BTTS Bets':>10} {'BTTS P&L':>10} {'Total P&L':>10}")
        print("  " + "-" * 62)
        for m in monthly:
            total_m = m['ou_profit'] + m['btts_profit']
            print(f"  {m['month']:<10} {m['ou_bets']:>10} ${m['ou_profit']:>9.2f} "
                  f"{m['btts_bets']:>10} ${m['btts_profit']:>9.2f} ${total_m:>9.2f}")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Market predictions accuracy report')
    parser.add_argument('--days', type=int, default=30, help='Last N days')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    db = DatabaseClient(DATABASE_URL)

    if args.start_date:
        display_market_report(db, start_date=args.start_date, end_date=args.end_date)
        display_market_pnl(db, start_date=args.start_date, end_date=args.end_date)
    else:
        display_market_report(db, days=args.days)
        display_market_pnl(db, days=args.days)


if __name__ == '__main__':
    main()
