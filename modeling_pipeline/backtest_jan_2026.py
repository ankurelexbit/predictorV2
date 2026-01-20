#!/usr/bin/env python3
"""
Comprehensive Backtest: January 1-20, 2026

Tests the live prediction system on historical data from Jan 2026.
For each day:
1. Fetches fixtures that occurred that day
2. Generates predictions using the live pipeline
3. Compares predictions to actual results
4. Calculates PnL based on production thresholds
5. Outputs detailed fixture-level analysis

This validates the claimed 23.7% ROI on fresh, forward-looking data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from predict_live import LiveFeatureCalculator, load_models
from production_thresholds import get_production_thresholds
from utils import setup_logger
import importlib.util

logger = setup_logger("backtest_jan_2026")

def load_xgboost_model():
    """Load the draw-tuned XGBoost model."""
    spec = importlib.util.spec_from_file_location(
        "xgboost_model",
        Path(__file__).parent / "06_model_xgboost.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from config import MODELS_DIR
    model_path = MODELS_DIR / "xgboost_model_draw_tuned.joblib"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None

    model = mod.XGBoostFootballModel()
    model.load(model_path)
    return model

def get_actual_results(start_date: str, end_date: str):
    """Get actual match results from the features dataset."""
    from config import PROCESSED_DATA_DIR

    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'

    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return pd.DataFrame()

    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])

    # Filter for date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    mask = (df['date'] >= start) & (df['date'] <= end)
    results = df[mask].copy()

    # Filter for matches with results and odds
    has_result = results['target'].notna()
    has_odds = (results['odds_home'] > 0) & (~results['odds_home'].isna())
    results = results[has_result & has_odds].copy()

    logger.info(f"Found {len(results)} matches with results and odds in date range")

    return results

def make_prediction_for_match(model, calculator, match_row):
    """Make a prediction for a single match."""
    try:
        # Build features
        features = calculator.build_features_for_match(
            home_team_id=int(match_row['home_team_id']),
            away_team_id=int(match_row['away_team_id']),
            fixture_date=match_row['date'],
            home_team_name=match_row.get('home_team_name'),
            away_team_name=match_row.get('away_team_name'),
            league_name=match_row.get('league_name'),
            fixture_id=match_row.get('fixture_id')
        )

        if not features:
            return None

        # Make prediction
        features_df = pd.DataFrame([features])
        probs = model.predict_proba(features_df, calibrated=True)[0]

        p_away, p_draw, p_home = probs

        return {
            'p_home': float(p_home),
            'p_draw': float(p_draw),
            'p_away': float(p_away)
        }

    except Exception as e:
        logger.warning(f"Error predicting match {match_row.get('fixture_id')}: {e}")
        return None

def apply_thresholds(predictions, thresholds):
    """Apply production thresholds to determine betting recommendation."""
    p_home = predictions['p_home']
    p_draw = predictions['p_draw']
    p_away = predictions['p_away']

    model_probs = {'home': p_home, 'draw': p_draw, 'away': p_away}

    best_bet = None
    best_prob = 0

    for outcome in ['home', 'draw', 'away']:
        if model_probs[outcome] > thresholds[outcome] and model_probs[outcome] > best_prob:
            best_bet = outcome
            best_prob = model_probs[outcome]

    return best_bet, best_prob

def calculate_pnl(bet_outcome, actual_outcome, odds, stake=100):
    """Calculate profit/loss for a bet."""
    if bet_outcome == actual_outcome:
        # Win
        profit = stake * odds - stake
        return profit
    else:
        # Loss
        return -stake

def main():
    print("=" * 80)
    print("LIVE PREDICTION SYSTEM BACKTEST: JANUARY 1-20, 2026")
    print("=" * 80)
    print()

    # Load model and thresholds
    logger.info("Loading model and thresholds...")
    model = load_xgboost_model()

    if model is None:
        logger.error("Failed to load model")
        return

    thresholds = get_production_thresholds()
    logger.info(f"Thresholds: H={thresholds['home']:.2f}, D={thresholds['draw']:.2f}, A={thresholds['away']:.2f}")

    # Get actual results
    logger.info("Fetching actual results...")
    results_df = get_actual_results('2026-01-01', '2026-01-20')

    if len(results_df) == 0:
        logger.error("No matches found in date range")
        return

    print(f"\n✅ Found {len(results_df)} matches in January 1-20, 2026")
    print()

    # Initialize calculator
    calculator = LiveFeatureCalculator()

    # Track results
    all_bets = []
    daily_stats = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})
    outcome_stats = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})

    # Process each match
    logger.info("Generating predictions and calculating PnL...")
    print("=" * 80)
    print("FIXTURE-LEVEL ANALYSIS")
    print("=" * 80)
    print()

    outcome_map = {0: 'away', 1: 'draw', 2: 'home'}

    for idx, (_, match) in enumerate(results_df.iterrows(), 1):
        match_date = match['date'].strftime('%Y-%m-%d')
        home_team = match.get('home_team_name', 'Unknown')
        away_team = match.get('away_team_name', 'Unknown')

        print(f"[{idx}/{len(results_df)}] {match_date} - {home_team} vs {away_team}")

        # Make prediction
        predictions = make_prediction_for_match(model, calculator, match)

        if predictions is None:
            print(f"  ⚠️  Could not generate prediction")
            print()
            continue

        p_home = predictions['p_home']
        p_draw = predictions['p_draw']
        p_away = predictions['p_away']

        print(f"  📊 Predictions: H={p_home*100:.1f}% D={p_draw*100:.1f}% A={p_away*100:.1f}%")

        # Get actual result
        actual_target = int(match['target'])
        actual_outcome = outcome_map[actual_target]
        actual_score = f"{int(match.get('home_goals', 0))}-{int(match.get('away_goals', 0))}"

        print(f"  ⚽ Actual: {actual_outcome.upper()} ({actual_score})")

        # Get odds
        odds_home = match.get('odds_home', 0)
        odds_draw = match.get('odds_draw', 0)
        odds_away = match.get('odds_away', 0)

        print(f"  💰 Odds: H={odds_home:.2f} D={odds_draw:.2f} A={odds_away:.2f}")

        # Apply thresholds
        bet_outcome, bet_prob = apply_thresholds(predictions, thresholds)

        if bet_outcome is None:
            print(f"  ⏭️  NO BET (no threshold exceeded)")
            print()
            continue

        # Calculate PnL
        odds_map = {'home': odds_home, 'draw': odds_draw, 'away': odds_away}
        bet_odds = odds_map[bet_outcome]

        pnl = calculate_pnl(bet_outcome, actual_outcome, bet_odds, stake=100)
        won = (bet_outcome == actual_outcome)

        # Print bet recommendation
        result_emoji = "✅ WIN" if won else "❌ LOSS"
        print(f"  🎯 BET: {bet_outcome.upper()} @ {bet_odds:.2f} (confidence: {bet_prob*100:.1f}%)")
        print(f"  {result_emoji}: {pnl:+.2f} (stake: $100)")
        print()

        # Record bet
        bet_record = {
            'date': match_date,
            'fixture_id': match.get('fixture_id'),
            'home_team': home_team,
            'away_team': away_team,
            'p_home': p_home,
            'p_draw': p_draw,
            'p_away': p_away,
            'bet_outcome': bet_outcome,
            'bet_odds': bet_odds,
            'bet_probability': bet_prob,
            'actual_outcome': actual_outcome,
            'actual_score': actual_score,
            'won': won,
            'pnl': pnl,
            'stake': 100
        }

        all_bets.append(bet_record)

        # Update stats
        daily_stats[match_date]['bets'] += 1
        daily_stats[match_date]['wins'] += (1 if won else 0)
        daily_stats[match_date]['profit'] += pnl

        outcome_stats[bet_outcome]['bets'] += 1
        outcome_stats[bet_outcome]['wins'] += (1 if won else 0)
        outcome_stats[bet_outcome]['profit'] += pnl

    # Calculate overall statistics
    print("=" * 80)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 80)
    print()

    if len(all_bets) == 0:
        print("❌ No bets placed during this period")
        return

    total_bets = len(all_bets)
    total_wins = sum(1 for b in all_bets if b['won'])
    total_losses = total_bets - total_wins
    total_staked = sum(b['stake'] for b in all_bets)
    total_profit = sum(b['pnl'] for b in all_bets)
    roi = (total_profit / total_staked) * 100
    win_rate = (total_wins / total_bets) * 100
    avg_win = sum(b['pnl'] for b in all_bets if b['won']) / total_wins if total_wins > 0 else 0
    avg_loss = sum(b['pnl'] for b in all_bets if not b['won']) / total_losses if total_losses > 0 else 0

    print(f"📅 Period: January 1-20, 2026")
    print(f"🎲 Total Matches Analyzed: {len(results_df)}")
    print(f"💰 Total Bets Placed: {total_bets} ({total_bets/len(results_df)*100:.1f}% of matches)")
    print(f"✅ Wins: {total_wins}")
    print(f"❌ Losses: {total_losses}")
    print(f"📊 Win Rate: {win_rate:.1f}%")
    print()
    print(f"💵 Total Staked: ${total_staked:.2f}")
    print(f"💸 Total Profit/Loss: ${total_profit:+.2f}")
    print(f"📈 ROI: {roi:+.1f}%")
    print()
    print(f"📊 Average Win: ${avg_win:+.2f}")
    print(f"📊 Average Loss: ${avg_loss:+.2f}")
    print(f"📊 Profit Factor: {abs(total_wins * avg_win / (total_losses * avg_loss)) if total_losses > 0 else float('inf'):.2f}")
    print()

    # Performance by outcome
    print("=" * 80)
    print("PERFORMANCE BY BET TYPE")
    print("=" * 80)
    print()

    for outcome in ['home', 'draw', 'away']:
        stats = outcome_stats[outcome]
        if stats['bets'] > 0:
            outcome_roi = (stats['profit'] / (stats['bets'] * 100)) * 100
            outcome_wr = (stats['wins'] / stats['bets']) * 100
            print(f"{outcome.upper()} Bets:")
            print(f"  Count: {stats['bets']}")
            print(f"  Wins: {stats['wins']} ({outcome_wr:.1f}%)")
            print(f"  Profit: ${stats['profit']:+.2f}")
            print(f"  ROI: {outcome_roi:+.1f}%")
            print()

    # Daily breakdown
    print("=" * 80)
    print("DAILY BREAKDOWN")
    print("=" * 80)
    print()

    for date in sorted(daily_stats.keys()):
        stats = daily_stats[date]
        if stats['bets'] > 0:
            daily_roi = (stats['profit'] / (stats['bets'] * 100)) * 100
            daily_wr = (stats['wins'] / stats['bets']) * 100
            print(f"{date}: {stats['bets']} bets, {stats['wins']}/{stats['bets']} wins ({daily_wr:.1f}%), ${stats['profit']:+.2f} ({daily_roi:+.1f}% ROI)")

    print()

    # Save detailed results
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)

    # Save detailed bets CSV
    bets_df = pd.DataFrame(all_bets)
    bets_csv = output_dir / 'backtest_jan_2026_detailed.csv'
    bets_df.to_csv(bets_csv, index=False)
    print(f"💾 Detailed bets saved to: {bets_csv}")

    # Save summary JSON
    summary = {
        'period': {
            'start': '2026-01-01',
            'end': '2026-01-20',
            'days': 20
        },
        'matches': {
            'total_analyzed': len(results_df),
            'bets_placed': total_bets,
            'bet_frequency_pct': round(total_bets/len(results_df)*100, 1)
        },
        'performance': {
            'total_bets': total_bets,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate_pct': round(win_rate, 1),
            'total_staked': round(total_staked, 2),
            'total_profit': round(total_profit, 2),
            'roi_pct': round(roi, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(abs(total_wins * avg_win / (total_losses * avg_loss)) if total_losses > 0 else 0, 2)
        },
        'by_outcome': {
            outcome: {
                'bets': stats['bets'],
                'wins': stats['wins'],
                'win_rate_pct': round((stats['wins'] / stats['bets']) * 100, 1) if stats['bets'] > 0 else 0,
                'profit': round(stats['profit'], 2),
                'roi_pct': round((stats['profit'] / (stats['bets'] * 100)) * 100, 1) if stats['bets'] > 0 else 0
            }
            for outcome, stats in outcome_stats.items() if stats['bets'] > 0
        },
        'thresholds_used': thresholds,
        'model': 'xgboost_model_draw_tuned.joblib'
    }

    summary_json = output_dir / 'backtest_jan_2026_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Summary saved to: {summary_json}")
    print()

    # Comparison to claimed performance
    print("=" * 80)
    print("COMPARISON TO CLAIMED PERFORMANCE")
    print("=" * 80)
    print()

    claimed_roi = 23.7
    claimed_wr = 68.2
    claimed_bets_per_day = 3.3

    print(f"{'Metric':<20} {'Claimed':<15} {'Actual':<15} {'Difference':<15}")
    print("-" * 65)
    print(f"{'ROI':<20} {claimed_roi:>6.1f}%{'':<8} {roi:>6.1f}%{'':<8} {roi-claimed_roi:+6.1f}%")
    print(f"{'Win Rate':<20} {claimed_wr:>6.1f}%{'':<8} {win_rate:>6.1f}%{'':<8} {win_rate-claimed_wr:+6.1f}%")
    print(f"{'Bets/Day':<20} {claimed_bets_per_day:>6.1f}{'':<9} {total_bets/20:>6.1f}{'':<9} {total_bets/20-claimed_bets_per_day:+6.1f}")
    print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    if roi >= 15:
        print("✅ EXCELLENT: System performing well above breakeven")
    elif roi >= 5:
        print("✅ GOOD: System is profitable")
    elif roi >= 0:
        print("⚠️  MARGINAL: Barely profitable, needs improvement")
    else:
        print("❌ POOR: System is losing money")

    print()

    if abs(roi - claimed_roi) <= 5:
        print("✅ Actual ROI matches claimed performance (within 5%)")
    elif abs(roi - claimed_roi) <= 10:
        print("⚠️  Actual ROI somewhat differs from claimed performance")
    else:
        print(f"❌ SIGNIFICANT DEVIATION: Actual ROI differs from claimed by {abs(roi - claimed_roi):.1f}%")

    print()

if __name__ == "__main__":
    main()
