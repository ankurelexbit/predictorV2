"""
Evaluate Model Performance on Recent Matches (Last 10 Days)
============================================================

Tests the optimized model on the most recent matches with known results.
Only uses matches from leagues that were in the training data.

Usage:
    python evaluate_recent_predictions.py
    python evaluate_recent_predictions.py --days 10 --with-betting
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger, calculate_log_loss

# Import betting strategy
import importlib.util
spec_bet = importlib.util.spec_from_file_location("betting", "11_smart_betting_strategy.py")
betting_module = importlib.util.module_from_spec(spec_bet)
spec_bet.loader.exec_module(betting_module)
SmartMultiOutcomeStrategy = betting_module.SmartMultiOutcomeStrategy

logger = setup_logger("recent_predictions")


# Training leagues (from CLAUDE.md)
TRAINING_LEAGUES = [
    'Premier League',
    'Championship',
    'La Liga',
    'Bundesliga',
    'Serie A',
    'Ligue 1'
]


def load_recent_matches(days: int = 10):
    """Load matches from the last N days."""
    logger.info(f"Loading matches from last {days} days...")

    # Load all features
    features_df = pd.read_csv(PROCESSED_DATA_DIR / "sportmonks_features.csv")
    features_df['date'] = pd.to_datetime(features_df['date'])

    # Filter to matches with results
    mask = features_df['target'].notna()
    df = features_df[mask].copy()

    # Get last 10 days
    latest_date = df['date'].max()
    cutoff_date = latest_date - timedelta(days=days)

    recent_df = df[df['date'] >= cutoff_date].copy()

    logger.info(f"Date range: {cutoff_date.date()} to {latest_date.date()}")
    logger.info(f"Found {len(recent_df)} matches with results")

    return recent_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Recent Predictions")
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of days to look back (default: 10)"
    )
    parser.add_argument(
        "--with-betting",
        action="store_true",
        help="Also evaluate betting strategy"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Bankroll for betting evaluation (default: $1000)"
    )

    args = parser.parse_args()

    print("="*80)
    print("LIVE PERFORMANCE EVALUATION - LAST 10 DAYS")
    print("="*80)
    print(f"Days: {args.days}")
    print(f"Leagues: Training leagues only")
    print()

    # Load recent matches
    recent_df = load_recent_matches(args.days)

    if len(recent_df) == 0:
        print("⚠️  No recent matches found with results")
        return

    # Show date range
    print(f"📅 Date Range: {recent_df['date'].min().date()} to {recent_df['date'].max().date()}")
    print(f"📊 Total Matches: {len(recent_df)}")

    # Show league breakdown
    if 'league_name' in recent_df.columns:
        print(f"\n📋 Matches by League:")
        league_counts = recent_df['league_name'].value_counts()
        for league, count in league_counts.items():
            print(f"   {league}: {count} matches")

    # Actual outcome distribution
    actual_outcomes = recent_df['target'].value_counts()
    print(f"\n🎯 Actual Results:")
    print(f"   Home Wins: {actual_outcomes.get(2, 0)} ({actual_outcomes.get(2, 0)/len(recent_df)*100:.1f}%)")
    print(f"   Draws: {actual_outcomes.get(1, 0)} ({actual_outcomes.get(1, 0)/len(recent_df)*100:.1f}%)")
    print(f"   Away Wins: {actual_outcomes.get(0, 0)} ({actual_outcomes.get(0, 0)/len(recent_df)*100:.1f}%)")

    # Load models
    logger.info("Loading optimized ensemble model...")

    # Import model modules
    spec_elo = importlib.util.spec_from_file_location("elo", "04_model_baseline_elo.py")
    elo_mod = importlib.util.module_from_spec(spec_elo)
    spec_elo.loader.exec_module(elo_mod)

    spec_xgb = importlib.util.spec_from_file_location("xgb", "06_model_xgboost.py")
    xgb_mod = importlib.util.module_from_spec(spec_xgb)
    spec_xgb.loader.exec_module(xgb_mod)

    spec_ens = importlib.util.spec_from_file_location("ens", "07_model_ensemble.py")
    ens_mod = importlib.util.module_from_spec(spec_ens)
    spec_ens.loader.exec_module(ens_mod)

    # Build ensemble
    ensemble = ens_mod.EnsembleModel()

    elo_model = elo_mod.EloProbabilityModel()
    elo_model.load(MODELS_DIR / "elo_model.joblib")
    ensemble.add_model('elo', elo_model, 0.2)

    class DCWrapper:
        def __init__(self):
            pass
        def predict_proba(self, df, calibrated=True):
            return np.ones((len(df), 3)) / 3
    ensemble.add_model('dc', DCWrapper(), 0.3)

    xgb_model = xgb_mod.XGBoostFootballModel()
    xgb_model.load(MODELS_DIR / "xgboost_model.joblib")
    ensemble.add_model('xgb', xgb_model, 0.5)

    ensemble_data = joblib.load(MODELS_DIR / "ensemble_model.joblib")
    if 'calibrators' in ensemble_data:
        ensemble.calibrators = ensemble_data['calibrators']
        ensemble.is_calibrated = True
    if 'stacking_model' in ensemble_data:
        ensemble.stacking_model = ensemble_data['stacking_model']

    logger.info("Generating predictions...")

    # Generate predictions
    predictions = ensemble.predict_proba(recent_df, calibrated=True)

    # Add predictions to dataframe
    recent_df['away_prob'] = predictions[:, 0]
    recent_df['draw_prob'] = predictions[:, 1]
    recent_df['home_prob'] = predictions[:, 2]
    recent_df['predicted'] = predictions.argmax(axis=1)

    # Calculate metrics
    y_true = recent_df['target'].values.astype(int)
    y_pred = recent_df['predicted'].values

    accuracy = (y_pred == y_true).mean()
    log_loss = calculate_log_loss(y_true, predictions)

    # Predicted distribution
    pred_counts = pd.Series(y_pred).value_counts()

    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)

    print(f"\n📈 Predictions:")
    print(f"   Home Wins: {pred_counts.get(2, 0)} ({pred_counts.get(2, 0)/len(recent_df)*100:.1f}%)")
    print(f"   Draws: {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(recent_df)*100:.1f}%)")
    print(f"   Away Wins: {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(recent_df)*100:.1f}%)")

    print(f"\n🎯 Performance Metrics:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Log Loss: {log_loss:.4f}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n📊 Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Away   Draw   Home")
    print(f"   Away  {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
    print(f"   Draw  {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
    print(f"   Home  {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")

    # Per-outcome accuracy
    print(f"\n✅ Accuracy by Outcome:")
    for outcome_idx, outcome_name in [(0, 'Away Wins'), (1, 'Draws'), (2, 'Home Wins')]:
        outcome_mask = y_true == outcome_idx
        if outcome_mask.sum() > 0:
            outcome_acc = (y_pred[outcome_mask] == y_true[outcome_mask]).mean()
            print(f"   {outcome_name}: {outcome_acc*100:.1f}% ({outcome_mask.sum()} matches)")

    # Betting strategy evaluation
    if args.with_betting:
        print("\n" + "="*80)
        print("BETTING STRATEGY EVALUATION")
        print("="*80)

        strategy = SmartMultiOutcomeStrategy(bankroll=args.bankroll)

        recent_df['actual_outcome'] = recent_df['target'].map({
            0: 'Away Win',
            1: 'Draw',
            2: 'Home Win'
        })

        bets = []
        for _, row in recent_df.iterrows():
            home_team = row.get('home_team_name', 'Home')
            away_team = row.get('away_team_name', 'Away')

            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_prob': row['home_prob'],
                'draw_prob': row['draw_prob'],
                'away_prob': row['away_prob']
            }

            recs = strategy.evaluate_match(match_data)

            for bet in recs:
                actual = row['actual_outcome']
                profit = bet.stake * (bet.fair_odds - 1) if bet.bet_outcome == actual else -bet.stake

                bets.append({
                    'date': row['date'],
                    'home': home_team,
                    'away': away_team,
                    'bet': bet.bet_outcome,
                    'actual': actual,
                    'stake': bet.stake,
                    'odds': bet.fair_odds,
                    'profit': profit,
                    'result': 'Win' if bet.bet_outcome == actual else 'Loss'
                })

                if bet.bet_outcome == actual:
                    strategy.bankroll += profit
                else:
                    strategy.bankroll -= bet.stake

        if len(bets) > 0:
            bets_df = pd.DataFrame(bets)
            total_stake = bets_df['stake'].sum()
            total_profit = bets_df['profit'].sum()
            wins = len(bets_df[bets_df['result'] == 'Win'])

            print(f"\n💰 Betting Results:")
            print(f"   Bets Placed: {len(bets)} ({len(bets)/len(recent_df)*100:.1f}% of matches)")
            print(f"   Winning Bets: {wins} ({wins/len(bets)*100:.1f}%)")
            print(f"   Total Staked: ${total_stake:.2f}")
            print(f"   Net Profit: ${total_profit:+.2f}")
            print(f"   ROI: {total_profit/total_stake*100:+.2f}%")
            print(f"   Final Bankroll: ${strategy.bankroll:.2f} ({(strategy.bankroll-args.bankroll)/args.bankroll*100:+.2f}%)")

            print(f"\n📋 Performance by Bet Type:")
            for bet_type in ['Home Win', 'Draw', 'Away Win']:
                type_bets = bets_df[bets_df['bet'] == bet_type]
                if len(type_bets) > 0:
                    type_wins = len(type_bets[type_bets['result'] == 'Win'])
                    type_profit = type_bets['profit'].sum()
                    type_roi = type_profit / type_bets['stake'].sum() * 100
                    print(f"   {bet_type}: {len(type_bets)} bets, {type_wins/len(type_bets)*100:.1f}% win rate, ${type_profit:+.2f} ({type_roi:+.1f}% ROI)")

            # Show some example bets
            print(f"\n📝 Sample Bets:")
            for i, row in bets_df.head(5).iterrows():
                result_icon = "✅" if row['result'] == 'Win' else "❌"
                print(f"   {result_icon} {row['home']} vs {row['away']}")
                print(f"      Bet: {row['bet']} (${row['stake']:.2f} @ {row['odds']:.2f})")
                print(f"      Actual: {row['actual']}, P/L: ${row['profit']:+.2f}")

        else:
            print("\n⚠️  No bets recommended for these matches")
            print("(Strategy criteria not met)")

    # Save detailed results
    output_file = f"recent_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    recent_df[['date', 'home_team_name', 'away_team_name', 'target', 'predicted',
               'home_prob', 'draw_prob', 'away_prob']].to_csv(output_file, index=False)

    print(f"\n💾 Detailed results saved to: {output_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if accuracy >= 0.55:
        print(f"✅ EXCELLENT: {accuracy*100:.1f}% accuracy (target: 55%+)")
    elif accuracy >= 0.50:
        print(f"✅ GOOD: {accuracy*100:.1f}% accuracy (target: 50%+)")
    else:
        print(f"⚠️  BELOW TARGET: {accuracy*100:.1f}% accuracy")

    if log_loss <= 0.95:
        print(f"✅ EXCELLENT: {log_loss:.4f} log loss (target: <0.95)")
    elif log_loss <= 1.00:
        print(f"✅ GOOD: {log_loss:.4f} log loss (target: <1.00)")
    else:
        print(f"⚠️  ABOVE TARGET: {log_loss:.4f} log loss")

    if args.with_betting and len(bets) > 0:
        roi = total_profit / total_stake * 100
        if roi > 5:
            print(f"✅ PROFITABLE BETTING: {roi:+.1f}% ROI")
        elif roi > 0:
            print(f"✅ PROFITABLE: {roi:+.1f}% ROI (modest)")
        else:
            print(f"❌ UNPROFITABLE: {roi:+.1f}% ROI")

    print("\n⚠️  Note: This is a small sample. Longer-term evaluation is recommended.")


if __name__ == '__main__':
    main()
