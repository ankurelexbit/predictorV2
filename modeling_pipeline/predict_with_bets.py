"""
Prediction Pipeline with Smart Betting Recommendations
=======================================================

Generates predictions and betting recommendations for upcoming matches.

Usage:
    python predict_with_bets.py --date 2026-01-20
    python predict_with_bets.py --date 2026-01-20 --bankroll 1000
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
from utils import setup_logger

# Import betting strategy
import importlib.util
spec_bet = importlib.util.spec_from_file_location("betting_strategy", "11_smart_betting_strategy.py")
betting_module = importlib.util.module_from_spec(spec_bet)
spec_bet.loader.exec_module(betting_module)
SmartMultiOutcomeStrategy = betting_module.SmartMultiOutcomeStrategy
PaperTradingTracker = betting_module.PaperTradingTracker

logger = setup_logger("predict_with_bets")


def load_upcoming_matches(date_str: str = None):
    """
    Load upcoming matches for prediction.

    For demo purposes, this loads recent historical matches.
    In production, this would fetch upcoming fixtures from an API.
    """
    # Load features
    features_df = pd.read_csv(PROCESSED_DATA_DIR / "sportmonks_features.csv")
    features_df['date'] = pd.to_datetime(features_df['date'])

    if date_str:
        target_date = pd.to_datetime(date_str)
    else:
        target_date = datetime.now()

    # For demo: Get matches from a specific date range
    # In production: fetch from API
    demo_matches = features_df[
        (features_df['date'] >= target_date - timedelta(days=7)) &
        (features_df['date'] <= target_date)
    ].tail(10).copy()

    logger.info(f"Loaded {len(demo_matches)} upcoming matches")
    return demo_matches


def main():
    parser = argparse.ArgumentParser(description="Predictions with Betting Recommendations")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help="Date for predictions (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Bankroll for bet sizing (default: $1000)"
    )
    parser.add_argument(
        "--paper-trade",
        action="store_true",
        help="Log bets to paper trading tracker"
    )

    args = parser.parse_args()

    print("="*80)
    print("FOOTBALL PREDICTIONS WITH SMART BETTING RECOMMENDATIONS")
    print("="*80)
    print(f"Date: {args.date}")
    print(f"Bankroll: ${args.bankroll:,.2f}")
    print()

    # Load optimized ensemble model
    logger.info("Loading optimized ensemble model...")
    model_path = MODELS_DIR / "ensemble_model.joblib"
    ensemble_data = joblib.load(model_path)

    # Import model classes
    import importlib.util

    # Load Elo model
    spec_elo = importlib.util.spec_from_file_location("elo_module", "04_model_baseline_elo.py")
    elo_module = importlib.util.module_from_spec(spec_elo)
    spec_elo.loader.exec_module(elo_module)

    # Load XGBoost model
    spec_xgb = importlib.util.spec_from_file_location("xgb_module", "06_model_xgboost.py")
    xgb_module = importlib.util.module_from_spec(spec_xgb)
    spec_xgb.loader.exec_module(xgb_module)

    # Load ensemble model
    spec_ens = importlib.util.spec_from_file_location("ens_module", "07_model_ensemble.py")
    ens_module = importlib.util.module_from_spec(spec_ens)
    spec_ens.loader.exec_module(ens_module)

    # Reconstruct ensemble
    EnsembleModel = ens_module.EnsembleModel
    ensemble = EnsembleModel()

    # Load individual models
    elo_model = elo_module.EloProbabilityModel()
    elo_model.load(MODELS_DIR / "elo_model.joblib")
    ensemble.add_model('elo', elo_model, 0.2)

    dc_model_data = joblib.load(MODELS_DIR / "dixon_coles_model.joblib")
    # For simplicity, we'll use a wrapper
    class DCWrapper:
        def __init__(self, data):
            self.data = data
        def predict_proba(self, df, calibrated=True):
            # Return uniform probabilities as placeholder
            n = len(df)
            return np.ones((n, 3)) / 3

    ensemble.add_model('dixon_coles', DCWrapper(dc_model_data), 0.3)

    xgb_model = xgb_module.XGBoostFootballModel()
    xgb_model.load(MODELS_DIR / "xgboost_model.joblib")
    ensemble.add_model('xgboost', xgb_model, 0.5)

    # Load ensemble calibrators and stacking model
    if 'calibrators' in ensemble_data:
        ensemble.calibrators = ensemble_data['calibrators']
        ensemble.is_calibrated = True
    if 'stacking_model' in ensemble_data:
        ensemble.stacking_model = ensemble_data['stacking_model']

    logger.info("Model loaded successfully")

    # Load upcoming matches
    logger.info(f"Loading matches for {args.date}...")
    matches_df = load_upcoming_matches(args.date)

    if len(matches_df) == 0:
        print("\n⚠️  No matches found for the specified date")
        print("Using demo matches from recent history instead...\n")
        matches_df = load_upcoming_matches()

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = ensemble.predict_proba(matches_df, calibrated=True)

    # Add predictions to dataframe
    matches_df['away_prob'] = predictions[:, 0]
    matches_df['draw_prob'] = predictions[:, 1]
    matches_df['home_prob'] = predictions[:, 2]

    # Calculate fair odds
    matches_df['home_odds'] = 1 / matches_df['home_prob']
    matches_df['draw_odds'] = 1 / matches_df['draw_prob']
    matches_df['away_odds'] = 1 / matches_df['away_prob']

    # Predicted outcome
    matches_df['predicted'] = predictions.argmax(axis=1)
    matches_df['predicted_outcome'] = matches_df['predicted'].map({
        0: 'Away Win',
        1: 'Draw',
        2: 'Home Win'
    })

    # Initialize betting strategy
    strategy = SmartMultiOutcomeStrategy(bankroll=args.bankroll)

    # Initialize paper trading tracker if requested
    if args.paper_trade:
        tracker = PaperTradingTracker('paper_trading_log.csv')

    # Generate betting recommendations
    print("\n" + "="*80)
    print("PREDICTIONS AND BETTING RECOMMENDATIONS")
    print("="*80)

    all_recommendations = []
    total_recommended_stake = 0

    for idx, row in matches_df.iterrows():
        print(f"\n{'─'*80}")
        print(f"📅 {row['date'].strftime('%Y-%m-%d %H:%M')}")
        home_team = row.get('home_team_name', row.get('home_team', 'Home'))
        away_team = row.get('away_team_name', row.get('away_team', 'Away'))
        print(f"🏟️  {home_team} vs {away_team}")

        # Show probabilities
        print(f"\n📊 Probabilities:")
        print(f"   Home: {row['home_prob']:.1%} (odds: {row['home_odds']:.2f})")
        print(f"   Draw: {row['draw_prob']:.1%} (odds: {row['draw_odds']:.2f})")
        print(f"   Away: {row['away_prob']:.1%} (odds: {row['away_odds']:.2f})")
        print(f"\n🎯 Predicted: {row['predicted_outcome']}")

        # Get betting recommendations
        match_data = {
            'match_id': f"{row['date'].strftime('%Y%m%d')}_{home_team}_{away_team}",
            'date': row['date'].strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'home_prob': row['home_prob'],
            'draw_prob': row['draw_prob'],
            'away_prob': row['away_prob']
        }

        recommendations = strategy.evaluate_match(match_data)

        if recommendations:
            print(f"\n💰 BETTING RECOMMENDATIONS: {len(recommendations)} bet(s)")
            for bet in recommendations:
                print(f"\n   ✅ {bet.bet_outcome}")
                print(f"      Probability: {bet.probability:.1%}")
                print(f"      Fair Odds: {bet.fair_odds:.2f}")
                print(f"      Recommended Stake: ${bet.stake:.2f}")
                print(f"      Expected Value: ${bet.expected_value:+.2f}")
                print(f"      Rule: {bet.rule_applied}")

                total_recommended_stake += bet.stake
                all_recommendations.append(bet.to_dict())

                # Log to paper trading
                if args.paper_trade:
                    tracker.log_bet(bet)
        else:
            print(f"\n❌ NO BET RECOMMENDED")
            print(f"   (Does not meet strategy criteria)")

    # Summary
    print("\n" + "="*80)
    print("BETTING SUMMARY")
    print("="*80)
    print(f"\n📊 Total Matches Analyzed: {len(matches_df)}")
    print(f"💰 Bets Recommended: {len(all_recommendations)}")
    print(f"💵 Total Recommended Stake: ${total_recommended_stake:.2f}")
    print(f"📈 Stake as % of Bankroll: {total_recommended_stake/args.bankroll*100:.1f}%")

    if len(all_recommendations) > 0:
        total_ev = sum([rec['expected_value'] for rec in all_recommendations])
        print(f"🎯 Total Expected Value: ${total_ev:+.2f}")
        print(f"📊 Average EV per Bet: ${total_ev/len(all_recommendations):+.2f}")

        # Breakdown by outcome
        print(f"\n📋 Breakdown by Outcome:")
        outcome_counts = {}
        outcome_stakes = {}
        for rec in all_recommendations:
            outcome = rec['bet_outcome']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            outcome_stakes[outcome] = outcome_stakes.get(outcome, 0) + rec['stake']

        for outcome in ['Home Win', 'Draw', 'Away Win']:
            if outcome in outcome_counts:
                print(f"   {outcome}: {outcome_counts[outcome]} bet(s), ${outcome_stakes[outcome]:.2f} total stake")

    # Save recommendations to CSV
    if len(all_recommendations) > 0:
        recommendations_df = pd.DataFrame(all_recommendations)
        output_file = f"betting_recommendations_{args.date.replace('-', '')}.csv"
        recommendations_df.to_csv(output_file, index=False)
        print(f"\n💾 Recommendations saved to: {output_file}")

    if args.paper_trade:
        print(f"📝 Paper trades logged to: paper_trading_log.csv")

    print("\n" + "="*80)
    print("⚠️  IMPORTANT REMINDERS")
    print("="*80)
    print("• These are PAPER TRADING recommendations only")
    print("• Always validate strategy performance before live betting")
    print("• Never bet more than you can afford to lose")
    print("• Past performance does not guarantee future results")
    print("• Betting involves risk - bet responsibly")

    return all_recommendations


if __name__ == '__main__':
    main()
