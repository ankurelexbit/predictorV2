"""
Backtest Smart Multi-Outcome Betting Strategy
==============================================

Tests the betting strategy on historical matches with known outcomes.

Usage:
    python backtest_betting_strategy.py
    python backtest_betting_strategy.py --bankroll 1000 --n-matches 200
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, PROCESSED_DATA_DIR, TEST_SEASONS
from utils import setup_logger, season_based_split

# Import betting strategy
import importlib.util
spec_bet = importlib.util.spec_from_file_location("betting_strategy", "11_smart_betting_strategy.py")
betting_module = importlib.util.module_from_spec(spec_bet)
spec_bet.loader.exec_module(betting_module)
SmartMultiOutcomeStrategy = betting_module.SmartMultiOutcomeStrategy

logger = setup_logger("backtest_strategy")


def main():
    parser = argparse.ArgumentParser(description="Backtest Betting Strategy")
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll (default: $1000)"
    )
    parser.add_argument(
        "--n-matches",
        type=int,
        default=None,
        help="Number of matches to backtest (default: all test set)"
    )

    args = parser.parse_args()

    print("="*80)
    print("SMART MULTI-OUTCOME BETTING STRATEGY - BACKTEST")
    print("="*80)
    print(f"Initial Bankroll: ${args.bankroll:,.2f}")
    print()

    # Load data
    logger.info("Loading historical data...")
    features_df = pd.read_csv(PROCESSED_DATA_DIR / "sportmonks_features.csv")
    features_df['date'] = pd.to_datetime(features_df['date'])

    # Filter to matches with results
    mask = features_df['target'].notna()
    df = features_df[mask].copy()

    # Split by season - use test set
    _, _, test_df = season_based_split(
        df, 'season_name',
        ["2019/2020", "2020/2021", "2021/2022"],
        ["2022/2023"],
        TEST_SEASONS
    )

    if args.n_matches:
        test_df = test_df.tail(args.n_matches)

    logger.info(f"Testing on {len(test_df)} matches from {TEST_SEASONS}")

    # Load ensemble model
    logger.info("Loading optimized ensemble model...")

    # Import model modules
    spec_elo = importlib.util.spec_from_file_location("elo_module", "04_model_baseline_elo.py")
    elo_module = importlib.util.module_from_spec(spec_elo)
    spec_elo.loader.exec_module(elo_module)

    spec_xgb = importlib.util.spec_from_file_location("xgb_module", "06_model_xgboost.py")
    xgb_module = importlib.util.module_from_spec(spec_xgb)
    spec_xgb.loader.exec_module(xgb_module)

    spec_ens = importlib.util.spec_from_file_location("ens_module", "07_model_ensemble.py")
    ens_module = importlib.util.module_from_spec(spec_ens)
    spec_ens.loader.exec_module(ens_module)

    # Build ensemble
    EnsembleModel = ens_module.EnsembleModel
    ensemble = EnsembleModel()

    elo_model = elo_module.EloProbabilityModel()
    elo_model.load(MODELS_DIR / "elo_model.joblib")
    ensemble.add_model('elo', elo_model, 0.2)

    dc_model_data = joblib.load(MODELS_DIR / "dixon_coles_model.joblib")
    class DCWrapper:
        def __init__(self, data):
            self.data = data
        def predict_proba(self, df, calibrated=True):
            n = len(df)
            return np.ones((n, 3)) / 3
    ensemble.add_model('dixon_coles', DCWrapper(dc_model_data), 0.3)

    xgb_model = xgb_module.XGBoostFootballModel()
    xgb_model.load(MODELS_DIR / "xgboost_model.joblib")
    ensemble.add_model('xgboost', xgb_model, 0.5)

    ensemble_data = joblib.load(MODELS_DIR / "ensemble_model.joblib")
    if 'calibrators' in ensemble_data:
        ensemble.calibrators = ensemble_data['calibrators']
        ensemble.is_calibrated = True
    if 'stacking_model' in ensemble_data:
        ensemble.stacking_model = ensemble_data['stacking_model']

    logger.info("Model loaded successfully")

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = ensemble.predict_proba(test_df, calibrated=True)

    # Add predictions to dataframe
    test_df['away_prob'] = predictions[:, 0]
    test_df['draw_prob'] = predictions[:, 1]
    test_df['home_prob'] = predictions[:, 2]

    # Map actual outcomes
    test_df['actual_outcome'] = test_df['target'].map({
        0: 'Away Win',
        1: 'Draw',
        2: 'Home Win'
    })

    # Initialize strategy
    logger.info("Running betting strategy backtest...")
    strategy = SmartMultiOutcomeStrategy(bankroll=args.bankroll)

    # Track all bets
    all_bets = []
    bankroll_history = [args.bankroll]

    for idx, row in test_df.iterrows():
        home_team = row.get('home_team_name', 'Home')
        away_team = row.get('away_team_name', 'Away')

        match_data = {
            'match_id': f"{idx}",
            'date': str(row['date'].date()),
            'home_team': home_team,
            'away_team': away_team,
            'home_prob': row['home_prob'],
            'draw_prob': row['draw_prob'],
            'away_prob': row['away_prob']
        }

        # Get recommendations
        recommendations = strategy.evaluate_match(match_data)

        # Process each bet
        for bet in recommendations:
            actual = row['actual_outcome']
            stake = bet.stake
            odds = bet.fair_odds

            # Determine result
            if bet.bet_outcome == actual:
                profit = stake * (odds - 1)
                result = 'Win'
                strategy.bankroll += profit
            else:
                profit = -stake
                result = 'Loss'
                strategy.bankroll -= stake

            bet_record = {
                'date': row['date'],
                'home_team': home_team,
                'away_team': away_team,
                'bet_outcome': bet.bet_outcome,
                'actual_outcome': actual,
                'probability': bet.probability,
                'odds': odds,
                'stake': stake,
                'result': result,
                'profit': profit,
                'rule': bet.rule_applied,
                'bankroll': strategy.bankroll
            }

            all_bets.append(bet_record)
            bankroll_history.append(strategy.bankroll)

    # Results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)

    if len(all_bets) == 0:
        print("\n⚠️  No bets placed - strategy criteria not met for any matches")
        print("This is normal for a conservative strategy.")
        return

    bets_df = pd.DataFrame(all_bets)

    total_bets = len(bets_df)
    winning_bets = len(bets_df[bets_df['result'] == 'Win'])
    losing_bets = len(bets_df[bets_df['result'] == 'Loss'])

    total_staked = bets_df['stake'].sum()
    total_profit = bets_df['profit'].sum()

    win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    final_bankroll = strategy.bankroll
    profit_pct = ((final_bankroll - args.bankroll) / args.bankroll * 100)

    print(f"\n📊 Overview:")
    print(f"   Period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"   Matches Analyzed: {len(test_df):,}")
    print(f"   Bets Placed: {total_bets:,}")
    print(f"   Bet Rate: {total_bets/len(test_df)*100:.1f}%")

    print(f"\n💰 Financial Performance:")
    print(f"   Initial Bankroll: ${args.bankroll:,.2f}")
    print(f"   Final Bankroll: ${final_bankroll:,.2f}")
    print(f"   Net Profit: ${total_profit:+,.2f}")
    print(f"   Profit %: {profit_pct:+.2f}%")
    print(f"   Total Staked: ${total_staked:,.2f}")
    print(f"   ROI: {roi:+.2f}%")

    print(f"\n📈 Bet Performance:")
    print(f"   Winning Bets: {winning_bets} ({win_rate:.1f}%)")
    print(f"   Losing Bets: {losing_bets} ({100-win_rate:.1f}%)")
    print(f"   Average Stake: ${bets_df['stake'].mean():.2f}")
    print(f"   Average Profit (Wins): ${bets_df[bets_df['result']=='Win']['profit'].mean():.2f}")
    print(f"   Average Loss: ${abs(bets_df[bets_df['result']=='Loss']['profit'].mean()):.2f}")

    # Breakdown by outcome
    print(f"\n📋 Performance by Bet Type:")
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        outcome_bets = bets_df[bets_df['bet_outcome'] == outcome]
        if len(outcome_bets) > 0:
            outcome_wins = len(outcome_bets[outcome_bets['result'] == 'Win'])
            outcome_profit = outcome_bets['profit'].sum()
            outcome_roi = (outcome_profit / outcome_bets['stake'].sum() * 100)
            print(f"\n   {outcome}:")
            print(f"     Bets: {len(outcome_bets)}")
            print(f"     Win Rate: {outcome_wins/len(outcome_bets)*100:.1f}%")
            print(f"     Profit: ${outcome_profit:+,.2f}")
            print(f"     ROI: {outcome_roi:+.2f}%")

    # Breakdown by rule
    print(f"\n🎯 Performance by Strategy Rule:")
    for rule in bets_df['rule'].unique():
        rule_bets = bets_df[bets_df['rule'] == rule]
        rule_wins = len(rule_bets[rule_bets['result'] == 'Win'])
        rule_profit = rule_bets['profit'].sum()
        rule_roi = (rule_profit / rule_bets['stake'].sum() * 100)
        print(f"\n   {rule}:")
        print(f"     Bets: {len(rule_bets)}")
        print(f"     Win Rate: {rule_wins/len(rule_bets)*100:.1f}%")
        print(f"     Profit: ${rule_profit:+,.2f}")
        print(f"     ROI: {rule_roi:+.2f}%")

    # Save results
    output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    bets_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    # Performance assessment
    print("\n" + "="*80)
    print("PERFORMANCE ASSESSMENT")
    print("="*80)

    if roi > 10:
        print("✅ EXCELLENT: ROI > 10% - Strategy is highly profitable")
    elif roi > 5:
        print("✅ GOOD: ROI > 5% - Strategy is profitable")
    elif roi > 0:
        print("⚠️  MARGINAL: ROI > 0% but < 5% - Strategy is slightly profitable")
    else:
        print("❌ UNPROFITABLE: ROI < 0% - Strategy needs adjustment")

    if win_rate > 50:
        print("✅ GOOD WIN RATE: > 50%")
    elif win_rate > 45:
        print("⚠️  ACCEPTABLE WIN RATE: 45-50%")
    else:
        print("❌ LOW WIN RATE: < 45%")

    print("\n⚠️  Remember: This is a backtest on historical data.")
    print("Past performance does not guarantee future results.")
    print("Always use paper trading before live betting!")


if __name__ == '__main__':
    main()
