"""
Apply betting strategy to predictions and calculate ROI with actual results.
"""

import pandas as pd
import sys
from pathlib import Path
import importlib.util

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from module with numeric prefix
spec = importlib.util.spec_from_file_location("betting_strategy", Path(__file__).parent / "11_smart_betting_strategy.py")
betting_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(betting_module)
SmartMultiOutcomeStrategy = betting_module.SmartMultiOutcomeStrategy


def apply_strategy_and_calculate_roi(validated_predictions_file: str, initial_bankroll: float = 1000.0):
    """
    Apply betting strategy to validated predictions and calculate actual ROI.

    Args:
        validated_predictions_file: CSV file with predictions and actual results
        initial_bankroll: Starting bankroll
    """
    print("=" * 80)
    print("BETTING STRATEGY PERFORMANCE")
    print("=" * 80)

    # Load validated predictions
    df = pd.read_csv(validated_predictions_file)
    print(f"\nLoaded {len(df)} validated predictions")

    # Initialize strategy
    strategy = SmartMultiOutcomeStrategy(
        bankroll=initial_bankroll,
        away_win_min_prob=0.50,
        draw_close_threshold=0.05,
        home_win_min_prob=0.51,
        kelly_fraction=0.25,
        max_stake_pct=0.05
    )

    print(f"\nInitial Bankroll: £{initial_bankroll:,.2f}")
    print("\nStrategy Rules:")
    print("  1. Bet away wins when probability ≥50%")
    print("  2. Bet draw when home/away probabilities within 5%")
    print("  3. Bet home wins when probability ≥51%")
    print("  4. Kelly Criterion: 0.25x (fractional Kelly)")
    print("  5. Max stake: 5% of bankroll")

    # Apply strategy to each match
    bets = []
    total_staked = 0.0
    total_return = 0.0
    current_bankroll = initial_bankroll

    for idx, row in df.iterrows():
        # Prepare match data
        match_data = {
            'match_id': f"{row['date']}_{row['home_team']}_{row['away_team']}",
            'date': row['date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_prob': row['home_win_prob'],
            'draw_prob': row['draw_prob'],
            'away_prob': row['away_win_prob']
        }

        # Update strategy bankroll
        strategy.bankroll = current_bankroll

        # Get betting recommendations
        recommendations = strategy.evaluate_match(match_data)

        # Process each bet
        for bet in recommendations:
            stake = bet.stake
            odds = bet.fair_odds
            bet_outcome = bet.bet_outcome
            actual_outcome = row['actual_outcome']

            # Calculate result
            if bet_outcome == actual_outcome:
                # Win
                profit = stake * (odds - 1)
                result = 'Win'
                current_bankroll += profit
                total_return += profit
            else:
                # Loss
                loss = stake
                result = 'Loss'
                current_bankroll -= loss
                total_return -= loss

            total_staked += stake

            bets.append({
                'date': bet.date,
                'home_team': bet.home_team,
                'away_team': bet.away_team,
                'score': f"{row['home_score']}-{row['away_score']}",
                'bet_outcome': bet_outcome,
                'actual_outcome': actual_outcome,
                'stake': stake,
                'odds': odds,
                'home_prob': bet.home_prob * 100,
                'draw_prob': bet.draw_prob * 100,
                'away_prob': bet.away_prob * 100,
                'rule': bet.rule_applied,
                'result': result,
                'profit_loss': profit if result == 'Win' else -loss,
                'bankroll_after': current_bankroll
            })

    # Convert to DataFrame
    bets_df = pd.DataFrame(bets)

    if len(bets_df) == 0:
        print("\n⚠️  No bets placed (no matches met strategy criteria)")
        return

    # Calculate metrics
    total_bets = len(bets_df)
    winning_bets = (bets_df['result'] == 'Win').sum()
    losing_bets = (bets_df['result'] == 'Loss').sum()
    win_rate = winning_bets / total_bets * 100 if total_bets > 0 else 0
    roi = (total_return / total_staked * 100) if total_staked > 0 else 0

    # Breakdown by outcome type
    home_bets = bets_df[bets_df['bet_outcome'] == 'Home Win']
    draw_bets = bets_df[bets_df['bet_outcome'] == 'Draw']
    away_bets = bets_df[bets_df['bet_outcome'] == 'Away Win']

    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    print(f"Total Bets Placed: {total_bets}")
    print(f"Winning Bets: {winning_bets} ({win_rate:.1f}%)")
    print(f"Losing Bets: {losing_bets}")
    print(f"Total Staked: £{total_staked:,.2f}")
    print(f"Net Profit/Loss: £{total_return:+,.2f}")
    print(f"ROI: {roi:+.2f}%")
    print(f"Final Bankroll: £{current_bankroll:,.2f}")
    print(f"Bankroll Change: {((current_bankroll - initial_bankroll) / initial_bankroll * 100):+.2f}%")

    print("\n" + "=" * 80)
    print("PERFORMANCE BY BET TYPE")
    print("=" * 80)

    for bet_type, subset in [('Home Win', home_bets), ('Draw', draw_bets), ('Away Win', away_bets)]:
        if len(subset) > 0:
            wins = (subset['result'] == 'Win').sum()
            total = len(subset)
            staked = subset['stake'].sum()
            profit = subset['profit_loss'].sum()
            type_roi = (profit / staked * 100) if staked > 0 else 0

            print(f"\n{bet_type}:")
            print(f"  Bets: {total} | Wins: {wins} ({wins/total*100:.1f}%)")
            print(f"  Staked: £{staked:,.2f} | P/L: £{profit:+,.2f} | ROI: {type_roi:+.2f}%")

    print("\n" + "=" * 80)
    print("ALL BETS PLACED")
    print("=" * 80)

    for idx, bet in bets_df.iterrows():
        result_emoji = "✅" if bet['result'] == 'Win' else "❌"
        print(f"\n{result_emoji} {bet['home_team']} {bet['score']} {bet['away_team']}")
        print(f"   Bet: {bet['bet_outcome']} (£{bet['stake']:.2f} @ {bet['odds']:.2f})")
        print(f"   Probabilities: H:{bet['home_prob']:.1f}% D:{bet['draw_prob']:.1f}% A:{bet['away_prob']:.1f}%")
        print(f"   Actual: {bet['actual_outcome']} | P/L: £{bet['profit_loss']:+.2f}")
        print(f"   Rule: {bet['rule']}")
        print(f"   Bankroll: £{bet['bankroll_after']:,.2f}")

    # Save detailed bets
    output_file = validated_predictions_file.replace('_validated.csv', '_bets.csv')
    bets_df.to_csv(output_file, index=False)
    print(f"\n✅ Betting results saved to: {output_file}")

    return {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'net_profit': total_return,
        'roi': roi,
        'final_bankroll': current_bankroll
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python apply_betting_strategy.py <validated_predictions_file> [initial_bankroll]")
        print("Example: python apply_betting_strategy.py predictions_jan_18_lineup_test_validated.csv 1000")
        sys.exit(1)

    validated_file = sys.argv[1]
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0

    apply_strategy_and_calculate_roi(validated_file, bankroll)
