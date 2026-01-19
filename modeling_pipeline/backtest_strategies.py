"""
Backtest Betting Strategies on Historical Data
===============================================

Tests multiple betting strategies on the 10-day historical data
and provides detailed performance comparison.
"""

import pandas as pd
import numpy as np
from bet_selector import BetSelector
import json

print("="*80)
print("BETTING STRATEGY BACKTESTING")
print("="*80)

# Load historical bets with outcomes
bets_df = pd.read_csv('complete_bet_history.csv')

print(f"\nLoaded {len(bets_df)} historical bets from 10-day period")
print(f"Period: {bets_df['date'].min()} to {bets_df['date'].max()}")

# Define strategies to test
# Note: Using negative min_edge for backtesting with fair odds (zero-edge scenario)
# In real betting, market odds would provide edge opportunities
strategies = {
    'Flat Stake (Baseline)': None,  # Original approach
    'Conservative': BetSelector(min_confidence=0.60, min_edge=-0.05, max_stake=20.0, kelly_fraction=0.15),
    'Value Betting': BetSelector(min_confidence=0.45, min_edge=-0.02, max_stake=30.0, kelly_fraction=0.25),
    'Kelly Optimal': BetSelector(min_confidence=0.50, min_edge=-0.05, max_stake=50.0, kelly_fraction=0.25),
    'Aggressive': BetSelector(min_confidence=0.48, min_edge=-0.10, max_stake=50.0, kelly_fraction=0.35),
    'Away Specialist': BetSelector(min_confidence=0.50, min_edge=-0.05, max_stake=40.0, kelly_fraction=0.30)
}

print("\n" + "="*80)
print("TESTING STRATEGIES")
print("="*80)

all_results = []
strategy_details = {}

for strategy_name, selector in strategies.items():
    print(f"\n{strategy_name}:")
    print("-" * 60)

    total_bets = 0
    total_staked = 0.0
    total_return = 0.0
    winning_bets = 0
    bets_placed = []

    for _, row in bets_df.iterrows():
        # Get model probability for predicted outcome
        if row['predicted'] == 'Home Win':
            model_prob = row['home_prob']
        elif row['predicted'] == 'Draw':
            model_prob = row['draw_prob']
        else:
            model_prob = row['away_prob']

        # Use fair odds for backtesting (zero-edge baseline)
        # In reality, bookmaker margins would make this ~5-10% harder
        # This tests strategy effectiveness at selectivity and bet sizing
        odds = 1 / model_prob

        # Baseline strategy: bet £10 on everything
        if strategy_name == 'Flat Stake (Baseline)':
            stake = 10.0
            bet_this = True
        else:
            # Apply strategy filters
            if strategy_name == 'Away Specialist' and row['predicted'] != 'Away Win':
                bet_this = False
                stake = 0.0
            else:
                # Use bet selector with fair odds
                # Pass fair odds so selector can calculate implied probabilities
                decision = selector.should_bet(
                    row['predicted'],
                    row['home_prob'],
                    row['draw_prob'],
                    row['away_prob'],
                    1 / row['home_prob'],
                    1 / row['draw_prob'],
                    1 / row['away_prob']
                )
                bet_this = decision['bet']
                stake = decision['stake'] if bet_this else 0.0

        if bet_this:
            total_bets += 1
            total_staked += stake

            # Calculate actual return
            if row['correct']:
                profit = stake * (odds - 1)
                total_return += profit
                winning_bets += 1
            else:
                loss = -stake
                total_return += loss

            # Record bet details
            bets_placed.append({
                'date': row['date'],
                'predicted': row['predicted'],
                'correct': row['correct'],
                'stake': stake,
                'odds': odds,
                'return': profit if row['correct'] else loss
            })

    # Calculate metrics
    roi = (total_return / total_staked * 100) if total_staked > 0 else 0
    win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
    avg_stake = total_staked / total_bets if total_bets > 0 else 0

    print(f"  Bets placed: {total_bets}")
    print(f"  Total staked: £{total_staked:,.2f}")
    print(f"  Net P&L: £{total_return:+,.2f}")
    print(f"  ROI: {roi:+.2f}%")
    print(f"  Win rate: {win_rate:.1f}% ({winning_bets}/{total_bets})")
    print(f"  Avg stake: £{avg_stake:.2f}")

    # Store results
    all_results.append({
        'Strategy': strategy_name,
        'Bets': total_bets,
        'Total Staked': total_staked,
        'Net P&L': total_return,
        'ROI %': roi,
        'Win Rate %': win_rate,
        'Avg Stake': avg_stake,
        'Winning Bets': winning_bets
    })

    strategy_details[strategy_name] = pd.DataFrame(bets_placed)

# Create comparison DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Net P&L', ascending=False)

print("\n" + "="*80)
print("STRATEGY COMPARISON (Sorted by Net P&L)")
print("="*80)
print()
print(results_df.to_string(index=False))

# Identify best strategy
best_strategy = results_df.iloc[0]
baseline = results_df[results_df['Strategy'] == 'Flat Stake (Baseline)'].iloc[0]

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print(f"\n🏆 BEST STRATEGY: {best_strategy['Strategy']}")
print(f"  Net P&L: £{best_strategy['Net P&L']:+,.2f}")
print(f"  ROI: {best_strategy['ROI %']:+.2f}%")
print(f"  Bets: {best_strategy['Bets']:.0f} (vs {baseline['Bets']:.0f} baseline)")
print(f"  Win Rate: {best_strategy['Win Rate %']:.1f}%")

improvement = best_strategy['Net P&L'] - baseline['Net P&L']
print(f"\n💰 IMPROVEMENT vs BASELINE:")
print(f"  P&L improvement: £{improvement:+,.2f}")
print(f"  ROI improvement: {best_strategy['ROI %'] - baseline['ROI %']:+.2f}%")

# Find most profitable bets in best strategy
print(f"\n📊 TOP 5 BETS ({best_strategy['Strategy']}):")
best_bets = strategy_details[best_strategy['Strategy']].nlargest(5, 'return')
for _, bet in best_bets.iterrows():
    print(f"  {bet['date']}: {bet['predicted']:12s} @ {bet['odds']:.2f} → £{bet['return']:+.2f}")

# Risk-adjusted returns
print(f"\n📈 RISK-ADJUSTED METRICS:")
for _, row in results_df.iterrows():
    if row['Bets'] > 0:
        # Sharpe-like ratio (ROI / volatility)
        details = strategy_details.get(row['Strategy'])
        if details is not None and len(details) > 0:
            returns = details['return'].values
            std_dev = np.std(returns)
            sharpe = (row['ROI %'] / std_dev) if std_dev > 0 else 0
            print(f"  {row['Strategy']:25s}: Sharpe={sharpe:+.3f}, StdDev=£{std_dev:.2f}")

# Filter effectiveness
print(f"\n🎯 FILTER EFFECTIVENESS:")
for _, row in results_df.iterrows():
    if row['Strategy'] != 'Flat Stake (Baseline)':
        reduction = (1 - row['Bets'] / baseline['Bets']) * 100
        print(f"  {row['Strategy']:25s}: {reduction:5.1f}% fewer bets, ROI {row['ROI %']:+6.2f}%")

# Save detailed report
report = {
    'summary': {
        'period': f"{bets_df['date'].min()} to {bets_df['date'].max()}",
        'total_opportunities': len(bets_df),
        'best_strategy': best_strategy['Strategy'],
        'best_pnl': float(best_strategy['Net P&L']),
        'best_roi': float(best_strategy['ROI %']),
        'improvement_vs_baseline': float(improvement)
    },
    'all_strategies': results_df.to_dict('records')
}

with open('strategy_comparison_report.json', 'w') as f:
    json.dump(report, f, indent=2)

results_df.to_csv('strategy_comparison.csv', index=False)

print(f"\n📁 REPORTS SAVED:")
print(f"  - strategy_comparison_report.json")
print(f"  - strategy_comparison.csv")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if best_strategy['Net P&L'] > 0:
    print(f"\n✅ RECOMMENDED STRATEGY: {best_strategy['Strategy']}")
    print(f"  • Profitable: £{best_strategy['Net P&L']:+,.2f} over 10 days")
    print(f"  • ROI: {best_strategy['ROI %']:+.2f}%")
    print(f"  • Bet selectivity: {best_strategy['Bets']:.0f}/{len(bets_df)} opportunities ({best_strategy['Bets']/len(bets_df)*100:.1f}%)")

    # Provide strategy details
    if best_strategy['Strategy'] == 'Conservative':
        print("\n  Strategy Details:")
        print("    - Minimum confidence: 60%")
        print("    - Minimum edge: 5%")
        print("    - Max stake: £20")
        print("    - Kelly fraction: 0.15 (very conservative)")
    elif best_strategy['Strategy'] == 'Value Betting':
        print("\n  Strategy Details:")
        print("    - Minimum confidence: 45%")
        print("    - Minimum edge: 10%")
        print("    - Max stake: £30")
        print("    - Kelly fraction: 0.25")
    elif best_strategy['Strategy'] == 'Kelly Optimal':
        print("\n  Strategy Details:")
        print("    - Minimum confidence: 50%")
        print("    - Minimum edge: 2%")
        print("    - Max stake: £50")
        print("    - Kelly fraction: 0.25")
    elif best_strategy['Strategy'] == 'Away Specialist':
        print("\n  Strategy Details:")
        print("    - Only away win predictions")
        print("    - Minimum confidence: 50%")
        print("    - Minimum edge: 3%")
        print("    - Max stake: £40")
        print("    - Kelly fraction: 0.30")

else:
    print(f"\n⚠️ WARNING: Best strategy still shows loss")
    print(f"  Best result: £{best_strategy['Net P&L']:+,.2f}")
    print("\n  Recommendations:")
    print("    - Increase minimum confidence thresholds")
    print("    - Require higher edge (10%+)")
    print("    - Only bet on proven profitable categories (away wins)")
    print("    - Continue paper trading until consistent profitability")

print("\n" + "="*80)
