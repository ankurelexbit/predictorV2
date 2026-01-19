"""
Simplified Backtesting - Focus on Confidence Filtering and Bet Sizing
========================================================================

Since we're backtesting with fair odds (zero edge), we test strategies based on:
1. Confidence thresholds (only bet on high-confidence predictions)
2. Outcome filtering (e.g., only away wins)
3. Bet sizing (conservative vs aggressive stakes)
"""

import pandas as pd
import numpy as np

print("="*80)
print("BETTING STRATEGY BACKTESTING")
print("="*80)

# Load historical bets
bets_df = pd.read_csv('complete_bet_history.csv')

print(f"\nLoaded {len(bets_df)} historical bets from 10-day period")
print(f"Period: {bets_df['date'].min()} to {bets_df['date'].max()}")

# Define strategies with simple rules
strategies = {
    'Flat Stake (Baseline)': {
        'min_confidence': 0.0,
        'outcome_filter': None,
        'stake': 10.0
    },
    'High Confidence (55%+)': {
        'min_confidence': 0.55,
        'outcome_filter': None,
        'stake': 10.0
    },
    'Very High Confidence (60%+)': {
        'min_confidence': 0.60,
        'outcome_filter': None,
        'stake': 10.0
    },
    'Away Wins Only': {
        'min_confidence': 0.0,
        'outcome_filter': 'Away Win',
        'stake': 10.0
    },
    'Away Wins 50%+ Confidence': {
        'min_confidence': 0.50,
        'outcome_filter': 'Away Win',
        'stake': 10.0
    },
    'Away Wins 53%+ Confidence': {
        'min_confidence': 0.53,
        'outcome_filter': 'Away Win',
        'stake': 10.0
    },
    'Home Wins 55%+ Confidence': {
        'min_confidence': 0.55,
        'outcome_filter': 'Home Win',
        'stake': 10.0
    },
    'Proportional Staking': {
        'min_confidence': 0.50,
        'outcome_filter': None,
        'stake': 'proportional'  # Stake = confidence * 20
    }
}

print("\n" + "="*80)
print("TESTING STRATEGIES")
print("="*80)

all_results = []

for strategy_name, config in strategies.items():
    print(f"\n{strategy_name}:")
    print("-" * 60)

    total_bets = 0
    total_staked = 0.0
    total_return = 0.0
    winning_bets = 0

    for _, row in bets_df.iterrows():
        # Get model probability for predicted outcome
        if row['predicted'] == 'Home Win':
            model_prob = row['home_prob']
        elif row['predicted'] == 'Draw':
            model_prob = row['draw_prob']
        else:
            model_prob = row['away_prob']

        # Apply filters
        bet_this = True

        # Confidence filter
        if model_prob < config['min_confidence']:
            bet_this = False

        # Outcome filter
        if config['outcome_filter'] and row['predicted'] != config['outcome_filter']:
            bet_this = False

        if bet_this:
            # Calculate stake
            if config['stake'] == 'proportional':
                stake = model_prob * 20  # Scale confidence to stake
            else:
                stake = config['stake']

            # Calculate odds and return
            odds = 1 / model_prob

            total_bets += 1
            total_staked += stake

            if row['correct']:
                profit = stake * (odds - 1)
                total_return += profit
                winning_bets += 1
            else:
                loss = -stake
                total_return += loss

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

# Filter effectiveness
print(f"\n🎯 FILTER EFFECTIVENESS:")
for _, row in results_df.iterrows():
    if row['Bets'] > 0 and row['Strategy'] != 'Flat Stake (Baseline)':
        reduction = (1 - row['Bets'] / baseline['Bets']) * 100
        print(f"  {row['Strategy']:35s}: {reduction:5.1f}% fewer bets, " +
              f"{row['Win Rate %']:5.1f}% win rate, ROI {row['ROI %']:+6.2f}%")

# Save reports
results_df.to_csv('strategy_comparison.csv', index=False)

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

import json
with open('strategy_comparison_report.json', 'w') as f:
    json.dump(report, f, indent=2)

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
    print(f"  • Win rate: {best_strategy['Win Rate %']:.1f}%")
else:
    print(f"\n⚠️ WARNING: Best strategy still shows loss")
    print(f"  Best result: £{best_strategy['Net P&L']:+,.2f}")
    print("\n  Recommendations:")
    print("    - Increase minimum confidence thresholds")
    print("    - Focus on proven profitable categories")
    print("    - Continue paper trading until consistent profitability")

print("\n" + "="*80)
