"""
Combined Multi-Outcome Strategy
================================

Tests combinations of:
- Away win predictions (proven profitable: +25.38% ROI)
- Draw predictions when model is uncertain (proven profitable: +14.97% ROI)
- High confidence predictions (60%+, proven profitable: +4.70% ROI)
"""

import pandas as pd

df = pd.read_csv('complete_bet_history.csv')

print("="*80)
print("COMBINED MULTI-OUTCOME STRATEGY TESTING")
print("="*80)

def apply_strategy(df, strategy_name, rules):
    """Apply betting rules and calculate P&L."""
    bets_placed = []

    for _, row in df.iterrows():
        bet = None
        stake = 10

        # Check each rule
        for rule in rules:
            result = rule(row)
            if result:
                bet = result
                break

        if bet:
            # Get odds for the bet outcome
            if bet == 'Home Win':
                odds = 1 / row['home_prob']
            elif bet == 'Draw':
                odds = 1 / row['draw_prob']
            else:  # Away Win
                odds = 1 / row['away_prob']

            # Calculate return
            correct = (row['actual'] == bet)
            if correct:
                profit = stake * (odds - 1)
                bets_placed.append({
                    'bet': bet,
                    'correct': True,
                    'return': profit,
                    'stake': stake
                })
            else:
                bets_placed.append({
                    'bet': bet,
                    'correct': False,
                    'return': -stake,
                    'stake': stake
                })

    if len(bets_placed) == 0:
        return None

    total_bets = len(bets_placed)
    total_staked = sum(b['stake'] for b in bets_placed)
    total_return = sum(b['return'] for b in bets_placed)
    winning_bets = sum(1 for b in bets_placed if b['correct'])
    win_rate = winning_bets / total_bets * 100
    roi = (total_return / total_staked * 100)

    return {
        'Strategy': strategy_name,
        'Bets': total_bets,
        'Wins': winning_bets,
        'Win Rate %': win_rate,
        'Total Staked': total_staked,
        'Net P&L': total_return,
        'ROI %': roi
    }

# Define strategies
strategies = {
    'Baseline (all predictions)': [
        lambda r: r['predicted']  # Bet on all predictions
    ],

    'Away Wins Only': [
        lambda r: 'Away Win' if r['predicted'] == 'Away Win' else None
    ],

    'Draw when low confidence (<50%)': [
        lambda r: 'Draw' if max(r['home_prob'], r['away_prob']) < 0.50 else None
    ],

    'Draw when home/away close (<12% diff)': [
        lambda r: 'Draw' if abs(r['home_prob'] - r['away_prob']) < 0.12 else None
    ],

    'Away Wins OR Draw if low confidence': [
        lambda r: 'Away Win' if r['predicted'] == 'Away Win' else None,
        lambda r: 'Draw' if max(r['home_prob'], r['away_prob']) < 0.50 else None
    ],

    'Away Wins OR Draw if close match': [
        lambda r: 'Away Win' if r['predicted'] == 'Away Win' else None,
        lambda r: 'Draw' if abs(r['home_prob'] - r['away_prob']) < 0.12 else None
    ],

    'High confidence (60%+) OR Away Wins': [
        lambda r: r['predicted'] if max(r['home_prob'], r['draw_prob'], r['away_prob']) >= 0.60 else None,
        lambda r: 'Away Win' if r['predicted'] == 'Away Win' else None
    ],

    'Away Wins OR Draw (low conf) OR High conf any outcome': [
        lambda r: 'Away Win' if r['predicted'] == 'Away Win' else None,
        lambda r: 'Draw' if max(r['home_prob'], r['away_prob']) < 0.50 else None,
        lambda r: r['predicted'] if max(r['home_prob'], r['draw_prob'], r['away_prob']) >= 0.60 else None
    ],

    'Smart Multi-Outcome': [
        # 1. Always bet away wins
        lambda r: 'Away Win' if r['predicted'] == 'Away Win' else None,
        # 2. Bet draw when close match
        lambda r: 'Draw' if abs(r['home_prob'] - r['away_prob']) < 0.10 else None,
        # 3. Bet very high confidence home wins
        lambda r: 'Home Win' if r['predicted'] == 'Home Win' and r['home_prob'] >= 0.65 else None
    ]
}

results = []

for strategy_name, rules in strategies.items():
    result = apply_strategy(df, strategy_name, rules)
    if result:
        results.append(result)

results_df = pd.DataFrame(results).sort_values('Net P&L', ascending=False)

print("\n" + "="*80)
print("COMBINED STRATEGY RESULTS (Sorted by Net P&L)")
print("="*80)
print()
print(results_df.to_string(index=False))

# Best strategy analysis
print("\n" + "="*80)
print("TOP 3 STRATEGIES")
print("="*80)

for i in range(min(3, len(results_df))):
    strategy = results_df.iloc[i]
    print(f"\n#{i+1}: {strategy['Strategy']}")
    print(f"  Net P&L: £{strategy['Net P&L']:+,.2f}")
    print(f"  ROI: {strategy['ROI %']:+.2f}%")
    print(f"  Bets: {strategy['Bets']:.0f}")
    print(f"  Win Rate: {strategy['Win Rate %']:.1f}% ({strategy['Wins']:.0f}/{strategy['Bets']:.0f})")
    print(f"  Total Staked: £{strategy['Total Staked']:,.2f}")

# Compare to baseline
baseline = results_df[results_df['Strategy'] == 'Baseline (all predictions)'].iloc[0]
best = results_df.iloc[0]

print("\n" + "="*80)
print("IMPROVEMENT vs BASELINE")
print("="*80)
print(f"\nBaseline: £{baseline['Net P&L']:+,.2f} ({baseline['ROI %']:+.2f}% ROI)")
print(f"Best:     £{best['Net P&L']:+,.2f} ({best['ROI %']:+.2f}% ROI)")
print(f"Improvement: £{best['Net P&L'] - baseline['Net P&L']:+,.2f} ({best['ROI %'] - baseline['ROI %']:+.2f}% ROI)")

print("\n" + "="*80)
