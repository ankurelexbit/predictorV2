"""
Test Draw Betting Strategies
=============================

Since the model never predicts draws, test alternative strategies:
1. Bet on draws when probabilities are close (home/away/draw within 10%)
2. Bet on draws when predicted outcome has low confidence
3. Bet on draws when draw_prob is highest even if not highest overall
"""

import pandas as pd

df = pd.read_csv('complete_bet_history.csv')

print("="*80)
print("DRAW BETTING STRATEGIES")
print("="*80)

strategies = {
    'Draw when all outcomes close (within 15%)': {
        'condition': lambda r: (max(r['home_prob'], r['draw_prob'], r['away_prob']) -
                               min(r['home_prob'], r['draw_prob'], r['away_prob'])) < 0.15
    },
    'Draw when predicted outcome < 50%': {
        'condition': lambda r: max(r['home_prob'], r['draw_prob'], r['away_prob']) < 0.50
    },
    'Draw when draw_prob > 27%': {
        'condition': lambda r: r['draw_prob'] > 0.27
    },
    'Draw when draw_prob > 26% AND predicted < 52%': {
        'condition': lambda r: r['draw_prob'] > 0.26 and max(r['home_prob'], r['away_prob']) < 0.52
    },
    'Draw when home_prob and away_prob are close (within 12%)': {
        'condition': lambda r: abs(r['home_prob'] - r['away_prob']) < 0.12
    }
}

results = []

for strategy_name, config in strategies.items():
    draw_bets = df[df.apply(config['condition'], axis=1)]

    if len(draw_bets) == 0:
        continue

    correct = (draw_bets['actual'] == 'Draw').sum()
    total = len(draw_bets)
    win_rate = correct / total * 100

    # Calculate P&L
    stake = 10
    total_staked = total * stake
    total_return = 0

    for _, row in draw_bets.iterrows():
        draw_odds = 1 / row['draw_prob']
        if row['actual'] == 'Draw':
            profit = stake * (draw_odds - 1)
            total_return += profit
        else:
            total_return -= stake

    roi = (total_return / total_staked * 100)

    results.append({
        'Strategy': strategy_name,
        'Bets': total,
        'Draws': correct,
        'Win Rate %': win_rate,
        'Total Staked': total_staked,
        'Net P&L': total_return,
        'ROI %': roi
    })

results_df = pd.DataFrame(results).sort_values('Net P&L', ascending=False)

print("\n" + "="*80)
print("DRAW STRATEGY RESULTS")
print("="*80)
print()
print(results_df.to_string(index=False))

# Find best
if len(results_df) > 0:
    best = results_df.iloc[0]
    print("\n" + "="*80)
    print("BEST DRAW STRATEGY")
    print("="*80)
    print(f"\n{best['Strategy']}")
    print(f"  Bets: {best['Bets']:.0f}")
    print(f"  Actual draws: {best['Draws']:.0f} ({best['Win Rate %']:.1f}%)")
    print(f"  Net P&L: £{best['Net P&L']:+.2f}")
    print(f"  ROI: {best['ROI %']:+.2f}%")

# Compare to actual distribution
print("\n" + "="*80)
print("COMPARISON TO ACTUAL OUTCOMES")
print("="*80)
print(f"\nActual outcome distribution:")
print(f"  Home wins: {(df['actual'] == 'Home Win').sum()} ({(df['actual'] == 'Home Win').sum()/len(df)*100:.1f}%)")
print(f"  Draws:     {(df['actual'] == 'Draw').sum()} ({(df['actual'] == 'Draw').sum()/len(df)*100:.1f}%)")
print(f"  Away wins: {(df['actual'] == 'Away Win').sum()} ({(df['actual'] == 'Away Win').sum()/len(df)*100:.1f}%)")

print(f"\nModel's average predictions:")
print(f"  Home prob: {df['home_prob'].mean():.1%}")
print(f"  Draw prob: {df['draw_prob'].mean():.1%}")
print(f"  Away prob: {df['away_prob'].mean():.1%}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nThe model needs RECALIBRATION:")
print("  1. Reduce home advantage from 50 to 30-35 Elo points")
print("  2. This will lower home_prob and allow draw predictions")
print("  3. Retrain the model with new home advantage")
print("\nCurrent issue: Model predicts Home ~47-54%, Draw ~25%, Away ~25-28%")
print("Reality:       Home 41%, Draw 27%, Away 33%")
print("\nHome advantage is STILL TOO HIGH!")
