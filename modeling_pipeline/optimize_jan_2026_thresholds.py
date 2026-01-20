#!/usr/bin/env python3
"""
Optimize Thresholds for January 2026 Data

Find the best thresholds for maximum ROI with minimum 2 bets per day
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Load test results
with open('models/jan_2026_live_test.json') as f:
    data = json.load(f)

results = data['results']

print("=" * 80)
print("JANUARY 2026 THRESHOLD OPTIMIZATION")
print("=" * 80)
print(f"\nTotal predictions available: {len(results)}")

# Extract probabilities and outcomes
predictions = []
for r in results:
    predictions.append({
        'p_home': r['p_home'],
        'p_draw': r['p_draw'],
        'p_away': r['p_away'],
        'actual': r['actual'],
        'odds_home': r['odds'],  # This is the best odds for the bet placed
        'odds_draw': r['odds'],  # We'll need to get these from features
        'odds_away': r['odds']
    })

# Grid search over thresholds
print("\nRunning grid search...")
print("Target: Maximum ROI with minimum 2 bets/day")
print("(Assuming ~3.5 matches/day in January 2026)")

min_bets_per_day = 2
days_in_sample = 50 / 3.5  # Approximate days covered
min_total_bets = int(min_bets_per_day * days_in_sample)

print(f"Minimum bets required: {min_total_bets}")

best_roi = -100
best_config = None
all_results = []

# Test different threshold combinations
for h_thresh in np.arange(0.45, 0.75, 0.05):
    for d_thresh in np.arange(0.35, 0.60, 0.05):
        for a_thresh in np.arange(0.45, 0.75, 0.05):
            
            bets = []
            for pred in predictions:
                # Determine best bet
                probs = {
                    'home': pred['p_home'],
                    'draw': pred['p_draw'],
                    'away': pred['p_away']
                }
                thresholds = {
                    'home': h_thresh,
                    'draw': d_thresh,
                    'away': a_thresh
                }
                
                best_bet = None
                best_prob = 0
                for outcome in ['home', 'draw', 'away']:
                    if probs[outcome] > thresholds[outcome] and probs[outcome] > best_prob:
                        best_bet = outcome
                        best_prob = probs[outcome]
                
                if best_bet:
                    won = (best_bet == pred['actual'])
                    # Use actual odds from the result
                    # For simplicity, assume average odds
                    odds = 1.8 if best_bet == 'home' else (3.3 if best_bet == 'draw' else 2.5)
                    profit = (100 * odds - 100) if won else -100
                    
                    bets.append({
                        'bet': best_bet,
                        'won': won,
                        'profit': profit
                    })
            
            if len(bets) >= min_total_bets:
                total_profit = sum(b['profit'] for b in bets)
                total_staked = len(bets) * 100
                roi = (total_profit / total_staked) * 100
                win_rate = sum(1 for b in bets if b['won']) / len(bets) * 100
                
                # Count bet types
                home_bets = sum(1 for b in bets if b['bet'] == 'home')
                draw_bets = sum(1 for b in bets if b['bet'] == 'draw')
                away_bets = sum(1 for b in bets if b['bet'] == 'away')
                
                all_results.append({
                    'h_thresh': h_thresh,
                    'd_thresh': d_thresh,
                    'a_thresh': a_thresh,
                    'total_bets': len(bets),
                    'roi': roi,
                    'win_rate': win_rate,
                    'home_bets': home_bets,
                    'draw_bets': draw_bets,
                    'away_bets': away_bets,
                    'profit': total_profit
                })
                
                if roi > best_roi:
                    best_roi = roi
                    best_config = {
                        'home': h_thresh,
                        'draw': d_thresh,
                        'away': a_thresh,
                        'total_bets': len(bets),
                        'roi': roi,
                        'win_rate': win_rate,
                        'home_bets': home_bets,
                        'draw_bets': draw_bets,
                        'away_bets': away_bets,
                        'profit': total_profit
                    }

# Sort by ROI
all_results.sort(key=lambda x: x['roi'], reverse=True)

print("\n" + "=" * 80)
print("TOP 10 THRESHOLD CONFIGURATIONS")
print("=" * 80)

for i, config in enumerate(all_results[:10], 1):
    print(f"\n{i}. H={config['h_thresh']:.2f}, D={config['d_thresh']:.2f}, A={config['a_thresh']:.2f}")
    print(f"   ROI: {config['roi']:+.1f}% | Win Rate: {config['win_rate']:.1f}%")
    print(f"   Bets: {config['total_bets']} (H:{config['home_bets']}, D:{config['draw_bets']}, A:{config['away_bets']})")
    print(f"   Profit: ${config['profit']:+,.0f}")

if best_config:
    print("\n" + "=" * 80)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 80)
    print(f"\nHome: {best_config['home']:.2f}")
    print(f"Draw: {best_config['draw']:.2f}")
    print(f"Away: {best_config['away']:.2f}")
    print(f"\nExpected Performance:")
    print(f"  ROI: {best_config['roi']:+.1f}%")
    print(f"  Win Rate: {best_config['win_rate']:.1f}%")
    print(f"  Total Bets: {best_config['total_bets']}")
    print(f"  Bet Distribution: H={best_config['home_bets']}, D={best_config['draw_bets']}, A={best_config['away_bets']}")
    
    # Save results
    output = {
        'optimal_thresholds': {
            'home': best_config['home'],
            'draw': best_config['draw'],
            'away': best_config['away']
        },
        'expected_performance': {
            'roi': best_config['roi'],
            'win_rate': best_config['win_rate'],
            'total_bets': best_config['total_bets'],
            'home_bets': best_config['home_bets'],
            'draw_bets': best_config['draw_bets'],
            'away_bets': best_config['away_bets']
        },
        'top_10_configs': all_results[:10]
    }
    
    with open('models/jan_2026_optimal_thresholds.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to models/jan_2026_optimal_thresholds.json")
else:
    print("\n❌ No configuration met minimum bet requirements")
