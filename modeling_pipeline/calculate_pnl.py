"""Calculate detailed P&L from prediction CSV files."""
import pandas as pd
import subprocess
from pathlib import Path
import json

print("="*80)
print("DETAILED P&L CALCULATION")
print("="*80)

# Read dates
with open('last_10_days.txt', 'r') as f:
    dates = [line.strip() for line in f.readlines()]

all_bets = []
daily_summary = []

for date in dates:
    pred_file = f'predictions_{date.replace("-", "_")}.csv'

    if not Path(pred_file).exists():
        print(f"\n⚠️  {date}: Prediction file not found")
        continue

    # Load predictions
    preds = pd.read_csv(pred_file)

    if len(preds) == 0:
        print(f"\n⚠️  {date}: No predictions")
        continue

    print(f"\n{date}: {len(preds)} predictions")

    # Get actual results
    result = subprocess.run(
        ['python', 'check_results.py', pred_file, date],
        capture_output=True,
        timeout=60,
        text=True
    )

    # Parse results from check_results output
    lines = result.stdout.split('\n')

    match_results = []
    current_match = None

    for i, line in enumerate(lines):
        # Look for match lines like "✅ Eredivisie - SC Heerenveen 0-2 FC Groningen"
        if ('✅' in line or '❌' in line) and ' - ' in line:
            # Extract match info
            is_correct = '✅' in line
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                league = parts[0].strip('✅❌ ')
                match_str = parts[1]

                # Next line has prediction and actual
                if i + 1 < len(lines):
                    pred_line = lines[i + 1]

                    if 'Predicted:' in pred_line and 'Actual:' in pred_line:
                        # Extract predicted and actual
                        pred_str = pred_line.split('Predicted:')[1].split('Actual:')[0].strip()
                        actual_str = pred_line.split('Actual:')[1].strip()

                        # Extract probabilities
                        probs = {'home': 0.33, 'draw': 0.33, 'away': 0.33}
                        if '(H:' in pred_line:
                            prob_part = pred_line.split('(')[1].split(')')[0]
                            for p in prob_part.split('|'):
                                if 'H:' in p:
                                    probs['home'] = float(p.split(':')[1].strip().rstrip('%')) / 100
                                elif 'D:' in p:
                                    probs['draw'] = float(p.split(':')[1].strip().rstrip('%')) / 100
                                elif 'A:' in p:
                                    probs['away'] = float(p.split(':')[1].strip().rstrip('%')) / 100

                        match_results.append({
                            'date': date,
                            'league': league,
                            'match': match_str.split(' ')[0] if ' ' in match_str else match_str[:30],
                            'predicted': pred_str,
                            'actual': actual_str,
                            'correct': is_correct,
                            'home_prob': probs['home'],
                            'draw_prob': probs['draw'],
                            'away_prob': probs['away']
                        })

    if match_results:
        print(f"  Parsed {len(match_results)} match results")
        all_bets.extend(match_results)

        correct = sum(1 for m in match_results if m['correct'])
        accuracy = correct / len(match_results) * 100
        daily_summary.append({
            'date': date,
            'matches': len(match_results),
            'correct': correct,
            'accuracy': accuracy
        })
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(match_results)})")
    else:
        print(f"  ⚠️  No match results parsed")

# Create DataFrame
bets_df = pd.DataFrame(all_bets)
daily_df = pd.DataFrame(daily_summary)

if len(bets_df) == 0:
    print("\n❌ No betting data collected!")
    exit(1)

print("\n" + "="*80)
print("P&L ANALYSIS")
print("="*80)

# Calculate implied odds (fair odds from our probabilities)
def get_odds(row):
    if row['predicted'] == 'Home Win':
        return 1 / row['home_prob']
    elif row['predicted'] == 'Draw':
        return 1 / row['draw_prob']
    else:  # Away Win
        return 1 / row['away_prob']

bets_df['implied_odds'] = bets_df.apply(get_odds, axis=1)

# Calculate returns with different stake amounts
STAKE = 10  # £10 per bet

def calculate_return(row, stake=STAKE):
    if row['correct']:
        return stake * (row['implied_odds'] - 1)  # Profit
    else:
        return -stake  # Loss

bets_df['return'] = bets_df.apply(lambda r: calculate_return(r, STAKE), axis=1)
bets_df['cumulative_return'] = bets_df['return'].cumsum()

# Overall P&L
total_bets = len(bets_df)
winning_bets = bets_df['correct'].sum()
losing_bets = total_bets - winning_bets
total_staked = total_bets * STAKE
total_return = bets_df['return'].sum()
roi = (total_return / total_staked * 100) if total_staked > 0 else 0

print(f"\n💷 P&L SUMMARY")
print(f"  Stake per bet: £{STAKE}")
print(f"  Total bets: {total_bets}")
print(f"  Winning bets: {winning_bets} ({winning_bets/total_bets*100:.1f}%)")
print(f"  Losing bets: {losing_bets} ({losing_bets/total_bets*100:.1f}%)")
print(f"  Total staked: £{total_staked:,.2f}")
print(f"  Total return: £{total_return:+,.2f}")
print(f"  ROI: {roi:+.2f}%")

# Breakdown by outcome type
print(f"\n📊 BREAKDOWN BY PREDICTION TYPE")
for pred_type in ['Home Win', 'Draw', 'Away Win']:
    subset = bets_df[bets_df['predicted'] == pred_type]
    if len(subset) > 0:
        type_bets = len(subset)
        type_correct = subset['correct'].sum()
        type_return = subset['return'].sum()
        type_roi = (type_return / (type_bets * STAKE) * 100)
        print(f"  {pred_type:12s}: {type_bets:3d} bets, {type_correct:3d} correct ({type_correct/type_bets*100:5.1f}%), P&L: £{type_return:+7.2f}, ROI: {type_roi:+6.2f}%")

# Daily breakdown
print(f"\n📅 DAILY BREAKDOWN")
daily_pnl = bets_df.groupby('date').agg({
    'return': 'sum',
    'correct': ['sum', 'count']
}).reset_index()
daily_pnl.columns = ['date', 'pnl', 'correct', 'total']
daily_pnl['accuracy'] = daily_pnl['correct'] / daily_pnl['total'] * 100
daily_pnl = daily_pnl.sort_values('pnl', ascending=False)

for _, row in daily_pnl.iterrows():
    print(f"  {row['date']}: {row['total']:2.0f} bets, {row['correct']:2.0f} correct ({row['accuracy']:5.1f}%), P&L: £{row['pnl']:+7.2f}")

# Best/worst bets
print(f"\n🏆 TOP 5 WINNING BETS")
top_wins = bets_df.nlargest(5, 'return')
for _, row in top_wins.iterrows():
    print(f"  {row['date']} | {row['league']:20s} | {row['predicted']:12s} @ {row['implied_odds']:.2f} → £{row['return']:+6.2f}")

print(f"\n💸 TOP 5 LOSING BETS")
top_losses = bets_df.nsmallest(5, 'return')
for _, row in top_losses.iterrows():
    pred_str = f"{row['predicted']:12s} @ {row['implied_odds']:.2f}"
    print(f"  {row['date']} | {row['league']:20s} | {pred_str} (Actual: {row['actual']:12s}) → £{row['return']:+6.2f}")

# Save detailed report
report = {
    'summary': {
        'period': f"{dates[0]} to {dates[-1]}",
        'total_bets': int(total_bets),
        'winning_bets': int(winning_bets),
        'losing_bets': int(losing_bets),
        'win_rate': float(winning_bets/total_bets*100),
        'stake_per_bet': float(STAKE),
        'total_staked': float(total_staked),
        'total_return': float(total_return),
        'net_pnl': float(total_return),
        'roi_percent': float(roi)
    },
    'daily_performance': daily_df.to_dict('records'),
    'all_bets': bets_df.to_dict('records')
}

with open('detailed_pnl_report.json', 'w') as f:
    json.dump(report, f, indent=2)

bets_df.to_csv('all_bets_with_pnl.csv', index=False)
daily_pnl.to_csv('daily_pnl_summary.csv', index=False)

print(f"\n📁 REPORTS SAVED:")
print(f"  - detailed_pnl_report.json")
print(f"  - all_bets_with_pnl.csv")
print(f"  - daily_pnl_summary.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
