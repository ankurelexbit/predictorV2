"""Simple P&L calculator using prediction CSVs and check_results output."""
import pandas as pd
import subprocess
from pathlib import Path
import re

print("="*80)
print("10-DAY P&L REPORT")
print("="*80)

# Read dates
with open('last_10_days.txt', 'r') as f:
    dates = [line.strip() for line in f.readlines()]

all_results = []
daily_stats = []

STAKE = 10  # £10 per bet

for date in dates:
    pred_file = f'predictions_{date.replace("-", "_")}.csv'

    if not Path(pred_file).exists():
        continue

    # Load predictions
    preds_df = pd.read_csv(pred_file)

    # Get actual results
    result = subprocess.run(
        ['python', 'check_results.py', pred_file, date],
        capture_output=True,
        timeout=60,
        text=True
    )

    output = result.stdout

    # Parse overall accuracy
    matches = 0
    correct = 0
    for line in output.split('\n'):
        if 'Finished Matches:' in line:
            matches = int(line.split(':')[1].strip())
        elif 'Correct Predictions:' in line:
            correct = int(line.split(':')[1].strip())

    if matches == 0:
        continue

    # Parse individual matches
    lines = output.split('\n')
    day_bets = []

    for i, line in enumerate(lines):
        if ('✅' in line or '❌' in line) and ' - ' in line and i + 1 < len(lines):
            is_correct = '✅' in line
            next_line = lines[i + 1]

            if 'Predicted:' in next_line:
                # Extract probabilities
                if '(H:' in next_line:
                    prob_match = re.search(r'H:\s*([\d.]+)%.*D:\s*([\d.]+)%.*A:\s*([\d.]+)%', next_line)
                    if prob_match:
                        home_prob = float(prob_match.group(1)) / 100
                        draw_prob = float(prob_match.group(2)) / 100
                        away_prob = float(prob_match.group(3)) / 100

                        # Extract prediction
                        pred = next_line.split('Predicted:')[1].split('(')[0].strip()

                        # Calculate implied odds
                        if pred == 'Home Win':
                            odds = 1 / home_prob
                        elif pred == 'Draw':
                            odds = 1 / draw_prob
                        else:  # Away Win
                            odds = 1 / away_prob

                        # Calculate return
                        if is_correct:
                            bet_return = STAKE * (odds - 1)  # Profit only
                        else:
                            bet_return = -STAKE  # Loss

                        day_bets.append({
                            'date': date,
                            'correct': is_correct,
                            'predicted': pred,
                            'odds': odds,
                            'return': bet_return
                        })

    if day_bets:
        day_df = pd.DataFrame(day_bets)
        all_results.extend(day_bets)

        day_pnl = day_df['return'].sum()
        day_correct = day_df['correct'].sum()
        day_total = len(day_df)
        day_accuracy = day_correct / day_total * 100

        daily_stats.append({
            'date': date,
            'bets': day_total,
            'correct': day_correct,
            'accuracy': day_accuracy,
            'pnl': day_pnl
        })

        print(f"\n{date}: {day_total} bets, {day_correct} correct ({day_accuracy:.1f}%), P&L: £{day_pnl:+.2f}")

# Overall statistics
if all_results:
    df = pd.DataFrame(all_results)
    daily_df = pd.DataFrame(daily_stats)

    total_bets = len(df)
    total_correct = df['correct'].sum()
    total_accuracy = total_correct / total_bets * 100
    total_staked = total_bets * STAKE
    total_pnl = df['return'].sum()
    roi = (total_pnl / total_staked * 100)

    print("\n" + "="*80)
    print("OVERALL P&L SUMMARY")
    print("="*80)

    print(f"\n💷 FINANCIAL SUMMARY")
    print(f"  Period: {dates[0]} to {dates[-1]}")
    print(f"  Stake per bet: £{STAKE}")
    print(f"  Total bets: {total_bets}")
    print(f"  Winning bets: {total_correct} ({total_accuracy:.1f}%)")
    print(f"  Losing bets: {total_bets - total_correct} ({100-total_accuracy:.1f}%)")
    print(f"  ")
    print(f"  Total staked: £{total_staked:,.2f}")
    print(f"  Total return: £{total_pnl:+,.2f}")
    print(f"  Net P&L: £{total_pnl:+,.2f}")
    print(f"  ROI: {roi:+.2f}%")

    # Breakdown by prediction type
    print(f"\n📊 BREAKDOWN BY OUTCOME")
    for pred_type in ['Home Win', 'Draw', 'Away Win']:
        subset = df[df['predicted'] == pred_type]
        if len(subset) > 0:
            type_bets = len(subset)
            type_correct = subset['correct'].sum()
            type_pnl = subset['return'].sum()
            type_staked = type_bets * STAKE
            type_roi = (type_pnl / type_staked * 100)
            avg_odds = subset['odds'].mean()

            print(f"  {pred_type:12s}: {type_bets:3d} bets @ avg {avg_odds:.2f}, " +
                  f"{type_correct:3d} wins ({type_correct/type_bets*100:5.1f}%), " +
                  f"P&L: £{type_pnl:+7.2f}, ROI: {type_roi:+6.2f}%")

    # Daily breakdown sorted by P&L
    print(f"\n📅 DAILY P&L (Best to Worst)")
    daily_sorted = daily_df.sort_values('pnl', ascending=False)
    for _, row in daily_sorted.iterrows():
        print(f"  {row['date']}: {row['bets']:2.0f} bets, {row['accuracy']:5.1f}% accuracy → £{row['pnl']:+7.2f}")

    # Statistics
    best_day = daily_df.loc[daily_df['pnl'].idxmax()]
    worst_day = daily_df.loc[daily_df['pnl'].idxmin()]
    avg_daily_pnl = daily_df['pnl'].mean()

    print(f"\n📈 DAILY STATISTICS")
    print(f"  Best day: {best_day['date']} (£{best_day['pnl']:+.2f})")
    print(f"  Worst day: {worst_day['date']} (£{worst_day['pnl']:+.2f})")
    print(f"  Average daily P&L: £{avg_daily_pnl:+.2f}")
    print(f"  Profitable days: {len(daily_df[daily_df['pnl'] > 0])}/{len(daily_df)}")

    # Save reports
    df.to_csv('10_day_all_bets.csv', index=False)
    daily_df.to_csv('10_day_daily_summary.csv', index=False)

    # Create JSON report
    import json
    report = {
        'period': f"{dates[0]} to {dates[-1]}",
        'stake_per_bet': STAKE,
        'total_bets': int(total_bets),
        'winning_bets': int(total_correct),
        'win_rate_percent': float(total_accuracy),
        'total_staked': float(total_staked),
        'net_pnl': float(total_pnl),
        'roi_percent': float(roi),
        'days_analyzed': len(daily_df),
        'profitable_days': int(len(daily_df[daily_df['pnl'] > 0])),
        'best_day': {
            'date': best_day['date'],
            'pnl': float(best_day['pnl'])
        },
        'worst_day': {
            'date': worst_day['date'],
            'pnl': float(worst_day['pnl'])
        }
    }

    with open('10_day_pnl_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📁 REPORTS SAVED:")
    print(f"  - 10_day_pnl_report.json")
    print(f"  - 10_day_all_bets.csv")
    print(f"  - 10_day_daily_summary.csv")

else:
    print("\n❌ No betting data found!")

print("\n" + "="*80)
