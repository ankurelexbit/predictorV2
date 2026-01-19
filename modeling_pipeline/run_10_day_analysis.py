"""Run 10-day prediction analysis with P&L calculation."""
import subprocess
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

print("="*80)
print("10-DAY PREDICTION ANALYSIS WITH P&L")
print("="*80)

# Read dates
with open('last_10_days.txt', 'r') as f:
    dates = [line.strip() for line in f.readlines()]

print(f"\nAnalyzing {len(dates)} days: {dates[0]} to {dates[-1]}")
print("\nThis will take several minutes...\n")

all_predictions = []
all_results = []
daily_stats = []

for i, date in enumerate(dates, 1):
    print(f"[{i}/{len(dates)}] Processing {date}...")

    pred_file = f'predictions_{date.replace("-", "_")}.csv'

    # Run predictions
    try:
        result = subprocess.run(
            ['python', 'predict_live.py', '--date', date, '--output', pred_file],
            capture_output=True,
            timeout=300,
            text=True
        )

        if result.returncode != 0:
            print(f"  ⚠️  Prediction failed: {result.stderr[:100]}")
            continue

    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Timeout")
        continue
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        continue

    # Check if file was created
    if not Path(pred_file).exists():
        print(f"  ⚠️  Prediction file not created")
        continue

    # Get actual results
    try:
        result = subprocess.run(
            ['python', 'check_results.py', pred_file, date],
            capture_output=True,
            timeout=60,
            text=True
        )

        output = result.stdout

        # Parse accuracy
        matches = 0
        correct = 0
        accuracy = 0.0

        for line in output.split('\n'):
            if 'Finished Matches:' in line:
                matches = int(line.split(':')[1].strip())
            elif 'Correct Predictions:' in line:
                correct = int(line.split(':')[1].strip())
            elif 'Accuracy:' in line and '%' in line:
                accuracy = float(line.split(':')[1].strip().replace('%', ''))

        if matches > 0:
            daily_stats.append({
                'date': date,
                'matches': matches,
                'correct': correct,
                'accuracy': accuracy,
                'pred_file': pred_file
            })
            print(f"  ✓ {matches} matches, {correct} correct ({accuracy:.1f}%)")
        else:
            print(f"  ⚠️  No finished matches")

    except Exception as e:
        print(f"  ⚠️  Error checking results: {e}")
        continue

print("\n" + "="*80)
print("PROCESSING COMPLETE - Generating detailed report...")
print("="*80)

if not daily_stats:
    print("\n❌ No data collected!")
    exit(1)

# Create summary DataFrame
df = pd.DataFrame(daily_stats)

# Calculate overall statistics
total_matches = df['matches'].sum()
total_correct = df['correct'].sum()
overall_accuracy = (total_correct / total_matches * 100) if total_matches > 0 else 0

print(f"\n📊 OVERALL STATISTICS")
print(f"  Days analyzed: {len(df)}")
print(f"  Total matches: {total_matches}")
print(f"  Correct predictions: {total_correct}")
print(f"  Overall accuracy: {overall_accuracy:.2f}%")

# Load all predictions for detailed P&L
print("\n💰 CALCULATING P&L...")

all_bets = []
for _, row in df.iterrows():
    pred_file = row['pred_file']
    if Path(pred_file).exists():
        preds = pd.read_csv(pred_file)

        # Get actual results for this date
        result = subprocess.run(
            ['python', 'check_results.py', pred_file, row['date']],
            capture_output=True,
            timeout=60,
            text=True
        )

        # Parse detailed results
        lines = result.stdout.split('\n')

        for i, line in enumerate(lines):
            if 'Predicted:' in line and 'Actual:' in line:
                # Extract match info
                if i > 0:
                    match_line = lines[i-1]
                    if ' - ' in match_line and ' vs ' in match_line:
                        parts = match_line.split(' - ', 1)
                        if len(parts) == 2:
                            league = parts[0].strip('✅❌ ')
                            teams_score = parts[1]

                            # Extract prediction and actual
                            if 'Predicted:' in line:
                                pred_part = line.split('Predicted:')[1].split('Actual:')[0].strip()
                                actual_part = line.split('Actual:')[1].strip()

                                # Extract probabilities if available
                                probs = {}
                                if '(H:' in line:
                                    prob_str = line.split('(')[1].split(')')[0]
                                    for p in prob_str.split('|'):
                                        if 'H:' in p:
                                            probs['home'] = float(p.split(':')[1].strip().rstrip('%')) / 100
                                        elif 'D:' in p:
                                            probs['draw'] = float(p.split(':')[1].strip().rstrip('%')) / 100
                                        elif 'A:' in p:
                                            probs['away'] = float(p.split(':')[1].strip().rstrip('%')) / 100

                                all_bets.append({
                                    'date': row['date'],
                                    'league': league,
                                    'teams': teams_score.split(' ')[0] if ' ' in teams_score else teams_score,
                                    'predicted': pred_part,
                                    'actual': actual_part,
                                    'correct': pred_part == actual_part,
                                    'home_prob': probs.get('home', 0.33),
                                    'draw_prob': probs.get('draw', 0.33),
                                    'away_prob': probs.get('away', 0.33)
                                })

bets_df = pd.DataFrame(all_bets)

if len(bets_df) > 0:
    print(f"  Total bets analyzed: {len(bets_df)}")

    # Calculate implied odds (fair odds from probabilities)
    bets_df['home_odds'] = 1 / bets_df['home_prob']
    bets_df['draw_odds'] = 1 / bets_df['draw_prob']
    bets_df['away_odds'] = 1 / bets_df['away_prob']

    # Calculate returns (assuming £10 stake per bet)
    stake = 10

    def calculate_return(row):
        if row['predicted'] == 'Home Win':
            odds = row['home_odds']
        elif row['predicted'] == 'Draw':
            odds = row['draw_odds']
        else:
            odds = row['away_odds']

        if row['correct']:
            return stake * odds - stake  # Profit
        else:
            return -stake  # Loss

    bets_df['return'] = bets_df.apply(calculate_return, axis=1)

    # Calculate cumulative P&L
    total_pnl = bets_df['return'].sum()
    total_staked = len(bets_df) * stake
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    print(f"\n💷 P&L SUMMARY (£{stake} stake per bet)")
    print(f"  Total staked: £{total_staked:,.2f}")
    print(f"  Total return: £{total_pnl:+,.2f}")
    print(f"  ROI: {roi:+.2f}%")
    print(f"  Winning bets: {bets_df['correct'].sum()} / {len(bets_df)} ({bets_df['correct'].mean()*100:.1f}%)")

    # Best and worst days
    daily_pnl = bets_df.groupby('date')['return'].sum().sort_values(ascending=False)

    print(f"\n📈 BEST DAYS")
    for date, pnl in daily_pnl.head(3).items():
        day_bets = len(bets_df[bets_df['date'] == date])
        print(f"  {date}: £{pnl:+.2f} ({day_bets} bets)")

    print(f"\n📉 WORST DAYS")
    for date, pnl in daily_pnl.tail(3).items():
        day_bets = len(bets_df[bets_df['date'] == date])
        print(f"  {date}: £{pnl:+.2f} ({day_bets} bets)")

    # Save detailed report
    report_data = {
        'summary': {
            'period': f"{dates[0]} to {dates[-1]}",
            'days_analyzed': len(df),
            'total_matches': int(total_matches),
            'correct_predictions': int(total_correct),
            'overall_accuracy': float(overall_accuracy),
            'stake_per_bet': stake,
            'total_staked': float(total_staked),
            'total_pnl': float(total_pnl),
            'roi_percent': float(roi)
        },
        'daily_performance': df.to_dict('records'),
        'all_bets': bets_df.to_dict('records')
    }

    with open('10_day_analysis_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)

    # Save CSV
    bets_df.to_csv('10_day_all_bets.csv', index=False)
    df.to_csv('10_day_daily_stats.csv', index=False)

    print(f"\n📁 REPORTS SAVED:")
    print(f"  - 10_day_analysis_report.json (detailed JSON)")
    print(f"  - 10_day_all_bets.csv (all bets)")
    print(f"  - 10_day_daily_stats.csv (daily summary)")

else:
    print("  ⚠️  No betting data could be parsed")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
