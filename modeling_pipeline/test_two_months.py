#!/usr/bin/env python3
"""
Test model performance over the last 2 months.
Sample representative days to get comprehensive performance statistics.
"""

import subprocess
import pandas as pd
from datetime import datetime, timedelta
import json

# Sample dates across last 2 months (Nov 19 - Jan 19)
# Focus on weekends and mid-week days with high match counts
test_dates = [
    # Late November 2025
    '2025-11-23', '2025-11-24', '2025-11-27', '2025-11-30',

    # December 2025
    '2025-12-01', '2025-12-07', '2025-12-08', '2025-12-14', '2025-12-15',
    '2025-12-21', '2025-12-22', '2025-12-26', '2025-12-28', '2025-12-29',

    # January 2026 (we already have some, but add more)
    '2026-01-01', '2026-01-04', '2026-01-05', '2026-01-11', '2026-01-12',
    '2026-01-14', '2026-01-15', '2026-01-17', '2026-01-18'
]

results = []

print("="*80)
print("TESTING MODEL PERFORMANCE - LAST 2 MONTHS")
print("="*80)
print(f"\nTesting {len(test_dates)} days from {test_dates[0]} to {test_dates[-1]}")
print("\nThis will take several minutes...\n")

for i, date in enumerate(test_dates, 1):
    print(f"[{i}/{len(test_dates)}] Testing {date}...", end=' ', flush=True)

    # Generate predictions
    pred_file = f'predictions_{date.replace("-", "_")}.csv'
    try:
        subprocess.run(
            ['python', 'predict_live.py', '--date', date, '--output', pred_file],
            capture_output=True,
            timeout=300,
            check=False
        )

        # Check results
        result = subprocess.run(
            ['python', 'check_results.py', pred_file, date],
            capture_output=True,
            timeout=60,
            text=True,
            check=False
        )

        # Parse output
        output = result.stdout

        # Extract stats
        matches = 0
        correct = 0
        accuracy = 0

        for line in output.split('\n'):
            if 'Finished Matches:' in line:
                matches = int(line.split(':')[1].strip())
            elif 'Correct Predictions:' in line:
                correct = int(line.split(':')[1].strip())
            elif 'Accuracy:' in line and '%' in line:
                accuracy = float(line.split(':')[1].strip().replace('%', ''))

        if matches > 0:
            results.append({
                'date': date,
                'matches': matches,
                'correct': correct,
                'accuracy': accuracy
            })
            print(f"{matches} matches, {correct} correct ({accuracy:.1f}%)")
        else:
            print("No finished matches")

    except Exception as e:
        print(f"Error: {e}")
        continue

# Create summary
if results:
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    total_matches = df['matches'].sum()
    total_correct = df['correct'].sum()
    overall_accuracy = (total_correct / total_matches * 100) if total_matches > 0 else 0

    print(f"\nTotal Days Tested: {len(df)}")
    print(f"Total Matches: {total_matches}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")

    print(f"\nAccuracy Statistics:")
    print(f"  Mean: {df['accuracy'].mean():.1f}%")
    print(f"  Median: {df['accuracy'].median():.1f}%")
    print(f"  Std Dev: {df['accuracy'].std():.1f}%")
    print(f"  Min: {df['accuracy'].min():.1f}%")
    print(f"  Max: {df['accuracy'].max():.1f}%")

    print(f"\nMatches Per Day:")
    print(f"  Mean: {df['matches'].mean():.1f}")
    print(f"  Total: {df['matches'].sum()}")

    # Performance categories
    excellent = df[df['accuracy'] >= 60]
    good = df[(df['accuracy'] >= 50) & (df['accuracy'] < 60)]
    average = df[(df['accuracy'] >= 40) & (df['accuracy'] < 50)]
    poor = df[df['accuracy'] < 40]

    print(f"\nPerformance Breakdown:")
    print(f"  Excellent (≥60%): {len(excellent)} days ({len(excellent)/len(df)*100:.1f}%)")
    print(f"  Good (50-59%): {len(good)} days ({len(good)/len(df)*100:.1f}%)")
    print(f"  Average (40-49%): {len(average)} days ({len(average)/len(df)*100:.1f}%)")
    print(f"  Poor (<40%): {len(poor)} days ({len(poor)/len(df)*100:.1f}%)")

    # Save detailed results
    df.to_csv('two_month_performance.csv', index=False)
    print(f"\nDetailed results saved to: two_month_performance.csv")

    # Best and worst days
    print("\n" + "="*80)
    print("BEST PERFORMING DAYS")
    print("="*80)
    for idx, row in df.nlargest(5, 'accuracy').iterrows():
        print(f"{row['date']}: {row['accuracy']:.1f}% ({row['correct']}/{row['matches']} correct)")

    print("\n" + "="*80)
    print("WORST PERFORMING DAYS")
    print("="*80)
    for idx, row in df.nsmallest(5, 'accuracy').iterrows():
        print(f"{row['date']}: {row['accuracy']:.1f}% ({row['correct']}/{row['matches']} correct)")

    # Monthly breakdown
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly = df.groupby('month').agg({
        'matches': 'sum',
        'correct': 'sum',
        'accuracy': 'mean'
    })
    monthly['actual_accuracy'] = (monthly['correct'] / monthly['matches'] * 100)

    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN")
    print("="*80)
    for month, row in monthly.iterrows():
        print(f"\n{month}:")
        print(f"  Matches: {row['matches']}")
        print(f"  Correct: {row['correct']}")
        print(f"  Accuracy: {row['actual_accuracy']:.1f}%")

else:
    print("\nNo results collected")
