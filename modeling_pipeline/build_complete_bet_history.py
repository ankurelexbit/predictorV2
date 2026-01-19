"""Build complete bet history with predictions, probabilities, and outcomes."""
import pandas as pd
import subprocess
from pathlib import Path
import re

print("="*80)
print("BUILDING COMPLETE BET HISTORY")
print("="*80)

# Read dates
with open('last_10_days.txt', 'r') as f:
    dates = [line.strip() for line in f.readlines() if line.strip()]

all_bets = []

for date in dates:
    pred_file = f'predictions_{date.replace("-", "_")}.csv'

    if not Path(pred_file).exists():
        print(f"\n⚠️  {date}: Prediction file not found")
        continue

    # Load predictions with probabilities
    preds_df = pd.read_csv(pred_file)

    if len(preds_df) == 0:
        print(f"\n⚠️  {date}: No predictions")
        continue

    print(f"\n{date}: {len(preds_df)} predictions")

    # Get actual results
    result = subprocess.run(
        ['python', 'check_results.py', pred_file, date],
        capture_output=True,
        timeout=60,
        text=True
    )

    output = result.stdout
    lines = output.split('\n')

    # Parse match results
    match_outcomes = {}

    for i, line in enumerate(lines):
        if ('✅' in line or '❌' in line) and ' - ' in line:
            is_correct = '✅' in line

            # Extract match info
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                league = parts[0].strip('✅❌ ')
                match_str = parts[1]

                # Next line has prediction details
                if i + 1 < len(lines):
                    pred_line = lines[i + 1]

                    if 'Predicted:' in pred_line and 'Actual:' in pred_line:
                        # Extract actual outcome
                        actual_str = pred_line.split('Actual:')[1].strip()

                        # Store with league + match_str as key for uniqueness
                        match_key = f"{league}|{match_str[:50]}"
                        match_outcomes[match_key] = {
                            'correct': is_correct,
                            'actual': actual_str,
                            'league': league
                        }

    # Match predictions to outcomes
    for _, pred_row in preds_df.iterrows():
        league = pred_row['league']
        home_team = pred_row['home_team']
        away_team = pred_row['away_team']

        # Try to find matching outcome by league and teams
        outcome = None
        for key, data in match_outcomes.items():
            if data['league'] == league:
                # Check if this match hasn't been used yet
                if 'used' not in data:
                    outcome = data
                    data['used'] = True  # Mark as used
                    break

        if outcome:
            # Calculate fair odds from probabilities
            home_odds = 1 / pred_row['home_win_prob']
            draw_odds = 1 / pred_row['draw_prob']
            away_odds = 1 / pred_row['away_win_prob']

            # Add complete bet record
            all_bets.append({
                'date': date,
                'league': pred_row['league'],
                'home_team': pred_row['home_team'],
                'away_team': pred_row['away_team'],
                'predicted': pred_row['predicted_outcome'],
                'actual': outcome['actual'],
                'correct': outcome['correct'],
                'home_prob': pred_row['home_win_prob'],
                'draw_prob': pred_row['draw_prob'],
                'away_prob': pred_row['away_win_prob'],
                'home_odds': home_odds,
                'draw_odds': draw_odds,
                'away_odds': away_odds
            })
        else:
            print(f"  ⚠️ No outcome found for: {league}")

# Create DataFrame
bets_df = pd.DataFrame(all_bets)

if len(bets_df) == 0:
    print("\n❌ No betting data collected!")
    exit(1)

print("\n" + "="*80)
print("COMPLETE BET HISTORY")
print("="*80)

print(f"\nTotal bets: {len(bets_df)}")
print(f"Correct predictions: {bets_df['correct'].sum()} ({bets_df['correct'].sum()/len(bets_df)*100:.1f}%)")
print(f"\nColumns: {list(bets_df.columns)}")

# Save complete dataset
bets_df.to_csv('complete_bet_history.csv', index=False)

print(f"\n✅ Saved complete_bet_history.csv with {len(bets_df)} bets")
print(f"   Columns include: probabilities, odds, predicted, actual, correct")
print("="*80)
