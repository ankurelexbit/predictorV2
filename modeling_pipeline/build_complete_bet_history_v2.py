"""Build complete bet history by parsing check_results output."""
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

    print(f"\n{date}: Processing...")

    # Get actual results
    result = subprocess.run(
        ['python', 'check_results.py', pred_file, date],
        capture_output=True,
        timeout=60,
        text=True
    )

    output = result.stdout
    lines = output.split('\n')

    # Parse matches
    for i, line in enumerate(lines):
        if ('✅' in line or '❌' in line) and ' - ' in line and i + 1 < len(lines):
            is_correct = '✅' in line

            # Parse match line: "✅ Bundesliga - Eintracht Frankfurt 3-3 Borussia Dortmund"
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                league = parts[0].strip('✅❌ ').strip()
                match_info = parts[1]

                # Extract team names (before the score)
                # Format: "Team1 X-Y Team2" or "Team1 vs Team2" if not finished
                score_match = re.search(r'(\d+)-(\d+)', match_info)
                if score_match:
                    # Extract actual result
                    home_score = int(score_match.group(1))
                    away_score = int(score_match.group(2))

                    if home_score > away_score:
                        actual_outcome = 'Home Win'
                    elif away_score > home_score:
                        actual_outcome = 'Away Win'
                    else:
                        actual_outcome = 'Draw'

                    # Get teams (split by score pattern)
                    teams_part = match_info.split(score_match.group(0))
                    if len(teams_part) == 2:
                        home_team = teams_part[0].strip()
                        away_team = teams_part[1].strip()
                    else:
                        home_team = "Unknown"
                        away_team = "Unknown"
                else:
                    actual_outcome = "Unknown"
                    home_team = "Unknown"
                    away_team = "Unknown"

                # Parse prediction line
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

                            # Calculate odds
                            home_odds = 1 / home_prob
                            draw_odds = 1 / draw_prob
                            away_odds = 1 / away_prob

                            all_bets.append({
                                'date': date,
                                'league': league,
                                'home_team': home_team,
                                'away_team': away_team,
                                'predicted': pred,
                                'actual': actual_outcome,
                                'correct': is_correct,
                                'home_prob': home_prob,
                                'draw_prob': draw_prob,
                                'away_prob': away_prob,
                                'home_odds': home_odds,
                                'draw_odds': draw_odds,
                                'away_odds': away_odds
                            })

    print(f"  Parsed {len([b for b in all_bets if b['date'] == date])} bets")

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
print(f"\nDate range: {bets_df['date'].min()} to {bets_df['date'].max()}")
print(f"Leagues: {bets_df['league'].nunique()} unique")

# Show sample
print(f"\nSample records:")
print(bets_df[['date', 'league', 'predicted', 'correct', 'home_prob', 'away_prob']].head())

# Save complete dataset
bets_df.to_csv('complete_bet_history.csv', index=False)

print(f"\n✅ Saved complete_bet_history.csv with {len(bets_df)} bets")
print(f"   Columns: {list(bets_df.columns)}")
print("="*80)
