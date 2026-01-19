"""
Validate predictions against actual results.

Fetches actual match results and compares with predictions.
"""

import pandas as pd
import requests
import json
from datetime import datetime
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://api.sportmonks.com/v3/football"
API_KEY = "LmEYRdsf8CSmiblNVTn6JfV0y0s8tc4aQsEkJ6JuoBhOWa3Hd1FEanGcrijo"


def get_actual_results(date_str: str):
    """Fetch actual results for a specific date."""
    url = f"{BASE_URL}/fixtures/between/{date_str}/{date_str}"
    params = {
        'api_token': API_KEY,
        'include': 'participants;scores'
    }

    response = requests.get(url, params=params, verify=False, timeout=30)
    response.raise_for_status()
    data = response.json()

    results = []
    for fixture in data['data']:
        participants = fixture.get('participants', [])
        if len(participants) < 2:
            continue

        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

        if not home_team or not away_team:
            continue

        # Get scores
        scores = fixture.get('scores', [])
        home_score = None
        away_score = None

        for score in scores:
            if score.get('description') == 'CURRENT' or score.get('description') == 'FT':
                participant_id = score.get('participant_id')
                score_val = score.get('score', {}).get('goals')

                if participant_id == home_team['id']:
                    home_score = score_val
                elif participant_id == away_team['id']:
                    away_score = score_val

        # Determine outcome
        actual_outcome = None
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                actual_outcome = 'Home Win'
            elif away_score > home_score:
                actual_outcome = 'Away Win'
            else:
                actual_outcome = 'Draw'

        results.append({
            'home_team': home_team['name'],
            'away_team': away_team['name'],
            'home_score': home_score,
            'away_score': away_score,
            'actual_outcome': actual_outcome,
            'league': fixture.get('league', {}).get('name', 'Unknown')
        })

    return pd.DataFrame(results)


def compare_predictions_vs_actual(predictions_file: str, date_str: str):
    """Compare predictions with actual results."""
    print("=" * 80)
    print("PREDICTION VALIDATION")
    print("=" * 80)

    # Load predictions
    predictions_df = pd.read_csv(predictions_file)
    print(f"\nLoaded {len(predictions_df)} predictions from {predictions_file}")

    # Fetch actual results
    print(f"\nFetching actual results for {date_str}...")
    actual_df = get_actual_results(date_str)
    print(f"Found {len(actual_df)} matches with results")

    # Merge predictions with actual results
    merged = predictions_df.merge(
        actual_df,
        on=['home_team', 'away_team'],
        how='left',
        suffixes=('_pred', '_actual')
    )

    # Filter only matches with actual results
    validated = merged[merged['actual_outcome'].notna()].copy()

    print(f"\nMatched {len(validated)} predictions with actual results")

    if len(validated) == 0:
        print("\n⚠️  No matches could be validated (no results found)")
        return

    # Calculate accuracy
    validated['correct'] = validated['predicted_outcome'] == validated['actual_outcome']

    total = len(validated)
    correct = validated['correct'].sum()
    accuracy = correct / total * 100

    # Calculate by outcome type
    home_wins_predicted = (validated['predicted_outcome'] == 'Home Win').sum()
    draws_predicted = (validated['predicted_outcome'] == 'Draw').sum()
    away_wins_predicted = (validated['predicted_outcome'] == 'Away Win').sum()

    home_wins_actual = (validated['actual_outcome'] == 'Home Win').sum()
    draws_actual = (validated['actual_outcome'] == 'Draw').sum()
    away_wins_actual = (validated['actual_outcome'] == 'Away Win').sum()

    # Print summary
    print("\n" + "=" * 80)
    print("OVERALL ACCURACY")
    print("=" * 80)
    print(f"Total Predictions: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.1f}%")

    print("\n" + "=" * 80)
    print("PREDICTION DISTRIBUTION")
    print("=" * 80)
    print(f"Home Wins Predicted: {home_wins_predicted} ({home_wins_predicted/total*100:.1f}%)")
    print(f"Draws Predicted: {draws_predicted} ({draws_predicted/total*100:.1f}%)")
    print(f"Away Wins Predicted: {away_wins_predicted} ({away_wins_predicted/total*100:.1f}%)")

    print("\n" + "=" * 80)
    print("ACTUAL DISTRIBUTION")
    print("=" * 80)
    print(f"Home Wins Actual: {home_wins_actual} ({home_wins_actual/total*100:.1f}%)")
    print(f"Draws Actual: {draws_actual} ({draws_actual/total*100:.1f}%)")
    print(f"Away Wins Actual: {away_wins_actual} ({away_wins_actual/total*100:.1f}%)")

    # Detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for idx, row in validated.iterrows():
        result_emoji = "✅" if row['correct'] else "❌"
        print(f"\n{result_emoji} {row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']}")
        print(f"   League: {row['league_actual']}")
        print(f"   Predicted: {row['predicted_outcome']} (H:{row['home_win_prob']:.1%} D:{row['draw_prob']:.1%} A:{row['away_win_prob']:.1%})")
        print(f"   Actual: {row['actual_outcome']}")

    # Confusion matrix
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print("\nActual →   Home Win  |  Draw  | Away Win")
    print("Predicted ↓")

    for pred_outcome in ['Home Win', 'Draw', 'Away Win']:
        row_str = f"{pred_outcome:10s} | "
        for actual_outcome in ['Home Win', 'Draw', 'Away Win']:
            count = ((validated['predicted_outcome'] == pred_outcome) &
                    (validated['actual_outcome'] == actual_outcome)).sum()
            row_str += f"{count:4d}    | "
        print(row_str)

    # Calculate probability calibration (Brier score)
    validated['brier_home'] = (validated['home_win_prob'] - (validated['actual_outcome'] == 'Home Win').astype(int)) ** 2
    validated['brier_draw'] = (validated['draw_prob'] - (validated['actual_outcome'] == 'Draw').astype(int)) ** 2
    validated['brier_away'] = (validated['away_win_prob'] - (validated['actual_outcome'] == 'Away Win').astype(int)) ** 2
    validated['brier_total'] = (validated['brier_home'] + validated['brier_draw'] + validated['brier_away']) / 3

    avg_brier = validated['brier_total'].mean()

    print("\n" + "=" * 80)
    print("PROBABILITY CALIBRATION")
    print("=" * 80)
    print(f"Average Brier Score: {avg_brier:.4f}")
    print(f"(Lower is better, perfect = 0.0, random = 0.67)")

    # Save detailed results
    output_file = predictions_file.replace('.csv', '_validated.csv')
    validated.to_csv(output_file, index=False)
    print(f"\n✅ Detailed validation saved to: {output_file}")

    return validated


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_predictions.py <predictions_file> [date_str]")
        print("Example: python validate_predictions.py predictions_jan_18_lineup_test.csv 2026-01-18")
        sys.exit(1)

    predictions_file = sys.argv[1]
    date_str = sys.argv[2] if len(sys.argv) > 2 else '2026-01-18'

    compare_predictions_vs_actual(predictions_file, date_str)
