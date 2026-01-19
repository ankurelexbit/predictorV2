#!/usr/bin/env python3
"""
Check prediction accuracy by comparing predictions with actual results.
"""

import pandas as pd
import requests
import urllib3
from datetime import datetime
import warnings
import os
from dotenv import load_dotenv

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Load environment
load_dotenv()

# API Configuration
API_KEY = "LmEYRdsf8CSmiblNVTn6JfV0y0s8tc4aQsEkJ6JuoBhOWa3Hd1FEanGcrijo"
BASE_URL = 'https://api.sportmonks.com/v3/football'


def fetch_actual_results(date_str):
    """Fetch actual results for a specific date."""
    print(f'Fetching actual results for {date_str}...')

    url = f"{BASE_URL}/fixtures/between/{date_str}/{date_str}"
    params = {
        'api_token': API_KEY,
        'include': 'participants;scores;state'
    }

    try:
        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or 'data' not in data:
            print('No fixtures found')
            return []

        fixtures = data['data']
        print(f'Found {len(fixtures)} fixtures from API')

        # Parse results
        results = []
        finished_count = 0

        for fixture in fixtures:
            state_id = fixture.get('state_id')
            state_name = fixture.get('state', {}).get('name', 'Unknown')

            # Only process finished matches (state_id 5 = FT, 8 = AET/Penalties)
            if state_id not in [5, 8]:
                continue

            finished_count += 1
            participants = fixture.get('participants', [])
            home_participant = None
            away_participant = None

            for p in participants:
                if p.get('meta', {}).get('location') == 'home':
                    home_participant = p
                elif p.get('meta', {}).get('location') == 'away':
                    away_participant = p

            if home_participant and away_participant:
                home_team = home_participant.get('name', 'Unknown')
                away_team = away_participant.get('name', 'Unknown')

                # Extract scores from scores array (type_id 1525 = CURRENT/final score)
                scores = fixture.get('scores', [])
                home_score = 0
                away_score = 0

                for score in scores:
                    if score.get('type_id') == 1525:  # CURRENT score
                        score_data = score.get('score', {})
                        goals = score_data.get('goals', 0)
                        participant = score_data.get('participant', '')

                        if participant == 'home':
                            home_score = goals
                        elif participant == 'away':
                            away_score = goals

                if home_score > away_score:
                    actual_outcome = 'Home Win'
                elif away_score > home_score:
                    actual_outcome = 'Away Win'
                else:
                    actual_outcome = 'Draw'

                results.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'actual_outcome': actual_outcome,
                    'state': state_name
                })

        print(f'Found {finished_count} finished matches')
        return results

    except Exception as e:
        print(f'Error fetching results: {e}')
        return []


def compare_predictions(predictions_file, date_str):
    """Compare predictions with actual results."""
    # Load predictions
    predictions = pd.read_csv(predictions_file)
    print(f'\nLoaded {len(predictions)} predictions from {predictions_file}')

    # Fetch actual results
    results = fetch_actual_results(date_str)

    if not results:
        print('\n⚠️  No finished matches found yet. Matches may still be in progress or not started.')
        return

    results_df = pd.DataFrame(results)

    # Merge predictions with results
    comparison = predictions.merge(
        results_df,
        on=['home_team', 'away_team'],
        how='inner'
    )

    if len(comparison) == 0:
        print('\n⚠️  No matching predictions found. Team names may not match between predictions and API.')
        print('\nPredicted teams:')
        for idx, row in predictions.head(5).iterrows():
            print(f"  {row['home_team']} vs {row['away_team']}")
        print('\nActual teams:')
        for idx, row in results_df.head(5).iterrows():
            print(f"  {row['home_team']} vs {row['away_team']}")
        return

    # Calculate accuracy
    comparison['correct'] = comparison['predicted_outcome'] == comparison['actual_outcome']
    accuracy = comparison['correct'].mean() * 100

    total_matches = len(comparison)
    correct_predictions = comparison['correct'].sum()

    print('\n' + '='*70)
    print('PREDICTION PERFORMANCE ANALYSIS')
    print('='*70)
    print(f'\nDate: {date_str}')
    print(f'Total Predictions Made: {len(predictions)}')
    print(f'Finished Matches: {total_matches}')
    print(f'Correct Predictions: {correct_predictions}')
    print(f'Accuracy: {accuracy:.1f}%')

    # Breakdown by outcome
    print('\n--- Prediction Breakdown ---')
    outcome_counts = comparison['predicted_outcome'].value_counts()
    for outcome, count in outcome_counts.items():
        correct_for_outcome = comparison[comparison['predicted_outcome'] == outcome]['correct'].sum()
        print(f'{outcome}: {count} predictions, {correct_for_outcome} correct ({correct_for_outcome/count*100:.1f}%)')

    # Show detailed results
    print('\n' + '='*70)
    print('DETAILED MATCH RESULTS')
    print('='*70 + '\n')

    for idx, row in comparison.iterrows():
        status = '✅' if row['correct'] else '❌'
        print(f"{status} {row['league']} - {row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']}")
        print(f"   Predicted: {row['predicted_outcome']} (H: {row['home_win_prob']*100:.1f}% | D: {row['draw_prob']*100:.1f}% | A: {row['away_win_prob']*100:.1f}%)")
        print(f"   Actual: {row['actual_outcome']}\n")

    # Show unmatched predictions (if any)
    unmatched = predictions[~predictions['home_team'].isin(comparison['home_team'])]
    if len(unmatched) > 0:
        print(f'\n⚠️  {len(unmatched)} predictions not yet finished:')
        for idx, row in unmatched.iterrows():
            print(f"  {row['home_team']} vs {row['away_team']}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python check_results.py <predictions_file> [date]')
        print('Example: python check_results.py predictions_jan_18.csv 2026-01-18')
        sys.exit(1)

    predictions_file = sys.argv[1]
    date_str = sys.argv[2] if len(sys.argv) > 2 else '2026-01-18'

    compare_predictions(predictions_file, date_str)
