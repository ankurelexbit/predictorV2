"""
Extract Market Features Available at 1 Hour Before Kickoff

This script filters odds to only include those that were available
at the 1-hour mark before kickoff, giving us a realistic backtest
of what signals would be available at prediction time.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from tqdm import tqdm

# Sharp vs soft bookmakers
SHARP_BOOKMAKERS = {2, 5, 34}
SOFT_BOOKMAKERS = {1, 3, 16, 20, 23, 29, 35}


def filter_odds_by_timing(odds_list: List[Dict], kickoff: datetime, hours_before: float = 1.0) -> List[Dict]:
    """
    Filter odds to only those available at N hours before kickoff.

    An odd is "available" if its latest_bookmaker_update is BEFORE the cutoff time.
    This simulates what odds we'd see at that point in time.
    """
    cutoff = kickoff - timedelta(hours=hours_before)

    filtered = []
    for odd in odds_list:
        updated = odd.get('latest_bookmaker_update', '')
        if not updated:
            continue

        try:
            updated_dt = datetime.strptime(updated[:19], '%Y-%m-%d %H:%M:%S')
            if updated_dt <= cutoff:
                filtered.append(odd)
        except:
            continue

    return filtered


def extract_1x2_odds(odds_list: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract 1X2 odds from filtered list"""
    result = {'home': [], 'draw': [], 'away': []}

    for odd in odds_list:
        market = odd.get('market_description', '') or odd.get('name', '')
        if market not in ['Fulltime Result', 'Match Winner', '1X2']:
            continue

        label = odd.get('label', '').lower()
        value_str = str(odd.get('value', 0)).replace(',', '')
        try:
            value = float(value_str)
        except:
            continue

        if value <= 1:
            continue

        entry = {
            'bookmaker_id': odd.get('bookmaker_id'),
            'odds': value,
            'implied_prob': 1 / value,
        }

        if label in ['home', '1', 'h']:
            result['home'].append(entry)
        elif label in ['draw', 'x', 'd']:
            result['draw'].append(entry)
        elif label in ['away', '2', 'a']:
            result['away'].append(entry)

    return result


def calculate_market_features(odds_1x2: Dict[str, List[Dict]]) -> Dict:
    """Calculate market features from filtered 1X2 odds"""
    features = {}

    for outcome in ['home', 'draw', 'away']:
        entries = odds_1x2[outcome]

        if not entries:
            features[f'{outcome}_best_odds_1hr'] = None
            features[f'{outcome}_implied_prob_1hr'] = None
            features[f'{outcome}_disagreement_1hr'] = None
            features[f'{outcome}_sharp_vs_soft_1hr'] = None
            continue

        odds_values = [e['odds'] for e in entries]
        probs = [e['implied_prob'] for e in entries]

        # Best odds
        features[f'{outcome}_best_odds_1hr'] = max(odds_values)
        features[f'{outcome}_implied_prob_1hr'] = 1 / max(odds_values)

        # Disagreement
        features[f'{outcome}_disagreement_1hr'] = max(probs) - min(probs) if len(probs) > 1 else 0

        # Sharp vs soft
        sharp_odds = [e['odds'] for e in entries if e['bookmaker_id'] in SHARP_BOOKMAKERS]
        soft_odds = [e['odds'] for e in entries if e['bookmaker_id'] in SOFT_BOOKMAKERS]

        if sharp_odds and soft_odds:
            features[f'{outcome}_sharp_vs_soft_1hr'] = np.mean(sharp_odds) - np.mean(soft_odds)
        else:
            features[f'{outcome}_sharp_vs_soft_1hr'] = 0

    # Count available bookmakers
    all_bookmakers = set()
    for outcome in ['home', 'draw', 'away']:
        for e in odds_1x2[outcome]:
            all_bookmakers.add(e['bookmaker_id'])
    features['num_bookmakers_1hr'] = len(all_bookmakers)

    return features


def process_fixture(fixture: Dict, hours_before: float = 1.0) -> Optional[Dict]:
    """Process a single fixture for 1hr-before market features"""

    fixture_id = fixture.get('id')
    if not fixture_id:
        return None

    # Parse kickoff time
    kickoff_str = fixture.get('starting_at', '')
    try:
        kickoff = datetime.strptime(kickoff_str, '%Y-%m-%d %H:%M:%S')
    except:
        return None

    # Filter odds to 1hr before
    all_odds = fixture.get('odds', [])
    filtered_odds = filter_odds_by_timing(all_odds, kickoff, hours_before)

    if not filtered_odds:
        return None

    # Extract 1X2 and calculate features
    odds_1x2 = extract_1x2_odds(filtered_odds)
    features = calculate_market_features(odds_1x2)

    # Only return if we have meaningful data
    if features.get('home_best_odds_1hr') is None:
        return None

    result = {
        'fixture_id': fixture_id,
        'match_date': kickoff_str,
    }
    result.update(features)

    return result


def process_all_fixtures(data_dir: str, output_path: str, leagues: List[int] = None, hours_before: float = 1.0):
    """Process all fixtures and extract 1hr-before market features"""

    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('*.json'))

    print(f"Processing {len(json_files)} fixture files")
    print(f"Extracting market features available {hours_before}hr before kickoff")

    all_results = []

    for json_file in tqdm(json_files, desc="Processing"):
        # Filter by league if specified
        if leagues:
            if not any(f'league_{lid}_' in json_file.name for lid in leagues):
                continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'data' in data:
                fixtures = data['data']
            elif isinstance(data, list):
                fixtures = data
            else:
                fixtures = [data]

            for fixture in fixtures:
                result = process_fixture(fixture, hours_before)
                if result:
                    all_results.append(result)

        except Exception as e:
            continue

    # Create DataFrame
    df = pd.DataFrame(all_results)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    print(f"\nExtracted {len(df):,} fixtures with 1hr-before market features")
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")

    # Compare coverage to closing odds
    closing_df = pd.read_csv('data/market_features.csv')
    overlap = len(set(df['fixture_id']) & set(closing_df['fixture_id']))
    print(f"Overlap with closing odds data: {overlap:,} fixtures")

    coverage = len(df) / len(closing_df) * 100
    print(f"Coverage: {coverage:.1f}% of fixtures have 1hr-before data")

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df


if __name__ == '__main__':
    df = process_all_fixtures(
        data_dir='data/historical/fixtures',
        output_path='data/market_features_1hr.csv',
        leagues=[8, 301, 384, 82, 564],
        hours_before=1.0
    )

    # Quick analysis
    print("\n" + "=" * 60)
    print("Feature Statistics (1hr before)")
    print("=" * 60)

    for col in df.columns:
        if col not in ['fixture_id', 'match_date']:
            non_null = df[col].notna().sum()
            if non_null > 0:
                mean_val = df[col].mean()
                print(f"  {col}: {non_null:,} non-null, mean={mean_val:.3f}")
