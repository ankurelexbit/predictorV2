"""
Extract Market Features from Raw Odds Data

This script processes the raw fixture JSON files to extract:
1. Bookmaker disagreement (uncertainty signal)
2. Line movements (sharp money signal)
3. Market implied probabilities
4. Sharp vs soft book spreads
5. Asian Handicap lines
6. Over/Under lines and odds

These features are completely unused in the current model but contain
valuable information about market sentiment and uncertainty.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import glob
from tqdm import tqdm


# Known sharp bookmakers (lower margins, move first)
SHARP_BOOKMAKERS = {2, 5, 34}  # Pinnacle-like books
SOFT_BOOKMAKERS = {1, 3, 16, 20, 23, 29, 35}  # Recreational books


def extract_1x2_odds(odds_list: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract 1X2 (Fulltime Result) odds from all bookmakers"""
    result = {'home': [], 'draw': [], 'away': []}

    for odd in odds_list:
        market = odd.get('market_description', '') or odd.get('name', '')
        if market not in ['Fulltime Result', 'Match Winner', '1X2']:
            continue

        label = odd.get('label', '').lower()
        bookmaker_id = odd.get('bookmaker_id')
        # Handle comma-separated numbers like "1,000.00"
        value_str = str(odd.get('value', 0)).replace(',', '')
        try:
            value = float(value_str)
        except:
            continue
        created = odd.get('created_at', '')
        updated = odd.get('latest_bookmaker_update', '')

        if value <= 1:
            continue

        entry = {
            'bookmaker_id': bookmaker_id,
            'odds': value,
            'implied_prob': 1 / value,
            'created_at': created,
            'updated_at': updated,
        }

        if label in ['home', '1', 'h']:
            result['home'].append(entry)
        elif label in ['draw', 'x', 'd']:
            result['draw'].append(entry)
        elif label in ['away', '2', 'a']:
            result['away'].append(entry)

    return result


def extract_asian_handicap(odds_list: List[Dict]) -> Dict:
    """Extract Asian Handicap lines and odds"""
    ah_odds = []

    for odd in odds_list:
        market = odd.get('market_description', '') or ''
        if 'Asian Handicap' not in market or 'Half' in market:
            continue

        handicap = odd.get('handicap')
        if handicap is None:
            continue

        try:
            handicap = float(handicap)
        except:
            continue

        label = odd.get('label', '').lower()
        if label not in ['home', 'away', '1', '2']:
            label = odd.get('name', '').lower()

        value_str = str(odd.get('value', 0)).replace(',', '')
        try:
            value = float(value_str)
        except:
            continue
        if value <= 1:
            continue

        ah_odds.append({
            'handicap': handicap,
            'side': 'home' if label in ['home', '1'] else 'away',
            'odds': value,
            'bookmaker_id': odd.get('bookmaker_id'),
        })

    # Find the main line (closest to even odds on both sides)
    if not ah_odds:
        return {'main_line': None, 'home_odds': None, 'away_odds': None}

    # Group by handicap
    by_line = defaultdict(list)
    for o in ah_odds:
        by_line[o['handicap']].append(o)

    # Find line with most balanced odds
    best_line = None
    best_balance = float('inf')
    for line, entries in by_line.items():
        home_odds = [e['odds'] for e in entries if e['side'] == 'home']
        away_odds = [e['odds'] for e in entries if e['side'] == 'away']
        if home_odds and away_odds:
            balance = abs(np.mean(home_odds) - np.mean(away_odds))
            if balance < best_balance:
                best_balance = balance
                best_line = line

    if best_line is not None:
        entries = by_line[best_line]
        home_odds = [e['odds'] for e in entries if e['side'] == 'home']
        away_odds = [e['odds'] for e in entries if e['side'] == 'away']
        return {
            'main_line': best_line,
            'home_odds': np.mean(home_odds) if home_odds else None,
            'away_odds': np.mean(away_odds) if away_odds else None,
        }

    return {'main_line': None, 'home_odds': None, 'away_odds': None}


def extract_over_under(odds_list: List[Dict]) -> Dict:
    """Extract Over/Under 2.5 goals odds"""
    ou_odds = {'over': [], 'under': []}

    for odd in odds_list:
        market = odd.get('market_description', '') or ''
        if market != 'Goals Over/Under':
            continue

        total = odd.get('total')
        if total != 2.5 and total != '2.5':
            continue

        label = odd.get('label', '').lower()
        value_str = str(odd.get('value', 0)).replace(',', '')
        try:
            value = float(value_str)
        except:
            continue
        if value <= 1:
            continue

        if 'over' in label:
            ou_odds['over'].append(value)
        elif 'under' in label:
            ou_odds['under'].append(value)

    return {
        'over_2_5_best': max(ou_odds['over']) if ou_odds['over'] else None,
        'over_2_5_avg': np.mean(ou_odds['over']) if ou_odds['over'] else None,
        'under_2_5_best': max(ou_odds['under']) if ou_odds['under'] else None,
        'under_2_5_avg': np.mean(ou_odds['under']) if ou_odds['under'] else None,
    }


def calculate_market_features(odds_1x2: Dict[str, List[Dict]]) -> Dict:
    """Calculate sophisticated market features from 1X2 odds"""

    features = {}

    for outcome in ['home', 'draw', 'away']:
        entries = odds_1x2[outcome]
        if not entries:
            features[f'{outcome}_best_odds'] = None
            features[f'{outcome}_avg_odds'] = None
            features[f'{outcome}_implied_prob'] = None
            features[f'{outcome}_bookmaker_disagreement'] = None
            features[f'{outcome}_sharp_vs_soft'] = None
            continue

        odds_values = [e['odds'] for e in entries]
        probs = [e['implied_prob'] for e in entries]

        # Best and average odds
        features[f'{outcome}_best_odds'] = max(odds_values)
        features[f'{outcome}_avg_odds'] = np.mean(odds_values)

        # Implied probability from best odds (what we'd bet at)
        features[f'{outcome}_implied_prob'] = 1 / max(odds_values)

        # Bookmaker disagreement (uncertainty signal)
        # Higher disagreement = more uncertainty = potential opportunity
        features[f'{outcome}_bookmaker_disagreement'] = max(probs) - min(probs)

        # Sharp vs soft book spread
        sharp_odds = [e['odds'] for e in entries if e['bookmaker_id'] in SHARP_BOOKMAKERS]
        soft_odds = [e['odds'] for e in entries if e['bookmaker_id'] in SOFT_BOOKMAKERS]

        if sharp_odds and soft_odds:
            # Positive = sharp books offering better odds = they think outcome more likely
            features[f'{outcome}_sharp_vs_soft'] = np.mean(sharp_odds) - np.mean(soft_odds)
        else:
            features[f'{outcome}_sharp_vs_soft'] = 0

    # Market overround (total implied probability)
    total_prob = sum([
        features.get('home_implied_prob', 0) or 0,
        features.get('draw_implied_prob', 0) or 0,
        features.get('away_implied_prob', 0) or 0,
    ])
    features['market_overround'] = total_prob

    # Normalized probabilities (remove margin)
    if total_prob > 0:
        features['market_home_prob_normalized'] = (features.get('home_implied_prob') or 0) / total_prob
        features['market_draw_prob_normalized'] = (features.get('draw_implied_prob') or 0) / total_prob
        features['market_away_prob_normalized'] = (features.get('away_implied_prob') or 0) / total_prob

    # Line movement (if we have timestamps)
    # Check first entry for timing info
    for outcome in ['home', 'draw', 'away']:
        entries = odds_1x2[outcome]
        if entries and entries[0].get('created_at') and entries[0].get('updated_at'):
            # We could calculate movement, but need opening odds which aren't always available
            pass

    return features


def process_fixture(fixture: Dict) -> Optional[Dict]:
    """Process a single fixture and extract all market features"""

    fixture_id = fixture.get('id')
    if not fixture_id:
        return None

    odds_list = fixture.get('odds', [])
    if not odds_list:
        return None

    # Extract basic info
    result = {
        'fixture_id': fixture_id,
        'league_id': fixture.get('league_id'),
        'season_id': fixture.get('season_id'),
        'match_date': fixture.get('starting_at'),
    }

    # Get scores for actual result
    scores = fixture.get('scores', [])
    home_score = away_score = None
    for s in scores:
        if s.get('description') == 'CURRENT':
            score_data = s.get('score', {})
            if score_data.get('participant') == 'home':
                home_score = score_data.get('goals', 0)
            elif score_data.get('participant') == 'away':
                away_score = score_data.get('goals', 0)

    result['home_score'] = home_score
    result['away_score'] = away_score

    if home_score is not None and away_score is not None:
        if home_score > away_score:
            result['result'] = 'H'
        elif home_score < away_score:
            result['result'] = 'A'
        else:
            result['result'] = 'D'
        result['total_goals'] = home_score + away_score

    # Extract 1X2 odds and calculate market features
    odds_1x2 = extract_1x2_odds(odds_list)
    market_features = calculate_market_features(odds_1x2)
    result.update(market_features)

    # Extract Asian Handicap
    ah = extract_asian_handicap(odds_list)
    result['ah_main_line'] = ah['main_line']
    result['ah_home_odds'] = ah['home_odds']
    result['ah_away_odds'] = ah['away_odds']

    # Extract Over/Under
    ou = extract_over_under(odds_list)
    result.update(ou)

    # Count bookmakers (liquidity indicator)
    result['num_bookmakers'] = len(set(o.get('bookmaker_id') for o in odds_list if o.get('bookmaker_id')))

    return result


def process_all_fixtures(data_dir: str, output_path: str, leagues: List[int] = None):
    """Process all fixture files and extract market features"""

    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('*.json'))

    print(f"Found {len(json_files)} fixture files")

    all_results = []

    for json_file in tqdm(json_files, desc="Processing files"):
        # Filter by league if specified
        if leagues:
            league_match = False
            for league_id in leagues:
                if f'league_{league_id}_' in json_file.name:
                    league_match = True
                    break
            if not league_match:
                continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Handle both list and dict structures
            if isinstance(data, dict) and 'data' in data:
                fixtures = data['data']
            elif isinstance(data, list):
                fixtures = data
            else:
                fixtures = [data]

            for fixture in fixtures:
                result = process_fixture(fixture)
                if result and result.get('home_best_odds'):  # Only include if we have odds
                    all_results.append(result)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by date
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    print(f"\nExtracted market features for {len(df):,} fixtures")
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df


def analyze_market_features(df: pd.DataFrame):
    """Analyze the predictive power of market features"""

    print("\n" + "=" * 60)
    print("Market Features Analysis")
    print("=" * 60)

    # Bookmaker disagreement analysis
    print("\n1. Bookmaker Disagreement vs Outcome")
    print("-" * 40)

    for outcome, result in [('home', 'H'), ('draw', 'D'), ('away', 'A')]:
        col = f'{outcome}_bookmaker_disagreement'
        if col not in df.columns:
            continue

        # Split into high/low disagreement
        median_disagreement = df[col].median()
        high_disagreement = df[df[col] > median_disagreement]
        low_disagreement = df[df[col] <= median_disagreement]

        high_win_rate = (high_disagreement['result'] == result).mean()
        low_win_rate = (low_disagreement['result'] == result).mean()

        print(f"{outcome.title()}:")
        print(f"  High disagreement (>{median_disagreement:.3f}): {high_win_rate:.1%} win rate")
        print(f"  Low disagreement:  {low_win_rate:.1%} win rate")

    # Sharp vs soft analysis
    print("\n2. Sharp vs Soft Book Spread")
    print("-" * 40)

    for outcome, result in [('home', 'H'), ('draw', 'D'), ('away', 'A')]:
        col = f'{outcome}_sharp_vs_soft'
        if col not in df.columns:
            continue

        # When sharps offer better odds (positive spread)
        sharp_favor = df[df[col] > 0.05]
        sharp_against = df[df[col] < -0.05]

        if len(sharp_favor) > 100:
            favor_win = (sharp_favor['result'] == result).mean()
            against_win = (sharp_against['result'] == result).mean() if len(sharp_against) > 100 else None

            print(f"{outcome.title()}:")
            print(f"  Sharp books favor (spread > 0.05): {favor_win:.1%} win rate ({len(sharp_favor)} samples)")
            if against_win:
                print(f"  Sharp books against (spread < -0.05): {against_win:.1%} win rate ({len(sharp_against)} samples)")

    # Asian Handicap line analysis
    print("\n3. Asian Handicap Lines")
    print("-" * 40)

    ah_df = df[df['ah_main_line'].notna()]
    if len(ah_df) > 0:
        print(f"Fixtures with AH data: {len(ah_df):,}")
        print(f"Main line distribution:")
        print(ah_df['ah_main_line'].value_counts().head(10))

    # Over/Under analysis
    print("\n4. Over/Under 2.5 Goals")
    print("-" * 40)

    ou_df = df[(df['over_2_5_best'].notna()) & (df['total_goals'].notna())]
    if len(ou_df) > 0:
        over_hit = (ou_df['total_goals'] > 2.5).mean()
        avg_over_odds = ou_df['over_2_5_avg'].mean()
        avg_under_odds = ou_df['under_2_5_avg'].mean()

        print(f"Fixtures with O/U data: {len(ou_df):,}")
        print(f"Over 2.5 hit rate: {over_hit:.1%}")
        print(f"Average Over 2.5 odds: {avg_over_odds:.2f}")
        print(f"Average Under 2.5 odds: {avg_under_odds:.2f}")

        # Value analysis
        over_ev = over_hit * avg_over_odds - 1
        under_ev = (1 - over_hit) * avg_under_odds - 1
        print(f"Blind Over 2.5 EV: {over_ev:.1%}")
        print(f"Blind Under 2.5 EV: {under_ev:.1%}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract market features from raw odds data')
    parser.add_argument('--data-dir', default='data/historical/fixtures',
                        help='Directory containing fixture JSON files')
    parser.add_argument('--output', default='data/market_features.csv',
                        help='Output CSV path')
    parser.add_argument('--leagues', type=int, nargs='+',
                        help='Filter to specific league IDs')
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis after extraction')

    args = parser.parse_args()

    df = process_all_fixtures(
        data_dir=args.data_dir,
        output_path=args.output,
        leagues=args.leagues
    )

    if args.analyze:
        analyze_market_features(df)
