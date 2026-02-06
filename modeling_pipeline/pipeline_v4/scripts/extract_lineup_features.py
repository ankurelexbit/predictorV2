"""
Extract Lineup Features Available at 1 Hour Before Kickoff

Lineups are typically announced 60-90 minutes before kickoff.
This script extracts features from:
- Starting XI ratings and quality
- Formation information
- Sidelined (injured/suspended) players
- Position-specific metrics

These features complement the 1hr market data to improve predictions.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Position mapping
POSITION_MAP = {
    24: 'goalkeeper',
    25: 'defender',
    26: 'midfielder',
    27: 'forward'
}

# Detail type IDs
RATING_TYPE_ID = 118
MINUTES_TYPE_ID = 119


def extract_starting_xi(lineups: List[Dict], team_id: int) -> List[Dict]:
    """Extract starting XI players for a team (type_id=12 for starters)"""
    return [
        p for p in lineups
        if p.get('team_id') == team_id and p.get('type_id') == 12
    ]


def get_player_rating(player: Dict) -> Optional[float]:
    """Extract player rating from details"""
    for detail in player.get('details', []):
        if detail.get('type_id') == RATING_TYPE_ID:
            value = detail.get('data', {}).get('value')
            if value and value > 1:  # Valid rating range is ~5.6-9.0
                return float(value)
    return None


def calculate_lineup_quality_features(starters: List[Dict]) -> Dict:
    """Calculate quality metrics from starting XI"""
    features = {}

    # Get all valid ratings
    ratings = []
    for player in starters:
        rating = get_player_rating(player)
        if rating:
            ratings.append(rating)

    if ratings:
        features['avg_starter_rating'] = np.mean(ratings)
        features['total_starter_rating'] = sum(ratings)
        features['min_starter_rating'] = min(ratings)
        features['max_starter_rating'] = max(ratings)
        features['rating_std'] = np.std(ratings) if len(ratings) > 1 else 0
        features['num_rated_starters'] = len(ratings)
    else:
        features['avg_starter_rating'] = None
        features['total_starter_rating'] = None
        features['min_starter_rating'] = None
        features['max_starter_rating'] = None
        features['rating_std'] = None
        features['num_rated_starters'] = 0

    return features


def calculate_position_ratings(starters: List[Dict]) -> Dict:
    """Calculate average ratings by position"""
    position_ratings = defaultdict(list)

    for player in starters:
        pos_id = player.get('position_id')
        position = POSITION_MAP.get(pos_id, 'unknown')
        rating = get_player_rating(player)
        if rating:
            position_ratings[position].append(rating)

    features = {}
    for position in ['goalkeeper', 'defender', 'midfielder', 'forward']:
        ratings = position_ratings.get(position, [])
        if ratings:
            features[f'{position}_avg_rating'] = np.mean(ratings)
            features[f'num_{position}s'] = len(ratings)
        else:
            features[f'{position}_avg_rating'] = None
            features[f'num_{position}s'] = 0

    return features


def parse_formation(formation_str: str) -> Dict:
    """Parse formation string like '4-3-3' into features"""
    if not formation_str or not isinstance(formation_str, str):
        return {'num_defenders': None, 'num_midfielders': None, 'num_forwards': None}

    parts = formation_str.split('-')
    try:
        parts = [int(p) for p in parts]

        # Standard interpretation: DEF-MID-FWD or DEF-MID1-MID2-FWD
        if len(parts) == 3:
            return {
                'num_defenders': parts[0],
                'num_midfielders': parts[1],
                'num_forwards': parts[2]
            }
        elif len(parts) == 4:
            return {
                'num_defenders': parts[0],
                'num_midfielders': parts[1] + parts[2],
                'num_forwards': parts[3]
            }
        elif len(parts) == 5:
            return {
                'num_defenders': parts[0],
                'num_midfielders': parts[1] + parts[2] + parts[3],
                'num_forwards': parts[4]
            }
    except:
        pass

    return {'num_defenders': None, 'num_midfielders': None, 'num_forwards': None}


def extract_formation_features(formations: List[Dict], team_id: int) -> Dict:
    """Extract formation features for a team"""
    features = {
        'formation': None,
        'num_defenders': None,
        'num_midfielders': None,
        'num_forwards': None,
        'is_defensive_formation': None
    }

    for f in formations:
        if f.get('participant_id') == team_id:
            formation_str = f.get('formation', '')
            features['formation'] = formation_str

            parsed = parse_formation(formation_str)
            features.update(parsed)

            # Defensive = 5+ defenders or 2+ defensive mids (5-x-x or 4-2-x-x)
            if parsed['num_defenders'] is not None:
                features['is_defensive_formation'] = 1 if parsed['num_defenders'] >= 5 else 0

            break

    return features


def extract_sidelined_features(sidelined: List[Dict], team_id: int) -> Dict:
    """Extract injury/suspension features for a team"""
    team_sidelined = [s for s in sidelined if s.get('participant_id') == team_id]

    return {
        'num_sidelined': len(team_sidelined),
        'has_sidelined': 1 if team_sidelined else 0
    }


def extract_team_lineup_features(
    fixture: Dict,
    team_id: int,
    prefix: str
) -> Dict:
    """Extract all lineup features for one team"""
    lineups = fixture.get('lineups', [])
    formations = fixture.get('formations', [])
    sidelined = fixture.get('sidelined', [])

    features = {}

    # Starting XI
    starters = extract_starting_xi(lineups, team_id)
    features['num_starters'] = len(starters)

    # Quality metrics
    quality = calculate_lineup_quality_features(starters)
    features.update(quality)

    # Position ratings
    positions = calculate_position_ratings(starters)
    features.update(positions)

    # Formation
    formation = extract_formation_features(formations, team_id)
    features.update(formation)

    # Sidelined
    sidelined_feats = extract_sidelined_features(sidelined, team_id)
    features.update(sidelined_feats)

    # Add prefix
    return {f'{prefix}_{k}': v for k, v in features.items()}


def extract_fixture_lineup_features(fixture: Dict) -> Optional[Dict]:
    """Extract lineup features for a fixture"""
    fixture_id = fixture.get('id')
    if not fixture_id:
        return None

    # Get team IDs from participants
    participants = fixture.get('participants', [])
    home_team_id = None
    away_team_id = None

    for p in participants:
        meta = p.get('meta', {})
        if meta.get('location') == 'home':
            home_team_id = p.get('id')
        elif meta.get('location') == 'away':
            away_team_id = p.get('id')

    if not home_team_id or not away_team_id:
        return None

    # Check if lineups exist
    lineups = fixture.get('lineups', [])
    if not lineups:
        return None

    # Extract features for both teams
    home_features = extract_team_lineup_features(fixture, home_team_id, 'home')
    away_features = extract_team_lineup_features(fixture, away_team_id, 'away')

    # Combine
    result = {
        'fixture_id': fixture_id,
        'match_date': fixture.get('starting_at', ''),
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
    }
    result.update(home_features)
    result.update(away_features)

    # Calculate differential features
    if home_features.get('home_avg_starter_rating') and away_features.get('away_avg_starter_rating'):
        result['rating_diff'] = (
            home_features['home_avg_starter_rating'] -
            away_features['away_avg_starter_rating']
        )
        result['total_rating_diff'] = (
            (home_features.get('home_total_starter_rating') or 0) -
            (away_features.get('away_total_starter_rating') or 0)
        )
    else:
        result['rating_diff'] = None
        result['total_rating_diff'] = None

    # Sidelined differential
    result['sidelined_diff'] = (
        (away_features.get('away_num_sidelined') or 0) -
        (home_features.get('home_num_sidelined') or 0)
    )

    return result


def process_all_fixtures(data_dir: str, output_path: str, leagues: List[int] = None):
    """Process all fixtures and extract lineup features"""
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('*.json'))

    print(f"Processing {len(json_files)} fixture files")
    print("Extracting lineup features...")

    all_results = []
    fixtures_with_lineups = 0
    fixtures_without_lineups = 0

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
                result = extract_fixture_lineup_features(fixture)
                if result and result.get('home_num_starters', 0) > 0:
                    all_results.append(result)
                    fixtures_with_lineups += 1
                else:
                    fixtures_without_lineups += 1

        except Exception as e:
            continue

    # Create DataFrame
    df = pd.DataFrame(all_results)

    if len(df) > 0:
        df['match_date'] = pd.to_datetime(df['match_date'])
        df = df.sort_values('match_date').reset_index(drop=True)

        print(f"\nExtracted {len(df):,} fixtures with lineup data")
        print(f"Fixtures without lineups: {fixtures_without_lineups:,}")
        print(f"Coverage: {len(df) / (len(df) + fixtures_without_lineups) * 100:.1f}%")
        print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")

        # Save
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    else:
        print("No lineup data found!")

    return df


def analyze_lineup_features(df: pd.DataFrame):
    """Analyze extracted lineup features"""
    print("\n" + "=" * 60)
    print("Lineup Feature Statistics")
    print("=" * 60)

    # Rating statistics
    print("\n--- Player Ratings ---")
    for col in ['home_avg_starter_rating', 'away_avg_starter_rating', 'rating_diff']:
        if col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0:
                mean = df[col].mean()
                std = df[col].std()
                print(f"  {col}: {valid:,} valid, mean={mean:.2f}, std={std:.2f}")

    # Position ratings
    print("\n--- Position-Specific Ratings ---")
    for pos in ['goalkeeper', 'defender', 'midfielder', 'forward']:
        col = f'home_{pos}_avg_rating'
        if col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0:
                mean = df[col].mean()
                print(f"  {pos}: {valid:,} valid, mean={mean:.2f}")

    # Formation distribution
    print("\n--- Formations ---")
    if 'home_formation' in df.columns:
        formations = df['home_formation'].value_counts().head(10)
        for form, count in formations.items():
            print(f"  {form}: {count:,} ({count/len(df)*100:.1f}%)")

    # Sidelined
    print("\n--- Sidelined Players ---")
    if 'home_num_sidelined' in df.columns:
        avg_home = df['home_num_sidelined'].mean()
        avg_away = df['away_num_sidelined'].mean()
        print(f"  Avg home sidelined: {avg_home:.2f}")
        print(f"  Avg away sidelined: {avg_away:.2f}")


if __name__ == '__main__':
    # Extract lineup features
    df = process_all_fixtures(
        data_dir='data/historical/fixtures',
        output_path='data/lineup_features.csv',
        leagues=[8, 301, 384, 82, 564]  # Top 5 leagues
    )

    if len(df) > 0:
        analyze_lineup_features(df)

        # Show sample
        print("\n--- Sample Data ---")
        print(df[['fixture_id', 'match_date', 'home_avg_starter_rating',
                  'away_avg_starter_rating', 'rating_diff', 'home_formation']].head(10))
