"""
Point-in-Time Correct Lineup Features

Uses the player historical database to generate lineup quality features
that are available at prediction time (1hr before kickoff).

Two modes:
1. Lineup announced: Use actual starting XI with historical player ratings
2. Lineup not announced: Use previous match's lineup as placeholder

This is POINT-IN-TIME CORRECT because:
- We only use player ratings from BEFORE the current match
- We never use post-match data for the current match
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .player_history_db import PlayerHistoryDB


class LineupFeatureGenerator:
    """
    Generate point-in-time correct lineup features.
    """

    def __init__(self, player_db: PlayerHistoryDB):
        """
        Args:
            player_db: Pre-built player history database
        """
        self.player_db = player_db

    def get_lineup_from_fixture(
        self,
        fixture: Dict,
        team_id: int
    ) -> Tuple[List[int], bool]:
        """
        Extract starting lineup from fixture data.

        Returns:
            Tuple of (player_ids, is_actual_lineup)
            - is_actual_lineup: True if from announced lineup, False if placeholder
        """
        lineups = fixture.get('lineups', [])

        # Get starters (type_id=12) for this team
        starters = [
            p.get('player_id')
            for p in lineups
            if p.get('team_id') == team_id and p.get('type_id') == 12
        ]

        # Filter out None values
        starters = [p for p in starters if p]

        if len(starters) >= 7:
            # We have an actual lineup
            return starters, True

        # No lineup announced - use previous match's lineup
        match_date = fixture.get('starting_at', '')
        prev_lineup = self.player_db.get_team_last_lineup(team_id, match_date)

        if prev_lineup:
            return prev_lineup, False

        return [], False

    def generate_team_features(
        self,
        player_ids: List[int],
        as_of_date: str,
        prefix: str,
        is_actual_lineup: bool
    ) -> Dict:
        """
        Generate features for a team's lineup.

        Args:
            player_ids: List of player IDs in starting XI
            as_of_date: Match date for point-in-time lookup
            prefix: 'home' or 'away'
            is_actual_lineup: Whether this is the announced lineup
        """
        features = {}

        # Get lineup quality
        quality = self.player_db.get_lineup_quality(player_ids, as_of_date)

        features[f'{prefix}_lineup_avg_rating'] = quality['avg_rating']
        features[f'{prefix}_lineup_total_rating'] = quality['total_rating']
        features[f'{prefix}_lineup_min_rating'] = quality['min_rating']
        features[f'{prefix}_lineup_max_rating'] = quality['max_rating']
        features[f'{prefix}_lineup_rated_players'] = quality['rated_players']
        features[f'{prefix}_lineup_coverage'] = quality['coverage']
        features[f'{prefix}_lineup_is_actual'] = 1 if is_actual_lineup else 0

        # Get individual player forms
        forms = []
        trends = []

        for player_id in player_ids:
            form = self.player_db.get_player_form(player_id, as_of_date)
            if form:
                forms.append(form['avg_rating'])
                trends.append(form['rating_trend'])

        if forms:
            features[f'{prefix}_lineup_form_avg'] = np.mean(forms)
            features[f'{prefix}_lineup_form_std'] = np.std(forms) if len(forms) > 1 else 0
            features[f'{prefix}_lineup_trend_avg'] = np.mean(trends)
            features[f'{prefix}_lineup_hot_players'] = sum(1 for t in trends if t > 0.2)
            features[f'{prefix}_lineup_cold_players'] = sum(1 for t in trends if t < -0.2)
        else:
            features[f'{prefix}_lineup_form_avg'] = None
            features[f'{prefix}_lineup_form_std'] = None
            features[f'{prefix}_lineup_trend_avg'] = None
            features[f'{prefix}_lineup_hot_players'] = 0
            features[f'{prefix}_lineup_cold_players'] = 0

        return features

    def generate_fixture_features(self, fixture: Dict) -> Optional[Dict]:
        """
        Generate all lineup features for a fixture.

        Returns None if insufficient data.
        """
        fixture_id = fixture.get('id')
        match_date = fixture.get('starting_at', '')

        if not fixture_id or not match_date:
            return None

        # Get team IDs
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

        # Get lineups
        home_lineup, home_actual = self.get_lineup_from_fixture(fixture, home_team_id)
        away_lineup, away_actual = self.get_lineup_from_fixture(fixture, away_team_id)

        if not home_lineup and not away_lineup:
            return None

        # Generate features
        result = {
            'fixture_id': fixture_id,
            'match_date': match_date,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
        }

        # Home features
        if home_lineup:
            home_features = self.generate_team_features(
                home_lineup, match_date, 'home', home_actual
            )
            result.update(home_features)

        # Away features
        if away_lineup:
            away_features = self.generate_team_features(
                away_lineup, match_date, 'away', away_actual
            )
            result.update(away_features)

        # Differential features
        if result.get('home_lineup_avg_rating') and result.get('away_lineup_avg_rating'):
            result['lineup_rating_diff'] = (
                result['home_lineup_avg_rating'] - result['away_lineup_avg_rating']
            )
            result['lineup_total_diff'] = (
                (result.get('home_lineup_total_rating') or 0) -
                (result.get('away_lineup_total_rating') or 0)
            )
            result['lineup_form_diff'] = (
                (result.get('home_lineup_form_avg') or 0) -
                (result.get('away_lineup_form_avg') or 0)
            )
        else:
            result['lineup_rating_diff'] = None
            result['lineup_total_diff'] = None
            result['lineup_form_diff'] = None

        return result


def generate_lineup_features_for_dataset(
    fixtures_dir: str,
    player_db_path: str,
    output_path: str,
    leagues: List[int] = None
) -> pd.DataFrame:
    """
    Generate lineup features for all fixtures.

    Args:
        fixtures_dir: Path to fixture JSON files
        player_db_path: Path to player history database
        output_path: Where to save the features CSV
        leagues: Optional league filter
    """
    # Load player database
    print("Loading player history database...")
    player_db = PlayerHistoryDB()
    player_db.load(player_db_path)

    generator = LineupFeatureGenerator(player_db)

    # Process fixtures
    data_path = Path(fixtures_dir)
    json_files = sorted(data_path.glob('*.json'))

    print(f"Processing {len(json_files)} fixture files...")

    all_results = []
    fixtures_with_features = 0
    fixtures_without_features = 0

    for json_file in tqdm(json_files, desc="Generating features"):
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
                result = generator.generate_fixture_features(fixture)
                if result and result.get('home_lineup_avg_rating'):
                    all_results.append(result)
                    fixtures_with_features += 1
                else:
                    fixtures_without_features += 1

        except Exception as e:
            continue

    # Create DataFrame
    df = pd.DataFrame(all_results)

    if len(df) > 0:
        df['match_date'] = pd.to_datetime(df['match_date'])
        df = df.sort_values('match_date').reset_index(drop=True)

        print(f"\nGenerated features for {len(df):,} fixtures")
        print(f"Fixtures without sufficient data: {fixtures_without_features:,}")
        print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")

        # Coverage stats
        actual_lineup_pct = (
            df['home_lineup_is_actual'].mean() + df['away_lineup_is_actual'].mean()
        ) / 2 * 100
        print(f"Actual lineup coverage: {actual_lineup_pct:.1f}%")

        # Save
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return df


if __name__ == '__main__':
    # First ensure player database exists
    import os

    db_path = 'data/player_history_db.pkl'

    if not os.path.exists(db_path):
        print("Building player database first...")
        from player_history_db import build_and_save_database
        build_and_save_database(
            fixtures_dir='data/historical/fixtures',
            output_path=db_path,
            leagues=[8, 301, 384, 82, 564]
        )

    # Generate lineup features
    df = generate_lineup_features_for_dataset(
        fixtures_dir='data/historical/fixtures',
        player_db_path=db_path,
        output_path='data/lineup_features_v2.csv',
        leagues=[8, 301, 384, 82, 564]
    )

    # Analysis
    print("\n" + "="*60)
    print("Feature Statistics")
    print("="*60)

    for col in df.columns:
        if col not in ['fixture_id', 'match_date', 'home_team_id', 'away_team_id']:
            valid = df[col].notna().sum()
            if valid > 0 and df[col].dtype in ['float64', 'int64']:
                mean = df[col].mean()
                std = df[col].std()
                print(f"  {col}: mean={mean:.3f}, std={std:.3f}")
