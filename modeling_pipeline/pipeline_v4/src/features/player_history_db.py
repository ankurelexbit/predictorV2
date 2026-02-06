"""
Player Historical Database

Builds and maintains a database of player performance metrics from historical matches.
This allows point-in-time correct lookups of player quality at any date.

Usage:
    db = PlayerHistoryDB()
    db.build_from_fixtures('data/historical/fixtures')

    # Get player's historical average rating as of a specific date
    rating = db.get_player_rating(player_id=1234, as_of_date='2025-01-15')

    # Get team's expected lineup quality
    quality = db.get_lineup_quality(player_ids=[1,2,3,...], as_of_date='2025-01-15')
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm


# Detail type IDs from SportMonks
RATING_TYPE_ID = 118
MINUTES_TYPE_ID = 119
GOALS_TYPE_ID = 52  # or similar
ASSISTS_TYPE_ID = 79


class PlayerHistoryDB:
    """
    Point-in-time correct player performance database.

    Stores historical match-by-match performance data and allows
    lookups of player quality metrics as of any date.
    """

    def __init__(self):
        # player_id -> list of (date, rating, minutes, team_id, is_starter)
        self.player_matches: Dict[int, List[Dict]] = defaultdict(list)

        # team_id -> list of (date, lineup_player_ids) for last lineup lookup
        self.team_lineups: Dict[int, List[Dict]] = defaultdict(list)

        # player_id -> player_name (for debugging)
        self.player_names: Dict[int, str] = {}

        # Cache for computed averages
        self._rating_cache: Dict[Tuple[int, str], float] = {}

        self.is_built = False

    def build_from_fixtures(self, fixtures_dir: str, leagues: List[int] = None):
        """
        Build player database from historical fixture JSON files.

        Args:
            fixtures_dir: Path to directory containing fixture JSON files
            leagues: Optional list of league IDs to filter
        """
        data_path = Path(fixtures_dir)
        json_files = sorted(data_path.glob('*.json'))

        print(f"Building player history database from {len(json_files)} files...")

        total_records = 0

        for json_file in tqdm(json_files, desc="Processing fixtures"):
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
                    records = self._process_fixture(fixture)
                    total_records += records

            except Exception as e:
                continue

        # Sort all player histories by date
        for player_id in self.player_matches:
            self.player_matches[player_id].sort(key=lambda x: x['date'])

        for team_id in self.team_lineups:
            self.team_lineups[team_id].sort(key=lambda x: x['date'])

        self.is_built = True

        print(f"\nDatabase built:")
        print(f"  Total player-match records: {total_records:,}")
        print(f"  Unique players: {len(self.player_matches):,}")
        print(f"  Teams with lineup history: {len(self.team_lineups):,}")

    def _process_fixture(self, fixture: Dict) -> int:
        """Process a single fixture and extract player data."""
        match_date = fixture.get('starting_at', '')
        if not match_date:
            return 0

        lineups = fixture.get('lineups', [])
        if not lineups:
            return 0

        records = 0
        team_starters: Dict[int, List[int]] = defaultdict(list)

        for player in lineups:
            player_id = player.get('player_id')
            team_id = player.get('team_id')
            player_name = player.get('player_name', '')
            type_id = player.get('type_id')  # 11=bench, 12=starter

            if not player_id or not team_id:
                continue

            # Store player name
            if player_name and player_id not in self.player_names:
                self.player_names[player_id] = player_name

            # Extract rating and minutes from details
            rating = None
            minutes = 0

            for detail in player.get('details', []):
                detail_type = detail.get('type_id')
                value = detail.get('data', {}).get('value')

                if detail_type == RATING_TYPE_ID and value and value > 1:
                    rating = float(value)
                elif detail_type == MINUTES_TYPE_ID and value:
                    minutes = int(value)

            # Only store if we have a valid rating
            if rating:
                is_starter = type_id == 12

                self.player_matches[player_id].append({
                    'date': match_date,
                    'rating': rating,
                    'minutes': minutes,
                    'team_id': team_id,
                    'is_starter': is_starter
                })
                records += 1

                # Track starters for team lineup history
                if is_starter:
                    team_starters[team_id].append(player_id)

        # Store team lineups
        for team_id, starter_ids in team_starters.items():
            if len(starter_ids) >= 7:  # At least 7 players identified as starters
                self.team_lineups[team_id].append({
                    'date': match_date,
                    'player_ids': starter_ids
                })

        return records

    def get_player_rating(
        self,
        player_id: int,
        as_of_date: str,
        n_matches: int = 10,
        min_matches: int = 3
    ) -> Optional[float]:
        """
        Get player's historical average rating as of a specific date.

        Args:
            player_id: The player's ID
            as_of_date: Date string (YYYY-MM-DD or full datetime)
            n_matches: Number of recent matches to average
            min_matches: Minimum matches required for valid rating

        Returns:
            Average rating or None if insufficient data
        """
        if player_id not in self.player_matches:
            return None

        # Parse as_of_date
        if isinstance(as_of_date, str):
            as_of = as_of_date[:10]  # Just the date part
        else:
            as_of = str(as_of_date)[:10]

        # Get matches before this date
        history = self.player_matches[player_id]
        prior_matches = [m for m in history if m['date'][:10] < as_of]

        if len(prior_matches) < min_matches:
            return None

        # Get most recent n_matches
        recent = prior_matches[-n_matches:]
        ratings = [m['rating'] for m in recent]

        return np.mean(ratings)

    def get_player_form(
        self,
        player_id: int,
        as_of_date: str,
        n_matches: int = 5
    ) -> Optional[Dict]:
        """
        Get player's recent form metrics.

        Returns dict with:
        - avg_rating: Average rating over last n matches
        - rating_trend: Change in rating (recent vs older)
        - matches_played: Number of matches in period
        - avg_minutes: Average minutes played
        """
        if player_id not in self.player_matches:
            return None

        as_of = as_of_date[:10] if isinstance(as_of_date, str) else str(as_of_date)[:10]

        history = self.player_matches[player_id]
        prior_matches = [m for m in history if m['date'][:10] < as_of]

        if len(prior_matches) < 3:
            return None

        recent = prior_matches[-n_matches:]
        older = prior_matches[-(2*n_matches):-n_matches] if len(prior_matches) >= 2*n_matches else []

        recent_ratings = [m['rating'] for m in recent]
        recent_minutes = [m['minutes'] for m in recent]

        result = {
            'avg_rating': np.mean(recent_ratings),
            'matches_played': len(recent),
            'avg_minutes': np.mean(recent_minutes) if recent_minutes else 0,
            'rating_std': np.std(recent_ratings) if len(recent_ratings) > 1 else 0,
        }

        # Calculate trend
        if older:
            older_ratings = [m['rating'] for m in older]
            result['rating_trend'] = np.mean(recent_ratings) - np.mean(older_ratings)
        else:
            result['rating_trend'] = 0

        return result

    def get_team_last_lineup(
        self,
        team_id: int,
        as_of_date: str
    ) -> Optional[List[int]]:
        """
        Get team's most recent lineup (player IDs) before a date.
        Used as placeholder when actual lineup not announced.
        """
        if team_id not in self.team_lineups:
            return None

        as_of = as_of_date[:10] if isinstance(as_of_date, str) else str(as_of_date)[:10]

        lineups = self.team_lineups[team_id]
        prior = [l for l in lineups if l['date'][:10] < as_of]

        if not prior:
            return None

        return prior[-1]['player_ids']

    def get_lineup_quality(
        self,
        player_ids: List[int],
        as_of_date: str,
        n_matches: int = 10
    ) -> Dict:
        """
        Calculate lineup quality metrics for a set of players.

        Returns:
        - avg_rating: Average historical rating of all players
        - total_rating: Sum of ratings
        - min_rating: Weakest link
        - max_rating: Best player
        - rated_players: Number of players with rating history
        - coverage: Fraction of players with ratings
        """
        ratings = []

        for player_id in player_ids:
            rating = self.get_player_rating(player_id, as_of_date, n_matches)
            if rating:
                ratings.append(rating)

        if not ratings:
            return {
                'avg_rating': None,
                'total_rating': None,
                'min_rating': None,
                'max_rating': None,
                'rated_players': 0,
                'coverage': 0.0
            }

        return {
            'avg_rating': np.mean(ratings),
            'total_rating': sum(ratings),
            'min_rating': min(ratings),
            'max_rating': max(ratings),
            'rated_players': len(ratings),
            'coverage': len(ratings) / len(player_ids) if player_ids else 0
        }

    def save(self, filepath: str):
        """Save database to disk."""
        data = {
            'player_matches': dict(self.player_matches),
            'team_lineups': dict(self.team_lineups),
            'player_names': self.player_names,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Database saved to {filepath}")

    def load(self, filepath: str):
        """Load database from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.player_matches = defaultdict(list, data['player_matches'])
        self.team_lineups = defaultdict(list, data['team_lineups'])
        self.player_names = data['player_names']
        self.is_built = True

        print(f"Database loaded: {len(self.player_matches):,} players")

    def get_stats(self) -> Dict:
        """Get database statistics."""
        total_records = sum(len(v) for v in self.player_matches.values())

        # Get date range
        all_dates = []
        for matches in self.player_matches.values():
            all_dates.extend(m['date'] for m in matches)

        return {
            'total_records': total_records,
            'unique_players': len(self.player_matches),
            'teams_with_lineups': len(self.team_lineups),
            'date_range': (min(all_dates)[:10], max(all_dates)[:10]) if all_dates else (None, None)
        }


def build_and_save_database(
    fixtures_dir: str = 'data/historical/fixtures',
    output_path: str = 'data/player_history_db.pkl',
    leagues: List[int] = None
):
    """Build and save player history database."""
    db = PlayerHistoryDB()
    db.build_from_fixtures(fixtures_dir, leagues)
    db.save(output_path)

    stats = db.get_stats()
    print(f"\nDatabase statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return db


if __name__ == '__main__':
    # Build database for top 5 leagues
    db = build_and_save_database(
        fixtures_dir='data/historical/fixtures',
        output_path='data/player_history_db.pkl',
        leagues=[8, 301, 384, 82, 564]
    )

    # Test lookups
    print("\n" + "="*60)
    print("Testing database lookups")
    print("="*60)

    # Get a sample player
    sample_player = list(db.player_matches.keys())[100]
    player_name = db.player_names.get(sample_player, 'Unknown')

    print(f"\nSample player: {player_name} (ID: {sample_player})")

    # Get their rating as of different dates
    for date in ['2024-01-01', '2024-06-01', '2025-01-01']:
        rating = db.get_player_rating(sample_player, date)
        form = db.get_player_form(sample_player, date)
        print(f"  As of {date}: rating={rating:.2f if rating else 'N/A'}, form={form}")
