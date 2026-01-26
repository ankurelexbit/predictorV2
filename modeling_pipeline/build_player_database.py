#!/usr/bin/env python3
"""
Build Player Statistics Database
=================================

Creates a comprehensive player statistics database by:
1. Reading historical lineups to identify all players
2. Aggregating player performance from team-level statistics
3. Querying SportMonks API for additional player details
4. Building player profiles with average statistics

This database is used for live predictions when lineups are announced.

Usage:
    python build_player_database.py --full          # Full rebuild
    python build_player_database.py --update        # Update recent players
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "sportmonks"
PROCESSED_DIR = DATA_DIR / "processed"
PLAYER_DB_DIR = PROCESSED_DIR / "player_database"
PLAYER_DB_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
API_KEY = "DQQStChRaPnjIryuZH2SxqJI5ufoA57wWsmFIuPCH2rvlBtm0G7Ch3mJoyE4"
BASE_URL = "https://api.sportmonks.com/v3/football"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("player_db_builder")

# Player stat type mapping (what we want to track)
PLAYER_STAT_TYPES = {
    # Performance
    188: 'rating',

    # Ball control
    120: 'touches',
    94: 'dispossessed',
    27273: 'possession_lost',

    # Passing
    81: 'accurate_passes',
    80: 'passes',
    116: 'accurate_passes_alt',
    117: 'key_passes',

    # Attacking
    52: 'goals',
    108: 'dribble_attempts',
    109: 'successful_dribbles',
    86: 'shots_on_target',
    42: 'shots_total',

    # Defending
    78: 'tackles',
    27267: 'tackles_won',
    100: 'interceptions',
    101: 'clearances',
    97: 'blocked_shots',

    # Duels
    105: 'total_duels',
    106: 'duels_won',
    1491: 'duels_lost',

    # Aerial
    107: 'aerials_won',
    27266: 'aerials_lost',
    27274: 'aerials_total',

    # Goalkeeper
    57: 'saves',
    104: 'saves_insidebox',
    1533: 'goals_conceded',

    # Other
    56: 'fouls',
    96: 'fouls_drawn',
    122: 'long_balls',
    123: 'long_balls_won',
}


class PlayerDatabaseBuilder:
    """Build and maintain player statistics database."""

    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
        self.player_stats = defaultdict(lambda: {
            'player_id': None,
            'player_name': None,
            'position_id': None,
            'team_id': None,
            'team_name': None,
            'matches_played': 0,
            'stats': defaultdict(list),  # List of values for averaging
            'last_updated': None
        })

    def load_historical_lineups(self) -> pd.DataFrame:
        """Load historical lineup data."""
        logger.info("Loading historical lineups...")
        lineups_file = RAW_DIR / "lineups.csv"

        if not lineups_file.exists():
            logger.error(f"Lineups file not found: {lineups_file}")
            return pd.DataFrame()

        df = pd.read_csv(lineups_file)
        logger.info(f"Loaded {len(df):,} lineup records")

        # Remove rows with missing player_id
        df = df.dropna(subset=['player_id'])
        df['player_id'] = df['player_id'].astype(int)

        logger.info(f"Found {df['player_id'].nunique():,} unique players")
        return df

    def load_historical_fixtures(self) -> pd.DataFrame:
        """Load historical fixture data with team-level aggregated player stats."""
        logger.info("Loading historical fixtures...")
        fixtures_file = RAW_DIR / "fixtures.csv"

        if not fixtures_file.exists():
            logger.error(f"Fixtures file not found: {fixtures_file}")
            return pd.DataFrame()

        df = pd.read_csv(fixtures_file, parse_dates=['date'])
        logger.info(f"Loaded {len(df):,} fixtures")
        return df

    def aggregate_player_stats_from_matches(
        self,
        lineups_df: pd.DataFrame,
        fixtures_df: pd.DataFrame
    ):
        """
        Aggregate player statistics from team-level stats.

        Strategy: Distribute team-level aggregated player stats among players
        who participated, weighted by position and minutes played (if available).
        """
        logger.info("Aggregating player statistics from match data...")

        # Get player stat columns from fixtures
        player_stat_cols = [col for col in fixtures_df.columns if 'player_stat' in col]

        if not player_stat_cols:
            logger.warning("No player stat columns found in fixtures")
            return

        # Join lineups with fixtures
        lineups_with_fixtures = lineups_df.merge(
            fixtures_df,
            on='fixture_id',
            how='inner'
        )

        logger.info(f"Processing {len(lineups_with_fixtures):,} lineup-fixture combinations")

        # Group by fixture to process each match
        for fixture_id, fixture_group in tqdm(lineups_with_fixtures.groupby('fixture_id'),
                                               desc="Processing fixtures"):
            home_team_id = fixture_group['home_team_id'].iloc[0]
            away_team_id = fixture_group['away_team_id'].iloc[0]
            match_date = fixture_group['date'].iloc[0]

            # Get home and away players
            home_players = fixture_group[fixture_group['team_id'] == home_team_id]
            away_players = fixture_group[fixture_group['team_id'] == away_team_id]

            # Process home team
            self._distribute_team_stats(
                home_players,
                fixture_group.iloc[0],
                'home',
                match_date
            )

            # Process away team
            self._distribute_team_stats(
                away_players,
                fixture_group.iloc[0],
                'away',
                match_date
            )

    def _distribute_team_stats(
        self,
        players: pd.DataFrame,
        fixture_row: pd.Series,
        side: str,
        match_date
    ):
        """
        Distribute team-level aggregated stats among players.

        Simple approach: Divide equally among players (starters get more weight).
        """
        if len(players) == 0:
            return

        # Count starters (type_id 11) and subs
        starters = players[players['type_id'] == 11]
        n_starters = len(starters)

        if n_starters == 0:
            return  # No starters, skip

        team_id = players['team_id'].iloc[0]
        team_name = fixture_row.get(f'{side}_team_name', 'Unknown')

        # Get team-level aggregated player stats for this side
        for col in fixture_row.index:
            if col.startswith(f'{side}_player_stat_'):
                # Extract stat type ID
                try:
                    stat_type_id = int(col.split('_')[-1])
                except ValueError:
                    continue

                if stat_type_id not in PLAYER_STAT_TYPES:
                    continue

                stat_name = PLAYER_STAT_TYPES[stat_type_id]
                team_total = fixture_row[col]

                if pd.isna(team_total) or team_total == 0:
                    continue

                # Distribute among starters (equal weight for simplicity)
                # In reality, this varies by player, but we don't have individual data
                per_player_avg = team_total / n_starters

                # Assign to each starter
                for _, player in starters.iterrows():
                    player_id = int(player['player_id'])
                    position_id = player.get('position_id')

                    # Initialize player if not exists
                    if self.player_stats[player_id]['player_id'] is None:
                        self.player_stats[player_id]['player_id'] = player_id
                        self.player_stats[player_id]['team_id'] = team_id
                        self.player_stats[player_id]['team_name'] = team_name
                        self.player_stats[player_id]['position_id'] = position_id

                    # Add stat value
                    self.player_stats[player_id]['stats'][stat_name].append(per_player_avg)

        # Increment match count for all starters
        for _, player in starters.iterrows():
            player_id = int(player['player_id'])
            self.player_stats[player_id]['matches_played'] += 1
            self.player_stats[player_id]['last_updated'] = datetime.now().isoformat()

    def fetch_player_details_from_api(self, player_ids: List[int], max_players: int = 1000):
        """
        Fetch player details from SportMonks API.

        This supplements our aggregate data with actual player names and details.
        """
        logger.info(f"Fetching player details from API for {min(len(player_ids), max_players)} players...")

        # Limit to avoid excessive API calls
        player_ids = player_ids[:max_players]

        for i, player_id in enumerate(tqdm(player_ids, desc="Fetching player details")):
            try:
                response = self.session.get(
                    f"{BASE_URL}/players/{player_id}",
                    params={"include": "position;nationality;teams"}
                )

                if response.status_code == 200:
                    data = response.json().get('data', {})

                    if self.player_stats[player_id]['player_id']:
                        # Update player details
                        self.player_stats[player_id]['player_name'] = data.get('display_name') or data.get('common_name')

                        # Position
                        position = data.get('position')
                        if position:
                            self.player_stats[player_id]['position_id'] = position.get('id')
                            self.player_stats[player_id]['position_name'] = position.get('name')

                # Rate limiting (3 requests per second = 180/min)
                if i % 3 == 0:
                    import time
                    time.sleep(1)

            except Exception as e:
                logger.warning(f"Error fetching player {player_id}: {e}")
                continue

    def calculate_player_averages(self):
        """Calculate average statistics for each player."""
        logger.info("Calculating player averages...")

        for player_id, player_data in self.player_stats.items():
            matches_played = player_data['matches_played']

            if matches_played == 0:
                continue

            # Calculate averages for each stat
            avg_stats = {}
            for stat_name, values in player_data['stats'].items():
                if len(values) > 0:
                    avg_stats[stat_name] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'samples': len(values)
                    }

            player_data['avg_stats'] = avg_stats

    def save_player_database(self, output_file: str = "player_stats_db.json"):
        """Save player database to JSON file."""
        output_path = PLAYER_DB_DIR / output_file

        logger.info(f"Saving player database to {output_path}...")

        def convert_to_native_types(obj):
            """Convert numpy/pandas types to native Python types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif pd.isna(obj):
                return None
            return obj

        # Convert defaultdict to regular dict for JSON serialization
        player_db = {}
        for player_id, data in self.player_stats.items():
            if data['matches_played'] > 0:  # Only save players with data
                player_db[str(player_id)] = {
                    'player_id': convert_to_native_types(data['player_id']),
                    'player_name': data['player_name'],
                    'position_id': convert_to_native_types(data['position_id']),
                    'position_name': data.get('position_name'),
                    'team_id': convert_to_native_types(data['team_id']),
                    'team_name': data['team_name'],
                    'matches_played': convert_to_native_types(data['matches_played']),
                    'avg_stats': convert_to_native_types(data.get('avg_stats', {})),
                    'last_updated': data['last_updated']
                }

        with open(output_path, 'w') as f:
            json.dump(player_db, f, indent=2)

        logger.info(f"✅ Saved {len(player_db):,} player profiles")

        # Also save summary statistics
        self._save_summary_stats(player_db)

        return output_path

    def _save_summary_stats(self, player_db: Dict):
        """Save summary statistics about the database."""
        summary_file = PLAYER_DB_DIR / "player_db_summary.txt"

        # Calculate summary stats
        total_players = len(player_db)
        players_with_names = sum(1 for p in player_db.values() if p['player_name'])

        # Group by position
        positions = defaultdict(int)
        for player in player_db.values():
            pos_name = player.get('position_name', 'Unknown')
            positions[pos_name] += 1

        # Stats coverage
        stat_coverage = defaultdict(int)
        for player in player_db.values():
            for stat_name in player.get('avg_stats', {}).keys():
                stat_coverage[stat_name] += 1

        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PLAYER DATABASE SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total Players: {total_players:,}\n")
            f.write(f"Players with Names: {players_with_names:,}\n")
            f.write(f"Built: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Players by Position:\n")
            for pos, count in sorted(positions.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {pos}: {count:,}\n")

            f.write("\nStatistic Coverage (# players with stat):\n")
            for stat, count in sorted(stat_coverage.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_players * 100
                f.write(f"  {stat}: {count:,} ({pct:.1f}%)\n")

        logger.info(f"Summary saved to {summary_file}")


def main():
    """Build player statistics database."""
    parser = argparse.ArgumentParser(description="Build player statistics database")
    parser.add_argument('--full', action='store_true', help='Full rebuild')
    parser.add_argument('--update', action='store_true', help='Update recent players')
    parser.add_argument('--fetch-api', action='store_true',
                       help='Fetch player details from API (requires API credits)')
    parser.add_argument('--max-api-players', type=int, default=1000,
                       help='Maximum players to fetch from API (default: 1000)')
    args = parser.parse_args()

    print("="*80)
    print("BUILDING PLAYER STATISTICS DATABASE")
    print("="*80)
    print()

    builder = PlayerDatabaseBuilder()

    # Load data
    lineups_df = builder.load_historical_lineups()
    fixtures_df = builder.load_historical_fixtures()

    if lineups_df.empty or fixtures_df.empty:
        logger.error("Cannot build database - missing source data")
        return

    # Aggregate player statistics
    builder.aggregate_player_stats_from_matches(lineups_df, fixtures_df)

    # Optionally fetch player details from API
    if args.fetch_api:
        player_ids = list(builder.player_stats.keys())
        player_ids = [pid for pid in player_ids if builder.player_stats[pid]['matches_played'] > 0]

        # Sort by matches played (prioritize active players)
        player_ids.sort(
            key=lambda pid: builder.player_stats[pid]['matches_played'],
            reverse=True
        )

        builder.fetch_player_details_from_api(
            player_ids,
            max_players=args.max_api_players
        )
    else:
        logger.info("Skipping API fetch (use --fetch-api to enable)")

    # Calculate averages
    builder.calculate_player_averages()

    # Save database
    output_file = builder.save_player_database()

    print()
    print("="*80)
    print("✅ PLAYER DATABASE BUILD COMPLETE")
    print("="*80)
    print(f"Output: {output_file}")
    print()
    print("Next steps:")
    print("  1. Review: cat data/processed/player_database/player_db_summary.txt")
    print("  2. Use in predictions: predict_live.py will automatically use this database")
    print()


if __name__ == '__main__':
    main()
