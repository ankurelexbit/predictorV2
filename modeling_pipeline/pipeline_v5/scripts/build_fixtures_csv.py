#!/usr/bin/env python3
"""
Build fixtures CSV from JSON files.

Processes JSON files one at a time to avoid memory issues.
Creates a lightweight CSV that can be loaded instantly.

Usage:
    python3 scripts/build_fixtures_csv.py
"""
import ijson
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_fixture_data(fixture: Dict) -> Optional[Dict]:
    """Extract essential fields from a fixture."""
    try:
        # Get participants
        participants = fixture.get('participants', [])
        if len(participants) != 2:
            return None

        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

        if not home_team or not away_team:
            return None

        # Get scores
        scores = fixture.get('scores', [])
        home_score = None
        away_score = None

        for score in scores:
            if score.get('description') == 'CURRENT':
                participant = score.get('score', {}).get('participant')
                goals = score.get('score', {}).get('goals')

                if participant == 'home':
                    home_score = goals
                elif participant == 'away':
                    away_score = goals

        # Determine result
        result = None
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                result = 'H'
            elif home_score < away_score:
                result = 'A'
            else:
                result = 'D'

        # Extract statistics summary
        statistics = fixture.get('statistics', [])
        stats_dict = extract_statistics(statistics, home_team.get('id'), away_team.get('id'))

        base_data = {
            'id': fixture.get('id'),
            'league_id': fixture.get('league_id'),
            'season_id': fixture.get('season_id'),
            'starting_at': fixture.get('starting_at'),
            'home_team_id': home_team.get('id'),
            'away_team_id': away_team.get('id'),
            'home_score': home_score,
            'away_score': away_score,
            'result': result,
            'state_id': fixture.get('state_id'),
        }

        # Merge statistics
        base_data.update(stats_dict)

        return base_data

    except Exception as e:
        return None


def extract_statistics(statistics: List[Dict], home_team_id: int, away_team_id: int) -> Dict:
    """Extract key statistics from fixture."""
    stats = {}

    # Stat type IDs we care about (SportMonks API)
    # Reference: data/reference/sportmonks_types.json
    STAT_TYPES = {
        # Shots
        42: 'shots_total',
        86: 'shots_on_target',
        49: 'shots_inside_box',
        50: 'shots_outside_box',
        580: 'big_chances',
        58: 'shots_blocked',
        # Possession & Attacks
        45: 'possession',
        43: 'attacks',
        44: 'dangerous_attacks',
        # Set pieces
        34: 'corners',
        51: 'offsides',
        # Defensive
        57: 'saves',
        78: 'tackles',
        100: 'interceptions',
        56: 'fouls',
        # Passing
        80: 'passes',
        81: 'passes_accurate',  # Successful Passes (116 not available)
    }

    for stat in statistics:
        type_id = stat.get('type_id')
        if type_id not in STAT_TYPES:
            continue

        stat_name = STAT_TYPES[type_id]
        participant_id = stat.get('participant_id')

        # Get the data value
        data = stat.get('data', {})
        value = data.get('value')

        if value is None:
            continue

        # Determine if home or away
        if participant_id == home_team_id:
            stats[f'home_{stat_name}'] = value
        elif participant_id == away_team_id:
            stats[f'away_{stat_name}'] = value

    return stats


def process_json_file(file_path: Path) -> List[Dict]:
    """Process a single JSON file using streaming."""
    fixtures = []

    try:
        with open(file_path, 'rb') as f:
            parser = ijson.items(f, 'item')

            for fixture in parser:
                data = extract_fixture_data(fixture)
                if data:
                    fixtures.append(data)

    except Exception as e:
        logger.warning(f"Failed to parse {file_path.name}: {e}")

    return fixtures


def main():
    fixtures_dir = Path('data/historical/fixtures')
    output_file = Path('data/processed/fixtures_with_stats.csv')

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(fixtures_dir.glob('*.json'))
    logger.info(f"Found {len(json_files)} JSON files")

    all_fixtures = []

    for i, file_path in enumerate(json_files, 1):
        logger.info(f"[{i}/{len(json_files)}] Processing {file_path.name}...")

        fixtures = process_json_file(file_path)
        all_fixtures.extend(fixtures)

        logger.info(f"  -> {len(fixtures)} fixtures extracted (total: {len(all_fixtures)})")

        # Periodic save to avoid losing progress
        if i % 50 == 0:
            logger.info(f"  Checkpoint: saving {len(all_fixtures)} fixtures...")
            df = pd.DataFrame(all_fixtures)
            df.to_csv(output_file, index=False)

    # Final save
    logger.info(f"\nCreating final CSV with {len(all_fixtures)} fixtures...")
    df = pd.DataFrame(all_fixtures)

    # Parse and sort by date
    df['starting_at'] = pd.to_datetime(df['starting_at'])
    df = df.sort_values('starting_at').reset_index(drop=True)

    # Save
    df.to_csv(output_file, index=False)

    logger.info(f"\n✅ Done! Saved to {output_file}")
    logger.info(f"   Total fixtures: {len(df)}")
    logger.info(f"   Date range: {df['starting_at'].min()} to {df['starting_at'].max()}")
    logger.info(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"\nNow run: python3 scripts/generate_training_data.py --output data/training_data.csv")


if __name__ == '__main__':
    main()
