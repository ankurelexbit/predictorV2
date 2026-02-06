#!/usr/bin/env python3
"""
Extract Market Features from Historical JSON
=============================================

Step 1: Extracts pre-match market features from fixture JSON files.
        No model needed — just raw market data → CSV.

Step 2 (train_bet_selector.py): Joins this CSV with model probabilities.

Usage:
    python3 scripts/extract_market_features.py
    python3 scripts/extract_market_features.py --min-date 2020-01-01
"""

import sys
import json
import argparse
import logging
import gc
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market.market_feature_extractor import MarketFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / 'data' / 'historical' / 'fixtures'
TOP_5_LEAGUES = {8, 82, 384, 564, 301}


def extract_result(fixture: dict) -> str:
    """Extract actual match result from fixture."""
    scores = fixture.get('scores', [])
    home_score = away_score = None
    for s in scores:
        if s.get('description') == 'CURRENT':
            score_data = s.get('score', {})
            if score_data.get('participant') == 'home':
                home_score = score_data.get('goals')
            elif score_data.get('participant') == 'away':
                away_score = score_data.get('goals')
    if home_score is not None and away_score is not None:
        if home_score > away_score:
            return 'H'
        elif home_score < away_score:
            return 'A'
        else:
            return 'D'
    return None


def process_json_file(json_file: Path, extractor: MarketFeatureExtractor, min_date: datetime) -> list:
    """Process one JSON file → list of {fixture_id, match_date, result, market features}."""
    records = []
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Error reading {json_file.name}: {e}")
        return records

    if isinstance(data, dict) and 'data' in data:
        fixtures = data['data']
    elif isinstance(data, list):
        fixtures = data
    else:
        fixtures = [data]

    for fix in fixtures:
        fixture_id = fix.get('id')
        if not fixture_id or not fix.get('odds') or not fix.get('starting_at'):
            continue

        try:
            fix_date = datetime.strptime(str(fix['starting_at']).strip(), '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            continue
        if fix_date < min_date:
            continue

        actual_result = extract_result(fix)
        if not actual_result:
            continue

        market_features = extractor.extract_from_json(fix)
        if market_features is None:
            continue

        record = {
            'fixture_id': fixture_id,
            'match_date': fix['starting_at'],
            'league_id': fix.get('league_id'),
            'actual_result': actual_result,
        }
        record.update(market_features)
        records.append(record)

    del data, fixtures
    gc.collect()
    return records


def main():
    parser = argparse.ArgumentParser(description='Extract market features from historical JSON')
    parser.add_argument('--output', default='data/market_features_raw.csv')
    parser.add_argument('--min-date', default='2017-01-01', help='Minimum fixture date')
    args = parser.parse_args()

    output_path = Path(__file__).parent.parent / args.output
    min_date = datetime.strptime(args.min_date, '%Y-%m-%d')
    extractor = MarketFeatureExtractor()

    json_files = sorted(DATA_DIR.glob('*.json'))
    json_files = [f for f in json_files if any(f'league_{lid}_' in f.name for lid in TOP_5_LEAGUES)]

    # Skip files whose date range ends before min_date (avoids loading huge pre-2022 files)
    import re
    filtered = []
    for f in json_files:
        match = re.search(r'_(\d{4}-\d{2}-\d{2})\.json$', f.name)
        if match:
            file_end = datetime.strptime(match.group(1), '%Y-%m-%d')
            if file_end < min_date:
                continue
        filtered.append(f)
    skipped = len(json_files) - len(filtered)
    json_files = filtered
    logger.info(f"Processing {len(json_files)} JSON files (skipped {skipped} pre-{args.min_date})...")

    all_records = []
    for i, json_file in enumerate(json_files):
        records = process_json_file(json_file, extractor, min_date)
        all_records.extend(records)
        if (i + 1) % 20 == 0 or i == len(json_files) - 1:
            logger.info(f"  {i+1}/{len(json_files)} files, {len(all_records)} fixtures extracted")

    if not all_records:
        logger.error("No features extracted.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    df = df.sort_values('match_date').reset_index(drop=True)

    logger.info(f"\nExtracted {len(df)} fixtures with market features")
    logger.info(f"Date range: {df['match_date'].iloc[0]} to {df['match_date'].iloc[-1]}")
    logger.info(f"Results: {df['actual_result'].value_counts().to_dict()}")

    # Quick feature stats
    for col in ['home_best_odds', 'draw_best_odds', 'away_best_odds', 'market_overround', 'num_bookmakers']:
        if col in df.columns:
            logger.info(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")

    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
