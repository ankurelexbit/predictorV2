#!/usr/bin/env python3
"""
Process Raw Data (CSV-only version)
====================================

This script processes raw football data CSV files and creates a consolidated
matches.csv file without using a database.

Usage:
    python 02_process_raw_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils import setup_logger
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np

logger = setup_logger("process_raw_data")


def normalize_team_name(name: str) -> str:
    """Normalize team name for consistency."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def process_csv_file(csv_path: Path) -> pd.DataFrame:
    """Process a single CSV file and return standardized match data."""
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return pd.DataFrame()
        
        # Extract league and season from filename
        filename = csv_path.stem  # e.g., "E0_2021"
        parts = filename.split('_')
        league_code = parts[0]
        season_year = parts[1] if len(parts) > 1 else '2024'
        
        # Convert season format (2021 -> 2020-2021)
        if len(season_year) == 4:
            year1 = '20' + season_year[:2]
            year2 = '20' + season_year[2:]
            season = f'{year1}-{year2}'
        else:
            season = season_year
        
        # Create processed dataframe
        processed_df = pd.DataFrame({
            'date': pd.to_datetime(df['Date'], dayfirst=True),
            'season': season,
            'league_code': league_code,
            'home_team': df['HomeTeam'],
            'away_team': df['AwayTeam'],
            'home_goals': df['FTHG'].astype(int),
            'away_goals': df['FTAG'].astype(int),
            'result': df['FTR'],
            'ht_home_goals': df.get('HTHG', np.nan),
            'ht_away_goals': df.get('HTAG', np.nan),
            'ht_result': df.get('HTR', ''),
        })
        
        # Add normalized team names
        processed_df['home_team_normalized'] = processed_df['home_team'].apply(normalize_team_name)
        processed_df['away_team_normalized'] = processed_df['away_team'].apply(normalize_team_name)
        
        # Add league names
        league_names = {
            'E0': 'Premier League',
            'E1': 'Championship', 
            'SP1': 'La Liga',
            'D1': 'Bundesliga',
            'I1': 'Serie A',
            'F1': 'Ligue 1'
        }
        processed_df['league_name'] = league_names.get(league_code, league_code)
        
        # Add match stats if available
        for col in ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']:
            if col in df.columns:
                new_col = {
                    'HS': 'home_shots',
                    'AS': 'away_shots',
                    'HST': 'home_shots_on_target',
                    'AST': 'away_shots_on_target',
                    'HC': 'home_corners',
                    'AC': 'away_corners',
                    'HF': 'home_fouls',
                    'AF': 'away_fouls',
                    'HY': 'home_yellows',
                    'AY': 'away_yellows',
                    'HR': 'home_reds',
                    'AR': 'away_reds'
                }.get(col, col.lower())
                processed_df[new_col] = df[col]
        
        # Extract best odds for each outcome
        home_odds_cols = [col for col in df.columns if col.endswith('H') and len(col) > 1]
        draw_odds_cols = [col for col in df.columns if col.endswith('D') and len(col) > 1]
        away_odds_cols = [col for col in df.columns if col.endswith('A') and len(col) > 1]
        
        if home_odds_cols:
            # Get best (highest) odds for each outcome
            processed_df['best_odds_home'] = df[home_odds_cols].max(axis=1)
            processed_df['avg_odds_home'] = df[home_odds_cols].mean(axis=1)
        
        if draw_odds_cols:
            processed_df['best_odds_draw'] = df[draw_odds_cols].max(axis=1)
            processed_df['avg_odds_draw'] = df[draw_odds_cols].mean(axis=1)
        
        if away_odds_cols:
            processed_df['best_odds_away'] = df[away_odds_cols].max(axis=1)
            processed_df['avg_odds_away'] = df[away_odds_cols].mean(axis=1)
        
        # Add source file
        processed_df['source_file'] = csv_path.name
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        return pd.DataFrame()


def main():
    """Process all raw CSV files and create consolidated matches.csv."""
    
    print("\n" + "=" * 60)
    print("PROCESSING RAW DATA (CSV-only)")
    print("=" * 60)
    
    # Get all CSV files
    raw_dir = RAW_DATA_DIR / "football_data_uk"
    csv_files = sorted([f for f in raw_dir.glob("*.csv") if f.name != "all_matches.csv"])
    
    if not csv_files:
        logger.error(f"No CSV files found in {raw_dir}")
        print(f"\n❌ No CSV files found in {raw_dir}")
        print("\nPlease run: python 01_data_collection.py")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    
    # Process each file
    all_matches = []
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        matches_df = process_csv_file(csv_file)
        if not matches_df.empty:
            all_matches.append(matches_df)
    
    # Combine all matches
    if all_matches:
        combined_df = pd.concat(all_matches, ignore_index=True)
        
        # Sort by date
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        # Add match_id
        combined_df['match_id'] = range(1, len(combined_df) + 1)
        
        # Reorder columns
        first_cols = ['match_id', 'date', 'season', 'league_code', 'league_name',
                      'home_team', 'away_team', 'home_goals', 'away_goals', 'result']
        other_cols = [col for col in combined_df.columns if col not in first_cols]
        combined_df = combined_df[first_cols + other_cols]
        
        # Save to processed directory
        output_path = PROCESSED_DATA_DIR / "matches.csv"
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Successfully processed {len(combined_df)} matches")
        print(f"📁 Saved to: {output_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"  Total matches: {len(combined_df):,}")
        print(f"  Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
        print(f"  Leagues: {', '.join(combined_df['league_code'].unique())}")
        print(f"  Seasons: {', '.join(sorted(combined_df['season'].unique()))}")
        
        # League breakdown
        print("\nMatches by league:")
        league_counts = combined_df.groupby(['league_code', 'league_name']).size()
        for (code, name), count in league_counts.items():
            print(f"  {code}: {name:<20} {count:>5,} matches")
        
    else:
        logger.error("No matches found in any CSV file")
        print("\n❌ No matches found in CSV files")


if __name__ == "__main__":
    main()