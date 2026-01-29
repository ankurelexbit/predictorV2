"""
Convert Standings JSON to CSV

Converts the standings data extracted from participants.meta
into a CSV file for use in the feature engineering pipeline.

Usage:
    python scripts/convert_standings_to_csv.py
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_standings_to_csv(
    input_dir: str = 'data/historical/standings',
    output_file: str = 'data/csv/standings.csv'
):
    """
    Convert standings JSON files to a single CSV.
    
    Args:
        input_dir: Directory containing standings JSON files
        output_file: Output CSV file path
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all standings data
    all_standings = []
    
    json_files = list(input_path.glob('fixture_*.json'))
    logger.info(f"Found {len(json_files)} standings files")
    
    for json_file in tqdm(json_files, desc="Converting"):
        try:
            with open(json_file, 'r') as f:
                standings_data = json.load(f)
            
            # standings_data is a list of 2 items (home and away)
            for team_standing in standings_data:
                all_standings.append(team_standing)
        
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_standings)
    
    # Sort by fixture_id and location
    df = df.sort_values(['fixture_id', 'location'])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Converted {len(all_standings)} standings records to {output_path}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nSample data:")
    print(df.head(10))


if __name__ == '__main__':
    convert_standings_to_csv()
