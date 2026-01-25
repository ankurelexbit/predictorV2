"""
Generate Training Features from Historical Data.

Processes downloaded historical data and generates complete feature vectors
for model training.

Usage:
    python scripts/generate_training_features.py --data-dir data/historical --output training_features.csv
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_pipeline import FeaturePipeline
from src.features.elo_calculator import EloCalculator
from src.utils.database import SupabaseDB


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Generate training features from historical data."""
    
    def __init__(self, data_dir: str, use_database: bool = True):
        """
        Initialize feature generator.
        
        Args:
            data_dir: Directory containing downloaded historical data
            use_database: Save features to Supabase (default: True)
        """
        self.data_dir = Path(data_dir)
        self.feature_pipeline = FeaturePipeline()
        self.use_database = use_database
        
        # Initialize database if needed
        if self.use_database:
            self.db = SupabaseDB()
            logger.info("Database connection initialized")
        else:
            self.db = None
        
        # Storage for processed data
        self.fixtures = []
        self.statistics = {}
        self.lineups = {}
        self.team_matches = defaultdict(list)
        
        logger.info(f"Initialized feature generator with data dir: {self.data_dir}")
    
    def load_fixtures(self) -> List[Dict]:
        """Load all fixtures from downloaded data."""
        logger.info("Loading fixtures...")
        
        fixtures_dir = self.data_dir / 'fixtures'
        all_fixtures = []
        
        # Load combined fixtures file if exists
        combined_file = list(fixtures_dir.glob('all_fixtures_*.json'))
        
        if combined_file:
            with open(combined_file[0], 'r') as f:
                all_fixtures = json.load(f)
            logger.info(f"Loaded {len(all_fixtures)} fixtures from combined file")
        else:
            # Load individual league files
            for fixture_file in fixtures_dir.glob('league_*.json'):
                with open(fixture_file, 'r') as f:
                    fixtures = json.load(f)
                    all_fixtures.extend(fixtures)
            logger.info(f"Loaded {len(all_fixtures)} fixtures from league files")
        
        # Sort by date
        all_fixtures.sort(key=lambda x: x.get('starting_at', ''))
        
        self.fixtures = all_fixtures
        return all_fixtures
    
    def load_statistics(self):
        """Load match statistics."""
        logger.info("Loading match statistics...")
        
        stats_dir = self.data_dir / 'statistics'
        
        for stats_file in tqdm(list(stats_dir.glob('fixture_*.json')), desc="Loading statistics"):
            fixture_id = int(stats_file.stem.split('_')[1])
            
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.statistics[fixture_id] = stats
        
        logger.info(f"Loaded statistics for {len(self.statistics)} fixtures")
    
    def parse_statistics(self, fixture_id: int, team_id: int) -> Dict:
        """
        Parse statistics for a team in a fixture.
        
        Args:
            fixture_id: Fixture ID
            team_id: Team ID
        
        Returns:
            Dictionary of parsed statistics
        """
        stats = self.statistics.get(fixture_id, [])
        
        team_stats = {}
        
        for stat_group in stats:
            if stat_group.get('participant_id') == team_id:
                # Parse statistics
                details = stat_group.get('details', [])
                
                for detail in details:
                    type_name = detail.get('type', {}).get('name', '')
                    value = detail.get('value', {}).get('total', 0)
                    
                    # Map to our expected keys
                    if 'Shots Total' in type_name:
                        team_stats['shots_total'] = value
                    elif 'Shots On Target' in type_name:
                        team_stats['shots_on_target'] = value
                    elif 'Shots Insidebox' in type_name:
                        team_stats['shots_insidebox'] = value
                    elif 'Shots Outsidebox' in type_name:
                        team_stats['shots_outsidebox'] = value
                    elif 'Big Chances Created' in type_name:
                        team_stats['big_chances_created'] = value
                    elif 'Corners' in type_name:
                        team_stats['corners'] = value
                    elif 'Attacks' in type_name and 'Dangerous' not in type_name:
                        team_stats['attacks'] = value
                    elif 'Dangerous Attacks' in type_name:
                        team_stats['dangerous_attacks'] = value
                    elif 'Possession' in type_name:
                        team_stats['possession'] = value
                    elif 'Passes' in type_name and 'Accurate' not in type_name:
                        team_stats['passes'] = value
                    elif 'Accurate Passes' in type_name:
                        team_stats['accurate_passes'] = value
                    elif 'Tackles' in type_name:
                        team_stats['tackles'] = value
                    elif 'Interceptions' in type_name:
                        team_stats['interceptions'] = value
                    elif 'Clearances' in type_name:
                        team_stats['clearances'] = value
        
        return team_stats
    
    def organize_team_matches(self):
        """Organize fixtures by team for historical lookback."""
        logger.info("Organizing matches by team...")
        
        for fixture in tqdm(self.fixtures, desc="Organizing"):
            fixture_id = fixture.get('id')
            match_date = datetime.fromisoformat(fixture.get('starting_at', '').replace('Z', '+00:00'))
            
            participants = fixture.get('participants', [])
            if len(participants) != 2:
                continue
            
            home_team = participants[0]
            away_team = participants[1]
            
            home_team_id = home_team.get('id')
            away_team_id = away_team.get('id')
            
            # Get scores
            scores = fixture.get('scores', [])
            home_goals = 0
            away_goals = 0
            
            for score in scores:
                if score.get('description') == 'CURRENT':
                    participant_id = score.get('participant_id')
                    goals = score.get('score', {}).get('goals', 0)
                    
                    if participant_id == home_team_id:
                        home_goals = goals
                    elif participant_id == away_team_id:
                        away_goals = goals
            
            # Determine result
            if home_goals > away_goals:
                home_result = 'W'
                away_result = 'L'
            elif home_goals < away_goals:
                home_result = 'L'
                away_result = 'W'
            else:
                home_result = 'D'
                away_result = 'D'
            
            # Get statistics
            home_stats = self.parse_statistics(fixture_id, home_team_id)
            away_stats = self.parse_statistics(fixture_id, away_team_id)
            
            # Store for home team
            self.team_matches[home_team_id].append({
                'fixture_id': fixture_id,
                'match_date': match_date,
                'is_home': True,
                'opponent_id': away_team_id,
                'goals_scored': home_goals,
                'goals_conceded': away_goals,
                'result': home_result,
                'team_stats': home_stats,
                'opponent_stats': away_stats,
            })
            
            # Store for away team
            self.team_matches[away_team_id].append({
                'fixture_id': fixture_id,
                'match_date': match_date,
                'is_home': False,
                'opponent_id': home_team_id,
                'goals_scored': away_goals,
                'goals_conceded': home_goals,
                'result': away_result,
                'team_stats': away_stats,
                'opponent_stats': home_stats,
            })
        
        # Sort each team's matches by date
        for team_id in self.team_matches:
            self.team_matches[team_id].sort(key=lambda x: x['match_date'])
        
        logger.info(f"Organized matches for {len(self.team_matches)} teams")
    
    def calculate_elo_ratings(self):
        """Calculate Elo ratings for all matches."""
        logger.info("Calculating Elo ratings...")
        
        elo_calc = self.feature_pipeline.elo_calc
        
        for fixture in tqdm(self.fixtures, desc="Calculating Elo"):
            fixture_id = fixture.get('id')
            match_date = datetime.fromisoformat(fixture.get('starting_at', '').replace('Z', '+00:00'))
            
            participants = fixture.get('participants', [])
            if len(participants) != 2:
                continue
            
            home_team_id = participants[0].get('id')
            away_team_id = participants[1].get('id')
            
            # Get scores
            scores = fixture.get('scores', [])
            home_goals = 0
            away_goals = 0
            
            for score in scores:
                if score.get('description') == 'CURRENT':
                    participant_id = score.get('participant_id')
                    goals = score.get('score', {}).get('goals', 0)
                    
                    if participant_id == home_team_id:
                        home_goals = goals
                    elif participant_id == away_team_id:
                        away_goals = goals
            
            # Update Elo ratings
            elo_calc.update_ratings(
                home_team_id,
                away_team_id,
                home_goals,
                away_goals,
                match_date
            )
        
        logger.info("Elo ratings calculated")
    
    def get_team_matches_before_date(
        self,
        team_id: int,
        target_date: datetime,
        n: int = None
    ) -> List[Dict]:
        """
        Get team's matches before a specific date.
        
        Args:
            team_id: Team ID
            target_date: Target date
            n: Number of matches (None = all)
        
        Returns:
            List of matches
        """
        team_matches = self.team_matches.get(team_id, [])
        
        # Filter matches before target date
        before_matches = [
            m for m in team_matches
            if m['match_date'] < target_date
        ]
        
        if n is not None:
            return before_matches[-n:]
        
        return before_matches
    
    def get_h2h_matches(
        self,
        team1_id: int,
        team2_id: int,
        target_date: datetime
    ) -> List[Dict]:
        """Get H2H matches between two teams before a date."""
        team1_matches = self.team_matches.get(team1_id, [])
        
        h2h = []
        for match in team1_matches:
            if match['match_date'] < target_date and match['opponent_id'] == team2_id:
                h2h.append({
                    'home_team_id': team1_id if match['is_home'] else team2_id,
                    'away_team_id': team2_id if match['is_home'] else team1_id,
                    'home_goals': match['goals_scored'] if match['is_home'] else match['goals_conceded'],
                    'away_goals': match['goals_conceded'] if match['is_home'] else match['goals_scored'],
                })
        
        return h2h
    
    def generate_features_list(self) -> List[Dict]:
        """
        Generate features for all fixtures.
        
        Returns:
            List of feature dictionaries
        """
        logger.info("Generating features for all fixtures...")
        
        all_features = []
        
        for fixture in tqdm(self.fixtures, desc="Generating features"):
            fixture_id = fixture.get('id')
            match_date = datetime.fromisoformat(fixture.get('starting_at', '').replace('Z', '+00:00'))
            
            participants = fixture.get('participants', [])
            if len(participants) != 2:
                continue
            
            home_team_id = participants[0].get('id')
            away_team_id = participants[1].get('id')
            
            # Get historical matches
            home_matches = self.get_team_matches_before_date(home_team_id, match_date)
            away_matches = self.get_team_matches_before_date(away_team_id, match_date)
            h2h_matches = self.get_h2h_matches(home_team_id, away_team_id, match_date)
            
            # Skip if not enough historical data
            if len(home_matches) < 3 or len(away_matches) < 3:
                continue
            
            try:
                # Calculate features
                features = self.feature_pipeline.calculate_features_for_match(
                    home_team_id,
                    away_team_id,
                    match_date,
                    home_matches,
                    away_matches,
                    h2h_matches
                )
                
                # Add metadata
                features['fixture_id'] = fixture_id
                features['match_date'] = match_date.isoformat()
                features['home_team_id'] = home_team_id
                features['away_team_id'] = away_team_id
                
                # Add target
                scores = fixture.get('scores', [])
                home_goals = 0
                away_goals = 0
                
                for score in scores:
                    if score.get('description') == 'CURRENT':
                        participant_id = score.get('participant_id')
                        goals = score.get('score', {}).get('goals', 0)
                        
                        if participant_id == home_team_id:
                            home_goals = goals
                        elif participant_id == away_team_id:
                            away_goals = goals
                
                features['home_goals'] = home_goals
                features['away_goals'] = away_goals
                
                if home_goals > away_goals:
                    features['target'] = 'H'
                elif home_goals < away_goals:
                    features['target'] = 'A'
                else:
                    features['target'] = 'D'
                
                # Prepare for database storage
                # Separate metadata from features
                feature_record = {
                    'fixture_id': fixture_id,
                    'match_date': match_date.isoformat(),
                    'features': {k: v for k, v in features.items() 
                                if k not in ['fixture_id', 'match_date', 'home_team_id', 'away_team_id', 
                                           'home_goals', 'away_goals', 'target']},
                    'target': features['target'],
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                }
                
                all_features.append(feature_record)
                
            except Exception as e:
                logger.error(f"Error generating features for fixture {fixture_id}: {e}")
                continue
        
        logger.info(f"Generated features for {len(all_features)} fixtures")
        
        return all_features
    
    def run(self, output_file: Optional[str] = None):
        """
        Run complete feature generation pipeline.
        
        Args:
            output_file: Optional CSV file path for export
        """
        logger.info("=" * 80)
        logger.info("STARTING FEATURE GENERATION")
        logger.info("=" * 80)
        
        # Load data from JSON
        self.load_fixtures()
        self.load_statistics()
        
        # Organize data
        self.organize_team_matches()
        
        # Calculate Elo ratings
        self.calculate_elo_ratings()
        
        # Generate features
        all_features = self.generate_features_list()
        
        # Save to database
        if self.use_database and all_features:
            logger.info(f"\nSaving {len(all_features)} feature vectors to Supabase...")
            try:
                self.db.insert_features_batch(all_features)
                logger.info("✅ Features saved to Supabase successfully")
            except Exception as e:
                logger.error(f"Error saving to database: {e}")
                raise
        
        # Optionally export to CSV
        if output_file:
            df = pd.DataFrame(all_features)
            df.to_csv(output_file, index=False)
            logger.info(f"✅ Exported {len(df)} feature vectors to {output_file}")
        
        # Summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total fixtures processed: {len(self.fixtures)}")
        logger.info(f"Feature vectors generated: {len(all_features)}")
        if all_features:
            logger.info(f"Features per vector: {len(all_features[0])}")
        if self.use_database:
            logger.info(f"Storage: Supabase database ✅")
        if output_file:
            logger.info(f"CSV export: {output_file} ✅")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate training features from historical data')
    parser.add_argument('--data-dir', default='data/historical', help='Directory with downloaded data')
    parser.add_argument('--output', default=None, help='Optional: Export to CSV file')
    parser.add_argument('--no-database', action='store_true', help='Skip database storage (CSV only)')
    
    args = parser.parse_args()
    
    # Run feature generation
    use_db = not args.no_database
    generator = FeatureGenerator(data_dir=args.data_dir, use_database=use_db)
    
    try:
        generator.run(output_file=args.output)
    except KeyboardInterrupt:
        logger.info("\nFeature generation interrupted by user")
    except Exception as e:
        logger.error(f"Feature generation failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
