"""
Database utility module for Supabase integration.
"""
import os
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
from supabase import create_client, Client
from config.database_config import DatabaseConfig


logger = logging.getLogger(__name__)


class SupabaseDB:
    """Supabase database client for storing and retrieving data."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.url = DatabaseConfig.SUPABASE_URL
        self.key = DatabaseConfig.SUPABASE_KEY
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        
        self.client: Client = create_client(self.url, self.key)
        logger.info("Supabase client initialized")
    
    # ========================================================================
    # MATCHES
    # ========================================================================
    
    def insert_match(self, match_data: Dict) -> Dict:
        """
        Insert a match record.
        
        Args:
            match_data: Match data dictionary
        
        Returns:
            Inserted record
        """
        try:
            result = self.client.table('matches').insert(match_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error inserting match: {e}")
            raise
    
    def insert_matches_batch(self, matches: List[Dict]) -> List[Dict]:
        """
        Insert multiple matches.
        
        Args:
            matches: List of match data
        
        Returns:
            List of inserted records
        """
        try:
            result = self.client.table('matches').insert(matches).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error inserting matches batch: {e}")
            raise
    
    def get_matches_between_dates(
        self,
        start_date: str,
        end_date: str,
        league_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get matches between dates.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league_id: Optional league filter
        
        Returns:
            List of matches
        """
        query = self.client.table('matches').select('*').gte('match_date', start_date).lte('match_date', end_date)
        
        if league_id:
            query = query.eq('league_id', league_id)
        
        result = query.execute()
        return result.data
    
    def get_team_matches(
        self,
        team_id: int,
        before_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get matches for a team.
        
        Args:
            team_id: Team ID
            before_date: Optional date filter
            limit: Optional limit
        
        Returns:
            List of matches
        """
        query = self.client.table('matches').select('*').or_(f'home_team_id.eq.{team_id},away_team_id.eq.{team_id}')
        
        if before_date:
            query = query.lt('match_date', before_date)
        
        query = query.order('match_date', desc=False)
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        return result.data
    
    # ========================================================================
    # MATCH STATISTICS
    # ========================================================================
    
    def insert_match_statistics(self, stats_data: Dict) -> Dict:
        """Insert match statistics."""
        try:
            result = self.client.table('match_statistics').insert(stats_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error inserting statistics: {e}")
            raise
    
    def insert_statistics_batch(self, statistics: List[Dict]) -> List[Dict]:
        """Insert multiple statistics records."""
        try:
            result = self.client.table('match_statistics').insert(statistics).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error inserting statistics batch: {e}")
            raise
    
    def get_match_statistics(self, fixture_id: int) -> List[Dict]:
        """Get statistics for a fixture."""
        result = self.client.table('match_statistics').select('*').eq('fixture_id', fixture_id).execute()
        return result.data
    
    # ========================================================================
    # ELO HISTORY
    # ========================================================================
    
    def insert_elo_record(self, elo_data: Dict) -> Dict:
        """Insert Elo rating record."""
        try:
            result = self.client.table('elo_history').insert(elo_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error inserting Elo record: {e}")
            raise
    
    def insert_elo_batch(self, elo_records: List[Dict]) -> List[Dict]:
        """Insert multiple Elo records."""
        try:
            result = self.client.table('elo_history').insert(elo_records).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error inserting Elo batch: {e}")
            raise
    
    def get_team_elo_history(
        self,
        team_id: int,
        before_date: Optional[str] = None
    ) -> List[Dict]:
        """Get Elo history for a team."""
        query = self.client.table('elo_history').select('*').eq('team_id', team_id)
        
        if before_date:
            query = query.lte('match_date', before_date)
        
        query = query.order('match_date', desc=False)
        
        result = query.execute()
        return result.data
    
    # ========================================================================
    # TRAINING FEATURES
    # ========================================================================
    
    def insert_training_features(self, features_data: Dict) -> Dict:
        """Insert training features."""
        try:
            result = self.client.table('training_features').insert(features_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error inserting training features: {e}")
            raise
    
    def insert_features_batch(self, features: List[Dict]) -> List[Dict]:
        """Insert multiple training features."""
        try:
            # Supabase has a limit on batch size, so chunk if needed
            batch_size = 1000
            all_results = []
            
            for i in range(0, len(features), batch_size):
                batch = features[i:i + batch_size]
                result = self.client.table('training_features').insert(batch).execute()
                all_results.extend(result.data)
            
            return all_results
        except Exception as e:
            logger.error(f"Error inserting features batch: {e}")
            raise
    
    def get_training_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Get training features."""
        query = self.client.table('training_features').select('*')
        
        if start_date:
            query = query.gte('match_date', start_date)
        if end_date:
            query = query.lte('match_date', end_date)
        
        query = query.order('match_date', desc=False)
        
        result = query.execute()
        return result.data
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def upsert_match(self, match_data: Dict) -> Dict:
        """
        Upsert match (insert or update if exists).
        
        Args:
            match_data: Match data
        
        Returns:
            Upserted record
        """
        try:
            result = self.client.table('matches').upsert(match_data, on_conflict='fixture_id').execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error upserting match: {e}")
            raise
    
    def delete_old_cache(self, table: str, days: int = 30):
        """
        Delete old cached data.
        
        Args:
            table: Table name
            days: Days to keep
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            result = self.client.table(table).delete().lt('created_at', cutoff_date).execute()
            logger.info(f"Deleted {len(result.data)} old records from {table}")
        except Exception as e:
            logger.error(f"Error deleting old cache: {e}")
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            result = self.client.table('matches').select('id').limit(1).execute()
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
