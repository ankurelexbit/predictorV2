"""
Supabase Database Module for Predictions
Handles storing and retrieving predictions from Supabase PostgreSQL
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

from config import DATABASE_URL

logger = logging.getLogger(__name__)

class PredictionsDB:
    """Database interface for storing predictions in Supabase."""
    
    def __init__(self, connection_url: str = None):
        """
        Initialize database connection.
        
        Args:
            connection_url: PostgreSQL connection URL (uses config.DATABASE_URL if not provided)
        """
        self.connection_url = connection_url or DATABASE_URL
        self.conn = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.connection_url)
            logger.info("✅ Connected to Supabase database")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def insert_prediction(self, prediction_data: Dict) -> Optional[int]:
        """
        Insert a new prediction into the database.
        
        Args:
            prediction_data: Dictionary containing prediction information
            
        Returns:
            Prediction ID if successful, None otherwise
        """
        if not self.conn:
            if not self.connect():
                return None
        
        try:
            with self.conn.cursor() as cur:
                sql = """
                INSERT INTO predictions (
                    fixture_id, match_date, home_team, away_team, league, league_id,
                    prob_home, prob_draw, prob_away,
                    recommended_bet, confidence,
                    odds_home, odds_draw, odds_away, best_odds,
                    features_count, model_version, thresholds, features,
                    kickoff_time, prediction_time, hours_before_kickoff,
                    home_lineup, away_lineup, lineup_available, 
                    lineup_coverage_home, lineup_coverage_away,
                    home_injuries_count, away_injuries_count,
                    home_injured_players, away_injured_players,
                    bookmaker_odds, best_odds_home, best_odds_draw, best_odds_away,
                    our_odds_home, our_odds_draw, our_odds_away,
                    used_lineup_data, used_injury_data, data_quality_score
                ) VALUES (
                    %(fixture_id)s, %(match_date)s, %(home_team)s, %(away_team)s, 
                    %(league)s, %(league_id)s,
                    %(prob_home)s, %(prob_draw)s, %(prob_away)s,
                    %(recommended_bet)s, %(confidence)s,
                    %(odds_home)s, %(odds_draw)s, %(odds_away)s, %(best_odds)s,
                    %(features_count)s, %(model_version)s, %(thresholds)s, %(features)s,
                    %(kickoff_time)s, %(prediction_time)s, %(hours_before_kickoff)s,
                    %(home_lineup)s, %(away_lineup)s, %(lineup_available)s,
                    %(lineup_coverage_home)s, %(lineup_coverage_away)s,
                    %(home_injuries_count)s, %(away_injuries_count)s,
                    %(home_injured_players)s, %(away_injured_players)s,
                    %(bookmaker_odds)s, %(best_odds_home)s, %(best_odds_draw)s, %(best_odds_away)s,
                    %(our_odds_home)s, %(our_odds_draw)s, %(our_odds_away)s,
                    %(used_lineup_data)s, %(used_injury_data)s, %(data_quality_score)s
                )
                ON CONFLICT (fixture_id, created_at) DO NOTHING
                RETURNING id;
                """
                
                cur.execute(sql, prediction_data)
                result = cur.fetchone()
                self.conn.commit()
                
                if result:
                    prediction_id = result[0]
                    logger.info(f"✅ Saved prediction ID {prediction_id} for {prediction_data['home_team']} vs {prediction_data['away_team']}")
                    return prediction_id
                else:
                    logger.warning(f"⚠️  Prediction already exists for fixture {prediction_data['fixture_id']}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to insert prediction: {e}")
            self.conn.rollback()
            return None
    
    def update_actual_result(self, fixture_id: int, home_goals: int, away_goals: int) -> bool:
        """
        Update prediction with actual match result.
        
        Args:
            fixture_id: Fixture ID
            home_goals: Actual home team goals
            away_goals: Actual away team goals
            
        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            if not self.connect():
                return False
        
        try:
            # Determine actual result
            if home_goals > away_goals:
                actual_result = 'HOME'
            elif home_goals < away_goals:
                actual_result = 'AWAY'
            else:
                actual_result = 'DRAW'
            
            with self.conn.cursor() as cur:
                sql = """
                UPDATE predictions
                SET 
                    actual_result = %s,
                    actual_home_goals = %s,
                    actual_away_goals = %s,
                    is_correct = (recommended_bet = %s),
                    profit_loss = CASE 
                        WHEN recommended_bet = 'NO_BET' THEN 0
                        WHEN recommended_bet = %s THEN (best_odds - 1) * 100
                        ELSE -100
                    END
                WHERE fixture_id = %s
                AND actual_result IS NULL;
                """
                
                cur.execute(sql, (actual_result, home_goals, away_goals, actual_result, actual_result, fixture_id))
                self.conn.commit()
                
                if cur.rowcount > 0:
                    logger.info(f"✅ Updated result for fixture {fixture_id}: {home_goals}-{away_goals}")
                    return True
                else:
                    logger.warning(f"⚠️  No prediction found for fixture {fixture_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to update result: {e}")
            self.conn.rollback()
            return False
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Get recent predictions.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        if not self.conn:
            if not self.connect():
                return []
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                SELECT *
                FROM predictions
                ORDER BY created_at DESC
                LIMIT %s;
                """
                
                cur.execute(sql, (limit,))
                results = cur.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch predictions: {e}")
            return []
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """
        Get performance statistics for the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.conn:
            if not self.connect():
                return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN recommended_bet != 'NO_BET' THEN 1 ELSE 0 END) as total_bets,
                    SUM(CASE WHEN is_correct = true THEN 1 ELSE 0 END) as correct_bets,
                    ROUND(AVG(CASE WHEN is_correct = true THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate_pct,
                    SUM(profit_loss) as total_profit,
                    ROUND(SUM(profit_loss) / NULLIF(SUM(CASE WHEN recommended_bet != 'NO_BET' THEN 1 ELSE 0 END), 0) * 100, 2) as roi_pct
                FROM predictions
                WHERE match_date >= NOW() - INTERVAL '%s days'
                AND recommended_bet != 'NO_BET'
                AND actual_result IS NOT NULL;
                """
                
                cur.execute(sql, (days,))
                result = cur.fetchone()
                
                return dict(result) if result else {}
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch performance stats: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def save_prediction_to_db(prediction: Dict) -> bool:
    """
    Convenience function to save a single prediction.
    
    Args:
        prediction: Prediction dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with PredictionsDB() as db:
            prediction_id = db.insert_prediction(prediction)
            return prediction_id is not None
    except Exception as e:
        logger.error(f"❌ Error saving prediction: {e}")
        return False
