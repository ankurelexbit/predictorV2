"""
02 - Data Storage & Schema
==========================

This notebook sets up the database schema and ingests raw data.

Database: PostgreSQL (Supabase)

Schema:
- teams: Normalized team information
- matches: Historical and upcoming matches
- odds_snapshots: Time-series odds data
- predictions: Model prediction snapshots
- user_bets: Bet tracking (for product)

Usage:
    python 02_data_storage.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Boolean, ForeignKey, Text, Index, CheckConstraint, UniqueConstraint,
    event
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, aliased
from sqlalchemy.sql import func
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATABASE_URL, RAW_DATA_DIR, PROCESSED_DATA_DIR, LEAGUES_CSV
from utils import (
    setup_logger, normalize_team_name, encode_result, 
    parse_date, get_season_from_date
)

# Setup
logger = setup_logger("data_storage")
Base = declarative_base()


# =============================================================================
# DATABASE MODELS
# =============================================================================

class Team(Base):
    """Normalized team information."""
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    normalized_name = Column(String(100), nullable=False, index=True)
    country = Column(String(50))
    
    # External IDs for API mapping
    football_data_uk_name = Column(String(100))  # Name in CSV files
    football_data_org_id = Column(Integer)
    api_football_id = Column(Integer)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Team(id={self.id}, name='{self.name}')>"


class League(Base):
    """League/Competition information."""
    __tablename__ = "leagues"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    code = Column(String(10), nullable=False, unique=True)  # E0, PL, etc.
    country = Column(String(50))
    
    # External IDs
    football_data_org_id = Column(Integer)
    api_football_id = Column(Integer)
    
    created_at = Column(DateTime, default=func.now())


class Match(Base):
    """
    Match information - central fact table.
    
    Stores both historical results and upcoming fixtures.
    """
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Match identifiers
    external_id = Column(String(50))  # ID from data source
    source = Column(String(50))  # football_data_uk, football_data_org, etc.
    
    # Match info
    league_id = Column(Integer, ForeignKey("leagues.id"), index=True)
    season = Column(String(20), nullable=False, index=True)  # "2023-2024"
    matchday = Column(Integer)
    date = Column(DateTime, nullable=False, index=True)
    
    # Teams
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    
    # Result (null for upcoming matches)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    result = Column(String(1))  # H, D, A
    result_numeric = Column(Integer)  # 0, 1, 2
    
    # Half-time (if available)
    ht_home_goals = Column(Integer)
    ht_away_goals = Column(Integer)
    
    # Match statistics (if available)
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellows = Column(Integer)
    away_yellows = Column(Integer)
    home_reds = Column(Integer)
    away_reds = Column(Integer)
    
    # Status
    status = Column(String(20), default="scheduled")  # scheduled, finished, postponed
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    league = relationship("League")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('date', 'home_team_id', 'away_team_id', name='unique_match'),
        Index('ix_match_date_league', 'date', 'league_id'),
    )
    
    def __repr__(self):
        return f"<Match(id={self.id}, {self.date}: home_id={self.home_team_id} vs away_id={self.away_team_id})>"


class OddsSnapshot(Base):
    """
    Time-series betting odds.
    
    Store multiple snapshots to track line movements.
    """
    __tablename__ = "odds_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)
    bookmaker = Column(String(50), nullable=False, index=True)
    
    # 1X2 odds (decimal)
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)
    
    # Implied probabilities (with vig)
    home_implied = Column(Float)
    draw_implied = Column(Float)
    away_implied = Column(Float)
    
    # Fair probabilities (vig removed)
    home_fair = Column(Float)
    draw_fair = Column(Float)
    away_fair = Column(Float)
    
    # Overround/vig
    overround = Column(Float)
    
    # Timestamps
    snapshot_time = Column(DateTime, nullable=False, default=func.now())
    source_updated_at = Column(DateTime)
    
    # Relationships
    match = relationship("Match")
    
    __table_args__ = (
        Index('ix_odds_match_book_time', 'match_id', 'bookmaker', 'snapshot_time'),
    )


class PredictionSnapshot(Base):
    """
    Model prediction snapshots - CRITICAL for transparency.
    
    Store every prediction with model version for auditability.
    """
    __tablename__ = "prediction_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    model_type = Column(String(50))  # elo, dixon_coles, xgboost, ensemble
    
    # Predicted probabilities
    p_home = Column(Float, nullable=False)
    p_draw = Column(Float, nullable=False)
    p_away = Column(Float, nullable=False)
    
    # Model confidence/uncertainty
    confidence_tier = Column(String(20))  # low, medium, high
    
    # Feature hash for reproducibility
    features_hash = Column(String(32))
    
    # Timestamp
    generated_at = Column(DateTime, nullable=False, default=func.now())
    
    # Relationships
    match = relationship("Match")
    
    # Constraints - probabilities must sum to 1
    __table_args__ = (
        CheckConstraint('ABS(p_home + p_draw + p_away - 1.0) < 0.001', name='probs_sum_to_one'),
        Index('ix_pred_match_model', 'match_id', 'model_version'),
    )


class TeamEloRating(Base):
    """
    Team Elo ratings over time.
    
    Track rating changes for visualization and debugging.
    """
    __tablename__ = "team_elo_ratings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    rating_date = Column(DateTime, nullable=False)
    match_id = Column(Integer, ForeignKey("matches.id"))  # Match that caused update
    
    # Metadata
    rating_type = Column(String(20), default="standard")  # standard, home_only, away_only
    
    # Relationships
    team = relationship("Team")
    match = relationship("Match")
    
    __table_args__ = (
        Index('ix_elo_team_date', 'team_id', 'rating_date'),
    )


class FeatureStore(Base):
    """
    Pre-computed features for matches.
    
    Speeds up inference by caching feature computation.
    """
    __tablename__ = "feature_store"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, unique=True)
    
    # Team strength features
    home_elo = Column(Float)
    away_elo = Column(Float)
    elo_diff = Column(Float)
    
    # Elo probabilities
    elo_prob_home = Column(Float)
    elo_prob_draw = Column(Float)
    elo_prob_away = Column(Float)
    
    # Form features
    home_form_3 = Column(Float)  # Points from last 3
    home_form_5 = Column(Float)
    home_form_10 = Column(Float)
    away_form_3 = Column(Float)
    away_form_5 = Column(Float)
    away_form_10 = Column(Float)
    
    # Goal features
    home_goals_scored_5 = Column(Float)
    home_goals_conceded_5 = Column(Float)
    away_goals_scored_5 = Column(Float)
    away_goals_conceded_5 = Column(Float)
    
    # Rest days
    home_rest_days = Column(Integer)
    away_rest_days = Column(Integer)
    
    # Head-to-head
    h2h_home_wins = Column(Integer)
    h2h_draws = Column(Integer)
    h2h_away_wins = Column(Integer)
    h2h_total_games = Column(Integer)
    
    # League context
    home_position = Column(Integer)
    away_position = Column(Integer)
    position_diff = Column(Integer)
    
    # Computed at
    computed_at = Column(DateTime, default=func.now())
    
    # Relationships
    match = relationship("Match")


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Manages database operations."""
    
    def __init__(self, db_url: str = DATABASE_URL):
        # PostgreSQL connection with connection pooling
        self.engine = create_engine(
            db_url, 
            echo=False,
            pool_size=10,  # Number of connections to maintain in pool
            max_overflow=20,  # Maximum overflow connections
            pool_pre_ping=True,  # Test connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        self.Session = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(self.engine)
        logger.info("Database tables dropped")
    
    def get_session(self):
        """Get a new database session."""
        return self.Session()


# =============================================================================
# DATA INGESTION
# =============================================================================

class DataIngester:
    """Ingest raw data into database."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def ingest_football_data_uk_csv(self, csv_path: Path) -> int:
        """
        Ingest a football-data.co.uk CSV file.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Number of matches ingested
        """
        logger.info(f"Ingesting {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Skip empty files
        if df.empty or 'HomeTeam' not in df.columns:
            logger.warning(f"Skipping empty or invalid file: {csv_path}")
            return 0
        
        session = self.db.get_session()
        matches_added = 0
        
        try:
            # Get or create league
            league_code = df['league_code'].iloc[0] if 'league_code' in df.columns else csv_path.stem.split('_')[0]
            league_info = LEAGUES_CSV.get(league_code, {"name": league_code, "country": "Unknown"})
            
            league = session.query(League).filter_by(code=league_code).first()
            if not league:
                league = League(
                    code=league_code,
                    name=league_info["name"],
                    country=league_info["country"]
                )
                session.add(league)
                session.flush()
            
            # Process each match
            for _, row in df.iterrows():
                # Skip rows without teams
                if pd.isna(row.get('HomeTeam')) or pd.isna(row.get('AwayTeam')):
                    continue
                
                # Get or create teams
                home_team = self._get_or_create_team(
                    session, 
                    row['HomeTeam'],
                    league_info["country"]
                )
                away_team = self._get_or_create_team(
                    session,
                    row['AwayTeam'],
                    league_info["country"]
                )
                
                # Parse date
                date_str = row.get('Date', '')
                match_date = parse_date(str(date_str))
                
                if match_date is None:
                    continue
                
                # Determine season
                season = row.get('season', get_season_from_date(match_date))
                
                # Check for existing match
                existing = session.query(Match).filter(
                    Match.date == match_date,
                    Match.home_team_id == home_team.id,
                    Match.away_team_id == away_team.id
                ).first()
                
                if existing:
                    continue
                
                # Create match
                home_goals = self._safe_int(row.get('FTHG'))
                away_goals = self._safe_int(row.get('FTAG'))
                result = row.get('FTR')
                
                match = Match(
                    league_id=league.id,
                    season=season,
                    date=match_date,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    result=result if pd.notna(result) else None,
                    result_numeric=encode_result(home_goals, away_goals) if home_goals is not None else None,
                    ht_home_goals=self._safe_int(row.get('HTHG')),
                    ht_away_goals=self._safe_int(row.get('HTAG')),
                    home_shots=self._safe_int(row.get('HS')),
                    away_shots=self._safe_int(row.get('AS')),
                    home_shots_on_target=self._safe_int(row.get('HST')),
                    away_shots_on_target=self._safe_int(row.get('AST')),
                    home_corners=self._safe_int(row.get('HC')),
                    away_corners=self._safe_int(row.get('AC')),
                    home_fouls=self._safe_int(row.get('HF')),
                    away_fouls=self._safe_int(row.get('AF')),
                    home_yellows=self._safe_int(row.get('HY')),
                    away_yellows=self._safe_int(row.get('AY')),
                    home_reds=self._safe_int(row.get('HR')),
                    away_reds=self._safe_int(row.get('AR')),
                    status="finished" if result else "scheduled",
                    source="football_data_uk"
                )
                session.add(match)
                session.flush()  # Flush to get the match ID
                matches_added += 1
                
                # Also store odds if available
                self._store_odds_from_csv(session, match, row)
            
            session.commit()
            logger.info(f"Added {matches_added} matches from {csv_path}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error ingesting {csv_path}: {e}")
            raise
        finally:
            session.close()
        
        return matches_added
    
    def _get_or_create_team(
        self,
        session,
        team_name: str,
        country: str = None
    ) -> Team:
        """Get existing team or create new one."""
        normalized = normalize_team_name(team_name)
        
        team = session.query(Team).filter_by(normalized_name=normalized).first()
        
        if not team:
            team = Team(
                name=team_name,
                normalized_name=normalized,
                country=country,
                football_data_uk_name=team_name
            )
            session.add(team)
            session.flush()
        
        return team
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert to int."""
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float."""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _store_odds_from_csv(self, session, match: Match, row: pd.Series):
        """Store betting odds from CSV row."""
        # Bookmaker columns in football-data.co.uk CSVs
        bookmakers = {
            'B365': ('B365H', 'B365D', 'B365A'),
            'BW': ('BWH', 'BWD', 'BWA'),
            'IW': ('IWH', 'IWD', 'IWA'),
            'PS': ('PSH', 'PSD', 'PSA'),
            'WH': ('WHH', 'WHD', 'WHA'),
            'VC': ('VCH', 'VCD', 'VCA'),
        }
        
        for book_name, (h_col, d_col, a_col) in bookmakers.items():
            home_odds = self._safe_float(row.get(h_col))
            draw_odds = self._safe_float(row.get(d_col))
            away_odds = self._safe_float(row.get(a_col))
            
            if home_odds and draw_odds and away_odds:
                # Calculate implied probabilities
                home_imp = 1 / home_odds
                draw_imp = 1 / draw_odds
                away_imp = 1 / away_odds
                overround = home_imp + draw_imp + away_imp
                
                # Calculate fair probabilities (remove vig)
                home_fair = home_imp / overround
                draw_fair = draw_imp / overround
                away_fair = away_imp / overround
                
                odds_snapshot = OddsSnapshot(
                    match_id=match.id,
                    bookmaker=book_name,
                    home_odds=home_odds,
                    draw_odds=draw_odds,
                    away_odds=away_odds,
                    home_implied=home_imp,
                    draw_implied=draw_imp,
                    away_implied=away_imp,
                    home_fair=home_fair,
                    draw_fair=draw_fair,
                    away_fair=away_fair,
                    overround=overround,
                    snapshot_time=match.date  # Use match date for historical odds
                )
                session.add(odds_snapshot)
    
    def ingest_all_csv_files(self, data_dir: Path = None) -> int:
        """Ingest all CSV files from the data directory."""
        data_dir = data_dir or (RAW_DATA_DIR / "football_data_uk")
        
        total_matches = 0
        
        for csv_file in sorted(data_dir.glob("*.csv")):
            if csv_file.name == "all_matches.csv":
                continue  # Skip combined file
            
            matches = self.ingest_football_data_uk_csv(csv_file)
            total_matches += matches
        
        logger.info(f"Total matches ingested: {total_matches}")
        return total_matches
    
    def ingest_all_csv_files_parallel(self, data_dir: Path = None, num_processes: int = None) -> int:
        """
        Ingest all CSV files from the data directory using multiprocessing.
        
        Args:
            data_dir: Directory containing CSV files
            num_processes: Number of processes to use (defaults to CPU count - 1)
        
        Returns:
            Total number of matches ingested
        """
        data_dir = data_dir or (RAW_DATA_DIR / "football_data_uk")
        
        # Get all CSV files
        csv_files = [f for f in sorted(data_dir.glob("*.csv")) 
                    if f.name != "all_matches.csv"]
        
        if not csv_files:
            logger.warning("No CSV files found to process")
            return 0
        
        # Determine number of processes
        if num_processes is None:
            num_processes = max(1, cpu_count() - 1)
        
        logger.info(f"Processing {len(csv_files)} files using {num_processes} processes")
        start_time = time.time()
        
        # First pass: Extract and deduplicate teams/leagues across all files
        logger.info("Phase 1: Extracting teams and leagues...")
        all_teams, all_leagues = self._extract_all_teams_and_leagues(csv_files, num_processes)
        
        # Insert teams and leagues first
        logger.info("Phase 2: Creating teams and leagues...")
        self._bulk_create_teams_and_leagues(all_teams, all_leagues)
        
        # Second pass: Process matches in parallel
        logger.info("Phase 3: Processing matches in parallel...")
        with Pool(processes=num_processes) as pool:
            # Create a partial function with the database URL
            process_func = partial(_process_csv_file_optimized, DATABASE_URL)
            
            # Process files in parallel
            results = pool.map(process_func, csv_files)
        
        # Sum up total matches
        total_matches = sum(results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Total matches ingested: {total_matches} in {elapsed_time:.2f} seconds")
        logger.info(f"Average: {elapsed_time/len(csv_files):.2f} seconds per file")
        
        return total_matches
    
    def _extract_all_teams_and_leagues(self, csv_files: List[Path], num_processes: int) -> tuple:
        """Extract unique teams and leagues from all CSV files."""
        with Pool(processes=num_processes) as pool:
            results = pool.map(_extract_teams_and_leagues_from_csv, csv_files)
        
        # Combine results
        all_teams = {}
        all_leagues = {}
        
        for teams, leagues in results:
            all_teams.update(teams)
            all_leagues.update(leagues)
        
        return all_teams, all_leagues
    
    def _bulk_create_teams_and_leagues(self, teams: Dict, leagues: Dict):
        """Bulk create teams and leagues in the database."""
        session = self.db.get_session()
        
        try:
            # Create leagues
            for code, info in leagues.items():
                existing = session.query(League).filter_by(code=code).first()
                if not existing:
                    league = League(
                        code=code,
                        name=info["name"],
                        country=info["country"]
                    )
                    session.add(league)
            
            session.commit()
            
            # Create teams
            for normalized_name, info in teams.items():
                existing = session.query(Team).filter_by(normalized_name=normalized_name).first()
                if not existing:
                    team = Team(
                        name=info["name"],
                        normalized_name=normalized_name,
                        country=info["country"],
                        football_data_uk_name=info["name"]
                    )
                    session.add(team)
            
            session.commit()
            logger.info(f"Created {len(leagues)} leagues and {len(teams)} teams")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating teams/leagues: {e}")
            raise
        finally:
            session.close()


# =============================================================================
# DATA EXPORT
# =============================================================================

class DataExporter:
    """Export data for model training."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def export_matches_df(
        self,
        seasons: List[str] = None,
        leagues: List[str] = None,
        include_odds: bool = True
    ) -> pd.DataFrame:
        """
        Export matches as pandas DataFrame for modeling.
        
        Args:
            seasons: Filter by seasons (e.g., ["2023-2024"])
            leagues: Filter by league codes (e.g., ["E0", "SP1"])
            include_odds: Include average market odds
        
        Returns:
            DataFrame with match data
        """
        session = self.db.get_session()
        
        try:
            # Create aliases for home and away teams
            HomeTeam = aliased(Team)
            AwayTeam = aliased(Team)
            
            # Base query
            query = session.query(
                Match.id.label('match_id'),
                Match.date,
                Match.season,
                League.code.label('league_code'),
                League.name.label('league_name'),
                HomeTeam.name.label('home_team'),
                AwayTeam.name.label('away_team'),
                Match.home_goals,
                Match.away_goals,
                Match.result,
                Match.result_numeric,
                Match.home_shots,
                Match.away_shots,
                Match.home_shots_on_target,
                Match.away_shots_on_target,
                Match.home_corners,
                Match.away_corners,
            ).join(
                League, Match.league_id == League.id
            ).join(
                HomeTeam, Match.home_team_id == HomeTeam.id
            ).join(
                AwayTeam, Match.away_team_id == AwayTeam.id
            ).filter(
                Match.status == 'finished',
                Match.result.isnot(None)
            )
            
            # Apply filters
            if seasons:
                query = query.filter(Match.season.in_(seasons))
            if leagues:
                query = query.filter(League.code.in_(leagues))
            
            # Order by date
            query = query.order_by(Match.date)
            
            # Execute and convert to DataFrame
            results = query.all()
            df = pd.DataFrame(results)
            
            if include_odds and not df.empty:
                # Add average market odds
                odds_df = self._get_average_odds(session, df['match_id'].tolist())
                df = df.merge(odds_df, on='match_id', how='left')
            
            logger.info(f"Exported {len(df)} matches")
            return df
            
        finally:
            session.close()
    
    def _get_average_odds(self, session, match_ids: List[int]) -> pd.DataFrame:
        """Get average odds across bookmakers."""
        from sqlalchemy import func as sql_func
        
        results = session.query(
            OddsSnapshot.match_id,
            sql_func.avg(OddsSnapshot.home_odds).label('avg_home_odds'),
            sql_func.avg(OddsSnapshot.draw_odds).label('avg_draw_odds'),
            sql_func.avg(OddsSnapshot.away_odds).label('avg_away_odds'),
            sql_func.avg(OddsSnapshot.home_fair).label('market_prob_home'),
            sql_func.avg(OddsSnapshot.draw_fair).label('market_prob_draw'),
            sql_func.avg(OddsSnapshot.away_fair).label('market_prob_away'),
        ).filter(
            OddsSnapshot.match_id.in_(match_ids)
        ).group_by(
            OddsSnapshot.match_id
        ).all()
        
        return pd.DataFrame(results)
    
    def export_to_csv(
        self,
        output_path: Path = None,
        **kwargs
    ) -> Path:
        """Export matches to CSV file."""
        df = self.export_matches_df(**kwargs)
        
        output_path = output_path or (PROCESSED_DATA_DIR / "matches.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported to {output_path}")
        return output_path


# =============================================================================
# MULTIPROCESSING HELPER FUNCTIONS
# =============================================================================

def _extract_teams_and_leagues_from_csv(csv_path: Path) -> tuple:
    """Extract unique teams and leagues from a CSV file."""
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
        
        if df.empty or 'HomeTeam' not in df.columns:
            return {}, {}
        
        # Extract league info
        league_code = csv_path.stem.split('_')[0]
        league_info = LEAGUES_CSV.get(league_code, {"name": league_code, "country": "Unknown"})
        leagues = {league_code: league_info}
        
        # Extract teams
        teams = {}
        all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
        
        for team_name in all_teams:
            normalized = normalize_team_name(team_name)
            teams[normalized] = {
                "name": team_name,
                "country": league_info["country"]
            }
        
        return teams, leagues
        
    except Exception as e:
        logger.error(f"Error extracting from {csv_path}: {e}")
        return {}, {}


def _process_csv_file_optimized(db_url: str, csv_path: Path) -> int:
    """
    Process a single CSV file with optimized database operations.
    This function runs in a separate process.
    """
    # Create new engine and session for this process
    engine = create_engine(db_url, pool_size=1, max_overflow=0)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        logger.info(f"Processing {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path, encoding='latin-1')
        
        if df.empty or 'HomeTeam' not in df.columns:
            return 0
        
        # Get league
        league_code = csv_path.stem.split('_')[0]
        league = session.query(League).filter_by(code=league_code).first()
        
        if not league:
            logger.error(f"League {league_code} not found!")
            return 0
        
        # Build team lookup
        team_names = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
        team_lookup = {}
        
        for team_name in team_names:
            normalized = normalize_team_name(team_name)
            team = session.query(Team).filter_by(normalized_name=normalized).first()
            if team:
                team_lookup[team_name] = team.id
        
        # Process matches in batches
        matches_data = []
        odds_data = []
        
        for _, row in df.iterrows():
            # Skip invalid rows
            if pd.isna(row.get('HomeTeam')) or pd.isna(row.get('AwayTeam')):
                continue
            
            home_team_id = team_lookup.get(row['HomeTeam'])
            away_team_id = team_lookup.get(row['AwayTeam'])
            
            if not home_team_id or not away_team_id:
                continue
            
            # Parse date
            match_date = parse_date(str(row.get('Date', '')))
            if not match_date:
                continue
            
            # Check if match exists
            existing = session.query(Match).filter(
                Match.date == match_date,
                Match.home_team_id == home_team_id,
                Match.away_team_id == away_team_id
            ).first()
            
            if existing:
                continue
            
            # Prepare match data
            home_goals = _safe_int(row.get('FTHG'))
            away_goals = _safe_int(row.get('FTAG'))
            result = row.get('FTR')
            
            match_dict = {
                'league_id': league.id,
                'season': row.get('season', get_season_from_date(match_date)),
                'date': match_date,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'result': result if pd.notna(result) else None,
                'result_numeric': encode_result(home_goals, away_goals) if home_goals is not None else None,
                'ht_home_goals': _safe_int(row.get('HTHG')),
                'ht_away_goals': _safe_int(row.get('HTAG')),
                'home_shots': _safe_int(row.get('HS')),
                'away_shots': _safe_int(row.get('AS')),
                'home_shots_on_target': _safe_int(row.get('HST')),
                'away_shots_on_target': _safe_int(row.get('AST')),
                'home_corners': _safe_int(row.get('HC')),
                'away_corners': _safe_int(row.get('AC')),
                'home_fouls': _safe_int(row.get('HF')),
                'away_fouls': _safe_int(row.get('AF')),
                'home_yellows': _safe_int(row.get('HY')),
                'away_yellows': _safe_int(row.get('AY')),
                'home_reds': _safe_int(row.get('HR')),
                'away_reds': _safe_int(row.get('AR')),
                'status': "finished" if result else "scheduled",
                'source': "football_data_uk"
            }
            
            matches_data.append(match_dict)
            
            # Prepare odds data (we'll link them after bulk insert)
            odds_row_data = _extract_odds_from_row(row, len(matches_data) - 1, match_date)
            odds_data.extend(odds_row_data)
        
        # Bulk insert matches
        if matches_data:
            session.bulk_insert_mappings(Match, matches_data)
            session.commit()
            
            # Get the inserted match IDs
            # We need to query back to get the IDs for odds
            match_ids = []
            for match_data in matches_data:
                match = session.query(Match).filter(
                    Match.date == match_data['date'],
                    Match.home_team_id == match_data['home_team_id'],
                    Match.away_team_id == match_data['away_team_id']
                ).first()
                if match:
                    match_ids.append(match.id)
            
            # Update odds data with actual match IDs
            for odds_record in odds_data:
                if odds_record['temp_match_idx'] < len(match_ids):
                    odds_record['match_id'] = match_ids[odds_record['temp_match_idx']]
                    del odds_record['temp_match_idx']
            
            # Bulk insert odds
            valid_odds = [o for o in odds_data if 'match_id' in o]
            if valid_odds:
                session.bulk_insert_mappings(OddsSnapshot, valid_odds)
                session.commit()
        
        logger.info(f"Processed {len(matches_data)} matches from {csv_path}")
        return len(matches_data)
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error processing {csv_path}: {e}")
        return 0
    finally:
        session.close()
        engine.dispose()


def _safe_int(value) -> Optional[int]:
    """Safely convert to int."""
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _safe_float(value) -> Optional[float]:
    """Safely convert to float."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _extract_odds_from_row(row: pd.Series, match_idx: int, match_date) -> List[Dict]:
    """Extract odds data from a CSV row."""
    bookmakers = {
        'B365': ('B365H', 'B365D', 'B365A'),
        'BW': ('BWH', 'BWD', 'BWA'),
        'IW': ('IWH', 'IWD', 'IWA'),
        'PS': ('PSH', 'PSD', 'PSA'),
        'WH': ('WHH', 'WHD', 'WHA'),
        'VC': ('VCH', 'VCD', 'VCA'),
    }
    
    odds_records = []
    
    for book_name, (h_col, d_col, a_col) in bookmakers.items():
        home_odds = _safe_float(row.get(h_col))
        draw_odds = _safe_float(row.get(d_col))
        away_odds = _safe_float(row.get(a_col))
        
        if home_odds and draw_odds and away_odds:
            # Calculate implied probabilities
            home_imp = 1 / home_odds
            draw_imp = 1 / draw_odds
            away_imp = 1 / away_odds
            overround = home_imp + draw_imp + away_imp
            
            # Calculate fair probabilities (remove vig)
            home_fair = home_imp / overround
            draw_fair = draw_imp / overround
            away_fair = away_imp / overround
            
            odds_records.append({
                'temp_match_idx': match_idx,
                'bookmaker': book_name,
                'home_odds': home_odds,
                'draw_odds': draw_odds,
                'away_odds': away_odds,
                'home_implied': home_imp,
                'draw_implied': draw_imp,
                'away_implied': away_imp,
                'home_fair': home_fair,
                'draw_fair': draw_fair,
                'away_fair': away_fair,
                'overround': overround,
                'snapshot_time': match_date
            })
    
    return odds_records


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution - setup database and ingest data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Setup and Data Ingestion")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables")
    parser.add_argument("--ingest", action="store_true", help="Ingest CSV data")
    parser.add_argument("--export", action="store_true", help="Export data for modeling")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing for ingestion")
    parser.add_argument("--processes", type=int, help="Number of processes for parallel ingestion")
    
    args = parser.parse_args()
    
    # If no args, run all
    if not any([args.reset, args.ingest, args.export, args.all]):
        args.all = True
    
    # Initialize
    db_manager = DatabaseManager()
    
    # Step 1: Create/reset tables
    if args.reset or args.all:
        logger.info("Creating database tables...")
        if args.reset:
            db_manager.drop_tables()
        db_manager.create_tables()
    
    # Step 2: Ingest data
    if args.ingest or args.all:
        logger.info("Ingesting data from CSV files...")
        ingester = DataIngester(db_manager)
        
        csv_dir = RAW_DATA_DIR / "football_data_uk"
        if csv_dir.exists():
            if args.parallel:
                total = ingester.ingest_all_csv_files_parallel(csv_dir, args.processes)
            else:
                total = ingester.ingest_all_csv_files(csv_dir)
            print(f"\nIngested {total} matches into database")
        else:
            print(f"\nNo data found in {csv_dir}")
            print("Run 01_data_collection.py first to download data")
    
    # Step 3: Export for modeling
    if args.export or args.all:
        logger.info("Exporting data for modeling...")
        exporter = DataExporter(db_manager)
        
        output_path = exporter.export_to_csv()
        print(f"\nExported data to {output_path}")
        
        # Print summary
        df = pd.read_csv(output_path)
        print(f"\nData Summary:")
        print(f"  Total matches: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Leagues: {df['league_code'].nunique()}")
        print(f"  Seasons: {df['season'].nunique()}")
        print(f"\nMatches by league:")
        print(df.groupby('league_code').size().to_string())


if __name__ == "__main__":
    main()
