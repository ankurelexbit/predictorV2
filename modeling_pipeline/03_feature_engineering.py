"""
03 - Feature Engineering
========================

This notebook builds all features needed for model training.

Key feature categories:
1. Team strength (Elo ratings)
2. Form (recent results)
3. Goal scoring/conceding patterns
4. Head-to-head history
5. Rest days
6. League context (position, points)

The Elo system is implemented here as it's both a feature AND a baseline model.

Usage:
    python 03_feature_engineering.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR,
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    ELO_HOME_ADVANTAGE,
    ELO_SEASON_REGRESSION,
    FORM_WINDOWS,
    ROLLING_WINDOWS,
    RANDOM_SEED,
)
from utils import setup_logger, encode_result, set_random_seed

# Setup
logger = setup_logger("feature_engineering")
set_random_seed(RANDOM_SEED)


# =============================================================================
# ELO RATING SYSTEM
# =============================================================================

class EloRatingSystem:
    """
    Elo rating system for football teams.
    
    This is both a feature generator AND a baseline prediction model.
    
    Key parameters:
    - K-factor: Controls how quickly ratings change (higher = more reactive)
    - Home advantage: Added to home team's rating for expected score calculation
    - Season regression: Regress ratings toward mean between seasons
    """
    
    def __init__(
        self,
        initial_rating: float = ELO_INITIAL_RATING,
        k_factor: float = ELO_K_FACTOR,
        home_advantage: float = ELO_HOME_ADVANTAGE,
        season_regression: float = ELO_SEASON_REGRESSION
    ):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.season_regression = season_regression
        
        # Store current ratings
        self.ratings: Dict[str, float] = defaultdict(lambda: initial_rating)
        
        # Store rating history for analysis
        self.rating_history: List[Dict] = []
        
        # Track current season for regression
        self.current_season = None
    
    def get_rating(self, team: str) -> float:
        """Get current Elo rating for a team."""
        return self.ratings[team]
    
    def get_expected_score(
        self,
        home_team: str,
        away_team: str,
        include_home_advantage: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate expected scores (win probabilities).
        
        Returns:
            Tuple of (home_expected, away_expected)
        """
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]
        
        if include_home_advantage:
            home_rating += self.home_advantage
        
        # Standard Elo expected score formula
        exp_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        exp_away = 1 - exp_home
        
        return exp_home, exp_away
    
    def get_win_draw_loss_probs(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float, float]:
        """
        Convert Elo ratings to 1X2 probabilities.
        
        This is a simple approach - more sophisticated methods exist.
        We use empirical draw rate adjustment based on Elo difference.
        """
        home_rating = self.ratings[home_team] + self.home_advantage
        away_rating = self.ratings[away_team]
        
        elo_diff = home_rating - away_rating
        
        # Base expected score
        exp_home = 1 / (1 + 10 ** (-elo_diff / 400))
        
        # Draw probability estimation
        # Higher when teams are closely matched, lower when big gap
        # Empirically, draws happen ~25-27% in top leagues
        base_draw_rate = 0.26
        elo_draw_factor = 1 - (abs(elo_diff) / 800)  # Reduce draws for mismatches
        elo_draw_factor = max(0.5, min(1.2, elo_draw_factor))
        
        p_draw = base_draw_rate * elo_draw_factor
        
        # Allocate remaining probability
        remaining = 1 - p_draw
        p_home = remaining * exp_home
        p_away = remaining * (1 - exp_home)
        
        return p_home, p_draw, p_away
    
    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        match_date: datetime = None,
        season: str = None
    ) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_date: Match date (for history tracking)
            season: Season string (for between-season regression)
        
        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        # Handle season change
        if season and season != self.current_season:
            self._apply_season_regression()
            self.current_season = season
        
        # Get current ratings
        old_home_rating = self.ratings[home_team]
        old_away_rating = self.ratings[away_team]
        
        # Calculate expected scores
        exp_home, exp_away = self.get_expected_score(home_team, away_team)
        
        # Calculate actual scores
        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Goal difference multiplier (rewards convincing wins)
        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            multiplier = 1.0
        elif goal_diff == 2:
            multiplier = 1.5
        else:
            multiplier = (11 + goal_diff) / 8  # ~1.75 for 3, ~2 for 4, etc.
        
        # Calculate rating changes
        home_change = self.k_factor * multiplier * (actual_home - exp_home)
        away_change = self.k_factor * multiplier * (actual_away - exp_away)
        
        # Update ratings
        self.ratings[home_team] = old_home_rating + home_change
        self.ratings[away_team] = old_away_rating + away_change
        
        # Record history
        if match_date:
            self.rating_history.append({
                'date': match_date,
                'team': home_team,
                'rating': self.ratings[home_team],
                'change': home_change
            })
            self.rating_history.append({
                'date': match_date,
                'team': away_team,
                'rating': self.ratings[away_team],
                'change': away_change
            })
        
        return self.ratings[home_team], self.ratings[away_team]
    
    def _apply_season_regression(self):
        """
        Regress ratings toward mean between seasons.
        
        This helps prevent rating explosion and accounts for
        team changes between seasons.
        """
        if not self.ratings:
            return
        
        mean_rating = np.mean(list(self.ratings.values()))
        
        for team in self.ratings:
            old_rating = self.ratings[team]
            new_rating = old_rating + self.season_regression * (mean_rating - old_rating)
            self.ratings[team] = new_rating
        
        logger.info(f"Applied season regression (mean={mean_rating:.1f})")
    
    def get_all_ratings(self) -> Dict[str, float]:
        """Get dictionary of all current ratings."""
        return dict(self.ratings)
    
    def get_rating_history_df(self) -> pd.DataFrame:
        """Get rating history as DataFrame."""
        return pd.DataFrame(self.rating_history)


# =============================================================================
# FEATURE CALCULATORS
# =============================================================================

class FormCalculator:
    """Calculate team form (recent results)."""
    
    def __init__(self, windows: List[int] = FORM_WINDOWS):
        self.windows = windows
        # Store recent matches per team: {team: [(date, home_away, result, goals_for, goals_against), ...]}
        self.team_matches: Dict[str, List] = defaultdict(list)
    
    def record_match(
        self,
        team: str,
        date: datetime,
        is_home: bool,
        goals_for: int,
        goals_against: int,
        result: str  # H, D, A (from team's perspective: W, D, L)
    ):
        """Record a match result for a team."""
        # Convert result to team perspective
        if is_home:
            team_result = result
        else:
            # Flip result for away team
            team_result = {'H': 'L', 'D': 'D', 'A': 'W'}.get(result, result)
        
        self.team_matches[team].append({
            'date': date,
            'is_home': is_home,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'result': team_result,
            'points': {'W': 3, 'D': 1, 'L': 0, 'H': 3 if is_home else 0, 'A': 0 if is_home else 3}.get(team_result, 0)
        })
        
        # Keep sorted by date
        self.team_matches[team].sort(key=lambda x: x['date'])
    
    def get_form(
        self,
        team: str,
        before_date: datetime,
        n_matches: int
    ) -> Dict[str, Any]:
        """
        Get form statistics for last N matches before a date.
        
        Returns:
            Dict with points, wins, draws, losses, goals_for, goals_against
        """
        # Get matches before date
        matches = [m for m in self.team_matches.get(team, []) if m['date'] < before_date]
        
        # Take last N
        recent = matches[-n_matches:] if len(matches) >= n_matches else matches
        
        if not recent:
            return {
                'points': None,
                'wins': None,
                'draws': None,
                'losses': None,
                'goals_for': None,
                'goals_against': None,
                'goal_diff': None,
                'matches_played': 0,
                'ppg': None,
                'goals_for_avg': None,
                'goals_against_avg': None
            }
        
        points = sum(m['points'] for m in recent)
        wins = sum(1 for m in recent if m['result'] == 'W' or (m['is_home'] and m['result'] == 'H') or (not m['is_home'] and m['result'] == 'A'))
        draws = sum(1 for m in recent if m['result'] == 'D')
        losses = len(recent) - wins - draws
        goals_for = sum(m['goals_for'] for m in recent)
        goals_against = sum(m['goals_against'] for m in recent)
        
        return {
            'points': points,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_diff': goals_for - goals_against,
            'matches_played': len(recent),
            'ppg': points / len(recent) if recent else None,  # Points per game
            'goals_for_avg': goals_for / len(recent) if recent else None,
            'goals_against_avg': goals_against / len(recent) if recent else None
        }
    
    def get_rest_days(self, team: str, match_date: datetime) -> Optional[int]:
        """Get days since team's last match."""
        matches = [m for m in self.team_matches.get(team, []) if m['date'] < match_date]
        
        if not matches:
            return None
        
        last_match = matches[-1]
        delta = match_date - last_match['date']
        return delta.days


class HeadToHeadCalculator:
    """Calculate head-to-head statistics."""
    
    def __init__(self):
        # Store H2H results: {(team1, team2): [matches]}
        self.h2h_matches: Dict[Tuple[str, str], List] = defaultdict(list)
    
    def record_match(
        self,
        home_team: str,
        away_team: str,
        date: datetime,
        home_goals: int,
        away_goals: int
    ):
        """Record a head-to-head match."""
        # Store in both directions for easy lookup
        key1 = (home_team, away_team)
        key2 = (away_team, home_team)
        
        match_data = {
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals
        }
        
        self.h2h_matches[key1].append(match_data)
        self.h2h_matches[key2].append(match_data)
    
    def get_h2h_stats(
        self,
        team1: str,
        team2: str,
        before_date: datetime,
        n_matches: int = 10
    ) -> Dict[str, Any]:
        """
        Get head-to-head statistics.
        
        Args:
            team1: First team (perspective team)
            team2: Second team
            before_date: Only consider matches before this date
            n_matches: Maximum matches to consider
        
        Returns:
            Dict with wins, draws, losses from team1's perspective
        """
        key = (team1, team2)
        matches = [m for m in self.h2h_matches.get(key, []) if m['date'] < before_date]
        recent = matches[-n_matches:] if matches else []
        
        if not recent:
            return {
                'h2h_team1_wins': None,
                'h2h_draws': None,
                'h2h_team2_wins': None,
                'h2h_total': 0,
                'h2h_team1_goals': None,
                'h2h_team2_goals': None,
                'h2h_team1_win_rate': None
            }
        
        team1_wins = 0
        team2_wins = 0
        draws = 0
        team1_goals = 0
        team2_goals = 0
        
        for m in recent:
            if m['home_team'] == team1:
                team1_goals += m['home_goals']
                team2_goals += m['away_goals']
                if m['home_goals'] > m['away_goals']:
                    team1_wins += 1
                elif m['home_goals'] < m['away_goals']:
                    team2_wins += 1
                else:
                    draws += 1
            else:  # team1 was away
                team1_goals += m['away_goals']
                team2_goals += m['home_goals']
                if m['away_goals'] > m['home_goals']:
                    team1_wins += 1
                elif m['away_goals'] < m['home_goals']:
                    team2_wins += 1
                else:
                    draws += 1
        
        return {
            'h2h_team1_wins': team1_wins,
            'h2h_draws': draws,
            'h2h_team2_wins': team2_wins,
            'h2h_total': len(recent),
            'h2h_team1_goals': team1_goals,
            'h2h_team2_goals': team2_goals,
            'h2h_team1_win_rate': team1_wins / len(recent) if recent else None
        }


class LeagueTableCalculator:
    """Track league standings."""
    
    def __init__(self):
        # {(league, season): {team: {points, gd, gf, ga, played}}}
        self.tables: Dict[Tuple[str, str], Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {'points': 0, 'gd': 0, 'gf': 0, 'ga': 0, 'played': 0})
        )
    
    def record_match(
        self,
        league: str,
        season: str,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int
    ):
        """Record match result in league table."""
        key = (league, season)
        
        # Update goals
        self.tables[key][home_team]['gf'] += home_goals
        self.tables[key][home_team]['ga'] += away_goals
        self.tables[key][home_team]['gd'] += (home_goals - away_goals)
        self.tables[key][home_team]['played'] += 1
        
        self.tables[key][away_team]['gf'] += away_goals
        self.tables[key][away_team]['ga'] += home_goals
        self.tables[key][away_team]['gd'] += (away_goals - home_goals)
        self.tables[key][away_team]['played'] += 1
        
        # Update points
        if home_goals > away_goals:
            self.tables[key][home_team]['points'] += 3
        elif home_goals < away_goals:
            self.tables[key][away_team]['points'] += 3
        else:
            self.tables[key][home_team]['points'] += 1
            self.tables[key][away_team]['points'] += 1
    
    def get_position(
        self,
        league: str,
        season: str,
        team: str
    ) -> Optional[int]:
        """Get team's current league position."""
        key = (league, season)
        
        if team not in self.tables[key]:
            return None
        
        # Sort teams by points, then goal difference
        sorted_teams = sorted(
            self.tables[key].items(),
            key=lambda x: (x[1]['points'], x[1]['gd'], x[1]['gf']),
            reverse=True
        )
        
        for i, (t, _) in enumerate(sorted_teams, 1):
            if t == team:
                return i
        
        return None
    
    def get_team_stats(
        self,
        league: str,
        season: str,
        team: str
    ) -> Dict[str, Any]:
        """Get team's current league statistics."""
        key = (league, season)
        stats = self.tables[key].get(team, {})
        
        return {
            'league_points': stats.get('points'),
            'league_gd': stats.get('gd'),
            'league_gf': stats.get('gf'),
            'league_ga': stats.get('ga'),
            'league_played': stats.get('played'),
            'league_position': self.get_position(league, season, team)
        }


# =============================================================================
# MAIN FEATURE BUILDER
# =============================================================================

class FeatureBuilder:
    """
    Main feature engineering pipeline.
    
    Processes matches chronologically and computes all features.
    """
    
    def __init__(self):
        self.elo_system = EloRatingSystem()
        self.form_calc = FormCalculator()
        self.h2h_calc = HeadToHeadCalculator()
        self.table_calc = LeagueTableCalculator()
    
    def build_features(
        self,
        matches_df: pd.DataFrame,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Build features for all matches.
        
        IMPORTANT: Features are computed using only data available
        BEFORE each match (no data leakage).
        
        Args:
            matches_df: DataFrame with match data
            include_target: Include result column for training
        
        Returns:
            DataFrame with features for each match
        """
        logger.info(f"Building features for {len(matches_df)} matches")
        
        # Sort by date (critical for time-series consistency)
        df = matches_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        features_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building features"):
            # Extract match info
            match_date = row['date']
            home_team = row['home_team']
            away_team = row['away_team']
            league = row.get('league_code', 'UNK')
            season = row.get('season', 'UNK')
            
            # -----------------------------
            # COMPUTE FEATURES (before match)
            # -----------------------------
            features = {
                'match_id': row.get('match_id', idx),
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'league_code': league,
                'season': season,
            }
            
            # 1. Elo ratings (before update)
            features['home_elo'] = self.elo_system.get_rating(home_team)
            features['away_elo'] = self.elo_system.get_rating(away_team)
            features['elo_diff'] = features['home_elo'] - features['away_elo'] + ELO_HOME_ADVANTAGE
            
            # Elo-based probabilities
            p_h, p_d, p_a = self.elo_system.get_win_draw_loss_probs(home_team, away_team)
            features['elo_prob_home'] = p_h
            features['elo_prob_draw'] = p_d
            features['elo_prob_away'] = p_a
            
            # 2. Form features
            for window in FORM_WINDOWS:
                home_form = self.form_calc.get_form(home_team, match_date, window)
                away_form = self.form_calc.get_form(away_team, match_date, window)
                
                features[f'home_form_{window}_points'] = home_form['points']
                features[f'home_form_{window}_ppg'] = home_form['ppg']
                features[f'home_form_{window}_gf'] = home_form['goals_for_avg']
                features[f'home_form_{window}_ga'] = home_form['goals_against_avg']
                
                features[f'away_form_{window}_points'] = away_form['points']
                features[f'away_form_{window}_ppg'] = away_form['ppg']
                features[f'away_form_{window}_gf'] = away_form['goals_for_avg']
                features[f'away_form_{window}_ga'] = away_form['goals_against_avg']
            
            # 3. Rest days
            features['home_rest_days'] = self.form_calc.get_rest_days(home_team, match_date)
            features['away_rest_days'] = self.form_calc.get_rest_days(away_team, match_date)
            if features['home_rest_days'] and features['away_rest_days']:
                features['rest_diff'] = features['home_rest_days'] - features['away_rest_days']
            else:
                features['rest_diff'] = None
            
            # 4. Head-to-head
            h2h = self.h2h_calc.get_h2h_stats(home_team, away_team, match_date)
            features['h2h_home_wins'] = h2h['h2h_team1_wins']
            features['h2h_draws'] = h2h['h2h_draws']
            features['h2h_away_wins'] = h2h['h2h_team2_wins']
            features['h2h_total'] = h2h['h2h_total']
            features['h2h_home_win_rate'] = h2h['h2h_team1_win_rate']
            
            # 5. League position
            home_stats = self.table_calc.get_team_stats(league, season, home_team)
            away_stats = self.table_calc.get_team_stats(league, season, away_team)
            
            features['home_position'] = home_stats['league_position']
            features['away_position'] = away_stats['league_position']
            features['home_league_points'] = home_stats['league_points']
            features['away_league_points'] = away_stats['league_points']
            
            if home_stats['league_position'] and away_stats['league_position']:
                features['position_diff'] = away_stats['league_position'] - home_stats['league_position']
            else:
                features['position_diff'] = None
            
            # 6. Market odds (if available)
            if 'market_prob_home' in row and pd.notna(row['market_prob_home']):
                features['market_prob_home'] = row['market_prob_home']
                features['market_prob_draw'] = row['market_prob_draw']
                features['market_prob_away'] = row['market_prob_away']
                features['avg_home_odds'] = row.get('avg_home_odds')
                features['avg_draw_odds'] = row.get('avg_draw_odds')
                features['avg_away_odds'] = row.get('avg_away_odds')
            
            # -----------------------------
            # TARGET VARIABLE
            # -----------------------------
            if include_target:
                features['result'] = row.get('result')
                features['result_numeric'] = row.get('result_numeric')
                features['home_goals'] = row.get('home_goals')
                features['away_goals'] = row.get('away_goals')
            
            features_list.append(features)
            
            # -----------------------------
            # UPDATE STATE (after computing features)
            # -----------------------------
            home_goals = row.get('home_goals')
            away_goals = row.get('away_goals')
            result = row.get('result')
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                # Update Elo
                self.elo_system.update_ratings(
                    home_team, away_team,
                    int(home_goals), int(away_goals),
                    match_date, season
                )
                
                # Update form
                self.form_calc.record_match(
                    home_team, match_date, True,
                    int(home_goals), int(away_goals), result
                )
                self.form_calc.record_match(
                    away_team, match_date, False,
                    int(away_goals), int(home_goals), result
                )
                
                # Update H2H
                self.h2h_calc.record_match(
                    home_team, away_team, match_date,
                    int(home_goals), int(away_goals)
                )
                
                # Update league table
                self.table_calc.record_match(
                    league, season,
                    home_team, away_team,
                    int(home_goals), int(away_goals)
                )
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"Built {len(features_df)} feature rows with {len(features_df.columns)} columns")
        
        return features_df
    
    def get_current_elo_ratings(self) -> pd.DataFrame:
        """Get current Elo ratings for all teams."""
        ratings = self.elo_system.get_all_ratings()
        df = pd.DataFrame([
            {'team': team, 'elo_rating': rating}
            for team, rating in ratings.items()
        ])
        return df.sort_values('elo_rating', ascending=False).reset_index(drop=True)
    
    def get_elo_history(self) -> pd.DataFrame:
        """Get Elo rating history."""
        return self.elo_system.get_rating_history_df()


# =============================================================================
# FEATURE SELECTION & PREPROCESSING
# =============================================================================

def get_feature_columns() -> List[str]:
    """Get list of feature columns for modeling."""
    features = [
        # Elo features
        'home_elo', 'away_elo', 'elo_diff',
        'elo_prob_home', 'elo_prob_draw', 'elo_prob_away',
        
        # Form features (using 5-match window as primary)
        'home_form_5_ppg', 'away_form_5_ppg',
        'home_form_5_gf', 'away_form_5_gf',
        'home_form_5_ga', 'away_form_5_ga',
        
        # Rest days
        'home_rest_days', 'away_rest_days', 'rest_diff',
        
        # Head-to-head
        'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
        'h2h_home_win_rate',
        
        # League position
        'home_position', 'away_position', 'position_diff',
    ]
    
    return features


def prepare_model_data(
    features_df: pd.DataFrame,
    feature_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        features_df: DataFrame with all features
        feature_cols: Columns to use as features
    
    Returns:
        X (features), y (target)
    """
    feature_cols = feature_cols or get_feature_columns()
    
    # Filter to rows with target
    df = features_df.dropna(subset=['result_numeric']).copy()
    
    # Select features
    X = df[feature_cols].copy()
    y = df['result_numeric'].astype(int)
    
    # Handle missing values
    # For numeric features, fill with median
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    return X, y


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Build features from processed match data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "matches.csv"),
        help="Input matches CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DATA_DIR / "features.csv"),
        help="Output features CSV"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    matches_df = pd.read_csv(args.input)
    print(f"\nLoaded {len(matches_df)} matches")
    
    # Build features
    builder = FeatureBuilder()
    features_df = builder.build_features(matches_df)
    
    # Save features
    output_path = Path(args.output)
    features_df.to_csv(output_path, index=False)
    print(f"\nSaved features to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Total matches: {len(features_df)}")
    print(f"Features: {len(features_df.columns)}")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    
    # Print feature columns
    print("\nFeature columns:")
    for col in get_feature_columns():
        non_null = features_df[col].notna().sum()
        pct = 100 * non_null / len(features_df)
        print(f"  {col}: {pct:.1f}% non-null")
    
    # Save Elo ratings
    elo_df = builder.get_current_elo_ratings()
    elo_path = PROCESSED_DATA_DIR / "elo_ratings.csv"
    elo_df.to_csv(elo_path, index=False)
    print(f"\nSaved Elo ratings to {elo_path}")
    
    # Print top/bottom Elo teams
    print("\nTop 10 teams by Elo:")
    print(elo_df.head(10).to_string(index=False))
    
    print("\nBottom 10 teams by Elo:")
    print(elo_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
