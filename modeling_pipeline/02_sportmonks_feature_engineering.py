#!/usr/bin/env python3
"""
Sportmonks Feature Engineering Pipeline

This script processes raw Sportmonks data and creates features for match prediction models.
It is designed to be the primary feature engineering pipeline for all model training.

Features include:
- Team form (rolling goals, shots, xG approximations)
- Elo ratings
- Attack/defense strength metrics
- Passing and possession statistics
- Injury/sidelined player counts
- Market odds features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_engineering')

# Paths
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw' / 'sportmonks'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STATISTIC TYPE ID MAPPING
# ============================================================
STAT_TYPE_MAP = {
    # Shooting
    34: 'corners',
    41: 'shots_off_target',
    42: 'shots_total',
    49: 'shots_insidebox',
    50: 'shots_outsidebox',
    52: 'stat_goals',  # Renamed to avoid conflict with home_goals/away_goals columns
    54: 'goal_attempts',
    58: 'shots_blocked',
    64: 'hit_woodwork',
    86: 'shots_on_target',

    # Attacks
    43: 'attacks',
    44: 'dangerous_attacks',
    580: 'big_chances_created',
    581: 'big_chances_missed',
    1527: 'counter_attacks',

    # Passing
    45: 'possession_pct',
    46: 'ball_safe',
    62: 'long_passes',
    63: 'short_passes',
    80: 'passes',
    81: 'successful_passes',
    82: 'successful_passes_pct',
    116: 'accurate_passes',
    117: 'key_passes',
    122: 'long_balls',
    123: 'long_balls_won',
    124: 'through_balls',
    125: 'through_balls_won',
    27264: 'successful_long_passes',
    27265: 'successful_long_passes_pct',

    # Crossing
    98: 'total_crosses',
    99: 'accurate_crosses',

    # Defense
    47: 'penalties',
    51: 'offsides',
    53: 'goal_kicks',
    55: 'free_kicks',
    56: 'fouls',
    57: 'saves',
    60: 'throwins',
    78: 'tackles',
    100: 'interceptions',
    101: 'clearances',
    102: 'clearances_won',
    103: 'punches',
    104: 'saves_insidebox',
    105: 'total_duels',
    106: 'duels_won',
    107: 'aerials_won',

    # Dribbling
    108: 'dribble_attempts',
    109: 'successful_dribbles',
    110: 'dribbled_past',
    1605: 'successful_dribbles_pct',

    # Headers
    65: 'successful_headers',
    70: 'headers',

    # Cards
    83: 'redcards',
    84: 'yellowcards',
    85: 'yellowred_cards',

    # Other match stats
    59: 'substitutions',
    94: 'dispossessed',
    95: 'offsides_provoked',
    96: 'fouls_drawn',
    97: 'blocked_shots',
    571: 'error_lead_to_goal',
    575: 'failed_to_score',
    583: 'last_man_tackle',
    584: 'good_high_claim',
    1491: 'duels_lost',
    27266: 'aerials_lost',
    27267: 'tackles_won',
    27273: 'possession_lost',
    27274: 'aerials_total',
    48997: 'error_lead_to_shot',
}

# Player-level statistics (aggregated to team level)
# These come from lineups.details and use home_player_stat_* / away_player_stat_* columns
PLAYER_STAT_TYPE_MAP = {
    # Defensive
    101: 'clearances',
    100: 'interceptions',
    97: 'blocked_shots',
    78: 'tackles',
    27267: 'tackles_won',

    # Aerial
    107: 'aerials_won',
    27266: 'aerials_lost',
    27274: 'aerials_total',

    # Ball control
    120: 'touches',
    27273: 'possession_lost',
    94: 'dispossessed',
    96: 'fouls_drawn',

    # Duels
    105: 'total_duels',
    106: 'duels_won',
    1491: 'duels_lost',

    # Passing
    116: 'accurate_passes',
    80: 'passes',
    117: 'key_passes',
    122: 'long_balls',
    123: 'long_balls_won',

    # Dribbling
    108: 'dribble_attempts',
    109: 'successful_dribbles',
    110: 'dribbled_past',

    # Shooting
    41: 'shots_off_target',
    42: 'shots_total',
    86: 'shots_on_target',
    58: 'shots_blocked',

    # Goalkeeper
    104: 'saves_insidebox',
    88: 'goals_conceded',
    57: 'saves',
    584: 'good_high_claim',
    583: 'last_man_tackle',

    # Performance
    118: 'rating',
    119: 'minutes_played',
    79: 'assists',

    # Other
    580: 'big_chances_created',
    581: 'big_chances_missed',
    48997: 'error_lead_to_shot',
}

# Event type mapping
EVENT_TYPE_MAP = {
    14: 'goal',
    15: 'own_goal',
    16: 'penalty',
    17: 'missed_penalty',
    18: 'substitution',
    19: 'yellowcard',
    20: 'redcard',
    21: 'yellowred_card',
}

# Sidelined type mapping (common injury/suspension types)
SIDELINED_TYPE_MAP = {
    536: 'injury',
    537: 'illness',
    549: 'injury',
    551: 'injury',
    587: 'suspension',
    742: 'injury',
    1682: 'unknown',
    1918: 'injury',
    1945: 'injury',
    1991: 'injury',
}

# ============================================================
# ELO RATING PARAMETERS
# ============================================================
ELO_K_FACTOR = 32
ELO_HOME_ADVANTAGE = 50
ELO_INITIAL_RATING = 1500

# ============================================================
# ROLLING WINDOW SIZES
# ============================================================
ROLLING_WINDOWS = [3, 5, 10]  # Last N matches


def load_raw_data():
    """Load all raw Sportmonks data files."""
    logger.info("Loading raw Sportmonks data...")

    fixtures = pd.read_csv(RAW_DATA_DIR / 'fixtures.csv')
    fixtures['date'] = pd.to_datetime(fixtures['date'])

    lineups = pd.read_csv(RAW_DATA_DIR / 'lineups.csv')
    events = pd.read_csv(RAW_DATA_DIR / 'events.csv')
    sidelined = pd.read_csv(RAW_DATA_DIR / 'sidelined.csv')
    standings = pd.read_csv(RAW_DATA_DIR / 'standings.csv')

    logger.info(f"  Fixtures: {len(fixtures)} rows")
    logger.info(f"  Lineups: {len(lineups)} rows")
    logger.info(f"  Events: {len(events)} rows")
    logger.info(f"  Sidelined: {len(sidelined)} rows")
    logger.info(f"  Standings: {len(standings)} rows")

    return fixtures, lineups, events, sidelined, standings


def rename_stat_columns(df):
    """Rename stat columns from IDs to meaningful names."""
    logger.info("Renaming statistic columns...")

    # Build rename mapping for team-level stats
    rename_map = {}
    for col in df.columns:
        # Team-level stats: home_stat_XX / away_stat_XX
        if col.startswith('home_stat_') or col.startswith('away_stat_'):
            prefix = col.split('_stat_')[0]  # 'home' or 'away'
            try:
                stat_id = int(col.split('_stat_')[1])
                if stat_id in STAT_TYPE_MAP:
                    new_name = f"{prefix}_{STAT_TYPE_MAP[stat_id]}"
                    rename_map[col] = new_name
            except ValueError:
                pass

        # Player-level stats (aggregated): home_player_stat_XX / away_player_stat_XX
        # ALWAYS use 'player_' prefix to avoid conflicts
        elif col.startswith('home_player_stat_') or col.startswith('away_player_stat_'):
            prefix = col.split('_player_stat_')[0]  # 'home' or 'away'
            try:
                stat_id = int(col.split('_player_stat_')[1])
                if stat_id in PLAYER_STAT_TYPE_MAP:
                    base_name = PLAYER_STAT_TYPE_MAP[stat_id]
                    # Always use player_ prefix to distinguish from team-level stats
                    new_name = f"{prefix}_player_{base_name}"
                    rename_map[col] = new_name
            except ValueError:
                pass

    df = df.rename(columns=rename_map)
    logger.info(f"  Renamed {len(rename_map)} columns (team + player-level)")

    return df


def calculate_xg_approximation(df):
    """
    Approximate xG from available shot statistics.

    xG formula based on shot location and type:
    - Shots inside box: ~0.12 xG each
    - Shots outside box: ~0.03 xG each
    - Big chances: ~0.35 xG each
    """
    logger.info("Calculating xG approximations...")

    # Reset index to avoid duplicate index issues
    df = df.reset_index(drop=True)

    # xG weights
    XG_INSIDE_BOX = 0.12
    XG_OUTSIDE_BOX = 0.03
    XG_BIG_CHANCE = 0.35

    for prefix in ['home', 'away']:
        inside_col = f'{prefix}_shots_insidebox'
        outside_col = f'{prefix}_shots_outsidebox'
        big_chance_col = f'{prefix}_big_chances_created'

        # Calculate xG from each component separately
        xg_inside = df[inside_col].fillna(0).values * XG_INSIDE_BOX if inside_col in df.columns else 0
        xg_outside = df[outside_col].fillna(0).values * XG_OUTSIDE_BOX if outside_col in df.columns else 0
        xg_big_chance = df[big_chance_col].fillna(0).values * (XG_BIG_CHANCE - XG_INSIDE_BOX) if big_chance_col in df.columns else 0

        # Sum them
        if isinstance(xg_inside, (int, float)):
            total_xg = xg_inside + xg_outside + xg_big_chance
        else:
            total_xg = xg_inside + xg_outside + xg_big_chance

        df[f'{prefix}_xg'] = total_xg

    return df


def calculate_elo_ratings(df):
    """Calculate Elo ratings for all teams."""
    logger.info("Calculating Elo ratings...")

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Initialize Elo ratings
    elo_ratings = {}

    home_elos = []
    away_elos = []

    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']

        # Get current ratings (or initialize)
        home_elo = elo_ratings.get(home_team, ELO_INITIAL_RATING)
        away_elo = elo_ratings.get(away_team, ELO_INITIAL_RATING)

        # Store pre-match ratings
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        # Skip if match not completed
        if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
            continue

        # Calculate expected scores (with home advantage)
        home_expected = 1 / (1 + 10 ** ((away_elo - home_elo - ELO_HOME_ADVANTAGE) / 400))
        away_expected = 1 - home_expected

        # Actual result
        if row['home_goals'] > row['away_goals']:
            home_actual = 1
            away_actual = 0
        elif row['home_goals'] < row['away_goals']:
            home_actual = 0
            away_actual = 1
        else:
            home_actual = 0.5
            away_actual = 0.5

        # Update ratings
        goal_diff = abs(row['home_goals'] - row['away_goals'])
        k_multiplier = 1 + (goal_diff - 1) * 0.1 if goal_diff > 1 else 1
        k = ELO_K_FACTOR * k_multiplier

        elo_ratings[home_team] = home_elo + k * (home_actual - home_expected)
        elo_ratings[away_team] = away_elo + k * (away_actual - away_expected)

    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    df['elo_diff'] = df['home_elo'] - df['away_elo']

    logger.info(f"  Calculated Elo for {len(elo_ratings)} teams")

    return df


def calculate_rolling_features(df, windows=ROLLING_WINDOWS):
    """Calculate rolling averages for key statistics."""
    logger.info("Calculating rolling features...")

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Key stats to calculate rolling averages for
    # Team-level stats (from statistics include)
    key_stats = [
        'goals', 'xg', 'shots_total', 'shots_on_target', 'possession_pct',
        'dangerous_attacks', 'corners', 'fouls', 'passes', 'successful_passes_pct',
        'big_chances_created', 'tackles', 'interceptions'
    ]

    # Player-level stats (from lineups.details)
    # These are ALWAYS named with 'player_' prefix to avoid conflicts with team stats
    player_stats = [
        'player_clearances', 'player_aerials_won', 'player_touches', 'player_rating',
        'player_total_duels', 'player_duels_won', 'player_possession_lost',
        'player_accurate_passes', 'player_dribble_attempts', 'player_successful_dribbles',
        'player_tackles_won', 'player_long_balls_won', 'player_dispossessed', 'player_fouls_drawn',
        'player_blocked_shots', 'player_duels_lost', 'player_aerials_lost', 'player_aerials_total',
        'player_saves', 'player_goals_conceded', 'player_saves_insidebox', 'player_good_high_claim',
    ]

    # Combine all for rolling calculations
    all_stats = key_stats + player_stats

    # Initialize columns
    for window in windows:
        for stat in all_stats:
            for side in ['home', 'away']:
                df[f'{side}_{stat}_{window}'] = np.nan
                df[f'{side}_{stat}_conceded_{window}'] = np.nan

    # Group by team and calculate rolling stats
    team_history = {}  # team_id -> list of (date, stats_dict, opponent_stats_dict)

    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        match_date = row['date']

        # Get home team's history
        if home_team in team_history:
            home_matches = [m for m in team_history[home_team] if m[0] < match_date]
            for window in windows:
                recent = home_matches[-window:] if len(home_matches) >= window else home_matches
                if recent:
                    for stat in all_stats:
                        col = f'home_{stat}'
                        if col in df.columns:
                            values = [m[1].get(stat, np.nan) for m in recent]
                            df.at[idx, f'home_{stat}_{window}'] = np.nanmean(values)

                        # Conceded stats (what opponents did)
                        conceded_values = [m[2].get(stat, np.nan) for m in recent]
                        df.at[idx, f'home_{stat}_conceded_{window}'] = np.nanmean(conceded_values)

        # Get away team's history
        if away_team in team_history:
            away_matches = [m for m in team_history[away_team] if m[0] < match_date]
            for window in windows:
                recent = away_matches[-window:] if len(away_matches) >= window else away_matches
                if recent:
                    for stat in all_stats:
                        col = f'away_{stat}'
                        if col in df.columns:
                            values = [m[1].get(stat, np.nan) for m in recent]
                            df.at[idx, f'away_{stat}_{window}'] = np.nanmean(values)

                        # Conceded stats
                        conceded_values = [m[2].get(stat, np.nan) for m in recent]
                        df.at[idx, f'away_{stat}_conceded_{window}'] = np.nanmean(conceded_values)

        # Skip incomplete matches for history
        if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
            continue

        # Store match stats for future calculations
        home_stats = {}
        away_stats = {}
        for stat in all_stats:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            if home_col in df.columns:
                home_stats[stat] = row[home_col]
            if away_col in df.columns:
                away_stats[stat] = row[away_col]

        # Add to team history
        if home_team not in team_history:
            team_history[home_team] = []
        if away_team not in team_history:
            team_history[away_team] = []

        team_history[home_team].append((match_date, home_stats, away_stats))
        team_history[away_team].append((match_date, away_stats, home_stats))

    logger.info(f"  Calculated rolling features for {len(windows)} windows")

    return df


def add_sidelined_features(df, sidelined):
    """Add injury/sidelined player counts to fixtures."""
    logger.info("Adding sidelined features...")

    # Already have counts in the fixtures data
    # Just ensure they're properly named
    if 'home_sidelined_count' in df.columns:
        df['home_injuries'] = df['home_sidelined_count']
    if 'away_sidelined_count' in df.columns:
        df['away_injuries'] = df['away_sidelined_count']

    # Calculate injury difference
    if 'home_injuries' in df.columns and 'away_injuries' in df.columns:
        df['injury_diff'] = df['home_injuries'] - df['away_injuries']

    return df


def add_standings_features(df, standings):
    """Add league position and points features."""
    logger.info("Adding standings features...")

    # Create season-team lookup
    standings_lookup = {}
    for _, row in standings.iterrows():
        key = (row['season_id'], row['team_id'])
        standings_lookup[key] = {
            'position': row['position'],
            'points': row['points'],
            'played': row['played']
        }

    # Add to fixtures
    home_positions = []
    away_positions = []
    home_points = []
    away_points = []

    for _, row in df.iterrows():
        season_id = row['season_id']
        home_key = (season_id, row['home_team_id'])
        away_key = (season_id, row['away_team_id'])

        home_data = standings_lookup.get(home_key, {})
        away_data = standings_lookup.get(away_key, {})

        home_positions.append(home_data.get('position', np.nan))
        away_positions.append(away_data.get('position', np.nan))
        home_points.append(home_data.get('points', np.nan))
        away_points.append(away_data.get('points', np.nan))

    df['home_position'] = home_positions
    df['away_position'] = away_positions
    df['home_points'] = home_points
    df['away_points'] = away_points
    df['position_diff'] = df['away_position'] - df['home_position']  # Positive = home team ranked higher
    df['points_diff'] = df['home_points'] - df['away_points']

    return df


def add_market_features(df):
    """Add market-derived features."""
    logger.info("Adding market features...")

    if 'odds_home' in df.columns and 'odds_away' in df.columns and 'odds_draw' in df.columns:
        # Convert odds to probabilities
        total = 1/df['odds_home'] + 1/df['odds_draw'] + 1/df['odds_away']
        df['market_prob_home'] = (1/df['odds_home']) / total
        df['market_prob_draw'] = (1/df['odds_draw']) / total
        df['market_prob_away'] = (1/df['odds_away']) / total

        # Market edge features
        df['market_home_away_ratio'] = df['market_prob_home'] / df['market_prob_away']
        df['market_favorite'] = np.where(
            df['market_prob_home'] > df['market_prob_away'], 1,
            np.where(df['market_prob_away'] > df['market_prob_home'], -1, 0)
        )

    return df


def add_contextual_features(df):
    """Add contextual features like round number, season stage, etc."""
    logger.info("Adding contextual features...")

    # Extract round number if available
    if 'round_name' in df.columns:
        df['round_num'] = pd.to_numeric(df['round_name'], errors='coerce')

    # Season progress (0-1)
    if 'round_num' in df.columns:
        df['season_progress'] = df['round_num'] / 38  # Assuming 38 matches per season

    # Is early season (first 5 rounds)
    df['is_early_season'] = (df.get('round_num', 99) <= 5).astype(int)

    # Day of week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df


def calculate_form_features(df):
    """Calculate team form features (wins/draws/losses in last N games)."""
    logger.info("Calculating form features...")

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Track results
    team_results = {}  # team_id -> list of (date, result, opponent_id)

    # Initialize columns
    for window in [3, 5]:
        df[f'home_wins_{window}'] = np.nan
        df[f'home_draws_{window}'] = np.nan
        df[f'home_losses_{window}'] = np.nan
        df[f'away_wins_{window}'] = np.nan
        df[f'away_draws_{window}'] = np.nan
        df[f'away_losses_{window}'] = np.nan
        df[f'home_form_{window}'] = np.nan  # Points from last N
        df[f'away_form_{window}'] = np.nan

    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        match_date = row['date']

        # Calculate form for home team
        if home_team in team_results:
            for window in [3, 5]:
                recent = [r for r in team_results[home_team] if r[0] < match_date][-window:]
                if recent:
                    wins = sum(1 for r in recent if r[1] == 'W')
                    draws = sum(1 for r in recent if r[1] == 'D')
                    losses = sum(1 for r in recent if r[1] == 'L')
                    df.at[idx, f'home_wins_{window}'] = wins
                    df.at[idx, f'home_draws_{window}'] = draws
                    df.at[idx, f'home_losses_{window}'] = losses
                    df.at[idx, f'home_form_{window}'] = wins * 3 + draws

        # Calculate form for away team
        if away_team in team_results:
            for window in [3, 5]:
                recent = [r for r in team_results[away_team] if r[0] < match_date][-window:]
                if recent:
                    wins = sum(1 for r in recent if r[1] == 'W')
                    draws = sum(1 for r in recent if r[1] == 'D')
                    losses = sum(1 for r in recent if r[1] == 'L')
                    df.at[idx, f'away_wins_{window}'] = wins
                    df.at[idx, f'away_draws_{window}'] = draws
                    df.at[idx, f'away_losses_{window}'] = losses
                    df.at[idx, f'away_form_{window}'] = wins * 3 + draws

        # Skip incomplete matches
        if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
            continue

        # Determine results
        if row['home_goals'] > row['away_goals']:
            home_result, away_result = 'W', 'L'
        elif row['home_goals'] < row['away_goals']:
            home_result, away_result = 'L', 'W'
        else:
            home_result, away_result = 'D', 'D'

        # Store results
        if home_team not in team_results:
            team_results[home_team] = []
        if away_team not in team_results:
            team_results[away_team] = []

        team_results[home_team].append((match_date, home_result, away_team))
        team_results[away_team].append((match_date, away_result, home_team))

    # Form difference
    for window in [3, 5]:
        df[f'form_diff_{window}'] = df[f'home_form_{window}'] - df[f'away_form_{window}']

    return df


def add_head_to_head_features(df):
    """Add head-to-head historical features."""
    logger.info("Adding head-to-head features...")

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Track H2H history
    h2h_history = {}  # frozenset({team1, team2}) -> list of (date, home_team, home_goals, away_goals)

    df['h2h_home_wins'] = np.nan
    df['h2h_away_wins'] = np.nan
    df['h2h_draws'] = np.nan
    df['h2h_home_goals_avg'] = np.nan
    df['h2h_away_goals_avg'] = np.nan

    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        match_date = row['date']

        key = frozenset({home_team, away_team})

        if key in h2h_history:
            past_matches = [m for m in h2h_history[key] if m[0] < match_date]
            if past_matches:
                home_wins = 0
                away_wins = 0
                draws = 0
                home_goals = []
                away_goals = []

                for _, h_team, hg, ag in past_matches:
                    # Normalize to current match perspective
                    if h_team == home_team:
                        if hg > ag:
                            home_wins += 1
                        elif ag > hg:
                            away_wins += 1
                        else:
                            draws += 1
                        home_goals.append(hg)
                        away_goals.append(ag)
                    else:
                        # Flip perspective
                        if ag > hg:
                            home_wins += 1
                        elif hg > ag:
                            away_wins += 1
                        else:
                            draws += 1
                        home_goals.append(ag)
                        away_goals.append(hg)

                df.at[idx, 'h2h_home_wins'] = home_wins
                df.at[idx, 'h2h_away_wins'] = away_wins
                df.at[idx, 'h2h_draws'] = draws
                df.at[idx, 'h2h_home_goals_avg'] = np.mean(home_goals)
                df.at[idx, 'h2h_away_goals_avg'] = np.mean(away_goals)

        # Skip incomplete matches
        if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
            continue

        # Store for future
        if key not in h2h_history:
            h2h_history[key] = []
        h2h_history[key].append((match_date, home_team, row['home_goals'], row['away_goals']))

    return df


def calculate_attack_defense_strength(df):
    """Calculate attack and defense strength metrics."""
    logger.info("Calculating attack/defense strength...")

    # Use rolling xG and goals conceded
    for window in [5, 10]:
        xg_col = f'home_xg_{window}'
        conc_col = f'home_goals_conceded_{window}'

        if xg_col in df.columns:
            # Normalize by league average (approximate)
            league_avg_xg = 1.3  # Typical EPL average
            df[f'home_attack_strength_{window}'] = df[xg_col] / league_avg_xg
            df[f'away_attack_strength_{window}'] = df[f'away_xg_{window}'] / league_avg_xg

        if conc_col in df.columns:
            df[f'home_defense_strength_{window}'] = df[conc_col] / league_avg_xg
            df[f'away_defense_strength_{window}'] = df[f'away_goals_conceded_{window}'] / league_avg_xg

    return df


def create_target_variable(df):
    """Create target variable for classification."""
    logger.info("Creating target variable...")

    # Result: 0 = Away win, 1 = Draw, 2 = Home win
    df['target'] = np.where(
        df['home_goals'] > df['away_goals'], 2,
        np.where(df['home_goals'] < df['away_goals'], 0, 1)
    )

    # Alternative: binary targets
    df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
    df['draw'] = (df['home_goals'] == df['away_goals']).astype(int)
    df['away_win'] = (df['home_goals'] < df['away_goals']).astype(int)

    return df


def calculate_ema_features(df, alpha=0.1):
    """Calculate Exponential Moving Averages for key stats.
    
    EMA gives more weight to recent matches vs hard cutoffs.
    
    Args:
        df: DataFrame with match data
        alpha: Decay factor (0.1 = 10% weight to new value)
    
    Returns:
        DataFrame with EMA features added
    """
    logger.info("Calculating EMA features...")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Stats to calculate EMA for
    stats = ['goals', 'xg', 'shots_total', 'shots_on_target', 'possession_pct']
    
    # Initialize EMA columns
    for stat in stats:
        df[f'home_{stat}_ema'] = np.nan
        df[f'away_{stat}_ema'] = np.nan
        df[f'home_{stat}_conceded_ema'] = np.nan
        df[f'away_{stat}_conceded_ema'] = np.nan
    
    # Track EMA by team
    team_ema = {}  # team_id -> {stat: ema_value}
    
    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        
        # Initialize if first match
        if home_team not in team_ema:
            team_ema[home_team] = {f'{s}': 0 for s in stats}
            team_ema[home_team].update({f'{s}_conceded': 0 for s in stats})
        if away_team not in team_ema:
            team_ema[away_team] = {f'{s}': 0 for s in stats}
            team_ema[away_team].update({f'{s}_conceded': 0 for s in stats})
        
        # Store current EMA (before updating)
        for stat in stats:
            df.at[idx, f'home_{stat}_ema'] = team_ema[home_team][stat]
            df.at[idx, f'away_{stat}_ema'] = team_ema[away_team][stat]
            df.at[idx, f'home_{stat}_conceded_ema'] = team_ema[home_team][f'{stat}_conceded']
            df.at[idx, f'away_{stat}_conceded_ema'] = team_ema[away_team][f'{stat}_conceded']
        
        # Skip if match incomplete
        if pd.isna(row.get('home_goals')) or pd.isna(row.get('away_goals')):
            continue
        
        # Update EMA with current match
        for stat in stats:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            
            if home_col in row and not pd.isna(row[home_col]):
                # Home team's own stat
                team_ema[home_team][stat] = (alpha * row[home_col] + 
                                             (1 - alpha) * team_ema[home_team][stat])
                # Away team conceded this stat
                team_ema[away_team][f'{stat}_conceded'] = (alpha * row[home_col] + 
                                                           (1 - alpha) * team_ema[away_team][f'{stat}_conceded'])
            
            if away_col in row and not pd.isna(row[away_col]):
                # Away team's own stat
                team_ema[away_team][stat] = (alpha * row[away_col] + 
                                             (1 - alpha) * team_ema[away_team][stat])
                # Home team conceded this stat
                team_ema[home_team][f'{stat}_conceded'] = (alpha * row[away_col] + 
                                                           (1 - alpha) * team_ema[home_team][f'{stat}_conceded'])
    
    logger.info(f"  Added {len(stats) * 4} EMA features")
    return df


def calculate_rest_days(df):
    """Calculate days since last match for each team.
    
    Captures fatigue effects (e.g., Champions League midweek games).
    
    Returns:
        DataFrame with rest days features added
    """
    logger.info("Calculating rest days...")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Initialize columns
    df['days_rest_home'] = np.nan
    df['days_rest_away'] = np.nan
    df['home_short_rest'] = 0
    df['away_short_rest'] = 0
    
    # Track last match date by team
    team_last_match = {}  # team_id -> last_match_date
    
    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        match_date = row['date']
        
        # Calculate rest days
        if home_team in team_last_match:
            days_rest = (match_date - team_last_match[home_team]).days
            df.at[idx, 'days_rest_home'] = days_rest
            df.at[idx, 'home_short_rest'] = 1 if days_rest < 4 else 0
        
        if away_team in team_last_match:
            days_rest = (match_date - team_last_match[away_team]).days
            df.at[idx, 'days_rest_away'] = days_rest
            df.at[idx, 'away_short_rest'] = 1 if days_rest < 4 else 0
        
        # Update last match date
        team_last_match[home_team] = match_date
        team_last_match[away_team] = match_date
    
    # Add rest difference feature
    df['rest_diff'] = df['days_rest_home'] - df['days_rest_away']
    
    logger.info("  Added 7 rest days features")
    return df


def select_features(df):
    """Select final feature columns for modeling."""

    # Core identification
    id_cols = ['fixture_id', 'date', 'season_id', 'league_id', 'home_team_id', 'away_team_id',
               'home_team_name', 'away_team_name']

    # Target
    target_cols = ['target', 'home_win', 'draw', 'away_win', 'home_goals', 'away_goals']

    # Elo features
    elo_cols = ['home_elo', 'away_elo', 'elo_diff']

    # Form features
    form_cols = []
    for window in [3, 5]:
        form_cols.extend([
            f'home_wins_{window}', f'home_draws_{window}', f'home_losses_{window}',
            f'away_wins_{window}', f'away_draws_{window}', f'away_losses_{window}',
            f'home_form_{window}', f'away_form_{window}', f'form_diff_{window}'
        ])

    # Rolling stat features (team + player-level)
    rolling_cols = []

    # Team-level rolling stats
    team_stats_list = ['goals', 'xg', 'shots_total', 'shots_on_target', 'possession_pct',
                       'dangerous_attacks', 'corners', 'passes', 'successful_passes_pct',
                       'big_chances_created', 'tackles', 'interceptions']

    # Player-level rolling stats (most impactful)
    player_stats_list = [
        'player_clearances', 'player_aerials_won', 'player_touches', 'player_rating',
        'player_total_duels', 'player_duels_won', 'player_possession_lost',
        'player_accurate_passes', 'player_dribble_attempts', 'player_successful_dribbles',
        'player_tackles_won', 'player_long_balls_won', 'player_dispossessed', 'player_fouls_drawn',
        'player_blocked_shots', 'player_duels_lost', 'player_aerials_lost', 'player_aerials_total',
        'player_saves', 'player_goals_conceded', 'player_saves_insidebox',
    ]

    all_rolling_stats = team_stats_list + player_stats_list

    for window in ROLLING_WINDOWS:
        for stat in all_rolling_stats:
            for side in ['home', 'away']:
                rolling_cols.append(f'{side}_{stat}_{window}')
                rolling_cols.append(f'{side}_{stat}_conceded_{window}')

    # Attack/Defense strength
    strength_cols = []
    for window in [5, 10]:
        strength_cols.extend([
            f'home_attack_strength_{window}', f'away_attack_strength_{window}',
            f'home_defense_strength_{window}', f'away_defense_strength_{window}'
        ])

    # Standings
    standings_cols = ['home_position', 'away_position', 'position_diff',
                      'home_points', 'away_points', 'points_diff']

    # H2H
    h2h_cols = ['h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
                'h2h_home_goals_avg', 'h2h_away_goals_avg']

    # Injury
    injury_cols = ['home_injuries', 'away_injuries', 'injury_diff']

    # Market
    market_cols = ['odds_home', 'odds_draw', 'odds_away',
                   'market_prob_home', 'market_prob_draw', 'market_prob_away',
                   'market_home_away_ratio', 'market_favorite']

    # Contextual
    context_cols = ['round_num', 'season_progress', 'is_early_season',
                    'day_of_week', 'is_weekend']
    
    # EMA features
    ema_stats = ['goals', 'xg', 'shots_total', 'shots_on_target', 'possession_pct']
    ema_cols = []
    for stat in ema_stats:
        ema_cols.extend([f'home_{stat}_ema', f'away_{stat}_ema',
                        f'home_{stat}_conceded_ema', f'away_{stat}_conceded_ema'])
    
    # Rest days features
    rest_cols = ['days_rest_home', 'days_rest_away', 'home_short_rest',
                 'away_short_rest', 'rest_diff']

    # Combine all
    all_cols = (id_cols + target_cols + elo_cols + form_cols + rolling_cols +
                strength_cols + standings_cols + h2h_cols + injury_cols +
                market_cols + context_cols + ema_cols + rest_cols)

    # Select only columns that exist
    available_cols = [c for c in all_cols if c in df.columns]

    return df[available_cols]


def main():
    """Main feature engineering pipeline."""
    logger.info("=" * 60)
    logger.info("SPORTMONKS FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # Load data
    fixtures, lineups, events, sidelined, standings = load_raw_data()

    # Filter to completed matches only for training
    completed_fixtures = fixtures[fixtures['state_id'] == 5].copy()
    logger.info(f"Completed matches: {len(completed_fixtures)}")

    # Rename stat columns
    df = rename_stat_columns(completed_fixtures)

    # Calculate xG
    df = calculate_xg_approximation(df)

    # Calculate Elo ratings
    df = calculate_elo_ratings(df)

    # Calculate rolling features
    df = calculate_rolling_features(df)

    # Add form features
    df = calculate_form_features(df)

    # Add H2H features
    df = add_head_to_head_features(df)

    # Add sidelined features
    df = add_sidelined_features(df, sidelined)

    # Add standings features
    df = add_standings_features(df, standings)

    # Add market features
    df = add_market_features(df)

    # Add contextual features
    df = add_contextual_features(df)

    # Calculate attack/defense strength
    df = calculate_attack_defense_strength(df)
    
    # Add EMA features
    df = calculate_ema_features(df, alpha=0.1)
    
    # Add rest days features
    df = calculate_rest_days(df)

    # Create target variable
    df = create_target_variable(df)

    # Select features
    df_features = select_features(df)

    # Remove rows with too many missing values
    min_features = 10
    feature_cols = [c for c in df_features.columns if c not in
                   ['fixture_id', 'date', 'season_id', 'league_id', 'home_team_id', 'away_team_id',
                    'home_team_name', 'away_team_name', 'target', 'home_win', 'draw',
                    'away_win', 'home_goals', 'away_goals']]
    df_features['feature_count'] = df_features[feature_cols].notna().sum(axis=1)
    df_clean = df_features[df_features['feature_count'] >= min_features].drop('feature_count', axis=1)

    # Save
    output_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    df_clean.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total features: {len(df_clean.columns)}")
    logger.info(f"  Total samples: {len(df_clean)}")
    logger.info(f"  Output: {output_path}")

    # Print feature summary
    logger.info("\nFeature groups:")
    logger.info(f"  - Elo features: 3")
    logger.info(f"  - Form features: {len([c for c in df_clean.columns if 'form' in c or 'wins_' in c or 'draws_' in c or 'losses_' in c])}")
    logger.info(f"  - Rolling stats: {len([c for c in df_clean.columns if any(f'_{w}' in c for w in ROLLING_WINDOWS)])}")
    logger.info(f"  - H2H features: {len([c for c in df_clean.columns if 'h2h' in c])}")
    logger.info(f"  - Market features: {len([c for c in df_clean.columns if 'market' in c or 'odds' in c])}")

    # Class distribution
    logger.info("\nTarget distribution:")
    for target, label in [(0, 'Away win'), (1, 'Draw'), (2, 'Home win')]:
        count = (df_clean['target'] == target).sum()
        pct = count / len(df_clean) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    return df_clean


if __name__ == '__main__':
    main()
