"""
Type ID Mappings for SportMonks API Data

This module contains all type_id mappings for match statistics, player statistics,
and event types from the SportMonks API.
"""

# Match Statistics Type IDs
MATCH_STAT_TYPE_IDS = {
    41: 'big_chances_created',
    42: 'shots_total',
    43: 'shots_on_target',
    44: 'shots_off_target',
    45: 'possession',
    46: 'passes_total',
    47: 'passes_accurate',
    49: 'attacks',
    50: 'dangerous_attacks',
    51: 'corners',
    52: 'offsides',
    53: 'fouls',
    54: 'yellow_cards',
    55: 'red_cards',
    56: 'saves',
    57: 'tackles',
    58: 'blocks',
    59: 'interceptions',
    60: 'clearances',
    78: 'throw_ins',
    79: 'hit_woodwork',
    80: 'passes_total_alt',
    81: 'passes_accurate_alt',
    82: 'pass_accuracy_pct',
    84: 'penalties_scored',
    86: 'shots_inside_box',
    87: 'shots_outside_box',
    98: 'free_kicks',
    99: 'goal_kicks',
    100: 'substitutions',
    106: 'yellow_red_cards',
    1605: 'pass_accuracy_pct_alt',
    27264: 'duels_won',
    27265: 'duels_total',
}

# Player Statistics Type IDs (Priority 0 - Critical)
PLAYER_STAT_TYPE_IDS_P0 = {
    118: 'rating',              # Player match rating (0-10)
    80: 'touches',              # Total touches
    119: 'minutes',             # Minutes played
    1584: 'passes_total',       # Total passes
    120: 'pass_accuracy',       # Pass accuracy %
    122: 'passes_accurate',     # Accurate passes count
    101: 'duels_won',           # Duels won
    27273: 'duels_total',       # Total duels
    56: 'shots',                # Shots
    1491: 'shots_on_target',    # Shots on target
    114: 'key_passes',          # Key passes
}

# Player Statistics Type IDs (Priority 1)
PLAYER_STAT_TYPE_IDS_P1 = {
    27266: 'dribbles_successful',
    27272: 'tackles_won',
    27271: 'interceptions',
    27269: 'clearances',
    27270: 'dribbles_attempted',
    27267: 'crosses',
    27268: 'cross_accuracy',
    116: 'accurate_passes_alt',
    117: 'dispossessed',
}

# Player Statistics Type IDs (Priority 2)
PLAYER_STAT_TYPE_IDS_P2 = {
    121: 'errors_leading_to_goal',
    123: 'penalties_committed',
    124: 'penalties_won',
    125: 'penalties_saved',
    40: 'captain',
    41: 'goals',
    42: 'assists',
    51: 'offsides',
    52: 'fouls_committed',
    57: 'fouls_drawn',
    58: 'saves_made',
    64: 'punches',
    78: 'catches',
    79: 'sweeper_clearances',
    83: 'throw_ins',
    84: 'goal_kicks',
    86: 'hit_woodwork',
    88: 'big_chances_missed',
    94: 'inside_box_saves',
    95: 'pen_committed',
    96: 'pen_won',
    97: 'pen_scored',
    98: 'pen_missed',
    99: 'pen_saved',
    103: 'high_claims',
    104: 'punches',
    105: 'runs_out',
    106: 'runs_out_succ',
    107: 'crosses_not_claimed',
    108: 'gk_smother',
    109: 'gk_throws',
    110: 'gk_long_balls',
    111: 'goal_kicks_long',
    112: 'goal_kicks_short',
    113: 'passes_long',
    115: 'passes_short',
}

# Combined player stats (all priorities)
PLAYER_STAT_TYPE_IDS = {
    **PLAYER_STAT_TYPE_IDS_P0,
    **PLAYER_STAT_TYPE_IDS_P1,
    **PLAYER_STAT_TYPE_IDS_P2,
}

# Event Type IDs
EVENT_TYPE_IDS = {
    14: 'goal',
    15: 'yellow_card',
    16: 'red_card',
    17: 'penalty',
    18: 'substitution',
    19: 'own_goal',
    79: 'var_decision',
    80: 'yellow_red_card',
}

# Lineup Type IDs
LINEUP_TYPE_IDS = {
    11: 'starter',
    12: 'substitute',
}

# Position IDs (common ones)
POSITION_IDS = {
    24: 'goalkeeper',
    25: 'defender',
    26: 'midfielder',
    27: 'forward',
}

# Reverse mappings for lookups
MATCH_STAT_ID_TO_NAME = MATCH_STAT_TYPE_IDS
MATCH_STAT_NAME_TO_ID = {v: k for k, v in MATCH_STAT_TYPE_IDS.items()}

PLAYER_STAT_ID_TO_NAME = PLAYER_STAT_TYPE_IDS
PLAYER_STAT_NAME_TO_ID = {v: k for k, v in PLAYER_STAT_TYPE_IDS.items()}

EVENT_ID_TO_NAME = EVENT_TYPE_IDS
EVENT_NAME_TO_ID = {v: k for k, v in EVENT_TYPE_IDS.items()}


def get_stat_name(type_id: int, category: str = 'player') -> str:
    """
    Get stat name from type_id.
    
    Args:
        type_id: The type_id to look up
        category: 'player', 'match', or 'event'
    
    Returns:
        Stat name or 'unknown_{type_id}'
    """
    if category == 'player':
        return PLAYER_STAT_ID_TO_NAME.get(type_id, f'unknown_{type_id}')
    elif category == 'match':
        return MATCH_STAT_ID_TO_NAME.get(type_id, f'unknown_{type_id}')
    elif category == 'event':
        return EVENT_ID_TO_NAME.get(type_id, f'unknown_{type_id}')
    else:
        return f'unknown_{type_id}'


def get_stat_id(stat_name: str, category: str = 'player') -> int:
    """
    Get type_id from stat name.
    
    Args:
        stat_name: The stat name to look up
        category: 'player', 'match', or 'event'
    
    Returns:
        type_id or None if not found
    """
    if category == 'player':
        return PLAYER_STAT_NAME_TO_ID.get(stat_name)
    elif category == 'match':
        return MATCH_STAT_NAME_TO_ID.get(stat_name)
    elif category == 'event':
        return EVENT_NAME_TO_ID.get(stat_name)
    else:
        return None
