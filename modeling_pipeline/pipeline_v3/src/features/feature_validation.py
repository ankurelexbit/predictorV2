"""
Feature Validation Framework

Validates features for data leakage and quality issues.
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Features that indicate data leakage
FORBIDDEN_FEATURES = [
    'result',
    'home_goals',
    'away_goals',
    'home_score',
    'away_score',
    'target',
    'target_home_win',
    'target_draw',
    'target_away_win',
    'winner',
    'goals',  # Unless it's goals_avg_5 or similar
]

# Features that should be rolling averages, not current match
SHOULD_BE_ROLLING = [
    'possession',
    'shots_total',
    'shots_on_target',
    'passes_total',
    'corners',
    'fouls',
    'yellow_cards',
    'red_cards',
]


def validate_features(features: Dict, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate features for data leakage and quality issues.
    
    Args:
        features: Dictionary of features
        strict: If True, raise errors for warnings too
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    warnings = []
    
    # Check for forbidden features
    for forbidden in FORBIDDEN_FEATURES:
        if forbidden in features:
            errors.append(f"LEAKAGE: '{forbidden}' in features - this is the target!")
    
    # Check for features that should be rolling averages
    for should_roll in SHOULD_BE_ROLLING:
        if should_roll in features:
            # Check if it's actually a rolling average
            if not any(suffix in should_roll for suffix in ['_5', '_10', '_avg', '_last']):
                warnings.append(
                    f"WARNING: '{should_roll}' should be a rolling average "
                    f"(e.g., '{should_roll}_avg_5')"
                )
    
    # Check for impossible values
    for key, value in features.items():
        if value is None:
            continue
            
        # Percentages should be 0-100
        if 'pct' in key or 'percentage' in key or 'accuracy' in key:
            if not (0 <= value <= 100):
                errors.append(f"Invalid value for {key}: {value} (should be 0-100)")
        
        # Ratings should be 0-10
        if 'rating' in key:
            if not (0 <= value <= 10):
                errors.append(f"Invalid rating for {key}: {value} (should be 0-10)")
        
        # Goals/shots should be non-negative
        if any(x in key for x in ['goals', 'shots', 'corners', 'fouls']):
            if value < 0:
                errors.append(f"Negative value for {key}: {value}")
    
    # Check for required fields
    required_fields = ['fixture_id']
    for field in required_fields:
        if field not in features:
            errors.append(f"Missing required field: {field}")
    
    # Log results
    if errors:
        logger.error(f"Feature validation failed with {len(errors)} errors:")
        for error in errors:
            logger.error(f"  - {error}")
    
    if warnings:
        logger.warning(f"Feature validation has {len(warnings)} warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    is_valid = len(errors) == 0 and (not strict or len(warnings) == 0)
    
    return is_valid, errors + warnings


def validate_temporal_cutoff(feature_date, match_date) -> bool:
    """
    Validate that feature data is from before the match.
    
    Args:
        feature_date: Date of the feature data
        match_date: Date of the match
    
    Returns:
        True if valid (feature_date < match_date)
    """
    return feature_date < match_date
