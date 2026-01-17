"""
Shared Utility Functions
========================

Common functions used across the pipeline.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import hashlib
import json

from config import LOG_LEVEL, LOG_FORMAT, RANDOM_SEED

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Setup a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_probabilities(probs: np.ndarray, tolerance: float = 0.001) -> bool:
    """Check that probabilities sum to 1 and are in valid range."""
    if probs is None or len(probs) == 0:
        return False
    
    # Check range [0, 1]
    if np.any(probs < 0) or np.any(probs > 1):
        return False
    
    # Check sum to 1
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        # Multiple predictions
        sums = probs.sum(axis=1)
        return np.all(np.abs(sums - 1.0) < tolerance)
    else:
        # Single prediction
        return abs(sum(probs) - 1.0) < tolerance


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Normalize probabilities to sum to 1."""
    if probs.ndim == 1:
        return probs / probs.sum()
    else:
        return probs / probs.sum(axis=1, keepdims=True)


# =============================================================================
# ODDS CONVERSION
# =============================================================================

def decimal_to_implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1:
        return 1.0
    return 1.0 / odds


def implied_prob_to_decimal(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0:
        return float('inf')
    return 1.0 / prob


def remove_vig(odds_home: float, odds_draw: float, odds_away: float) -> Tuple[float, float, float]:
    """
    Remove bookmaker's vig from 1X2 odds to get fair probabilities.
    
    Uses basic normalization method. For more sophisticated approaches,
    consider Shin's method or power method.
    """
    # Convert to implied probabilities
    p_home = decimal_to_implied_prob(odds_home)
    p_draw = decimal_to_implied_prob(odds_draw)
    p_away = decimal_to_implied_prob(odds_away)
    
    # Total (overround/vig)
    total = p_home + p_draw + p_away
    
    # Normalize to remove vig
    return p_home / total, p_draw / total, p_away / total


def calculate_edge(model_prob: float, market_prob: float) -> float:
    """Calculate edge as model probability minus market implied probability."""
    return model_prob - market_prob


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """Calculate expected value of a bet."""
    return model_prob * decimal_odds - 1.0


# =============================================================================
# MATCH OUTCOME ENCODING
# =============================================================================

OUTCOME_MAP = {
    "H": 0,  # Home win
    "D": 1,  # Draw
    "A": 2,  # Away win
}

OUTCOME_REVERSE = {v: k for k, v in OUTCOME_MAP.items()}


def encode_result(home_goals: int, away_goals: int) -> int:
    """Encode match result as 0 (home), 1 (draw), 2 (away)."""
    if home_goals > away_goals:
        return 0
    elif home_goals == away_goals:
        return 1
    else:
        return 2


def result_to_label(result: str) -> int:
    """Convert result string (H/D/A) to numeric label."""
    return OUTCOME_MAP.get(result, -1)


def label_to_result(label: int) -> str:
    """Convert numeric label to result string."""
    return OUTCOME_REVERSE.get(label, "?")


# =============================================================================
# DATE UTILITIES
# =============================================================================

def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
    """Try to parse date string with multiple formats."""
    if formats is None:
        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d/%m/%y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def get_season_from_date(date: datetime) -> str:
    """
    Determine season from date.
    Assumes season runs Aug-May (e.g., Aug 2023 - May 2024 = "2023-2024")
    """
    if date.month >= 8:  # Aug onwards = start of new season
        return f"{date.year}-{date.year + 1}"
    else:  # Jan-Jul = second half of season
        return f"{date.year - 1}-{date.year}"


# =============================================================================
# TEAM NAME NORMALIZATION
# =============================================================================

# Common variations to standardize
TEAM_NAME_MAPPINGS = {
    # English
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Wolves": "Wolverhampton",
    "Wolverhampton Wanderers": "Wolverhampton",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Sheffield United": "Sheffield Utd",
    "Nottingham Forest": "Nott'm Forest",
    
    # Spanish
    "Atletico Madrid": "Atl. Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Athletic Club": "Ath Bilbao",
    "Real Betis Balompie": "Betis",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "RC Celta": "Celta",
    "Deportivo Alaves": "Alaves",
    
    # German
    "Bayern Munich": "Bayern",
    "FC Bayern München": "Bayern",
    "Borussia Dortmund": "Dortmund",
    "Borussia M'gladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "RB Leipzig": "Leipzig",
    "Bayer Leverkusen": "Leverkusen",
    "Bayer 04 Leverkusen": "Leverkusen",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "Eintracht Frankfurt": "Ein Frankfurt",
    
    # Italian
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Napoli": "Napoli",
    "SSC Napoli": "Napoli",
    "Hellas Verona": "Verona",
    
    # French
    "Paris Saint Germain": "PSG",
    "Paris Saint-Germain": "PSG",
    "Paris SG": "PSG",
    "Olympique Marseille": "Marseille",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyon": "Lyon",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to consistent format."""
    # Strip whitespace
    name = name.strip()
    
    # Check direct mapping
    if name in TEAM_NAME_MAPPINGS:
        return TEAM_NAME_MAPPINGS[name]
    
    # Check case-insensitive
    name_lower = name.lower()
    for original, normalized in TEAM_NAME_MAPPINGS.items():
        if original.lower() == name_lower:
            return normalized
    
    return name


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate multiclass log loss.
    
    Args:
        y_true: True labels (integers 0, 1, 2)
        y_pred: Predicted probabilities (N x 3 array)
        eps: Small value to avoid log(0)
    
    Returns:
        Log loss value
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n_samples = len(y_true)
    
    # Extract probability of true class for each sample
    log_loss = 0.0
    for i, true_label in enumerate(y_true):
        log_loss -= np.log(y_pred[i, int(true_label)])
    
    return log_loss / n_samples


def calculate_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate multiclass Brier score.
    
    Args:
        y_true: True labels (integers 0, 1, 2)
        y_pred: Predicted probabilities (N x 3 array)
    
    Returns:
        Brier score (lower is better)
    """
    n_samples = len(y_true)
    n_classes = y_pred.shape[1]
    
    # One-hot encode true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, int(label)] = 1
    
    # Brier score = mean squared error
    return np.mean(np.sum((y_pred - y_true_onehot) ** 2, axis=1))


def calculate_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculate calibration error for each class.
    
    Args:
        y_true: True labels (integers 0, 1, 2)
        y_pred: Predicted probabilities (N x 3 array)
        n_bins: Number of bins for calibration
    
    Returns:
        Dict with calibration metrics per class
    """
    n_classes = y_pred.shape[1]
    results = {}
    
    for class_idx in range(n_classes):
        probs = y_pred[:, class_idx]
        true_binary = (y_true == class_idx).astype(int)
        
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges[1:-1])
        
        bin_means = []
        bin_trues = []
        bin_counts = []
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_means.append(probs[mask].mean())
                bin_trues.append(true_binary[mask].mean())
                bin_counts.append(mask.sum())
        
        # Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = sum(bin_counts)
        for mean, true, count in zip(bin_means, bin_trues, bin_counts):
            ece += (count / total_samples) * abs(mean - true)
        
        results[f"class_{class_idx}"] = {
            "ece": ece,
            "bin_means": bin_means,
            "bin_trues": bin_trues,
            "bin_counts": bin_counts,
        }
    
    # Overall ECE (average across classes)
    results["overall_ece"] = np.mean([results[f"class_{i}"]["ece"] for i in range(n_classes)])
    
    return results


# =============================================================================
# HASHING & REPRODUCIBILITY
# =============================================================================

def hash_features(features: Dict) -> str:
    """Create a hash of feature values for reproducibility tracking."""
    feature_str = json.dumps(features, sort_keys=True, default=str)
    return hashlib.md5(feature_str.encode()).hexdigest()[:16]


def set_random_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set other library seeds if available
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# =============================================================================
# DATA SPLITTING
# =============================================================================

def time_based_split(
    df: pd.DataFrame,
    date_column: str,
    train_end: str,
    val_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by time (critical for sports prediction).
    
    Args:
        df: DataFrame with match data
        date_column: Name of date column
        train_end: End date for training (exclusive)
        val_end: End date for validation (exclusive)
    
    Returns:
        train_df, val_df, test_df
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train_df = df[df[date_column] < train_end_dt]
    val_df = df[(df[date_column] >= train_end_dt) & (df[date_column] < val_end_dt)]
    test_df = df[df[date_column] >= val_end_dt]
    
    return train_df, val_df, test_df


def season_based_split(
    df: pd.DataFrame,
    season_column: str,
    train_seasons: List[str],
    val_seasons: List[str],
    test_seasons: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by season.
    
    Args:
        df: DataFrame with match data
        season_column: Name of season column
        train_seasons: List of seasons for training
        val_seasons: List of seasons for validation
        test_seasons: List of seasons for testing
    
    Returns:
        train_df, val_df, test_df
    """
    train_df = df[df[season_column].isin(train_seasons)]
    val_df = df[df[season_column].isin(val_seasons)]
    test_df = df[df[season_column].isin(test_seasons)]
    
    return train_df, val_df, test_df


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics as a table."""
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:<30} {value:>15.4f}")
        else:
            print(f"{name:<30} {str(value):>15}")
    
    print(f"{'=' * 50}\n")


def print_prediction_sample(
    fixture: Dict,
    prediction: np.ndarray,
    market_probs: Optional[np.ndarray] = None
):
    """Pretty print a single prediction."""
    print(f"\n{fixture['home_team']} vs {fixture['away_team']}")
    print(f"{'Outcome':<10} {'Model':>10} {'Market':>10} {'Edge':>10}")
    print("-" * 45)
    
    outcomes = ["Home", "Draw", "Away"]
    for i, outcome in enumerate(outcomes):
        model_prob = prediction[i]
        if market_probs is not None:
            market_prob = market_probs[i]
            edge = model_prob - market_prob
            print(f"{outcome:<10} {model_prob:>9.1%} {market_prob:>9.1%} {edge:>+9.1%}")
        else:
            print(f"{outcome:<10} {model_prob:>9.1%}")
