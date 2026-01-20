"""
Production Thresholds Configuration

This file contains the optimal betting thresholds for the live prediction pipeline.
These thresholds are calibrated for the retrained balanced model (Draw 1.5x, Away 1.3x weights).

Last Updated: 2026-01-21
Calibration Method: Q4 2025 backtest (889 matches) with balanced class weights
Expected Performance: 25.0% ROI, 7.7 bets/day, 63.6% win rate
"""

# Optimal thresholds for retrained balanced model
OPTIMAL_THRESHOLDS = {
    'home': 0.48,  # Home win probability threshold (lowered from 0.50)
    'draw': 0.35,  # Draw probability threshold (lowered from 0.40)
    'away': 0.45,  # Away win probability threshold (lowered from 0.60)
}

# Expected performance metrics
EXPECTED_PERFORMANCE = {
    'roi': 25.0,
    'win_rate': 63.6,
    'bets_per_day': 7.7,
    'bet_frequency_pct': 77.6
}

# Calibration details
CALIBRATION_INFO = {
    'date': '2026-01-21',
    'method': 'q4_2025_backtest_balanced_model',
    'matches_tested': 889,
    'model_version': 'retrained_balanced_weights',
    'class_weights': 'Draw 1.5x, Away 1.3x',
    'features': '71 core features (no player stats)'
}

def get_production_thresholds():
    """Get the production thresholds for betting."""
    return OPTIMAL_THRESHOLDS.copy()

def get_expected_performance():
    """Get expected performance metrics."""
    return EXPECTED_PERFORMANCE.copy()
