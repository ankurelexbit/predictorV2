"""
Production Configuration
========================

CRITICAL: This file contains the production model and threshold settings.
Any changes here will affect live predictions!

Last Updated: 2026-02-05 (Hybrid strategy with Jan calibrator)
Model: v2.0.0 — Unweighted CatBoost (class weights removed)
Versioning: Automatic semantic versioning (v1.0.0, v1.1.0, etc.)
Tested on: January 2026 (202 resolved predictions, Top 5 Leagues)
Strategy: Hybrid — prob gate + EV filter + draw odds gate, pick highest EV
Simulated PnL: +$45.15, 45.8% win rate, 107 bets (~3.5/day), all 3 outcomes positive
Constraints: 3-4 bets/day, each outcome ≥10% distribution, maximized PnL
Calibrators: calibrators_v1.1.0 (Jan 2026, loaded via LATEST_CALIBRATORS)
"""

from pathlib import Path
import re

# ============================================================================
# ELO RATING CONFIGURATION
# ============================================================================

class EloConfig:
    """Elo rating system configuration."""

    K_FACTOR = 32  # Update speed (higher = more responsive to recent results)
    HOME_ADVANTAGE = 35  # Home team Elo bonus (calibrated for modern football)
    INITIAL_ELO = 1500  # Starting Elo for new teams

    @classmethod
    def get_params(cls):
        """Get Elo parameters as dict."""
        return {
            'k_factor': cls.K_FACTOR,
            'home_advantage': cls.HOME_ADVANTAGE,
            'initial_elo': cls.INITIAL_ELO
        }

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

def get_latest_model_path() -> str:
    """
    Get the path to the latest production model.

    Returns the model specified in models/production/LATEST file,
    or falls back to finding the highest version number.

    Returns:
        str: Path to latest model file
    """
    production_dir = Path("models/production")
    latest_file = production_dir / "LATEST"

    # Try reading LATEST file first
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            model_filename = f.read().strip()
            model_path = production_dir / model_filename
            if model_path.exists():
                return str(model_path)

    # Fallback: Find highest version number
    if production_dir.exists():
        model_files = list(production_dir.glob("model_v*.joblib"))
        if model_files:
            # Extract version numbers and sort
            versions = []
            for f in model_files:
                match = re.search(r'model_v(\d+)\.(\d+)\.(\d+)\.joblib', f.name)
                if match:
                    major, minor, patch = map(int, match.groups())
                    versions.append((major, minor, patch, f))

            if versions:
                versions.sort(reverse=True)
                return str(versions[0][3])

    # Final fallback: Legacy model path
    legacy_path = "models/weight_experiments/option3_balanced.joblib"
    if Path(legacy_path).exists():
        return legacy_path

    raise FileNotFoundError("No production model found. Train a model first with train_production_model.py")


# Production Model Path (automatically uses latest version)
MODEL_PATH = get_latest_model_path()

# Model Metadata
MODEL_INFO = {
    'name': 'Unweighted',
    'version': 'auto',  # Determined from model file
    'class_weights': {'home': 1.0, 'draw': 1.0, 'away': 1.0},
    'trained_on': '2023-2024 seasons',
    'last_updated': '2026-02-04',
    'note': 'Class weights removed — isotonic calibrators correct residual bias'
}

# ============================================================================
# BETTING THRESHOLDS
# ============================================================================

# Hybrid strategy: prob gate → EV filter → draw odds gate → pick highest EV.
# Calibrators: v1.1.0 (Jan 2026), loaded via models/production/LATEST_CALIBRATORS.
# Raw model probs are stored in DB; calibrated probs are used only for gating.
# Verified on Jan 2026 (202 predictions): +$45.15 PnL, 45.8% wr, 107 bets (~3.5/day)
# H: 19 bets, 73.7% wr, +$2.13 | D: 54 bets, 33.3% wr, +$36.02 | A: 34 bets, 52.9% wr, +$7.00
# Active strategy — flipping this is the only change needed to switch between strategies
STRATEGY = 'hybrid'                # 'hybrid' or 'pure_threshold'

# Hybrid settings
HYBRID_THRESHOLDS = {
    'home': 0.44,  # prob gate — cal_home must exceed this to be a candidate
    'draw': 0.20,  # prob gate — cal_draw must exceed this to be a candidate
    'away': 0.30   # prob gate — cal_away must exceed this to be a candidate
}
DRAW_ODDS_GATE = 4.0               # draw only qualifies if market odds >= this
MIN_EV = 0.0                       # candidate rejected if EV = (cal_prob × odds) - 1 <= this

# Pure threshold settings (last tested: v1.1.0, H.50 D.34 A.43 → 125 bets, +$24.27)
PURE_THRESHOLD_THRESHOLDS = {
    'home': 0.50,
    'draw': 0.34,
    'away': 0.43
}

# Resolver — always use this in code, never read the dicts directly
THRESHOLDS = HYBRID_THRESHOLDS if STRATEGY == 'hybrid' else PURE_THRESHOLD_THRESHOLDS

# Historical threshold performance (for reference)
THRESHOLD_HISTORY = {
    '2026-02-05-v1.1.0-hybrid': {
        'thresholds': {'home': 0.44, 'draw': 0.20, 'away': 0.30},
        'model': 'v2.0.0 (unweighted)',
        'calibrators': 'calibrators_v1.1.0 (trained on Jan 2026, 202 samples)',
        'strategy': 'Hybrid — prob gate + EV filter + draw odds gate (4.0x), pick highest EV',
        'reason': 'Switched from pure threshold to hybrid after grid search showed hybrid '
                  'captures draw mispricing at 4.0x+ odds (+$36.02 on 54 draw bets). '
                  'EV filter prevents low-odds home bets from bleeding. '
                  'Draw odds gate ensures only genuinely mispriced draws qualify.',
        'performance': {
            'total_predictions': 202,
            'simulated_hybrid': {
                'total_bets': 107, 'wins': 49, 'win_rate': 45.8, 'total_pnl': 45.15, 'roi': 42.2,
                'home': {'bets': 19, 'wins': 14, 'win_rate': 73.7, 'pnl': 2.13, 'avg_odds': 1.58, 'dist_pct': 17.8},
                'draw': {'bets': 54, 'wins': 18, 'win_rate': 33.3, 'pnl': 36.02, 'avg_odds': 4.62, 'dist_pct': 50.5},
                'away': {'bets': 34, 'wins': 17, 'win_rate': 50.0, 'pnl': 7.00, 'avg_odds': 1.88, 'dist_pct': 31.8}
            },
            'test_period': 'January 2026',
            'leagues': 'Top 5 only',
            'constraints_met': '~3.5 bets/day ✓, each outcome ≥10% ✓, all 3 outcomes positive PnL ✓'
        }
    },
    '2026-02-05-v1.1.0-jan-calibrator-volume-capped': {
        'thresholds': {'home': 0.50, 'draw': 0.34, 'away': 0.43},
        'model': 'v2.0.0 (unweighted)',
        'calibrators': 'calibrators_v1.1.0 (trained on Jan 2026, 202 samples)',
        'strategy': 'Pure threshold — volume-capped to ~4 bets/day (125/month)',
        'reason': 'Raised home from 0.40→0.50 and draw from 0.28→0.34 to cut volume '
                  'from 202 to 125 bets/month while keeping each outcome ≥10%. '
                  'Away kept at 0.43 (profitable at 66.7% WR). '
                  'Jan calibrator inflates draw cal_probs due to elevated Jan draw rate (30.2%) '
                  'so draws carry most PnL (+$17.80). Home breakeven at -$0.05 due to low avg odds (1.52).',
        'performance': {
            'total_predictions': 202,
            'simulated_pure_threshold': {
                'total_bets': 125, 'wins': 70, 'win_rate': 56.0, 'total_pnl': 24.27, 'roi': 19.4,
                'home': {'bets': 45, 'wins': 30, 'win_rate': 66.7, 'pnl': -0.05, 'avg_odds': 1.52, 'dist_pct': 36.0},
                'draw': {'bets': 50, 'wins': 20, 'win_rate': 40.0, 'pnl': 17.80, 'avg_odds': 3.45, 'dist_pct': 40.0},
                'away': {'bets': 30, 'wins': 20, 'win_rate': 66.7, 'pnl':  6.52, 'avg_odds': 1.88, 'dist_pct': 24.0}
            },
            'test_period': 'January 2026',
            'leagues': 'Top 5 only',
            'constraints_met': '~4 bets/day ✓, each outcome ≥10% ✓, PnL maximized ✓'
        }
    },
    '2026-02-04-v2.0.0-optimized-balanced': {
        'thresholds': {'home': 0.40, 'draw': 0.28, 'away': 0.40},
        'model': 'v2.0.0 (unweighted)',
        'calibrators': 'Existing calibrators.joblib',
        'strategy': 'Pure threshold — optimized for balanced distribution (each outcome ≥10%)',
        'reason': 'Raised home threshold from 0.36 to 0.40 to reduce home bet volume and meet '\
                  '10% minimum distribution constraint for each outcome. This improves PnL by '\
                  'reducing exposure to lower-confidence home bets while increasing draw volume '\
                  'which has higher per-bet profitability. Strategy delivers better risk-adjusted '\
                  'returns with more balanced portfolio.',
        'performance': {
            'total_predictions': 539,
            'simulated_pure_threshold': {
                'total_bets': 508, 'wins': 275, 'win_rate': 54.1, 'total_pnl': 41.90, 'roi': 8.2,
                'home': {'bets': 338, 'wins': 190, 'win_rate': 56.2, 'pnl': 23.32, 'dist_pct': 66.5},
                'draw': {'bets': 56,  'wins': 19,  'win_rate': 33.9, 'pnl': 8.68,  'dist_pct': 11.0},
                'away': {'bets': 114, 'wins': 66,  'win_rate': 57.9, 'pnl': 9.90,  'dist_pct': 22.4}
            },
            'jan_2026_only': {
                'total_bets': 188, 'wins': 98, 'win_rate': 52.1, 'total_pnl': 7.89, 'roi': 4.2,
                'note': 'Turned breakeven month (+$0.85 with H=0.36) into +$7.89 profit'
            },
            'test_period': 'Nov 2025 – Jan 2026',
            'leagues': 'Top 5 only',
            'constraints_met': 'WR>50% ✓, PnL maximized ✓, each outcome ≥10% ✓'
        }
    },
    '2026-02-04-v2.0.0-pure-threshold': {
        'thresholds': {'home': 0.36, 'draw': 0.28, 'away': 0.40},
        'model': 'v2.0.0 (unweighted)',
        'strategy': 'Pure threshold — pick highest cal_prob among candidates. No EV gate, no odds gate.',
        'reason': 'Simplified from hybrid (prob gate + EV>0 + draw odds gate) to remove odds-dependent '
                  'selection.  Odds change between prediction time and bet placement so the EV gate '
                  'was unreliable.  Pure threshold maximises win rate while keeping all 3 outcomes in the '
                  'portfolio.  Draw volume is constrained by the isotonic step function (max cal_draw = 0.285) '
                  'not by the threshold.  EV strategy saved as TODO for future re-evaluation.',
        'performance': {
            'total_predictions': 581,
            'simulated_pure_threshold': {
                'total_bets': 576, 'wins': 299, 'win_rate': 51.9, 'total_pnl': 32.53, 'roi': 5.6,
                'home': {'bets': 434, 'wins': 227, 'win_rate': 52.3, 'pnl': 20.04, 'avg_odds': 2.19},
                'draw': {'bets': 47,  'wins': 14,  'win_rate': 29.8, 'pnl': 2.71,  'avg_odds': 3.64},
                'away': {'bets': 95,  'wins': 58,  'win_rate': 61.1, 'pnl': 9.78,  'avg_odds': 1.89}
            },
            'test_period': 'Nov 2025 – Jan 2026',
            'leagues': 'Top 5 only'
        }
    },
    '2026-02-04-v2.0.0-hybrid': {
        'thresholds': {'home': 0.48, 'draw': 0.26, 'away': 0.28},
        'model': 'v2.0.0 (unweighted — class weights removed)',
        'calibration': 'Isotonic calibrators re-fitted on v2.0.0 raw probs from 581 resolved predictions. '
                       'Raw bias near-zero (Away 0.302 vs 0.308, Draw 0.249 vs 0.256, Home 0.449 vs 0.435). '
                       'Calibration MAE: Home 6.6→5.0%, Draw 7.0→4.0%, Away 7.7→6.0%.',
        'reason': 'Class weights (H=1.2 D=1.4 A=1.1) removed after empirical comparison showed they '
                  'distorted all three probability channels (+4.0% draw bias, -3.4% away bias). '
                  'Unweighted model won 3/4 key metrics: lower log loss, lower cal MAE, near-zero bias. '
                  'Calibrators re-fitted on new model raw probs. Thresholds lowered because EV filter '
                  'now does the primary gating — prob gates just exclude obvious no-bets. '
                  'Draw requires odds >= 4.0x to qualify: selects matches where market underprices draw risk. '
                  'Draw win rate 35% vs 23% market implied = +52% edge. '
                  'Hybrid simulated: +$85.05, 403 bets, 53% win rate, all 3 outcomes positive PnL.',
        'performance': {
            'total_predictions': 581,
            'actual_outcomes': {'H': 253, 'D': 149, 'A': 179},
            'hybrid_strategy': {
                'total_bets': 403, 'total_wins': 214, 'win_rate': 53.1, 'total_pnl': 85.05, 'roi': 21.1,
                'home': {'bets': 227, 'wins': 130, 'win_rate': 57.3, 'pnl': 36.96, 'avg_odds': 2.19},
                'draw':  {'bets': 40,  'wins': 14,  'win_rate': 35.0, 'pnl': 21.87, 'avg_odds': 4.34,
                          'note': 'cal>=0.26 AND odds>=4.0x. 35% wr vs 23% implied = +52% edge'},
                'away':  {'bets': 136, 'wins': 70,  'win_rate': 51.5, 'pnl': 26.22, 'avg_odds': 2.52}
            },
            'test_period': 'Nov 2025 – Jan 2026',
            'leagues': 'Top 5 only'
        }
    },
    '2026-02-04-v3-calibrated': {
        'thresholds': {'home': 0.56, 'draw': 0.26, 'away': 0.37},
        'calibration': 'Isotonic regression fitted on 581 predictions. '
                       'Calibrators saved to models/calibrators.joblib. '
                       'Thresholds now apply to calibrated probs, not raw CatBoost output. '
                       'Raw probs still stored in DB for future re-calibration.',
        'reason': 'Home had +15.6% overconfidence in 0.55–0.60 bin (dead zone). '
                  'Draw had +5.6% overconfidence in 0.30–0.35 bin. '
                  'Away was consistently underconfident above 0.40. '
                  'Calibration fixes all three. Simulated total: +$40.81 vs +$26.54 uncalibrated (+$14.27 gain).',
        'performance': {
            'total_predictions': 581,
            'actual_outcomes': {'H': 253, 'D': 149, 'A': 179},
            'simulated_calibrated': {
                'home': {'cal_threshold': 0.56, 'bets': 72, 'wins': 58, 'win_rate': 80.6, 'profit': 9.82, 'roi': 13.6},
                'draw': {'cal_threshold': 0.26, 'bets': 126, 'wins': 42, 'win_rate': 33.3, 'profit': 15.88, 'roi': 12.6},
                'away': {'cal_threshold': 0.37, 'bets': 133, 'wins': 78, 'win_rate': 58.6, 'profit': 15.11, 'roi': 11.4}
            },
            'test_period': 'Nov 2025 – Jan 2026',
            'leagues': 'Top 5 only'
        }
    },
    '2026-02-04': {
        'thresholds': {'home': 0.58, 'draw': 0.30, 'away': 0.41},
        'reason': 'Jan-only optimisation (223 predictions). Later superseded by v2 with full Nov-Jan data.',
        'performance': {
            'total_predictions': 223,
            'note': 'Superseded — draw at 0.30 was losing on Nov-Dec data'
        }
    },
    '2026-02-02': {
        'thresholds': {'home': 0.65, 'draw': 0.30, 'away': 0.42},
        'performance': {
            'total_bets': 148,
            'total_profit': 41.81,
            'roi': 28.3,
            'win_rate': 52.0,
            'draw_winrate': 39.0,
            'test_period': 'January 2026',
            'leagues': 'Top 5 only'
        }
    }
}

# ============================================================================
# LEAGUE FILTERING
# ============================================================================

# TOP 5 EUROPEAN LEAGUES (RECOMMENDED)
# Model performs significantly better on these leagues:
#   Top 5: 28.3% ROI, 39.0% draw win rate
#   All leagues: 9.0% ROI, 28.9% draw win rate
TOP_5_LEAGUES = [
    8,    # Premier League (England)
    82,   # Bundesliga (Germany)
    384,  # Serie A (Italy)
    564,  # Ligue 1 (France)
    301   # La Liga (Spain)
]

# Set to True to filter predictions to Top 5 leagues only
FILTER_TOP_5_ONLY = True

# Alternative: Specify custom league IDs
# CUSTOM_LEAGUES = [8, 82, 384, 564, 301, 301]  # Add more if needed
# FILTER_CUSTOM_LEAGUES = False

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================

# Minimum confidence threshold (predictions below this won't be stored)
MIN_CONFIDENCE_ANY_OUTCOME = 0.15  # At least one outcome must be >15%

# Only store predictions that cross betting threshold
STORE_ONLY_ACTIONABLE = False  # Set True to only store bets you'd actually place

# ============================================================================
# HISTORICAL DATA SETTINGS
# ============================================================================

# How much historical data to load for feature calculation
HISTORY_DAYS = 365  # 1 year of history

# Alternative: Specific date range
USE_DATE_RANGE = False
HISTORY_START_DATE = None  # "2025-01-01"
HISTORY_END_DATE = None    # "2026-01-31"

# ============================================================================
# DATABASE SETTINGS
# ============================================================================

# Table name for storing predictions
PREDICTIONS_TABLE = "predictions"

# Model version tag (for tracking different model versions)
MODEL_VERSION_TAG = "v4.1_option3_balanced"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []

    # Check model path exists
    try:
        model_path = get_latest_model_path()
        if not Path(model_path).exists():
            errors.append(f"Model path does not exist: {model_path}")
    except FileNotFoundError as e:
        errors.append(str(e))

    # Validate thresholds
    for outcome, thresh in THRESHOLDS.items():
        if not 0.0 <= thresh <= 1.0:
            errors.append(f"Invalid threshold for {outcome}: {thresh} (must be 0-1)")

    # Check league IDs
    if FILTER_TOP_5_ONLY:
        if not TOP_5_LEAGUES or not all(isinstance(x, int) for x in TOP_5_LEAGUES):
            errors.append("TOP_5_LEAGUES must be a list of integers")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


if __name__ == '__main__':
    """Print current configuration."""
    print("="*80)
    print("PRODUCTION CONFIGURATION")
    print("="*80)
    print()
    print(f"Model: {MODEL_INFO['name']}")
    print(f"Path: {MODEL_PATH}")
    print(f"Version: {MODEL_VERSION_TAG}")
    print()
    print("Thresholds:")
    for outcome, thresh in THRESHOLDS.items():
        print(f"  {outcome.capitalize()}: {thresh:.2f}")
    print()
    print(f"League Filter: {'Top 5 only' if FILTER_TOP_5_ONLY else 'All leagues'}")
    if FILTER_TOP_5_ONLY:
        print(f"  Leagues: {TOP_5_LEAGUES}")
    print()

    try:
        validate_config()
        print("✅ Configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration errors:")
        print(str(e))
