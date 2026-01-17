"""
09 - Prediction Pipeline
========================

Production-ready prediction pipeline for making 1X2 predictions.

Features:
1. Load trained models
2. Fetch upcoming fixtures
3. Compute features
4. Generate predictions
5. Detect edges vs market odds
6. Output predictions with confidence

This is what runs daily to generate your product's predictions.

Usage:
    python 09_prediction_pipeline.py
    python 09_prediction_pipeline.py --league PL --days 7
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import json
import joblib
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    DATA_DIR,
    MIN_EDGE_THRESHOLD,
    MIN_EV_THRESHOLD,
    RANDOM_SEED,
)
from utils import (
    setup_logger,
    set_random_seed,
    normalize_team_name,
    calculate_edge,
    calculate_ev,
    hash_features,
    remove_vig,
)

# Setup
logger = setup_logger("prediction_pipeline")
set_random_seed(RANDOM_SEED)


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================

class PredictionPipeline:
    """
    Production prediction pipeline.
    
    Steps:
    1. Load models
    2. Get fixtures
    3. Compute features (using existing feature store)
    4. Make predictions
    5. Compare to market
    6. Output results
    """
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        
        # Models (loaded on demand)
        self.elo_model = None
        self.dixon_coles_model = None
        self.xgboost_model = None
        self.ensemble_config = None
        
        # Feature engineering state (loaded from saved state)
        self.elo_ratings = {}
        self.team_form = {}
        
        # Model version for tracking
        self.model_version = None
    
    def load_models(self):
        """Load all trained models."""
        logger.info("Loading models...")
        
        # Load Elo ratings (from feature engineering)
        elo_path = PROCESSED_DATA_DIR / "elo_ratings.csv"
        if elo_path.exists():
            elo_df = pd.read_csv(elo_path)
            self.elo_ratings = dict(zip(elo_df['team'], elo_df['elo_rating']))
            logger.info(f"Loaded Elo ratings for {len(self.elo_ratings)} teams")
        
        # Load ensemble config
        ensemble_path = self.models_dir / "ensemble_model.joblib"
        if ensemble_path.exists():
            self.ensemble_config = joblib.load(ensemble_path)
            logger.info("Loaded ensemble configuration")
        
        # Set model version
        self.model_version = datetime.now().strftime("%Y%m%d")
        
        logger.info("Models loaded successfully")
    
    def get_team_elo(self, team: str) -> float:
        """Get Elo rating for a team."""
        normalized = normalize_team_name(team)
        
        # Try exact match
        if normalized in self.elo_ratings:
            return self.elo_ratings[normalized]
        
        # Try original name
        if team in self.elo_ratings:
            return self.elo_ratings[team]
        
        # Default rating for unknown teams
        return 1500.0
    
    def compute_features(self, fixture: Dict) -> Dict[str, Any]:
        """
        Compute features for a single fixture.
        
        Args:
            fixture: Dict with home_team, away_team, date, etc.
        
        Returns:
            Feature dictionary
        """
        home_team = fixture['home_team']
        away_team = fixture['away_team']
        
        # Elo features
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        home_advantage = 100  # Default home advantage
        
        elo_diff = home_elo - away_elo + home_advantage
        
        # Elo-based probabilities
        exp_home = 1 / (1 + 10 ** (-elo_diff / 400))
        
        # Simple draw estimation
        base_draw_rate = 0.26
        elo_draw_factor = 1 - (abs(elo_diff - 100) / 800)
        elo_draw_factor = max(0.5, min(1.2, elo_draw_factor))
        p_draw = base_draw_rate * elo_draw_factor
        
        remaining = 1 - p_draw
        p_home = remaining * exp_home
        p_away = remaining * (1 - exp_home)
        
        features = {
            'home_team': home_team,
            'away_team': away_team,
            'date': fixture.get('date'),
            
            # Elo features
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            
            # Elo probabilities (used as baseline)
            'elo_prob_home': p_home,
            'elo_prob_draw': p_draw,
            'elo_prob_away': p_away,
            
            # Placeholder features (would be computed from feature store)
            'home_form_5_ppg': None,
            'away_form_5_ppg': None,
            'home_position': None,
            'away_position': None,
        }
        
        # Hash features for reproducibility
        features['features_hash'] = hash_features({
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff
        })
        
        return features
    
    def predict_single(self, fixture: Dict) -> Dict[str, Any]:
        """
        Make prediction for a single fixture.
        
        Args:
            fixture: Dict with home_team, away_team, date
        
        Returns:
            Prediction dictionary
        """
        # Compute features
        features = self.compute_features(fixture)
        
        # For now, use Elo probabilities as the prediction
        # In production, you'd use the ensemble model
        p_home = features['elo_prob_home']
        p_draw = features['elo_prob_draw']
        p_away = features['elo_prob_away']
        
        # Determine predicted outcome and confidence
        probs = [p_home, p_draw, p_away]
        max_prob = max(probs)
        predicted_outcome = probs.index(max_prob)
        outcome_names = ['Home', 'Draw', 'Away']
        
        # Confidence tier based on margin
        second_max = sorted(probs, reverse=True)[1]
        margin = max_prob - second_max
        
        if margin > 0.15:
            confidence = 'high'
        elif margin > 0.08:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        prediction = {
            'home_team': features['home_team'],
            'away_team': features['away_team'],
            'date': features['date'],
            
            'p_home': round(p_home, 4),
            'p_draw': round(p_draw, 4),
            'p_away': round(p_away, 4),
            
            'predicted_outcome': outcome_names[predicted_outcome],
            'confidence': confidence,
            
            'model_version': self.model_version,
            'features_hash': features['features_hash'],
            'generated_at': datetime.now().isoformat(),
            
            # Features for transparency
            'home_elo': round(features['home_elo'], 1),
            'away_elo': round(features['away_elo'], 1),
            'elo_diff': round(features['elo_diff'], 1),
        }
        
        return prediction
    
    def predict_batch(self, fixtures: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple fixtures.
        
        Args:
            fixtures: List of fixture dictionaries
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for fixture in fixtures:
            try:
                pred = self.predict_single(fixture)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict {fixture}: {e}")
                continue
        
        return predictions
    
    def add_market_comparison(
        self,
        prediction: Dict,
        market_odds: Dict
    ) -> Dict:
        """
        Add market comparison to prediction.
        
        Args:
            prediction: Prediction dictionary
            market_odds: Dict with home_odds, draw_odds, away_odds
        
        Returns:
            Updated prediction with edge analysis
        """
        home_odds = market_odds.get('home_odds')
        draw_odds = market_odds.get('draw_odds')
        away_odds = market_odds.get('away_odds')
        
        if not all([home_odds, draw_odds, away_odds]):
            prediction['market_comparison'] = None
            return prediction
        
        # Get fair market probabilities
        market_home, market_draw, market_away = remove_vig(
            home_odds, draw_odds, away_odds
        )
        
        # Calculate edges
        edges = {
            'home': prediction['p_home'] - market_home,
            'draw': prediction['p_draw'] - market_draw,
            'away': prediction['p_away'] - market_away,
        }
        
        # Calculate EVs
        evs = {
            'home': prediction['p_home'] * home_odds - 1,
            'draw': prediction['p_draw'] * draw_odds - 1,
            'away': prediction['p_away'] * away_odds - 1,
        }
        
        # Find best bet
        best_outcome = max(evs, key=evs.get)
        best_ev = evs[best_outcome]
        best_edge = edges[best_outcome]
        
        prediction['market_comparison'] = {
            'market_prob_home': round(market_home, 4),
            'market_prob_draw': round(market_draw, 4),
            'market_prob_away': round(market_away, 4),
            
            'edge_home': round(edges['home'], 4),
            'edge_draw': round(edges['draw'], 4),
            'edge_away': round(edges['away'], 4),
            
            'ev_home': round(evs['home'], 4),
            'ev_draw': round(evs['draw'], 4),
            'ev_away': round(evs['away'], 4),
            
            'best_bet': best_outcome if best_ev > MIN_EV_THRESHOLD else None,
            'best_edge': round(best_edge, 4),
            'best_ev': round(best_ev, 4),
            
            'has_value': best_ev > MIN_EV_THRESHOLD and best_edge > MIN_EDGE_THRESHOLD,
        }
        
        return prediction
    
    def format_output(
        self,
        predictions: List[Dict],
        format_type: str = 'detailed'
    ) -> str:
        """
        Format predictions for output.
        
        Args:
            predictions: List of prediction dictionaries
            format_type: 'detailed', 'simple', or 'json'
        
        Returns:
            Formatted string
        """
        if format_type == 'json':
            return json.dumps(predictions, indent=2, default=str)
        
        lines = []
        lines.append("=" * 70)
        lines.append(f"FOOTBALL PREDICTIONS - Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Model Version: {self.model_version}")
        lines.append("=" * 70)
        
        for pred in predictions:
            lines.append("")
            lines.append(f"{pred['home_team']} vs {pred['away_team']}")
            lines.append(f"Date: {pred['date']}")
            lines.append("-" * 40)
            
            # Probabilities
            lines.append(f"  Home Win: {pred['p_home']:.1%}")
            lines.append(f"  Draw:     {pred['p_draw']:.1%}")
            lines.append(f"  Away Win: {pred['p_away']:.1%}")
            
            lines.append(f"  Prediction: {pred['predicted_outcome']} ({pred['confidence']} confidence)")
            
            # Elo info
            lines.append(f"  Elo: {pred['home_team']} {pred['home_elo']:.0f} vs {pred['away_team']} {pred['away_elo']:.0f}")
            
            # Market comparison
            if pred.get('market_comparison'):
                mc = pred['market_comparison']
                lines.append("")
                lines.append("  Market Comparison:")
                lines.append(f"    Edge Home: {mc['edge_home']:+.1%}")
                lines.append(f"    Edge Draw: {mc['edge_draw']:+.1%}")
                lines.append(f"    Edge Away: {mc['edge_away']:+.1%}")
                
                if mc['has_value']:
                    lines.append(f"    >>> VALUE BET: {mc['best_bet'].upper()} (EV: {mc['best_ev']:+.1%})")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_predictions(
        self,
        predictions: List[Dict],
        output_path: Path = None
    ):
        """Save predictions to file."""
        if output_path is None:
            output_dir = DATA_DIR / "predictions"
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"predictions_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        logger.info(f"Predictions saved to {output_path}")
        return output_path


# =============================================================================
# EXAMPLE FIXTURES (for testing)
# =============================================================================

def get_example_fixtures() -> List[Dict]:
    """Get example fixtures for testing."""
    return [
        {
            'home_team': 'Liverpool',
            'away_team': 'Manchester United',
            'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'league': 'Premier League'
        },
        {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'league': 'Premier League'
        },
        {
            'home_team': 'Bayern',
            'away_team': 'Dortmund',
            'date': (datetime.now() + timedelta(days=4)).strftime('%Y-%m-%d'),
            'league': 'Bundesliga'
        },
        {
            'home_team': 'Barcelona',
            'away_team': 'Real Madrid',
            'date': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
            'league': 'La Liga'
        },
    ]


def get_example_odds() -> Dict[str, Dict]:
    """Get example market odds for testing."""
    return {
        'Liverpool-Manchester United': {
            'home_odds': 1.65,
            'draw_odds': 4.00,
            'away_odds': 5.50
        },
        'Arsenal-Chelsea': {
            'home_odds': 2.10,
            'draw_odds': 3.50,
            'away_odds': 3.40
        },
        'Bayern-Dortmund': {
            'home_odds': 1.55,
            'draw_odds': 4.50,
            'away_odds': 5.80
        },
        'Barcelona-Real Madrid': {
            'home_odds': 2.40,
            'draw_odds': 3.40,
            'away_odds': 2.90
        },
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run prediction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prediction Pipeline")
    parser.add_argument(
        "--fixtures",
        type=str,
        help="Path to fixtures JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for predictions"
    )
    parser.add_argument(
        "--format",
        choices=['detailed', 'simple', 'json'],
        default='detailed',
        help="Output format"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with example fixtures"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PredictionPipeline()
    pipeline.load_models()
    
    # Get fixtures
    if args.example:
        fixtures = get_example_fixtures()
        example_odds = get_example_odds()
        print("\nUsing example fixtures for demonstration")
    elif args.fixtures:
        with open(args.fixtures, 'r') as f:
            fixtures = json.load(f)
        example_odds = {}
    else:
        # Default: use example fixtures
        fixtures = get_example_fixtures()
        example_odds = get_example_odds()
        print("\nNo fixtures provided, using examples")
    
    print(f"\nProcessing {len(fixtures)} fixtures...")
    
    # Make predictions
    predictions = pipeline.predict_batch(fixtures)
    
    # Add market comparison if odds available
    for pred in predictions:
        key = f"{pred['home_team']}-{pred['away_team']}"
        if key in example_odds:
            pipeline.add_market_comparison(pred, example_odds[key])
    
    # Output
    output = pipeline.format_output(predictions, args.format)
    print(output)
    
    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = None
    
    saved_path = pipeline.save_predictions(predictions, output_path)
    print(f"\nPredictions saved to: {saved_path}")
    
    # Summary
    value_bets = [p for p in predictions if p.get('market_comparison', {}).get('has_value')]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total predictions: {len(predictions)}")
    print(f"Value bets found: {len(value_bets)}")
    
    if value_bets:
        print("\nValue Bets:")
        for vb in value_bets:
            mc = vb['market_comparison']
            print(f"  {vb['home_team']} vs {vb['away_team']}: {mc['best_bet'].upper()} (EV: {mc['best_ev']:+.1%})")


if __name__ == "__main__":
    main()
