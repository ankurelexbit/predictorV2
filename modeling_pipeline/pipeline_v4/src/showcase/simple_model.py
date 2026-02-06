"""
Sports Analytics Made Easy - Simple 1X2 Prediction Model

A showcase model that uses interpretable features to predict
Home Win / Draw / Away Win outcomes.

Key Principles:
1. Simple, interpretable features (29 core features)
2. Market-aware (uses bookmaker odds for edge detection)
3. Transparent predictions with confidence levels
4. Easy to understand value signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from catboost import CatBoostClassifier
import joblib


@dataclass
class CoreFeatures:
    """29 interpretable core features for 1X2 prediction"""

    TEAM_STRENGTH = [
        'home_elo', 'away_elo', 'elo_diff',
    ]

    LEAGUE_POSITION = [
        'home_league_position', 'away_league_position', 'position_diff',
    ]

    RECENT_FORM = [
        'home_points_last_5', 'away_points_last_5',
        'home_wins_last_5', 'away_wins_last_5',
        'home_draws_last_5', 'away_draws_last_5',
    ]

    GOALS = [
        'home_goals_scored_last_5', 'away_goals_scored_last_5',
        'home_goals_conceded_last_5', 'away_goals_conceded_last_5',
    ]

    EXPECTED_GOALS = [
        'home_derived_xg_per_match_5', 'away_derived_xg_per_match_5',
        'home_derived_xga_per_match_5', 'away_derived_xga_per_match_5',
    ]

    HEAD_TO_HEAD = [
        'h2h_home_wins_last_5', 'h2h_draws_last_5', 'h2h_away_wins_last_5',
    ]

    HOME_ADVANTAGE = [
        'home_home_win_pct', 'away_away_win_pct', 'home_advantage_strength',
    ]

    CONTEXT = [
        'home_days_since_last_match', 'away_days_since_last_match',
        'is_derby_match',
    ]

    @classmethod
    def all_features(cls) -> List[str]:
        """Return all 29 core features"""
        return (
            cls.TEAM_STRENGTH +
            cls.LEAGUE_POSITION +
            cls.RECENT_FORM +
            cls.GOALS +
            cls.EXPECTED_GOALS +
            cls.HEAD_TO_HEAD +
            cls.HOME_ADVANTAGE +
            cls.CONTEXT
        )

    @classmethod
    def feature_categories(cls) -> Dict[str, List[str]]:
        """Return features grouped by category"""
        return {
            'Team Strength': cls.TEAM_STRENGTH,
            'League Position': cls.LEAGUE_POSITION,
            'Recent Form': cls.RECENT_FORM,
            'Goals': cls.GOALS,
            'Expected Goals': cls.EXPECTED_GOALS,
            'Head to Head': cls.HEAD_TO_HEAD,
            'Home Advantage': cls.HOME_ADVANTAGE,
            'Context': cls.CONTEXT,
        }


@dataclass
class MarketFeatures:
    """Features derived from bookmaker odds"""

    FEATURES = [
        'market_home_prob',       # Implied probability from best odds
        'market_draw_prob',
        'market_away_prob',
        'bookmaker_disagreement', # Max - min implied prob across books
        'market_overround',       # Total implied prob (>100% = margin)
    ]

    @staticmethod
    def extract_from_odds(
        home_odds: List[float],
        draw_odds: List[float],
        away_odds: List[float]
    ) -> Dict[str, float]:
        """Extract market features from bookmaker odds lists"""

        # Best odds (highest = best for bettor)
        best_home = max(home_odds) if home_odds else 0
        best_draw = max(draw_odds) if draw_odds else 0
        best_away = max(away_odds) if away_odds else 0

        # Implied probabilities from best odds
        home_prob = 1/best_home if best_home > 0 else 0
        draw_prob = 1/best_draw if best_draw > 0 else 0
        away_prob = 1/best_away if best_away > 0 else 0

        # Bookmaker disagreement (uncertainty signal)
        home_probs = [1/o for o in home_odds if o > 0]
        disagreement = max(home_probs) - min(home_probs) if len(home_probs) > 1 else 0

        # Market overround
        overround = home_prob + draw_prob + away_prob

        return {
            'market_home_prob': home_prob,
            'market_draw_prob': draw_prob,
            'market_away_prob': away_prob,
            'bookmaker_disagreement': disagreement,
            'market_overround': overround,
        }


@dataclass
class Prediction:
    """A single match prediction with explanation"""
    fixture_id: int
    home_team: str
    away_team: str
    match_date: str

    # Model probabilities
    prob_home: float
    prob_draw: float
    prob_away: float

    # Market probabilities (from odds)
    market_prob_home: float
    market_prob_draw: float
    market_prob_away: float

    # Value signals
    edge_home: float  # model_prob - market_prob
    edge_draw: float
    edge_away: float

    # Recommendation
    best_bet: Optional[str]  # 'Home', 'Draw', 'Away', or None
    confidence: str  # 'High', 'Medium', 'Low'
    edge: float

    def to_dict(self) -> Dict:
        return {
            'fixture_id': self.fixture_id,
            'match': f'{self.home_team} vs {self.away_team}',
            'date': self.match_date,
            'model_probs': {
                'home': round(self.prob_home, 3),
                'draw': round(self.prob_draw, 3),
                'away': round(self.prob_away, 3),
            },
            'market_probs': {
                'home': round(self.market_prob_home, 3),
                'draw': round(self.market_prob_draw, 3),
                'away': round(self.market_prob_away, 3),
            },
            'edges': {
                'home': round(self.edge_home, 3),
                'draw': round(self.edge_draw, 3),
                'away': round(self.edge_away, 3),
            },
            'recommendation': {
                'bet': self.best_bet,
                'confidence': self.confidence,
                'edge': round(self.edge, 3) if self.edge else None,
            }
        }


class SimpleModel:
    """
    Sports Analytics Made Easy - Simple 1X2 Model

    Uses 29 core interpretable features + market features
    to predict Home/Draw/Away outcomes and identify value bets.
    """

    # Target encoding
    TARGET_MAP = {'H': 2, 'D': 1, 'A': 0}
    TARGET_REVERSE = {2: 'Home', 1: 'Draw', 0: 'Away'}

    # Confidence thresholds
    HIGH_CONFIDENCE_EDGE = 0.08   # 8%+ edge
    MEDIUM_CONFIDENCE_EDGE = 0.05  # 5%+ edge
    MIN_BET_EDGE = 0.03           # 3% minimum edge to bet

    def __init__(self):
        self.model = None
        self.feature_names = CoreFeatures.all_features()

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model on historical data.

        Args:
            train_df: Training data with features and 'result' column
            val_df: Optional validation data
            verbose: Print training progress

        Returns:
            Training metrics dict
        """
        # Prepare features
        X_train = train_df[self.feature_names].copy()
        y_train = train_df['result'].map(self.TARGET_MAP)

        # Handle missing values
        X_train = X_train.fillna(0)

        # Initialize CatBoost with simple parameters
        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50 if val_df is not None else None,
        )

        # Prepare validation set if provided
        eval_set = None
        if val_df is not None:
            X_val = val_df[self.feature_names].fillna(0)
            y_val = val_df['result'].map(self.TARGET_MAP)
            eval_set = (X_val, y_val)

        # Train
        if verbose:
            print(f"Training on {len(X_train):,} matches with {len(self.feature_names)} features...")

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)

        # Calculate training metrics
        train_probs = self.model.predict_proba(X_train)
        train_preds = train_probs.argmax(axis=1)
        train_acc = (train_preds == y_train).mean()

        metrics = {
            'train_accuracy': train_acc,
            'train_samples': len(X_train),
            'features_used': len(self.feature_names),
        }

        if val_df is not None:
            val_probs = self.model.predict_proba(X_val)
            val_preds = val_probs.argmax(axis=1)
            val_acc = (val_preds == y_val).mean()
            metrics['val_accuracy'] = val_acc
            metrics['val_samples'] = len(X_val)

        if verbose:
            print(f"Training accuracy: {train_acc:.1%}")
            if val_df is not None:
                print(f"Validation accuracy: {val_acc:.1%}")

        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get raw probabilities for each outcome.

        Returns:
            Array of shape (n_samples, 3) with [P(Away), P(Draw), P(Home)]
        """
        X = df[self.feature_names].fillna(0)
        return self.model.predict_proba(X)

    def predict_with_market(
        self,
        df: pd.DataFrame,
        home_odds: List[float],
        draw_odds: List[float],
        away_odds: List[float],
    ) -> Prediction:
        """
        Make a prediction for a single match with market context.

        Args:
            df: Single row DataFrame with features
            home_odds: List of home odds from different bookmakers
            draw_odds: List of draw odds
            away_odds: List of away odds

        Returns:
            Prediction object with full analysis
        """
        # Get model probabilities
        probs = self.predict_proba(df)[0]
        prob_away, prob_draw, prob_home = probs

        # Get market probabilities
        market = MarketFeatures.extract_from_odds(home_odds, draw_odds, away_odds)
        market_home = market['market_home_prob']
        market_draw = market['market_draw_prob']
        market_away = market['market_away_prob']

        # Calculate edges
        edge_home = prob_home - market_home
        edge_draw = prob_draw - market_draw
        edge_away = prob_away - market_away

        # Find best bet
        edges = {'Home': edge_home, 'Draw': edge_draw, 'Away': edge_away}
        best_outcome = max(edges, key=edges.get)
        best_edge = edges[best_outcome]

        # Determine confidence and recommendation
        if best_edge >= self.HIGH_CONFIDENCE_EDGE:
            confidence = 'High'
            best_bet = best_outcome
        elif best_edge >= self.MEDIUM_CONFIDENCE_EDGE:
            confidence = 'Medium'
            best_bet = best_outcome
        elif best_edge >= self.MIN_BET_EDGE:
            confidence = 'Low'
            best_bet = best_outcome
        else:
            confidence = 'None'
            best_bet = None
            best_edge = 0

        return Prediction(
            fixture_id=int(df['fixture_id'].iloc[0]) if 'fixture_id' in df.columns else 0,
            home_team=str(df['home_team_id'].iloc[0]) if 'home_team_id' in df.columns else 'Home',
            away_team=str(df['away_team_id'].iloc[0]) if 'away_team_id' in df.columns else 'Away',
            match_date=str(df['match_date'].iloc[0]) if 'match_date' in df.columns else '',
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
            market_prob_home=market_home,
            market_prob_draw=market_draw,
            market_prob_away=market_away,
            edge_home=edge_home,
            edge_draw=edge_draw,
            edge_away=edge_away,
            best_bet=best_bet,
            confidence=confidence,
            edge=best_edge,
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance ranked by impact"""
        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)

    def explain_prediction(self, prediction: Prediction) -> str:
        """Generate human-readable explanation of a prediction"""
        lines = [
            f"Match: {prediction.home_team} vs {prediction.away_team}",
            f"Date: {prediction.match_date}",
            "",
            "Model Probabilities:",
            f"  Home Win: {prediction.prob_home:.1%}",
            f"  Draw:     {prediction.prob_draw:.1%}",
            f"  Away Win: {prediction.prob_away:.1%}",
            "",
            "Market Probabilities (from odds):",
            f"  Home Win: {prediction.market_prob_home:.1%}",
            f"  Draw:     {prediction.market_prob_draw:.1%}",
            f"  Away Win: {prediction.market_prob_away:.1%}",
            "",
            "Value Edges (model - market):",
            f"  Home Win: {prediction.edge_home:+.1%}",
            f"  Draw:     {prediction.edge_draw:+.1%}",
            f"  Away Win: {prediction.edge_away:+.1%}",
            "",
        ]

        if prediction.best_bet:
            lines.extend([
                f"Recommendation: {prediction.best_bet}",
                f"Confidence: {prediction.confidence}",
                f"Edge: {prediction.edge:.1%}",
            ])
        else:
            lines.append("Recommendation: No value bet identified")

        return "\n".join(lines)

    def save(self, path: str):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
        }, path)

    def load(self, path: str):
        """Load model from file"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']


def train_showcase_model(
    data_path: str = 'data/training_data.csv',
    output_path: str = 'models/showcase_simple_v1.joblib',
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[SimpleModel, Dict]:
    """
    Train the showcase model on historical data.

    Uses chronological split (no data leakage).
    """
    print("=" * 60)
    print("Sports Analytics Made Easy - Training Simple 1X2 Model")
    print("=" * 60)
    print()

    # Load data
    df = pd.read_csv(data_path)
    df = df.sort_values('match_date').reset_index(drop=True)

    print(f"Loaded {len(df):,} matches")
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
    print()

    # Chronological split
    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df):,} matches (up to {train_df['match_date'].max()})")
    print(f"Val:   {len(val_df):,} matches")
    print(f"Test:  {len(test_df):,} matches (from {test_df['match_date'].min()})")
    print()

    # Train model
    model = SimpleModel()
    metrics = model.train(train_df, val_df, verbose=True)

    # Evaluate on test set
    print("\n" + "=" * 40)
    print("Test Set Evaluation")
    print("=" * 40)

    test_probs = model.predict_proba(test_df)
    test_preds = test_probs.argmax(axis=1)
    test_actual = test_df['result'].map(SimpleModel.TARGET_MAP)

    test_acc = (test_preds == test_actual).mean()
    print(f"Test accuracy: {test_acc:.1%}")

    # Per-outcome accuracy
    for outcome, code in SimpleModel.TARGET_MAP.items():
        mask = test_actual == code
        if mask.sum() > 0:
            outcome_acc = (test_preds[mask] == code).mean()
            print(f"  {outcome}: {outcome_acc:.1%} ({mask.sum()} samples)")

    metrics['test_accuracy'] = test_acc
    metrics['test_samples'] = len(test_df)

    # Feature importance
    print("\n" + "=" * 40)
    print("Top 10 Most Important Features")
    print("=" * 40)
    importance = model.get_feature_importance()
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.1f}")

    # Save model
    model.save(output_path)
    print(f"\nModel saved to: {output_path}")

    return model, metrics


if __name__ == '__main__':
    train_showcase_model()
