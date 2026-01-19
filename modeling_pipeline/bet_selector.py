"""
Sophisticated Bet Selection System
===================================

Evaluates predictions and decides which bets to place based on:
- Confidence levels
- Expected value (EV)
- Kelly Criterion
- Risk management
- Historical performance patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class BetSelector:
    """Sophisticated bet selection engine."""

    def __init__(
        self,
        min_confidence: float = 0.50,
        min_edge: float = 0.0,
        max_stake: float = 50.0,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25
    ):
        """
        Initialize bet selector.

        Args:
            min_confidence: Minimum probability to consider betting (0-1)
            min_edge: Minimum edge required (model_prob - implied_prob)
            max_stake: Maximum stake per bet
            bankroll: Total bankroll for Kelly calculations
            kelly_fraction: Fraction of Kelly stake to use (0.25 = quarter Kelly)
        """
        self.min_confidence = min_confidence
        self.min_edge = min_edge
        self.max_stake = max_stake
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction

        # Historical performance tracking
        self.performance_by_type = {}
        self.performance_by_confidence = {}

    def calculate_edge(
        self,
        model_prob: float,
        implied_odds: float
    ) -> float:
        """
        Calculate betting edge.

        Edge = Model Probability - Implied Probability from Odds

        Args:
            model_prob: Model's predicted probability
            implied_odds: Fair odds (1/probability)

        Returns:
            Edge as decimal (positive = value bet)
        """
        implied_prob = 1 / implied_odds
        return model_prob - implied_prob

    def calculate_kelly_stake(
        self,
        model_prob: float,
        odds: float,
        bankroll: float
    ) -> float:
        """
        Calculate optimal Kelly Criterion stake.

        Kelly % = (odds * model_prob - 1) / (odds - 1)

        Args:
            model_prob: Model's predicted probability
            odds: Decimal odds
            bankroll: Available bankroll

        Returns:
            Optimal stake amount
        """
        if odds <= 1.0 or model_prob <= 0:
            return 0.0

        # Kelly formula
        kelly_pct = (odds * model_prob - 1) / (odds - 1)

        # Apply fractional Kelly for safety
        kelly_pct = kelly_pct * self.kelly_fraction

        # Ensure non-negative
        kelly_pct = max(0, kelly_pct)

        # Calculate stake
        stake = kelly_pct * bankroll

        # Cap at max stake
        stake = min(stake, self.max_stake)

        return stake

    def calculate_expected_value(
        self,
        model_prob: float,
        odds: float,
        stake: float
    ) -> float:
        """
        Calculate expected value of a bet.

        EV = (model_prob * profit) - ((1 - model_prob) * stake)

        Args:
            model_prob: Model's predicted probability
            odds: Decimal odds
            stake: Stake amount

        Returns:
            Expected value in currency units
        """
        profit = stake * (odds - 1)
        loss = stake
        ev = (model_prob * profit) - ((1 - model_prob) * loss)
        return ev

    def should_bet(
        self,
        predicted_outcome: str,
        home_prob: float,
        draw_prob: float,
        away_prob: float,
        home_odds: float,
        draw_odds: float,
        away_odds: float
    ) -> Dict:
        """
        Decide whether to place a bet and calculate stake.

        Args:
            predicted_outcome: Model's prediction ('Home Win', 'Draw', 'Away Win')
            home_prob: Predicted home win probability
            draw_prob: Predicted draw probability
            away_prob: Predicted away win probability
            home_odds: Fair odds for home win
            draw_odds: Fair odds for draw
            away_odds: Fair odds for away win

        Returns:
            Dict with decision, stake, edge, ev, and reasoning
        """
        # Map outcome to probability and odds
        prob_map = {
            'Home Win': (home_prob, home_odds),
            'Draw': (draw_prob, draw_odds),
            'Away Win': (away_prob, away_odds)
        }

        model_prob, odds = prob_map[predicted_outcome]

        # Calculate metrics
        edge = self.calculate_edge(model_prob, odds)
        kelly_stake = self.calculate_kelly_stake(model_prob, odds, self.bankroll)
        ev = self.calculate_expected_value(model_prob, odds, kelly_stake)

        # Decision criteria
        reasons = []

        # Criterion 1: Minimum confidence
        if model_prob < self.min_confidence:
            return {
                'bet': False,
                'stake': 0.0,
                'model_prob': model_prob,
                'odds': odds,
                'edge': edge,
                'ev': ev,
                'reason': f'Low confidence ({model_prob:.1%} < {self.min_confidence:.1%})'
            }

        reasons.append(f'Confidence: {model_prob:.1%}')

        # Criterion 2: Minimum edge
        if edge < self.min_edge:
            return {
                'bet': False,
                'stake': 0.0,
                'model_prob': model_prob,
                'odds': odds,
                'edge': edge,
                'ev': ev,
                'reason': f'Insufficient edge ({edge:+.1%} < {self.min_edge:+.1%})'
            }

        reasons.append(f'Edge: {edge:+.1%}')

        # Criterion 3: Positive EV
        if ev <= 0:
            return {
                'bet': False,
                'stake': 0.0,
                'model_prob': model_prob,
                'odds': odds,
                'edge': edge,
                'ev': ev,
                'reason': f'Negative EV (£{ev:.2f})'
            }

        reasons.append(f'EV: £{ev:+.2f}')

        # Criterion 4: Kelly stake > minimum
        if kelly_stake < 1.0:  # Minimum £1 bet
            return {
                'bet': False,
                'stake': 0.0,
                'model_prob': model_prob,
                'odds': odds,
                'edge': edge,
                'ev': ev,
                'reason': f'Kelly stake too small (£{kelly_stake:.2f})'
            }

        # All criteria passed - PLACE BET
        return {
            'bet': True,
            'stake': round(kelly_stake, 2),
            'model_prob': model_prob,
            'odds': odds,
            'edge': edge,
            'ev': ev,
            'reason': ' | '.join(reasons)
        }


class BettingStrategy:
    """Pre-configured betting strategies."""

    @staticmethod
    def conservative() -> BetSelector:
        """High confidence, low risk strategy."""
        return BetSelector(
            min_confidence=0.60,  # 60%+ confidence
            min_edge=0.05,  # 5%+ edge
            max_stake=20.0,
            bankroll=1000.0,
            kelly_fraction=0.15  # Very conservative Kelly
        )

    @staticmethod
    def value_betting() -> BetSelector:
        """Focus on positive expected value."""
        return BetSelector(
            min_confidence=0.45,  # Lower confidence OK if value exists
            min_edge=0.10,  # 10%+ edge required
            max_stake=30.0,
            bankroll=1000.0,
            kelly_fraction=0.25  # Quarter Kelly
        )

    @staticmethod
    def kelly_optimal() -> BetSelector:
        """Optimal growth using Kelly Criterion."""
        return BetSelector(
            min_confidence=0.50,
            min_edge=0.02,  # Any positive edge
            max_stake=50.0,
            bankroll=1000.0,
            kelly_fraction=0.25  # Quarter Kelly for safety
        )

    @staticmethod
    def aggressive() -> BetSelector:
        """Higher risk, higher reward."""
        return BetSelector(
            min_confidence=0.48,
            min_edge=0.0,  # No edge requirement
            max_stake=50.0,
            bankroll=1000.0,
            kelly_fraction=0.35  # More aggressive Kelly
        )

    @staticmethod
    def away_specialist() -> BetSelector:
        """Conservative but focuses on away wins (proven profitable)."""
        return BetSelector(
            min_confidence=0.50,
            min_edge=0.03,
            max_stake=40.0,
            bankroll=1000.0,
            kelly_fraction=0.30
        )


def analyze_bet_selection(
    predictions_df: pd.DataFrame,
    selector: BetSelector,
    strategy_name: str = "Custom"
) -> pd.DataFrame:
    """
    Analyze which bets would be selected by the strategy.

    Args:
        predictions_df: DataFrame with predictions (from CSV)
        selector: BetSelector instance
        strategy_name: Name of strategy for reporting

    Returns:
        DataFrame with bet decisions and analysis
    """
    results = []

    for _, row in predictions_df.iterrows():
        # Calculate fair odds from probabilities
        home_odds = 1 / row['home_win_prob']
        draw_odds = 1 / row['draw_prob']
        away_odds = 1 / row['away_win_prob']

        # Get bet decision
        decision = selector.should_bet(
            row['predicted_outcome'],
            row['home_win_prob'],
            row['draw_prob'],
            row['away_win_prob'],
            home_odds,
            draw_odds,
            away_odds
        )

        # Add to results
        results.append({
            'date': row['date'],
            'league': row['league'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'predicted': row['predicted_outcome'],
            'home_prob': row['home_win_prob'],
            'draw_prob': row['draw_prob'],
            'away_prob': row['away_win_prob'],
            'bet_decision': decision['bet'],
            'stake': decision['stake'],
            'odds': decision['odds'],
            'model_prob': decision['model_prob'],
            'edge': decision['edge'],
            'expected_value': decision['ev'],
            'reason': decision['reason'],
            'strategy': strategy_name
        })

    return pd.DataFrame(results)


def compare_strategies(all_bets_csv: str) -> pd.DataFrame:
    """
    Compare different betting strategies on historical data.

    Args:
        all_bets_csv: Path to CSV with bet outcomes

    Returns:
        Comparison DataFrame
    """
    # Load historical bets with actual outcomes
    df = pd.read_csv(all_bets_csv)

    strategies = {
        'Conservative': BettingStrategy.conservative(),
        'Value Betting': BettingStrategy.value_betting(),
        'Kelly Optimal': BettingStrategy.kelly_optimal(),
        'Aggressive': BettingStrategy.aggressive(),
        'Away Specialist': BettingStrategy.away_specialist()
    }

    strategy_results = []

    for name, selector in strategies.items():
        # Track performance
        total_bets = 0
        total_staked = 0.0
        total_return = 0.0
        winning_bets = 0

        for _, row in df.iterrows():
            # Get odds (fair odds from probabilities)
            if row['predicted'] == 'Home Win':
                odds = 1 / row['home_prob']
                model_prob = row['home_prob']
            elif row['predicted'] == 'Draw':
                odds = 1 / row['draw_prob']
                model_prob = row['draw_prob']
            else:
                odds = 1 / row['away_prob']
                model_prob = row['away_prob']

            # Filter away specialist to only away wins
            if name == 'Away Specialist' and row['predicted'] != 'Away Win':
                continue

            # Get decision
            decision = selector.should_bet(
                row['predicted'],
                row['home_prob'],
                row['draw_prob'],
                row['away_prob'],
                1 / row['home_prob'],
                1 / row['draw_prob'],
                1 / row['away_prob']
            )

            if decision['bet']:
                total_bets += 1
                stake = decision['stake']
                total_staked += stake

                # Calculate actual return
                if row['correct']:
                    profit = stake * (odds - 1)
                    total_return += profit
                    winning_bets += 1
                else:
                    total_return -= stake

        # Calculate metrics
        roi = (total_return / total_staked * 100) if total_staked > 0 else 0
        win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0

        strategy_results.append({
            'Strategy': name,
            'Total Bets': total_bets,
            'Total Staked': total_staked,
            'Net P&L': total_return,
            'ROI %': roi,
            'Winning Bets': winning_bets,
            'Win Rate %': win_rate
        })

    return pd.DataFrame(strategy_results)


if __name__ == '__main__':
    print("Bet Selection System Loaded!")
    print("\nAvailable Strategies:")
    print("  - Conservative: High confidence (60%+), 5%+ edge")
    print("  - Value Betting: 10%+ edge required")
    print("  - Kelly Optimal: Quarter Kelly with any positive edge")
    print("  - Aggressive: Lower thresholds, higher stakes")
    print("  - Away Specialist: Conservative away-only strategy")
