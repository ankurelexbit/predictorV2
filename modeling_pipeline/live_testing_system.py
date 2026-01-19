"""
Live Testing System - True API-Based Predictions
=================================================

Comprehensive system for testing model on real upcoming matches:
1. Fetch upcoming matches from SportMonks API
2. Generate live predictions using real-time data
3. Track predictions and wait for results
4. Evaluate performance and betting strategy
5. Generate reports

Usage:
    # Make predictions for today's matches
    python live_testing_system.py --predict-today

    # Update results for pending predictions
    python live_testing_system.py --update-results

    # Generate performance report
    python live_testing_system.py --report

    # Full workflow: predict + check results + report
    python live_testing_system.py --full
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR
from utils import setup_logger, calculate_log_loss
import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup
logger = setup_logger("live_testing")

# Sportmonks API configuration
API_KEY = "LmEYRdsf8CSmiblNVTn6JfV0y0s8tc4aQsEkJ6JuoBhOWa3Hd1FEanGcrijo"
BASE_URL = "https://api.sportmonks.com/v3/football"

# Target leagues (from training)
TRAINING_LEAGUES = {
    8: "Premier League",
    9: "Championship",
    564: "La Liga",
    82: "Bundesliga",
    384: "Serie A",
    301: "Ligue 1"
}

# Live predictions tracking file
PREDICTIONS_DIR = Path(__file__).parent / "live_predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)


class LivePredictionTracker:
    """Track live predictions and results."""

    def __init__(self, tracking_file: Path = PREDICTIONS_DIR / "live_predictions_tracker.json"):
        self.tracking_file = tracking_file
        self.predictions = self._load_predictions()

    def _load_predictions(self) -> Dict:
        """Load predictions from tracking file."""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {"predictions": []}

    def _save_predictions(self):
        """Save predictions to tracking file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)

    def add_prediction(self, prediction: Dict):
        """Add a new prediction."""
        prediction['prediction_time'] = datetime.now().isoformat()
        prediction['status'] = 'pending'
        self.predictions['predictions'].append(prediction)
        self._save_predictions()
        logger.info(f"Added prediction: {prediction['home_team']} vs {prediction['away_team']}")

    def update_result(self, fixture_id: int, result: Dict):
        """Update prediction with actual result."""
        for pred in self.predictions['predictions']:
            if pred['fixture_id'] == fixture_id:
                pred['status'] = 'completed'
                pred['result_time'] = datetime.now().isoformat()
                pred['actual_home_goals'] = result['home_goals']
                pred['actual_away_goals'] = result['away_goals']

                # Determine actual outcome
                if result['home_goals'] > result['away_goals']:
                    pred['actual_outcome'] = 'Home Win'
                    pred['actual_outcome_encoded'] = 2
                elif result['home_goals'] < result['away_goals']:
                    pred['actual_outcome'] = 'Away Win'
                    pred['actual_outcome_encoded'] = 0
                else:
                    pred['actual_outcome'] = 'Draw'
                    pred['actual_outcome_encoded'] = 1

                # Check if prediction was correct
                pred['correct'] = (pred['predicted_outcome'] == pred['actual_outcome'])

                logger.info(f"Updated result: {pred['home_team']} {result['home_goals']}-{result['away_goals']} {pred['away_team']}")
                break

        self._save_predictions()

    def get_pending_predictions(self) -> List[Dict]:
        """Get predictions awaiting results."""
        return [p for p in self.predictions['predictions'] if p['status'] == 'pending']

    def get_completed_predictions(self) -> List[Dict]:
        """Get predictions with results."""
        return [p for p in self.predictions['predictions'] if p['status'] == 'completed']

    def clear_old_predictions(self, days: int = 30):
        """Remove predictions older than N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        original_count = len(self.predictions['predictions'])
        self.predictions['predictions'] = [
            p for p in self.predictions['predictions']
            if p['prediction_time'] > cutoff
        ]
        removed = original_count - len(self.predictions['predictions'])
        if removed > 0:
            self._save_predictions()
            logger.info(f"Removed {removed} old predictions")


def fetch_upcoming_fixtures(date: str = None, league_ids: List[int] = None) -> List[Dict]:
    """
    Fetch upcoming fixtures from SportMonks API.

    Args:
        date: Date in YYYY-MM-DD format (default: today)
        league_ids: List of league IDs to filter (default: training leagues)

    Returns:
        List of fixture dictionaries
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if league_ids is None:
        league_ids = list(TRAINING_LEAGUES.keys())

    logger.info(f"Fetching fixtures for {date} from training leagues")

    try:
        url = f"{BASE_URL}/fixtures/between/{date}/{date}"
        params = {
            'api_token': API_KEY,
            'include': 'participants;league;state',
            'filters': f'fixtureLeagues:{",".join(map(str, league_ids))}'
        }

        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or 'data' not in data:
            logger.warning(f"No fixtures found for {date}")
            return []

        fixtures = []
        for fixture in data['data']:
            # Skip if already finished
            state_id = fixture.get('state_id')
            if state_id == 5:  # FT (already finished)
                continue

            # Extract participants
            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            league = fixture.get('league', {})
            league_id = league.get('id')

            # Only include training leagues
            if league_id not in TRAINING_LEAGUES:
                continue

            fixtures.append({
                'fixture_id': fixture['id'],
                'date': fixture.get('starting_at'),
                'league_id': league_id,
                'league_name': TRAINING_LEAGUES.get(league_id, 'Unknown'),
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'state_id': state_id
            })

        logger.info(f"Found {len(fixtures)} upcoming fixtures in training leagues")
        return fixtures

    except Exception as e:
        logger.error(f"Error fetching fixtures: {e}")
        return []


def fetch_fixture_result(fixture_id: int) -> Optional[Dict]:
    """
    Fetch result for a completed fixture.

    Args:
        fixture_id: Sportmonks fixture ID

    Returns:
        Dict with home_goals, away_goals, or None if not finished
    """
    try:
        url = f"{BASE_URL}/fixtures/{fixture_id}"
        params = {
            'api_token': API_KEY,
            'include': 'scores;state'
        }

        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or 'data' not in data:
            return None

        fixture = data['data']

        # Check if finished
        state_id = fixture.get('state_id')
        if state_id != 5:  # Not finished
            return None

        # Extract scores
        scores = fixture.get('scores', [])
        home_goals = None
        away_goals = None

        for score in scores:
            if score.get('description') == 'CURRENT':
                score_data = score.get('score', {})
                participant = score_data.get('participant', '')
                goals = score_data.get('goals', 0)

                if participant == 'home':
                    home_goals = goals
                elif participant == 'away':
                    away_goals = goals

        if home_goals is not None and away_goals is not None:
            return {
                'home_goals': home_goals,
                'away_goals': away_goals,
                'state_id': state_id
            }

        return None

    except Exception as e:
        logger.error(f"Error fetching result for fixture {fixture_id}: {e}")
        return None


def generate_predictions_for_fixtures(fixtures: List[Dict]) -> List[Dict]:
    """
    Generate predictions for fixtures using predict_live.py logic.

    Args:
        fixtures: List of fixture dictionaries

    Returns:
        List of predictions with probabilities
    """
    logger.info(f"Generating predictions for {len(fixtures)} fixtures")

    # Import prediction modules
    import importlib.util

    # Load LiveFeatureCalculator from predict_live
    spec = importlib.util.spec_from_file_location("predict_live", "predict_live.py")
    predict_live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict_live)

    LiveFeatureCalculator = predict_live.LiveFeatureCalculator

    # Load models
    logger.info("Loading models...")
    models = predict_live.load_models(model_name='stacking')

    if not models or 'stacking' not in models:
        logger.error("Failed to load stacking ensemble model")
        return []

    model = models['stacking']

    # Initialize feature calculator
    calculator = LiveFeatureCalculator()

    predictions = []

    for fixture in fixtures:
        try:
            logger.info(f"Predicting: {fixture['home_team_name']} vs {fixture['away_team_name']}")

            # Build features from live API data
            features = calculator.build_features_for_match(
                fixture['home_team_id'],
                fixture['away_team_id'],
                pd.to_datetime(fixture['date']),
                home_team_name=fixture['home_team_name'],
                away_team_name=fixture['away_team_name'],
                league_name=fixture['league_name']
            )

            if not features:
                logger.warning(f"Could not build features for fixture {fixture['fixture_id']}")
                continue

            # Convert to DataFrame
            feature_df = pd.DataFrame([features])

            # Add team info for model
            feature_df['home_team_id'] = fixture['home_team_id']
            feature_df['away_team_id'] = fixture['away_team_id']
            feature_df['home_team_name'] = fixture['home_team_name']
            feature_df['away_team_name'] = fixture['away_team_name']

            # Make prediction
            probs = model.predict_proba(feature_df)[0]

            # Format: [away, draw, home]
            prediction = {
                'fixture_id': fixture['fixture_id'],
                'date': fixture['date'],
                'league_name': fixture['league_name'],
                'home_team': fixture['home_team_name'],
                'away_team': fixture['away_team_name'],
                'home_team_id': fixture['home_team_id'],
                'away_team_id': fixture['away_team_id'],
                'home_prob': float(probs[2]),
                'draw_prob': float(probs[1]),
                'away_prob': float(probs[0]),
                'predicted_outcome': ['Away Win', 'Draw', 'Home Win'][np.argmax(probs)],
                'predicted_outcome_encoded': int(np.argmax(probs)),
                'model_used': 'stacking',
                'data_source': 'live_api'
            }

            predictions.append(prediction)

            logger.info(f"  Home: {prediction['home_prob']:.1%}, Draw: {prediction['draw_prob']:.1%}, Away: {prediction['away_prob']:.1%}")
            logger.info(f"  → {prediction['predicted_outcome']}")

        except Exception as e:
            logger.error(f"Error predicting fixture {fixture['fixture_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"Successfully generated {len(predictions)} predictions")
    return predictions


def predict_today():
    """Fetch today's matches and generate predictions."""
    print("=" * 80)
    print("LIVE PREDICTION SYSTEM - PREDICT TODAY'S MATCHES")
    print("=" * 80)

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\nDate: {today}")
    print(f"Leagues: {', '.join(TRAINING_LEAGUES.values())}")

    # Fetch upcoming fixtures
    print("\nFetching upcoming fixtures...")
    fixtures = fetch_upcoming_fixtures(date=today)

    if not fixtures:
        print("\n⚠️  No upcoming fixtures found for today in training leagues")
        return

    print(f"Found {len(fixtures)} upcoming matches")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = generate_predictions_for_fixtures(fixtures)

    if not predictions:
        print("\n⚠️  Failed to generate predictions")
        return

    # Save predictions to tracker
    tracker = LivePredictionTracker()

    for pred in predictions:
        tracker.add_prediction(pred)

    # Display predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS GENERATED")
    print("=" * 80)

    for pred in predictions:
        print(f"\n{pred['league_name']}")
        print(f"{pred['date']}")
        print(f"{pred['home_team']} vs {pred['away_team']}")
        print(f"  Home: {pred['home_prob']:.1%}")
        print(f"  Draw: {pred['draw_prob']:.1%}")
        print(f"  Away: {pred['away_prob']:.1%}")
        print(f"  → Prediction: {pred['predicted_outcome']}")

    # Save to CSV
    pred_df = pd.DataFrame(predictions)
    csv_file = PREDICTIONS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pred_df.to_csv(csv_file, index=False)

    print(f"\n✅ Predictions saved to: {csv_file}")
    print(f"✅ Tracking {len(predictions)} predictions")


def update_results():
    """Check for results of pending predictions."""
    print("=" * 80)
    print("LIVE PREDICTION SYSTEM - UPDATE RESULTS")
    print("=" * 80)

    tracker = LivePredictionTracker()
    pending = tracker.get_pending_predictions()

    if not pending:
        print("\n⚠️  No pending predictions to update")
        return

    print(f"\nChecking results for {len(pending)} pending predictions...")

    updated = 0
    still_pending = 0

    for pred in pending:
        fixture_id = pred['fixture_id']
        print(f"\nChecking: {pred['home_team']} vs {pred['away_team']}")

        result = fetch_fixture_result(fixture_id)

        if result:
            tracker.update_result(fixture_id, result)
            print(f"  ✅ Result: {result['home_goals']}-{result['away_goals']}")
            updated += 1

            # Small delay to respect rate limits
            time.sleep(0.5)
        else:
            print(f"  ⏳ Still pending")
            still_pending += 1

    print("\n" + "=" * 80)
    print(f"✅ Updated {updated} results")
    print(f"⏳ {still_pending} still pending")
    print("=" * 80)


def generate_report():
    """Generate performance report for completed predictions."""
    print("=" * 80)
    print("LIVE PREDICTION SYSTEM - PERFORMANCE REPORT")
    print("=" * 80)

    tracker = LivePredictionTracker()
    completed = tracker.get_completed_predictions()

    if not completed:
        print("\n⚠️  No completed predictions to report")
        return

    # Convert to DataFrame
    df = pd.DataFrame(completed)

    print(f"\n📊 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Predictions Analyzed: {len(df)}")
    print(f"📊 Date Range: {df['date'].min()} to {df['date'].max()}")

    # Calculate metrics
    y_true = df['actual_outcome_encoded'].values
    y_pred = df['predicted_outcome_encoded'].values
    probs = df[['away_prob', 'draw_prob', 'home_prob']].values

    accuracy = (y_pred == y_true).mean()
    log_loss = calculate_log_loss(y_true, probs)

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Log Loss: {log_loss:.4f}")

    # Accuracy by outcome
    print(f"\n📊 Accuracy by Outcome:")
    for outcome_idx, outcome_name in [(0, 'Away Wins'), (1, 'Draws'), (2, 'Home Wins')]:
        outcome_mask = y_true == outcome_idx
        if outcome_mask.sum() > 0:
            outcome_acc = (y_pred[outcome_mask] == y_true[outcome_mask]).mean()
            print(f"   {outcome_name}: {outcome_acc*100:.1f}% ({outcome_mask.sum()} matches)")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n📊 Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Away   Draw   Home")
    print(f"   Away  {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
    print(f"   Draw  {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
    print(f"   Home  {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")

    # Betting strategy evaluation
    print("\n" + "=" * 80)
    print("BETTING STRATEGY EVALUATION")
    print("=" * 80)

    # Import betting strategy
    import importlib.util
    spec_bet = importlib.util.spec_from_file_location("betting", "11_smart_betting_strategy.py")
    betting_module = importlib.util.module_from_spec(spec_bet)
    spec_bet.loader.exec_module(betting_module)
    SmartMultiOutcomeStrategy = betting_module.SmartMultiOutcomeStrategy

    strategy = SmartMultiOutcomeStrategy(bankroll=1000.0)

    df['actual_outcome_name'] = df['actual_outcome']

    bets = []
    for _, row in df.iterrows():
        match_data = {
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_prob': row['home_prob'],
            'draw_prob': row['draw_prob'],
            'away_prob': row['away_prob']
        }

        recommendations = strategy.evaluate_match(match_data)

        for bet in recommendations:
            actual = row['actual_outcome_name']
            profit = bet.stake * (bet.fair_odds - 1) if bet.bet_outcome == actual else -bet.stake

            bets.append({
                'date': row['date'],
                'home': row['home_team'],
                'away': row['away_team'],
                'bet': bet.bet_outcome,
                'actual': actual,
                'stake': bet.stake,
                'odds': bet.fair_odds,
                'profit': profit,
                'result': 'Win' if bet.bet_outcome == actual else 'Loss'
            })

            if bet.bet_outcome == actual:
                strategy.bankroll += profit
            else:
                strategy.bankroll -= bet.stake

    if len(bets) > 0:
        bets_df = pd.DataFrame(bets)
        total_stake = bets_df['stake'].sum()
        total_profit = bets_df['profit'].sum()
        wins = len(bets_df[bets_df['result'] == 'Win'])

        print(f"\n💰 Betting Results:")
        print(f"   Bets Placed: {len(bets)} ({len(bets)/len(df)*100:.1f}% of matches)")
        print(f"   Winning Bets: {wins} ({wins/len(bets)*100:.1f}%)")
        print(f"   Total Staked: ${total_stake:.2f}")
        print(f"   Net Profit: ${total_profit:+.2f}")
        print(f"   ROI: {total_profit/total_stake*100:+.2f}%")
        print(f"   Final Bankroll: ${strategy.bankroll:.2f} ({(strategy.bankroll-1000)/1000*100:+.2f}%)")

        print(f"\n📋 Performance by Bet Type:")
        for bet_type in ['Home Win', 'Draw', 'Away Win']:
            type_bets = bets_df[bets_df['bet'] == bet_type]
            if len(type_bets) > 0:
                type_wins = len(type_bets[type_bets['result'] == 'Win'])
                type_profit = type_bets['profit'].sum()
                type_roi = type_profit / type_bets['stake'].sum() * 100
                print(f"   {bet_type}: {len(type_bets)} bets, {type_wins/len(type_bets)*100:.1f}% win rate, ${type_profit:+.2f} ({type_roi:+.1f}% ROI)")
    else:
        print("\n⚠️  No bets recommended for these matches")

    # Save report
    report_file = PREDICTIONS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LIVE PREDICTION SYSTEM - PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Predictions Analyzed: {len(df)}\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Log Loss: {log_loss:.4f}\n")
        if len(bets) > 0:
            f.write(f"\nBetting ROI: {total_profit/total_stake*100:+.2f}%\n")
            f.write(f"Net Profit: ${total_profit:+.2f}\n")

    print(f"\n✅ Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Live Testing System")
    parser.add_argument("--predict-today", action="store_true", help="Predict today's matches")
    parser.add_argument("--update-results", action="store_true", help="Update results for pending predictions")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--full", action="store_true", help="Run full workflow (predict + update + report)")
    parser.add_argument("--date", help="Specific date (YYYY-MM-DD) for predictions")

    args = parser.parse_args()

    if args.full:
        predict_today()
        print("\nWaiting 5 seconds before checking results...")
        time.sleep(5)
        update_results()
        print("\n")
        generate_report()
    elif args.predict_today or args.date:
        predict_today()
    elif args.update_results:
        update_results()
    elif args.report:
        generate_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
