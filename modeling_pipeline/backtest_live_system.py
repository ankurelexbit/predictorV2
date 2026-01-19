"""
Backtest Live Prediction System on Last 10 Days
================================================

Run the live prediction system on matches from the last 10 days
to validate performance with actual results.

This simulates what would have happened if we ran predictions daily
over the last 10 days using real-time API data.

Usage:
    python backtest_live_system.py --days 10
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
import json
import time

sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logger, calculate_log_loss
import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = setup_logger("backtest_live")

# API config
API_KEY = "LmEYRdsf8CSmiblNVTn6JfV0y0s8tc4aQsEkJ6JuoBhOWa3Hd1FEanGcrijo"
BASE_URL = "https://api.sportmonks.com/v3/football"

# Training leagues
TRAINING_LEAGUES = {
    8: "Premier League",
    9: "Championship",
    564: "La Liga",
    82: "Bundesliga",
    384: "Serie A",
    301: "Ligue 1"
}


def fetch_finished_fixtures(start_date: str, end_date: str) -> List[Dict]:
    """Fetch finished fixtures from date range."""
    logger.info(f"Fetching finished fixtures from {start_date} to {end_date}")

    try:
        url = f"{BASE_URL}/fixtures/between/{start_date}/{end_date}"
        params = {
            'api_token': API_KEY,
            'include': 'participants;league;scores;state',
            'filters': f'fixtureLeagues:{",".join(map(str, TRAINING_LEAGUES.keys()))}'
        }

        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or 'data' not in data:
            logger.warning(f"No fixtures found")
            return []

        fixtures = []
        for fixture in data['data']:
            # Only include finished matches
            state_id = fixture.get('state_id')
            if state_id != 5:  # Not finished
                continue

            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            league = fixture.get('league', {})
            league_id = league.get('id')

            if league_id not in TRAINING_LEAGUES:
                continue

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

            if home_goals is None or away_goals is None:
                continue

            # Determine result
            if home_goals > away_goals:
                result = 'Home Win'
                result_encoded = 2
            elif home_goals < away_goals:
                result = 'Away Win'
                result_encoded = 0
            else:
                result = 'Draw'
                result_encoded = 1

            fixtures.append({
                'fixture_id': fixture['id'],
                'date': fixture.get('starting_at'),
                'league_id': league_id,
                'league_name': TRAINING_LEAGUES.get(league_id, 'Unknown'),
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'home_goals': home_goals,
                'away_goals': away_goals,
                'result': result,
                'result_encoded': result_encoded
            })

        logger.info(f"Found {len(fixtures)} finished fixtures")
        return fixtures

    except Exception as e:
        logger.error(f"Error fetching fixtures: {e}")
        return []


def generate_predictions_for_fixtures(fixtures: List[Dict]) -> List[Dict]:
    """Generate predictions using live feature calculation."""
    logger.info(f"Generating predictions for {len(fixtures)} fixtures")

    import importlib.util

    # Load predict_live module
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
    calculator = LiveFeatureCalculator()

    predictions = []
    total = len(fixtures)

    for idx, fixture in enumerate(fixtures, 1):
        try:
            logger.info(f"[{idx}/{total}] Predicting: {fixture['home_team_name']} vs {fixture['away_team_name']}")

            # Build features from live API data (as if predicting before the match)
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
            feature_df['home_team_id'] = fixture['home_team_id']
            feature_df['away_team_id'] = fixture['away_team_id']
            feature_df['home_team_name'] = fixture['home_team_name']
            feature_df['away_team_name'] = fixture['away_team_name']

            # Make prediction
            probs = model.predict_proba(feature_df)[0]

            prediction = {
                'fixture_id': fixture['fixture_id'],
                'date': fixture['date'],
                'league_name': fixture['league_name'],
                'home_team': fixture['home_team_name'],
                'away_team': fixture['away_team_name'],
                'home_prob': float(probs[2]),
                'draw_prob': float(probs[1]),
                'away_prob': float(probs[0]),
                'predicted_outcome': ['Away Win', 'Draw', 'Home Win'][np.argmax(probs)],
                'predicted_encoded': int(np.argmax(probs)),
                'actual_home_goals': fixture['home_goals'],
                'actual_away_goals': fixture['away_goals'],
                'actual_outcome': fixture['result'],
                'actual_encoded': fixture['result_encoded'],
                'correct': (['Away Win', 'Draw', 'Home Win'][np.argmax(probs)] == fixture['result'])
            }

            predictions.append(prediction)

            logger.info(f"  Predicted: {prediction['predicted_outcome']} | Actual: {prediction['actual_outcome']} | {'✅' if prediction['correct'] else '❌'}")

            # Small delay to respect rate limits
            if idx % 10 == 0:
                time.sleep(2)

        except Exception as e:
            logger.error(f"Error predicting fixture {fixture['fixture_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"Successfully generated {len(predictions)} predictions")
    return predictions


def evaluate_betting_strategy(predictions: List[Dict], initial_bankroll: float = 1000.0) -> Dict:
    """Evaluate betting strategy on predictions."""
    logger.info("Evaluating betting strategy...")

    # Import betting strategy
    import importlib.util
    spec_bet = importlib.util.spec_from_file_location("betting", "11_smart_betting_strategy.py")
    betting_module = importlib.util.module_from_spec(spec_bet)
    spec_bet.loader.exec_module(betting_module)
    SmartMultiOutcomeStrategy = betting_module.SmartMultiOutcomeStrategy

    strategy = SmartMultiOutcomeStrategy(bankroll=initial_bankroll)

    bets = []

    for pred in predictions:
        match_data = {
            'home_team': pred['home_team'],
            'away_team': pred['away_team'],
            'home_prob': pred['home_prob'],
            'draw_prob': pred['draw_prob'],
            'away_prob': pred['away_prob']
        }

        recommendations = strategy.evaluate_match(match_data)

        for bet in recommendations:
            actual = pred['actual_outcome']
            profit = bet.stake * (bet.fair_odds - 1) if bet.bet_outcome == actual else -bet.stake

            bets.append({
                'date': pred['date'],
                'home': pred['home_team'],
                'away': pred['away_team'],
                'bet_outcome': bet.bet_outcome,
                'actual_outcome': actual,
                'stake': bet.stake,
                'odds': bet.fair_odds,
                'profit': profit,
                'result': 'Win' if bet.bet_outcome == actual else 'Loss',
                'rule': bet.rule_applied,
                'ev': bet.expected_value
            })

            # Update bankroll
            if bet.bet_outcome == actual:
                strategy.bankroll += profit
            else:
                strategy.bankroll -= bet.stake

    if not bets:
        return {
            'total_bets': 0,
            'message': 'No bets placed'
        }

    bets_df = pd.DataFrame(bets)

    total_stake = bets_df['stake'].sum()
    total_profit = bets_df['profit'].sum()
    wins = len(bets_df[bets_df['result'] == 'Win'])

    return {
        'total_bets': len(bets),
        'winning_bets': wins,
        'win_rate': wins / len(bets) * 100,
        'total_staked': total_stake,
        'net_profit': total_profit,
        'roi': (total_profit / total_stake * 100) if total_stake > 0 else 0,
        'final_bankroll': strategy.bankroll,
        'bankroll_change': (strategy.bankroll - initial_bankroll) / initial_bankroll * 100,
        'bets_df': bets_df
    }


def generate_report(predictions: List[Dict], betting_results: Dict, output_file: Path):
    """Generate comprehensive report."""
    df = pd.DataFrame(predictions)

    # Calculate metrics
    y_true = df['actual_encoded'].values
    y_pred = df['predicted_encoded'].values
    probs = df[['away_prob', 'draw_prob', 'home_prob']].values

    accuracy = (y_pred == y_true).mean()
    log_loss = calculate_log_loss(y_true, probs)

    # Outcome distribution
    actual_dist = df['actual_outcome'].value_counts()
    pred_dist = df['predicted_outcome'].value_counts()

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LIVE PREDICTION SYSTEM - 10 DAY BACKTEST REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Matches Analyzed: {len(df)}")
    report_lines.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    report_lines.append("")

    # Model Performance
    report_lines.append("=" * 80)
    report_lines.append("MODEL PERFORMANCE")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Overall Metrics:")
    report_lines.append(f"  Accuracy: {accuracy*100:.2f}%")
    report_lines.append(f"  Log Loss: {log_loss:.4f}")
    report_lines.append("")

    # Prediction vs Actual Distribution
    report_lines.append("Outcome Distribution:")
    report_lines.append(f"  Actual:")
    for outcome, count in actual_dist.items():
        pct = count / len(df) * 100
        report_lines.append(f"    {outcome}: {count} ({pct:.1f}%)")
    report_lines.append(f"  Predicted:")
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        count = pred_dist.get(outcome, 0)
        pct = count / len(df) * 100
        report_lines.append(f"    {outcome}: {count} ({pct:.1f}%)")
    report_lines.append("")

    # Accuracy by outcome
    report_lines.append("Accuracy by Outcome:")
    for outcome_idx, outcome_name in [(0, 'Away Wins'), (1, 'Draws'), (2, 'Home Wins')]:
        outcome_mask = y_true == outcome_idx
        if outcome_mask.sum() > 0:
            outcome_acc = (y_pred[outcome_mask] == y_true[outcome_mask]).mean()
            report_lines.append(f"  {outcome_name}: {outcome_acc*100:.1f}% ({outcome_mask.sum()} matches)")
    report_lines.append("")

    # Confusion Matrix
    report_lines.append("Confusion Matrix:")
    report_lines.append("              Predicted")
    report_lines.append("              Away   Draw   Home")
    report_lines.append(f"   Away  {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
    report_lines.append(f"   Draw  {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
    report_lines.append(f"   Home  {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")
    report_lines.append("")

    # Betting Performance
    report_lines.append("=" * 80)
    report_lines.append("BETTING STRATEGY PERFORMANCE")
    report_lines.append("=" * 80)
    report_lines.append("")

    if betting_results['total_bets'] > 0:
        report_lines.append(f"Overall Betting Results:")
        report_lines.append(f"  Total Bets: {betting_results['total_bets']} ({betting_results['total_bets']/len(df)*100:.1f}% of matches)")
        report_lines.append(f"  Winning Bets: {betting_results['winning_bets']}")
        report_lines.append(f"  Win Rate: {betting_results['win_rate']:.1f}%")
        report_lines.append(f"  Total Staked: ${betting_results['total_staked']:.2f}")
        report_lines.append(f"  Net Profit: ${betting_results['net_profit']:+.2f}")
        report_lines.append(f"  ROI: {betting_results['roi']:+.2f}%")
        report_lines.append(f"  Initial Bankroll: $1,000.00")
        report_lines.append(f"  Final Bankroll: ${betting_results['final_bankroll']:.2f}")
        report_lines.append(f"  Bankroll Change: {betting_results['bankroll_change']:+.2f}%")
        report_lines.append("")

        # By bet type
        bets_df = betting_results['bets_df']
        report_lines.append("Performance by Bet Type:")
        for bet_type in ['Home Win', 'Draw', 'Away Win']:
            type_bets = bets_df[bets_df['bet_outcome'] == bet_type]
            if len(type_bets) > 0:
                type_wins = len(type_bets[type_bets['result'] == 'Win'])
                type_profit = type_bets['profit'].sum()
                type_roi = type_profit / type_bets['stake'].sum() * 100
                report_lines.append(f"  {bet_type}:")
                report_lines.append(f"    Bets: {len(type_bets)}")
                report_lines.append(f"    Win Rate: {type_wins/len(type_bets)*100:.1f}%")
                report_lines.append(f"    Profit: ${type_profit:+.2f}")
                report_lines.append(f"    ROI: {type_roi:+.1f}%")
        report_lines.append("")

        # Top profitable bets
        report_lines.append("Top 5 Profitable Bets:")
        top_bets = bets_df.nlargest(5, 'profit')
        for _, bet in top_bets.iterrows():
            report_lines.append(f"  {bet['home']} vs {bet['away']}")
            report_lines.append(f"    Bet: {bet['bet_outcome']} @ {bet['odds']:.2f}")
            report_lines.append(f"    Result: {bet['actual_outcome']} → {bet['result']}")
            report_lines.append(f"    Profit: ${bet['profit']:+.2f}")
        report_lines.append("")

        # Worst bets
        report_lines.append("Worst 5 Bets:")
        worst_bets = bets_df.nsmallest(5, 'profit')
        for _, bet in worst_bets.iterrows():
            report_lines.append(f"  {bet['home']} vs {bet['away']}")
            report_lines.append(f"    Bet: {bet['bet_outcome']} @ {bet['odds']:.2f}")
            report_lines.append(f"    Result: {bet['actual_outcome']} → {bet['result']}")
            report_lines.append(f"    Loss: ${bet['profit']:+.2f}")
    else:
        report_lines.append("No bets placed")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Save report
    report_text = "\n".join(report_lines)

    with open(output_file, 'w') as f:
        f.write(report_text)

    # Print to console
    print(report_text)

    return report_text


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Live System on Last N Days")
    parser.add_argument("--days", type=int, default=10, help="Number of days to backtest")

    args = parser.parse_args()

    print("=" * 80)
    print("LIVE PREDICTION SYSTEM - BACKTEST ON LAST 10 DAYS")
    print("=" * 80)
    print()

    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=args.days)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"Date Range: {start_str} to {end_str}")
    print(f"Leagues: {', '.join(TRAINING_LEAGUES.values())}")
    print()

    # Fetch finished fixtures
    print("Step 1: Fetching finished fixtures from API...")
    fixtures = fetch_finished_fixtures(start_str, end_str)

    if not fixtures:
        print("❌ No fixtures found")
        return

    print(f"✅ Found {len(fixtures)} finished fixtures")
    print()

    # Generate predictions
    print("Step 2: Generating predictions using live feature calculation...")
    print("(This will take several minutes due to API rate limiting)")
    print()

    predictions = generate_predictions_for_fixtures(fixtures)

    if not predictions:
        print("❌ Failed to generate predictions")
        return

    print()
    print(f"✅ Generated {len(predictions)} predictions")
    print()

    # Evaluate betting strategy
    print("Step 3: Evaluating betting strategy...")
    betting_results = evaluate_betting_strategy(predictions)
    print(f"✅ Analyzed {betting_results.get('total_bets', 0)} bets")
    print()

    # Generate report
    print("Step 4: Generating comprehensive report...")
    output_file = Path(f"LIVE_10_DAY_BACKTEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    generate_report(predictions, betting_results, output_file)

    print()
    print("=" * 80)
    print(f"✅ Report saved to: {output_file}")

    # Save detailed data
    pred_df = pd.DataFrame(predictions)
    pred_csv = Path(f"10_day_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"✅ Predictions saved to: {pred_csv}")

    if betting_results.get('total_bets', 0) > 0:
        bets_csv = Path(f"10_day_bets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        betting_results['bets_df'].to_csv(bets_csv, index=False)
        print(f"✅ Bets saved to: {bets_csv}")

    print("=" * 80)


if __name__ == "__main__":
    main()
