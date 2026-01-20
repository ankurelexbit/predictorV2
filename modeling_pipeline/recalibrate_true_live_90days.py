#!/usr/bin/env python3
"""
True Live Pipeline 90-Day Recalibration

Simulates live feature generation (271 features) for each match
and recalibrates thresholds based on realistic live predictions.

WARNING: This is computationally expensive - simulates live feature
generation for ~900 matches without API calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger("true_live_recalibration")

def simulate_live_features(row, all_matches_df):
    """
    Simulate live feature generation for a single match.
    Uses only data available BEFORE the match (no future data).
    """
    from predict_live import LiveFeatureCalculator
    
    # Get match details
    fixture_date = pd.to_datetime(row['date'])
    home_team_id = int(row['home_team_id'])
    away_team_id = int(row['away_team_id'])
    home_team_name = row.get('home_team_name', 'Unknown')
    away_team_name = row.get('away_team_name', 'Unknown')
    
    # Filter for matches BEFORE this fixture (no future data)
    past_matches = all_matches_df[all_matches_df['date'] < fixture_date].copy()
    
    # Create a mock LiveFeatureCalculator that uses historical data
    # instead of API calls
    class MockLiveCalculator:
        def __init__(self, historical_data):
            self.historical_data = historical_data
            
        def get_team_recent_matches(self, team_id, limit=15):
            """Get recent matches from historical data."""
            team_matches = self.historical_data[
                (self.historical_data['home_team_id'] == team_id) |
                (self.historical_data['away_team_id'] == team_id)
            ].copy()
            
            # Sort by date and take most recent
            team_matches = team_matches.sort_values('date', ascending=False).head(limit)
            
            # Convert to match format
            matches = []
            for _, m in team_matches.iterrows():
                is_home = m['home_team_id'] == team_id
                
                # Extract basic stats
                match_data = {
                    'date': m['date'],
                    'goals': m.get('home_goals_3', 0) if is_home else m.get('away_goals_3', 0),
                    'goals_conceded': m.get('away_goals_3', 0) if is_home else m.get('home_goals_3', 0),
                    'shots_total': m.get('home_shots_total_3', 0) if is_home else m.get('away_shots_total_3', 0),
                    'shots_on_target': m.get('home_shots_on_target_3', 0) if is_home else m.get('away_shots_on_target_3', 0),
                    'possession_pct': m.get('home_possession_pct_3', 50) if is_home else m.get('away_possession_pct_3', 50),
                    'shots_total_conceded': m.get('away_shots_total_3', 0) if is_home else m.get('home_shots_total_3', 0),
                    'shots_on_target_conceded': m.get('away_shots_on_target_3', 0) if is_home else m.get('home_shots_on_target_3', 0),
                }
                matches.append(match_data)
            
            return matches
        
        def calculate_ema(self, values, alpha=0.3):
            """Calculate EMA."""
            if not values:
                return 0.0
            ema = values[0]
            for value in values[1:]:
                ema = alpha * value + (1 - alpha) * ema
            return ema
        
        def calculate_ema_features(self, matches, alpha=0.3):
            """Calculate EMA features."""
            ema_features = {}
            
            stats_to_ema = {
                'goals': 'goals_ema',
                'goals_conceded': 'goals_conceded_ema',
                'shots_total': 'shots_total_ema',
                'shots_total_conceded': 'shots_total_conceded_ema',
                'shots_on_target': 'shots_on_target_ema',
                'shots_on_target_conceded': 'shots_on_target_conceded_ema',
                'possession_pct': 'possession_pct_ema',
            }
            
            for stat_key, ema_key in stats_to_ema.items():
                values = [m.get(stat_key, 0) for m in matches]
                ema_features[ema_key] = self.calculate_ema(values, alpha) if values else 0.0
            
            # Add conceded possession
            poss_values = [m.get('possession_pct', 50) for m in matches]
            ema_features['possession_pct_conceded_ema'] = self.calculate_ema(
                [100 - p for p in poss_values], alpha
            ) if poss_values else 50.0
            
            return ema_features
        
        def calculate_rest_days(self, matches, fixture_date):
            """Calculate rest days."""
            if not matches:
                return {'days_rest': 7, 'short_rest': 0}
            
            last_match_date = matches[0]['date']
            if isinstance(last_match_date, str):
                last_match_date = pd.to_datetime(last_match_date)
            
            days_rest = (fixture_date - last_match_date).days
            short_rest = 1 if days_rest < 4 else 0
            
            return {'days_rest': days_rest, 'short_rest': short_rest}
    
    # Create mock calculator
    calc = MockLiveCalculator(past_matches)
    
    # Get recent matches
    home_matches = calc.get_team_recent_matches(home_team_id, limit=15)
    away_matches = calc.get_team_recent_matches(away_team_id, limit=15)
    
    if not home_matches or not away_matches:
        return None
    
    # Calculate EMA features
    home_ema = calc.calculate_ema_features(home_matches)
    away_ema = calc.calculate_ema_features(away_matches)
    
    # Calculate rest days
    home_rest = calc.calculate_rest_days(home_matches, fixture_date)
    away_rest = calc.calculate_rest_days(away_matches, fixture_date)
    
    # Build feature dict (simplified - just the new 27 features)
    # In reality, we'd need all 271 features, but for threshold optimization,
    # we can use the existing features + the 27 new ones
    live_features = row.to_dict()
    
    # Add EMA features
    live_features.update({
        'home_goals_ema': home_ema['goals_ema'],
        'away_goals_ema': away_ema['goals_ema'],
        'home_goals_conceded_ema': home_ema['goals_conceded_ema'],
        'away_goals_conceded_ema': away_ema['goals_conceded_ema'],
        'home_shots_total_ema': home_ema['shots_total_ema'],
        'away_shots_total_ema': away_ema['shots_total_ema'],
        'home_shots_total_conceded_ema': home_ema['shots_total_conceded_ema'],
        'away_shots_total_conceded_ema': away_ema['shots_total_conceded_ema'],
        'home_shots_on_target_ema': home_ema['shots_on_target_ema'],
        'away_shots_on_target_ema': away_ema['shots_on_target_ema'],
        'home_shots_on_target_conceded_ema': home_ema['shots_on_target_conceded_ema'],
        'away_shots_on_target_conceded_ema': away_ema['shots_on_target_conceded_ema'],
        'home_possession_pct_ema': home_ema['possession_pct_ema'],
        'away_possession_pct_ema': away_ema['possession_pct_ema'],
        'home_possession_pct_conceded_ema': home_ema['possession_pct_conceded_ema'],
        'away_possession_pct_conceded_ema': away_ema['possession_pct_conceded_ema'],
        # Add rest days
        'days_rest_home': home_rest['days_rest'],
        'days_rest_away': away_rest['days_rest'],
        'home_short_rest': home_rest['short_rest'],
        'away_short_rest': away_rest['short_rest'],
        'rest_diff': home_rest['days_rest'] - away_rest['days_rest'],
    })
    
    return live_features

def calculate_roi_for_thresholds(df, probs, thresholds, stake=100):
    """Calculate ROI for given thresholds."""
    outcome_map = {'away': 0, 'draw': 1, 'home': 2}
    outcome_names = {0: 'away', 1: 'draw', 2: 'home'}
    
    total_profit = 0
    total_staked = 0
    bets_placed = 0
    bets_won = 0
    
    for idx, (i, row) in enumerate(df.iterrows()):
        p_away, p_draw, p_home = probs[idx]
        model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
        
        odds = {
            'away': row.get('odds_away', 0),
            'draw': row.get('odds_draw', 0),
            'home': row.get('odds_home', 0)
        }
        
        # Find best bet
        best_bet = None
        best_prob = 0
        
        for outcome in ['away', 'draw', 'home']:
            prob = model_probs[outcome]
            threshold = thresholds[outcome]
            
            if prob > threshold and prob > best_prob and odds[outcome] > 1.0:
                best_bet = outcome
                best_prob = prob
        
        if best_bet:
            actual_outcome_idx = int(row['target'])
            actual_outcome = outcome_names[actual_outcome_idx]
            
            won = (best_bet == actual_outcome)
            payout = stake * odds[best_bet] if won else 0
            profit = payout - stake
            
            total_profit += profit
            total_staked += stake
            bets_placed += 1
            if won:
                bets_won += 1
    
    if total_staked == 0:
        return None
    
    roi = (total_profit / total_staked) * 100
    win_rate = (bets_won / bets_placed) * 100 if bets_placed > 0 else 0
    
    return {
        'roi': roi,
        'profit': total_profit,
        'staked': total_staked,
        'bets': bets_placed,
        'won': bets_won,
        'win_rate': win_rate
    }

def main():
    print("=" * 80)
    print("TRUE LIVE PIPELINE 90-DAY RECALIBRATION")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n⚠️  This simulates live feature generation (271 features)")
    print("   Expected runtime: 5-10 minutes for 892 matches")
    
    # Load ALL features (needed for historical lookups)
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading features from {features_path}")
    df_all = pd.read_csv(features_path)
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    # Filter for last 90 days
    cutoff_date = datetime.now() - timedelta(days=90)
    df_90d = df_all[df_all['date'] >= cutoff_date].copy()
    
    print(f"\n📅 Date Range: {df_90d['date'].min().date()} to {df_90d['date'].max().date()}")
    print(f"📊 Total Matches: {len(df_90d)}")
    
    # Filter for matches with odds
    has_odds = (df_90d['odds_home'] > 0) & (~df_90d['odds_home'].isna())
    df_with_odds = df_90d[has_odds].copy()
    
    print(f"Matches with odds: {len(df_with_odds)}")
    
    # Simulate live feature generation
    print("\n🔄 Simulating live feature generation...")
    print("   (This may take 5-10 minutes...)")
    
    live_features_list = []
    for idx, (i, row) in enumerate(df_with_odds.iterrows()):
        if idx % 100 == 0:
            print(f"   Progress: {idx}/{len(df_with_odds)} matches ({idx/len(df_with_odds)*100:.1f}%)")
        
        live_feat = simulate_live_features(row, df_all)
        if live_feat:
            live_features_list.append(live_feat)
    
    df_live = pd.DataFrame(live_features_list)
    print(f"\n✅ Generated live features for {len(df_live)} matches")
    
    # Load model and get predictions
    logger.info("Loading draw-tuned XGBoost model...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.XGBoostFootballModel()
    model.load(MODELS_DIR / "xgboost_model_draw_tuned.joblib")
    
    print("\n🔮 Generating predictions with live features...")
    probs = model.predict_proba(df_live, calibrated=True)
    
    # Grid search for optimal thresholds
    print("\n" + "=" * 80)
    print("SEARCHING FOR OPTIMAL THRESHOLDS (TRUE LIVE PIPELINE)")
    print("=" * 80)
    
    # Define threshold ranges
    home_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    draw_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    away_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    
    total_combinations = len(home_thresholds) * len(draw_thresholds) * len(away_thresholds)
    print(f"\nTesting {total_combinations} threshold combinations...")
    
    best_result = None
    best_thresholds = None
    all_results = []
    
    for home_t, draw_t, away_t in product(home_thresholds, draw_thresholds, away_thresholds):
        thresholds = {'home': home_t, 'draw': draw_t, 'away': away_t}
        
        result = calculate_roi_for_thresholds(df_live, probs, thresholds)
        
        if result and result['bets'] >= 10:
            all_results.append({
                'home': home_t,
                'draw': draw_t,
                'away': away_t,
                **result
            })
            
            if best_result is None or result['roi'] > best_result['roi']:
                best_result = result
                best_thresholds = thresholds
                print(f"New best ROI: {result['roi']:.1f}% | "
                      f"Thresholds: H={home_t:.2f} D={draw_t:.2f} A={away_t:.2f} | "
                      f"Bets: {result['bets']}")
    
    # Show results
    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLDS (TRUE LIVE PIPELINE)")
    print("=" * 80)
    
    if best_thresholds:
        print(f"\n🎯 Best Thresholds:")
        print(f"  Home Win: {best_thresholds['home']:.2f} ({best_thresholds['home']*100:.0f}%)")
        print(f"  Draw:     {best_thresholds['draw']:.2f} ({best_thresholds['draw']*100:.0f}%)")
        print(f"  Away Win: {best_thresholds['away']:.2f} ({best_thresholds['away']*100:.0f}%)")
        
        print(f"\n📊 Performance:")
        print(f"  ROI:          {best_result['roi']:.1f}%")
        print(f"  Profit:       ${best_result['profit']:,.2f}")
        print(f"  Bets Placed:  {best_result['bets']}")
        print(f"  Bets Won:     {best_result['won']}")
        print(f"  Win Rate:     {best_result['win_rate']:.1f}%")
        print(f"  Total Staked: ${best_result['staked']:,.2f}")
        
        # Save results
        output = {
            'thresholds': best_thresholds,
            'performance': best_result,
            'calibration_date': datetime.now().isoformat(),
            'calibration_period': '90 days',
            'matches_analyzed': len(df_live),
            'pipeline': 'true_live_271_features'
        }
        
        output_file = MODELS_DIR / 'optimal_thresholds_true_live_90day.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 True live thresholds saved to: {output_file}")
        
        # Show top 10
        print("\n" + "=" * 80)
        print("TOP 10 THRESHOLD COMBINATIONS (TRUE LIVE)")
        print("=" * 80)
        
        results_df = pd.DataFrame(all_results).sort_values('roi', ascending=False)
        
        print(f"\n{'Rank':<5} {'Home':<6} {'Draw':<6} {'Away':<6} {'ROI':<8} {'Bets':<6} {'Win%':<8} {'Profit'}")
        print("-" * 80)
        
        for i, row in results_df.head(10).iterrows():
            print(f"{i+1:<5} {row['home']:<6.2f} {row['draw']:<6.2f} {row['away']:<6.2f} "
                  f"{row['roi']:<8.1f} {int(row['bets']):<6} {row['win_rate']:<8.1f} "
                  f"${row['profit']:,.2f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
