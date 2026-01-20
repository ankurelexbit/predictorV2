#!/usr/bin/env python3
"""
Optimal Betting Threshold Finder

Analyzes last 2 years of data to find optimal probability thresholds
for each outcome (Home/Draw/Away) that maximize ROI.

Strategy: Only bet when model probability exceeds threshold for that outcome.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
from sklearn.metrics import log_loss, accuracy_score
import itertools

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger("threshold_optimization")

def calculate_roi(df, probs, thresholds, stake=100):
    """
    Calculate ROI for given probability thresholds.
    
    Args:
        df: DataFrame with actual outcomes and odds
        probs: Model probabilities [away, draw, home]
        thresholds: Dict with 'away', 'draw', 'home' threshold values
        stake: Bet amount per match
    
    Returns:
        Dict with ROI metrics
    """
    total_staked = 0
    total_return = 0
    bets_placed = 0
    bets_won = 0
    
    outcome_names = {0: 'away', 1: 'draw', 2: 'home'}
    
    for idx, (i, row) in enumerate(df.iterrows()):
        # Get model probabilities
        p_away, p_draw, p_home = probs[idx]
        model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
        
        # Get odds
        odds = {
            'away': row.get('odds_away', 0),
            'draw': row.get('odds_draw', 0),
            'home': row.get('odds_home', 0)
        }
        
        # Find best bet (highest probability above threshold)
        best_bet = None
        best_prob = 0
        
        for outcome in ['away', 'draw', 'home']:
            prob = model_probs[outcome]
            threshold = thresholds[outcome]
            
            if prob > threshold and prob > best_prob and odds[outcome] > 1.0:
                best_bet = outcome
                best_prob = prob
        
        # Place bet if we found one
        if best_bet:
            bets_placed += 1
            total_staked += stake
            
            # Check if won
            actual_outcome_idx = int(row['target'])
            actual_outcome = outcome_names[actual_outcome_idx]
            
            if best_bet == actual_outcome:
                bets_won += 1
                total_return += stake * odds[best_bet]
            # else: lost, return is 0
    
    if bets_placed == 0:
        return {
            'roi': 0,
            'profit': 0,
            'bets': 0,
            'win_rate': 0,
            'avg_odds': 0
        }
    
    profit = total_return - total_staked
    roi = profit / total_staked
    win_rate = bets_won / bets_placed
    
    return {
        'roi': roi,
        'profit': profit,
        'bets': bets_placed,
        'wins': bets_won,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'total_return': total_return
    }

def main():
    print("=" * 80)
    print("OPTIMAL BETTING THRESHOLD FINDER")
    print("=" * 80)
    
    # Load features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for last 2 years
    cutoff_date = datetime.now() - timedelta(days=730)  # 2 years
    df_2y = df[df['date'] >= cutoff_date].copy()
    
    print(f"\nAnalyzing {len(df_2y)} matches from last 2 years")
    print(f"Date range: {df_2y['date'].min().date()} to {df_2y['date'].max().date()}")
    
    # Load draw-tuned model
    logger.info("Loading draw-tuned XGBoost model...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.XGBoostFootballModel()
    model.load(MODELS_DIR / "xgboost_model_draw_tuned.joblib")
    
    # Get predictions
    print("\nGenerating predictions...")
    probs = model.predict_proba(df_2y, calibrated=True)
    
    # Check we have odds
    if 'odds_home' not in df_2y.columns or df_2y['odds_home'].isna().all():
        print("ERROR: No odds data available!")
        return
    
    # Filter to matches with odds
    has_odds = df_2y['odds_home'].notna() & df_2y['odds_draw'].notna() & df_2y['odds_away'].notna()
    df_with_odds = df_2y[has_odds].copy()
    probs_with_odds = probs[has_odds]
    
    print(f"Matches with odds: {len(df_with_odds)}")
    
    if len(df_with_odds) == 0:
        print("ERROR: No matches with odds data!")
        return
    
    # Grid search for optimal thresholds
    print("\n" + "=" * 80)
    print("SEARCHING FOR OPTIMAL THRESHOLDS")
    print("=" * 80)
    
    # Test different threshold combinations
    # Start with reasonable ranges
    threshold_range = np.arange(0.35, 0.75, 0.05)  # 35% to 70% in 5% steps
    
    best_roi = -float('inf')
    best_thresholds = None
    best_metrics = None
    
    results = []
    
    print(f"\nTesting {len(threshold_range)**3} threshold combinations...")
    print("(This may take a minute...)\n")
    
    total_combinations = len(threshold_range) ** 3
    tested = 0
    
    for t_home in threshold_range:
        for t_draw in threshold_range:
            for t_away in threshold_range:
                tested += 1
                
                thresholds = {
                    'home': t_home,
                    'draw': t_draw,
                    'away': t_away
                }
                
                metrics = calculate_roi(df_with_odds, probs_with_odds, thresholds)
                
                # Only consider if we placed at least 50 bets
                if metrics['bets'] >= 50:
                    results.append({
                        'threshold_home': t_home,
                        'threshold_draw': t_draw,
                        'threshold_away': t_away,
                        **metrics
                    })
                    
                    if metrics['roi'] > best_roi:
                        best_roi = metrics['roi']
                        best_thresholds = thresholds.copy()
                        best_metrics = metrics.copy()
                        
                        print(f"New best ROI: {best_roi:.1%} | Thresholds: H={t_home:.2f} D={t_draw:.2f} A={t_away:.2f} | Bets: {metrics['bets']}")
                
                # Progress indicator
                if tested % 100 == 0:
                    print(f"Progress: {tested}/{total_combinations} ({tested/total_combinations*100:.1f}%)", end='\r')
    
    print("\n")
    
    # Results
    print("=" * 80)
    print("OPTIMAL THRESHOLDS FOUND")
    print("=" * 80)
    
    if best_thresholds:
        print(f"\n🎯 Best Thresholds:")
        print(f"  Home Win: {best_thresholds['home']:.2f} ({best_thresholds['home']*100:.0f}%)")
        print(f"  Draw:     {best_thresholds['draw']:.2f} ({best_thresholds['draw']*100:.0f}%)")
        print(f"  Away Win: {best_thresholds['away']:.2f} ({best_thresholds['away']*100:.0f}%)")
        
        print(f"\n📊 Performance with Optimal Thresholds:")
        print(f"  ROI:          {best_metrics['roi']:.1%}")
        print(f"  Profit:       ${best_metrics['profit']:,.2f}")
        print(f"  Bets Placed:  {best_metrics['bets']}")
        print(f"  Bets Won:     {best_metrics['wins']}")
        print(f"  Win Rate:     {best_metrics['win_rate']:.1%}")
        print(f"  Total Staked: ${best_metrics['total_staked']:,.2f}")
        print(f"  Total Return: ${best_metrics['total_return']:,.2f}")
        
        # Show top 10 threshold combinations
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('roi', ascending=False)
            
            print("\n" + "=" * 80)
            print("TOP 10 THRESHOLD COMBINATIONS")
            print("=" * 80)
            print(f"\n{'Rank':<5} {'Home':<6} {'Draw':<6} {'Away':<6} {'ROI':<8} {'Bets':<6} {'Win%':<8} {'Profit'}")
            print("-" * 80)
            
            for i, row in results_df.head(10).iterrows():
                print(f"{results_df.index.get_loc(i)+1:<5} "
                      f"{row['threshold_home']:.2f}   "
                      f"{row['threshold_draw']:.2f}   "
                      f"{row['threshold_away']:.2f}   "
                      f"{row['roi']:>6.1%}  "
                      f"{int(row['bets']):<6} "
                      f"{row['win_rate']:>6.1%}  "
                      f"${row['profit']:>8,.2f}")
        
        # Interpretation
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        
        print("\n💡 How to Use These Thresholds:")
        print(f"  1. Get model probabilities for a match")
        print(f"  2. Only bet if probability > threshold:")
        print(f"     - Bet Home if P(Home) > {best_thresholds['home']:.2f}")
        print(f"     - Bet Draw if P(Draw) > {best_thresholds['draw']:.2f}")
        print(f"     - Bet Away if P(Away) > {best_thresholds['away']:.2f}")
        print(f"  3. Bet on the outcome with highest probability above threshold")
        
        print("\n📈 Expected Results:")
        print(f"  - You'll place ~{best_metrics['bets']/len(df_with_odds)*100:.1f}% of matches")
        print(f"  - Win rate: {best_metrics['win_rate']:.1%}")
        print(f"  - ROI: {best_metrics['roi']:.1%}")
        
        # Save results
        if results:
            results_df.to_csv(MODELS_DIR / 'threshold_optimization_results.csv', index=False)
            print(f"\n💾 Full results saved to: {MODELS_DIR / 'threshold_optimization_results.csv'}")
        
        # Save best thresholds
        import json
        threshold_file = MODELS_DIR / 'optimal_thresholds.json'
        with open(threshold_file, 'w') as f:
            json.dump({
                'thresholds': best_thresholds,
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in best_metrics.items()},
                'date_analyzed': datetime.now().isoformat(),
                'matches_analyzed': len(df_with_odds)
            }, f, indent=2)
        
        print(f"💾 Optimal thresholds saved to: {threshold_file}")
        
    else:
        print("⚠️  Could not find optimal thresholds (not enough bets placed)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
