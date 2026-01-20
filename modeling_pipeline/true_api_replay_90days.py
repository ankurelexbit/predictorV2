#!/usr/bin/env python3
"""
True Production Replay with Real API Calls

This script replicates the EXACT production workflow:
1. For each historical match, make REAL API calls to fetch data
2. Generate features on-the-fly (271 features)
3. Make predictions with draw-tuned model
4. Apply optimal thresholds
5. Compare to actual results
6. Calculate comprehensive PnL

WARNING: This will take 6-8 hours and consume significant API quota.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger
from predict_live import LiveFeatureCalculator, load_models

logger = setup_logger("true_api_replay")

def main():
    print("=" * 80)
    print("TRUE PRODUCTION REPLAY WITH REAL API CALLS")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n⚠️  WARNING: This will make REAL API calls!")
    print("   - Sample size: 200 matches (randomly sampled from 90 days)")
    print("   - Expected runtime: ~2 hours")
    print("   - API calls: ~200 matches × 30-40 API calls each = ~7,000 total calls")
    print("   - This will consume API quota")
    
    response = input("\nAre you sure you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Load thresholds
    threshold_file = MODELS_DIR / 'optimal_thresholds_true_live_90day.json'
    
    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            data = json.load(f)
        thresholds = data['thresholds']
    else:
        thresholds = {'home': 0.65, 'draw': 0.45, 'away': 0.70}
    
    print(f"\n🎯 Using Optimal Thresholds:")
    print(f"   Home: {thresholds['home']:.2f}")
    print(f"   Draw: {thresholds['draw']:.2f}")
    print(f"   Away: {thresholds['away']:.2f}")
    
    # Load historical data (for match list and actual results)
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading match list from {features_path}")
    df_all = pd.read_csv(features_path)
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    # Filter for last 90 days
    cutoff_date = datetime.now() - timedelta(days=90)
    df_90d = df_all[df_all['date'] >= cutoff_date].copy()
    
    # Filter for matches with odds
    has_odds = (df_90d['odds_home'] > 0) & (~df_90d['odds_home'].isna())
    df_with_odds = df_90d[has_odds].copy()
    
    # Sample 200 random matches to reduce runtime
    sample_size = min(200, len(df_with_odds))
    df_matches = df_with_odds.sample(n=sample_size, random_state=42).sort_values('date')
    
    print(f"\n📊 Sampled {sample_size} matches from {len(df_with_odds)} total")
    print(f"   This reduces runtime from ~{len(df_with_odds)*30/3600:.1f}h to ~{sample_size*30/3600:.1f}h")
    
    print(f"\n📅 Replay Period: {df_matches['date'].min().date()} to {df_matches['date'].max().date()}")
    print(f"📊 Total Matches: {len(df_matches)}")
    
    # Initialize live feature calculator
    print("\n🔧 Initializing live feature calculator...")
    calculator = LiveFeatureCalculator()
    
    # Load model
    print("🔮 Loading draw-tuned XGBoost model...")
    models = load_models('xgboost')
    
    if 'xgboost' not in models:
        print("❌ Failed to load model")
        return
    
    model = models['xgboost']
    
    # Process each match with REAL API calls
    print("\n" + "=" * 80)
    print("PROCESSING MATCHES WITH REAL API CALLS")
    print("=" * 80)
    
    print(f"\n⏱️  Estimated time: {len(df_matches) * 30 / 3600:.1f} hours")
    print(f"📞 Estimated API calls: ~{len(df_matches) * 35:,}")
    
    results = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for idx, (i, row) in enumerate(df_matches.iterrows(), 1):
        elapsed = time.time() - start_time
        avg_time_per_match = elapsed / idx if idx > 0 else 30
        remaining_matches = len(df_matches) - idx
        eta_seconds = remaining_matches * avg_time_per_match
        eta_hours = eta_seconds / 3600
        
        print(f"\n[{idx}/{len(df_matches)}] {row['home_team_name']} vs {row['away_team_name']}")
        print(f"   Date: {row['date'].date()}")
        print(f"   Elapsed: {elapsed/60:.1f}m | ETA: {eta_hours:.1f}h | Avg: {avg_time_per_match:.1f}s/match")
        
        try:
            # Make REAL API calls to build features
            print(f"   🔄 Making API calls...")
            
            match_start = time.time()
            
            features = calculator.build_features_for_match(
                home_team_id=int(row['home_team_id']),
                away_team_id=int(row['away_team_id']),
                fixture_date=row['date'],
                home_team_name=row.get('home_team_name'),
                away_team_name=row.get('away_team_name'),
                league_name=row.get('league_name'),
                fixture_id=row.get('fixture_id')
            )
            
            match_time = time.time() - match_start
            
            if not features:
                print(f"   ❌ Failed to build features")
                failed += 1
                continue
            
            print(f"   ✅ Built {len(features)} features in {match_time:.1f}s")
            
            # Make prediction
            features_df = pd.DataFrame([features])
            probs = model.predict_proba(features_df, calibrated=True)[0]
            
            p_away, p_draw, p_home = probs
            
            print(f"   🎯 Predictions: H={p_home*100:.1f}% D={p_draw*100:.1f}% A={p_away*100:.1f}%")
            
            # Apply thresholds
            model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
            
            odds = {
                'away': row.get('odds_away', 0),
                'draw': row.get('odds_draw', 0),
                'home': row.get('odds_home', 0)
            }
            
            best_bet = None
            best_prob = 0
            
            for outcome in ['away', 'draw', 'home']:
                prob = model_probs[outcome]
                threshold = thresholds[outcome]
                
                if prob > threshold and prob > best_prob and odds[outcome] > 1.0:
                    best_bet = outcome
                    best_prob = prob
            
            # Get actual outcome
            outcome_names = {0: 'away', 1: 'draw', 2: 'home'}
            actual_outcome_idx = int(row['target'])
            actual_outcome = outcome_names[actual_outcome_idx]
            
            if best_bet:
                won = (best_bet == actual_outcome)
                stake = 100
                payout = stake * odds[best_bet] if won else 0
                profit = payout - stake
                
                result_str = "✅ Won" if won else "❌ Lost"
                print(f"   💰 BET {best_bet.upper()} @ {odds[best_bet]:.2f} → {result_str} (${profit:+.2f})")
                
                results.append({
                    'date': row['date'],
                    'home_team': row['home_team_name'],
                    'away_team': row['away_team_name'],
                    'bet_on': best_bet,
                    'probability': best_prob,
                    'odds': odds[best_bet],
                    'actual_outcome': actual_outcome,
                    'won': won,
                    'stake': stake,
                    'payout': payout,
                    'profit': profit,
                    'p_home': p_home,
                    'p_draw': p_draw,
                    'p_away': p_away,
                    'api_time': match_time
                })
            else:
                print(f"   ⏭️  No bet (no threshold exceeded)")
            
            successful += 1
            
            # Save progress every 10 matches
            if idx % 10 == 0:
                temp_df = pd.DataFrame(results)
                temp_file = MODELS_DIR / 'true_api_replay_progress.csv'
                temp_df.to_csv(temp_file, index=False)
                print(f"   💾 Progress saved ({len(results)} bets so far)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
            continue
    
    # Final results
    print("\n" + "=" * 80)
    print("REPLAY COMPLETE")
    print("=" * 80)
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total Runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"📊 Matches Processed: {successful}/{len(df_matches)}")
    print(f"❌ Failed: {failed}")
    print(f"💰 Bets Placed: {len(results)}")
    
    if len(results) == 0:
        print("\n⚠️  No bets placed")
        return
    
    # Calculate performance
    results_df = pd.DataFrame(results)
    
    total_bets = len(results_df)
    total_won = results_df['won'].sum()
    total_staked = results_df['stake'].sum()
    total_profit = results_df['profit'].sum()
    roi = (total_profit / total_staked) * 100
    win_rate = (total_won / total_bets) * 100
    
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Overall:")
    print(f"   Bets:       {total_bets}")
    print(f"   Won:        {total_won} ({win_rate:.1f}%)")
    print(f"   Profit:     ${total_profit:+,.2f}")
    print(f"   ROI:        {roi:+.1f}%")
    
    # Save final results
    output_file = MODELS_DIR / 'true_api_replay_90days_final.csv'
    results_df.to_csv(output_file, index=False)
    
    summary = {
        'replay_date': datetime.now().isoformat(),
        'runtime_hours': total_time / 3600,
        'matches_processed': successful,
        'matches_failed': failed,
        'total_bets': total_bets,
        'bets_won': int(total_won),
        'win_rate': float(win_rate),
        'total_profit': float(total_profit),
        'roi': float(roi),
        'thresholds': thresholds
    }
    
    summary_file = MODELS_DIR / 'true_api_replay_90days_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"   {output_file}")
    print(f"   {summary_file}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
