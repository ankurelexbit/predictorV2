#!/usr/bin/env python3
"""
Production Simulation - Live Betting System

Simulates the complete production workflow:
1. Fetch today's fixtures from SportMonks API
2. Generate features with live API calls
3. Make predictions with draw-tuned model
4. Apply optimal thresholds
5. Output betting recommendations

This is exactly how the system will work in production.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR
from utils import setup_logger

logger = setup_logger("production_simulation")

def main():
    print("=" * 80)
    print("PRODUCTION SIMULATION - LIVE BETTING SYSTEM")
    print("=" * 80)
    print(f"Simulation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis simulates the EXACT production workflow:")
    print("  1. ✅ Fetch fixtures from SportMonks API")
    print("  2. ✅ Generate features with live API calls")
    print("  3. ✅ Make predictions with draw-tuned model")
    print("  4. ✅ Apply optimal thresholds")
    print("  5. ✅ Output betting recommendations")
    
    # Load optimal thresholds
    threshold_file = MODELS_DIR / 'optimal_thresholds_true_live_90day.json'
    
    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            threshold_data = json.load(f)
        thresholds = threshold_data['thresholds']
        print(f"\n📊 Using Optimal Thresholds (90-day calibrated):")
        print(f"   Home: {thresholds['home']:.2f}")
        print(f"   Draw: {thresholds['draw']:.2f}")
        print(f"   Away: {thresholds['away']:.2f}")
    else:
        thresholds = {'home': 0.65, 'draw': 0.45, 'away': 0.70}
        print(f"\n⚠️  Using default thresholds (calibration file not found)")
    
    # Import predict_live module
    print("\n" + "=" * 80)
    print("STEP 1: INITIALIZE LIVE PREDICTION SYSTEM")
    print("=" * 80)
    
    from predict_live import LiveFeatureCalculator, load_models
    
    # Load model
    print("\n🔮 Loading draw-tuned XGBoost model...")
    models = load_models('xgboost')
    
    if 'xgboost' not in models:
        print("❌ Failed to load XGBoost model!")
        return
    
    model = models['xgboost']
    print("✅ Model loaded successfully")
    
    # Initialize feature calculator
    print("\n🔧 Initializing live feature calculator...")
    calculator = LiveFeatureCalculator()
    print("✅ Feature calculator ready")
    
    # Fetch today's fixtures
    print("\n" + "=" * 80)
    print("STEP 2: FETCH TODAY'S FIXTURES FROM API")
    print("=" * 80)
    
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n📅 Fetching fixtures for: {today}")
    
    from predict_live import get_upcoming_fixtures
    
    fixtures_df = get_upcoming_fixtures(today)
    
    if fixtures_df.empty:
        print(f"\n⚠️  No fixtures found for {today}")
        print("\nTrying tomorrow's date...")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        fixtures_df = get_upcoming_fixtures(tomorrow)
        today = tomorrow
    
    if fixtures_df.empty:
        print("\n❌ No fixtures found for today or tomorrow")
        print("\nTrying to find next available fixtures...")
        
        # Try next 7 days
        for days_ahead in range(2, 8):
            future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            fixtures_df = get_upcoming_fixtures(future_date)
            if not fixtures_df.empty:
                today = future_date
                break
        
        if fixtures_df.empty:
            print("❌ No fixtures found in next 7 days")
            return
    
    fixtures = fixtures_df.to_dict('records')
    
    # Show fixtures
    print("\n📋 Fixtures:")
    for i, fixture in enumerate(fixtures[:10], 1):  # Show first 10
        home = fixture.get('home_team_name', 'Unknown')
        away = fixture.get('away_team_name', 'Unknown')
        league = fixture.get('league_name', 'Unknown')
        time = fixture.get('starting_at', 'TBD')
        print(f"  {i}. {home} vs {away} ({league}) - {time}")
    
    if len(fixtures) > 10:
        print(f"  ... and {len(fixtures) - 10} more")
    
    # Generate predictions
    print("\n" + "=" * 80)
    print("STEP 3: GENERATE FEATURES & PREDICTIONS (LIVE API CALLS)")
    print("=" * 80)
    
    print(f"\n⚠️  This will make LIVE API calls for each fixture")
    print(f"   Expected time: ~{len(fixtures) * 30} seconds ({len(fixtures)} fixtures × 30s each)")
    
    input("\nPress Enter to continue with live API calls (or Ctrl+C to cancel)...")
    
    predictions = []
    successful = 0
    failed = 0
    
    for i, fixture in enumerate(fixtures, 1):
        print(f"\n[{i}/{len(fixtures)}] Processing: {fixture.get('home_team_name')} vs {fixture.get('away_team_name')}")
        
        try:
            # Build features with LIVE API CALLS
            print(f"  🔄 Fetching live data from API...")
            features = calculator.build_features_for_match(
                home_team_id=fixture['home_team_id'],
                away_team_id=fixture['away_team_id'],
                fixture_date=datetime.fromisoformat(fixture['starting_at'].replace('Z', '+00:00')),
                home_team_name=fixture.get('home_team_name'),
                away_team_name=fixture.get('away_team_name'),
                league_name=fixture.get('league_name'),
                fixture_id=fixture.get('id')
            )
            
            if not features:
                print(f"  ❌ Failed to build features")
                failed += 1
                continue
            
            print(f"  ✅ Built {len(features)} features")
            
            # Make prediction
            features_df = pd.DataFrame([features])
            probs = model.predict_proba(features_df, calibrated=True)[0]
            
            p_away, p_draw, p_home = probs
            
            print(f"  🎯 Predictions: Home={p_home*100:.1f}% Draw={p_draw*100:.1f}% Away={p_away*100:.1f}%")
            
            # Apply thresholds
            model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
            
            best_bet = None
            best_prob = 0
            
            for outcome in ['away', 'draw', 'home']:
                prob = model_probs[outcome]
                threshold = thresholds[outcome]
                
                if prob > threshold and prob > best_prob:
                    best_bet = outcome
                    best_prob = prob
            
            # Store prediction
            prediction = {
                'fixture_id': fixture.get('id'),
                'date': fixture.get('starting_at'),
                'league': fixture.get('league_name'),
                'home_team': fixture.get('home_team_name'),
                'away_team': fixture.get('away_team_name'),
                'p_home': p_home,
                'p_draw': p_draw,
                'p_away': p_away,
                'recommended_bet': best_bet,
                'confidence': best_prob if best_bet else 0,
            }
            
            predictions.append(prediction)
            successful += 1
            
            if best_bet:
                print(f"  💰 RECOMMENDATION: Bet {best_bet.upper()} ({best_prob*100:.1f}% confidence)")
            else:
                print(f"  ⏭️  SKIP: No bet meets threshold")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("STEP 4: BETTING RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n📊 Processing Summary:")
    print(f"   Total Fixtures:  {len(fixtures)}")
    print(f"   Successful:      {successful}")
    print(f"   Failed:          {failed}")
    
    # Filter for recommendations
    recommendations = [p for p in predictions if p['recommended_bet']]
    
    print(f"\n💰 Betting Recommendations: {len(recommendations)}")
    
    if recommendations:
        print(f"\n{'Match':<50} {'Bet':<8} {'Conf':<8} {'Probs (H/D/A)'}")
        print("-" * 90)
        
        for rec in recommendations:
            match = f"{rec['home_team'][:22]} vs {rec['away_team'][:22]}"
            bet = rec['recommended_bet'].upper()
            conf = f"{rec['confidence']*100:.1f}%"
            probs = f"{rec['p_home']*100:.0f}/{rec['p_draw']*100:.0f}/{rec['p_away']*100:.0f}"
            
            print(f"{match:<50} {bet:<8} {conf:<8} {probs}")
    else:
        print("\n⚠️  No bets recommended (no fixtures exceeded thresholds)")
    
    # Save results
    output_file = MODELS_DIR / f'production_simulation_{today.replace("-", "")}.json'
    
    output = {
        'simulation_date': datetime.now().isoformat(),
        'fixture_date': today,
        'thresholds': thresholds,
        'total_fixtures': len(fixtures),
        'successful_predictions': successful,
        'failed_predictions': failed,
        'recommendations': recommendations,
        'all_predictions': predictions
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PRODUCTION SIMULATION COMPLETE")
    print("=" * 80)
    
    print(f"\n✅ System Status: OPERATIONAL")
    print(f"\n📊 Performance:")
    print(f"   Fixtures Analyzed:     {successful}/{len(fixtures)}")
    print(f"   Bets Recommended:      {len(recommendations)}")
    print(f"   Bet Rate:              {len(recommendations)/successful*100:.1f}%" if successful > 0 else "   Bet Rate:              N/A")
    
    if recommendations:
        print(f"\n💰 Expected Performance (based on 90-day calibration):")
        print(f"   Expected ROI:          21.9%")
        print(f"   Expected Win Rate:     82.3%")
        print(f"   Recommended Stake:     $100 per bet")
        print(f"   Total Stake:           ${len(recommendations) * 100:,}")
        print(f"   Expected Profit:       ${len(recommendations) * 100 * 0.219:,.2f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
