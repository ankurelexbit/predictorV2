#!/usr/bin/env python3
"""
Weekly Threshold Recalibration

After model retraining, recalibrates thresholds on the latest data
to ensure optimal betting performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger("threshold_recalibration")

def main():
    logger.info("=" * 80)
    logger.info("WEEKLY THRESHOLD RECALIBRATION")
    logger.info("=" * 80)
    
    # Load latest features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Use last 90 days for calibration
    cutoff = datetime.now() - timedelta(days=90)
    df_90d = df[df['date'] >= cutoff]
    
    # Filter for matches with odds
    has_odds = (df_90d['odds_home'] > 0) & (~df_90d['odds_home'].isna())
    df_cal = df_90d[has_odds].copy()
    
    logger.info(f"Calibration data: {len(df_cal)} matches from last 90 days")
    
    # Load newly trained model
    import importlib.util
    spec = importlib.util.spec_from_file_location("xgboost_model", 
                                                   Path(__file__).parent.parent / "06_model_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.XGBoostFootballModel()
    model.load(MODELS_DIR / "xgboost_model_draw_tuned.joblib")
    
    # Generate predictions
    logger.info("Generating predictions...")
    probs = model.predict_proba(df_cal, calibrated=True)
    
    # Grid search for optimal thresholds
    logger.info("Searching for optimal thresholds...")
    
    home_range = [0.45, 0.50, 0.55, 0.60]
    draw_range = [0.35, 0.40, 0.45]
    away_range = [0.55, 0.60, 0.65]
    
    best_roi = -float('inf')
    best_thresholds = None
    
    from itertools import product
    
    for h, d, a in product(home_range, draw_range, away_range):
        thresholds = {'home': h, 'draw': d, 'away': a}
        
        # Calculate performance
        outcome_map = {0: 'away', 1: 'draw', 2: 'home'}
        bets = []
        
        for idx, (i, row) in enumerate(df_cal.iterrows()):
            p_away, p_draw, p_home = probs[idx]
            model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
            
            best_bet = None
            best_prob = 0
            
            for outcome in ['away', 'draw', 'home']:
                if model_probs[outcome] > thresholds[outcome] and model_probs[outcome] > best_prob:
                    best_bet = outcome
                    best_prob = model_probs[outcome]
            
            if best_bet:
                actual = outcome_map[int(row['target'])]
                won = (best_bet == actual)
                
                odds = {
                    'home': row['odds_home'],
                    'draw': row['odds_draw'],
                    'away': row['odds_away']
                }
                
                profit = (100 * odds[best_bet] - 100) if won else -100
                bets.append(profit)
        
        if len(bets) >= 20:  # Minimum 20 bets
            roi = (sum(bets) / (len(bets) * 100)) * 100
            
            if roi > best_roi:
                best_roi = roi
                best_thresholds = thresholds
                best_bets = len(bets)
                best_wins = sum(1 for p in bets if p > 0)
    
    if best_thresholds:
        logger.info(f"\n✅ Optimal thresholds found:")
        logger.info(f"   Home: {best_thresholds['home']:.2f}")
        logger.info(f"   Draw: {best_thresholds['draw']:.2f}")
        logger.info(f"   Away: {best_thresholds['away']:.2f}")
        logger.info(f"   ROI: {best_roi:.1f}%")
        logger.info(f"   Bets: {best_bets}")
        logger.info(f"   Win Rate: {best_wins/best_bets*100:.1f}%")
        
        # Save thresholds
        output = {
            'thresholds': best_thresholds,
            'calibration_date': datetime.now().isoformat(),
            'calibration_method': 'weekly_recalibration',
            'expected_performance': {
                'roi': float(best_roi),
                'bets': int(best_bets),
                'win_rate': float(best_wins/best_bets*100)
            }
        }
        
        output_file = MODELS_DIR / 'optimal_thresholds_production.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\n💾 Thresholds saved to {output_file}")
        
        # Update production_thresholds.py
        py_file = Path(__file__).parent.parent / 'production_thresholds.py'
        
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Update threshold values
        new_content = content
        new_content = new_content.replace(
            f"'home': 0.50",
            f"'home': {best_thresholds['home']:.2f}"
        )
        new_content = new_content.replace(
            f"'draw': 0.40",
            f"'draw': {best_thresholds['draw']:.2f}"
        )
        new_content = new_content.replace(
            f"'away': 0.60",
            f"'away': {best_thresholds['away']:.2f}"
        )
        
        with open(py_file, 'w') as f:
            f.write(new_content)
        
        logger.info(f"✅ Updated {py_file}")
        
    else:
        logger.error("❌ Could not find optimal thresholds!")
        sys.exit(1)

if __name__ == "__main__":
    main()
