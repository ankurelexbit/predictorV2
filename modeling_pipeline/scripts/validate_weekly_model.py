#!/usr/bin/env python3
"""
Validate Weekly Model

Validates the newly trained model before deployment.
Returns:
- 0: Validation passed (deploy)
- 1: Validation failed (don't deploy)
- 2: Warning (deploy with caution)
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

logger = setup_logger("model_validation")

def main():
    logger.info("=" * 80)
    logger.info("WEEKLY MODEL VALIDATION")
    logger.info("=" * 80)
    
    # Load test data (last 30 days)
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    cutoff = datetime.now() - timedelta(days=30)
    df_test = df[df['date'] >= cutoff]
    
    has_odds = (df_test['odds_home'] > 0) & (~df_test['odds_home'].isna())
    df_test = df_test[has_odds].copy()
    
    logger.info(f"Test data: {len(df_test)} matches from last 30 days")
    
    # Load model
    import importlib.util
    spec = importlib.util.spec_from_file_location("xgboost_model", 
                                                   Path(__file__).parent.parent / "06_model_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.XGBoostFootballModel()
    model.load(MODELS_DIR / "xgboost_model_draw_tuned.joblib")
    
    # Load thresholds
    threshold_file = MODELS_DIR / 'optimal_thresholds_production.json'
    with open(threshold_file) as f:
        config = json.load(f)
    thresholds = config['thresholds']
    
    # Generate predictions
    probs = model.predict_proba(df_test, calibrated=True)
    
    # Calculate performance
    outcome_map = {0: 'away', 1: 'draw', 2: 'home'}
    bets = []
    
    for idx, (i, row) in enumerate(df_test.iterrows()):
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
            bets.append({'won': won, 'profit': profit})
    
    if len(bets) == 0:
        logger.error("❌ No bets placed - validation failed!")
        sys.exit(1)
    
    # Calculate metrics
    total_bets = len(bets)
    total_won = sum(1 for b in bets if b['won'])
    total_profit = sum(b['profit'] for b in bets)
    roi = (total_profit / (total_bets * 100)) * 100
    win_rate = (total_won / total_bets) * 100
    
    logger.info(f"\n📊 Validation Results:")
    logger.info(f"   Bets: {total_bets}")
    logger.info(f"   Won: {total_won} ({win_rate:.1f}%)")
    logger.info(f"   ROI: {roi:.1f}%")
    logger.info(f"   Profit: ${total_profit:+,.2f}")
    
    # Validation criteria
    MIN_ROI = 10.0
    MIN_WIN_RATE = 55.0
    TARGET_ROI = 20.0
    TARGET_WIN_RATE = 65.0
    
    logger.info(f"\n🎯 Validation Criteria:")
    logger.info(f"   Minimum ROI: {MIN_ROI}%")
    logger.info(f"   Minimum Win Rate: {MIN_WIN_RATE}%")
    logger.info(f"   Target ROI: {TARGET_ROI}%")
    logger.info(f"   Target Win Rate: {TARGET_WIN_RATE}%")
    
    # Determine result
    if roi >= TARGET_ROI and win_rate >= TARGET_WIN_RATE:
        logger.info(f"\n✅ VALIDATION PASSED - Excellent performance!")
        sys.exit(0)
    elif roi >= MIN_ROI and win_rate >= MIN_WIN_RATE:
        logger.info(f"\n⚠️  VALIDATION WARNING - Acceptable but below target")
        sys.exit(2)
    else:
        logger.info(f"\n❌ VALIDATION FAILED - Performance below minimum")
        sys.exit(1)

if __name__ == "__main__":
    main()
