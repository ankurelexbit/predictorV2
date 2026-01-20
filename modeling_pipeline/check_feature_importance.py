#!/usr/bin/env python3
"""
Check XGBoost Feature Importance - Alternative Method

Uses get_score() from XGBoost booster directly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR

def check_feature_importance():
    """Check XGBoost feature importance using booster."""
    
    print("=" * 80)
    print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Load model
    print("\n1. Loading XGBoost model...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.XGBoostFootballModel()
    model.load(MODELS_DIR / "xgboost_model_draw_tuned.joblib")
    
    # Try to get feature importance from booster
    print("\n2. Extracting feature importance from booster...")
    
    try:
        # Get importance scores
        importance_dict = model.model.get_booster().get_score(importance_type='gain')
        
        # Convert to dataframe
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        print(f"✅ Extracted importance for {len(importance_df)} features")
        
        # Show top 30 features
        print("\n" + "=" * 80)
        print("TOP 30 MOST IMPORTANT FEATURES")
        print("=" * 80)
        
        print(f"\n{'Rank':<5} {'Feature':<50} {'Importance':<12} {'Type'}")
        print("-" * 80)
        
        for idx, row in importance_df.head(30).iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Categorize feature
            if 'ema' in feature.lower():
                feature_type = "EMA ❌"
            elif 'rest' in feature.lower() or 'days' in feature.lower():
                feature_type = "REST ❌"
            elif 'player' in feature.lower():
                feature_type = "PLAYER ⚠️"
            elif 'elo' in feature.lower():
                feature_type = "ELO ✅"
            elif any(x in feature.lower() for x in ['goals', 'xg', 'shots']):
                feature_type = "STATS ✅"
            elif 'form' in feature.lower():
                feature_type = "FORM ✅"
            else:
                feature_type = "OTHER ✅"
            
            rank = list(importance_df.index).index(idx) + 1
            print(f"{rank:<5} {feature:<50} {importance:<12.1f} {feature_type}")
        
        # Count by category
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE BY CATEGORY (Top 30)")
        print("=" * 80)
        
        top_30 = importance_df.head(30)
        
        categories = {
            'EMA': sum(1 for f in top_30['feature'] if 'ema' in f.lower()),
            'Rest Days': sum(1 for f in top_30['feature'] if 'rest' in f.lower() or 'days' in f.lower()),
            'Player Stats': sum(1 for f in top_30['feature'] if 'player' in f.lower()),
            'Elo': sum(1 for f in top_30['feature'] if 'elo' in f.lower()),
            'Goals/xG/Shots': sum(1 for f in top_30['feature'] if any(x in f.lower() for x in ['goals', 'xg', 'shots'])),
            'Form': sum(1 for f in top_30['feature'] if 'form' in f.lower()),
        }
        
        print(f"\n{'Category':<20} {'Count in Top 30':<15} {'Status'}")
        print("-" * 60)
        for cat, count in categories.items():
            if cat in ['EMA', 'Rest Days']:
                status = "❌ MISSING IN LIVE"
            elif cat == 'Player Stats':
                status = "⚠️  APPROXIMATED"
            else:
                status = "✅ AVAILABLE"
            print(f"{cat:<20} {count:<15} {status}")
        
        # Critical missing features
        print("\n" + "=" * 80)
        print("CRITICAL MISSING FEATURES IN TOP 30")
        print("=" * 80)
        
        missing_in_top30 = []
        for idx, row in top_30.iterrows():
            feature = row['feature']
            if 'ema' in feature.lower() or 'rest' in feature.lower():
                missing_in_top30.append((feature, row['importance']))
        
        if missing_in_top30:
            print(f"\n⚠️  Found {len(missing_in_top30)} critical features missing in live pipeline:")
            for feat, imp in missing_in_top30:
                print(f"   - {feat}: {imp:.1f}")
        else:
            print("\n✅ No critical features missing in top 30")
        
        # Overall statistics
        print("\n" + "=" * 80)
        print("OVERALL FEATURE IMPORTANCE STATISTICS")
        print("=" * 80)
        
        total_ema = sum(1 for f in importance_df['feature'] if 'ema' in f.lower())
        total_rest = sum(1 for f in importance_df['feature'] if 'rest' in f.lower())
        
        ema_importance = importance_df[importance_df['feature'].str.contains('ema', case=False)]['importance'].sum()
        rest_importance = importance_df[importance_df['feature'].str.contains('rest', case=False)]['importance'].sum()
        total_importance = importance_df['importance'].sum()
        
        print(f"\nEMA Features:")
        print(f"  Count: {total_ema}")
        print(f"  Total Importance: {ema_importance:.1f} ({ema_importance/total_importance*100:.1f}% of total)")
        
        print(f"\nRest Days Features:")
        print(f"  Count: {total_rest}")
        print(f"  Total Importance: {rest_importance:.1f} ({rest_importance/total_importance*100:.1f}% of total)")
        
        print(f"\nCombined Missing Features:")
        print(f"  Total Importance: {(ema_importance + rest_importance)/total_importance*100:.1f}% of model")
        
        # Save to file
        importance_df.to_csv(MODELS_DIR / 'feature_importance.csv', index=False)
        print(f"\n💾 Full feature importance saved to: {MODELS_DIR / 'feature_importance.csv'}")
        
        return importance_df, missing_in_top30
        
    except Exception as e:
        print(f"❌ Error extracting feature importance: {e}")
        return None, []

if __name__ == "__main__":
    importance_df, missing = check_feature_importance()
    
    if missing:
        print("\n" + "=" * 80)
        print("⚠️  ACTION REQUIRED")
        print("=" * 80)
        print(f"\n{len(missing)} critical features are missing from live pipeline!")
        print("These features are in the TOP 30 most important.")
        print("\nYou MUST add EMA and rest days calculations to predict_live.py")
