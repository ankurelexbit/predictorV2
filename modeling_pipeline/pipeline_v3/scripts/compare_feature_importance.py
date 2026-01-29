#!/usr/bin/env python3
"""
Compare XGBoost Feature Importance: Parent vs V3
Shows which features each model actually uses/values
"""

import joblib
import pandas as pd
import sys
from pathlib import Path
import importlib.util

# Load parent model class
spec = importlib.util.spec_from_file_location(
    "xgboost_model", 
    Path('/Users/ankurgupta/code/predictorV2/modeling_pipeline/06_model_xgboost.py')
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules['xgboost_model'] = mod

def load_parent_importance():
    """Load parent model feature importance"""
    model_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/models/xgboost_model_draw_tuned.joblib'
    data = joblib.load(model_path)
    
    # Get feature importance
    importance = data.get('feature_importance', {})
    
    if not importance:
        print("No feature importance found in parent model")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importance.items()
    ])
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return df

def load_v3_importance():
    """Load V3 model feature importance"""
    model_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3/models/xgboost_model.joblib'
    data = joblib.load(model_path)
    
    importance = data.get('feature_importance', {})
    
    if not importance:
        print("No feature importance found in V3 model")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importance.items()
    ])
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return df

def main():
    print("=" * 80)
    print("XGBOOST FEATURE IMPORTANCE COMPARISON")
    print("=" * 80)
    
    # Load importance
    parent_imp = load_parent_importance()
    v3_imp = load_v3_importance()
    
    if parent_imp.empty or v3_imp.empty:
        print("Could not load feature importance from one or both models")
        return
    
    print(f"\nParent Model: {len(parent_imp)} features with importance")
    print(f"V3 Model:     {len(v3_imp)} features with importance")
    
    # Top features comparison
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES BY IMPORTANCE")
    print("=" * 80)
    
    print(f"\n{'PARENT PIPELINE':<50} {'PIPELINE V3':<50}")
    print(f"{'Feature':<35} {'Importance':<15} {'Feature':<35} {'Importance':<15}")
    print("-" * 100)
    
    for i in range(min(20, len(parent_imp), len(v3_imp))):
        p_feat = parent_imp.iloc[i]['feature']
        p_imp = parent_imp.iloc[i]['importance']
        v_feat = v3_imp.iloc[i]['feature']
        v_imp = v3_imp.iloc[i]['importance']
        
        print(f"{p_feat:<35} {p_imp:<15.2f} {v_feat:<35} {v_imp:<15.2f}")
    
    # Analyze top features
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Get top N features
    top_n = 20
    parent_top = set(parent_imp.head(top_n)['feature'])
    v3_top = set(v3_imp.head(top_n)['feature'])
    
    common_top = parent_top.intersection(v3_top)
    
    print(f"\nTop {top_n} Features:")
    print(f"  Common in both: {len(common_top)}")
    print(f"  Parent-only:    {len(parent_top - v3_top)}")
    print(f"  V3-only:        {len(v3_top - parent_top)}")
    
    if common_top:
        print(f"\n  ✅ Common High-Value Features:")
        for feat in sorted(common_top):
            p_rank = parent_imp[parent_imp['feature'] == feat].index[0] + 1
            v_rank = v3_imp[v3_imp['feature'] == feat].index[0] + 1
            print(f"    - {feat:<40} (Parent: #{p_rank}, V3: #{v_rank})")
    
    parent_only_top = parent_top - v3_top
    if parent_only_top:
        print(f"\n  ⚠️  High-Value in Parent, Missing/Low in V3:")
        for feat in sorted(parent_only_top):
            p_rank = parent_imp[parent_imp['feature'] == feat].index[0] + 1
            p_imp = parent_imp[parent_imp['feature'] == feat]['importance'].values[0]
            
            # Check if exists in V3
            if feat in v3_imp['feature'].values:
                v_rank = v3_imp[v3_imp['feature'] == feat].index[0] + 1
                print(f"    - {feat:<40} (Parent: #{p_rank}, V3: #{v_rank} ⬇️)")
            else:
                print(f"    - {feat:<40} (Parent: #{p_rank}, V3: MISSING ❌)")
    
    v3_only_top = v3_top - parent_top
    if v3_only_top:
        print(f"\n  ✨ High-Value in V3, Missing/Low in Parent:")
        for feat in sorted(list(v3_only_top)[:10]):
            v_rank = v3_imp[v3_imp['feature'] == feat].index[0] + 1
            v_imp = v3_imp[v3_imp['feature'] == feat]['importance'].values[0]
            print(f"    - {feat:<40} (V3: #{v_rank}, Importance: {v_imp:.1f})")
    
    # Importance distribution
    print("\n" + "=" * 80)
    print("IMPORTANCE DISTRIBUTION")
    print("=" * 80)
    
    parent_total = parent_imp['importance'].sum()
    v3_total = v3_imp['importance'].sum()
    
    parent_top10_pct = parent_imp.head(10)['importance'].sum() / parent_total * 100
    v3_top10_pct = v3_imp.head(10)['importance'].sum() / v3_total * 100
    
    parent_top20_pct = parent_imp.head(20)['importance'].sum() / parent_total * 100
    v3_top20_pct = v3_imp.head(20)['importance'].sum() / v3_total * 100
    
    print(f"\nImportance Concentration:")
    print(f"  Top 10 features:")
    print(f"    Parent: {parent_top10_pct:.1f}% of total importance")
    print(f"    V3:     {v3_top10_pct:.1f}% of total importance")
    print(f"\n  Top 20 features:")
    print(f"    Parent: {parent_top20_pct:.1f}% of total importance")
    print(f"    V3:     {v3_top20_pct:.1f}% of total importance")
    
    # Key insights
    print("\n" + "=" * 80)
    print("🔍 KEY INSIGHTS")
    print("=" * 80)
    
    print(f"\n1. Feature Concentration:")
    if parent_top10_pct > v3_top10_pct:
        print(f"   - Parent model is MORE concentrated on top features ({parent_top10_pct:.1f}% vs {v3_top10_pct:.1f}%)")
        print(f"   - Suggests parent has clearer signal from fewer features")
    else:
        print(f"   - V3 model is MORE concentrated on top features ({v3_top10_pct:.1f}% vs {parent_top10_pct:.1f}%)")
    
    print(f"\n2. Top Feature Overlap:")
    overlap_pct = len(common_top) / top_n * 100
    print(f"   - {overlap_pct:.0f}% of top {top_n} features are common")
    if overlap_pct < 50:
        print(f"   - Models rely on DIFFERENT features for predictions")
    else:
        print(f"   - Models have SIMILAR feature preferences")
    
    # Save detailed report
    report_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3/feature_importance_comparison.csv'
    
    # Merge importance
    merged = parent_imp.merge(
        v3_imp, 
        on='feature', 
        how='outer', 
        suffixes=('_parent', '_v3')
    ).fillna(0)
    
    merged = merged.sort_values('importance_parent', ascending=False)
    merged.to_csv(report_path, index=False)
    
    print(f"\n📄 Detailed comparison saved to: {report_path}")

if __name__ == "__main__":
    main()
