#!/usr/bin/env python3
"""
Detailed Feature Analysis: Parent Pipeline vs Pipeline V3
"""

import pandas as pd
import joblib
import sys
from pathlib import Path
from collections import defaultdict

def categorize_features(features):
    """Categorize features by type"""
    categories = defaultdict(list)
    
    for feat in features:
        if 'elo' in feat.lower():
            categories['Elo Ratings'].append(feat)
        elif 'form' in feat.lower():
            categories['Form Metrics'].append(feat)
        elif 'player' in feat.lower():
            categories['Player Statistics'].append(feat)
        elif 'xg' in feat.lower():
            categories['Expected Goals (xG)'].append(feat)
        elif 'attack' in feat.lower() or 'defense' in feat.lower():
            categories['Attack/Defense Strength'].append(feat)
        elif 'h2h' in feat.lower():
            categories['Head-to-Head'].append(feat)
        elif 'position' in feat.lower() or 'points' in feat.lower():
            categories['League Standing'].append(feat)
        elif 'injury' in feat.lower() or 'injuries' in feat.lower():
            categories['Injuries'].append(feat)
        elif any(x in feat.lower() for x in ['shot', 'pass', 'tackle', 'intercept', 'possession']):
            categories['Match Statistics'].append(feat)
        elif any(x in feat.lower() for x in ['goal', 'win', 'draw', 'loss']):
            categories['Results/Goals'].append(feat)
        elif any(x in feat.lower() for x in ['season', 'round', 'weekend', 'days']):
            categories['Temporal Features'].append(feat)
        else:
            categories['Other'].append(feat)
    
    return categories

def load_parent_features():
    """Load parent model features"""
    model_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/models/xgboost_model_draw_tuned.joblib'
    data = joblib.load(model_path)
    return sorted(data['feature_columns'])

def load_v3_features():
    """Load V3 model features"""
    model_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3/models/xgboost_model.joblib'
    data = joblib.load(model_path)
    return sorted(data['feature_columns'])

def main():
    print("=" * 80)
    print("DETAILED FEATURE ANALYSIS: PARENT vs PIPELINE V3")
    print("=" * 80)
    
    # Load features
    parent_features = load_parent_features()
    v3_features = load_v3_features()
    
    print(f"\n📊 SUMMARY")
    print(f"  Parent Pipeline: {len(parent_features)} features")
    print(f"  Pipeline V3:     {len(v3_features)} features")
    
    # Categorize
    parent_cats = categorize_features(parent_features)
    v3_cats = categorize_features(v3_features)
    
    # Compare categories
    print(f"\n" + "=" * 80)
    print("FEATURE BREAKDOWN BY CATEGORY")
    print("=" * 80)
    
    all_categories = sorted(set(list(parent_cats.keys()) + list(v3_cats.keys())))
    
    print(f"\n{'Category':<30} {'Parent':<10} {'V3':<10} {'Difference'}")
    print("-" * 80)
    for cat in all_categories:
        parent_count = len(parent_cats.get(cat, []))
        v3_count = len(v3_cats.get(cat, []))
        diff = v3_count - parent_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{cat:<30} {parent_count:<10} {v3_count:<10} {diff_str}")
    
    # Detailed category analysis
    print(f"\n" + "=" * 80)
    print("DETAILED CATEGORY COMPARISON")
    print("=" * 80)
    
    for cat in all_categories:
        parent_feats = set(parent_cats.get(cat, []))
        v3_feats = set(v3_cats.get(cat, []))
        
        if not parent_feats and not v3_feats:
            continue
            
        print(f"\n{'─' * 80}")
        print(f"📁 {cat}")
        print(f"{'─' * 80}")
        
        # Only in parent
        only_parent = parent_feats - v3_feats
        if only_parent:
            print(f"\n  ⚠️  ONLY IN PARENT ({len(only_parent)}):")
            for f in sorted(only_parent)[:10]:
                print(f"    - {f}")
            if len(only_parent) > 10:
                print(f"    ... and {len(only_parent) - 10} more")
        
        # Only in V3
        only_v3 = v3_feats - parent_feats
        if only_v3:
            print(f"\n  ✨ ONLY IN V3 ({len(only_v3)}):")
            for f in sorted(only_v3)[:10]:
                print(f"    - {f}")
            if len(only_v3) > 10:
                print(f"    ... and {len(only_v3) - 10} more")
        
        # Common
        common = parent_feats.intersection(v3_feats)
        if common:
            print(f"\n  ✅ COMMON ({len(common)}):")
            for f in sorted(common)[:5]:
                print(f"    - {f}")
            if len(common) > 5:
                print(f"    ... and {len(common) - 5} more")
    
    # Key insights
    print(f"\n" + "=" * 80)
    print("🔍 KEY INSIGHTS")
    print("=" * 80)
    
    parent_set = set(parent_features)
    v3_set = set(v3_features)
    
    overlap = parent_set.intersection(v3_set)
    only_parent = parent_set - v3_set
    only_v3 = v3_set - parent_set
    
    print(f"\n1. Feature Overlap:")
    print(f"   - Common features: {len(overlap)} ({len(overlap)/len(parent_features)*100:.1f}% of parent)")
    print(f"   - Parent-only: {len(only_parent)} ({len(only_parent)/len(parent_features)*100:.1f}%)")
    print(f"   - V3-only: {len(only_v3)} ({len(only_v3)/len(v3_features)*100:.1f}%)")
    
    print(f"\n2. Critical Missing Features in V3:")
    critical = []
    for feat in only_parent:
        if 'player' in feat.lower():
            critical.append(feat)
    print(f"   - Player-based features: {len(critical)}")
    for f in sorted(critical):
        print(f"     • {f}")
    
    print(f"\n3. Feature Engineering Approach:")
    print(f"   - Parent: Curated, specific metrics (71 features)")
    print(f"   - V3: Comprehensive, auto-generated (150 features)")
    print(f"   - Parent focuses on proven predictors")
    print(f"   - V3 casts a wider net, relies on model to select")
    
    # Save detailed report
    report_path = '/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3/feature_comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("PARENT PIPELINE FEATURES\n")
        f.write("=" * 80 + "\n")
        for feat in parent_features:
            f.write(f"{feat}\n")
        
        f.write("\n\nPIPELINE V3 FEATURES\n")
        f.write("=" * 80 + "\n")
        for feat in v3_features:
            f.write(f"{feat}\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
