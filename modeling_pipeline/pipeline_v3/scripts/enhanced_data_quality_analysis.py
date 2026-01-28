#!/usr/bin/env python3
"""
Enhanced Data Quality Analysis with Visualizations

This script provides a detailed data quality report with:
- Statistical summaries
- Data completeness metrics
- Feature generation readiness
- Actionable recommendations

Usage:
    python scripts/enhanced_data_quality_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_data_quality():
    """Comprehensive data quality analysis with detailed metrics."""
    
    data_dir = Path('data/csv')
    
    print("\n" + "=" * 100)
    print("ENHANCED DATA QUALITY ANALYSIS - PIPELINE V3")
    print("=" * 100)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Directory: {data_dir}")
    
    # Load data
    print("\n📂 Loading CSV files...")
    fixtures = pd.read_csv(data_dir / 'fixtures.csv')
    statistics = pd.read_csv(data_dir / 'statistics.csv', low_memory=False)
    lineups = pd.read_csv(data_dir / 'lineups.csv', low_memory=False)
    sidelined = pd.read_csv(data_dir / 'sidelined.csv')
    
    print(f"   ✅ Fixtures: {len(fixtures):,} rows")
    print(f"   ✅ Statistics: {len(statistics):,} rows")
    print(f"   ✅ Lineups: {len(lineups):,} rows")
    print(f"   ✅ Sidelined: {len(sidelined):,} rows")
    
    # ============================================================================
    # 1. FIXTURES ANALYSIS
    # ============================================================================
    print("\n" + "=" * 100)
    print("1. FIXTURES DATA QUALITY")
    print("=" * 100)
    
    print(f"\n📊 Volume:")
    print(f"   Total fixtures: {len(fixtures):,}")
    print(f"   Unique fixture IDs: {fixtures['fixture_id'].nunique():,}")
    print(f"   Duplicate IDs: {len(fixtures) - fixtures['fixture_id'].nunique():,}")
    
    fixtures['date'] = pd.to_datetime(fixtures['starting_at'])
    print(f"\n📅 Temporal Coverage:")
    print(f"   Date range: {fixtures['date'].min()} to {fixtures['date'].max()}")
    print(f"   Years: {fixtures['date'].dt.year.min()} - {fixtures['date'].dt.year.max()}")
    print(f"   Total years: {fixtures['date'].dt.year.nunique()}")
    
    print(f"\n🏆 League Coverage:")
    print(f"   Unique leagues: {fixtures['league_id'].nunique()}")
    print(f"   Unique teams: {pd.concat([fixtures['home_team_id'], fixtures['away_team_id']]).nunique()}")
    
    print(f"\n🎯 Result Distribution:")
    result_counts = fixtures['result'].value_counts()
    total_with_result = result_counts.sum()
    for result, count in result_counts.items():
        pct = (count / total_with_result) * 100
        print(f"   {result}: {count:,} ({pct:.1f}%)")
    
    missing_results = fixtures['result'].isnull().sum()
    if missing_results > 0:
        print(f"   Missing: {missing_results:,} ({(missing_results/len(fixtures))*100:.1f}%)")
    
    print(f"\n✅ Data Quality:")
    print(f"   Duplicate fixture_ids: {fixtures.duplicated(subset=['fixture_id']).sum():,}")
    print(f"   Negative scores: {((fixtures['home_score'] < 0) | (fixtures['away_score'] < 0)).sum()}")
    
    # Check result consistency
    fixtures_with_result = fixtures[fixtures['result'].notna() & fixtures['home_score'].notna() & fixtures['away_score'].notna()].copy()
    if len(fixtures_with_result) > 0:
        fixtures_with_result['expected_result'] = fixtures_with_result.apply(
            lambda row: 'H' if row['home_score'] > row['away_score'] 
            else ('A' if row['home_score'] < row['away_score'] else 'D'),
            axis=1
        )
        mismatches = (fixtures_with_result['result'] != fixtures_with_result['expected_result']).sum()
        print(f"   Result/Score mismatches: {mismatches:,}")
    
    # ============================================================================
    # 2. STATISTICS ANALYSIS
    # ============================================================================
    print("\n" + "=" * 100)
    print("2. STATISTICS DATA QUALITY")
    print("=" * 100)
    
    print(f"\n📊 Volume:")
    print(f"   Total rows: {len(statistics):,}")
    print(f"   Unique fixtures: {statistics['fixture_id'].nunique():,}")
    print(f"   Expected rows (2 per fixture): {fixtures['fixture_id'].nunique() * 2:,}")
    
    # Coverage
    fixture_ids = set(fixtures['fixture_id'].unique())
    stats_fixture_ids = set(statistics['fixture_id'].unique())
    coverage = len(stats_fixture_ids & fixture_ids)
    print(f"\n📈 Coverage:")
    print(f"   Fixtures with statistics: {coverage:,} / {len(fixture_ids):,} ({(coverage/len(fixture_ids))*100:.1f}%)")
    print(f"   Missing statistics: {len(fixture_ids - stats_fixture_ids):,}")
    
    # Key statistics availability
    print(f"\n🎯 Key Statistics Availability:")
    key_stats = {
        'shots_total': 'Shots Total',
        'shots_on_target': 'Shots on Target',
        'shots_inside_box': 'Shots Inside Box',
        'shots_outside_box': 'Shots Outside Box',
        'possession': 'Possession %',
        'passes_total': 'Passes Total',
        'passes_accurate': 'Passes Accurate',
        'interceptions': 'Interceptions',
        'fouls': 'Fouls',
        'attacks': 'Attacks',
        'dangerous_attacks': 'Dangerous Attacks',
    }
    
    for col, name in key_stats.items():
        if col in statistics.columns:
            available = statistics[col].notna().sum()
            total = len(statistics)
            pct = (available / total) * 100
            status = "✅" if pct > 90 else ("⚠️" if pct > 50 else "❌")
            print(f"   {status} {name:25s}: {available:7,} / {total:7,} ({pct:5.1f}%)")
        else:
            print(f"   ❌ {name:25s}: NOT FOUND")
    
    # ============================================================================
    # 3. LINEUPS ANALYSIS
    # ============================================================================
    print("\n" + "=" * 100)
    print("3. LINEUPS DATA QUALITY")
    print("=" * 100)
    
    print(f"\n📊 Volume:")
    print(f"   Total rows: {len(lineups):,}")
    print(f"   Unique fixtures: {lineups['fixture_id'].nunique():,}")
    print(f"   Unique players: {lineups['player_id'].nunique():,}")
    
    # Coverage
    lineup_fixture_ids = set(lineups['fixture_id'].unique())
    coverage = len(lineup_fixture_ids & fixture_ids)
    print(f"\n📈 Coverage:")
    print(f"   Fixtures with lineups: {coverage:,} / {len(fixture_ids):,} ({(coverage/len(fixture_ids))*100:.1f}%)")
    print(f"   Missing lineups: {len(fixture_ids - lineup_fixture_ids):,}")
    
    # Players per fixture
    players_per_fixture = lineups.groupby('fixture_id').size()
    print(f"\n⚽ Lineup Composition:")
    print(f"   Avg players per fixture: {players_per_fixture.mean():.1f}")
    print(f"   Median: {players_per_fixture.median():.0f}")
    print(f"   Min: {players_per_fixture.min()}")
    print(f"   Max: {players_per_fixture.max()}")
    
    if 'is_starter' in lineups.columns:
        starter_counts = lineups['is_starter'].value_counts()
        print(f"\n   Starters: {starter_counts.get(True, 0):,} ({(starter_counts.get(True, 0)/len(lineups))*100:.1f}%)")
        print(f"   Bench: {starter_counts.get(False, 0):,} ({(starter_counts.get(False, 0)/len(lineups))*100:.1f}%)")
    
    # Player statistics
    print(f"\n🎯 Player Statistics Availability:")
    player_stats = {
        'rating': 'Rating',
        'minutes_played': 'Minutes Played',
        'goals': 'Goals',
        'assists': 'Assists',
        'shots_total': 'Shots',
        'passes_total': 'Passes',
        'tackles': 'Tackles',
        'interceptions': 'Interceptions',
    }
    
    for col, name in player_stats.items():
        if col in lineups.columns:
            available = lineups[col].notna().sum()
            total = len(lineups)
            pct = (available / total) * 100
            status = "✅" if pct > 70 else ("⚠️" if pct > 30 else "❌")
            print(f"   {status} {name:20s}: {available:9,} / {total:9,} ({pct:5.1f}%)")
        else:
            print(f"   ❌ {name:20s}: NOT FOUND")
    
    # Rating distribution
    if 'rating' in lineups.columns:
        ratings = pd.to_numeric(lineups['rating'], errors='coerce').dropna()
        if len(ratings) > 0:
            print(f"\n📊 Rating Distribution:")
            print(f"   Count: {len(ratings):,}")
            print(f"   Mean: {ratings.mean():.2f}")
            print(f"   Median: {ratings.median():.2f}")
            print(f"   Min: {ratings.min():.2f}")
            print(f"   Max: {ratings.max():.2f}")
            print(f"   Std: {ratings.std():.2f}")
            
            # Check for invalid ratings
            invalid = ((ratings < 0) | (ratings > 10)).sum()
            if invalid > 0:
                print(f"   ⚠️ Invalid ratings (< 0 or > 10): {invalid:,}")
    
    # ============================================================================
    # 4. SIDELINED ANALYSIS
    # ============================================================================
    print("\n" + "=" * 100)
    print("4. SIDELINED DATA QUALITY")
    print("=" * 100)
    
    print(f"\n📊 Volume:")
    print(f"   Total rows: {len(sidelined):,}")
    print(f"   Unique fixtures: {sidelined['fixture_id'].nunique():,}")
    print(f"   Unique players: {sidelined['player_id'].nunique():,}")
    
    # Check for dates
    print(f"\n📅 Date Availability:")
    for col in ['start_date', 'end_date', 'category']:
        if col in sidelined.columns:
            available = sidelined[col].notna().sum()
            total = len(sidelined)
            pct = (available / total) * 100
            status = "✅" if pct > 90 else ("⚠️" if pct > 50 else "❌")
            print(f"   {status} {col:15s}: {available:7,} / {total:7,} ({pct:5.1f}%)")
        else:
            print(f"   ❌ {col:15s}: NOT FOUND")
    
    # ============================================================================
    # 5. CROSS-TABLE ANALYSIS
    # ============================================================================
    print("\n" + "=" * 100)
    print("5. CROSS-TABLE DATA COMPLETENESS")
    print("=" * 100)
    
    print(f"\n🔗 Coverage Summary:")
    print(f"   Total unique fixtures: {len(fixture_ids):,}")
    print(f"   With statistics: {len(stats_fixture_ids & fixture_ids):,} ({(len(stats_fixture_ids & fixture_ids)/len(fixture_ids))*100:.1f}%)")
    print(f"   With lineups: {len(lineup_fixture_ids & fixture_ids):,} ({(len(lineup_fixture_ids & fixture_ids)/len(fixture_ids))*100:.1f}%)")
    
    # Complete data
    complete = fixture_ids & stats_fixture_ids & lineup_fixture_ids
    print(f"\n✅ Fixtures with COMPLETE data (fixtures + stats + lineups):")
    print(f"   {len(complete):,} / {len(fixture_ids):,} ({(len(complete)/len(fixture_ids))*100:.1f}%)")
    
    # ============================================================================
    # 6. FEATURE GENERATION READINESS
    # ============================================================================
    print("\n" + "=" * 100)
    print("6. FEATURE GENERATION READINESS")
    print("=" * 100)
    
    complete_pct = (len(complete) / len(fixture_ids)) * 100
    
    print(f"\n📊 Pillar 1: Fundamentals (50 features)")
    print(f"   Status: ✅ READY")
    print(f"   Coverage: {complete_pct:.1f}%")
    print(f"   Features: Elo, Form, H2H, League Position, Home Advantage")
    
    print(f"\n📊 Pillar 2: Modern Analytics (60 features)")
    stats_coverage = (len(stats_fixture_ids & fixture_ids) / len(fixture_ids)) * 100
    status = "✅ READY" if stats_coverage > 90 else "⚠️ LIMITED"
    print(f"   Status: {status}")
    print(f"   Coverage: {stats_coverage:.1f}%")
    print(f"   Features: Derived xG, Shot Analysis, Defensive Metrics, Attack Patterns")
    
    print(f"\n📊 Pillar 3: Hidden Edges (40 features)")
    lineup_coverage = (len(lineup_fixture_ids & fixture_ids) / len(fixture_ids)) * 100
    
    # Check sidelined dates
    sidelined_dates_available = False
    if 'start_date' in sidelined.columns and 'end_date' in sidelined.columns:
        sidelined_dates_pct = (sidelined['start_date'].notna().sum() / len(sidelined)) * 100
        sidelined_dates_available = sidelined_dates_pct > 50
    
    if lineup_coverage > 90 and sidelined_dates_available:
        status = "✅ READY"
        feasible = 40
    elif lineup_coverage > 90:
        status = "⚠️ PARTIAL"
        feasible = 15
    else:
        status = "❌ LIMITED"
        feasible = 10
    
    print(f"   Status: {status}")
    print(f"   Coverage: {lineup_coverage:.1f}% (lineups)")
    print(f"   Feasible features: ~{feasible}/40")
    print(f"   Available: Momentum, Fixture-Adjusted")
    if not sidelined_dates_available:
        print(f"   Missing: Player Availability (25 features) - sidelined dates not available")
    
    # Total
    total_feasible = 50 + (60 if stats_coverage > 90 else 30) + feasible
    print(f"\n🎯 TOTAL FEASIBLE FEATURES: ~{total_feasible}/150")
    
    # ============================================================================
    # 7. RECOMMENDATIONS
    # ============================================================================
    print("\n" + "=" * 100)
    print("7. RECOMMENDATIONS")
    print("=" * 100)
    
    issues = []
    warnings = []
    
    # Check statistics coverage
    if stats_coverage < 90:
        issues.append(f"Statistics coverage is {stats_coverage:.1f}% (target: >90%)")
    
    # Check lineup coverage
    if lineup_coverage < 80:
        warnings.append(f"Lineup coverage is {lineup_coverage:.1f}% (target: >80%)")
    
    # Check duplicates
    duplicates = fixtures.duplicated(subset=['fixture_id']).sum()
    if duplicates > 0:
        warnings.append(f"{duplicates:,} duplicate fixture IDs - deduplicate before training")
    
    # Check sidelined dates
    if not sidelined_dates_available:
        issues.append("Sidelined data missing dates - cannot implement player availability features")
    
    # Check result mismatches
    if 'expected_result' in fixtures_with_result.columns:
        mismatches = (fixtures_with_result['result'] != fixtures_with_result['expected_result']).sum()
        if mismatches > 0:
            issues.append(f"{mismatches:,} result/score mismatches detected")
    
    print(f"\n📋 Status:")
    if len(issues) == 0 and len(warnings) == 0:
        print("   ✅ Data quality is EXCELLENT!")
        print("   ✅ Ready for training dataset generation")
    else:
        if len(issues) > 0:
            print(f"\n   ❌ Critical Issues ({len(issues)}):")
            for issue in issues:
                print(f"      • {issue}")
        
        if len(warnings) > 0:
            print(f"\n   ⚠️ Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"      • {warning}")
        
        if len(issues) == 0:
            print("\n   ✅ Data quality is GOOD (minor warnings only)")
            print("   ✅ Ready for training dataset generation")
    
    print(f"\n💡 Next Steps:")
    if duplicates > 0:
        print(f"   1. Deduplicate fixtures: df.drop_duplicates(subset=['fixture_id'])")
    print(f"   2. Generate training dataset: python scripts/generate_training_data.py")
    print(f"   3. Train model: python scripts/train_model.py")
    
    if not sidelined_dates_available:
        print(f"\n   Optional: Investigate sidelined data to unlock 25 player availability features")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    analyze_data_quality()
