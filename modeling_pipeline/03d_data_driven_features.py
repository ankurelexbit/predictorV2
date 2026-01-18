#!/usr/bin/env python3
"""
Data-Driven Player Features
===========================

Features based on ACTUAL patterns found in the data:
- Minimal rotation effect
- Moderate activity is beneficial  
- High variance teams identified
- Instability can be positive

Usage:
    python 03d_data_driven_features.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

# Based on analysis results
HIGH_VARIANCE_TEAMS = [
    'Fulham', 'Schalke 04', 'Peterboro', 'Crotone', 'Girona',
    'Darmstadt', 'Blackpool', 'Newcastle', 'Atalanta', 'Plymouth'
]

LOW_VARIANCE_TEAMS = [
    'Sevilla', 'Troyes', 'Juventus', 'Oxford', 'Brescia',
    'Benevento', 'Espanol', 'Spal', 'Greuther Furth', 'Paderborn'
]

def create_data_driven_features():
    """Create features based on actual data patterns."""
    
    print("\nDATA-DRIVEN FEATURE ENGINEERING")
    print("=" * 50)
    
    # Load data
    matches_df = pd.read_csv('data/processed/matches.csv')
    features_df = pd.read_csv('data/processed/features.csv')
    
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    enhanced_df = features_df.copy()
    
    # 1. ACTIVITY LEVEL (not fatigue - activity is good!)
    print("\n1. Adding activity level features...")
    
    for idx, row in features_df.iterrows():
        if idx % 1000 == 0:
            print(f"   Processing {idx}/{len(features_df)}...")
            
        # Count recent games
        home_recent = matches_df[
            ((matches_df['home_team'] == row['home_team']) | 
             (matches_df['away_team'] == row['home_team'])) &
            (matches_df['date'] < row['date']) &
            (matches_df['date'] >= row['date'] - timedelta(days=10))
        ]
        
        away_recent = matches_df[
            ((matches_df['home_team'] == row['away_team']) | 
             (matches_df['away_team'] == row['away_team'])) &
            (matches_df['date'] < row['date']) &
            (matches_df['date'] >= row['date'] - timedelta(days=10))
        ]
        
        enhanced_df.loc[idx, 'home_games_10d'] = len(home_recent)
        enhanced_df.loc[idx, 'away_games_10d'] = len(away_recent)
    
    # Optimal activity is 2 games in 10 days
    enhanced_df['home_optimal_activity'] = (enhanced_df['home_games_10d'] == 2).astype(int)
    enhanced_df['away_optimal_activity'] = (enhanced_df['away_games_10d'] == 2).astype(int)
    
    # Too much rest is bad (0 games)
    enhanced_df['home_too_rested'] = (enhanced_df['home_games_10d'] == 0).astype(int)
    enhanced_df['away_too_rested'] = (enhanced_df['away_games_10d'] == 0).astype(int)
    
    # 2. TEAM VARIANCE CATEGORIES
    print("\n2. Adding variance-based team categories...")
    
    enhanced_df['home_high_variance'] = enhanced_df['home_team'].isin(HIGH_VARIANCE_TEAMS).astype(int)
    enhanced_df['away_high_variance'] = enhanced_df['away_team'].isin(HIGH_VARIANCE_TEAMS).astype(int)
    enhanced_df['home_low_variance'] = enhanced_df['home_team'].isin(LOW_VARIANCE_TEAMS).astype(int)
    enhanced_df['away_low_variance'] = enhanced_df['away_team'].isin(LOW_VARIANCE_TEAMS).astype(int)
    
    # High variance team with good form = dangerous
    enhanced_df['home_variance_form'] = (
        enhanced_df['home_high_variance'] * 
        enhanced_df['home_form_5_ppg'].fillna(1.5)
    )
    enhanced_df['away_variance_form'] = (
        enhanced_df['away_high_variance'] * 
        enhanced_df['away_form_5_ppg'].fillna(1.5)
    )
    
    # 3. TACTICAL FLEXIBILITY (instability as positive)
    print("\n3. Calculating tactical flexibility...")
    
    # Teams with varying goal patterns might be tactically flexible
    # Calculate recent goal variance for each team
    for idx, row in features_df.iterrows():
        if idx % 2000 == 0:
            print(f"   Processing {idx}/{len(features_df)}...")
            
        # Get last 5 home games
        home_recent_goals = matches_df[
            (matches_df['home_team'] == row['home_team']) &
            (matches_df['date'] < row['date'])
        ].tail(5)['home_goals'].values
        
        away_recent_goals = matches_df[
            (matches_df['away_team'] == row['away_team']) &
            (matches_df['date'] < row['date'])
        ].tail(5)['away_goals'].values
        
        if len(home_recent_goals) >= 3:
            enhanced_df.loc[idx, 'home_tactical_flexibility'] = np.std(home_recent_goals)
        else:
            enhanced_df.loc[idx, 'home_tactical_flexibility'] = 1.0
            
        if len(away_recent_goals) >= 3:
            enhanced_df.loc[idx, 'away_tactical_flexibility'] = np.std(away_recent_goals)
        else:
            enhanced_df.loc[idx, 'away_tactical_flexibility'] = 1.0
    
    # 4. MATCH IMPORTANCE (teams perform better with moderate activity)
    print("\n4. Adding match importance indicators...")
    
    # League position difference as proxy for importance
    enhanced_df['position_gap'] = abs(enhanced_df['home_position'] - enhanced_df['away_position'])
    enhanced_df['high_stakes_match'] = (enhanced_df['position_gap'] <= 3).astype(int)
    
    # Teams with optimal activity in high stakes matches
    enhanced_df['home_primed_for_big_match'] = (
        enhanced_df['home_optimal_activity'] * 
        enhanced_df['high_stakes_match']
    )
    enhanced_df['away_primed_for_big_match'] = (
        enhanced_df['away_optimal_activity'] * 
        enhanced_df['high_stakes_match']
    )
    
    # 5. MOMENTUM FEATURES (activity breeds confidence)
    print("\n5. Adding momentum features...")
    
    # Active teams with good form have momentum
    enhanced_df['home_momentum'] = (
        (enhanced_df['home_games_10d'] >= 1) * 
        enhanced_df['home_form_5_ppg'].fillna(1.5)
    )
    enhanced_df['away_momentum'] = (
        (enhanced_df['away_games_10d'] >= 1) * 
        enhanced_df['away_form_5_ppg'].fillna(1.5)
    )
    
    # Save enhanced features
    output_path = Path('data/processed/features_data_driven.csv')
    enhanced_df.to_csv(output_path, index=False)
    
    # Summary
    new_features = [col for col in enhanced_df.columns if col not in features_df.columns]
    
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 50)
    print(f"✅ Added {len(new_features)} data-driven features")
    print(f"📁 Saved to {output_path}")
    
    print("\nNew features based on ACTUAL patterns:")
    for feat in new_features[:10]:
        print(f"  - {feat}")
    
    print("\nKey insights incorporated:")
    print("  ✓ Optimal activity is 2 games/10 days (not rest)")
    print("  ✓ High variance teams identified by name")
    print("  ✓ Tactical flexibility (variance) can be positive")
    print("  ✓ Too much rest (0 games) is detrimental")
    print("  ✓ Activity + form = momentum")
    
    return enhanced_df


if __name__ == "__main__":
    create_data_driven_features()