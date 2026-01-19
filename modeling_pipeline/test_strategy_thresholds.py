"""Quick test to find good thresholds"""
import pandas as pd
import numpy as np

# Load test set
df = pd.read_csv('data/processed/sportmonks_features.csv')
df = df[df['season_name'].isin(['2023/2024'])].tail(500)

print("Analysis of test set predictions:")
print(f"Total matches: {len(df)}")

# Check target distribution
if 'target' in df.columns:
    print(f"\nActual outcomes:")
    print(df['target'].value_counts())
    print(df['target'].value_counts(normalize=True))
    
# Check if we have probabilities from previous runs
if 'home_prob' in df.columns:
    print(f"\nAverage probabilities:")
    print(f"  Home: {df['home_prob'].mean():.1%}")
    print(f"  Draw: {df['draw_prob'].mean():.1%}")
    print(f"  Away: {df['away_prob'].mean():.1%}")
    
    print(f"\nProbability distributions:")
    print(f"  Home prob > 50%: {(df['home_prob'] > 0.50).sum()} matches")
    print(f"  Home prob > 60%: {(df['home_prob'] > 0.60).sum()} matches")
    print(f"  Away prob > 40%: {(df['away_prob'] > 0.40).sum()} matches")
    print(f"  Away prob > 50%: {(df['away_prob'] > 0.50).sum()} matches")
else:
    print("\nNo probability columns found - need to run predictions first")
