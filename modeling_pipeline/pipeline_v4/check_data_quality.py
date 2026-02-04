#!/usr/bin/env python3
"""Check training data for null, zero, and default values."""

import pandas as pd
import numpy as np

# Load training data
print("Loading training data...")
df = pd.read_csv('data/training_data.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Check for null values
print("\n" + "="*80)
print("NULL/NaN VALUES CHECK")
print("="*80)
null_counts = df.isnull().sum()
null_cols = null_counts[null_counts > 0].sort_values(ascending=False)

if len(null_cols) > 0:
    print(f"\n❌ Found {len(null_cols)} columns with null values:")
    for col, count in null_cols.items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} nulls ({pct:.2f}%)")
else:
    print("\n✓ No null values found")

# Check for all-zero columns
print("\n" + "="*80)
print("ALL-ZERO COLUMNS CHECK")
print("="*80)
numeric_cols = df.select_dtypes(include=[np.number]).columns
zero_cols = []

for col in numeric_cols:
    if (df[col] == 0).all():
        zero_cols.append(col)

if zero_cols:
    print(f"\n❌ Found {len(zero_cols)} columns with ALL zeros:")
    for col in zero_cols:
        print(f"  {col}")
else:
    print("\n✓ No all-zero columns found")

# Check for high percentage of zeros
print("\n" + "="*80)
print("HIGH ZERO PERCENTAGE CHECK (>95%)")
print("="*80)
high_zero_cols = []

for col in numeric_cols:
    zero_pct = (df[col] == 0).sum() / len(df) * 100
    if zero_pct > 95 and col not in zero_cols:  # Already reported all-zero columns
        high_zero_cols.append((col, zero_pct))

if high_zero_cols:
    print(f"\n⚠️  Found {len(high_zero_cols)} columns with >95% zeros:")
    for col, pct in sorted(high_zero_cols, key=lambda x: x[1], reverse=True):
        print(f"  {col}: {pct:.2f}% zeros")
else:
    print("\n✓ No columns with excessive zeros")

# Check for constant/default values
print("\n" + "="*80)
print("CONSTANT VALUE CHECK")
print("="*80)
constant_cols = []

for col in numeric_cols:
    if df[col].nunique() == 1:
        constant_cols.append((col, df[col].iloc[0]))

if constant_cols:
    print(f"\n❌ Found {len(constant_cols)} columns with constant values:")
    for col, val in constant_cols:
        print(f"  {col}: always {val}")
else:
    print("\n✓ No constant columns found")

# Check specific feature categories
print("\n" + "="*80)
print("FEATURE CATEGORY ANALYSIS")
print("="*80)

# Pillar 1 features
pillar1_features = [col for col in df.columns if any(x in col.lower() for x in ['elo', 'position', 'form', 'h2h', 'home_adv'])]
print(f"\nPillar 1 (Fundamentals): {len(pillar1_features)} features")
pillar1_issues = [col for col in pillar1_features if col in zero_cols or col in [c for c, _ in high_zero_cols]]
if pillar1_issues:
    print(f"  ⚠️  Issues in: {pillar1_issues}")

# Pillar 2 features
pillar2_features = [col for col in df.columns if any(x in col.lower() for x in ['xg', 'shot', 'attack', 'defense', 'possession'])]
print(f"\nPillar 2 (Modern Analytics): {len(pillar2_features)} features")
pillar2_issues = [col for col in pillar2_features if col in zero_cols or col in [c for c, _ in high_zero_cols]]
if pillar2_issues:
    print(f"  ⚠️  Issues in: {pillar2_issues}")

# Pillar 3 features
pillar3_features = [col for col in df.columns if any(x in col.lower() for x in ['momentum', 'fixture', 'player', 'context', 'streak'])]
print(f"\nPillar 3 (Hidden Edges): {len(pillar3_features)} features")
pillar3_issues = [col for col in pillar3_features if col in zero_cols or col in [c for c, _ in high_zero_cols]]
if pillar3_issues:
    print(f"  ⚠️  Issues in: {pillar3_issues}")

# Sample some rows to show actual data
print("\n" + "="*80)
print("SAMPLE DATA (first 5 rows, first 15 columns)")
print("="*80)
print(df.iloc[:5, :15].to_string())

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
total_issues = len(null_cols) + len(zero_cols) + len(high_zero_cols) + len(constant_cols)
if total_issues == 0:
    print("✓ Training data looks good - no major issues detected!")
else:
    print(f"⚠️  Found {total_issues} potential issues:")
    print(f"  - {len(null_cols)} columns with nulls")
    print(f"  - {len(zero_cols)} all-zero columns")
    print(f"  - {len(high_zero_cols)} columns with >95% zeros")
    print(f"  - {len(constant_cols)} constant columns")
