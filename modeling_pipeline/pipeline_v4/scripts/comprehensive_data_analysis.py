"""
Comprehensive Data Quality Analysis for V4 Training Data.

Analyzes:
- Missing values
- Feature distributions
- Constant/low-variance features
- Feature correlations
- Outliers
- Target distribution
- Data quality issues
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_missing_values(df):
    """Analyze missing values."""
    print("\n" + "=" * 80)
    print("MISSING VALUES ANALYSIS")
    print("=" * 80)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'column': missing.index,
        'missing_count': missing.values,
        'missing_pct': missing_pct.values
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)

    if len(missing_df) > 0:
        print(f"\nFeatures with missing values: {len(missing_df)}")
        print(missing_df.to_string(index=False))

        # High missing features
        high_missing = missing_df[missing_df['missing_pct'] > 50]
        if len(high_missing) > 0:
            print(f"\n⚠️  WARNING: {len(high_missing)} features have >50% missing values!")
            print("Consider removing these features:")
            print(high_missing['column'].tolist())
    else:
        print("✅ No missing values found!")

    return missing_df

def analyze_constant_features(df, features):
    """Find constant or near-constant features."""
    print("\n" + "=" * 80)
    print("CONSTANT/LOW-VARIANCE FEATURES")
    print("=" * 80)

    constant_features = []
    low_variance_features = []

    for col in features:
        nunique = df[col].nunique()
        variance = df[col].var()

        if nunique == 1:
            constant_features.append(col)
        elif nunique <= 3 or variance < 0.01:
            low_variance_features.append((col, nunique, variance))

    if constant_features:
        print(f"\n⚠️  WARNING: {len(constant_features)} constant features (remove these):")
        for f in constant_features:
            print(f"  - {f}: {df[f].iloc[0]}")

    if low_variance_features:
        print(f"\n⚠️  {len(low_variance_features)} low-variance features:")
        for f, nu, var in sorted(low_variance_features, key=lambda x: x[2])[:10]:
            print(f"  - {f}: unique={nu}, variance={var:.6f}")

    if not constant_features and not low_variance_features:
        print("✅ No constant or low-variance features!")

    return constant_features, low_variance_features

def analyze_distributions(df, features):
    """Analyze feature distributions."""
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 80)

    stats = df[features].describe().T
    stats['zeros_pct'] = (df[features] == 0).sum() / len(df) * 100
    stats['skewness'] = df[features].skew()
    stats['kurtosis'] = df[features].kurtosis()

    # High skewness
    high_skew = stats[abs(stats['skewness']) > 3].sort_values('skewness', ascending=False)
    if len(high_skew) > 0:
        print(f"\n⚠️  {len(high_skew)} features with high skewness (|skew| > 3):")
        print(high_skew[['mean', 'std', 'skewness']].head(10).to_string())
        print("\nConsider log transformation for these features")

    # High percentage of zeros
    high_zeros = stats[stats['zeros_pct'] > 50].sort_values('zeros_pct', ascending=False)
    if len(high_zeros) > 0:
        print(f"\n⚠️  {len(high_zeros)} features with >50% zeros:")
        print(high_zeros[['zeros_pct', 'mean', 'std']].head(10).to_string())

    # Save full stats
    stats_file = Path('data/feature_statistics.csv')
    stats.to_csv(stats_file)
    print(f"\n✅ Full statistics saved to: {stats_file}")

    return stats

def analyze_correlations(df, features, threshold=0.95):
    """Find highly correlated features."""
    print("\n" + "=" * 80)
    print(f"HIGH CORRELATION ANALYSIS (threshold={threshold})")
    print("=" * 80)

    # Calculate correlation matrix
    corr_matrix = df[features].corr().abs()

    # Find high correlations (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    if high_corr_pairs:
        print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated pairs (correlation > {threshold}):")
        high_corr_df = pd.DataFrame(high_corr_pairs, columns=['feature_1', 'feature_2', 'correlation'])
        high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
        print(high_corr_df.head(20).to_string(index=False))
        print("\nConsider removing one feature from each pair to reduce multicollinearity")

        # Save to file
        corr_file = Path('data/high_correlations.csv')
        high_corr_df.to_csv(corr_file, index=False)
        print(f"\n✅ Full correlation analysis saved to: {corr_file}")
    else:
        print(f"✅ No feature pairs with correlation > {threshold}")

    return high_corr_pairs

def analyze_outliers(df, features):
    """Detect outliers using IQR method."""
    print("\n" + "=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)

    outlier_summary = []

    for col in features:
        # Skip boolean/binary columns
        if df[col].nunique() <= 2:
            continue

        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:  # Skip constant features
                continue

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(df)) * 100

            if outlier_pct > 5:
                outlier_summary.append({
                    'feature': col,
                    'outlier_count': outliers,
                    'outlier_pct': outlier_pct,
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'Q1': Q1,
                    'Q3': Q3
                })
        except Exception as e:
            print(f"  Skipping {col}: {e}")
            continue

    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary).sort_values('outlier_pct', ascending=False)
        print(f"\n⚠️  {len(outlier_df)} features with >5% outliers:")
        print(outlier_df.head(20).to_string(index=False))
        print("\nThese may need capping or transformation")
    else:
        print("✅ No features with excessive outliers (>5%)")

    return outlier_summary

def analyze_target_distribution(df):
    """Analyze target variable distribution."""
    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION")
    print("=" * 80)

    target_counts = df['result'].value_counts()
    target_pct = df['result'].value_counts(normalize=True) * 100

    print(f"\nTotal samples: {len(df)}")
    print("\nClass distribution:")
    for outcome in ['H', 'D', 'A']:
        if outcome in target_counts.index:
            print(f"  {outcome}: {target_counts[outcome]:,} ({target_pct[outcome]:.2f}%)")

    # Check for imbalance
    max_pct = target_pct.max()
    min_pct = target_pct.min()
    imbalance_ratio = max_pct / min_pct

    if imbalance_ratio > 2:
        print(f"\n⚠️  WARNING: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        print("Consider using class weights or resampling techniques")
    else:
        print(f"\n✅ Classes are reasonably balanced (ratio: {imbalance_ratio:.2f})")

    return target_counts

def analyze_feature_groups(df):
    """Analyze features by pillar."""
    print("\n" + "=" * 80)
    print("FEATURE GROUP ANALYSIS")
    print("=" * 80)

    # Group features by pillar
    pillar1_keywords = ['elo', 'position', 'points', 'wins', 'draws', 'goals', 'h2h', 'home_', 'away_']
    pillar2_keywords = ['xg', 'shot', 'tackle', 'ppda', 'attack', 'possession', 'dangerous']
    pillar3_keywords = ['trend', 'streak', 'weighted', 'opponent', 'player', 'rest', 'derby']

    features = [c for c in df.columns if c not in ['fixture_id', 'home_team_id', 'away_team_id',
                                                     'season_id', 'league_id', 'match_date',
                                                     'home_score', 'away_score', 'result']]

    pillar1 = [f for f in features if any(k in f.lower() for k in ['elo', 'position', 'league', 'form', 'h2h', 'advantage'])]
    pillar2 = [f for f in features if any(k in f.lower() for k in ['xg', 'shot', 'tackle', 'ppda', 'attack', 'possession', 'dangerous', 'defensive'])]
    pillar3 = [f for f in features if any(k in f.lower() for k in ['trend', 'streak', 'weighted', 'opponent', 'player', 'rest', 'derby', 'relegation', 'top_6', 'bottom'])]

    # Remove overlap (feature can be in multiple)
    pillar2 = [f for f in pillar2 if f not in pillar1]
    pillar3 = [f for f in pillar3 if f not in pillar1 and f not in pillar2]
    other = [f for f in features if f not in pillar1 and f not in pillar2 and f not in pillar3]

    print(f"\nPillar 1 (Fundamentals): {len(pillar1)} features")
    print(f"Pillar 2 (Modern Analytics): {len(pillar2)} features")
    print(f"Pillar 3 (Hidden Edges): {len(pillar3)} features")
    print(f"Other: {len(other)} features")
    print(f"Total: {len(features)} features")

    # Check data quality by pillar
    for name, group in [('Pillar 1', pillar1), ('Pillar 2', pillar2), ('Pillar 3', pillar3)]:
        missing = df[group].isnull().sum().sum()
        total_values = len(df) * len(group)
        missing_pct = (missing / total_values) * 100
        print(f"\n{name}: {missing_pct:.2f}% missing values")

def main():
    print("=" * 80)
    print("COMPREHENSIVE DATA QUALITY ANALYSIS")
    print("=" * 80)

    # Load data
    data_file = Path('data/training_data.csv')
    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file)

    print(f"Shape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Identify feature columns
    metadata_cols = ['fixture_id', 'home_team_id', 'away_team_id', 'season_id',
                     'league_id', 'match_date', 'home_score', 'away_score', 'result']
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Metadata columns: {len(metadata_cols)}")

    # Run analyses
    missing_df = analyze_missing_values(df)
    constant_features, low_variance = analyze_constant_features(df, feature_cols)
    stats = analyze_distributions(df, feature_cols)
    high_corr = analyze_correlations(df, feature_cols, threshold=0.95)
    outliers = analyze_outliers(df, feature_cols)
    target_dist = analyze_target_distribution(df)
    analyze_feature_groups(df)

    # Summary recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    issues = []

    if len(missing_df) > 0:
        high_missing = missing_df[missing_df['missing_pct'] > 50]
        if len(high_missing) > 0:
            issues.append(f"Remove {len(high_missing)} features with >50% missing values")

    if constant_features:
        issues.append(f"Remove {len(constant_features)} constant features")

    if high_corr and len(high_corr) > 20:
        issues.append(f"Remove redundant features from {len(high_corr)} highly correlated pairs")

    if issues:
        print("\n🔴 DATA QUALITY ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ Overall data quality looks good!")

    print("\n📊 NEXT STEPS:")
    print("  1. Review feature_statistics.csv for detailed distributions")
    print("  2. Review high_correlations.csv for redundant features")
    print("  3. Consider feature engineering improvements")
    print("  4. Try advanced models (LightGBM, CatBoost, Neural Networks)")
    print("  5. Implement ensemble methods (stacking, blending)")

if __name__ == '__main__':
    main()
