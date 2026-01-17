#!/usr/bin/env python3
"""
Data Quality Check Script
=========================

This script performs comprehensive checks on the features and data integrity:
1. Missing value analysis
2. Feature distribution checks
3. Data consistency validation
4. Outlier detection
5. Correlation analysis

Usage:
    python check_data_quality.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR
from utils import setup_logger

# Setup
logger = setup_logger("data_quality")


def load_data():
    """Load all processed data files."""
    matches_path = PROCESSED_DATA_DIR / "matches.csv"
    features_path = PROCESSED_DATA_DIR / "features.csv"
    elo_path = PROCESSED_DATA_DIR / "elo_ratings.csv"
    
    logger.info(f"Loading data from {PROCESSED_DATA_DIR}")
    
    matches_df = pd.read_csv(matches_path, parse_dates=['date'])
    features_df = pd.read_csv(features_path, parse_dates=['date'])
    elo_df = pd.read_csv(elo_path)
    
    return matches_df, features_df, elo_df


def check_missing_values(df, name):
    """Analyze missing values in dataframe."""
    print(f"\n{'='*60}")
    print(f"MISSING VALUE ANALYSIS: {name}")
    print(f"{'='*60}")
    
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    if len(missing_stats) > 0:
        print(f"\nColumns with missing values:")
        print(missing_stats.to_string(index=False))
    else:
        print("\n✓ No missing values found!")
    
    return missing_stats


def check_data_consistency(matches_df, features_df):
    """Check consistency between matches and features."""
    print(f"\n{'='*60}")
    print(f"DATA CONSISTENCY CHECKS")
    print(f"{'='*60}")
    
    issues = []
    
    # 1. Check if all matches have features
    matches_ids = set(matches_df['match_id'])
    features_ids = set(features_df['match_id'])
    
    missing_features = matches_ids - features_ids
    extra_features = features_ids - matches_ids
    
    if missing_features:
        issues.append(f"❌ {len(missing_features)} matches without features")
    else:
        print("✓ All matches have corresponding features")
    
    if extra_features:
        issues.append(f"❌ {len(extra_features)} features without matches")
    else:
        print("✓ No orphaned features")
    
    # 2. Check date consistency
    merged = pd.merge(matches_df[['match_id', 'date']], 
                     features_df[['match_id', 'date']], 
                     on='match_id', 
                     suffixes=('_match', '_feature'))
    
    date_mismatch = merged[merged['date_match'] != merged['date_feature']]
    if len(date_mismatch) > 0:
        issues.append(f"❌ {len(date_mismatch)} date mismatches between matches and features")
    else:
        print("✓ All dates match between matches and features")
    
    # 3. Check result consistency
    result_check = pd.merge(matches_df[['match_id', 'result', 'home_goals', 'away_goals']], 
                           features_df[['match_id', 'result']], 
                           on='match_id', 
                           suffixes=('_match', '_feature'))
    
    result_mismatch = result_check[result_check['result_match'] != result_check['result_feature']]
    if len(result_mismatch) > 0:
        issues.append(f"❌ {len(result_mismatch)} result mismatches")
    else:
        print("✓ All results consistent")
    
    # 4. Check goal encoding
    for _, row in matches_df.iterrows():
        if pd.notna(row['home_goals']) and pd.notna(row['away_goals']):
            if row['home_goals'] > row['away_goals'] and row['result'] != 'H':
                issues.append(f"❌ Match {row['match_id']}: Home win but result={row['result']}")
            elif row['home_goals'] < row['away_goals'] and row['result'] != 'A':
                issues.append(f"❌ Match {row['match_id']}: Away win but result={row['result']}")
            elif row['home_goals'] == row['away_goals'] and row['result'] != 'D':
                issues.append(f"❌ Match {row['match_id']}: Draw but result={row['result']}")
    
    if not any("goal encoding" in str(issue) for issue in issues):
        print("✓ Goal encoding consistent with results")
    
    return issues


def check_feature_sanity(features_df):
    """Check if features have sensible values."""
    print(f"\n{'='*60}")
    print(f"FEATURE SANITY CHECKS")
    print(f"{'='*60}")
    
    issues = []
    
    # 1. Check Elo ratings range
    elo_cols = ['home_elo', 'away_elo']
    for col in elo_cols:
        min_val = features_df[col].min()
        max_val = features_df[col].max()
        print(f"\n{col}: min={min_val:.1f}, max={max_val:.1f}")
        
        if min_val < 1000 or max_val > 2000:
            issues.append(f"⚠️  {col} has unusual range: [{min_val:.1f}, {max_val:.1f}]")
    
    # 2. Check probabilities sum to 1
    prob_cols = ['elo_prob_home', 'elo_prob_draw', 'elo_prob_away']
    prob_sum = features_df[prob_cols].sum(axis=1)
    prob_errors = features_df[abs(prob_sum - 1.0) > 0.001]
    
    if len(prob_errors) > 0:
        issues.append(f"❌ {len(prob_errors)} rows where Elo probabilities don't sum to 1")
    else:
        print("\n✓ All Elo probabilities sum to 1")
    
    # 3. Check form values
    form_cols = [col for col in features_df.columns if 'form' in col and 'ppg' in col]
    for col in form_cols:
        valid_data = features_df[features_df[col].notna()]
        if len(valid_data) > 0:
            max_ppg = valid_data[col].max()
            if max_ppg > 3.0:
                issues.append(f"⚠️  {col} has max value {max_ppg:.2f} (>3.0 points per game)")
    
    # 4. Check rest days
    rest_cols = ['home_rest_days', 'away_rest_days']
    for col in rest_cols:
        valid_data = features_df[features_df[col].notna()]
        if len(valid_data) > 0:
            max_rest = valid_data[col].max()
            min_rest = valid_data[col].min()
            if max_rest > 30:
                issues.append(f"⚠️  {col} has unusually high value: {max_rest} days")
            if min_rest < 1:
                issues.append(f"⚠️  {col} has value < 1: {min_rest} days")
    
    # 5. Check positions
    position_cols = ['home_position', 'away_position']
    for col in position_cols:
        valid_data = features_df[features_df[col].notna()]
        if len(valid_data) > 0:
            min_pos = valid_data[col].min()
            max_pos = valid_data[col].max()
            if min_pos < 1:
                issues.append(f"❌ {col} has position < 1: {min_pos}")
            if max_pos > 24:  # Max teams in Championship
                issues.append(f"⚠️  {col} has position > 24: {max_pos}")
    
    # 6. Check head-to-head consistency
    h2h_matches = features_df[features_df['h2h_total'].notna()]
    h2h_sum_check = h2h_matches['h2h_home_wins'] + h2h_matches['h2h_draws'] + h2h_matches['h2h_away_wins']
    h2h_errors = h2h_matches[h2h_sum_check != h2h_matches['h2h_total']]
    
    if len(h2h_errors) > 0:
        issues.append(f"❌ {len(h2h_errors)} rows where H2H wins+draws+losses != total")
    else:
        print("\n✓ Head-to-head statistics are consistent")
    
    return issues


def check_outliers(features_df):
    """Detect outliers in numerical features."""
    print(f"\n{'='*60}")
    print(f"OUTLIER DETECTION")
    print(f"{'='*60}")
    
    numerical_cols = features_df.select_dtypes(include=[np.number]).columns
    outlier_stats = []
    
    for col in numerical_cols:
        if col in ['match_id', 'result_numeric']:
            continue
            
        data = features_df[col].dropna()
        if len(data) == 0:
            continue
        
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        if len(outliers) > 0:
            outlier_stats.append({
                'column': col,
                'outlier_count': len(outliers),
                'outlier_pct': len(outliers) / len(data) * 100,
                'min_outlier': outliers.min(),
                'max_outlier': outliers.max(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    
    outlier_df = pd.DataFrame(outlier_stats).sort_values('outlier_pct', ascending=False)
    
    print("\nColumns with outliers (IQR method):")
    print(outlier_df[outlier_df['outlier_pct'] > 1.0].to_string(index=False))
    
    return outlier_df


def check_correlations(features_df):
    """Check feature correlations."""
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    # Select numerical features
    feature_cols = [col for col in features_df.columns 
                   if col not in ['match_id', 'date', 'league_code', 'season', 
                                 'home_team', 'away_team', 'result']]
    
    corr_matrix = features_df[feature_cols].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8 and not pd.isna(corr_val):
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (|r| > 0.8):")
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print("\n✓ No highly correlated feature pairs found")
    
    return corr_matrix


def plot_feature_distributions(features_df, output_dir):
    """Plot distributions of key features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Select key features to plot
    key_features = [
        'home_elo', 'away_elo', 'elo_diff',
        'home_form_5_ppg', 'away_form_5_ppg',
        'home_rest_days', 'away_rest_days',
        'home_position', 'away_position'
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(key_features):
        if idx < len(axes):
            data = features_df[col].dropna()
            axes[idx].hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            
            # Add statistics
            mean_val = data.mean()
            median_val = data.median()
            axes[idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved feature distribution plots to {output_dir / 'feature_distributions.png'}")
    plt.close()


def generate_report(matches_df, features_df, elo_df, issues_list):
    """Generate a comprehensive data quality report."""
    report_path = PROCESSED_DATA_DIR / "data_quality_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("DATA QUALITY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Data summary
        f.write("DATA SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total matches: {len(matches_df)}\n")
        f.write(f"Total features rows: {len(features_df)}\n")
        f.write(f"Total teams: {len(elo_df)}\n")
        f.write(f"Date range: {matches_df['date'].min()} to {matches_df['date'].max()}\n")
        f.write(f"Leagues: {matches_df['league_code'].nunique()}\n")
        f.write(f"Seasons: {matches_df['season'].nunique()}\n\n")
        
        # Feature summary
        f.write("FEATURE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total features: {len(features_df.columns)}\n")
        f.write(f"Numerical features: {len(features_df.select_dtypes(include=[np.number]).columns)}\n")
        f.write(f"Categorical features: {len(features_df.select_dtypes(include=['object']).columns)}\n\n")
        
        # Issues found
        f.write("ISSUES FOUND\n")
        f.write("-"*40 + "\n")
        if issues_list:
            for issue in issues_list:
                f.write(f"• {issue}\n")
        else:
            f.write("✓ No major issues found!\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
    
    print(f"\nGenerated comprehensive report: {report_path}")


def main():
    """Run all data quality checks."""
    logger.info("Starting data quality checks")
    
    # Load data
    matches_df, features_df, elo_df = load_data()
    
    print(f"\nData loaded successfully:")
    print(f"  Matches: {len(matches_df)} rows")
    print(f"  Features: {len(features_df)} rows") 
    print(f"  Elo ratings: {len(elo_df)} teams")
    
    all_issues = []
    
    # 1. Missing value analysis
    missing_matches = check_missing_values(matches_df, "Matches")
    missing_features = check_missing_values(features_df, "Features")
    
    # 2. Data consistency checks
    consistency_issues = check_data_consistency(matches_df, features_df)
    all_issues.extend(consistency_issues)
    
    # 3. Feature sanity checks
    sanity_issues = check_feature_sanity(features_df)
    all_issues.extend(sanity_issues)
    
    # 4. Outlier detection
    outlier_df = check_outliers(features_df)
    
    # 5. Correlation analysis
    corr_matrix = check_correlations(features_df)
    
    # 6. Plot distributions
    plot_feature_distributions(features_df, PROCESSED_DATA_DIR)
    
    # 7. Generate report
    generate_report(matches_df, features_df, elo_df, all_issues)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    if all_issues:
        print(f"\n⚠️  Found {len(all_issues)} potential issues")
        print("\nMost critical issues:")
        for issue in all_issues[:5]:
            print(f"  • {issue}")
    else:
        print("\n✅ All data quality checks passed!")
    
    print(f"\nData quality checks complete!")
    logger.info("Data quality checks completed")


if __name__ == "__main__":
    main()