#!/usr/bin/env python3
"""
Feature Quality & Data Integrity Validation
============================================

Comprehensive validation of engineered features for:
- Data completeness and missing values
- Feature distributions and outliers
- Time-based ordering and leakage
- Rolling window correctness
- Duplicate detection
- Feature correlations
- Statistical sanity checks

Usage:
    python validate_features.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data/processed")
FEATURES_FILE = DATA_DIR / "sportmonks_features.csv"
OUTPUT_DIR = Path("data/validation")
OUTPUT_DIR.mkdir(exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class FeatureValidator:
    """Comprehensive feature validation."""

    def __init__(self, features_file: Path):
        """Load features for validation."""
        print("="*80)
        print("FEATURE QUALITY & DATA INTEGRITY VALIDATION")
        print("="*80)
        print(f"\nLoading features from: {features_file}")

        self.df = pd.read_csv(features_file)
        self.df['date'] = pd.to_datetime(self.df['date'])

        print(f"  Loaded {len(self.df)} matches")
        print(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"  Total columns: {len(self.df.columns)}")

        self.issues = []
        self.warnings = []

    def check_basic_integrity(self):
        """Check basic data integrity."""
        print("\n" + "="*80)
        print("1. BASIC DATA INTEGRITY")
        print("="*80)

        # Check for duplicates
        dups = self.df.duplicated(subset=['fixture_id']).sum()
        if dups > 0:
            self.issues.append(f"Found {dups} duplicate fixture_ids")
            print(f"  ✗ DUPLICATE FIXTURES: {dups} duplicates found")
        else:
            print(f"  ✓ No duplicate fixtures")

        # Check date ordering
        is_sorted = self.df['date'].is_monotonic_increasing
        if not is_sorted:
            self.issues.append("Data is not sorted by date")
            print(f"  ✗ DATE ORDER: Data not sorted chronologically")
        else:
            print(f"  ✓ Data sorted chronologically")

        # Check for future dates
        future_dates = (self.df['date'] > pd.Timestamp.now()).sum()
        if future_dates > 0:
            self.warnings.append(f"{future_dates} matches have future dates")
            print(f"  ⚠ FUTURE DATES: {future_dates} matches (expected for upcoming fixtures)")
        else:
            print(f"  ✓ No future dates")

        # Check required columns
        required = ['fixture_id', 'date', 'home_team_name', 'away_team_name', 'home_goals', 'away_goals']
        missing_cols = [col for col in required if col not in self.df.columns]
        if missing_cols:
            self.issues.append(f"Missing required columns: {missing_cols}")
            print(f"  ✗ MISSING COLUMNS: {missing_cols}")
        else:
            print(f"  ✓ All required columns present")

        # Check for matches with results
        has_result = self.df['home_goals'].notna() & self.df['away_goals'].notna()
        print(f"  ✓ Matches with results: {has_result.sum()} ({has_result.mean()*100:.1f}%)")

    def check_missing_values(self):
        """Analyze missing value patterns."""
        print("\n" + "="*80)
        print("2. MISSING VALUE ANALYSIS")
        print("="*80)

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)

        # Show top missing columns
        top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False).head(20)

        if len(top_missing) == 0:
            print("  ✓ No missing values found!")
        else:
            print(f"\n  Top 20 columns with missing values:")
            print("  " + "-"*60)
            for col, pct in top_missing.items():
                count = missing[col]
                status = "⚠" if pct > 50 else "○"
                print(f"  {status} {col:40s} {count:6d} ({pct:5.1f}%)")

        # Expected missing values for early-season matches
        rolling_features = [c for c in self.df.columns if any(w in c for w in ['_3', '_5', '_10'])]
        if rolling_features:
            early_missing = self.df[rolling_features].head(10).isnull().mean().mean() * 100
            print(f"\n  ✓ Early-season missing (expected): {early_missing:.1f}%")
            print(f"    (First 3-10 games lack rolling statistics - this is normal)")

        # Critical features that should never be missing
        critical_features = ['elo_diff', 'home_elo', 'away_elo', 'points_diff', 'position_diff']
        critical_missing = missing[[c for c in critical_features if c in self.df.columns]]

        if critical_missing.sum() > 0:
            self.issues.append(f"Critical features have missing values: {critical_missing[critical_missing > 0].to_dict()}")
            print(f"\n  ✗ CRITICAL MISSING: {critical_missing[critical_missing > 0].to_dict()}")
        else:
            print(f"\n  ✓ No missing values in critical features")

        # Save missing value report
        missing_report = pd.DataFrame({
            'column': missing.index,
            'missing_count': missing.values,
            'missing_pct': missing_pct.values
        }).sort_values('missing_pct', ascending=False)

        missing_report.to_csv(OUTPUT_DIR / "missing_values_report.csv", index=False)
        print(f"\n  Report saved: {OUTPUT_DIR / 'missing_values_report.csv'}")

    def check_feature_distributions(self):
        """Check feature distributions for outliers and anomalies."""
        print("\n" + "="*80)
        print("3. FEATURE DISTRIBUTION ANALYSIS")
        print("="*80)

        # Get numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['fixture_id', 'season_id', 'league_id']]

        print(f"\n  Analyzing {len(feature_cols)} numeric features...")

        # Check for infinite values
        inf_counts = {}
        for col in feature_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count

        if inf_counts:
            self.issues.append(f"Infinite values found: {inf_counts}")
            print(f"\n  ✗ INFINITE VALUES:")
            for col, count in list(inf_counts.items())[:10]:
                print(f"    - {col}: {count} infinite values")
        else:
            print(f"  ✓ No infinite values")

        # Check for extreme outliers (> 5 std from mean)
        outlier_summary = []
        for col in feature_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0 and data.std() > 0:
                    z_scores = np.abs((data - data.mean()) / data.std())
                    extreme_outliers = (z_scores > 5).sum()
                    if extreme_outliers > 0:
                        outlier_summary.append({
                            'feature': col,
                            'extreme_outliers': extreme_outliers,
                            'pct': extreme_outliers / len(data) * 100,
                            'min': data.min(),
                            'max': data.max(),
                            'mean': data.mean(),
                            'std': data.std()
                        })

        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary).sort_values('extreme_outliers', ascending=False).head(10)
            print(f"\n  Top 10 features with extreme outliers (>5 std):")
            print("  " + "-"*60)
            for _, row in outlier_df.iterrows():
                print(f"  ○ {row['feature']:40s} {int(row['extreme_outliers']):4d} ({row['pct']:.2f}%)")

            outlier_df.to_csv(OUTPUT_DIR / "outliers_report.csv", index=False)
            print(f"\n  Report saved: {OUTPUT_DIR / 'outliers_report.csv'}")
        else:
            print(f"  ✓ No extreme outliers detected")

        # Distribution plots for key features
        key_features = ['elo_diff', 'points_diff', 'position_diff', 'home_xg', 'away_xg']
        key_features = [f for f in key_features if f in self.df.columns]

        if key_features:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()

            for i, col in enumerate(key_features[:6]):
                data = self.df[col].dropna()
                axes[i].hist(data, bins=50, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'{col}\nMean: {data.mean():.2f}, Std: {data.std():.2f}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].axvline(data.mean(), color='red', linestyle='--', label='Mean')
                axes[i].legend()

            # Hide unused subplots
            for i in range(len(key_features), 6):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150, bbox_inches='tight')
            print(f"\n  Plots saved: {OUTPUT_DIR / 'feature_distributions.png'}")
            plt.close()

    def check_time_leakage(self):
        """Check for data leakage from future information."""
        print("\n" + "="*80)
        print("4. TIME-BASED LEAKAGE DETECTION")
        print("="*80)

        leakage_found = False

        # Check if rolling features use only past data
        # Sample random match and verify rolling stats
        completed = self.df[self.df['home_goals'].notna()].copy()

        if len(completed) > 100:
            sample_idx = completed.sample(min(10, len(completed)), random_state=42).index

            print(f"\n  Validating rolling window calculations on {len(sample_idx)} sample matches...")

            for idx in sample_idx:
                match_date = self.df.loc[idx, 'date']
                home_team = self.df.loc[idx, 'home_team_name']
                away_team = self.df.loc[idx, 'away_team_name']

                # Get past matches for home team
                home_past = self.df[
                    (self.df['date'] < match_date) &
                    ((self.df['home_team_name'] == home_team) | (self.df['away_team_name'] == home_team)) &
                    (self.df['home_goals'].notna())
                ].sort_values('date', ascending=False)

                # Check if we have enough history
                if 'home_points_5' in self.df.columns:
                    feature_value = self.df.loc[idx, 'home_points_5']
                    if pd.notna(feature_value) and len(home_past) < 5:
                        self.warnings.append(f"Match {idx}: home_points_5 exists but <5 past matches")
                        leakage_found = True

        if not leakage_found:
            print(f"  ✓ No obvious time leakage detected")
        else:
            print(f"  ⚠ Potential time leakage issues found (see warnings)")

        # Check feature creation date logic
        print(f"\n  Checking feature availability by match number...")

        # Features should not exist for first N matches
        self.df['match_number'] = self.df.groupby('home_team_name').cumcount() + 1

        early_matches = self.df[self.df['match_number'] <= 3]
        if 'home_points_5' in self.df.columns:
            early_with_feature = early_matches['home_points_5'].notna().sum()
            if early_with_feature > 0:
                self.warnings.append(f"{early_with_feature} early matches have 5-game rolling features")
                print(f"  ⚠ {early_with_feature} matches in first 3 have 5-game features (should be NaN)")
            else:
                print(f"  ✓ Early matches correctly have NaN for rolling features")

    def check_rolling_window_correctness(self):
        """Verify rolling window features are calculated correctly."""
        print("\n" + "="*80)
        print("5. ROLLING WINDOW VALIDATION")
        print("="*80)

        # Sample a few matches and manually verify rolling calculations
        completed = self.df[self.df['home_goals'].notna()].copy()

        if len(completed) > 50:
            # Pick a match with complete history
            sample = completed.iloc[50:60]

            print(f"\n  Manual verification of rolling windows (sample of {len(sample)} matches)...")

            errors = []
            for idx in sample.index:
                match_date = self.df.loc[idx, 'date']
                home_team = self.df.loc[idx, 'home_team_name']

                # Get past 5 home matches
                home_past = self.df[
                    (self.df['date'] < match_date) &
                    (self.df['home_team_name'] == home_team) &
                    (self.df['home_goals'].notna())
                ].sort_values('date', ascending=False).head(5)

                if len(home_past) >= 5 and 'home_points_5' in self.df.columns:
                    # Calculate expected points
                    expected_points = 0
                    for _, past_match in home_past.iterrows():
                        if past_match['home_goals'] > past_match['away_goals']:
                            expected_points += 3
                        elif past_match['home_goals'] == past_match['away_goals']:
                            expected_points += 1

                    expected_avg = expected_points / 5
                    actual_avg = self.df.loc[idx, 'home_points_5']

                    if pd.notna(actual_avg) and abs(expected_avg - actual_avg) > 0.01:
                        errors.append({
                            'fixture_id': self.df.loc[idx, 'fixture_id'],
                            'expected': expected_avg,
                            'actual': actual_avg,
                            'diff': abs(expected_avg - actual_avg)
                        })

            if errors:
                self.issues.append(f"Rolling window calculation errors: {len(errors)} discrepancies")
                print(f"\n  ✗ CALCULATION ERRORS: {len(errors)} discrepancies found")
                for err in errors[:5]:
                    print(f"    Fixture {err['fixture_id']}: expected {err['expected']:.2f}, got {err['actual']:.2f}")
            else:
                print(f"  ✓ Rolling window calculations verified correct")
        else:
            print(f"  ⚠ Not enough completed matches for validation")

    def check_correlations(self):
        """Check feature correlations."""
        print("\n" + "="*80)
        print("6. FEATURE CORRELATION ANALYSIS")
        print("="*80)

        # Get numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['fixture_id', 'season_id', 'league_id', 'match_number']]

        # Calculate correlations with target
        if 'result' in self.df.columns:
            target_corr = self.df[feature_cols + ['result']].corr()['result'].drop('result').abs().sort_values(ascending=False)

            print(f"\n  Top 10 features correlated with match result:")
            print("  " + "-"*60)
            for feat, corr in target_corr.head(10).items():
                print(f"  ○ {feat:40s} {corr:.4f}")

            # Save correlation report
            target_corr.to_csv(OUTPUT_DIR / "target_correlations.csv", header=['correlation'])

        # Check for highly correlated feature pairs (multicollinearity)
        print(f"\n  Checking for multicollinearity...")

        # Sample features to avoid memory issues
        sample_features = feature_cols[:100] if len(feature_cols) > 100 else feature_cols
        corr_matrix = self.df[sample_features].corr().abs()

        # Find highly correlated pairs (>0.95)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        if high_corr_pairs:
            print(f"  ⚠ Found {len(high_corr_pairs)} highly correlated pairs (>0.95):")
            for pair in high_corr_pairs[:10]:
                print(f"    - {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")

            pd.DataFrame(high_corr_pairs).to_csv(OUTPUT_DIR / "high_correlations.csv", index=False)
        else:
            print(f"  ✓ No extreme multicollinearity detected")

        # Create correlation heatmap for top features
        if 'result' in self.df.columns:
            top_features = target_corr.head(20).index.tolist()

            plt.figure(figsize=(14, 12))
            sns.heatmap(
                self.df[top_features].corr(),
                annot=False,
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True
            )
            plt.title('Correlation Heatmap - Top 20 Features by Target Correlation')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
            print(f"\n  Heatmap saved: {OUTPUT_DIR / 'correlation_heatmap.png'}")
            plt.close()

    def check_feature_importance_readiness(self):
        """Check if features are ready for model training."""
        print("\n" + "="*80)
        print("7. MODEL READINESS CHECK")
        print("="*80)

        # Check train/val/test split viability
        total_matches = len(self.df)
        completed = self.df['home_goals'].notna().sum()

        print(f"\n  Total matches: {total_matches}")
        print(f"  Completed matches: {completed} ({completed/total_matches*100:.1f}%)")

        # Recommended split
        test_size = int(completed * 0.15)
        val_size = int(completed * 0.15)
        train_size = completed - test_size - val_size

        print(f"\n  Recommended split (15%/15%/70%):")
        print(f"    Train: {train_size} matches")
        print(f"    Val:   {val_size} matches")
        print(f"    Test:  {test_size} matches")

        if train_size < 500:
            self.warnings.append(f"Training set may be too small ({train_size} < 500)")
            print(f"  ⚠ Training set may be small for robust model")
        else:
            print(f"  ✓ Sufficient data for train/val/test split")

        # Check feature variance
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        zero_var_features = []

        for col in numeric_cols:
            if self.df[col].nunique() == 1:
                zero_var_features.append(col)

        if zero_var_features:
            self.warnings.append(f"Zero-variance features: {zero_var_features}")
            print(f"\n  ⚠ Zero-variance features (should remove):")
            for feat in zero_var_features[:10]:
                print(f"    - {feat}")
        else:
            print(f"  ✓ No zero-variance features")

        # Check feature scale differences
        feature_cols = [c for c in numeric_cols if c not in ['fixture_id', 'season_id', 'league_id']]
        feature_ranges = self.df[feature_cols].agg(['min', 'max']).T
        feature_ranges['range'] = feature_ranges['max'] - feature_ranges['min']

        large_scale_diff = (feature_ranges['range'].max() / (feature_ranges['range'].min() + 1e-10)) > 1000

        if large_scale_diff:
            print(f"\n  ⚠ Large scale differences detected (consider normalization)")
            print(f"    Max range: {feature_ranges['range'].max():.2f}")
            print(f"    Min range: {feature_ranges['range'].min():.2f}")
        else:
            print(f"  ✓ Feature scales are reasonable")

    def generate_summary_report(self):
        """Generate final validation summary."""
        print("\n" + "="*80)
        print("8. VALIDATION SUMMARY")
        print("="*80)

        total_checks = 7

        print(f"\n  Validation completed: {total_checks} check categories")
        print(f"  Critical issues found: {len(self.issues)}")
        print(f"  Warnings: {len(self.warnings)}")

        # Write summary to file
        with open(OUTPUT_DIR / "validation_summary.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEATURE QUALITY & DATA INTEGRITY VALIDATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {FEATURES_FILE}\n")
            f.write(f"Validation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total matches: {len(self.df)}\n")
            f.write(f"Total features: {len(self.df.columns)}\n\n")

            f.write("CRITICAL ISSUES:\n")
            f.write("-" * 80 + "\n")
            if self.issues:
                for i, issue in enumerate(self.issues, 1):
                    f.write(f"{i}. {issue}\n")
            else:
                f.write("None - Data integrity verified!\n")

            f.write("\nWARNINGS:\n")
            f.write("-" * 80 + "\n")
            if self.warnings:
                for i, warning in enumerate(self.warnings, 1):
                    f.write(f"{i}. {warning}\n")
            else:
                f.write("None\n")

            f.write("\nGENERATED REPORTS:\n")
            f.write("-" * 80 + "\n")
            f.write("- missing_values_report.csv\n")
            f.write("- outliers_report.csv\n")
            f.write("- target_correlations.csv\n")
            f.write("- high_correlations.csv\n")
            f.write("- feature_distributions.png\n")
            f.write("- correlation_heatmap.png\n")

        print(f"\n  Summary saved: {OUTPUT_DIR / 'validation_summary.txt'}")

        # Final verdict
        print("\n" + "="*80)
        if len(self.issues) == 0:
            print("✓ VALIDATION PASSED - Features are ready for model training!")
        else:
            print("✗ VALIDATION FAILED - Please review and fix issues before training")
        print("="*80)

        print(f"\nAll reports saved to: {OUTPUT_DIR}/")

    def run_all_checks(self):
        """Run all validation checks."""
        self.check_basic_integrity()
        self.check_missing_values()
        self.check_feature_distributions()
        self.check_time_leakage()
        self.check_rolling_window_correctness()
        self.check_correlations()
        self.check_feature_importance_readiness()
        self.generate_summary_report()


def main():
    """Run feature validation."""
    if not FEATURES_FILE.exists():
        print(f"Error: Features file not found at {FEATURES_FILE}")
        print("Please run feature engineering first: python 02_sportmonks_feature_engineering.py")
        return

    validator = FeatureValidator(FEATURES_FILE)
    validator.run_all_checks()


if __name__ == "__main__":
    main()
