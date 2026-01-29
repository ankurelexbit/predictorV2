#!/usr/bin/env python3
"""
Feature Engineering Validation Script.

Validates the training_data.csv file for:
1. Feature completeness (all expected features present)
2. Data quality (nulls, infinities, outliers)
3. Data leakage detection (future data in features)
4. Feature distributions and correlations
5. Target variable balance

Usage:
    python scripts/validate_features_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validate engineered features for quality and sanity."""
    
    def __init__(self, training_data_path: str = 'data/csv/training_data_complete_v2.csv'):
        self.training_data_path = Path(training_data_path)
        self.df = None
        self.issues = []
        
    def load_data(self):
        """Load training data."""
        logger.info("=" * 80)
        logger.info("LOADING TRAINING DATA")
        logger.info("=" * 80)
        
        if not self.training_data_path.exists():
            logger.error(f"❌ Training data not found: {self.training_data_path}")
            logger.error("Run regenerate_training_data.py first")
            return False
        
        try:
            self.df = pd.read_csv(self.training_data_path)
            logger.info(f"✅ Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
            logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def check_feature_completeness(self):
        """Check if all expected features are present."""
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE COMPLETENESS CHECK")
        logger.info("=" * 80)
        
        # Expected feature categories and approximate counts
        expected_categories = {
            'target': ['result'],  # Target variable
            'metadata': ['fixture_id', 'date', 'home_team_id', 'away_team_id'],
            'elo': 5,  # Elo features
            'position': 12,  # League position features
            'form': 15,  # Form features (L3, L5, L10)
            'h2h': 10,  # Head-to-head features
            'xg': 8,  # Expected goals features
            'shots': 10,  # Shots features
            'defense': 8,  # Defense features
            'momentum': 12,  # Momentum features
            'player': 49,  # Player statistics (Phase 1)
            'events': 32,  # Match events (Phase 2)
            'formation': 12,  # Formation features (Phase 3)
            'injury': 16,  # Injury features (Phase 4)
            'odds': 6,  # Betting odds (Phase 5)
            'temporal': 8,  # Temporal features (Phase 6)
        }
        
        # Check required columns
        required_cols = ['fixture_id', 'result']
        missing_required = [col for col in required_cols if col not in self.df.columns]
        
        if missing_required:
            logger.error(f"❌ Missing required columns: {missing_required}")
            self.issues.append(f"Missing required columns: {missing_required}")
            return False
        
        logger.info(f"✅ All required columns present")
        
        # Count features by category (rough estimate)
        feature_cols = [col for col in self.df.columns 
                       if col not in ['fixture_id', 'date', 'home_team_id', 'away_team_id', 'result']]
        
        logger.info(f"Total feature columns: {len(feature_cols)}")
        
        # Expected range: 200-300 features after selection
        if len(feature_cols) < 150:
            logger.warning(f"⚠️ Low feature count: {len(feature_cols)} (expected 200-300)")
            self.issues.append(f"Low feature count: {len(feature_cols)}")
        elif len(feature_cols) > 350:
            logger.warning(f"⚠️ High feature count: {len(feature_cols)} (expected 200-300)")
            self.issues.append(f"High feature count: {len(feature_cols)}")
        else:
            logger.info(f"✅ Feature count in expected range: {len(feature_cols)}")
        
        return True
    
    def check_data_quality(self):
        """Check for data quality issues."""
        logger.info("\n" + "=" * 80)
        logger.info("DATA QUALITY CHECK")
        logger.info("=" * 80)
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['fixture_id', 'date', 'home_team_id', 'away_team_id', 'result']]
        
        issues_found = False
        
        # 1. Check for nulls
        logger.info("\n1. Checking for null values...")
        null_counts = self.df[feature_cols].isnull().sum()
        high_null_features = null_counts[null_counts > len(self.df) * 0.1]  # >10% null
        
        if len(high_null_features) > 0:
            logger.warning(f"⚠️ {len(high_null_features)} features with >10% nulls:")
            for feat, count in high_null_features.head(10).items():
                pct = (count / len(self.df)) * 100
                logger.warning(f"  - {feat}: {pct:.1f}% ({count:,} rows)")
            self.issues.append(f"{len(high_null_features)} features with >10% nulls")
            issues_found = True
        else:
            logger.info("✅ No features with excessive nulls")
        
        # 2. Check for infinities
        logger.info("\n2. Checking for infinite values...")
        inf_counts = np.isinf(self.df[feature_cols].select_dtypes(include=[np.number])).sum()
        inf_features = inf_counts[inf_counts > 0]
        
        if len(inf_features) > 0:
            logger.error(f"❌ {len(inf_features)} features with infinite values:")
            for feat, count in inf_features.head(10).items():
                logger.error(f"  - {feat}: {count:,} infinite values")
            self.issues.append(f"{len(inf_features)} features with infinite values")
            issues_found = True
        else:
            logger.info("✅ No infinite values found")
        
        # 3. Check for constant features
        logger.info("\n3. Checking for constant features...")
        numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns
        constant_features = []
        
        for col in numeric_cols:
            if self.df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            logger.warning(f"⚠️ {len(constant_features)} constant features (no variance):")
            for feat in constant_features[:10]:
                logger.warning(f"  - {feat}")
            self.issues.append(f"{len(constant_features)} constant features")
            issues_found = True
        else:
            logger.info("✅ No constant features found")
        
        # 4. Check for extreme outliers (>10 std from mean)
        logger.info("\n4. Checking for extreme outliers...")
        outlier_features = []
        
        for col in numeric_cols:
            if self.df[col].std() > 0:  # Skip zero-variance
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                extreme_count = (z_scores > 10).sum()
                
                if extreme_count > len(self.df) * 0.01:  # >1% extreme outliers
                    outlier_features.append((col, extreme_count))
        
        if outlier_features:
            logger.warning(f"⚠️ {len(outlier_features)} features with >1% extreme outliers:")
            for feat, count in sorted(outlier_features, key=lambda x: x[1], reverse=True)[:10]:
                pct = (count / len(self.df)) * 100
                logger.warning(f"  - {feat}: {pct:.1f}% ({count:,} rows)")
            self.issues.append(f"{len(outlier_features)} features with extreme outliers")
        else:
            logger.info("✅ No excessive outliers found")
        
        return not issues_found
    
    def check_target_distribution(self):
        """Check target variable distribution."""
        logger.info("\n" + "=" * 80)
        logger.info("TARGET VARIABLE CHECK")
        logger.info("=" * 80)
        
        if 'result' not in self.df.columns:
            logger.error("❌ Target variable 'result' not found")
            return False
        
        # Check for valid values
        valid_results = {'H', 'D', 'A'}
        actual_results = set(self.df['result'].unique())
        
        if not actual_results.issubset(valid_results):
            logger.error(f"❌ Invalid result values: {actual_results - valid_results}")
            self.issues.append(f"Invalid result values: {actual_results - valid_results}")
            return False
        
        # Check distribution
        result_counts = self.df['result'].value_counts()
        result_pcts = (result_counts / len(self.df)) * 100
        
        logger.info("Result distribution:")
        for result, count in result_counts.items():
            pct = result_pcts[result]
            logger.info(f"  {result}: {count:,} ({pct:.1f}%)")
        
        # Check for severe imbalance (any class <10% or >60%)
        if result_pcts.min() < 10:
            logger.warning(f"⚠️ Severe class imbalance: minimum class has {result_pcts.min():.1f}%")
            self.issues.append(f"Severe class imbalance: {result_pcts.min():.1f}%")
        elif result_pcts.max() > 60:
            logger.warning(f"⚠️ Severe class imbalance: maximum class has {result_pcts.max():.1f}%")
            self.issues.append(f"Severe class imbalance: {result_pcts.max():.1f}%")
        else:
            logger.info("✅ Target distribution is balanced")
        
        return True
    
    def check_data_leakage(self):
        """Check for potential data leakage."""
        logger.info("\n" + "=" * 80)
        logger.info("DATA LEAKAGE CHECK")
        logger.info("=" * 80)
        
        # Features that should NOT exist (would indicate leakage)
        leakage_keywords = [
            'final', 'actual', 'outcome', 'winner', 'loser',
            'full_time', 'ft_', 'match_result'
        ]
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['fixture_id', 'date', 'home_team_id', 'away_team_id', 'result']]
        
        suspicious_features = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in leakage_keywords):
                suspicious_features.append(col)
        
        if suspicious_features:
            logger.error(f"❌ Suspicious features (potential leakage):")
            for feat in suspicious_features:
                logger.error(f"  - {feat}")
            self.issues.append(f"{len(suspicious_features)} suspicious features")
            return False
        else:
            logger.info("✅ No obvious data leakage detected")
        
        # Check for perfect correlations with target
        logger.info("\nChecking for perfect correlations with target...")
        numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns
        
        # Encode target
        target_encoded = self.df['result'].map({'H': 1, 'D': 0, 'A': -1})
        
        perfect_corr_features = []
        for col in numeric_cols:
            if self.df[col].std() > 0:  # Skip zero-variance
                corr = abs(self.df[col].corr(target_encoded))
                if corr > 0.99:  # Nearly perfect correlation
                    perfect_corr_features.append((col, corr))
        
        if perfect_corr_features:
            logger.error(f"❌ Features with near-perfect correlation to target:")
            for feat, corr in perfect_corr_features:
                logger.error(f"  - {feat}: {corr:.4f}")
            self.issues.append(f"{len(perfect_corr_features)} features with perfect correlation")
            return False
        else:
            logger.info("✅ No perfect correlations found")
        
        return True
    
    def check_feature_correlations(self):
        """Check for highly correlated features."""
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE CORRELATION CHECK")
        logger.info("=" * 80)
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['fixture_id', 'date', 'home_team_id', 'away_team_id', 'result']]
        
        numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns
        
        logger.info(f"Computing correlations for {len(numeric_cols)} numeric features...")
        
        # Sample if too many features (for speed)
        if len(numeric_cols) > 300:
            logger.info(f"Sampling 300 features for correlation check...")
            numeric_cols = np.random.choice(numeric_cols, 300, replace=False)
        
        corr_matrix = self.df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs (>0.95)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            logger.warning(f"⚠️ {len(high_corr_pairs)} highly correlated feature pairs (>0.95):")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
                logger.warning(f"  - {feat1} <-> {feat2}: {corr:.4f}")
            logger.info("Consider removing redundant features during feature selection")
        else:
            logger.info("✅ No highly correlated feature pairs found")
        
        return True
    
    def generate_summary_stats(self):
        """Generate summary statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 80)
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['fixture_id', 'date', 'home_team_id', 'away_team_id', 'result']]
        
        numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns
        
        logger.info(f"\nDataset shape: {self.df.shape}")
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"Numeric features: {len(numeric_cols)}")
        logger.info(f"Categorical features: {len(feature_cols) - len(numeric_cols)}")
        
        # Memory usage
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Memory usage: {memory_mb:.1f} MB")
        
        # Feature value ranges
        logger.info("\nFeature value ranges (sample):")
        sample_features = np.random.choice(numeric_cols, min(5, len(numeric_cols)), replace=False)
        for feat in sample_features:
            logger.info(f"  {feat}: [{self.df[feat].min():.2f}, {self.df[feat].max():.2f}]")
        
        return True
    
    def run_full_validation(self):
        """Run complete validation suite."""
        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING VALIDATION")
        logger.info("=" * 80)
        
        # Load data
        if not self.load_data():
            return False
        
        # Run all checks
        results = {
            'completeness': self.check_feature_completeness(),
            'data_quality': self.check_data_quality(),
            'target_distribution': self.check_target_distribution(),
            'data_leakage': self.check_data_leakage(),
            'correlations': self.check_feature_correlations(),
            'summary': self.generate_summary_stats(),
        }
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        for check, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{check.upper()}: {status}")
        
        if self.issues:
            logger.warning(f"\n⚠️ Total issues found: {len(self.issues)}")
            for issue in self.issues:
                logger.warning(f"  - {issue}")
        
        all_passed = all(results.values())
        
        if all_passed and not self.issues:
            logger.info("\n🎉 ALL VALIDATIONS PASSED!")
            logger.info("Training data is ready for model training")
            return True
        elif all_passed:
            logger.warning("\n⚠️ VALIDATION PASSED WITH WARNINGS")
            logger.warning("Review warnings before proceeding to training")
            return True
        else:
            logger.error("\n❌ VALIDATION FAILED")
            logger.error("Fix issues before proceeding to training")
            return False


def main():
    """Main entry point."""
    validator = FeatureValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
