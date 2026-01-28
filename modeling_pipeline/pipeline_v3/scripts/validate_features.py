#!/usr/bin/env python3
"""
Comprehensive Feature Validation Script

Performs sanity checks on all 150 features in the training data:
- Missing value analysis
- Range validation
- Distribution analysis
- Logical consistency checks
- Correlation analysis
- Outlier detection

Usage:
    python scripts/validate_features.py --input data/csv/training_data_complete.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def setup_logging(log_file: str = 'feature_validation.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


class FeatureValidator:
    """Comprehensive feature validation."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize validator with training data."""
        self.df = df
        self.logger = logging.getLogger(__name__)
        
        # Separate features from metadata
        self.metadata_cols = [
            'fixture_id', 'match_date', 'home_team_id', 'away_team_id',
            'league_id', 'season_id', 'home_score', 'away_score', 'result'
        ]
        self.feature_cols = [col for col in df.columns if col not in self.metadata_cols]
        
        self.logger.info(f"Loaded {len(df)} samples with {len(self.feature_cols)} features")
        
        # Define expected ranges for different feature types
        self.expected_ranges = self._define_expected_ranges()
        
        # Results storage
        self.validation_results = {
            'missing_values': {},
            'range_violations': {},
            'distribution_issues': {},
            'logical_inconsistencies': [],
            'outliers': {},
            'summary': {}
        }
    
    def _define_expected_ranges(self) -> Dict:
        """Define expected ranges for feature validation."""
        return {
            # Elo features (actual Elo values)
            'home_elo': (1000, 2000),
            'away_elo': (1000, 2000),
            'elo_diff': (-500, 500),
            'elo_diff_with_ha': (-500, 500),
            'elo_change': (-200, 200),
            'elo_vs_league_avg': (-400, 400),
            
            # Position features
            'league_position': (1, 25),
            'position_diff': (-24, 24),
            'points': (0, 120),
            'points_diff': (-120, 120),
            'points_per_game': (0, 3),
            'points_at_home': (0, 400),  # Cumulative over all history
            'points_away': (0, 400),
            
            # Binary features (0 or 1 only)
            'in_top_6': (0, 1),
            'in_bottom_3': (0, 1),
            'is_derby': (0, 1),
            'underdog': (0, 1),
            'pressure': (0, 1),
            
            # Form features
            'points_last': (0, 30),
            'wins': (0, 10),
            'draws': (0, 10),
            'goals_scored': (0, 50),
            'goals_conceded': (0, 50),
            'goal_diff': (-30, 30),
            
            # xG features (can be higher than 5 for cumulative/per match)
            'derived_xg_per_match': (0, 20),  # Some teams have very high xG
            'derived_xga_per_match': (0, 20),
            'derived_xgd': (-10, 10),
            'goals_vs_xg': (-20, 20),  # Cumulative difference
            'ga_vs_xga': (-20, 20),
            'xg_per_shot': (0, 0.5),
            'big_chances': (0, 15),
            'big_chance_conversion': (0, 2),  # Can be > 1 if few chances
            'xg_trend': (-3, 3),
            
            # Shot features
            'shots_per_match': (0, 40),
            'shots_on_target': (0, 30),
            'inside_box_shot': (0, 30),  # Count, not percentage
            'outside_box_shot': (0, 40),  # Count, not percentage
            'shot_accuracy': (0, 5),  # Ratio, can be > 1
            'shots_per_goal': (0, 100),
            'shots_conceded': (0, 40),
            
            # Defensive features
            'ppda': (0, 100),  # Can be high for passive teams
            'tackles_per_90': (0, 40),
            'interceptions_per_90': (0, 40),
            'defensive_actions': (0, 80),
            'possession_pct': (0, 100),
            'passes_per_match': (0, 1000),
            
            # Attack features
            'attacks_per_match': (0, 250),
            'dangerous_attacks_per_match': (0, 150),
            'dangerous_attack_ratio': (0, 150),  # Ratio, can be > 1
            'shots_per_attack': (0, 50),
            
            # Momentum features
            'points_trend': (-1, 1),  # Linear regression slope
            'weighted_form': (0, 3),
            'win_streak': (0, 30),
            'unbeaten_streak': (0, 40),
            'clean_sheet_streak': (0, 20),
            'goals_trend': (-1, 1),
            
            # Fixture adjusted features
            'avg_opponent_elo': (1200, 1800),
            'points_vs_strong': (0, 3),
            'points_vs_weak': (0, 3),
            'goals_vs_strong': (0, 10),
            'goals_vs_weak': (0, 10),
            
            # Player quality (proxies)
            'lineup_quality_proxy': (-10, 10),
            'squad_depth_proxy': (0, 2),
            'consistency_rating': (0, 2),
            'recent_performance': (0, 3),
            'goal_threat': (0, 5),
            
            # Context features
            'days_since_last_match': (0, 60),
            'rest_advantage': (-30, 30),
            
            # H2H features
            'h2h_wins': (0, 10),
            'h2h_draws': (0, 10),
            'h2h_goals_avg': (0, 5),
            'h2h_win_pct': (0, 1),
            'h2h_btts_pct': (0, 1),
            'h2h_over_2_5_pct': (0, 1),
            
            # Home advantage
            'home_win_pct': (0, 1),
            'away_win_pct': (0, 1),
            'home_advantage_strength': (-2, 2),
        }
    
    def validate_all(self) -> Dict:
        """Run all validation checks."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING COMPREHENSIVE FEATURE VALIDATION")
        self.logger.info("=" * 80 + "\n")
        
        # 1. Missing values
        self.logger.info("1. Checking missing values...")
        self._check_missing_values()
        
        # 2. Range validation
        self.logger.info("\n2. Validating feature ranges...")
        self._validate_ranges()
        
        # 3. Distribution analysis
        self.logger.info("\n3. Analyzing distributions...")
        self._analyze_distributions()
        
        # 4. Logical consistency
        self.logger.info("\n4. Checking logical consistency...")
        self._check_logical_consistency()
        
        # 5. Outlier detection
        self.logger.info("\n5. Detecting outliers...")
        self._detect_outliers()
        
        # 6. Correlation analysis
        self.logger.info("\n6. Analyzing correlations...")
        self._analyze_correlations()
        
        # 7. Generate summary
        self.logger.info("\n7. Generating summary...")
        self._generate_summary()
        
        return self.validation_results
    
    def _check_missing_values(self):
        """Check for missing values in each feature."""
        for col in self.feature_cols:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            if missing_count > 0:
                self.validation_results['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
                
                if missing_pct > 50:
                    self.logger.warning(f"  ⚠️  {col}: {missing_pct:.1f}% missing ({missing_count} values)")
                elif missing_pct > 10:
                    self.logger.info(f"  ⚡ {col}: {missing_pct:.1f}% missing ({missing_count} values)")
        
        total_missing = sum(r['count'] for r in self.validation_results['missing_values'].values())
        self.logger.info(f"\n  Total missing values: {total_missing}")
        self.logger.info(f"  Features with missing values: {len(self.validation_results['missing_values'])}")
    
    def _validate_ranges(self):
        """Validate that features are within expected ranges."""
        violations = 0
        
        for col in self.feature_cols:
            # Skip if all NaN
            if self.df[col].isna().all():
                continue
            
            # Get actual range
            actual_min = self.df[col].min()
            actual_max = self.df[col].max()
            
            # Find expected range
            expected_range = None
            for pattern, range_vals in self.expected_ranges.items():
                if pattern in col.lower():
                    expected_range = range_vals
                    break
            
            if expected_range:
                expected_min, expected_max = expected_range
                
                # Check violations
                if actual_min < expected_min or actual_max > expected_max:
                    violations += 1
                    self.validation_results['range_violations'][col] = {
                        'expected': expected_range,
                        'actual': (float(actual_min), float(actual_max)),
                        'violation_type': []
                    }
                    
                    if actual_min < expected_min:
                        self.validation_results['range_violations'][col]['violation_type'].append('min')
                        self.logger.warning(f"  ⚠️  {col}: min={actual_min:.2f} < expected {expected_min}")
                    
                    if actual_max > expected_max:
                        self.validation_results['range_violations'][col]['violation_type'].append('max')
                        self.logger.warning(f"  ⚠️  {col}: max={actual_max:.2f} > expected {expected_max}")
        
        self.logger.info(f"\n  Range violations: {violations}")
    
    def _analyze_distributions(self):
        """Analyze feature distributions for anomalies."""
        for col in self.feature_cols:
            # Skip if all NaN
            if self.df[col].isna().all():
                continue
            
            values = self.df[col].dropna()
            
            # Calculate statistics
            mean = values.mean()
            std = values.std()
            skew = values.skew()
            kurtosis = values.kurtosis()
            
            # Check for issues
            issues = []
            
            # Check for zero variance
            if std < 0.001:
                issues.append('zero_variance')
            
            # Check for extreme skewness
            if abs(skew) > 5:
                issues.append('extreme_skew')
            
            # Check for extreme kurtosis
            if abs(kurtosis) > 10:
                issues.append('extreme_kurtosis')
            
            # Check for constant values
            if values.nunique() == 1:
                issues.append('constant')
            
            if issues:
                self.validation_results['distribution_issues'][col] = {
                    'mean': float(mean),
                    'std': float(std),
                    'skew': float(skew),
                    'kurtosis': float(kurtosis),
                    'issues': issues
                }
                
                self.logger.warning(f"  ⚠️  {col}: {', '.join(issues)}")
        
        self.logger.info(f"\n  Features with distribution issues: {len(self.validation_results['distribution_issues'])}")
    
    def _check_logical_consistency(self):
        """Check for logical inconsistencies between features."""
        inconsistencies = []
        
        # 1. Elo diff should equal home_elo - away_elo
        if 'elo_diff' in self.df.columns and 'home_elo' in self.df.columns:
            calculated_diff = self.df['home_elo'] - self.df['away_elo']
            diff_error = (self.df['elo_diff'] - calculated_diff).abs().max()
            
            if diff_error > 1:
                inconsistencies.append({
                    'check': 'elo_diff = home_elo - away_elo',
                    'max_error': float(diff_error)
                })
                self.logger.warning(f"  ⚠️  Elo diff inconsistency: max error = {diff_error:.2f}")
        
        # 2. Points should be consistent with wins/draws
        if all(col in self.df.columns for col in ['home_points_last_5', 'home_wins_last_5', 'home_draws_last_5']):
            expected_points = self.df['home_wins_last_5'] * 3 + self.df['home_draws_last_5']
            points_error = (self.df['home_points_last_5'] - expected_points).abs().max()
            
            if points_error > 1:
                inconsistencies.append({
                    'check': 'points_last_5 = wins*3 + draws',
                    'max_error': float(points_error)
                })
                self.logger.warning(f"  ⚠️  Points inconsistency: max error = {points_error:.2f}")
        
        # 3. Position diff should equal home_position - away_position
        if 'position_diff' in self.df.columns and 'home_league_position' in self.df.columns:
            calculated_pos_diff = self.df['home_league_position'] - self.df['away_league_position']
            pos_error = (self.df['position_diff'] - calculated_pos_diff).abs().max()
            
            if pos_error > 1:
                inconsistencies.append({
                    'check': 'position_diff = home_position - away_position',
                    'max_error': float(pos_error)
                })
                self.logger.warning(f"  ⚠️  Position diff inconsistency: max error = {pos_error:.2f}")
        
        # 4. xGD should equal xG - xGA
        if all(col in self.df.columns for col in ['home_derived_xgd_5', 'home_derived_xg_per_match_5', 'home_derived_xga_per_match_5']):
            calculated_xgd = self.df['home_derived_xg_per_match_5'] - self.df['home_derived_xga_per_match_5']
            xgd_error = (self.df['home_derived_xgd_5'] - calculated_xgd).abs().max()
            
            if xgd_error > 0.1:
                inconsistencies.append({
                    'check': 'xgd = xg - xga',
                    'max_error': float(xgd_error)
                })
                self.logger.warning(f"  ⚠️  xGD inconsistency: max error = {xgd_error:.2f}")
        
        # 5. Binary features should only be 0 or 1
        binary_features = [col for col in self.feature_cols if any(
            pattern in col for pattern in ['in_top_6', 'in_bottom_3', 'is_derby', 'underdog', 'pressure']
        )]
        
        for col in binary_features:
            if col in self.df.columns:
                unique_vals = self.df[col].dropna().unique()
                if not all(val in [0, 0.0, 1, 1.0] for val in unique_vals):
                    inconsistencies.append({
                        'check': f'{col} should be binary (0 or 1)',
                        'unique_values': list(map(float, unique_vals))
                    })
                    self.logger.warning(f"  ⚠️  {col} has non-binary values: {unique_vals}")
        
        self.validation_results['logical_inconsistencies'] = inconsistencies
        self.logger.info(f"\n  Logical inconsistencies found: {len(inconsistencies)}")
    
    def _detect_outliers(self):
        """Detect outliers using IQR method."""
        outlier_threshold = 0.05  # Report if >5% outliers
        
        for col in self.feature_cols:
            # Skip if all NaN
            if self.df[col].isna().all():
                continue
            
            values = self.df[col].dropna()
            
            # Calculate IQR
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Count outliers
            outliers = ((values < lower_bound) | (values > upper_bound)).sum()
            outlier_pct = (outliers / len(values)) * 100
            
            if outlier_pct > outlier_threshold * 100:
                self.validation_results['outliers'][col] = {
                    'count': int(outliers),
                    'percentage': round(outlier_pct, 2),
                    'bounds': (float(lower_bound), float(upper_bound)),
                    'extreme_values': {
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
                }
                
                if outlier_pct > 10:
                    self.logger.warning(f"  ⚠️  {col}: {outlier_pct:.1f}% outliers ({outliers} values)")
        
        self.logger.info(f"\n  Features with significant outliers: {len(self.validation_results['outliers'])}")
    
    def _analyze_correlations(self):
        """Analyze feature correlations."""
        # Calculate correlation matrix
        corr_matrix = self.df[self.feature_cols].corr()
        
        # Find highly correlated pairs (>0.95)
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) > 0.95 and not pd.isna(corr_val):
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        self.validation_results['high_correlations'] = high_corr_pairs
        
        if high_corr_pairs:
            self.logger.info(f"\n  Highly correlated pairs (>0.95): {len(high_corr_pairs)}")
            for pair in high_corr_pairs[:5]:  # Show first 5
                self.logger.info(f"    {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    
    def _generate_summary(self):
        """Generate validation summary."""
        summary = {
            'total_samples': len(self.df),
            'total_features': len(self.feature_cols),
            'features_with_missing': len(self.validation_results['missing_values']),
            'features_with_range_violations': len(self.validation_results['range_violations']),
            'features_with_distribution_issues': len(self.validation_results['distribution_issues']),
            'logical_inconsistencies': len(self.validation_results['logical_inconsistencies']),
            'features_with_outliers': len(self.validation_results['outliers']),
            'highly_correlated_pairs': len(self.validation_results.get('high_correlations', [])),
        }
        
        # Calculate overall health score (0-100)
        max_issues = len(self.feature_cols)
        total_issues = (
            summary['features_with_missing'] * 0.5 +  # Missing values less critical
            summary['features_with_range_violations'] * 1.0 +
            summary['features_with_distribution_issues'] * 0.3 +
            summary['logical_inconsistencies'] * 2.0 +  # Most critical
            summary['features_with_outliers'] * 0.2
        )
        
        health_score = max(0, 100 - (total_issues / max_issues * 100))
        summary['health_score'] = round(health_score, 1)
        
        self.validation_results['summary'] = summary
        
        # Print summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total samples: {summary['total_samples']}")
        self.logger.info(f"Total features: {summary['total_features']}")
        self.logger.info(f"\nIssues found:")
        self.logger.info(f"  Features with missing values: {summary['features_with_missing']}")
        self.logger.info(f"  Features with range violations: {summary['features_with_range_violations']}")
        self.logger.info(f"  Features with distribution issues: {summary['features_with_distribution_issues']}")
        self.logger.info(f"  Logical inconsistencies: {summary['logical_inconsistencies']}")
        self.logger.info(f"  Features with outliers: {summary['features_with_outliers']}")
        self.logger.info(f"  Highly correlated pairs: {summary['highly_correlated_pairs']}")
        self.logger.info(f"\n🎯 Overall Health Score: {summary['health_score']}/100")
        self.logger.info("=" * 80 + "\n")
    
    def save_report(self, output_path: str):
        """Save detailed validation report."""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        self.logger.info(f"Detailed report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive feature validation for training data'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with training data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='feature_validation_report.json',
        help='Output JSON report file (default: feature_validation_report.json)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='feature_validation.log',
        help='Log file path (default: feature_validation.log)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("FEATURE VALIDATION SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output report: {args.output}")
    logger.info(f"Log file: {args.log_file}")
    logger.info("=" * 80 + "\n")
    
    try:
        # Load data
        logger.info("Loading training data...")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns\n")
        
        # Run validation
        validator = FeatureValidator(df)
        results = validator.validate_all()
        
        # Save report
        validator.save_report(args.output)
        
        # Exit code based on health score
        health_score = results['summary']['health_score']
        
        if health_score >= 90:
            logger.info("✅ Validation PASSED - Data quality is excellent!")
            return 0
        elif health_score >= 70:
            logger.warning("⚠️  Validation PASSED with warnings - Data quality is acceptable")
            return 0
        else:
            logger.error("❌ Validation FAILED - Data quality issues detected")
            return 1
        
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
