#!/usr/bin/env python3
"""
Comprehensive Data Quality Analysis for CSV Files

This script analyzes the quality of converted CSV data to ensure it's ready
for training dataset generation. It checks:
- Data completeness
- Missing values
- Join quality between tables
- Data integrity
- Temporal coverage
- Statistical distributions

Usage:
    python scripts/analyze_data_quality.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Comprehensive data quality analyzer for CSV files."""
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize analyzer and load data."""
        self.data_dir = Path(data_dir)
        self.fixtures = None
        self.statistics = None
        self.lineups = None
        self.sidelined = None
        
        logger.info("Loading CSV files...")
        self._load_data()
    
    def _load_data(self):
        """Load all CSV files."""
        try:
            self.fixtures = pd.read_csv(self.data_dir / 'fixtures.csv')
            logger.info(f"✅ Loaded fixtures: {len(self.fixtures):,} rows")
        except Exception as e:
            logger.error(f"❌ Failed to load fixtures: {e}")
        
        try:
            # Try with error handling for malformed lines
            try:
                self.statistics = pd.read_csv(self.data_dir / 'statistics.csv', on_bad_lines='skip')
            except TypeError:
                # Older pandas version
                self.statistics = pd.read_csv(self.data_dir / 'statistics.csv', error_bad_lines=False, warn_bad_lines=True)
            logger.info(f"✅ Loaded statistics: {len(self.statistics):,} rows")
        except Exception as e:
            logger.error(f"❌ Failed to load statistics: {e}")
        
        try:
            # Try with error handling for malformed lines
            try:
                self.lineups = pd.read_csv(self.data_dir / 'lineups.csv', on_bad_lines='skip')
            except TypeError:
                # Older pandas version
                self.lineups = pd.read_csv(self.data_dir / 'lineups.csv', error_bad_lines=False, warn_bad_lines=True)
            logger.info(f"✅ Loaded lineups: {len(self.lineups):,} rows")
        except Exception as e:
            logger.error(f"❌ Failed to load lineups: {e}")
        
        try:
            self.sidelined = pd.read_csv(self.data_dir / 'sidelined.csv')
            logger.info(f"✅ Loaded sidelined: {len(self.sidelined):,} rows")
        except Exception as e:
            logger.error(f"❌ Failed to load sidelined: {e}")
    
    def analyze_fixtures(self):
        """Analyze fixtures data quality."""
        print("\n" + "=" * 80)
        print("1. FIXTURES ANALYSIS")
        print("=" * 80)
        
        df = self.fixtures
        
        print(f"\n📊 Basic Statistics:")
        print(f"  Total fixtures: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Date range
        if 'starting_at' in df.columns:
            df['date'] = pd.to_datetime(df['starting_at'])
            print(f"\n📅 Temporal Coverage:")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Years covered: {df['date'].dt.year.nunique()}")
            
            # Fixtures per year
            print(f"\n  Fixtures per year:")
            yearly = df['date'].dt.year.value_counts().sort_index()
            for year, count in yearly.items():
                print(f"    {year}: {count:,}")
        
        # Missing values
        print(f"\n❓ Missing Values:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Count': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string())
        else:
            print("  ✅ No missing values!")
        
        # Result distribution
        if 'result' in df.columns:
            print(f"\n🎯 Result Distribution:")
            result_counts = df['result'].value_counts()
            total_with_result = result_counts.sum()
            for result, count in result_counts.items():
                pct = (count / total_with_result) * 100
                print(f"  {result}: {count:,} ({pct:.1f}%)")
            
            missing_results = df['result'].isnull().sum()
            if missing_results > 0:
                print(f"  Missing: {missing_results:,} ({(missing_results/len(df))*100:.1f}%)")
        
        # League coverage
        if 'league_id' in df.columns:
            print(f"\n🏆 League Coverage:")
            print(f"  Unique leagues: {df['league_id'].nunique()}")
            print(f"  Top 10 leagues by fixtures:")
            league_counts = df['league_id'].value_counts().head(10)
            for league_id, count in league_counts.items():
                print(f"    League {league_id}: {count:,}")
        
        # Team coverage
        if 'home_team_id' in df.columns and 'away_team_id' in df.columns:
            all_teams = pd.concat([df['home_team_id'], df['away_team_id']])
            print(f"\n⚽ Team Coverage:")
            print(f"  Unique teams: {all_teams.nunique():,}")
        
        # Data integrity checks
        print(f"\n✅ Data Integrity:")
        print(f"  Duplicate fixture_ids: {df.duplicated(subset=['fixture_id']).sum()}")
        
        if 'home_score' in df.columns and 'away_score' in df.columns:
            negative_scores = ((df['home_score'] < 0) | (df['away_score'] < 0)).sum()
            print(f"  Negative scores: {negative_scores}")
            
            # Check result consistency
            df_with_result = df[df['result'].notna() & df['home_score'].notna() & df['away_score'].notna()].copy()
            if len(df_with_result) > 0:
                df_with_result['expected_result'] = df_with_result.apply(
                    lambda row: 'H' if row['home_score'] > row['away_score'] 
                    else ('A' if row['home_score'] < row['away_score'] else 'D'),
                    axis=1
                )
                mismatches = (df_with_result['result'] != df_with_result['expected_result']).sum()
                print(f"  Result/Score mismatches: {mismatches}")
    
    def analyze_statistics(self):
        """Analyze statistics data quality."""
        print("\n" + "=" * 80)
        print("2. STATISTICS ANALYSIS")
        print("=" * 80)
        
        if self.statistics is None:
            print("  ⚠️ Statistics data not loaded - skipping analysis")
            return
        
        df = self.statistics
        
        print(f"\n📊 Basic Statistics:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Coverage
        if 'fixture_id' in df.columns:
            print(f"\n📈 Coverage:")
            print(f"  Unique fixtures: {df['fixture_id'].nunique():,}")
            
            # Rows per fixture (should be 2: home + away)
            rows_per_fixture = df.groupby('fixture_id').size()
            print(f"\n  Rows per fixture distribution:")
            for count, freq in rows_per_fixture.value_counts().sort_index().items():
                print(f"    {count} rows: {freq:,} fixtures")
        
        # Critical statistics availability
        print(f"\n🎯 Critical Statistics Availability:")
        critical_stats = [
            'shots_total', 'shots_on_target', 'shots_inside_box', 'shots_outside_box',
            'possession', 'corners', 'tackles', 'interceptions', 'fouls',
            'passes_total', 'passes_accurate', 'attacks', 'dangerous_attacks'
        ]
        
        for stat in critical_stats:
            if stat in df.columns:
                available = df[stat].notna().sum()
                total = len(df)
                pct = (available / total) * 100
                status = "✅" if pct > 90 else ("⚠️" if pct > 50 else "❌")
                print(f"  {status} {stat:25s}: {available:7,} / {total:7,} ({pct:5.1f}%)")
            else:
                print(f"  ❌ {stat:25s}: NOT FOUND")
        
        # xG availability (derived)
        if 'xG' in df.columns:
            xg_available = df['xG'].notna().sum()
            xg_pct = (xg_available / len(df)) * 100
            print(f"\n  ✅ Derived xG: {xg_available:,} / {len(df):,} ({xg_pct:.1f}%)")
        
        # Join quality with fixtures
        if self.fixtures is not None and 'fixture_id' in df.columns:
            print(f"\n🔗 Join Quality with Fixtures:")
            fixture_ids = set(self.fixtures['fixture_id'].unique())
            stats_fixture_ids = set(df['fixture_id'].unique())
            
            in_both = len(fixture_ids & stats_fixture_ids)
            only_fixtures = len(fixture_ids - stats_fixture_ids)
            only_stats = len(stats_fixture_ids - fixture_ids)
            
            print(f"  Fixtures with statistics: {in_both:,} / {len(fixture_ids):,} ({(in_both/len(fixture_ids))*100:.1f}%)")
            print(f"  Fixtures without statistics: {only_fixtures:,}")
            print(f"  Statistics without fixtures: {only_stats:,}")
    
    def analyze_lineups(self):
        """Analyze lineups data quality."""
        print("\n" + "=" * 80)
        print("3. LINEUPS ANALYSIS")
        print("=" * 80)
        
        if self.lineups is None:
            print("  ⚠️ Lineups data not loaded - skipping analysis")
            return
        
        df = self.lineups
        
        print(f"\n📊 Basic Statistics:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Coverage
        if 'fixture_id' in df.columns:
            print(f"\n📈 Coverage:")
            print(f"  Unique fixtures with lineups: {df['fixture_id'].nunique():,}")
            print(f"  Unique players: {df['player_id'].nunique():,}")
            
            # Players per fixture
            players_per_fixture = df.groupby('fixture_id').size()
            print(f"\n  Players per fixture:")
            print(f"    Mean: {players_per_fixture.mean():.1f}")
            print(f"    Median: {players_per_fixture.median():.0f}")
            print(f"    Min: {players_per_fixture.min()}")
            print(f"    Max: {players_per_fixture.max()}")
        
        # Starters vs bench
        if 'is_starter' in df.columns:
            print(f"\n⚽ Lineup Composition:")
            starter_counts = df['is_starter'].value_counts()
            for is_starter, count in starter_counts.items():
                role = "Starters" if is_starter else "Bench"
                print(f"  {role}: {count:,} ({(count/len(df))*100:.1f}%)")
        
        # Player statistics availability
        print(f"\n🎯 Player Statistics Availability:")
        player_stats = ['rating', 'minutes_played', 'goals', 'assists', 'shots_total', 
                       'passes_total', 'tackles', 'interceptions']
        
        for stat in player_stats:
            if stat in df.columns:
                available = df[stat].notna().sum()
                total = len(df)
                pct = (available / total) * 100
                status = "✅" if pct > 70 else ("⚠️" if pct > 30 else "❌")
                print(f"  {status} {stat:20s}: {available:7,} / {total:7,} ({pct:5.1f}%)")
            else:
                print(f"  ❌ {stat:20s}: NOT FOUND")
        
        # Rating distribution
        if 'rating' in df.columns:
            ratings = pd.to_numeric(df['rating'], errors='coerce').dropna()
            if len(ratings) > 0:
                print(f"\n📊 Rating Distribution:")
                print(f"  Mean: {ratings.mean():.2f}")
                print(f"  Median: {ratings.median():.2f}")
                print(f"  Min: {ratings.min():.2f}")
                print(f"  Max: {ratings.max():.2f}")
                print(f"  Std: {ratings.std():.2f}")
        
        # Join quality with fixtures
        if self.fixtures is not None and 'fixture_id' in df.columns:
            print(f"\n🔗 Join Quality with Fixtures:")
            fixture_ids = set(self.fixtures['fixture_id'].unique())
            lineup_fixture_ids = set(df['fixture_id'].unique())
            
            in_both = len(fixture_ids & lineup_fixture_ids)
            only_fixtures = len(fixture_ids - lineup_fixture_ids)
            
            print(f"  Fixtures with lineups: {in_both:,} / {len(fixture_ids):,} ({(in_both/len(fixture_ids))*100:.1f}%)")
            print(f"  Fixtures without lineups: {only_fixtures:,}")
    
    def analyze_sidelined(self):
        """Analyze sidelined data quality."""
        print("\n" + "=" * 80)
        print("4. SIDELINED ANALYSIS")
        print("=" * 80)
        
        if self.sidelined is None:
            print("  ⚠️ Sidelined data not loaded - skipping analysis")
            return
        
        df = self.sidelined
        
        print(f"\n📊 Basic Statistics:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Coverage
        if 'player_id' in df.columns:
            print(f"\n📈 Coverage:")
            print(f"  Unique players sidelined: {df['player_id'].nunique():,}")
            print(f"  Unique fixtures: {df['fixture_id'].nunique():,}")
        
        # Reason distribution
        if 'reason' in df.columns:
            print(f"\n🏥 Sidelined Reasons (Top 15):")
            reason_counts = df['reason'].value_counts().head(15)
            for reason, count in reason_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {reason:30s}: {count:5,} ({pct:5.1f}%)")
        
        # Date coverage
        if 'start_date' in df.columns and 'end_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            
            print(f"\n📅 Temporal Coverage:")
            print(f"  Earliest start: {df['start_date'].min()}")
            print(f"  Latest end: {df['end_date'].max()}")
            
            # Missing dates
            missing_start = df['start_date'].isnull().sum()
            missing_end = df['end_date'].isnull().sum()
            print(f"\n  Missing start_date: {missing_start:,} ({(missing_start/len(df))*100:.1f}%)")
            print(f"  Missing end_date: {missing_end:,} ({(missing_end/len(df))*100:.1f}%)")
    
    def analyze_cross_table_quality(self):
        """Analyze quality across multiple tables."""
        print("\n" + "=" * 80)
        print("5. CROSS-TABLE ANALYSIS")
        print("=" * 80)
        
        if self.fixtures is None:
            print("  ⚠️ Fixtures not loaded, skipping cross-table analysis")
            return
        
        fixture_ids = set(self.fixtures['fixture_id'].unique())
        
        print(f"\n🔗 Data Coverage Across Tables:")
        print(f"  Total fixtures: {len(fixture_ids):,}")
        
        if self.statistics is not None:
            stats_coverage = len(set(self.statistics['fixture_id'].unique()) & fixture_ids)
            print(f"  With statistics: {stats_coverage:,} ({(stats_coverage/len(fixture_ids))*100:.1f}%)")
        
        if self.lineups is not None:
            lineup_coverage = len(set(self.lineups['fixture_id'].unique()) & fixture_ids)
            print(f"  With lineups: {lineup_coverage:,} ({(lineup_coverage/len(fixture_ids))*100:.1f}%)")
        
        if self.sidelined is not None:
            sidelined_coverage = len(set(self.sidelined['fixture_id'].unique()) & fixture_ids)
            print(f"  With sidelined data: {sidelined_coverage:,} ({(sidelined_coverage/len(fixture_ids))*100:.1f}%)")
        
        # Complete data availability
        if self.statistics is not None and self.lineups is not None:
            stats_ids = set(self.statistics['fixture_id'].unique())
            lineup_ids = set(self.lineups['fixture_id'].unique())
            
            complete = fixture_ids & stats_ids & lineup_ids
            print(f"\n✅ Fixtures with COMPLETE data (fixtures + stats + lineups):")
            print(f"  {len(complete):,} / {len(fixture_ids):,} ({(len(complete)/len(fixture_ids))*100:.1f}%)")
    
    def generate_summary(self):
        """Generate overall data quality summary."""
        print("\n" + "=" * 80)
        print("6. OVERALL QUALITY SUMMARY")
        print("=" * 80)
        
        issues = []
        warnings = []
        
        # Check fixtures
        if self.fixtures is not None:
            if self.fixtures['result'].isnull().sum() > len(self.fixtures) * 0.1:
                issues.append("⚠️ More than 10% of fixtures missing results")
            
            if 'home_score' in self.fixtures.columns:
                if self.fixtures['home_score'].isnull().sum() > len(self.fixtures) * 0.1:
                    issues.append("⚠️ More than 10% of fixtures missing scores")
        
        # Check statistics
        if self.statistics is not None:
            critical_stats = ['shots_total', 'shots_on_target', 'possession']
            for stat in critical_stats:
                if stat in self.statistics.columns:
                    missing_pct = (self.statistics[stat].isnull().sum() / len(self.statistics)) * 100
                    if missing_pct > 50:
                        issues.append(f"❌ {stat} missing in {missing_pct:.1f}% of rows")
                    elif missing_pct > 20:
                        warnings.append(f"⚠️ {stat} missing in {missing_pct:.1f}% of rows")
        
        # Check lineups
        if self.lineups is not None and self.fixtures is not None:
            lineup_coverage = (self.lineups['fixture_id'].nunique() / self.fixtures['fixture_id'].nunique()) * 100
            if lineup_coverage < 50:
                issues.append(f"⚠️ Only {lineup_coverage:.1f}% of fixtures have lineup data")
        
        print(f"\n📋 Quality Assessment:")
        
        if len(issues) == 0 and len(warnings) == 0:
            print("  ✅ Data quality is EXCELLENT!")
            print("  ✅ Ready for training dataset generation")
        else:
            if len(issues) > 0:
                print(f"\n  ❌ Critical Issues ({len(issues)}):")
                for issue in issues:
                    print(f"    {issue}")
            
            if len(warnings) > 0:
                print(f"\n  ⚠️ Warnings ({len(warnings)}):")
                for warning in warnings:
                    print(f"    {warning}")
            
            if len(issues) == 0:
                print("\n  ✅ Data quality is GOOD (minor warnings only)")
                print("  ✅ Ready for training dataset generation")
            else:
                print("\n  ⚠️ Consider addressing critical issues before proceeding")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if self.fixtures is not None and self.statistics is not None:
            fixture_ids = set(self.fixtures['fixture_id'].unique())
            stats_ids = set(self.statistics['fixture_id'].unique())
            missing_stats = len(fixture_ids - stats_ids)
            
            if missing_stats > 0:
                print(f"  • {missing_stats:,} fixtures missing statistics - consider re-running data collection")
        
        if self.lineups is not None and self.fixtures is not None:
            lineup_coverage = (self.lineups['fixture_id'].nunique() / self.fixtures['fixture_id'].nunique()) * 100
            if lineup_coverage < 80:
                print(f"  • Lineup coverage is {lineup_coverage:.1f}% - feature engine will use team averages as fallback")
        
        if self.statistics is not None:
            if 'xG' in self.statistics.columns:
                xg_coverage = (self.statistics['xG'].notna().sum() / len(self.statistics)) * 100
                if xg_coverage > 95:
                    print(f"  • Derived xG successfully calculated for {xg_coverage:.1f}% of rows ✅")
        
        print(f"\n  • Proceed to: python scripts/generate_training_data.py")
    
    def run_full_analysis(self):
        """Run complete data quality analysis."""
        print("\n" + "=" * 80)
        print("CSV DATA QUALITY ANALYSIS")
        print("=" * 80)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.analyze_fixtures()
        self.analyze_statistics()
        self.analyze_lineups()
        self.analyze_sidelined()
        self.analyze_cross_table_quality()
        self.generate_summary()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)


def main():
    """Main entry point."""
    analyzer = DataQualityAnalyzer(data_dir='data/csv')
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
