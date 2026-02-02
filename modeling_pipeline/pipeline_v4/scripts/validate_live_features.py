#!/usr/bin/env python3
"""
Live Prediction Feature Sanity Check
=====================================

Validates that features generated during live prediction are correct and complete.
Checks for:
- Missing values (NaN/None)
- Feature value ranges
- Standings accuracy (via API comparison)
- Historical data availability
- Feature consistency
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FeatureSanityChecker:
    """Comprehensive feature validation for live predictions."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pipeline = StandaloneLivePipeline(api_key)
        self.pipeline.load_model()

    def check_fixture_features(self, fixture: dict) -> dict:
        """
        Generate and validate features for a single fixture.

        Returns dict with validation results.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"VALIDATING: {fixture['home_team_name']} vs {fixture['away_team_name']}")
        logger.info(f"League: {fixture['league_name']}")
        logger.info(f"Date: {fixture['starting_at']}")
        logger.info(f"{'='*80}\n")

        # Generate features
        logger.info("📊 Generating features...")
        features = self.pipeline.generate_features(fixture)

        if not features:
            return {'status': 'FAILED', 'error': 'Feature generation failed'}

        # Convert to DataFrame for analysis
        features_df = pd.DataFrame([features])

        results = {
            'status': 'PASS',
            'fixture': f"{fixture['home_team_name']} vs {fixture['away_team_name']}",
            'checks': {}
        }

        # Check 1: Missing values
        logger.info("✓ Check 1: Missing Values")
        missing = features_df.isnull().sum()
        missing_features = missing[missing > 0]

        if len(missing_features) > 0:
            logger.warning(f"  ⚠️  Found {len(missing_features)} features with missing values:")
            for feat, count in missing_features.items():
                logger.warning(f"    - {feat}: {count} missing")
            results['checks']['missing_values'] = 'WARN'
        else:
            logger.info(f"  ✅ No missing values (all {len(features)} features populated)")
            results['checks']['missing_values'] = 'PASS'

        # Check 2: Feature ranges
        logger.info("\n✓ Check 2: Feature Value Ranges")
        range_issues = self._check_feature_ranges(features)

        if range_issues:
            logger.warning(f"  ⚠️  Found {len(range_issues)} features with unexpected ranges:")
            for issue in range_issues[:5]:  # Show first 5
                logger.warning(f"    - {issue}")
            results['checks']['feature_ranges'] = 'WARN'
        else:
            logger.info(f"  ✅ All features in expected ranges")
            results['checks']['feature_ranges'] = 'PASS'

        # Check 3: Standings features
        logger.info("\n✓ Check 3: Standings Features Validation")
        standings_check = self._validate_standings(fixture, features)

        if standings_check['status'] == 'PASS':
            logger.info(f"  ✅ Standings features accurate")
            logger.info(f"    Home position: {standings_check['home_position']} (API verified)")
            logger.info(f"    Away position: {standings_check['away_position']} (API verified)")
            logger.info(f"    Home points: {standings_check['home_points']}")
            logger.info(f"    Away points: {standings_check['away_points']}")
            results['checks']['standings'] = 'PASS'
        else:
            logger.warning(f"  ⚠️  Standings validation failed: {standings_check.get('error')}")
            results['checks']['standings'] = 'WARN'

        # Check 4: Historical data availability
        logger.info("\n✓ Check 4: Historical Data Availability")
        history_check = self._check_historical_data(fixture)

        logger.info(f"  Home team ({fixture['home_team_name']}): {history_check['home_matches']} matches")
        logger.info(f"  Away team ({fixture['away_team_name']}): {history_check['away_matches']} matches")

        if history_check['home_matches'] >= 10 and history_check['away_matches'] >= 10:
            logger.info(f"  ✅ Sufficient historical data for rolling features")
            results['checks']['historical_data'] = 'PASS'
        else:
            logger.warning(f"  ⚠️  Insufficient historical data (need 10+ per team)")
            results['checks']['historical_data'] = 'WARN'

        # Check 5: Feature groups completeness
        logger.info("\n✓ Check 5: Feature Groups Completeness")
        groups = self._check_feature_groups(features)

        for group, count in groups.items():
            logger.info(f"  {group}: {count} features")

        total = sum(groups.values())
        if total == 162:
            logger.info(f"  ✅ All 162 features present")
            results['checks']['completeness'] = 'PASS'
        else:
            logger.warning(f"  ⚠️  Expected 162 features, got {total}")
            results['checks']['completeness'] = 'WARN'

        # Check 6: Feature correlations (sanity checks)
        logger.info("\n✓ Check 6: Feature Correlation Sanity Checks")
        correlation_checks = self._check_correlations(features)

        if correlation_checks['issues']:
            logger.warning(f"  ⚠️  Found {len(correlation_checks['issues'])} correlation issues:")
            for issue in correlation_checks['issues'][:3]:
                logger.warning(f"    - {issue}")
            results['checks']['correlations'] = 'WARN'
        else:
            logger.info(f"  ✅ Feature correlations look reasonable")
            results['checks']['correlations'] = 'PASS'

        return results

    def _check_feature_ranges(self, features: dict) -> list:
        """Check if features are in expected ranges."""
        issues = []

        # Elo should be 1000-2500
        if 'home_elo' in features:
            if not (1000 <= features['home_elo'] <= 2500):
                issues.append(f"home_elo={features['home_elo']} (expected 1000-2500)")

        if 'away_elo' in features:
            if not (1000 <= features['away_elo'] <= 2500):
                issues.append(f"away_elo={features['away_elo']} (expected 1000-2500)")

        # League positions should be 1-30 (max league size)
        if 'home_league_position' in features and features['home_league_position'] is not None:
            if not (1 <= features['home_league_position'] <= 30):
                issues.append(f"home_league_position={features['home_league_position']} (expected 1-30)")

        # Points per game should be 0-3
        if 'home_points_per_game' in features and features['home_points_per_game'] is not None:
            if not (0 <= features['home_points_per_game'] <= 3):
                issues.append(f"home_points_per_game={features['home_points_per_game']} (expected 0-3)")

        # xG should be 0-10
        if 'home_avg_xg_last_10' in features and features['home_avg_xg_last_10'] is not None:
            if features['home_avg_xg_last_10'] < 0 or features['home_avg_xg_last_10'] > 10:
                issues.append(f"home_avg_xg_last_10={features['home_avg_xg_last_10']} (expected 0-10)")

        # Probabilities should be 0-1
        prob_features = [k for k in features.keys() if 'prob' in k.lower()]
        for feat in prob_features:
            if features[feat] is not None:
                if not (0 <= features[feat] <= 1):
                    issues.append(f"{feat}={features[feat]} (expected 0-1)")

        return issues

    def _validate_standings(self, fixture: dict, features: dict) -> dict:
        """Validate standings features against API."""
        try:
            # Fetch actual standings from API
            season_id = fixture.get('season_id')
            if not season_id:
                return {'status': 'SKIP', 'error': 'No season_id'}

            standings = self.pipeline.fetch_season_standings(season_id)
            if standings is None:
                return {'status': 'SKIP', 'error': 'Could not fetch standings'}

            # Get team standings
            home_standing = standings[standings['team_id'] == fixture['home_team_id']]
            away_standing = standings[standings['team_id'] == fixture['away_team_id']]

            if len(home_standing) == 0 or len(away_standing) == 0:
                return {'status': 'SKIP', 'error': 'Teams not in standings'}

            home_pos_api = int(home_standing.iloc[0]['position'])
            away_pos_api = int(away_standing.iloc[0]['position'])
            home_pts_api = int(home_standing.iloc[0]['points'])
            away_pts_api = int(away_standing.iloc[0]['points'])

            # Compare with features
            home_pos_feat = features.get('home_league_position')
            away_pos_feat = features.get('away_league_position')
            home_pts_feat = features.get('home_points')
            away_pts_feat = features.get('away_points')

            # Check if they match
            pos_match = (home_pos_api == home_pos_feat and away_pos_api == away_pos_feat)
            pts_match = (home_pts_api == home_pts_feat and away_pts_api == away_pts_feat)

            if pos_match and pts_match:
                return {
                    'status': 'PASS',
                    'home_position': home_pos_api,
                    'away_position': away_pos_api,
                    'home_points': home_pts_api,
                    'away_points': away_pts_api
                }
            else:
                return {
                    'status': 'FAIL',
                    'error': f"Mismatch - API: H{home_pos_api}/A{away_pos_api}, Features: H{home_pos_feat}/A{away_pos_feat}"
                }

        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}

    def _check_historical_data(self, fixture: dict) -> dict:
        """Check how much historical data was available."""
        end_date = fixture['starting_at'][:10]

        home_matches = self.pipeline.fetch_team_recent_matches(
            fixture['home_team_id'],
            end_date,
            num_matches=20,
            days_back=180
        )

        away_matches = self.pipeline.fetch_team_recent_matches(
            fixture['away_team_id'],
            end_date,
            num_matches=20,
            days_back=180
        )

        return {
            'home_matches': len(home_matches),
            'away_matches': len(away_matches)
        }

    def _check_feature_groups(self, features: dict) -> dict:
        """Count features by group."""
        groups = {
            'Pillar 1 - Elo': 0,
            'Pillar 1 - Standings': 0,
            'Pillar 1 - Form': 0,
            'Pillar 1 - H2H': 0,
            'Pillar 1 - Home Advantage': 0,
            'Pillar 2 - xG': 0,
            'Pillar 2 - Shots': 0,
            'Pillar 2 - Defense': 0,
            'Pillar 2 - Attacks': 0,
            'Pillar 3 - Momentum': 0,
            'Pillar 3 - Parity': 0,
            'Pillar 3 - Draw Features': 0,
        }

        for key in features.keys():
            if 'elo' in key:
                groups['Pillar 1 - Elo'] += 1
            elif any(x in key for x in ['position', 'points', 'top_6', 'bottom_3', 'goal_difference']):
                groups['Pillar 1 - Standings'] += 1
            elif any(x in key for x in ['form', 'wins_', 'draws_', 'points_3', 'points_5', 'points_10']):
                groups['Pillar 1 - Form'] += 1
            elif 'h2h' in key:
                groups['Pillar 1 - H2H'] += 1
            elif 'home_advantage' in key or 'venue' in key:
                groups['Pillar 1 - Home Advantage'] += 1
            elif 'xg' in key.lower():
                groups['Pillar 2 - xG'] += 1
            elif 'shot' in key:
                groups['Pillar 2 - Shots'] += 1
            elif any(x in key for x in ['tackles', 'interceptions', 'clearances', 'blocks']):
                groups['Pillar 2 - Defense'] += 1
            elif 'attack' in key or 'dribble' in key or 'cross' in key:
                groups['Pillar 2 - Attacks'] += 1
            elif 'momentum' in key or 'trend' in key:
                groups['Pillar 3 - Momentum'] += 1
            elif 'parity' in key or 'difference' in key:
                groups['Pillar 3 - Parity'] += 1
            elif 'draw' in key:
                groups['Pillar 3 - Draw Features'] += 1

        return groups

    def _check_correlations(self, features: dict) -> dict:
        """Check if related features have sensible correlations."""
        issues = []

        # Elo difference and position difference should correlate
        if all(k in features for k in ['elo_diff', 'position_diff']):
            elo_diff = features['elo_diff']
            pos_diff = features.get('position_diff', 0)

            # Higher Elo should generally mean better position (lower number)
            if elo_diff > 100 and pos_diff is not None and pos_diff > 5:
                issues.append("Elo diff suggests home stronger, but position diff suggests otherwise")

        # Points and position should correlate
        if all(k in features for k in ['home_points', 'away_points', 'home_league_position', 'away_league_position']):
            if features['home_points'] is not None and features['away_points'] is not None:
                if features['home_points'] > features['away_points']:
                    if features['home_league_position'] is not None and features['away_league_position'] is not None:
                        if features['home_league_position'] > features['away_league_position']:
                            issues.append("Home has more points but worse position")

        # Form points should be reasonable
        if 'home_form_points_last_10' in features and features['home_form_points_last_10'] is not None:
            if features['home_form_points_last_10'] > 30:  # Max 30 points in 10 games
                issues.append(f"home_form_points_last_10={features['home_form_points_last_10']} (max should be 30)")

        return {'issues': issues}


def main():
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        print("❌ SPORTMONKS_API_KEY not set")
        sys.exit(1)

    print("=" * 80)
    print("LIVE PREDICTION FEATURE SANITY CHECK")
    print("=" * 80)
    print("\nThis script validates features generated during live prediction.")
    print("Checks: missing values, ranges, standings accuracy, historical data\n")

    # Initialize checker
    checker = FeatureSanityChecker(api_key)

    # Fetch today's fixtures
    today = datetime.now().strftime('%Y-%m-%d')
    fixtures = checker.pipeline.fetch_upcoming_fixtures(today, today)

    if not fixtures:
        print("❌ No fixtures found for today")
        sys.exit(1)

    print(f"✅ Found {len(fixtures)} fixtures for {today}\n")

    # Check first 3 fixtures (representative sample)
    num_to_check = min(3, len(fixtures))
    print(f"Validating {num_to_check} fixtures (representative sample)...\n")

    results = []
    for i in range(num_to_check):
        fixture = fixtures[i]
        result = checker.check_fixture_features(fixture)
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    all_checks = ['missing_values', 'feature_ranges', 'standings', 'historical_data', 'completeness', 'correlations']

    for check in all_checks:
        check_results = [r['checks'].get(check, 'N/A') for r in results]
        pass_count = sum(1 for r in check_results if r == 'PASS')
        warn_count = sum(1 for r in check_results if r == 'WARN')

        status = '✅' if pass_count == len(results) else ('⚠️' if warn_count > 0 else '❌')
        print(f"{status} {check.replace('_', ' ').title()}: {pass_count}/{len(results)} passed")

    print("\n" + "=" * 80)

    # Overall status
    all_pass = all(all(r['checks'].get(check) == 'PASS' for check in all_checks if check in r['checks'])
                   for r in results)

    if all_pass:
        print("✅ ALL CHECKS PASSED - Features are valid and accurate")
    else:
        print("⚠️  SOME CHECKS FAILED - Review warnings above")

    print("=" * 80)


if __name__ == '__main__':
    main()
