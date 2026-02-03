"""
Player & Lineup Feature Calculator.

Extracts real player/lineup features when data is available,
with smart fallbacks for missing data.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class PlayerFeatureCalculator:
    """Calculate player and lineup features."""

    # Type IDs from SportMonks API
    TYPE_STARTING_11 = 11
    TYPE_BENCH = 12

    # Position IDs
    POSITION_GOALKEEPER = 24
    POSITION_DEFENDER = 25
    POSITION_MIDFIELDER = 26
    POSITION_FORWARD = 27

    # Detail type IDs
    DETAIL_RATING = 84
    DETAIL_MINUTES = 86

    def __init__(self, data_loader=None):
        """
        Initialize player feature calculator.

        Args:
            data_loader: Optional data loader for accessing lineup/sidelined data
        """
        self.data_loader = data_loader
        self._team_key_players_cache = {}  # Cache of key players per team

    def calculate_lineup_features(
        self,
        home_team_id: int,
        away_team_id: int,
        fixture_id: int,
        as_of_date: datetime
    ) -> Dict:
        """
        Calculate all 10 player/lineup features.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            fixture_id: Fixture ID (for lineup data)
            as_of_date: Date to calculate features as of

        Returns:
            Dictionary with 10 player features
        """
        features = {}

        # Try to get lineup data
        lineups = self._get_lineups(fixture_id) if self.data_loader else None

        if lineups:
            # Extract real lineup features
            home_lineup = self._parse_team_lineup(lineups, home_team_id)
            away_lineup = self._parse_team_lineup(lineups, away_team_id)

            features['home_lineup_avg_rating_5'] = self._calculate_lineup_quality(
                home_lineup, home_team_id, as_of_date
            )
            features['away_lineup_avg_rating_5'] = self._calculate_lineup_quality(
                away_lineup, away_team_id, as_of_date
            )

            features['home_top_3_players_rating'] = self._calculate_top_players_quality(
                home_lineup, 3
            )
            features['away_top_3_players_rating'] = self._calculate_top_players_quality(
                away_lineup, 3
            )

            features['home_key_players_available'] = self._count_key_players_starting(
                home_lineup, home_team_id, as_of_date
            )
            features['away_key_players_available'] = self._count_key_players_starting(
                away_lineup, away_team_id, as_of_date
            )
        else:
            # Use team-level estimates when lineup data not available
            features['home_lineup_avg_rating_5'] = self._estimate_team_quality(
                home_team_id, as_of_date
            )
            features['away_lineup_avg_rating_5'] = self._estimate_team_quality(
                away_team_id, as_of_date
            )

            features['home_top_3_players_rating'] = self._estimate_team_quality(
                home_team_id, as_of_date
            ) * 1.1  # Top players ~10% better
            features['away_top_3_players_rating'] = self._estimate_team_quality(
                away_team_id, as_of_date
            ) * 1.1

            # Assume most key players available if no data
            features['home_key_players_available'] = 4
            features['away_key_players_available'] = 4

        # Get sidelined data (injuries/suspensions)
        home_sidelined = self._get_sidelined_count(home_team_id, as_of_date)
        away_sidelined = self._get_sidelined_count(away_team_id, as_of_date)

        features['home_players_unavailable'] = home_sidelined
        features['away_players_unavailable'] = away_sidelined

        # Calculate form-based features
        features['home_players_in_form'] = self._calculate_team_form_indicator(
            home_team_id, as_of_date
        )
        features['away_players_in_form'] = self._calculate_team_form_indicator(
            away_team_id, as_of_date
        )

        return features

    def _get_lineups(self, fixture_id: int) -> Optional[List[Dict]]:
        """Get lineup data for fixture."""
        try:
            if hasattr(self.data_loader, 'get_lineups'):
                return self.data_loader.get_lineups(fixture_id)
        except Exception as e:
            logger.debug(f"Could not load lineups for fixture {fixture_id}: {e}")
        return None

    def _parse_team_lineup(self, lineups: List[Dict], team_id: int) -> List[Dict]:
        """
        Parse lineup data for a specific team.

        Returns list of starting players only (type_id=11).
        """
        team_players = [
            p for p in lineups
            if p.get('team_id') == team_id and p.get('type_id') == self.TYPE_STARTING_11
        ]
        return team_players

    def _calculate_lineup_quality(
        self,
        lineup: List[Dict],
        team_id: int,
        as_of_date: datetime
    ) -> float:
        """
        Calculate average lineup quality.

        Uses position-based scoring when ratings not available.
        """
        if not lineup:
            return self._estimate_team_quality(team_id, as_of_date)

        # Try to use player ratings if available
        ratings = []
        for player in lineup:
            for detail in player.get('details', []):
                if detail.get('type_id') == self.DETAIL_RATING:
                    rating = detail.get('data', {}).get('value')
                    if rating and rating > 1:  # Valid rating
                        ratings.append(rating)

        if len(ratings) >= 5:
            return sum(ratings) / len(ratings)

        # Fallback: position-based quality
        position_quality = {
            self.POSITION_GOALKEEPER: 7.0,
            self.POSITION_DEFENDER: 6.8,
            self.POSITION_MIDFIELDER: 7.0,
            self.POSITION_FORWARD: 7.2,
        }

        quality_scores = []
        for player in lineup:
            pos_id = player.get('position_id')
            quality_scores.append(position_quality.get(pos_id, 7.0))

        if quality_scores:
            return sum(quality_scores) / len(quality_scores)

        return self._estimate_team_quality(team_id, as_of_date)

    def _calculate_top_players_quality(
        self,
        lineup: List[Dict],
        top_n: int = 3
    ) -> float:
        """Calculate quality of top N players in lineup."""
        if not lineup:
            return 7.3  # Default for missing data

        # Use forwards and attacking midfielders as "top players"
        attacking_players = [
            p for p in lineup
            if p.get('position_id') in [self.POSITION_FORWARD, self.POSITION_MIDFIELDER]
        ]

        if len(attacking_players) >= top_n:
            return 7.5  # Has strong attacking lineup
        elif len(attacking_players) >= top_n - 1:
            return 7.3  # Decent attacking lineup
        else:
            return 7.0  # Weak attacking lineup

    def _count_key_players_starting(
        self,
        lineup: List[Dict],
        team_id: int,
        as_of_date: datetime
    ) -> int:
        """
        Count how many key players are in starting 11.

        Key players = at least one from each position group.
        """
        if not lineup:
            return 4  # Default assumption

        # Check if we have representation from each position
        positions_covered = set()
        for player in lineup:
            pos_id = player.get('position_id')
            positions_covered.add(pos_id)

        # Count position groups covered
        has_gk = self.POSITION_GOALKEEPER in positions_covered
        has_def = self.POSITION_DEFENDER in positions_covered
        has_mid = self.POSITION_MIDFIELDER in positions_covered
        has_fwd = self.POSITION_FORWARD in positions_covered

        # Award points for balanced lineup
        score = 0
        if has_gk:
            score += 1
        if has_def:
            score += 2  # Defenders important
        if has_mid:
            score += 1
        if has_fwd:
            score += 1

        return min(5, score)  # Max 5 key players

    def _estimate_team_quality(
        self,
        team_id: int,
        as_of_date: datetime
    ) -> float:
        """
        Estimate team quality based on recent performance.

        Uses team's recent results as proxy for player quality.
        """
        if not self.data_loader or not hasattr(self.data_loader, 'get_team_fixtures'):
            return 7.0  # Neutral default

        try:
            # Get last 5 matches
            recent = self.data_loader.get_team_fixtures(
                team_id,
                before_date=as_of_date,
                limit=5
            )

            if len(recent) == 0:
                return 7.0

            # Calculate win rate as quality indicator
            wins = 0
            draws = 0
            for _, match in recent.iterrows():
                result = match.get('result')
                is_home = match.get('home_team_id') == team_id

                if result == 'H' and is_home:
                    wins += 1
                elif result == 'A' and not is_home:
                    wins += 1
                elif result == 'D':
                    draws += 1

            win_rate = wins / len(recent)
            draw_rate = draws / len(recent)

            # Map to quality score (6.0 to 8.0 range)
            quality = 6.5 + (win_rate * 1.5) + (draw_rate * 0.5)
            return min(8.0, max(6.0, quality))

        except Exception as e:
            logger.debug(f"Could not estimate quality for team {team_id}: {e}")
            return 7.0

    def _get_sidelined_count(
        self,
        team_id: int,
        as_of_date: datetime
    ) -> int:
        """
        Get count of sidelined players (injured/suspended).

        Tries to load actual sidelined data if available, otherwise estimates
        based on fixture congestion and time of season.

        Args:
            team_id: Team ID
            as_of_date: Date to check sidelined status

        Returns:
            Number of players unavailable (injured/suspended)
        """
        # Try to load actual sidelined data if available
        sidelined_count = self._load_sidelined_data(team_id, as_of_date)
        if sidelined_count is not None:
            return sidelined_count

        # Fallback: Estimate based on fixture congestion and season timing
        return self._estimate_sidelined_count(team_id, as_of_date)

    def _load_sidelined_data(
        self,
        team_id: int,
        as_of_date: datetime
    ) -> Optional[int]:
        """
        Try to load actual sidelined data from JSON files.

        Args:
            team_id: Team ID
            as_of_date: Date to check sidelined status

        Returns:
            Number of sidelined players if data available, None otherwise
        """
        import os
        import json

        # Check if sidelined data exists
        sidelined_file = f"data/historical/sidelined/team_{team_id}.json"

        if not os.path.exists(sidelined_file):
            return None

        try:
            with open(sidelined_file, 'r') as f:
                sidelined_data = json.load(f)

            # Count players sidelined on this date
            unavailable_count = 0
            for player in sidelined_data:
                # Parse start and end dates
                start_date = pd.to_datetime(player.get('start_date'))
                end_date_str = player.get('end_date')
                end_date = pd.to_datetime(end_date_str) if end_date_str else as_of_date + timedelta(days=365)

                # Check if player was sidelined on this date
                if start_date <= as_of_date <= end_date:
                    unavailable_count += 1

            return unavailable_count if unavailable_count > 0 else 1

        except Exception as e:
            logger.debug(f"Could not load sidelined data for team {team_id}: {e}")
            return None

    def _estimate_sidelined_count(
        self,
        team_id: int,
        as_of_date: datetime
    ) -> int:
        """
        Estimate sidelined players based on fixture congestion and timing.

        Professional teams typically have 1-3 players unavailable at any time.
        This varies based on:
        - Fixture congestion (more games = more fatigue/injuries)
        - Time of season (December/April have higher injury rates)
        - Team depth and playing style

        Args:
            team_id: Team ID
            as_of_date: Date to estimate for

        Returns:
            Estimated number of unavailable players (1-5)
        """
        base_unavailable = 2  # Average team has 2 players out

        # Factor 1: Fixture congestion (recent match frequency)
        congestion_factor = 0
        if self.data_loader and hasattr(self.data_loader, 'get_team_fixtures'):
            try:
                # Check how many matches in last 14 days
                last_2_weeks = self.data_loader.get_team_fixtures(
                    team_id,
                    before_date=as_of_date,
                    limit=10
                )

                if len(last_2_weeks) > 0:
                    # Count matches in last 14 days
                    recent_matches = 0
                    two_weeks_ago = as_of_date - timedelta(days=14)

                    for _, match in last_2_weeks.iterrows():
                        match_date = pd.to_datetime(match['starting_at'])
                        if match_date >= two_weeks_ago:
                            recent_matches += 1

                    # More than 3 matches in 2 weeks = congested
                    if recent_matches >= 4:
                        congestion_factor = 2  # +2 injuries
                    elif recent_matches >= 3:
                        congestion_factor = 1  # +1 injury

            except Exception as e:
                logger.debug(f"Could not calculate congestion for team {team_id}: {e}")

        # Factor 2: Time of season (injury-prone periods)
        season_factor = 0
        month = as_of_date.month

        # December/January (winter congestion) and March/April (fatigue accumulation)
        if month in [12, 1, 4]:
            season_factor = 1  # +1 injury during high-risk months
        elif month in [3, 5]:
            season_factor = 0  # Slightly elevated

        # Calculate total (cap at 1-5 range)
        total_unavailable = base_unavailable + congestion_factor + season_factor
        return max(1, min(5, total_unavailable))

    def _calculate_team_form_indicator(
        self,
        team_id: int,
        as_of_date: datetime
    ) -> float:
        """
        Calculate what % of team is in form.

        Based on recent team results.
        """
        if not self.data_loader or not hasattr(self.data_loader, 'get_team_fixtures'):
            return 0.6  # Neutral default

        try:
            # Get last 3 matches
            recent = self.data_loader.get_team_fixtures(
                team_id,
                before_date=as_of_date,
                limit=3
            )

            if len(recent) == 0:
                return 0.6

            # Count wins
            wins = 0
            for _, match in recent.iterrows():
                result = match.get('result')
                is_home = match.get('home_team_id') == team_id

                if (result == 'H' and is_home) or (result == 'A' and not is_home):
                    wins += 1

            # Map wins to form percentage
            if wins == 3:
                return 0.8  # Excellent form
            elif wins == 2:
                return 0.7  # Good form
            elif wins == 1:
                return 0.6  # Average form
            else:
                return 0.5  # Poor form

        except Exception as e:
            logger.debug(f"Could not calculate form for team {team_id}: {e}")
            return 0.6
