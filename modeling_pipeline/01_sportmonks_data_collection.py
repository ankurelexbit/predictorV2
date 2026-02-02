#!/usr/bin/env python3
"""
01 - Sportmonks Data Collection Pipeline
=========================================

Comprehensive data collection from Sportmonks Football API.
This replaces all other data sources (football-data.co.uk, odds API, etc.)

Data Collected:
- Fixtures with full match details
- Statistics (40+ metrics per team per match)
- Events (goals, cards, substitutions, VAR)
- Lineups & formations
- Sidelined players (injuries/suspensions)
- Pre-match odds (136+ markets)
- Standings
- Team & player information

Usage:
    python 01_sportmonks_data_collection.py --full              # Full historical collection
    python 01_sportmonks_data_collection.py --update            # Update last 30 days
    python 01_sportmonks_data_collection.py --update --days 7   # Weekly update (last 7 days)
    python 01_sportmonks_data_collection.py --season 23690      # Specific season
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
API_KEY = "DQQStChRaPnjIryuZH2SxqJI5ufoA57wWsmFIuPCH2rvlBtm0G7Ch3mJoyE4"
BASE_URL = "https://api.sportmonks.com/v3/football"
CORE_URL = "https://api.sportmonks.com/v3/core"

# Output directories
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw" / "sportmonks"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Target leagues (ID: Name)
TARGET_LEAGUES = {
    8: "Premier League",
    9: "Championship",
    564: "La Liga",
    82: "Bundesliga",
    384: "Serie A",
    301: "Ligue 1",
    2: "Champions League",
    5: "Europa League",
}

# Rate limiting - Optimized for SportMonks API
REQUESTS_PER_MINUTE = 50  # 3000 per hour = 50 per minute
# No artificial delay - response time (~0.43s) naturally paces requests

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sportmonks")


class SportmonksAPI:
    """Sportmonks Football API client."""

    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
        # Optimized: Connection pooling for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        self.last_request_time = 0

    def _rate_limit(self):
        """Track request timing (no artificial delay - response time paces requests)."""
        self.last_request_time = time.time()

    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting and error handling."""
        self._rate_limit()

        url = f"{BASE_URL}/{endpoint}"
        params = params or {}

        try:
            resp = self.session.get(url, params=params, timeout=120)  # Optimized: Increased timeout for large payloads
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:  # Rate limited
                logger.warning("Rate limited, waiting 60s...")
                time.sleep(60)
                return self._request(endpoint, params)
            elif resp.status_code == 403:
                logger.error(f"Access denied: {resp.text[:200]}")
                return {"data": [], "error": resp.text}
            else:
                logger.error(f"HTTP error {resp.status_code}: {resp.text[:200]}")
                return {"data": [], "error": str(e)}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"data": [], "error": str(e)}

    def _paginate(self, endpoint: str, params: Dict = None, max_pages: int = 100) -> List[Dict]:
        """Fetch all pages of a paginated endpoint."""
        params = params or {}
        params["per_page"] = 100  # Optimized: Increased from 25 to 100 for fewer requests

        all_data = []
        page = 1

        while page <= max_pages:
            params["page"] = page
            response = self._request(endpoint, params)

            data = response.get("data", [])
            if not data:
                break

            all_data.extend(data)

            # Check pagination
            pagination = response.get("pagination", {})
            has_more = pagination.get("has_more", False)

            if not has_more:
                break

            page += 1
            logger.debug(f"  Fetched page {page-1}, total items: {len(all_data)}")

        return all_data

    # =========================================================================
    # LEAGUES & SEASONS
    # =========================================================================

    def get_leagues(self) -> List[Dict]:
        """Get all available leagues."""
        return self._paginate("leagues")

    def get_seasons(self, league_id: int = None) -> List[Dict]:
        """Get seasons, optionally filtered by league."""
        params = {}
        if league_id:
            params["filters"] = f"seasonLeagues:{league_id}"
        return self._paginate("seasons", params)

    def get_season_by_id(self, season_id: int) -> Dict:
        """Get specific season details."""
        response = self._request(f"seasons/{season_id}", {
            "include": "league;stages;rounds"
        })
        return response.get("data", {})

    # =========================================================================
    # FIXTURES
    # =========================================================================

    def get_fixtures_by_season(
        self,
        season_id: int,
        include_full_data: bool = True
    ) -> List[Dict]:
        """
        Get all fixtures for a season with optional full data.

        Args:
            season_id: Season ID
            include_full_data: Whether to include statistics, lineups, etc.
        """
        params = {"filters": f"fixtureSeasons:{season_id}"}

        if include_full_data:
            # Include essential data only - Optimized for speed
            # NOTE: lineups.details gives us player-level statistics (touches, clearances, etc.)
            includes = [
                "participants",
                "scores",
                "statistics",
                "events",
                "lineups.details",  # Player-level statistics
                "formations",
                "sidelined",
                "odds",
                "state",
                # Removed: coaches, venue, round, stage, league, weatherReport (not needed for modeling)
            ]
            params["include"] = ";".join(includes)

        return self._paginate(f"fixtures", params)

    def get_fixtures_by_date_range(
        self,
        start_date: str,
        end_date: str,
        league_ids: List[int] = None
    ) -> List[Dict]:
        """
        Get fixtures between dates.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league_ids: Optional list of league IDs to filter
        """
        params = {
            "include": "participants;scores;statistics;events;lineups.details;formations;sidelined;odds;venue;coaches"
        }

        if league_ids:
            params["filters"] = f"fixtureLeagues:{','.join(map(str, league_ids))}"

        return self._paginate(f"fixtures/between/{start_date}/{end_date}", params)

    def get_fixture_by_id(self, fixture_id: int) -> Dict:
        """Get single fixture with full details including player-level stats."""
        response = self._request(f"fixtures/{fixture_id}", {
            "include": "participants;scores;statistics;events;lineups.details;coaches;formations;sidelined;odds;venue;state;round;stage;league;weatherReport"
        })
        return response.get("data", {})

    # =========================================================================
    # STANDINGS
    # =========================================================================

    def get_standings_by_season(self, season_id: int) -> List[Dict]:
        """Get league standings for a season."""
        response = self._request(f"standings/seasons/{season_id}", {
            "include": "participant;details"
        })
        return response.get("data", [])

    # =========================================================================
    # TEAMS & PLAYERS
    # =========================================================================

    def get_teams_by_season(self, season_id: int) -> List[Dict]:
        """Get all teams in a season."""
        return self._paginate(f"teams/seasons/{season_id}", {
            "include": "venue;coaches"
        })

    def get_squad(self, team_id: int, season_id: int = None) -> List[Dict]:
        """Get team squad."""
        endpoint = f"squads/teams/{team_id}"
        if season_id:
            endpoint = f"squads/seasons/{season_id}/teams/{team_id}"

        response = self._request(endpoint, {"include": "player;position"})
        return response.get("data", [])

    def get_player(self, player_id: int) -> Dict:
        """Get player details."""
        response = self._request(f"players/{player_id}", {
            "include": "position;nationality;teams;statistics"
        })
        return response.get("data", {})

    # =========================================================================
    # ODDS
    # =========================================================================

    def get_odds_by_fixture(self, fixture_id: int) -> List[Dict]:
        """Get all odds for a fixture."""
        return self._paginate(f"odds/fixtures/{fixture_id}")

    def get_markets(self) -> List[Dict]:
        """Get all betting markets."""
        return self._paginate("markets")

    def get_bookmakers(self) -> List[Dict]:
        """Get all bookmakers."""
        return self._paginate("bookmakers")


# =============================================================================
# DATA PROCESSING
# =============================================================================

# Player-level statistics that should be summed (counts)
PLAYER_STATS_SUM = {
    41, 42, 51, 52, 56, 57, 58, 78, 79, 80, 83, 84, 86, 88,  # Basic
    94, 96, 97, 98, 99, 100, 101, 104, 105, 106, 107, 108, 109, 110,  # Advanced
    116, 117, 119, 120, 122, 123, 580, 581, 583, 584,  # More advanced
    1491, 27266, 27267, 27273, 27274,  # Extended
    48997,  # Error Lead To Shot
}

# Player-level statistics that should be averaged (percentages, ratings)
PLAYER_STATS_AVG = {
    82, 118,  # Pass %, Rating
    1533, 1584, 27268, 27270, 27275, 27276,  # Various percentages
}


def aggregate_player_stats(lineups: List[Dict]) -> Dict[int, float]:
    """
    Aggregate player-level statistics to team level.

    For count stats: sum across all players
    For percentage stats: average across players with data

    Args:
        lineups: List of lineup entries with 'details' containing player stats

    Returns:
        Dictionary mapping stat_id -> aggregated value
    """
    aggregated = {}
    count_by_stat = {}

    for lineup in lineups:
        details = lineup.get("details") or []
        for detail in details:
            stat_id = detail.get("type_id")

            # Extract numeric value from different possible structures
            # API returns: {'id': ..., 'type_id': 96, 'data': {'value': 2}}
            data = detail.get("data", {})
            if isinstance(data, dict):
                stat_value = data.get("value") or data.get("total")
            else:
                stat_value = data

            # Also try the old 'value' key for backwards compatibility
            if stat_value is None:
                value = detail.get("value", {})
                if isinstance(value, dict):
                    stat_value = value.get("value") or value.get("total")
                else:
                    stat_value = value

            if stat_value is None:
                continue

            try:
                stat_value = float(stat_value)
            except (TypeError, ValueError):
                continue

            # Aggregate
            if stat_id in PLAYER_STATS_AVG:
                # Average - track count for averaging later
                if stat_id not in aggregated:
                    aggregated[stat_id] = 0
                    count_by_stat[stat_id] = 0
                aggregated[stat_id] += stat_value
                count_by_stat[stat_id] += 1
            else:
                # Sum (default)
                if stat_id not in aggregated:
                    aggregated[stat_id] = 0
                aggregated[stat_id] += stat_value

    # Calculate averages for percentage stats
    for stat_id in PLAYER_STATS_AVG:
        if stat_id in aggregated and count_by_stat.get(stat_id, 0) > 0:
            aggregated[stat_id] = aggregated[stat_id] / count_by_stat[stat_id]

    return aggregated


def process_fixture(fixture: Dict) -> Dict:
    """
    Process a raw fixture into a flat structure for analysis.

    Returns:
        Processed fixture dictionary with all relevant fields
    """
    # Basic info
    processed = {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("starting_at"),
        "timestamp": fixture.get("starting_at_timestamp"),
        "league_id": fixture.get("league_id"),
        "season_id": fixture.get("season_id"),
        "round_id": fixture.get("round_id"),
        "stage_id": fixture.get("stage_id"),
        "venue_id": fixture.get("venue_id"),
        "state_id": fixture.get("state_id"),
        "name": fixture.get("name"),
        "leg": fixture.get("leg"),
        "length": fixture.get("length"),
        "has_odds": fixture.get("has_odds"),
    }

    # League info
    league = fixture.get("league") or {}
    if isinstance(league, dict):
        processed["league_name"] = league.get("name")
        processed["league_country_id"] = league.get("country_id")

    # Round info
    round_data = fixture.get("round") or {}
    if isinstance(round_data, dict):
        processed["round_name"] = round_data.get("name")

    # Venue info
    venue = fixture.get("venue") or {}
    if isinstance(venue, dict):
        processed["venue_name"] = venue.get("name")
        processed["venue_city"] = venue.get("city_id")

    # Weather
    weather = fixture.get("weatherReport") or {}
    if isinstance(weather, dict):
        processed["weather_type"] = weather.get("type")
        processed["weather_temp_c"] = weather.get("temperature", {}).get("temp") if isinstance(weather.get("temperature"), dict) else None
        processed["weather_humidity"] = weather.get("humidity")
        processed["weather_wind_speed"] = weather.get("wind", {}).get("speed") if isinstance(weather.get("wind"), dict) else None

    # Participants (teams)
    participants = fixture.get("participants") or []
    home_team = None
    away_team = None

    for p in participants:
        meta = p.get("meta", {})
        if meta.get("location") == "home":
            home_team = p
        else:
            away_team = p

    if home_team:
        processed["home_team_id"] = home_team.get("id")
        processed["home_team_name"] = home_team.get("name")
        processed["home_team_short"] = home_team.get("short_code")

    if away_team:
        processed["away_team_id"] = away_team.get("id")
        processed["away_team_name"] = away_team.get("name")
        processed["away_team_short"] = away_team.get("short_code")

    # Scores
    scores = fixture.get("scores") or []
    for score in scores:
        desc = score.get("description", "")
        participant = score.get("score", {}).get("participant", "")
        goals = score.get("score", {}).get("goals", 0)

        if desc == "CURRENT":
            if participant == "home":
                processed["home_goals"] = goals
            elif participant == "away":
                processed["away_goals"] = goals
        elif desc == "1ST_HALF":
            if participant == "home":
                processed["home_goals_ht"] = goals
            elif participant == "away":
                processed["away_goals_ht"] = goals

    # Determine result
    if processed.get("home_goals") is not None and processed.get("away_goals") is not None:
        if processed["home_goals"] > processed["away_goals"]:
            processed["result"] = "H"
        elif processed["home_goals"] < processed["away_goals"]:
            processed["result"] = "A"
        else:
            processed["result"] = "D"

    # Coaches
    coaches = fixture.get("coaches") or []
    for coach in coaches:
        meta = coach.get("meta", {})
        if meta.get("participant_id") == processed.get("home_team_id"):
            processed["home_coach_id"] = coach.get("id")
            processed["home_coach_name"] = coach.get("name")
        elif meta.get("participant_id") == processed.get("away_team_id"):
            processed["away_coach_id"] = coach.get("id")
            processed["away_coach_name"] = coach.get("name")

    # Formations
    formations = fixture.get("formations") or []
    for f in formations:
        if f.get("participant_id") == processed.get("home_team_id"):
            processed["home_formation"] = f.get("formation")
        elif f.get("participant_id") == processed.get("away_team_id"):
            processed["away_formation"] = f.get("formation")

    # Statistics
    statistics = fixture.get("statistics") or []
    for stat in statistics:
        type_id = stat.get("type_id")
        participant_id = stat.get("participant_id")
        value = stat.get("data", {}).get("value")

        prefix = "home" if participant_id == processed.get("home_team_id") else "away"
        processed[f"{prefix}_stat_{type_id}"] = value

    # Events summary
    events = fixture.get("events") or []
    home_goals_list = []
    away_goals_list = []
    home_cards = {"yellow": 0, "red": 0}
    away_cards = {"yellow": 0, "red": 0}

    for event in events:
        event_type = event.get("type_id")
        participant_id = event.get("participant_id")
        minute = event.get("minute", 0)

        is_home = participant_id == processed.get("home_team_id")

        # Goals (type 14 = goal, 15 = own goal, 16 = penalty)
        if event_type in [14, 16]:
            if is_home:
                home_goals_list.append(minute)
            else:
                away_goals_list.append(minute)
        elif event_type == 15:  # Own goal
            if is_home:
                away_goals_list.append(minute)
            else:
                home_goals_list.append(minute)
        # Cards
        elif event_type == 19:  # Yellow
            if is_home:
                home_cards["yellow"] += 1
            else:
                away_cards["yellow"] += 1
        elif event_type in [20, 21]:  # Red or Yellow-Red
            if is_home:
                home_cards["red"] += 1
            else:
                away_cards["red"] += 1

    processed["home_goal_minutes"] = ",".join(map(str, home_goals_list))
    processed["away_goal_minutes"] = ",".join(map(str, away_goals_list))
    processed["home_yellows"] = home_cards["yellow"]
    processed["away_yellows"] = away_cards["yellow"]
    processed["home_reds"] = home_cards["red"]
    processed["away_reds"] = away_cards["red"]

    # Lineups count and player-level statistics
    lineups = fixture.get("lineups") or []
    home_lineup = [l for l in lineups if l.get("team_id") == processed.get("home_team_id")]
    away_lineup = [l for l in lineups if l.get("team_id") == processed.get("away_team_id")]
    processed["home_lineup_count"] = len(home_lineup)
    processed["away_lineup_count"] = len(away_lineup)

    # Aggregate player-level statistics to team level
    # Player stats are in lineup.details
    home_player_stats = aggregate_player_stats(home_lineup)
    away_player_stats = aggregate_player_stats(away_lineup)

    for stat_id, value in home_player_stats.items():
        processed[f"home_player_stat_{stat_id}"] = value
    for stat_id, value in away_player_stats.items():
        processed[f"away_player_stat_{stat_id}"] = value

    # Sidelined (injuries/suspensions)
    sidelined = fixture.get("sidelined") or []
    home_sidelined = len([s for s in sidelined if s.get("participant_id") == processed.get("home_team_id")])
    away_sidelined = len([s for s in sidelined if s.get("participant_id") == processed.get("away_team_id")])
    processed["home_sidelined_count"] = home_sidelined
    processed["away_sidelined_count"] = away_sidelined

    # Odds (extract 1X2 main market)
    odds = fixture.get("odds") or []
    for odd in odds:
        # Market 1 is typically 1X2
        if odd.get("market_id") == 1:
            label = odd.get("label", "").lower()
            value = odd.get("value")
            if label in ["1", "home"]:
                if "odds_home" not in processed or processed.get("odds_home") is None:
                    processed["odds_home"] = float(value) if value else None
            elif label in ["x", "draw"]:
                if "odds_draw" not in processed or processed.get("odds_draw") is None:
                    processed["odds_draw"] = float(value) if value else None
            elif label in ["2", "away"]:
                if "odds_away" not in processed or processed.get("odds_away") is None:
                    processed["odds_away"] = float(value) if value else None

    return processed


def process_lineup(lineup: Dict, fixture_id: int) -> Dict:
    """Process a lineup entry."""
    return {
        "fixture_id": fixture_id,
        "player_id": lineup.get("player_id"),
        "team_id": lineup.get("team_id"),
        "position_id": lineup.get("position_id"),
        "formation_position": lineup.get("formation_field"),
        "type_id": lineup.get("type_id"),  # starter vs substitute
        "jersey_number": lineup.get("jersey_number"),
    }


def process_event(event: Dict) -> Dict:
    """Process an event entry."""
    return {
        "event_id": event.get("id"),
        "fixture_id": event.get("fixture_id"),
        "participant_id": event.get("participant_id"),
        "player_id": event.get("player_id"),
        "related_player_id": event.get("related_player_id"),
        "type_id": event.get("type_id"),
        "minute": event.get("minute"),
        "extra_minute": event.get("extra_minute"),
        "result": event.get("result"),
        "info": event.get("info"),
        "player_name": event.get("player_name"),
    }


def process_sidelined(sidelined: Dict) -> Dict:
    """Process a sidelined (injury) entry."""
    return {
        "sidelined_id": sidelined.get("id"),
        "fixture_id": sidelined.get("fixture_id"),
        "participant_id": sidelined.get("participant_id"),
        "player_id": sidelined.get("player_id"),
        "type_id": sidelined.get("type_id"),  # injury type
        "sideline_id": sidelined.get("sideline_id"),
    }


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_season_data(api: SportmonksAPI, season_id: int, league_name: str) -> Dict[str, pd.DataFrame]:
    """
    Collect all data for a season.

    Returns:
        Dict with DataFrames: fixtures, lineups, events, sidelined, standings
    """
    logger.info(f"Collecting data for season {season_id} ({league_name})")

    # Get fixtures with full data
    fixtures_raw = api.get_fixtures_by_season(season_id, include_full_data=True)
    logger.info(f"  Found {len(fixtures_raw)} fixtures")

    if not fixtures_raw:
        return {}

    # Process fixtures
    fixtures = []
    lineups = []
    events = []
    sidelined = []

    for fix in tqdm(fixtures_raw, desc=f"Processing {league_name}"):
        # Main fixture data
        fixtures.append(process_fixture(fix))

        fixture_id = fix.get("id")

        # Lineups
        for l in fix.get("lineups", []):
            lineups.append(process_lineup(l, fixture_id))

        # Events
        for e in fix.get("events", []):
            events.append(process_event(e))

        # Sidelined
        for s in fix.get("sidelined", []):
            sidelined.append(process_sidelined(s))

    # Get standings
    standings_raw = api.get_standings_by_season(season_id)
    standings = []
    for s in standings_raw:
        standings.append({
            "season_id": season_id,
            "team_id": s.get("participant_id"),
            "position": s.get("position"),
            "points": s.get("points"),
            "played": s.get("details", [{}])[0].get("value") if s.get("details") else None,
        })

    return {
        "fixtures": pd.DataFrame(fixtures),
        "lineups": pd.DataFrame(lineups),
        "events": pd.DataFrame(events),
        "sidelined": pd.DataFrame(sidelined),
        "standings": pd.DataFrame(standings),
    }


def collect_season_wrapper(args):
    """Wrapper for parallel season collection."""
    season_id, season_name, league_name, api = args
    try:
        data = collect_season_data(api, season_id, f"{league_name} {season_name}")
        if data:
            # Add metadata
            if not data["fixtures"].empty:
                data["fixtures"]["league_name"] = league_name
                data["fixtures"]["season_name"] = season_name
        return data
    except Exception as e:
        logger.error(f"Error collecting {league_name} {season_name}: {e}")
        return None


def collect_all_data(
    leagues: Dict[int, str] = TARGET_LEAGUES,
    seasons: List[int] = None,
    min_year: int = 2019,
    parallel: bool = True,
    max_workers: int = 4
) -> Dict[str, pd.DataFrame]:
    """
    Collect data for all target leagues and seasons.

    Args:
        leagues: Dict of league_id -> league_name
        seasons: Optional specific season IDs to collect
        min_year: Minimum year to collect (for filtering seasons)
        parallel: Whether to use parallel processing (default: True)
        max_workers: Number of parallel workers (default: 4)

    Returns:
        Dict with combined DataFrames
    """
    api = SportmonksAPI()

    all_fixtures = []
    all_lineups = []
    all_events = []
    all_sidelined = []
    all_standings = []

    # Collect season tasks
    season_tasks = []

    # Get seasons for each league
    for league_id, league_name in leagues.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTING: {league_name} (ID: {league_id})")
        logger.info(f"{'='*60}")

        league_seasons = api.get_seasons(league_id)

        # Filter seasons
        valid_seasons = []
        for s in league_seasons:
            season_name = s.get("name", "")
            # Extract year from season name (e.g., "2023/2024" -> 2023)
            try:
                year = int(season_name.split("/")[0])
                if year >= min_year:
                    valid_seasons.append(s)
            except:
                continue

        logger.info(f"Found {len(valid_seasons)} seasons since {min_year}")

        # Create tasks for parallel processing
        for season in valid_seasons:
            season_id = season.get("id")
            season_name = season.get("name")
            # Each worker gets its own API instance to avoid conflicts
            season_tasks.append((season_id, season_name, league_name, SportmonksAPI()))

    # Process seasons in parallel or sequentially
    if parallel and len(season_tasks) > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"PARALLEL PROCESSING: {len(season_tasks)} seasons with {max_workers} workers")
        logger.info(f"{'='*60}\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(collect_season_wrapper, season_tasks))

        # Aggregate results
        for data in results:
            if data:
                if not data["fixtures"].empty:
                    all_fixtures.append(data["fixtures"])

                if not data["lineups"].empty:
                    all_lineups.append(data["lineups"])

                if not data["events"].empty:
                    all_events.append(data["events"])

                if not data["sidelined"].empty:
                    all_sidelined.append(data["sidelined"])

                if not data["standings"].empty:
                    all_standings.append(data["standings"])
    else:
        # Sequential processing (fallback)
        for season_id, season_name, league_name, _ in season_tasks:
            data = collect_season_data(api, season_id, f"{league_name} {season_name}")

            if data:
                if not data["fixtures"].empty:
                    data["fixtures"]["league_name"] = league_name
                    data["fixtures"]["season_name"] = season_name
                    all_fixtures.append(data["fixtures"])

                if not data["lineups"].empty:
                    all_lineups.append(data["lineups"])

                if not data["events"].empty:
                    all_events.append(data["events"])

                if not data["sidelined"].empty:
                    all_sidelined.append(data["sidelined"])

                if not data["standings"].empty:
                    all_standings.append(data["standings"])

    # Combine all data
    result = {}

    if all_fixtures:
        result["fixtures"] = pd.concat(all_fixtures, ignore_index=True)
        logger.info(f"Total fixtures: {len(result['fixtures'])}")

    if all_lineups:
        result["lineups"] = pd.concat(all_lineups, ignore_index=True)
        logger.info(f"Total lineup entries: {len(result['lineups'])}")

    if all_events:
        result["events"] = pd.concat(all_events, ignore_index=True)
        logger.info(f"Total events: {len(result['events'])}")

    if all_sidelined:
        result["sidelined"] = pd.concat(all_sidelined, ignore_index=True)
        logger.info(f"Total sidelined entries: {len(result['sidelined'])}")

    if all_standings:
        result["standings"] = pd.concat(all_standings, ignore_index=True)
        logger.info(f"Total standings entries: {len(result['standings'])}")

    return result


def save_data(data: Dict[str, pd.DataFrame], output_dir: Path = RAW_DIR):
    """Save collected data to CSV files."""
    for name, df in data.items():
        if df is not None and not df.empty:
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info(f"Saved {name}: {len(df)} rows -> {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sportmonks Data Collection")
    parser.add_argument("--full", action="store_true", help="Full historical collection")
    parser.add_argument("--update", action="store_true", help="Update recent matches")
    parser.add_argument("--days", type=int, default=30, help="Number of days to update (default: 30, use 7 for weekly)")
    parser.add_argument("--season", type=int, help="Collect specific season by ID")
    parser.add_argument("--min-year", type=int, default=2016, help="Minimum year for full collection")
    parser.add_argument("--leagues", type=str, help="Comma-separated league IDs")

    args = parser.parse_args()

    print("=" * 60)
    print("SPORTMONKS DATA COLLECTION")
    print("=" * 60)

    api = SportmonksAPI()

    # Determine leagues to collect
    if args.leagues:
        league_ids = [int(x) for x in args.leagues.split(",")]
        leagues = {lid: TARGET_LEAGUES.get(lid, f"League {lid}") for lid in league_ids}
    else:
        leagues = TARGET_LEAGUES

    if args.season:
        # Collect single season
        season_info = api.get_season_by_id(args.season)
        league_name = season_info.get("league", {}).get("name", "Unknown")
        data = collect_season_data(api, args.season, league_name)

    elif args.update:
        # Update last N days (default 30, or use --days for custom)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

        logger.info(f"Updating fixtures from {start_date} to {end_date} ({args.days} days)")
        fixtures_raw = api.get_fixtures_by_date_range(
            start_date, end_date,
            list(leagues.keys())
        )

        fixtures = [process_fixture(f) for f in fixtures_raw]
        data = {"fixtures": pd.DataFrame(fixtures)}

    else:
        # Full collection
        data = collect_all_data(
            leagues=leagues,
            min_year=args.min_year
        )

    # Save data
    if data:
        save_data(data)

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)

        for name, df in data.items():
            if df is not None and not df.empty:
                print(f"  {name}: {len(df)} rows")
    else:
        print("No data collected!")


if __name__ == "__main__":
    main()
