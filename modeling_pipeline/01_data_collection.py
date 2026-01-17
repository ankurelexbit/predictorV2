"""
01 - Data Collection
====================

This notebook fetches historical and current match data from multiple sources:
1. football-data.co.uk - Free historical CSV data (best for training)
2. Football-Data.org API - Fixtures, results (free tier)
3. API-Football - Detailed stats, lineups (free tier)
4. The Odds API - Betting odds (free tier)

Run this script to download all required data for model training.

Usage:
    python 01_data_collection.py

    # Or run specific collectors:
    python 01_data_collection.py --source csv
    python 01_data_collection.py --source football_data_api
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RAW_DATA_DIR,
    FOOTBALL_DATA_API_KEY,
    API_FOOTBALL_KEY,
    API_FOOTBALL_HOST,
    ODDS_API_KEY,
    LEAGUES_CSV,
    LEAGUES_API,
    LEAGUES_API_FOOTBALL,
    CSV_SEASONS,
    HISTORICAL_SEASONS,
    RATE_LIMITS,
)
from utils import setup_logger, normalize_team_name

# Setup logging
logger = setup_logger("data_collection")


# =============================================================================
# SOURCE 1: FOOTBALL-DATA.CO.UK (FREE HISTORICAL CSVs)
# =============================================================================
# Best source for historical data - no API key needed, comprehensive coverage

class FootballDataUKCollector:
    """
    Collector for football-data.co.uk CSV files.
    
    This is your PRIMARY source for historical training data:
    - Free, no API key
    - Goes back 20+ years
    - Includes betting odds from multiple bookmakers
    - Reliable and well-structured
    
    Data dictionary: https://www.football-data.co.uk/notes.txt
    """
    
    BASE_URL = "https://www.football-data.co.uk"
    
    def __init__(self, output_dir: Path = RAW_DATA_DIR / "football_data_uk"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_csv_url(self, league_code: str, season: str) -> str:
        """
        Construct URL for CSV download.
        
        Args:
            league_code: E0, E1, SP1, D1, I1, F1
            season: Two-digit format like "2324" for 2023-2024
        """
        return f"{self.BASE_URL}/mmz4281/{season}/{league_code}.csv"
    
    def download_season(self, league_code: str, season: str) -> Optional[pd.DataFrame]:
        """Download single season CSV."""
        url = self.get_csv_url(league_code, season)
        
        try:
            logger.info(f"Downloading {league_code} season {season} from {url}")
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(url, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"Failed to decode {url}")
                return None
            
            # Add metadata
            df['league_code'] = league_code
            df['season'] = self._convert_season_format(season)
            
            # Save locally
            output_file = self.output_dir / f"{league_code}_{season}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} matches to {output_file}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None
    
    def _convert_season_format(self, season: str) -> str:
        """Convert '2324' to '2023-2024'."""
        if len(season) == 4:
            start_year = int("20" + season[:2])
            end_year = int("20" + season[2:])
            return f"{start_year}-{end_year}"
        return season
    
    def download_all(
        self,
        leagues: List[str] = None,
        seasons: List[str] = None
    ) -> pd.DataFrame:
        """
        Download all specified leagues and seasons.
        
        Args:
            leagues: List of league codes (default: all configured)
            seasons: List of seasons in "2324" format (default: all configured)
        
        Returns:
            Combined DataFrame of all matches
        """
        leagues = leagues or list(LEAGUES_CSV.keys())
        seasons = seasons or CSV_SEASONS
        
        all_data = []
        
        for league in tqdm(leagues, desc="Leagues"):
            for season in tqdm(seasons, desc=f"  {league} seasons", leave=False):
                df = self.download_season(league, season)
                if df is not None:
                    all_data.append(df)
                
                # Be nice to the server
                time.sleep(1)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Save combined file
            combined_file = self.output_dir / "all_matches.csv"
            combined.to_csv(combined_file, index=False)
            logger.info(f"Saved combined data: {len(combined)} matches to {combined_file}")
            
            return combined
        
        return pd.DataFrame()
    
    def get_column_descriptions(self) -> Dict[str, str]:
        """Return column descriptions for football-data.co.uk CSVs."""
        return {
            # Match info
            "Div": "League Division",
            "Date": "Match Date (dd/mm/yy)",
            "Time": "Time of match kick off",
            "HomeTeam": "Home Team",
            "AwayTeam": "Away Team",
            
            # Result
            "FTHG": "Full Time Home Team Goals",
            "FTAG": "Full Time Away Team Goals",
            "FTR": "Full Time Result (H=Home Win, D=Draw, A=Away Win)",
            "HTHG": "Half Time Home Team Goals",
            "HTAG": "Half Time Away Team Goals",
            "HTR": "Half Time Result",
            
            # Statistics (may not be available for all leagues/seasons)
            "HS": "Home Team Shots",
            "AS": "Away Team Shots",
            "HST": "Home Team Shots on Target",
            "AST": "Away Team Shots on Target",
            "HF": "Home Team Fouls Committed",
            "AF": "Away Team Fouls Committed",
            "HC": "Home Team Corners",
            "AC": "Away Team Corners",
            "HY": "Home Team Yellow Cards",
            "AY": "Away Team Yellow Cards",
            "HR": "Home Team Red Cards",
            "AR": "Away Team Red Cards",
            
            # Betting Odds (key bookmakers)
            "B365H": "Bet365 home win odds",
            "B365D": "Bet365 draw odds",
            "B365A": "Bet365 away win odds",
            "BWH": "Betway home win odds",
            "BWD": "Betway draw odds",
            "BWA": "Betway away win odds",
            "PSH": "Pinnacle home win odds",
            "PSD": "Pinnacle draw odds",
            "PSA": "Pinnacle away win odds",
            
            # Market averages
            "AvgH": "Market average home win odds",
            "AvgD": "Market average draw odds",
            "AvgA": "Market average away win odds",
            "MaxH": "Market maximum home win odds",
            "MaxD": "Market maximum draw win odds",
            "MaxA": "Market maximum away win odds",
        }


# =============================================================================
# SOURCE 2: FOOTBALL-DATA.ORG API (FREE TIER)
# =============================================================================
# Good for current fixtures and recent results

class FootballDataOrgCollector:
    """
    Collector for Football-Data.org API.
    
    Free tier: 10 requests/minute
    Good for: Current fixtures, standings, recent results
    
    API docs: https://www.football-data.org/documentation/quickstart
    """
    
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self, api_key: str = FOOTBALL_DATA_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "X-Auth-Token": api_key
        })
        self.output_dir = RAW_DATA_DIR / "football_data_org"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_count = 0
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting (10 requests/minute)."""
        self.request_count += 1
        
        # Wait if needed
        elapsed = time.time() - self.last_request_time
        if elapsed < 6:  # 10 req/min = 1 req/6 sec
            time.sleep(6 - elapsed)
        
        self.last_request_time = time.time()
    
    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated GET request."""
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_competitions(self) -> List[Dict]:
        """Get list of available competitions."""
        data = self._get("competitions")
        if data:
            return data.get("competitions", [])
        return []
    
    def get_matches(
        self,
        competition_code: str,
        date_from: str = None,
        date_to: str = None,
        status: str = None
    ) -> List[Dict]:
        """
        Get matches for a competition.
        
        Args:
            competition_code: PL, BL1, SA, etc.
            date_from: YYYY-MM-DD
            date_to: YYYY-MM-DD
            status: SCHEDULED, LIVE, IN_PLAY, PAUSED, FINISHED
        """
        params = {}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        if status:
            params["status"] = status
        
        data = self._get(f"competitions/{competition_code}/matches", params)
        if data:
            return data.get("matches", [])
        return []
    
    def get_standings(self, competition_code: str) -> Dict:
        """Get current standings for a competition."""
        return self._get(f"competitions/{competition_code}/standings")
    
    def get_upcoming_fixtures(
        self,
        competition_code: str,
        days_ahead: int = 7
    ) -> pd.DataFrame:
        """Get upcoming fixtures for next N days."""
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        matches = self.get_matches(
            competition_code,
            date_from=today,
            date_to=end_date,
            status="SCHEDULED"
        )
        
        if not matches:
            return pd.DataFrame()
        
        # Convert to DataFrame
        fixtures = []
        for match in matches:
            fixtures.append({
                "match_id": match["id"],
                "competition": match["competition"]["name"],
                "matchday": match.get("matchday"),
                "date": match["utcDate"],
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_team_id": match["homeTeam"]["id"],
                "away_team_id": match["awayTeam"]["id"],
            })
        
        df = pd.DataFrame(fixtures)
        
        # Save
        output_file = self.output_dir / f"fixtures_{competition_code}_{today}.json"
        with open(output_file, "w") as f:
            json.dump(matches, f, indent=2)
        
        return df
    
    def get_season_matches(
        self,
        competition_code: str,
        season: int = None
    ) -> pd.DataFrame:
        """
        Get all matches for a season.
        
        Args:
            competition_code: League code (PL, BL1, etc.)
            season: Season start year (e.g., 2023 for 2023-2024)
        """
        params = {}
        if season:
            params["season"] = season
        
        data = self._get(f"competitions/{competition_code}/matches", params)
        
        if not data or "matches" not in data:
            return pd.DataFrame()
        
        matches = []
        for match in data["matches"]:
            matches.append({
                "match_id": match["id"],
                "date": match["utcDate"],
                "matchday": match.get("matchday"),
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_goals": match["score"]["fullTime"]["home"],
                "away_goals": match["score"]["fullTime"]["away"],
                "status": match["status"],
            })
        
        df = pd.DataFrame(matches)
        
        # Add result column
        def calc_result(row):
            if pd.isna(row["home_goals"]) or pd.isna(row["away_goals"]):
                return None
            if row["home_goals"] > row["away_goals"]:
                return "H"
            elif row["home_goals"] < row["away_goals"]:
                return "A"
            return "D"
        
        df["result"] = df.apply(calc_result, axis=1)
        
        return df


# =============================================================================
# SOURCE 3: API-FOOTBALL (FREE TIER VIA RAPIDAPI)
# =============================================================================
# Detailed statistics, lineups, injuries

class APIFootballCollector:
    """
    Collector for API-Football via RapidAPI.
    
    Free tier: 100 requests/day
    Good for: Detailed match stats, lineups, injuries, player data
    
    API docs: https://www.api-football.com/documentation-v3
    """
    
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
    
    def __init__(
        self,
        api_key: str = API_FOOTBALL_KEY,
        api_host: str = API_FOOTBALL_HOST
    ):
        self.session = requests.Session()
        self.session.headers.update({
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": api_host
        })
        self.output_dir = RAW_DATA_DIR / "api_football"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.daily_requests = 0
    
    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated GET request."""
        if self.daily_requests >= 100:
            logger.warning("Daily API limit reached (100 requests)")
            return None
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            self.daily_requests += 1
            
            data = response.json()
            
            # Check for API errors
            if data.get("errors"):
                logger.error(f"API error: {data['errors']}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_fixtures(
        self,
        league_id: int,
        season: int,
        from_date: str = None,
        to_date: str = None
    ) -> List[Dict]:
        """Get fixtures for a league."""
        params = {
            "league": league_id,
            "season": season,
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        data = self._get("fixtures", params)
        if data:
            return data.get("response", [])
        return []
    
    def get_fixture_statistics(self, fixture_id: int) -> Optional[Dict]:
        """Get detailed statistics for a specific fixture."""
        data = self._get("fixtures/statistics", {"fixture": fixture_id})
        if data:
            return data.get("response", [])
        return None
    
    def get_injuries(self, league_id: int, season: int) -> List[Dict]:
        """Get injury data for a league/season."""
        data = self._get("injuries", {"league": league_id, "season": season})
        if data:
            return data.get("response", [])
        return []
    
    def get_standings(self, league_id: int, season: int) -> List[Dict]:
        """Get league standings."""
        data = self._get("standings", {"league": league_id, "season": season})
        if data:
            return data.get("response", [])
        return []
    
    def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> List[Dict]:
        """Get head-to-head history between two teams."""
        data = self._get("fixtures/headtohead", {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last
        })
        if data:
            return data.get("response", [])
        return []
    
    def download_season_data(
        self,
        league_name: str,
        season: int
    ) -> pd.DataFrame:
        """Download all fixtures with stats for a season."""
        league_id = LEAGUES_API_FOOTBALL.get(league_name)
        if not league_id:
            logger.error(f"Unknown league: {league_name}")
            return pd.DataFrame()
        
        fixtures = self.get_fixtures(league_id, season)
        
        if not fixtures:
            return pd.DataFrame()
        
        matches = []
        for fixture in fixtures:
            match_data = {
                "fixture_id": fixture["fixture"]["id"],
                "date": fixture["fixture"]["date"],
                "venue": fixture["fixture"].get("venue", {}).get("name"),
                "home_team": fixture["teams"]["home"]["name"],
                "away_team": fixture["teams"]["away"]["name"],
                "home_team_id": fixture["teams"]["home"]["id"],
                "away_team_id": fixture["teams"]["away"]["id"],
                "home_goals": fixture["goals"]["home"],
                "away_goals": fixture["goals"]["away"],
                "status": fixture["fixture"]["status"]["short"],
            }
            matches.append(match_data)
        
        df = pd.DataFrame(matches)
        
        # Save
        output_file = self.output_dir / f"{league_name}_{season}.json"
        with open(output_file, "w") as f:
            json.dump(fixtures, f, indent=2)
        
        return df


# =============================================================================
# SOURCE 4: THE ODDS API (FREE TIER)
# =============================================================================
# Betting odds from multiple bookmakers

class OddsAPICollector:
    """
    Collector for The Odds API.
    
    Free tier: 500 requests/month
    Good for: Live betting odds from multiple bookmakers
    
    API docs: https://the-odds-api.com/liveapi/guides/v4/
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Sport keys for football
    SPORT_KEYS = {
        "Premier League": "soccer_epl",
        "Championship": "soccer_efl_champ",
        "La Liga": "soccer_spain_la_liga",
        "Bundesliga": "soccer_germany_bundesliga",
        "Serie A": "soccer_italy_serie_a",
        "Ligue 1": "soccer_france_ligue_one",
    }
    
    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.output_dir = RAW_DATA_DIR / "odds_api"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated GET request."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        params = params or {}
        params["apiKey"] = self.api_key
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Log remaining requests
            remaining = response.headers.get("x-requests-remaining")
            used = response.headers.get("x-requests-used")
            logger.info(f"Odds API: {used} used, {remaining} remaining this month")
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Odds API request failed: {e}")
            return None
    
    def get_sports(self) -> List[Dict]:
        """Get list of available sports."""
        return self._get("sports") or []
    
    def get_odds(
        self,
        sport_key: str,
        regions: str = "uk,eu",
        markets: str = "h2h",
        odds_format: str = "decimal"
    ) -> List[Dict]:
        """
        Get current odds for a sport.
        
        Args:
            sport_key: e.g., "soccer_epl"
            regions: Bookmaker regions (uk, us, eu, au)
            markets: Bet markets (h2h = 1X2, spreads, totals)
            odds_format: decimal or american
        """
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        
        return self._get(f"sports/{sport_key}/odds", params) or []
    
    def get_current_odds_df(self, league_name: str) -> pd.DataFrame:
        """Get current 1X2 odds as DataFrame."""
        sport_key = self.SPORT_KEYS.get(league_name)
        if not sport_key:
            logger.error(f"Unknown league: {league_name}")
            return pd.DataFrame()
        
        odds_data = self.get_odds(sport_key)
        
        if not odds_data:
            return pd.DataFrame()
        
        records = []
        for event in odds_data:
            home_team = event["home_team"]
            away_team = event["away_team"]
            commence_time = event["commence_time"]
            
            for bookmaker in event.get("bookmakers", []):
                book_name = bookmaker["key"]
                
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                        
                        records.append({
                            "event_id": event["id"],
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": book_name,
                            "home_odds": outcomes.get(home_team),
                            "draw_odds": outcomes.get("Draw"),
                            "away_odds": outcomes.get(away_team),
                            "last_update": bookmaker["last_update"],
                        })
        
        df = pd.DataFrame(records)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"odds_{league_name}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        return df


# =============================================================================
# MAIN COLLECTION ORCHESTRATOR
# =============================================================================

class DataCollector:
    """Main orchestrator for all data collection."""
    
    def __init__(self):
        self.csv_collector = FootballDataUKCollector()
        self.fd_api_collector = FootballDataOrgCollector()
        # Only initialize if API keys are set
        if API_FOOTBALL_KEY != "YOUR_RAPIDAPI_KEY_HERE":
            self.api_football_collector = APIFootballCollector()
        else:
            self.api_football_collector = None
            logger.warning("API-Football not configured (no API key)")
        
        if ODDS_API_KEY != "YOUR_ODDS_API_KEY_HERE":
            self.odds_collector = OddsAPICollector()
        else:
            self.odds_collector = None
            logger.warning("Odds API not configured (no API key)")
    
    def collect_historical_data(self) -> pd.DataFrame:
        """
        Collect historical match data from football-data.co.uk.
        This is the PRIMARY data source for model training.
        """
        logger.info("=" * 60)
        logger.info("Collecting historical data from football-data.co.uk")
        logger.info("=" * 60)
        
        df = self.csv_collector.download_all()
        
        logger.info(f"Collected {len(df)} historical matches")
        return df
    
    def collect_current_fixtures(self) -> pd.DataFrame:
        """Collect upcoming fixtures from Football-Data.org."""
        logger.info("=" * 60)
        logger.info("Collecting upcoming fixtures")
        logger.info("=" * 60)
        
        all_fixtures = []
        
        for code, info in LEAGUES_API.items():
            logger.info(f"Fetching fixtures for {info['name']}")
            df = self.fd_api_collector.get_upcoming_fixtures(code)
            if not df.empty:
                all_fixtures.append(df)
        
        if all_fixtures:
            combined = pd.concat(all_fixtures, ignore_index=True)
            logger.info(f"Collected {len(combined)} upcoming fixtures")
            return combined
        
        return pd.DataFrame()
    
    def collect_current_odds(self) -> pd.DataFrame:
        """Collect current betting odds."""
        if self.odds_collector is None:
            logger.warning("Odds API not configured")
            return pd.DataFrame()
        
        logger.info("=" * 60)
        logger.info("Collecting current odds")
        logger.info("=" * 60)
        
        all_odds = []
        
        for league_name in ["Premier League", "La Liga", "Bundesliga"]:
            logger.info(f"Fetching odds for {league_name}")
            df = self.odds_collector.get_current_odds_df(league_name)
            if not df.empty:
                all_odds.append(df)
        
        if all_odds:
            combined = pd.concat(all_odds, ignore_index=True)
            logger.info(f"Collected odds for {len(combined)} events")
            return combined
        
        return pd.DataFrame()
    
    def collect_all(self):
        """Run complete data collection pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING FULL DATA COLLECTION")
        logger.info("=" * 60)
        
        # 1. Historical data (main training source)
        historical_df = self.collect_historical_data()
        
        # 2. Current fixtures (for predictions)
        fixtures_df = self.collect_current_fixtures()
        
        # 3. Current odds (for edge detection)
        odds_df = self.collect_current_odds()
        
        # Summary
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Historical matches: {len(historical_df)}")
        logger.info(f"Upcoming fixtures: {len(fixtures_df)}")
        logger.info(f"Current odds records: {len(odds_df)}")
        
        return {
            "historical": historical_df,
            "fixtures": fixtures_df,
            "odds": odds_df,
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Football Data Collection")
    parser.add_argument(
        "--source",
        choices=["csv", "football_data_api", "api_football", "odds", "all"],
        default="all",
        help="Data source to collect from"
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="Specific leagues to collect (default: all)"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Specific seasons in '2324' format (default: all configured)"
    )
    
    args = parser.parse_args()
    
    collector = DataCollector()
    
    if args.source == "csv":
        df = collector.csv_collector.download_all(
            leagues=args.leagues,
            seasons=args.seasons
        )
        print(f"\nDownloaded {len(df)} matches from football-data.co.uk")
        print(f"Data saved to: {collector.csv_collector.output_dir}")
        
    elif args.source == "football_data_api":
        df = collector.collect_current_fixtures()
        print(f"\nCollected {len(df)} upcoming fixtures")
        
    elif args.source == "odds":
        df = collector.collect_current_odds()
        print(f"\nCollected {len(df)} odds records")
        
    elif args.source == "all":
        data = collector.collect_all()
        print("\n" + "=" * 60)
        print("DATA COLLECTION SUMMARY")
        print("=" * 60)
        for key, df in data.items():
            print(f"{key}: {len(df)} records")


if __name__ == "__main__":
    main()
