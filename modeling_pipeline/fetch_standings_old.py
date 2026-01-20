"""Fetch real league standings from ESPN API (free, no auth required)."""
import requests
import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ESPN league codes for major European leagues
ESPN_LEAGUE_CODES = {
    'Premier League': 'eng.1',
    'Championship': 'eng.2',
    'La Liga': 'esp.1',
    'La Liga 2': 'esp.2',
    'Serie A': 'ita.1',
    'Serie B': 'ita.2',
    'Bundesliga': 'ger.1',
    '2. Bundesliga': 'ger.2',
    'Ligue 1': 'fra.1',
    'Ligue 2': 'fra.2',
    'Eredivisie': 'ned.1',
    'Primeira Liga': 'por.1',
    'Pro League': 'bel.1',
    'Super Lig': 'tur.1',
    'Scottish Premiership': 'sco.1',
    'Swiss Super League': 'sui.1',
    'Greek Super League': 'gre.1',  # Added
}

# Team name mappings (Sportmonks name -> ESPN name variations)
TEAM_NAME_VARIATIONS = {
    'Paris': ['Paris Saint-Germain', 'PSG', 'Paris SG'],
    'Wolverhampton Wanderers': ['Wolves', 'Wolverhampton'],
    'Newcastle United': ['Newcastle', 'Newcastle United'],
    'Tottenham Hotspur': ['Tottenham', 'Spurs'],
    'Brighton & Hove Albion': ['Brighton', 'Brighton & Hove Albion'],
    'West Ham United': ['West Ham', 'West Ham United'],
    # German teams
    'Bayer 04 Leverkusen': ['Bayer Leverkusen', 'Leverkusen'],
    'FC Bayern München': ['Bayern Munich', 'Bayern'],
    'Borussia Dortmund': ['Dortmund'],
    'RB Leipzig': ['Leipzig'],
    # Greek teams
    'Olympiacos F.C.': ['Olympiacos', 'Olympiakos'],
    'Panathinaikos': ['Panathinaikos Athens'],
    'AEK Athens': ['AEK Athens F.C.'],
    # Italian teams
    'Inter': ['Inter Milan', 'Internazionale'],
    'AC Milan': ['Milan', 'AC Milan'],
    # Spanish teams
    'Atlético Madrid': ['Atletico Madrid', 'Atlético de Madrid'],
    'Athletic Bilbao': ['Athletic Club'],
}


def fetch_espn_standings(league_code: str) -> Optional[Dict]:
    """Fetch standings for a league from ESPN API.

    Args:
        league_code: ESPN league code (e.g., 'eng.1', 'esp.1')

    Returns:
        Dict mapping team name to {position, points} or None if failed
    """
    url = f'https://site.api.espn.com/apis/v2/sports/soccer/{league_code}/standings'

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'children' not in data:
            logger.warning(f"No standings data for {league_code}")
            return None

        standings_data = data['children'][0]['standings']['entries']

        # Build lookup dict
        standings = {}
        for entry in standings_data:
            team_name = entry['team']['displayName']

            # Extract position and points from stats array
            position = None
            points = None
            for stat in entry['stats']:
                if stat['name'] == 'rank':
                    position = int(stat['value'])
                elif stat['name'] == 'points':
                    points = int(stat['value'])

            if position and points is not None:
                standings[team_name] = {
                    'position': position,
                    'points': points
                }

        logger.info(f"Fetched standings for {league_code}: {len(standings)} teams")
        return standings

    except Exception as e:
        logger.error(f"Failed to fetch standings for {league_code}: {e}")
        return None


def get_team_standings(team_name: str, league_name: str = None) -> Optional[Dict]:
    """Get position and points for a team.

    Args:
        team_name: Name of the team
        league_name: League name (if known)

    Returns:
        Dict with 'position' and 'points' or None
    """
    # If league specified, try that first
    if league_name and league_name in ESPN_LEAGUE_CODES:
        league_code = ESPN_LEAGUE_CODES[league_name]
        standings = fetch_espn_standings(league_code)

        if standings:
            # Try exact match
            if team_name in standings:
                return standings[team_name]

            # Try case-insensitive match
            for espn_name, data in standings.items():
                if team_name.lower() == espn_name.lower():
                    return data

            # Try partial match (e.g., "Feyenoord" matches "Feyenoord Rotterdam")
            for espn_name, data in standings.items():
                if team_name.lower() in espn_name.lower() or espn_name.lower() in team_name.lower():
                    logger.info(f"Partial match: '{team_name}' -> '{espn_name}' in {league_name}")
                    return data

            # Try variations
            if team_name in TEAM_NAME_VARIATIONS:
                for variation in TEAM_NAME_VARIATIONS[team_name]:
                    if variation in standings:
                        return standings[variation]
                    # Case-insensitive check
                    for espn_name in standings:
                        if variation.lower() == espn_name.lower():
                            return standings[espn_name]

    # If not found or no league specified, try all major leagues
    for league_name, league_code in ESPN_LEAGUE_CODES.items():
        standings = fetch_espn_standings(league_code)

        if standings:
            # Try exact match
            if team_name in standings:
                logger.info(f"Found {team_name} in {league_name}")
                return standings[team_name]

            # Try case-insensitive
            for espn_name, data in standings.items():
                if team_name.lower() == espn_name.lower():
                    logger.info(f"Found {team_name} as {espn_name} in {league_name}")
                    return data

    logger.warning(f"Could not find standings for team: {team_name}")
    return None


def test_espn_api():
    """Test ESPN API with sample teams."""
    print("=" * 80)
    print("TESTING ESPN STANDINGS API")
    print("=" * 80)

    test_teams = [
        ('Arsenal', 'Premier League'),
        ('Real Madrid', 'La Liga'),
        ('Bayern Munich', 'Bundesliga'),
        ('Paris', 'Ligue 1'),
        ('Feyenoord', 'Eredivisie'),
    ]

    for team, league in test_teams:
        print(f"\n{team} ({league}):")
        result = get_team_standings(team, league)
        if result:
            print(f"  Position: {result['position']}")
            print(f"  Points: {result['points']}")
        else:
            print("  NOT FOUND")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_espn_api()
