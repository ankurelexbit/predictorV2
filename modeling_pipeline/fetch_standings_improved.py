#!/usr/bin/env python3
"""
Improved team name matching with fuzzy matching and automatic learning.
"""

import requests
import json
from typing import Dict, Optional, List
import logging
from pathlib import Path
from difflib import SequenceMatcher

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
    'Greek Super League': 'gre.1',
}

# Cache file for learned team name mappings
CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEAM_MAPPING_CACHE = CACHE_DIR / "team_name_mappings.json"

# Load cached mappings
def load_cached_mappings() -> Dict[str, str]:
    """Load previously learned team name mappings."""
    if TEAM_MAPPING_CACHE.exists():
        try:
            with open(TEAM_MAPPING_CACHE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

# Save cached mappings
def save_cached_mappings(mappings: Dict[str, str]):
    """Save learned team name mappings."""
    try:
        with open(TEAM_MAPPING_CACHE, 'w') as f:
            json.dump(mappings, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save team mappings: {e}")

# Global cache
CACHED_TEAM_MAPPINGS = load_cached_mappings()


def normalize_team_name(name: str) -> str:
    """Normalize team name for better matching."""
    # Remove common suffixes/prefixes
    name = name.strip()
    
    # Remove common patterns
    patterns_to_remove = [
        'F.C.', 'FC', 'A.C.', 'AC', 'S.C.', 'SC',
        'CF', 'C.F.', 'United', 'City', 'Town',
        '04', '1899', '1900', '1893', '1909',  # Foundation years
    ]
    
    normalized = name
    for pattern in patterns_to_remove:
        # Remove as whole word
        normalized = normalized.replace(f' {pattern} ', ' ')
        normalized = normalized.replace(f' {pattern}', '')
        normalized = normalized.replace(f'{pattern} ', '')
    
    # Clean up extra spaces
    normalized = ' '.join(normalized.split())
    
    return normalized.strip()


def fuzzy_match_score(name1: str, name2: str) -> float:
    """Calculate fuzzy match score between two team names."""
    # Normalize both names
    norm1 = normalize_team_name(name1).lower()
    norm2 = normalize_team_name(name2).lower()
    
    # Direct match
    if norm1 == norm2:
        return 1.0
    
    # Check if one contains the other
    if norm1 in norm2 or norm2 in norm1:
        return 0.9
    
    # Use SequenceMatcher for fuzzy matching
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Bonus for matching key words
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    common_words = words1 & words2
    
    if common_words:
        word_bonus = len(common_words) / max(len(words1), len(words2)) * 0.3
        ratio += word_bonus
    
    return min(ratio, 1.0)


def fetch_espn_standings(league_code: str) -> Optional[Dict]:
    """Fetch standings for a league from ESPN API."""
    url = f'https://site.api.espn.com/apis/v2/sports/soccer/{league_code}/standings'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'children' not in data:
            return None
        
        standings_data = data['children'][0]['standings']['entries']
        
        # Build lookup dict
        standings = {}
        for entry in standings_data:
            team_name = entry['team']['displayName']
            
            # Extract position and points
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
        
        return standings
        
    except Exception as e:
        logger.debug(f"Failed to fetch standings for {league_code}: {e}")
        return None


def find_best_match(team_name: str, standings: Dict, threshold: float = 0.7) -> Optional[tuple]:
    """Find best matching team name in standings using fuzzy matching."""
    best_match = None
    best_score = threshold
    
    for espn_name in standings.keys():
        score = fuzzy_match_score(team_name, espn_name)
        
        if score > best_score:
            best_score = score
            best_match = espn_name
    
    if best_match:
        logger.info(f"Fuzzy matched '{team_name}' → '{best_match}' (score: {best_score:.2f})")
        return best_match, best_score
    
    return None


def get_team_standings(team_name: str, league_name: str = None) -> Optional[Dict]:
    """Get position and points for a team with fuzzy matching."""
    
    # Check cache first
    cache_key = f"{team_name}|{league_name or 'any'}"
    if cache_key in CACHED_TEAM_MAPPINGS:
        cached_espn_name = CACHED_TEAM_MAPPINGS[cache_key]
        logger.debug(f"Using cached mapping: {team_name} → {cached_espn_name}")
        # Still need to fetch standings to get current position/points
        team_name_to_lookup = cached_espn_name
    else:
        team_name_to_lookup = team_name
    
    # If league specified, try that first
    if league_name and league_name in ESPN_LEAGUE_CODES:
        league_code = ESPN_LEAGUE_CODES[league_name]
        standings = fetch_espn_standings(league_code)
        
        if standings:
            # Try exact match
            if team_name_to_lookup in standings:
                return standings[team_name_to_lookup]
            
            # Try fuzzy matching
            match_result = find_best_match(team_name, standings, threshold=0.7)
            if match_result:
                espn_name, score = match_result
                
                # Cache this mapping for future use
                CACHED_TEAM_MAPPINGS[cache_key] = espn_name
                save_cached_mappings(CACHED_TEAM_MAPPINGS)
                
                return standings[espn_name]
    
    # Try all leagues with fuzzy matching
    for league_name, league_code in ESPN_LEAGUE_CODES.items():
        standings = fetch_espn_standings(league_code)
        
        if standings:
            # Try exact match
            if team_name_to_lookup in standings:
                logger.info(f"Found {team_name} in {league_name}")
                return standings[team_name_to_lookup]
            
            # Try fuzzy matching
            match_result = find_best_match(team_name, standings, threshold=0.75)
            if match_result:
                espn_name, score = match_result
                
                # Cache this mapping
                cache_key = f"{team_name}|{league_name}"
                CACHED_TEAM_MAPPINGS[cache_key] = espn_name
                save_cached_mappings(CACHED_TEAM_MAPPINGS)
                
                logger.info(f"Found {team_name} as {espn_name} in {league_name}")
                return standings[espn_name]
    
    logger.warning(f"Could not find standings for team: {team_name}")
    return None


def test_fuzzy_matching():
    """Test fuzzy matching with various team names."""
    print("=" * 80)
    print("TESTING FUZZY TEAM NAME MATCHING")
    print("=" * 80)
    
    test_cases = [
        ('Bayer 04 Leverkusen', 'Bundesliga'),
        ('Olympiacos F.C.', 'Greek Super League'),
        ('FC Bayern München', 'Bundesliga'),
        ('Inter', 'Serie A'),
        ('Paris', 'Ligue 1'),
        ('Atlético Madrid', 'La Liga'),
    ]
    
    for team, league in test_cases:
        print(f"\n{team} ({league}):")
        result = get_team_standings(team, league)
        if result:
            print(f"  ✅ Position: {result['position']}, Points: {result['points']}")
        else:
            print(f"  ❌ NOT FOUND")
    
    # Show cached mappings
    print("\n" + "=" * 80)
    print("CACHED TEAM MAPPINGS")
    print("=" * 80)
    for key, value in CACHED_TEAM_MAPPINGS.items():
        print(f"  {key} → {value}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_fuzzy_matching()
