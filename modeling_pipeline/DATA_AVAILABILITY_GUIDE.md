# Pre-Match Data Availability & Sources Guide

**Critical Question:** What data is available BEFORE a match starts, and where do we get it?

---

## TL;DR - Data Availability Timeline

| Data Type | Available When | Source | Update Frequency |
|-----------|---------------|--------|------------------|
| **Elo Ratings** | ✅ Pre-match | Your database | After each match |
| **Form/Rolling Stats** | ✅ Pre-match | Your database | After each match |
| **Standings** | ✅ Pre-match | Sportmonks API | After each match |
| **H2H History** | ✅ Pre-match | Your database | Static (historical) |
| **Injuries** | ✅ Pre-match (~24h before) | Sportmonks API | Daily |
| **Odds** | ✅ Pre-match (live updates) | Sportmonks/Odds APIs | Real-time |
| **Match Result** | ❌ POST-match only | N/A | After match ends |
| **Actual Goals** | ❌ POST-match only | N/A | After match ends |

---

## Feature Availability Breakdown

### ✅ Available PRE-Match (Predictive Features)

#### 1. **Elo Ratings** (3 features)
```
home_elo, away_elo, elo_diff
```

**When available:** Immediately after last match
**Source:** Your own calculation/database
**How to get:**
```python
# Maintain Elo ratings in your database
# Update after each match completes
elo_ratings = {
    'Liverpool': 1850,
    'Man City': 1920,
    # ... all teams
}

home_elo = elo_ratings['Liverpool']
away_elo = elo_ratings['Man City']
elo_diff = home_elo - away_elo
```

**Data source:** Self-maintained (calculate from match results)

---

#### 2. **Form Features** (18 features)
```
home_wins_3, home_draws_3, home_losses_3, home_form_3
away_wins_3, away_draws_3, away_losses_3, away_form_3
(same for 5-game and 10-game windows)
```

**When available:** After each match (within hours)
**Source:** Your database of completed matches
**How to get:**
```python
# Query last 5 matches for Liverpool
last_5_matches = db.query("""
    SELECT result FROM matches
    WHERE (home_team = 'Liverpool' OR away_team = 'Liverpool')
    AND date < '2024-01-20'  -- upcoming match date
    ORDER BY date DESC
    LIMIT 5
""")

# Calculate form
wins = count_wins(last_5_matches)
draws = count_draws(last_5_matches)
form = (wins * 3 + draws * 1) / 5
```

**Data source:**
- **Your database** (matches you've already collected)
- **Sportmonks API:** `/fixtures` endpoint (historical results)

---

#### 3. **Rolling Statistics** (390 features)
```
All rolling stats: goals_3, xg_5, shots_10, possession_3, etc.
```

**When available:** After each match (12-24 hours for full stats)
**Source:** Sportmonks API + your database
**How to get:**

```python
# Get last 5 completed matches with full statistics
matches = sportmonks_api.get_fixtures(
    team_id=liverpool_id,
    filters=f"fixtureSeasons:{current_season}",
    include="statistics,lineups.details"  # Full match data
)

# Calculate rolling averages
home_goals_5 = mean([m['home_goals'] for m in last_5_home_matches])
home_xg_5 = mean([m['home_xg'] for m in last_5_home_matches])
home_possession_5 = mean([m['home_possession'] for m in last_5_matches])
# ... etc for all 390 rolling features
```

**Data sources:**
- **Sportmonks API:** `/fixtures` with `include=statistics,lineups.details`
- **Your database:** Pre-calculated rolling features (recommended)

**Lag time:** 12-24 hours after match (wait for official stats)

---

#### 4. **Standings** (6 features)
```
home_position, away_position, position_diff
home_points, away_points, points_diff
```

**When available:** Updated within hours after each match
**Source:** Sportmonks API or official league sites
**How to get:**

```python
# Sportmonks API - Standings endpoint
standings = sportmonks_api.get_standings_by_season(season_id)

liverpool_standing = next(s for s in standings if s['team_id'] == liverpool_id)
man_city_standing = next(s for s in standings if s['team_id'] == man_city_id)

home_position = liverpool_standing['position']  # e.g., 2
home_points = liverpool_standing['points']      # e.g., 65
away_position = man_city_standing['position']   # e.g., 1
away_points = man_city_standing['points']       # e.g., 70

position_diff = home_position - away_position   # 2 - 1 = 1
points_diff = home_points - away_points         # 65 - 70 = -5
```

**Data sources:**
- **Sportmonks API:** `/standings` endpoint
- **Official sites:** premierleague.com, laliga.com (scraping)
- **Football-Data.org:** Free API with standings

---

#### 5. **Head-to-Head (H2H)** (5 features)
```
h2h_home_wins, h2h_away_wins, h2h_draws
h2h_home_goals_avg, h2h_away_goals_avg
```

**When available:** Always (historical data)
**Source:** Your database or Sportmonks API
**How to get:**

```python
# Query historical matchups
h2h_matches = db.query("""
    SELECT * FROM matches
    WHERE (home_team = 'Liverpool' AND away_team = 'Man City')
       OR (home_team = 'Man City' AND away_team = 'Liverpool')
    ORDER BY date DESC
    LIMIT 10
""")

h2h_home_wins = count_home_wins(h2h_matches, 'Liverpool')
h2h_away_wins = count_away_wins(h2h_matches, 'Man City')
h2h_draws = count_draws(h2h_matches)
```

**Data sources:**
- **Your database:** Historical matches
- **Sportmonks API:** `/head-to-head/{team1}/{team2}`

---

#### 6. **Injuries/Suspensions** (3 features)
```
home_injuries, away_injuries, injury_diff
```

**When available:** 24-48 hours before match (team news)
**Source:** Sportmonks API or team announcements
**How to get:**

```python
# Sportmonks - Sidelined endpoint
sidelined = sportmonks_api.get_sidelined(
    team_id=liverpool_id,
    date=match_date
)

home_injuries = len([p for p in sidelined if p['type'] in ['injury', 'suspended']])
```

**Data sources:**
- **Sportmonks API:** `/sidelined` endpoint
- **Official team sites:** Injury news (scraping)
- **Third-party:** FantasyPremierLeague API, Transfermarkt

**Update frequency:** Daily, with official team news 24-48h before match

---

#### 7. **Market Odds** (8 features)
```
odds_home, odds_draw, odds_away
market_prob_home, market_prob_draw, market_prob_away
market_home_away_ratio, market_favorite
```

**When available:** Days/weeks before match, updated until kickoff
**Source:** Betting APIs
**How to get:**

```python
# Sportmonks - Odds endpoint (premium feature)
odds = sportmonks_api.get_odds(
    fixture_id=fixture_id,
    market_id=1  # 1x2 market (home/draw/away)
)

odds_home = odds[0]['home']  # e.g., 2.10
odds_draw = odds[0]['draw']  # e.g., 3.40
odds_away = odds[0]['away']  # e.g., 3.50

# Calculate implied probabilities
market_prob_home = 1 / odds_home
market_prob_draw = 1 / odds_draw
market_prob_away = 1 / odds_away

# Normalize to sum to 1.0
total = market_prob_home + market_prob_draw + market_prob_away
market_prob_home /= total
market_prob_draw /= total
market_prob_away /= total
```

**Data sources:**
- **Sportmonks API:** `/odds` endpoint (premium subscription)
- **The Odds API:** https://the-odds-api.com (free tier: 500 requests/month)
- **Betfair API:** Real-time exchange odds
- **Oddschecker:** Scraping (be careful with ToS)
- **RapidAPI:** Various odds providers

**Update frequency:** Real-time until kickoff

---

#### 8. **Contextual Features** (5 features)
```
round_num, season_progress, is_early_season, day_of_week, is_weekend
```

**When available:** Always (calculated from match date)
**Source:** Your code (calculated)
**How to get:**

```python
from datetime import datetime

match_date = datetime(2024, 1, 20)  # Upcoming match
season_start = datetime(2023, 8, 15)

# Calculate
day_of_week = match_date.weekday()  # 0=Monday, 6=Sunday
is_weekend = day_of_week in [5, 6]  # Saturday or Sunday

# If you have season info
round_num = 21  # From fixture data
days_since_start = (match_date - season_start).days
season_progress = days_since_start / 280  # ~280 days in season
is_early_season = round_num <= 10
```

**Data sources:** Self-calculated from match metadata

---

### ❌ NOT Available PRE-Match (Target Variables)

#### Match Results & Goals
```
target, home_win, draw, away_win
home_goals, away_goals
```

**When available:** ONLY after match ends
**Purpose:** Training targets, not features for prediction
**Usage:** These are what you're PREDICTING, not inputs

---

## Data Sources Reference

### Primary Source: Sportmonks API

**Endpoint Structure:**
```
Base URL: https://api.sportmonks.com/v3/football

Key Endpoints:
├── /fixtures                    # Match data
│   └── ?include=statistics      # Team stats
│   └── ?include=lineups.details # Player stats
├── /standings                   # League tables
├── /odds                        # Betting odds (premium)
├── /sidelined                   # Injuries/suspensions
└── /head-to-head/{t1}/{t2}     # H2H history
```

**Cost:**
- Free tier: 180 requests/minute
- Premium: $19-99/month for odds & advanced stats

**What you get:**
```python
import requests

API_KEY = "your_key"
headers = {"Authorization": API_KEY}

# Get upcoming fixture with pre-match data
response = requests.get(
    "https://api.sportmonks.com/v3/football/fixtures/18535517",
    headers=headers,
    params={
        "include": "participants,statistics,odds,sidelined"
    }
)

fixture = response.json()['data']

# Available pre-match:
home_team_id = fixture['participants'][0]['id']
away_team_id = fixture['participants'][1]['id']
match_date = fixture['starting_at']
odds = fixture.get('odds', [])
injuries = fixture.get('sidelined', [])
```

---

### Alternative Free Sources

#### 1. **Football-Data.org API**
- **URL:** https://www.football-data.org/
- **Free tier:** 10 requests/minute
- **Data:** Fixtures, standings, basic stats
- **No odds:** Free tier doesn't include betting data

#### 2. **API-Football (RapidAPI)**
- **URL:** https://rapidapi.com/api-football/
- **Free tier:** 100 requests/day
- **Data:** Comprehensive (fixtures, stats, odds, injuries)

#### 3. **The Odds API**
- **URL:** https://the-odds-api.com/
- **Free tier:** 500 requests/month
- **Data:** Betting odds only (but comprehensive)
- **Perfect for:** Getting pre-match odds

#### 4. **Official League APIs**
- **Premier League:** Limited official API
- **La Liga:** No public API
- **Bundesliga:** OpenLigaDB (free, German leagues)

---

## Pre-Match Prediction Pipeline

### Step 1: Identify Upcoming Match
```python
# From Sportmonks or manual input
upcoming_match = {
    'fixture_id': 18535517,
    'date': '2024-01-20 15:00:00',
    'home_team': 'Liverpool',
    'away_team': 'Manchester City',
    'season_id': 23614
}
```

### Step 2: Gather Pre-Match Data

#### A. From Your Database (Instant)
```python
# Elo ratings (you maintain these)
home_elo = db.get_elo('Liverpool')        # Your DB
away_elo = db.get_elo('Manchester City')

# Form & rolling stats (pre-calculated)
home_form_5 = db.get_rolling_stat('Liverpool', 'form', window=5)
home_goals_5 = db.get_rolling_stat('Liverpool', 'goals', window=5)
# ... all 390 rolling features

# H2H history (historical)
h2h = db.get_h2h('Liverpool', 'Manchester City', limit=10)
```

**Recommended:** Pre-calculate and store rolling features in your database.

#### B. From Sportmonks API (Real-time)
```python
# Current standings (updates after each match)
standings = api.get_standings(season_id=23614)
home_position = get_position(standings, 'Liverpool')
home_points = get_points(standings, 'Liverpool')

# Injuries (updated daily, critical 24-48h before match)
injuries_home = api.get_sidelined(team_id=liverpool_id, date=match_date)
injuries_away = api.get_sidelined(team_id=man_city_id, date=match_date)

# Odds (real-time until kickoff)
odds = api.get_odds(fixture_id=18535517)
```

#### C. Calculate Contextual Features
```python
# From match date
day_of_week = parse_date(match_date).weekday()
is_weekend = day_of_week >= 5
```

### Step 3: Construct Feature Vector
```python
# Build complete feature vector (465 features)
features = {
    'fixture_id': 18535517,
    'date': match_date,
    'home_team_name': 'Liverpool',
    'away_team_name': 'Manchester City',

    # Elo
    'home_elo': home_elo,
    'away_elo': away_elo,
    'elo_diff': home_elo - away_elo,

    # Form (from your DB)
    'home_form_5': home_form_5,
    'away_form_5': away_form_5,

    # Rolling stats (from your DB) - 390 features
    'home_goals_5': home_goals_5,
    'home_xg_5': home_xg_5,
    # ... all rolling features

    # Standings (from API)
    'home_position': home_position,
    'away_position': away_position,
    'position_diff': home_position - away_position,
    'home_points': home_points,
    'away_points': away_points,
    'points_diff': home_points - away_points,

    # H2H (from your DB)
    'h2h_home_wins': h2h['home_wins'],
    'h2h_draws': h2h['draws'],

    # Injuries (from API)
    'home_injuries': len(injuries_home),
    'away_injuries': len(injuries_away),

    # Odds (from API)
    'odds_home': odds['home'],
    'odds_draw': odds['draw'],
    'odds_away': odds['away'],

    # Contextual
    'day_of_week': day_of_week,
    'is_weekend': is_weekend,
}

# Convert to DataFrame
import pandas as pd
X = pd.DataFrame([features])
```

### Step 4: Make Prediction
```python
import joblib

# Load trained model
model = joblib.load('models/xgboost_model.joblib')

# Predict
probabilities = model.predict_proba(X)

print(f"Liverpool vs Man City predictions:")
print(f"  Home win: {probabilities[0][2]:.1%}")
print(f"  Draw:     {probabilities[0][1]:.1%}")
print(f"  Away win: {probabilities[0][0]:.1%}")
```

---

## Data Freshness Requirements

| Feature Group | Max Age | Critical? | Update Method |
|---------------|---------|-----------|---------------|
| **Elo ratings** | After last match | Yes | Your DB (update after match) |
| **Rolling stats** | 1-2 days | Yes | Your DB (update after match) |
| **Standings** | 1 day | Yes | API call or your DB |
| **Injuries** | 1 day | Medium | Daily API call |
| **Odds** | Minutes | Medium | Real-time API |
| **H2H** | Months | Low | Static (rarely changes) |

**Critical path for predictions:**
1. Maintain Elo ratings (self-managed)
2. Pre-calculate rolling features (self-managed)
3. Fetch current standings (API or cache)
4. Get latest injuries (API, 24h before match)
5. Fetch odds (API, close to kickoff)

---

## Recommended Architecture

### Database Schema
```sql
-- Store calculated features
CREATE TABLE team_features (
    team_id INT,
    date DATE,
    elo_rating FLOAT,
    form_5 FLOAT,
    goals_5 FLOAT,
    xg_5 FLOAT,
    -- ... all rolling features
    updated_at TIMESTAMP
);

-- Store matches
CREATE TABLE matches (
    fixture_id INT PRIMARY KEY,
    date TIMESTAMP,
    home_team_id INT,
    away_team_id INT,
    home_goals INT,
    away_goals INT,
    -- ... result data
);

-- Store injuries (cache)
CREATE TABLE injuries (
    team_id INT,
    date DATE,
    player_name VARCHAR,
    type VARCHAR,  -- 'injury' or 'suspended'
);

-- Store odds (cache)
CREATE TABLE odds (
    fixture_id INT,
    timestamp TIMESTAMP,
    odds_home FLOAT,
    odds_draw FLOAT,
    odds_away FLOAT,
);
```

### Update Schedule
```python
# Daily cron job (00:00 UTC)
def daily_update():
    # 1. Update completed matches
    new_matches = sportmonks_api.get_fixtures(date=yesterday)
    db.insert_matches(new_matches)

    # 2. Recalculate Elo ratings
    update_elo_ratings(new_matches)

    # 3. Recalculate rolling features
    update_rolling_features(new_matches)

    # 4. Update standings
    standings = sportmonks_api.get_standings(current_season)
    db.update_standings(standings)

    # 5. Update injuries
    injuries = sportmonks_api.get_sidelined(date=today)
    db.update_injuries(injuries)

# Pre-match (1-2 hours before kickoff)
def pre_match_update(fixture_id):
    # Get latest odds
    odds = sportmonks_api.get_odds(fixture_id)
    db.update_odds(fixture_id, odds)

    # Get latest injury news
    injuries = sportmonks_api.get_sidelined(fixture_id)
    db.update_injuries_for_match(fixture_id, injuries)
```

---

## Cost Estimate

### Sportmonks API
- **Free tier:** 180 req/min (sufficient for development)
- **Hobby:** $19/month (includes odds)
- **Professional:** $49/month (higher limits)

**Usage estimate:**
- Daily updates: ~50 requests/day
- Pre-match predictions: ~5 requests/match
- **Total for Premier League season:** ~4,000 requests (well within free tier)

### The Odds API
- **Free:** 500 requests/month
- **Paid:** $10/month for 10,000 requests

**Usage estimate:**
- 1 request per upcoming match
- **Total for Premier League season:** ~380 requests (free tier OK)

### Total Monthly Cost
- **Development:** $0 (free tiers)
- **Production:** $19-29/month (Sportmonks Hobby + odds backup)

---

## Quick Start Example

```python
from sportmonks import SportmonksAPI
from database import Database
import joblib

# Initialize
api = SportmonksAPI(api_key="your_key")
db = Database()
model = joblib.load('models/xgboost_model.joblib')

# Predict upcoming match
def predict_match(home_team, away_team, match_date):
    # 1. Get pre-calculated features from your DB
    home_features = db.get_team_features(home_team, as_of_date=match_date)
    away_features = db.get_team_features(away_team, as_of_date=match_date)

    # 2. Get fresh standings from API (or cache)
    standings = api.get_standings(current_season_id)
    home_standing = get_team_standing(standings, home_team)
    away_standing = get_team_standing(standings, away_team)

    # 3. Get injuries from API
    home_injuries = api.get_sidelined(home_team, match_date)
    away_injuries = api.get_sidelined(away_team, match_date)

    # 4. Get odds from API
    odds = api.get_odds_for_match(home_team, away_team, match_date)

    # 5. Build feature vector
    X = build_feature_vector(
        home_features, away_features,
        home_standing, away_standing,
        home_injuries, away_injuries,
        odds, match_date
    )

    # 6. Predict
    probs = model.predict_proba(X)[0]

    return {
        'home_win': probs[2],
        'draw': probs[1],
        'away_win': probs[0]
    }

# Example usage
predictions = predict_match('Liverpool', 'Manchester City', '2024-01-20')
print(predictions)
# {'home_win': 0.42, 'draw': 0.28, 'away_win': 0.30}
```

---

## Summary

### ✅ What IS Available Pre-Match
1. **Elo ratings** - Your database
2. **Form & rolling stats** - Your database (pre-calculated)
3. **Standings** - Sportmonks API or cache
4. **H2H history** - Your database
5. **Injuries** - Sportmonks API (24-48h before)
6. **Odds** - Sportmonks/Odds API (real-time)
7. **Contextual** - Calculated from match date

### ❌ What is NOT Available Pre-Match
1. **Match result** (target variable)
2. **Actual goals scored** (what you're predicting)

### Data Sources
- **Primary:** Sportmonks API ($0-19/month)
- **Odds:** The Odds API ($0-10/month)
- **Self-maintained:** Elo ratings, rolling features
- **Cache:** Standings, injuries (update daily)

### Critical Success Factors
1. **Maintain your own database** - Don't recalculate everything at prediction time
2. **Pre-calculate rolling features** - Update after each match
3. **Cache standings** - Update daily, not per prediction
4. **Get odds close to kickoff** - They're most accurate 1-2 hours before
5. **Update injuries regularly** - Official team news 24-48h before match

---

**Next step:** Set up your database schema and start collecting historical data!
