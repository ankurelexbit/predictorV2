# SportMonks API Data Mapping
## Complete Reference for Data Sources

**Version:** 1.0  
**Last Updated:** January 25, 2026

---

## 📊 Available Statistics from SportMonks API

### Core Match Statistics

| Statistic | Type ID | Endpoint | Description | Used For |
|-----------|---------|----------|-------------|----------|
| **Shots Total** | 42 | `/statistics` | Total shots attempted | xG calculation, shot analysis |
| **Shots On Target** | - | `/statistics` | Shots on target | xG accuracy multiplier |
| **Shots Inside Box** | 49 | `/statistics` | Shots from inside penalty box | Derived xG (0.12 per shot) |
| **Shots Outside Box** | 50 | `/statistics` | Shots from outside box | Derived xG (0.03 per shot) |
| **Big Chances Created** | 580 | `/statistics` | Clear goal opportunities | Derived xG (0.35 per chance) |
| **Big Chances Missed** | 581 | `/statistics` | Big chances not converted | Finishing efficiency |
| **Corners** | 34 | `/statistics` | Corner kicks | Set piece xG (0.03 per corner) |
| **Attacks** | - | `/statistics` | Total attacks | Attack volume |
| **Dangerous Attacks** | - | `/statistics` | Dangerous attacks | Attack quality |
| **Possession** | - | `/statistics` | Ball possession % | Possession features |
| **Passes** | 80 | `/statistics` | Total passes | Passing features |
| **Accurate Passes** | 81 | `/statistics` | Successful passes | Pass accuracy |
| **Tackles** | 78 | `/statistics` | Tackles attempted | PPDA calculation |
| **Interceptions** | 100 | `/statistics` | Passes intercepted | PPDA calculation |
| **Clearances** | 101 | `/statistics` | Defensive clearances | Defensive actions |
| **Fouls** | 56 | `/statistics` | Fouls committed | Discipline |
| **Yellow Cards** | 84 | `/statistics` | Yellow cards | Discipline |
| **Red Cards** | 83 | `/statistics` | Red cards | Discipline |

### Player Statistics

| Statistic | Type ID | Endpoint | Description | Used For |
|-----------|---------|----------|-------------|----------|
| **Player Rating** | 118 | `/players/statistics` | Match rating (0-10) | Lineup quality |
| **Minutes Played** | 119 | `/players/statistics` | Time on pitch | Player availability |
| **Key Passes** | 117 | `/players/statistics` | Passes leading to shots | Creative output |
| **Dribbles** | 108 | `/players/statistics` | Dribbles attempted | Ball progression |
| **Successful Dribbles** | 109 | `/players/statistics` | Completed dribbles | Dribbling efficiency |

---

## 🔧 API Endpoints Reference

### 1. Fixtures & Results
```
GET /v3/football/fixtures
GET /v3/football/fixtures/{id}
GET /v3/football/fixtures/between/{start}/{end}
```

**Returns:** Match details, scores, dates, teams

**Used For:**
- Recent form calculation
- H2H history
- Match scheduling

### 2. Statistics
```
GET /v3/football/statistics/fixtures/{fixture_id}
```

**Returns:** Match statistics (shots, possession, etc.)

**Used For:**
- Derived xG calculation
- Shot analysis
- Defensive metrics
- Attack patterns

### 3. Standings
```
GET /v3/football/standings/seasons/{season_id}
GET /v3/football/standings/live/leagues/{league_id}
```

**Returns:** League table, points, position, form

**Used For:**
- League position features
- Points features
- Season context

### 4. Head-to-Head
```
GET /v3/football/fixtures/head-to-head/{team1}/{team2}
```

**Returns:** Historical matchups

**Used For:**
- H2H win/draw/loss records
- H2H goals
- H2H patterns

### 5. Player Statistics
```
GET /v3/football/players/{player_id}/statistics
GET /v3/football/lineups/fixtures/{fixture_id}
```

**Returns:** Player performance data, lineups

**Used For:**
- Lineup quality
- Player ratings
- Availability

---

## 💾 Database Schema

### Tables Required

#### 1. `matches`
```sql
CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    fixture_id INT UNIQUE,
    league_id INT,
    season_id INT,
    home_team_id INT,
    away_team_id INT,
    match_date TIMESTAMP,
    home_goals INT,
    away_goals INT,
    result VARCHAR(10), -- 'H', 'D', 'A'
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 2. `match_statistics`
```sql
CREATE TABLE match_statistics (
    id SERIAL PRIMARY KEY,
    fixture_id INT REFERENCES matches(fixture_id),
    team_id INT,
    shots_total INT,
    shots_on_target INT,
    shots_insidebox INT,
    shots_outsidebox INT,
    big_chances_created INT,
    corners INT,
    attacks INT,
    dangerous_attacks INT,
    possession FLOAT,
    passes INT,
    tackles INT,
    interceptions INT,
    clearances INT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 3. `elo_ratings`
```sql
CREATE TABLE elo_ratings (
    id SERIAL PRIMARY KEY,
    team_id INT,
    season_id INT,
    match_date TIMESTAMP,
    elo_rating FLOAT,
    elo_change FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 4. `derived_features`
```sql
CREATE TABLE derived_features (
    id SERIAL PRIMARY KEY,
    fixture_id INT REFERENCES matches(fixture_id),
    team_id INT,
    derived_xg FLOAT,
    derived_xga FLOAT,
    derived_xgd FLOAT,
    points_last_5 INT,
    goals_scored_last_5 INT,
    -- ... all 150-180 features
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔄 Data Collection Workflow

### Daily Update Process

```python
# 1. Fetch recent fixtures
fixtures = api.get_fixtures_between(yesterday, today)

# 2. For each fixture, get statistics
for fixture in fixtures:
    stats = api.get_fixture_statistics(fixture.id)
    store_match_statistics(stats)
    
# 3. Update Elo ratings
for fixture in completed_fixtures:
    update_elo_ratings(fixture)
    
# 4. Calculate derived features
for fixture in upcoming_fixtures:
    features = calculate_all_features(fixture)
    store_derived_features(features)
```

---

## 📈 Feature Calculation Pipeline

### Step-by-Step Process

```
1. RAW DATA (API)
   ↓
2. MATCH STATISTICS (Database)
   ↓
3. ELO RATINGS (Calculated)
   ↓
4. DERIVED xG (Calculated from stats)
   ↓
5. ROLLING FEATURES (Last 3/5/10 matches)
   ↓
6. MOMENTUM FEATURES (Trends, streaks)
   ↓
7. FIXTURE-ADJUSTED (Opponent strength)
   ↓
8. FINAL FEATURE VECTOR (150-180 features)
```

---

**See FEATURE_DICTIONARY.md for complete feature list**
