# Point-in-Time Training Data Strategy
## Building Historical Features for Model Training

**Created:** January 25, 2026  
**Critical Issue:** Training data must reflect point-in-time knowledge, not future information

---

## 🎯 The Problem

**Training Data Leakage:**
When building features for a match on **2024-10-15**, we can only use data available **before** that date. We cannot use:
- ❌ Live match statistics (not yet played)
- ❌ Future player ratings (not yet earned)
- ❌ Future injuries (not yet occurred)

**We need:** Historical snapshots of all features as they existed at prediction time.

---

## ✅ Feature Availability Analysis

### Category 1: Fully Historical (✅ Available)

These can be calculated from historical data:

| Feature Group | Count | Data Source | Historical? | Notes |
|---------------|-------|-------------|-------------|-------|
| **Elo Ratings** | 10 | Calculated from past results | ✅ YES | Reconstruct from match history |
| **League Position & Points** | 12 | API: `/standings/historical` | ✅ YES | Available for past dates |
| **Recent Form** | 15 | API: `/fixtures` (past matches) | ✅ YES | Filter by date |
| **H2H History** | 8 | API: `/fixtures/head-to-head` | ✅ YES | Historical matchups |
| **Derived xG** | 25 | Calculated from match stats | ✅ YES | Stats available post-match |
| **Shot Analysis** | 15 | API: `/statistics` (historical) | ✅ YES | Match statistics saved |
| **Defensive Metrics** | 12 | API: `/statistics` (historical) | ✅ YES | Tackles, interceptions saved |
| **Attack Patterns** | 8 | API: `/statistics` (historical) | ✅ YES | Attacks, dangerous attacks |
| **Momentum** | 12 | Calculated from form | ✅ YES | Derived from historical data |

**Total Fully Historical: ~117 features**

---

### Category 2: Partially Historical (⚠️ Requires Tracking)

These need prospective data collection:

| Feature Group | Count | Data Source | Historical? | Solution |
|---------------|-------|-------------|-------------|----------|
| **Player Lineup Quality** | 8 | API: `/lineups` | ⚠️ PARTIAL | Only if lineups were saved historically |
| **Key Player Availability** | 8 | API: `/sidelined` | ⚠️ PARTIAL | Need historical injury tracking |
| **Injuries & Suspensions** | 5 | API: `/sidelined` | ⚠️ PARTIAL | Need historical tracking |
| **Player Form** | 4 | API: `/players/statistics` | ⚠️ PARTIAL | Season stats available, not point-in-time |
| **Fixture-Adjusted** | 10 | Calculated | ✅ YES | Can calculate from Elo |
| **Situational Context** | 8 | Mixed | ✅ MOSTLY | Rest days calculable, derbies known |

**Total Partially Historical: ~43 features**

---

### Category 3: Not Historical (❌ Need Prospective Collection)

| Feature | Issue | Solution |
|---------|-------|----------|
| **Exact lineup 1h before** | Not saved historically | Use season averages for historical data |
| **Injury status at prediction time** | Not timestamped | Track going forward, use averages for past |
| **Player form last 3 matches** | Need point-in-time ratings | Use season averages for historical data |

---

## 🔧 Implementation Strategy

### Approach 1: Full Historical Reconstruction (140-150 features)

**What we CAN build from historical data:**

```python
def build_historical_training_data(start_date, end_date):
    """
    Build training data for matches between start_date and end_date.
    Uses only data available BEFORE each match.
    """
    
    matches = get_matches_between(start_date, end_date)
    training_data = []
    
    for match in matches:
        match_date = match.date
        
        # 1. Get historical standings (as of match_date - 1 day)
        standings = get_standings_at_date(match.league_id, match_date - 1)
        
        # 2. Calculate Elo ratings (reconstruct from all past matches)
        home_elo = calculate_elo_at_date(match.home_team_id, match_date)
        away_elo = calculate_elo_at_date(match.away_team_id, match_date)
        
        # 3. Get recent form (last 5 matches before match_date)
        home_form = get_form_before_date(match.home_team_id, match_date, n=5)
        away_form = get_form_before_date(match.away_team_id, match_date, n=5)
        
        # 4. Calculate derived xG from past matches
        home_xg_5 = calculate_avg_xg_before_date(match.home_team_id, match_date, n=5)
        away_xg_5 = calculate_avg_xg_before_date(match.away_team_id, match_date, n=5)
        
        # 5. Get H2H history (all matches before match_date)
        h2h = get_h2h_before_date(match.home_team_id, match.away_team_id, match_date)
        
        # ... build all 140-150 features
        
        features = {
            'match_id': match.id,
            'match_date': match_date,
            'home_elo': home_elo,
            'away_elo': away_elo,
            # ... all features
        }
        
        training_data.append(features)
    
    return pd.DataFrame(training_data)
```

---

### Approach 2: Hybrid (165-190 features with prospective tracking)

**For new data going forward, track everything:**

```python
# Database schema for prospective tracking
CREATE TABLE feature_snapshots (
    id SERIAL PRIMARY KEY,
    fixture_id INT,
    snapshot_time TIMESTAMP,  -- When features were calculated
    match_time TIMESTAMP,      -- Actual match time
    
    -- All 165-190 features stored as JSON or columns
    features JSONB,
    
    -- Metadata
    lineup_available BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

# Daily feature calculation (run 2 hours before each match)
def calculate_and_store_features_daily():
    """
    Calculate features for upcoming matches and store snapshot.
    This creates point-in-time training data for future use.
    """
    upcoming_matches = get_matches_next_24_hours()
    
    for match in upcoming_matches:
        # Calculate all features using only data available NOW
        features = calculate_all_features(match)
        
        # Store snapshot
        store_feature_snapshot(
            fixture_id=match.id,
            snapshot_time=datetime.now(),
            match_time=match.start_time,
            features=features,
            lineup_available=check_lineup_available(match.id)
        )
```

---

## 📊 SportMonks API Historical Capabilities

### ✅ What IS Available Historically

```python
# 1. Historical match results and statistics
GET /v3/football/fixtures/between/{start_date}/{end_date}
# Returns: All matches in date range with results

GET /v3/football/statistics/fixtures/{fixture_id}
# Returns: Full match statistics (shots, possession, etc.)
# Available for ANY past match

# 2. Historical standings
GET /v3/football/standings/seasons/{season_id}
# Returns: Final standings for completed season
# Note: Point-in-time standings need to be calculated

# 3. Historical H2H
GET /v3/football/fixtures/head-to-head/{team1}/{team2}
# Returns: All historical matchups
# Can filter by date

# 4. Player season statistics
GET /v3/football/players/{player_id}/statistics/seasons/{season_id}
# Returns: Aggregated season stats
# Note: Not point-in-time, but season totals
```

### ❌ What is NOT Available Historically

```python
# 1. Point-in-time standings
# API only gives final season standings
# Solution: Calculate from match results

# 2. Historical lineups with timestamps
# Lineups available, but not "when they were announced"
# Solution: Assume lineups known 1h before for historical data

# 3. Historical injury/suspension status at specific dates
# Current sidelined players available, not historical
# Solution: Use season averages for historical training data

# 4. Point-in-time player ratings
# Season aggregates available, not match-by-match historical
# Solution: Use season averages or track going forward
```

---

## 🗄️ Database Schema for Historical Features

```sql
-- Store historical Elo ratings
CREATE TABLE elo_history (
    id SERIAL PRIMARY KEY,
    team_id INT,
    season_id INT,
    match_date DATE,
    elo_rating FLOAT,
    elo_change FLOAT,
    INDEX idx_team_date (team_id, match_date)
);

-- Store historical standings (calculated)
CREATE TABLE standings_history (
    id SERIAL PRIMARY KEY,
    team_id INT,
    season_id INT,
    as_of_date DATE,
    league_position INT,
    points INT,
    matches_played INT,
    INDEX idx_team_date (team_id, as_of_date)
);

-- Store historical derived xG
CREATE TABLE xg_history (
    id SERIAL PRIMARY KEY,
    fixture_id INT,
    team_id INT,
    match_date DATE,
    derived_xg FLOAT,
    derived_xga FLOAT,
    shots_insidebox INT,
    shots_outsidebox INT,
    big_chances INT,
    INDEX idx_team_date (team_id, match_date)
);

-- Store complete feature vectors (for training)
CREATE TABLE training_features (
    id SERIAL PRIMARY KEY,
    fixture_id INT,
    match_date DATE,
    features JSONB,  -- All 140-190 features
    target VARCHAR(10),  -- 'H', 'D', 'A'
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔄 Historical Data Collection Workflow

### Phase 1: Backfill Historical Data (One-time)

```python
def backfill_historical_features(seasons=['2022-2023', '2023-2024', '2024-2025']):
    """
    Build training data for past seasons.
    """
    
    for season in seasons:
        print(f"Processing season: {season}")
        
        # 1. Get all matches for season
        matches = api.get_season_fixtures(season)
        
        # 2. For each match, calculate features using only prior data
        for match in tqdm(matches):
            # Skip if already processed
            if feature_exists(match.id):
                continue
            
            # Calculate features as of match_date - 1 hour
            features = calculate_historical_features(
                match=match,
                as_of_date=match.date - timedelta(hours=1)
            )
            
            # Store
            store_training_features(
                fixture_id=match.id,
                match_date=match.date,
                features=features,
                target=match.result
            )
            
        print(f"Completed {season}: {len(matches)} matches")
```

### Phase 2: Daily Updates (Ongoing)

```python
def daily_feature_collection():
    """
    Run daily to collect features for upcoming matches.
    Creates point-in-time snapshots for future training.
    """
    
    # Get matches in next 24-48 hours
    upcoming = api.get_upcoming_fixtures(hours=48)
    
    for match in upcoming:
        # Calculate features NOW (point-in-time)
        features = calculate_all_features(
            match=match,
            include_lineups=True,  # If available
            include_injuries=True   # Current status
        )
        
        # Store snapshot
        store_feature_snapshot(
            fixture_id=match.id,
            snapshot_time=datetime.now(),
            features=features
        )
    
    print(f"Collected features for {len(upcoming)} upcoming matches")
```

---

## ✅ Recommended Approach

### For Historical Training Data (2022-2025)

**Use 140-150 features that can be fully reconstructed:**

1. ✅ Elo Ratings (10) - Reconstruct from match history
2. ✅ League Position & Points (12) - Calculate from results
3. ✅ Recent Form (15) - From past fixtures
4. ✅ H2H (8) - Historical matchups
5. ✅ Derived xG (25) - From match statistics
6. ✅ Shot Analysis (15) - From match statistics
7. ✅ Defensive Metrics (12) - From match statistics
8. ✅ Attack Patterns (8) - From match statistics
9. ✅ Momentum (12) - From form trends
10. ✅ Fixture-Adjusted (10) - From Elo
11. ✅ Context (8) - Rest days, position pressure
12. ⚠️ Player Features (15) - Use season averages as proxy

**Total: ~140-150 features with high quality historical data**

### For Prospective Data (2026+)

**Track all 165-190 features going forward:**

1. All 140-150 historical features
2. + Full 25 player features (lineups, injuries, form)
3. Store daily snapshots before each match

---

## 📈 Expected Impact

| Approach | Features | Historical Data | Prospective Data | Model Quality |
|----------|----------|----------------|------------------|---------------|
| **Minimal** | 100 | ✅ Easy | ✅ Easy | 70% |
| **Recommended** | 140-150 | ✅ Feasible | ✅ Good | 90% |
| **Full** | 165-190 | ⚠️ Partial | ✅ Excellent | 100% |

**Recommendation:** Start with 140-150 features for historical training, expand to 165-190 for live predictions.

---

## 🚀 Implementation Priority

1. **Week 1-2:** Build historical feature reconstruction (140-150 features)
2. **Week 3:** Backfill 2022-2025 seasons
3. **Week 4:** Set up daily prospective tracking (165-190 features)
4. **Week 5+:** Train model on historical data, use full features for live predictions

---

**Bottom Line:** We CAN build high-quality training data from historical API data for 140-150 features. Player features need prospective tracking for full accuracy.
