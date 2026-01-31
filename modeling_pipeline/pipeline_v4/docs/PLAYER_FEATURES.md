# Player & Lineup Features - Extended
## Key Player Impact & Availability

**Last Updated:** January 25, 2026  
**Total Features:** 25 (expanded from 10)

---

## 🎯 Why Player Features Matter

**Impact:** Key player absence can swing match probability by 5-15%
- Missing star striker: -10% win probability
- Missing key defender: +0.3 xGA per match
- Missing playmaker: -0.2 xG per match

---

## 📊 Player Features (25 total)

### Group 1: Lineup Quality (8 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_lineup_avg_rating_5` | API: `/lineups` + `/players/statistics` | Avg player rating last 5 | Float | Overall team quality | ⭐⭐⭐⭐⭐ |
| `away_lineup_avg_rating_5` | API: `/lineups` + `/players/statistics` | Avg player rating last 5 | Float | Overall team quality | ⭐⭐⭐⭐⭐ |
| `home_starting_11_avg_rating` | Calculated | Avg rating of starting 11 | Float | Lineup strength | ⭐⭐⭐⭐⭐ |
| `away_starting_11_avg_rating` | Calculated | Avg rating of starting 11 | Float | Lineup strength | ⭐⭐⭐⭐⭐ |
| `home_top_3_players_rating` | Calculated | Avg of 3 best players | Float | Star player quality | ⭐⭐⭐⭐ |
| `away_top_3_players_rating` | Calculated | Avg of 3 best players | Float | Star player quality | ⭐⭐⭐⭐ |
| `home_bench_strength` | Calculated | Avg bench player rating | Float | Squad depth | ⭐⭐⭐ |
| `away_bench_strength` | Calculated | Avg bench player rating | Float | Squad depth | ⭐⭐⭐ |

### Group 2: Key Player Availability (8 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_key_players_available` | API: `/sidelined` | Count of top 5 players available | Int | Star availability (0-5) | ⭐⭐⭐⭐⭐ |
| `away_key_players_available` | API: `/sidelined` | Count of top 5 players available | Int | Star availability (0-5) | ⭐⭐⭐⭐⭐ |
| `home_top_scorer_available` | API: `/sidelined` | Binary: top scorer playing | Bool | Main striker available | ⭐⭐⭐⭐⭐ |
| `away_top_scorer_available` | API: `/sidelined` | Binary: top scorer playing | Bool | Main striker available | ⭐⭐⭐⭐⭐ |
| `home_top_assister_available` | API: `/sidelined` | Binary: top assister playing | Bool | Playmaker available | ⭐⭐⭐⭐ |
| `away_top_assister_available` | API: `/sidelined` | Binary: top assister playing | Bool | Playmaker available | ⭐⭐⭐⭐ |
| `home_gk_quality` | Calculated | Goalkeeper rating | Float | GK strength | ⭐⭐⭐⭐ |
| `away_gk_quality` | Calculated | Goalkeeper rating | Float | GK strength | ⭐⭐⭐⭐ |

### Group 3: Injuries & Suspensions (5 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_players_injured` | API: `/sidelined` | Count injured players | Int | Injury crisis indicator | ⭐⭐⭐⭐ |
| `away_players_injured` | API: `/sidelined` | Count injured players | Int | Injury crisis indicator | ⭐⭐⭐⭐ |
| `home_players_suspended` | API: `/sidelined` | Count suspended players | Int | Suspension impact | ⭐⭐⭐⭐ |
| `away_players_suspended` | API: `/sidelined` | Count suspended players | Int | Suspension impact | ⭐⭐⭐⭐ |
| `home_key_players_missing` | Calculated | Top 5 players injured/suspended | Int | Critical absences (0-5) | ⭐⭐⭐⭐⭐ |

### Group 4: Player Form (4 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_players_in_form` | Calculated | % players with rating > 7.0 | Float | Team confidence | ⭐⭐⭐⭐ |
| `away_players_in_form` | Calculated | % players with rating > 7.0 | Float | Team confidence | ⭐⭐⭐⭐ |
| `home_avg_player_form_3` | Calculated | Avg player rating last 3 | Float | Recent player form | ⭐⭐⭐⭐ |
| `away_avg_player_form_3` | Calculated | Avg player rating last 3 | Float | Recent player form | ⭐⭐⭐⭐ |

---

## 🔧 Implementation Strategy

### Lineup Availability Scenarios

#### Scenario 1: Lineups Available (60-70% of matches)
**When:** 1-2 hours before kickoff
**Data:** Full starting 11 + bench
**Features:** All 25 player features calculated

```python
if lineup_available:
    features['home_starting_11_avg_rating'] = calculate_lineup_quality(lineup)
    features['home_key_players_available'] = count_key_players(lineup)
    # ... all 25 features
```

#### Scenario 2: Lineups Not Available (30-40% of matches)
**When:** More than 2 hours before kickoff
**Data:** Historical averages only
**Features:** Use team averages as fallback

```python
else:  # No lineup
    features['home_starting_11_avg_rating'] = team_avg_rating_season
    features['home_key_players_available'] = 5  # Assume all available
    # ... use historical averages
```

---

## 📊 Key Player Identification

### Top 5 Key Players per Team

**Criteria for "key player":**
1. **Highest avg rating** (last 10 matches)
2. **Most minutes played** (>70% of matches)
3. **Goal/assist contribution** (top 3 in team)
4. **Market value** (if available)

**Example:**
```python
def identify_key_players(team_id, season_id):
    players = get_team_players(team_id, season_id)
    
    # Score each player
    for player in players:
        score = (
            player.avg_rating * 0.4 +
            (player.minutes_played / total_minutes) * 0.3 +
            (player.goals + player.assists) * 0.2 +
            (player.market_value / max_value) * 0.1
        )
        player.key_score = score
    
    # Return top 5
    return sorted(players, key=lambda x: x.key_score, reverse=True)[:5]
```

---

## 🎯 Impact Quantification

### Expected Impact of Key Player Absence

| Player Type | xG Impact | Win Prob Impact | Example |
|-------------|-----------|-----------------|---------|
| **Top Striker** | -0.3 xG | -10% | Haaland, Kane, Mbappé |
| **Playmaker** | -0.2 xG | -8% | De Bruyne, Odegaard |
| **Key Defender** | +0.3 xGA | -7% | Van Dijk, Saliba |
| **Goalkeeper** | +0.2 xGA | -5% | Alisson, Ederson |
| **Star Winger** | -0.15 xG | -6% | Saka, Vinicius Jr |

### Adjustment Formula

```python
def adjust_for_missing_players(base_xg, missing_players):
    """Adjust xG based on missing key players."""
    adjustment = 0
    
    for player in missing_players:
        if player.position == 'Forward':
            adjustment -= 0.3
        elif player.position == 'Midfielder':
            adjustment -= 0.2
        elif player.position == 'Defender':
            adjustment += 0.3  # More xGA
        elif player.position == 'Goalkeeper':
            adjustment += 0.2  # More xGA
    
    adjusted_xg = base_xg + adjustment
    return max(0, adjusted_xg)  # Can't be negative
```

---

## 📡 Data Sources

### SportMonks API Endpoints

```
# Lineups (when available)
GET /v3/football/lineups/fixtures/{fixture_id}

# Player statistics
GET /v3/football/players/{player_id}/statistics/seasons/{season_id}

# Sidelined players (injuries/suspensions)
GET /v3/football/teams/{team_id}/sidelined

# Player details
GET /v3/football/players/{player_id}
```

### Database Tables

```sql
-- Key players tracking
CREATE TABLE key_players (
    id SERIAL PRIMARY KEY,
    team_id INT,
    season_id INT,
    player_id INT,
    player_name VARCHAR(255),
    position VARCHAR(50),
    avg_rating FLOAT,
    key_score FLOAT,
    is_key_player BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Player availability
CREATE TABLE player_availability (
    id SERIAL PRIMARY KEY,
    fixture_id INT,
    team_id INT,
    player_id INT,
    is_available BOOLEAN,
    reason VARCHAR(100), -- 'injured', 'suspended', 'rested'
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔄 Weekly Update Process

```python
def update_key_players_weekly():
    """Update key player list every week."""
    for team in all_teams:
        # Recalculate key players based on recent form
        key_players = identify_key_players(team.id, current_season)
        
        # Update database
        update_key_players_table(team.id, key_players)
        
        # Log changes
        log_key_player_changes(team.id, key_players)
```

---

## ✅ Feature Validation

### Expected Correlations

- **Lineup quality** ↔ **Match result:** 0.35-0.45
- **Key players missing** ↔ **xG:** -0.25 to -0.35
- **Player form** ↔ **Team form:** 0.50-0.60

### A/B Testing

Test model performance:
- **With player features:** Expected +3-5% ROI
- **Without player features:** Baseline

---

**This expands player features from 10 to 25, with focus on key player availability!**
