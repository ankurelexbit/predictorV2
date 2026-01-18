# Player Data Integration Guide

## Overview

Adding player-level data can significantly improve prediction accuracy by capturing:
- Team strength beyond historical results
- Impact of injuries and suspensions  
- Player form and fatigue
- Squad depth and rotation patterns
- Key player dependencies

## Data Sources

### 1. **Free APIs**
- **Football-Data.org** (10 calls/min)
  - Squad lists
  - Basic player info
  - Limited lineup data
  
- **API-Football** (100 calls/day free)
  - Detailed lineups
  - Player statistics
  - Injury reports
  - Player ratings

### 2. **Web Scraping Options**
- **Transfermarkt.com**
  - Market values
  - Injury history
  - Squad details
  
- **PhysioRoom.com**
  - Comprehensive injury data
  - Return dates
  
- **WhoScored.com**
  - Player ratings
  - Detailed match stats

### 3. **Premium Data Providers**
- **Opta** - Professional-grade player stats
- **StatsBomb** - Advanced metrics
- **Wyscout** - Scouting data

## Key Player Features to Add

### Team Strength Metrics
```python
# Squad quality
- average_player_market_value
- total_squad_value
- squad_size
- squad_age_profile
- international_players_count

# Positional strength  
- defenders_avg_rating
- midfielders_avg_rating
- attackers_avg_rating
- goalkeeper_rating
```

### Availability Features
```python
# Injuries & Suspensions
- players_injured_count
- key_players_missing
- days_since_last_injury_crisis
- suspension_count

# Specific positions
- striker_available
- main_goalkeeper_available
- center_backs_available
```

### Form Features
```python
# Recent performance
- top_scorer_last_5_games
- goalkeeper_clean_sheets_recent
- team_avg_player_rating_recent
- key_player_goal_drought
```

### Fatigue & Rotation
```python
# Workload
- avg_minutes_played_last_5
- players_played_90min_last_game
- days_since_last_match
- european_competition_midweek
```

## Implementation Steps

### Step 1: Collect Player Data
```bash
# Run player data collection
python 01b_collect_player_data.py
```

This creates:
- `data/raw/player_data/squads_YYYYMMDD.csv`
- `data/raw/player_data/injuries_YYYYMMDD.csv`
- `data/raw/player_data/lineups_YYYYMMDD.csv`

### Step 2: Engineer Player Features
```bash
# Enhance existing features with player data
python 03b_player_feature_engineering.py
```

This creates:
- `data/processed/features_with_players.csv`

### Step 3: Update Models
Models can now use the enhanced features:
```python
# In model scripts, use:
features_df = pd.read_csv('data/processed/features_with_players.csv')
```

## Example: Adding Injury Data

### Manual Collection (CSV)
Create `injuries_current.csv`:
```csv
team_name,player_name,position,injury_type,severity,expected_return
Liverpool,Mohamed Salah,Forward,Hamstring,Major,2024-02-15
Liverpool,Virgil van Dijk,Defender,Knee,Minor,2024-02-01
Man City,Kevin De Bruyne,Midfielder,Muscle,Major,2024-02-20
```

### Automated Collection (API)
```python
def get_injuries_api_football(league_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/injuries"
    params = {"league": league_id, "season": "2023"}
    response = requests.get(url, headers=headers, params=params)
    return response.json()['response']
```

## Impact on Model Performance

Based on research, adding player data typically improves:
- **Accuracy**: +3-5% improvement
- **Upset predictions**: Better at catching when key players missing
- **Calibration**: More accurate probability estimates
- **ROI**: Higher returns due to market inefficiencies

## Best Practices

1. **Update Frequency**
   - Squad data: Weekly
   - Injuries: Daily
   - Lineups: After each match

2. **Feature Engineering**
   - Normalize by position (striker vs defender impact)
   - Weight by player importance
   - Consider recent form

3. **Handling Missing Data**
   - Use team averages for missing players
   - Implement "unknown player" category
   - Flag when data is incomplete

## Quick Start Example

```python
# 1. Collect latest squad data
from modeling_pipeline.src.player_data import collect_squads
squads = collect_squads(['Premier League', 'La Liga'])

# 2. Get current injuries  
injuries = get_injury_report(date='2024-01-15')

# 3. Engineer features
features_df = pd.read_csv('features.csv')
features_df = add_player_features(features_df, squads, injuries)

# 4. Train model with enhanced features
model = XGBoostModel()
model.train(features_df)
```

## Data Quality Checks

Always validate player data:
- Squad sizes reasonable (20-30 players)
- Injury counts realistic (0-10 per team)
- Market values in expected range
- No duplicate players
- Positions properly mapped

## Future Enhancements

1. **Advanced Metrics**
   - xG (expected goals) per player
   - Defensive actions
   - Progressive passes

2. **Tactical Features**
   - Formation changes
   - Playing style metrics
   - Manager preferences

3. **External Factors**
   - Travel distance
   - Weather conditions
   - Referee assignments