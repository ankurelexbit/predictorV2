# Pre-Match Data Checklist

Quick reference for what data you need and where to get it.

---

## 🎯 The Answer: YES, All Features ARE Available Pre-Match!

**Key insight:** Your features are calculated from PAST matches, not the upcoming one.

---

## Data Flow Diagram

```
PAST MATCHES                    YOUR DATABASE              UPCOMING MATCH
(completed)                     (maintained)               (to predict)
─────────────                   ─────────────              ──────────────

Match Results     ─────►    Elo Ratings         ─────►    
  2024-01-01              Liverpool: 1850                 Liverpool vs
  Liverpool 2-1           Man City:  1920                 Man City
  Wolves                                                   2024-01-20
                                                           15:00
Match Stats       ─────►    Rolling Features    ─────►    
  Shots: 15                home_goals_5: 2.1             [PREDICT]
  xG: 2.3                  home_xg_5: 2.0
  Possession: 62%          home_possession_5: 59%

League Table      ─────►    Standings           ─────►
  1. Man City: 70pts       home_position: 2
  2. Liverpool: 65pts      away_position: 1
                           points_diff: -5

                           H2H History         ─────►
                           Last 10 meetings
                           Liverpool: 4 wins

REAL-TIME APIs              FRESH DATA                    
──────────────              ──────────                    

Sportmonks API    ─────►    Injuries            ─────►
Team News                   home_injuries: 2
                           away_injuries: 0

The Odds API      ─────►    Betting Odds        ─────►
Bookmakers                  odds_home: 2.10
                           odds_away: 3.50
```

---

## ✅ Pre-Match Data Checklist

### Before Making a Prediction, You Need:

| Category | Features | Source | How Fresh? | Critical? |
|----------|----------|--------|------------|-----------|
| ✅ **Elo Ratings** | 3 | Your DB | After last match | ⭐⭐⭐ |
| ✅ **Form (3/5 games)** | 18 | Your DB | After last match | ⭐⭐⭐ |
| ✅ **Rolling Stats** | 390 | Your DB | 1-2 days old | ⭐⭐⭐ |
| ✅ **Standings** | 6 | API/Cache | 1 day old | ⭐⭐⭐ |
| ✅ **H2H History** | 5 | Your DB | Months old OK | ⭐⭐ |
| ✅ **Injuries** | 3 | API | 24h before match | ⭐⭐ |
| ✅ **Odds** | 8 | API | Real-time | ⭐⭐ |
| ✅ **Context** | 5 | Calculate | N/A | ⭐ |
| ❌ **Match Result** | 6 | N/A | NOT AVAILABLE | - |

**Total pre-match features:** 438 (out of 465)  
**Not available pre-match:** 27 (match results + goals - these are targets!)

---

## 📊 Where to Get Each Feature

### Your Database (Self-Maintained)
**438 features - The bulk of your model**

```python
# You maintain these by updating after each match
features_from_db = {
    # Elo (3)
    'home_elo': db.get_elo('Liverpool'),
    'away_elo': db.get_elo('Man City'),
    'elo_diff': home_elo - away_elo,

    # Form (18) - pre-calculated
    'home_form_5': db.get_rolling('Liverpool', 'form', 5),
    'away_form_5': db.get_rolling('Man City', 'form', 5),

    # Rolling stats (390) - pre-calculated
    'home_goals_5': db.get_rolling('Liverpool', 'goals', 5),
    'home_xg_5': db.get_rolling('Liverpool', 'xg', 5),
    # ... all 390 rolling features

    # H2H (5) - historical
    'h2h_home_wins': db.get_h2h('Liverpool', 'Man City')['home_wins'],
}
```

**How to populate:**
1. Run `01_sportmonks_data_collection.py` daily
2. Run `02_sportmonks_feature_engineering.py` daily
3. Store features in your database

---

### Sportmonks API (Real-time)
**17 features - Fresh data**

```python
import requests

API_KEY = "your_key"
headers = {"Authorization": API_KEY}

# Standings (6 features)
standings = requests.get(
    f"https://api.sportmonks.com/v3/football/standings/seasons/{season_id}",
    headers=headers
).json()

features_from_api = {
    'home_position': get_position(standings, liverpool_id),
    'home_points': get_points(standings, liverpool_id),
    # ... standings features
}

# Injuries (3 features) - 24-48h before match
sidelined = requests.get(
    f"https://api.sportmonks.com/v3/football/fixtures/{fixture_id}",
    headers=headers,
    params={"include": "sidelined"}
).json()

features_from_api.update({
    'home_injuries': count_injuries(sidelined, home_team_id),
    'away_injuries': count_injuries(sidelined, away_team_id),
})

# Odds (8 features) - real-time
odds = requests.get(
    f"https://api.sportmonks.com/v3/football/fixtures/{fixture_id}",
    headers=headers,
    params={"include": "odds"}
).json()

features_from_api.update({
    'odds_home': odds['home'],
    'odds_draw': odds['draw'],
    'odds_away': odds['away'],
})
```

**Cost:** Free tier (180 req/min) or $19/month

---

### Alternative: The Odds API
**8 features - Just odds**

```python
import requests

API_KEY = "your_odds_api_key"

response = requests.get(
    "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/",
    params={
        "apiKey": API_KEY,
        "regions": "uk",
        "markets": "h2h",  # home/draw/away
    }
).json()

# Find your match
match = next(m for m in response 
             if 'Liverpool' in m['home_team'] and 'Man City' in m['away_team'])

odds_home = match['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
odds_away = match['bookmakers'][0]['markets'][0]['outcomes'][1]['price']
odds_draw = match['bookmakers'][0]['markets'][0]['outcomes'][2]['price']
```

**Cost:** Free (500 req/month) or $10/month

---

### Calculate Yourself
**5 features - Contextual**

```python
from datetime import datetime

match_date = datetime(2024, 1, 20, 15, 0)

contextual_features = {
    'day_of_week': match_date.weekday(),  # 5 = Saturday
    'is_weekend': match_date.weekday() >= 5,  # True
    'round_num': 21,  # From fixture metadata
    'season_progress': 0.55,  # ~halfway through season
    'is_early_season': False,  # round_num > 10
}
```

**Cost:** Free

---

## 🚀 Quick Start: Make Your First Prediction

### Step 1: Set Up Data Collection

```bash
# Install dependencies
pip install sportmonks-python pandas scikit-learn xgboost

# Set your API key in config.py
echo "SPORTMONKS_API_KEY = 'your_key_here'" >> config.py

# Collect historical data (one-time, ~25 min)
python 01_sportmonks_data_collection.py

# Generate features (one-time, ~30 sec)
python 02_sportmonks_feature_engineering.py

# Train model (one-time, ~5 sec)
python 06_model_xgboost.py
```

### Step 2: Create Prediction Script

```python
# predict_match.py
import pandas as pd
import joblib
from sportmonks import SportmonksAPI
from database import get_team_features, get_standings

def predict_upcoming_match(home_team, away_team, match_date):
    """Predict match outcome using pre-match data."""

    # 1. Load model
    model = joblib.load('models/xgboost_model.joblib')

    # 2. Get features from your database
    home_features = get_team_features(home_team, as_of=match_date)
    away_features = get_team_features(away_team, as_of=match_date)

    # 3. Get fresh standings (API or cache)
    standings = get_standings()  # Your function
    home_pos = standings[home_team]['position']
    home_pts = standings[home_team]['points']

    # 4. Build feature vector (all 465 features)
    features = {
        **home_features,  # Elo, form, rolling stats
        **away_features,
        'home_position': home_pos,
        'home_points': home_pts,
        # ... all features
    }

    X = pd.DataFrame([features])

    # 5. Predict
    probabilities = model.predict_proba(X)[0]

    return {
        'home_win': probabilities[2],
        'draw': probabilities[1],
        'away_win': probabilities[0],
    }

# Usage
result = predict_upcoming_match('Liverpool', 'Manchester City', '2024-01-20')
print(f"Predictions: {result}")
# {'home_win': 0.42, 'draw': 0.28, 'away_win': 0.30}
```

### Step 3: Daily Updates

```python
# update_data.py (run daily via cron)
from sportmonks import SportmonksAPI
from database import update_features

def daily_update():
    """Update database with latest matches and features."""
    api = SportmonksAPI()

    # 1. Get yesterday's completed matches
    matches = api.get_fixtures(date=yesterday)

    # 2. Update Elo ratings
    update_elo_ratings(matches)

    # 3. Recalculate rolling features
    update_rolling_features(matches)

    # 4. Update standings
    standings = api.get_standings(current_season)
    update_standings(standings)

    print("✓ Database updated with latest data")

# Run via cron: 0 2 * * * python update_data.py
```

---

## 💡 Pro Tips

### 1. Cache API Calls
```python
import redis
import json

cache = redis.Redis()

def get_standings_cached(season_id):
    """Cache standings for 24 hours."""
    cache_key = f"standings:{season_id}"

    # Try cache first
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Fetch from API
    standings = sportmonks_api.get_standings(season_id)

    # Cache for 24 hours
    cache.setex(cache_key, 86400, json.dumps(standings))

    return standings
```

### 2. Pre-calculate Everything
```python
# Don't do this at prediction time (slow):
def predict_slow(home, away):
    # Calculate 390 rolling features on the fly... ❌
    for window in [3, 5, 10]:
        for stat in stats:
            calculate_rolling_avg(home, stat, window)

# Do this instead (fast):
def predict_fast(home, away):
    # Get pre-calculated features from DB ✅
    features = db.get_all_features(home, away)
    return model.predict(features)
```

### 3. Update Schedule
```
Daily (02:00 AM):
  ✓ Fetch yesterday's matches
  ✓ Update Elo ratings
  ✓ Recalculate rolling features
  ✓ Update standings

Pre-match (1-2 hours before):
  ✓ Get latest injuries
  ✓ Fetch current odds

Weekly:
  ✓ Retrain model (optional)
```

---

## 📋 Data Availability Summary

### Available RIGHT NOW (from past matches):
- ✅ Elo ratings
- ✅ Form metrics (3/5/10 games)
- ✅ All rolling statistics (390 features)
- ✅ H2H history
- ✅ Attack/defense strength

### Available 1-2 HOURS BEFORE MATCH:
- ✅ Current standings
- ✅ Latest injuries
- ✅ Current betting odds

### NOT AVAILABLE (these are what you predict):
- ❌ Match result
- ❌ Goals scored

### The Insight 💡
**Your model predicts the future using only the past.**

Every feature is calculated from:
- Previous matches (rolling windows)
- Current league state (standings)
- External data (odds, injuries)

Nothing comes from the match you're predicting!

---

## Next Steps

1. **Read:** `DATA_AVAILABILITY_GUIDE.md` (detailed)
2. **Set up:** Sportmonks API account (free)
3. **Run:** Data collection pipeline
4. **Build:** Database schema for feature storage
5. **Create:** Prediction script (see examples above)
6. **Automate:** Daily update cron job

---

**Questions?**
- Detailed guide: `DATA_AVAILABILITY_GUIDE.md`
- Feature list: `FEATURE_LIST.md`
- Validation: `FEATURE_VALIDATION_GUIDE.md`

**You're ready to make predictions!** 🎯
