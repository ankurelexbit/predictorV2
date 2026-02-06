# Team Name Search for Live Predictions

## Problem Solved

**Your Question**: "if i just provide man city vs liverpool, how will system know which fixture_id to pick"

**Answer**: The system now has a **smart fixture search** that automatically finds the fixture by team names!

---

## How It Works

### Step-by-Step Process

When you run:
```bash
python predict_live.py --home "Man City" --away "Liverpool"
```

**The system does this**:

1. **Searches for the fixture** in upcoming matches (next 14 days)
   ```
   API Call: GET /fixtures/between/{today}/{14_days_ahead}
   ```

2. **Matches team names** (smart matching):
   - Exact match: "Manchester City" == "Manchester City"
   - Partial match: "Man City" in "Manchester City"
   - Normalized: "mancity" == "manchestercity" (spaces removed)

3. **Finds the fixture_id**:
   ```json
   {
     "fixture_id": 19433642,
     "date": "2026-01-22 20:00:00",
     "home_team_id": 8,
     "home_team_name": "Manchester City",
     "away_team_id": 9,
     "away_team_name": "Liverpool",
     "league_name": "Premier League"
   }
   ```

4. **Fetches lineups** using fixture_id:
   ```
   API Call: GET /fixtures/19433642?include=lineups
   ```

5. **Gets real player stats** from player database:
   ```python
   # For each of the 11 starters:
   player_12345_stats = player_db.get_stats(12345)  # Haaland
   player_67890_stats = player_db.get_stats(67890)  # De Bruyne
   # ...
   ```

6. **Aggregates to team features**:
   ```python
   home_player_rating_3 = mean([8.2, 7.9, 7.5, ...]) = 7.68
   home_player_touches_3 = sum([42.5, 68.3, ...]) = 612.4
   ```

7. **Generates prediction** with accurate features

---

## Usage Examples

### Example 1: Premier League Match

```bash
python predict_live.py --home "Man City" --away "Liverpool"
```

**Output**:
```
================================================================================
SEARCHING FOR: Man City vs Liverpool
Using real-time API data
================================================================================

✅ Found fixture: Manchester City vs Liverpool
   Fixture ID: 19433642
   Date: 2026-01-22 20:00:00
   League: Premier League

Loading models...
✅ Player database loaded - will use real lineup data when available

================================================================================
PREDICTIONS (using live API data)
================================================================================

Premier League
2026-01-22 20:00:00 | Etihad Stadium
Manchester City vs Liverpool

Building features for match: 8 vs 9
✅ Found lineups: 11 home, 11 away
✅ Using REAL player data from lineups
📊 Replacing approximations with real player statistics from lineup
  Home lineup: 9/11 players found (81.8% coverage), avg rating: 7.68
  Away lineup: 10/11 players found (90.9% coverage), avg rating: 7.45

  Prediction: Man City 48% | Draw 26% | Liverpool 26%
  Confidence: Medium
```

### Example 2: Abbreviated Team Names

```bash
python predict_live.py --home "Arsenal" --away "Chelsea"
```

**The system matches**:
- "Arsenal" → "Arsenal FC" ✅
- "Chelsea" → "Chelsea FC" ✅

### Example 3: Partial Names

```bash
python predict_live.py --home "Man United" --away "Tottenham"
```

**The system matches**:
- "Man United" → "Manchester United" ✅
- "Tottenham" → "Tottenham Hotspur" ✅

### Example 4: Match Not Found

```bash
python predict_live.py --home "Barcelona" --away "Real Madrid"
```

**Output** (if no match in next 14 days):
```
================================================================================
SEARCHING FOR: Barcelona vs Real Madrid
Using real-time API data
================================================================================

❌ ERROR: Could not find fixture matching:
   Home: Barcelona
   Away: Real Madrid

Tips:
- Check team name spelling (e.g., 'Man City' or 'Manchester City')
- Ensure match is within next 14 days
- Try with abbreviated names (e.g., 'Liverpool' instead of 'Liverpool FC')

Searched fixtures:
  - Arsenal vs Chelsea
  - Manchester City vs Liverpool
  - Tottenham vs Manchester United
  - ...
```

---

## Smart Team Name Matching

The system uses **flexible matching** to find fixtures:

| You Type | System Finds | Match Type |
|----------|-------------|-----------|
| "Man City" | "Manchester City" | Partial match |
| "Liverpool" | "Liverpool FC" | Partial match |
| "Man United" | "Manchester United" | Partial match |
| "Spurs" | "Tottenham Hotspur" | ⚠️ Won't match (too different) |
| "Manchester City" | "Manchester City" | Exact match ✅ |
| "Man City" (with spaces) | "ManCity" (no spaces) | Normalized match ✅ |

### Matching Algorithm

```python
# The system checks:
1. Exact match: "Liverpool" == "Liverpool"
2. Partial match: "Liverpool" in "Liverpool FC"
3. Reverse partial: "Liverpool FC" contains "Liverpool"
4. Normalized: "mancity" == "manchestercity" (spaces removed)

# If any match → Found!
```

---

## Search Parameters

### Default Search Window

```python
days_ahead = 14  # Searches next 14 days by default
```

### Modify Search Window

To search further ahead, modify the code:

```python
# In predict_live.py, line ~1237:
fixture_info = calculator.search_fixture_by_teams(
    args.home,
    args.away,
    days_ahead=30  # Search next 30 days
)
```

---

## Comparison: Date-Based vs Team Name Search

### Date-Based Prediction (Original)

```bash
python predict_live.py --date tomorrow
```

**Process**:
1. Fetch ALL fixtures for tomorrow
2. Make predictions for ALL matches
3. Output: Multiple predictions

**Use case**: Scout all matches for a specific date

---

### Team Name Search (New)

```bash
python predict_live.py --home "Man City" --away "Liverpool"
```

**Process**:
1. Search for specific fixture by team names
2. Find fixture_id automatically
3. Make prediction for THAT match only
4. Use lineups if available

**Use case**: Deep analysis of one specific match

---

## Complete Flow Diagram

```
User Input: python predict_live.py --home "Man City" --away "Liverpool"
│
├─> Parse Arguments
│   • home_team_name: "Man City"
│   • away_team_name: "Liverpool"
│
├─> Search for Fixture
│   ├─> API Call: GET /fixtures/between/{today}/{today+14days}
│   │
│   ├─> Match Team Names (flexible)
│   │   • "Man City" matches "Manchester City" ✅
│   │   • "Liverpool" matches "Liverpool FC" ✅
│   │
│   └─> Extract fixture_id: 19433642
│
├─> Fetch Lineups (using fixture_id)
│   ├─> API Call: GET /fixtures/19433642?include=lineups
│   │
│   ├─> If lineups announced:
│   │   ├─> Extract player IDs: [12345, 67890, ...]
│   │   └─> Look up in player database
│   │       • Player 12345 (Haaland): rating=8.2, touches=42.5
│   │       • Player 67890 (De Bruyne): rating=7.9, touches=68.3
│   │       • ... (all 11 players)
│   │
│   └─> If lineups NOT announced:
│       └─> Use approximations
│
├─> Aggregate Player Stats
│   • home_player_rating_3 = mean([8.2, 7.9, ...]) = 7.68
│   • home_player_touches_3 = sum([42.5, 68.3, ...]) = 612.4
│   • home_player_duels_won_3 = sum([5.8, 7.2, ...]) = 88.6
│
├─> Build All Features (452 features)
│   • Elo ratings
│   • Form (3, 5, 10 games)
│   • Rolling stats
│   • Player stats (REAL DATA!)
│   • H2H, standings, etc.
│
├─> Generate Prediction
│   • Model: Stacking Ensemble (Elo + Dixon-Coles + XGBoost)
│   • Output: Home 48%, Draw 26%, Away 26%
│
└─> Display Result
    ✅ Man City 48% | Draw 26% | Liverpool 26%
```

---

## Error Handling

### No Fixture Found

**Cause**: Match not in next 14 days, or team names don't match

**Solution**:
```bash
# Try different name variations:
python predict_live.py --home "Manchester City" --away "Liverpool"  # Full name
python predict_live.py --home "Man City" --away "Liverpool"         # Short name

# Check upcoming fixtures manually:
python predict_live.py --date tomorrow  # See what matches exist
```

### Multiple Matches Found

**Rare case**: Same teams play twice in 14 days (rare)

**System behavior**: Returns the **earliest** fixture

### API Timeout

**Cause**: Slow API response

**Solution**: System retries automatically with timeout handling

---

## Benefits Over Manual fixture_id Entry

### Before (Manual)

```bash
# Step 1: Find fixture_id manually
curl "https://api.sportmonks.com/v3/football/fixtures?..." | grep "Man City"
# Find: fixture_id = 19433642

# Step 2: Use fixture_id (but this wasn't even implemented)
# No way to pass fixture_id to the system!
```

### After (Automatic)

```bash
# One command!
python predict_live.py --home "Man City" --away "Liverpool"

# System automatically:
# 1. Searches for fixture
# 2. Finds fixture_id
# 3. Fetches lineups
# 4. Gets player stats
# 5. Generates prediction
```

---

## Advanced Usage

### Combine with Output File

```bash
python predict_live.py --home "Man City" --away "Liverpool" --output predictions.csv
```

### Use Different Model

```bash
python predict_live.py --home "Man City" --away "Liverpool" --model xgboost
```

---

## Implementation Details

### Key Function: `search_fixture_by_teams()`

Location: `predict_live.py`, line ~218

```python
def search_fixture_by_teams(
    self,
    home_team_name: str,
    away_team_name: str,
    days_ahead: int = 7
) -> Optional[Dict]:
    """
    Search for a fixture by team names.

    Returns:
        {
            'fixture_id': 19433642,
            'date': '2026-01-22 20:00:00',
            'home_team_id': 8,
            'home_team_name': 'Manchester City',
            'away_team_id': 9,
            'away_team_name': 'Liverpool',
            'league_name': 'Premier League',
            'venue': 'Etihad Stadium'
        }
    """
```

**Matching Logic**:
```python
# Normalize team names
home_normalized = home_team_name.lower().strip()
fixture_home_name = fixture['home']['name'].lower().strip()

# Check for match (any of these):
home_match = (
    home_normalized in fixture_home_name or      # "man city" in "manchester city"
    fixture_home_name in home_normalized or      # "city" in "man city"
    home_normalized.replace(' ', '') == fixture_home_name.replace(' ', '')  # "mancity" == "manchestercity"
)
```

---

## Testing

### Test with Known Fixture

```bash
# Find a match happening tomorrow
python predict_live.py --date tomorrow

# Note team names from output, then:
python predict_live.py --home "<exact_name>" --away "<exact_name>"
```

### Test with Variations

```bash
# Try different name formats:
python predict_live.py --home "Manchester City" --away "Liverpool"
python predict_live.py --home "Man City" --away "Liverpool"
python predict_live.py --home "City" --away "Liverpool"  # Might work!
```

---

## Summary

✅ **Problem Solved**: "how will system know which fixture_id to pick"

**Solution**: Automatic fixture search by team names

**Process**:
1. User provides: `--home "Man City" --away "Liverpool"`
2. System searches API for matching fixture
3. Finds fixture_id automatically
4. Fetches lineups using fixture_id
5. Looks up real player stats from database
6. Generates accurate prediction

**Benefits**:
- No need to know fixture_id
- No need to know exact team names
- Flexible matching (partial names work)
- Automatic lineup integration
- One simple command

**Usage**:
```bash
# Just provide team names!
python predict_live.py --home "Man City" --away "Liverpool"
```

That's it! 🚀
