# Live Prediction Pipeline - User Guide

This guide shows how to use the V4 live prediction pipeline to make predictions for upcoming matches.

---

## Overview

The live prediction pipeline (`scripts/predict_live_v4.py`) makes predictions for upcoming fixtures by:

1. **Downloading live data** from SportMonks API for specified dates
2. **Generating 162 features** using the same FeatureOrchestrator used in training
3. **Loading production model** (Conservative CatBoost with draw features)
4. **Making predictions** and displaying probabilities
5. **Saving results** to CSV (optional)

---

## Prerequisites

### 1. Set API Key
```bash
export SPORTMONKS_API_KEY="your_api_key_here"
```

### 2. Have Historical Data
The pipeline needs historical data to calculate features (Elo, form, standings, etc.):
```bash
# Ensure you have historical data
ls data/historical/fixtures/*.json

# If not, download it first
python3 scripts/backfill_historical_data.py --start-date 2023-08-01 --end-date 2024-05-31
```

### 3. Have Production Model
```bash
# Ensure model exists
ls models/with_draw_features/conservative_with_draw_features.joblib
```

---

## Usage

### Basic Usage

**Predict for today:**
```bash
python3 scripts/predict_live_v4.py --date today
```

**Predict for tomorrow:**
```bash
python3 scripts/predict_live_v4.py --date tomorrow
```

**Predict for specific date:**
```bash
python3 scripts/predict_live_v4.py --date 2026-02-15
```

### Date Range

**Predict for multiple days:**
```bash
python3 scripts/predict_live_v4.py \
  --start-date 2026-02-15 \
  --end-date 2026-02-17
```

### Filter by League

**Premier League only (league_id=8):**
```bash
python3 scripts/predict_live_v4.py --date today --league-id 8
```

**Common League IDs:**
- 8: Premier League
- 564: La Liga
- 82: Bundesliga
- 384: Serie A
- 301: Ligue 1

### Save to CSV

**Save predictions to file:**
```bash
python3 scripts/predict_live_v4.py \
  --date today \
  --output predictions/2026-02-15.csv
```

### Custom Model

**Use a different model:**
```bash
python3 scripts/predict_live_v4.py \
  --date today \
  --model models/custom_model.joblib
```

---

## Example Output

```
================================================================================
LIVE PREDICTION PIPELINE - V4
================================================================================
Date range: 2026-02-15 to 2026-02-15
Fetching fixtures from 2026-02-15 to 2026-02-15
✅ Found 12 upcoming fixtures
Downloading detailed data for 12 fixtures...
  ✓ Downloaded fixture 12345678
  ✓ Downloaded fixture 12345679
  ...
✅ Downloaded 12 fixtures to data/historical/fixtures
Generating 162 features for 12 fixtures...
  ✓ Manchester City vs Liverpool
  ✓ Arsenal vs Chelsea
  ...
✅ Generated features for 12 fixtures
✅ Loaded model from models/with_draw_features/conservative_with_draw_features.joblib
Making predictions for 12 fixtures...
✅ Predictions complete

================================================================================
PREDICTIONS
================================================================================

Premier League
2026-02-15T15:00:00+00:00
Manchester City vs Liverpool
  Home Win: 45.2%
  Draw:     28.3%
  Away Win: 26.5%
  → Prediction: Home Win (confidence: 45.2%)

Premier League
2026-02-15T15:00:00+00:00
Arsenal vs Chelsea
  Home Win: 52.1%
  Draw:     25.8%
  Away Win: 22.1%
  → Prediction: Home Win (confidence: 52.1%)

...

✅ Predictions saved to predictions.csv
================================================================================
✅ PIPELINE COMPLETE
================================================================================
```

---

## Output CSV Format

The predictions CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `fixture_id` | SportMonks fixture ID |
| `match_date` | Match kickoff time (ISO format) |
| `league_name` | League name |
| `home_team_name` | Home team name |
| `away_team_name` | Away team name |
| `away_win_prob` | Probability of away win (0-1) |
| `draw_prob` | Probability of draw (0-1) |
| `home_win_prob` | Probability of home win (0-1) |
| `predicted_outcome` | 0=Away, 1=Draw, 2=Home |
| `predicted_outcome_label` | "Away Win", "Draw", or "Home Win" |
| `confidence` | Confidence score (max probability) |

---

## How It Works

### Step 1: Fetch Fixtures from API
```
SportMonks API → Upcoming fixtures for date range
```

### Step 2: Download Detailed Data
```
For each fixture:
  Download full fixture data (statistics, lineups, scores)
  Save to data/historical/fixtures/{date}_{fixture_id}.json
```

### Step 3: Generate Features
```
Reload FeatureOrchestrator → Generate 162 features per fixture
  ├─ Pillar 1: Fundamentals (50 features)
  ├─ Pillar 2: Modern Analytics (60 features)
  └─ Pillar 3: Hidden Edges (52 features including 12 draw features)
```

### Step 4: Make Predictions
```
Load production model → Predict probabilities → Display results
```

---

## Advanced Usage

### Predict Without Downloading Raw Data

If you want to skip saving raw JSON files:
```bash
python3 scripts/predict_live_v4.py --date today --no-download
```

**Note:** Features will still be generated from historical data, but the latest fixture data won't be saved.

### Batch Prediction for Week

Predict for the entire upcoming week:
```bash
python3 scripts/predict_live_v4.py \
  --start-date $(date +%Y-%m-%d) \
  --end-date $(date -d "+7 days" +%Y-%m-%d) \
  --output predictions/week_$(date +%Y%m%d).csv
```

### Filter Multiple Leagues

To predict for multiple leagues, run the script separately for each:
```bash
# Premier League
python3 scripts/predict_live_v4.py --date today --league-id 8 --output pl_predictions.csv

# La Liga
python3 scripts/predict_live_v4.py --date today --league-id 564 --output laliga_predictions.csv

# Combine later with:
cat pl_predictions.csv laliga_predictions.csv > all_predictions.csv
```

---

## Integration with Other Tools

### Load Predictions in Python

```python
import pandas as pd

# Load predictions
df = pd.read_csv('predictions.csv')

# Filter high confidence predictions
high_conf = df[df['confidence'] > 0.6]

# Find draw predictions
draws = df[df['predicted_outcome_label'] == 'Draw']

# Find upset predictions (away win with high confidence)
upsets = df[(df['predicted_outcome_label'] == 'Away Win') & (df['confidence'] > 0.5)]
```

### Combine with Betting Odds

```python
import pandas as pd

# Load predictions
predictions = pd.read_csv('predictions.csv')

# Load odds from external source
odds = pd.read_csv('betting_odds.csv')

# Merge
merged = predictions.merge(odds, on=['home_team_name', 'away_team_name'])

# Find value bets (model probability > implied odds probability)
merged['home_implied_prob'] = 1 / merged['home_odds']
merged['value_bet'] = merged['home_win_prob'] > merged['home_implied_prob']

print(merged[merged['value_bet']])
```

---

## Troubleshooting

### Error: "SPORTMONKS_API_KEY environment variable not set"

**Solution:**
```bash
export SPORTMONKS_API_KEY="your_api_key_here"

# Add to ~/.bashrc or ~/.zshrc for persistence:
echo 'export SPORTMONKS_API_KEY="your_api_key"' >> ~/.bashrc
```

### Error: "Model not found"

**Solution:** Train the model first:
```bash
python3 scripts/train_production_model.py \
  --data data/training_data_with_draw_features.csv \
  --output models/with_draw_features/conservative_with_draw_features.joblib
```

### Error: "No fixtures found"

**Possible causes:**
- No matches scheduled for that date
- League ID filter is too restrictive
- API rate limit reached

**Solution:** Try a different date or remove league filter

### Error: "Insufficient data to generate features"

**Cause:** Missing historical data for teams

**Solution:** Download more historical data:
```bash
python3 scripts/backfill_historical_data.py --start-date 2022-01-01 --end-date 2024-12-31
```

### Slow Performance

**Optimization tips:**
1. Convert JSON to CSV first (100x faster):
   ```bash
   python3 scripts/convert_json_to_csv.py
   ```

2. Use league filter to reduce fixtures:
   ```bash
   python3 scripts/predict_live_v4.py --date today --league-id 8
   ```

3. Use `--no-download` if you don't need raw fixture JSONs

---

## Production Deployment

### Automated Daily Predictions

Set up a cron job to generate predictions every morning:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 6 AM daily)
0 6 * * * cd /path/to/pipeline_v4 && python3 scripts/predict_live_v4.py --date today --output predictions/$(date +\%Y\%m\%d).csv >> logs/predictions.log 2>&1
```

### Weekly Predictions

Generate predictions for the entire week:

```bash
# Add to crontab (runs Sunday at 8 PM)
0 20 * * 0 cd /path/to/pipeline_v4 && python3 scripts/predict_live_v4.py --start-date $(date +\%Y-\%m-\%d) --end-date $(date -d "+7 days" +\%Y-\%m-\%d) --output predictions/week_$(date +\%Y\%m\%d).csv >> logs/predictions.log 2>&1
```

---

## API Limits & Best Practices

### SportMonks API Rate Limits

- Free tier: 180 requests/hour
- Each fixture download = 1 request
- Plan accordingly for date ranges

### Best Practices

1. **Cache fixture data:** Downloaded fixtures are saved, no need to re-download
2. **Use league filters:** Reduce number of fixtures processed
3. **Run during off-peak hours:** Better API response times
4. **Monitor API usage:** Track requests to avoid hitting limits

---

## Next Steps

1. **Analyze prediction accuracy:** Compare predictions against actual results
2. **Calibrate probabilities:** Adjust model if over/under-confident
3. **Integrate with betting strategy:** Use predictions for value bet identification
4. **Build dashboard:** Visualize predictions in web interface
5. **Automate retraining:** Keep model updated with latest data

For questions or issues, see:
- `PRODUCTION_FILES.md` - Complete production file reference
- `README.md` - V4 pipeline overview
- `docs/FEATURE_FRAMEWORK.md` - Feature engineering details
