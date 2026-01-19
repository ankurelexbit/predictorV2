# 🎯 Live Testing System Guide

## Overview

The Live Testing System provides **true live prediction testing** using real-time API data from SportMonks. This is different from the historical backtest - it makes predictions on matches that haven't happened yet, then tracks results when they complete.

## 🔄 How It Works

### Data Flow

```
1. PREDICT
   ↓
   Fetch upcoming fixtures from SportMonks API (no results yet)
   ↓
   Fetch recent team data from API (last 15 matches)
   ↓
   Calculate features in real-time (Elo, form, stats)
   ↓
   Generate predictions with stacking ensemble
   ↓
   Save predictions to tracker

2. WAIT
   ↓
   Matches are played in real life
   ↓

3. UPDATE
   ↓
   Fetch results from SportMonks API
   ↓
   Update tracker with actual outcomes
   ↓

4. EVALUATE
   ↓
   Calculate accuracy, log loss
   ↓
   Test betting strategy
   ↓
   Generate performance report
```

## 🚀 Quick Start

### 1. Predict Today's Matches

```bash
python live_testing_system.py --predict-today
```

This will:
- Fetch all upcoming matches from training leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Championship)
- Generate predictions using live API data
- Save predictions to tracking file

**Output:**
```
PREDICTIONS GENERATED
================================================================================

Premier League
2026-01-19 15:00:00
Arsenal vs Liverpool
  Home: 45.3%
  Draw: 28.1%
  Away: 26.6%
  → Prediction: Home Win

✅ Predictions saved to: live_predictions/predictions_20260119_143022.csv
✅ Tracking 8 predictions
```

### 2. Update Results (After Matches Complete)

```bash
python live_testing_system.py --update-results
```

This will:
- Check all pending predictions
- Fetch results from API for finished matches
- Update tracker with actual outcomes

**Output:**
```
UPDATE RESULTS
================================================================================

Checking: Arsenal vs Liverpool
  ✅ Result: 2-1

✅ Updated 8 results
⏳ 0 still pending
```

### 3. Generate Performance Report

```bash
python live_testing_system.py --report
```

This will:
- Analyze all completed predictions
- Calculate model performance metrics
- Evaluate betting strategy
- Generate report file

**Output:**
```
PERFORMANCE REPORT
================================================================================

📊 Predictions Analyzed: 8
📊 Accuracy: 62.5%
📊 Log Loss: 0.9234

💰 Betting Results:
   ROI: +18.3%
   Net Profit: +$12.45

✅ Report saved to: live_predictions/report_20260119_201534.txt
```

### 4. Full Workflow (All in One)

```bash
python live_testing_system.py --full
```

Runs: predict → wait 5s → update → report

## 📁 File Structure

```
modeling_pipeline/
├── live_testing_system.py           # Main script
├── live_predictions/                # Tracking directory
│   ├── live_predictions_tracker.json  # Master tracking file
│   ├── predictions_20260119.csv       # Daily predictions
│   └── report_20260119.txt            # Performance reports
```

## 📊 Tracking File Format

The `live_predictions_tracker.json` file stores all predictions:

```json
{
  "predictions": [
    {
      "fixture_id": 19456789,
      "date": "2026-01-19 15:00:00",
      "league_name": "Premier League",
      "home_team": "Arsenal",
      "away_team": "Liverpool",
      "home_prob": 0.453,
      "draw_prob": 0.281,
      "away_prob": 0.266,
      "predicted_outcome": "Home Win",
      "predicted_outcome_encoded": 2,
      "status": "completed",
      "actual_home_goals": 2,
      "actual_away_goals": 1,
      "actual_outcome": "Home Win",
      "actual_outcome_encoded": 2,
      "correct": true,
      "prediction_time": "2026-01-19T14:30:22",
      "result_time": "2026-01-19T20:15:33"
    }
  ]
}
```

## 🔍 What Makes This "Live" Testing?

### ✅ True Live Testing (This System)

- Fetches **upcoming matches** from API (no results yet)
- Uses **real-time team data** (latest form, stats, Elo)
- Makes predictions **before matches happen**
- Waits for results from API
- **Cannot train on this data** (true future data)

### ❌ Historical Backtest (Previous System)

- Uses existing CSV data
- Matches already have results
- Tests on "recent" historical data
- Still useful but not truly live

## 📈 Metrics Tracked

### Model Performance
- **Accuracy**: Overall prediction accuracy
- **Log Loss**: Probabilistic prediction quality
- **Accuracy by Outcome**: Home/Draw/Away breakdown
- **Confusion Matrix**: Detailed prediction vs actual

### Betting Strategy
- **ROI**: Return on Investment
- **Win Rate**: Percentage of winning bets
- **Net Profit**: Total profit/loss
- **Performance by Bet Type**: Home/Draw/Away breakdown

## 🎯 Data Sources

### SportMonks API Integration

The system uses multiple SportMonks endpoints:

1. **Fixtures Endpoint**
   ```
   /fixtures/between/{date}/{date}
   ```
   - Fetches upcoming matches
   - Filters by training leagues
   - Includes team IDs and basic info

2. **Team Stats Endpoint**
   ```
   /teams/{id}?include=latest
   ```
   - Fetches last 15 matches
   - Includes statistics, scores
   - Used for feature calculation

3. **Standings Endpoint** (via ESPN)
   ```
   ESPN API for real-time standings
   ```
   - Current league position
   - Current points total

4. **Results Endpoint**
   ```
   /fixtures/{id}?include=scores
   ```
   - Fetches final scores
   - Updates predictions when matches complete

## ⚙️ Configuration

### Training Leagues

Only matches from these leagues are predicted (same as training data):

```python
TRAINING_LEAGUES = {
    8: "Premier League",
    9: "Championship",
    564: "La Liga",
    82: "Bundesliga",
    384: "Serie A",
    301: "Ligue 1"
}
```

### API Configuration

```python
API_KEY = "LmEYRdsf8CSmiblNVTn6JfV0y0s8tc4aQsEkJ6JuoBhOWa3Hd1FEanGcrijo"
BASE_URL = "https://api.sportmonks.com/v3/football"
```

## 🔄 Typical Workflow

### Daily Predictions

**Morning (before matches):**
```bash
# Generate predictions for today
python live_testing_system.py --predict-today
```

**Evening (after matches):**
```bash
# Update results
python live_testing_system.py --update-results

# Generate report
python live_testing_system.py --report
```

### Weekly Review

```bash
# Check all completed predictions from last 7 days
python live_testing_system.py --report
```

## 📊 Example Report

```
================================================================================
LIVE PREDICTION SYSTEM - PERFORMANCE REPORT
================================================================================

📊 Report Date: 2026-01-19 20:15:33
📊 Predictions Analyzed: 45
📊 Date Range: 2026-01-13 to 2026-01-19

================================================================================
MODEL PERFORMANCE
================================================================================

Accuracy: 55.6%
Log Loss: 0.9845

📊 Accuracy by Outcome:
   Away Wins: 45.5% (11 matches)
   Draws: 12.5% (16 matches)
   Home Wins: 88.9% (18 matches)

📊 Confusion Matrix:
              Predicted
              Away   Draw   Home
   Away      5      0      6
   Draw      8      2      6
   Home      2      0     16

================================================================================
BETTING STRATEGY EVALUATION
================================================================================

💰 Betting Results:
   Bets Placed: 35 (77.8% of matches)
   Winning Bets: 18 (51.4%)
   Total Staked: $35.00
   Net Profit: +$5.67
   ROI: +16.2%
   Final Bankroll: $1,005.67 (+0.57%)

📋 Performance by Bet Type:
   Home Win: 12 bets, 75.0% win rate, +$4.23 (+35.3% ROI)
   Draw: 15 bets, 33.3% win rate, +$3.12 (+20.8% ROI)
   Away Win: 8 bets, 37.5% win rate, -$1.68 (-21.0% ROI)

✅ Report saved to: live_predictions/report_20260119_201533.txt
```

## 🚨 Important Notes

### Rate Limiting
- SportMonks API: 180 requests/minute
- System automatically handles rate limiting
- Delays between requests to avoid hitting limits

### API Costs
- Each prediction requires ~3-5 API calls:
  - 1 call for upcoming fixtures
  - 2 calls for team recent matches
  - 1-2 calls for standings

### Match Timing
- Predictions should be made at least 1 hour before kickoff
- Results typically available within 10 minutes after match ends
- Some matches may have delayed results

### Data Accuracy
- Uses same feature calculation as training
- Elo ratings loaded from training data
- Recent form calculated from live API data

## 🔧 Troubleshooting

### No Fixtures Found
```
⚠️  No upcoming fixtures found for today in training leagues
```
**Solution**: Check if there are matches scheduled in training leagues today

### Failed to Generate Predictions
```
⚠️  Failed to generate predictions
```
**Solution**:
1. Check API key is valid
2. Ensure models are trained (run pipeline)
3. Check internet connection

### No Results Available
```
⏳ Still pending
```
**Solution**: Match hasn't finished yet, wait and run `--update-results` later

## 📈 Comparison with Historical Backtest

| Aspect | Historical Backtest | Live Testing |
|--------|---------------------|--------------|
| **Data** | Existing CSV (past matches) | Live API (upcoming matches) |
| **Timing** | All data available | Must wait for results |
| **Features** | Pre-calculated | Calculated in real-time |
| **Sample Size** | Large (500+ matches) | Small (daily matches) |
| **Purpose** | Model validation | Real-world testing |
| **Results** | Instant | Delayed (after matches) |

## 🎯 Next Steps

1. **Run daily predictions** for 30 days to build sample size
2. **Compare with historical backtest** to validate consistency
3. **Track betting strategy** over time
4. **Monitor for model drift** (performance degradation)
5. **Consider model retraining** if performance drops

## 📝 Command Reference

```bash
# Predict today's matches
python live_testing_system.py --predict-today

# Update results for pending predictions
python live_testing_system.py --update-results

# Generate performance report
python live_testing_system.py --report

# Full workflow (predict → update → report)
python live_testing_system.py --full
```

---

**Status**: ✅ Ready for Live Testing
**Last Updated**: 2026-01-19
**Version**: 1.0
