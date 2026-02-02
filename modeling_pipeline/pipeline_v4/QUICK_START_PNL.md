# Quick Start: PnL Tracking

Complete setup for tracking betting performance with real market odds.

## 🚀 One-Time Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export SPORTMONKS_API_KEY="your_sportmonks_api_key"
export DATABASE_URL="postgresql://user:pass@host:5432/db"
```

### 3. Run Database Migration
```bash
# Adds new columns for odds, features, and PnL tracking
python3 scripts/migrate_database.py
```

Output:
```
================================================================================
DATABASE MIGRATION - PNL TRACKING
================================================================================

📊 Checking current schema...
🔧 Adding 8 new columns...
   Adding column: best_home_odds (FLOAT)
   Adding column: best_draw_odds (FLOAT)
   Adding column: best_away_odds (FLOAT)
   Adding column: avg_home_odds (FLOAT)
   Adding column: avg_draw_odds (FLOAT)
   Adding column: avg_away_odds (FLOAT)
   Adding column: odds_count (INTEGER DEFAULT 0)
   Adding column: features (JSONB)

✅ Migration completed successfully!
```

## 📊 Daily Workflow

### Morning: Generate Predictions
```bash
# Predict today's and tomorrow's matches
python3 scripts/predict_production.py --days-ahead 2
```

**What happens:**
- Fetches upcoming fixtures with odds from SportMonks API
- Generates predictions using 162 features
- Stores predictions with all features and market odds in database
- Applies betting strategy (thresholds: Home 48%, Draw 35%, Away 45%)

### Backtesting on Past Data
```bash
# Test model on past fixtures (e.g., January 2026)
python3 scripts/predict_production.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --include-finished

# Results are already available, so calculate PnL immediately
python3 scripts/update_results.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31

# View backtesting performance
python3 scripts/get_pnl.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31
```

**Use backtesting to:**
- Evaluate new models before deploying
- Find optimal betting thresholds
- Test performance on specific leagues or time periods
- Analyze feature importance for winning bets

### Evening: Update Results
```bash
# Update results for matches completed in last 2 days
python3 scripts/update_results.py --days-back 2
```

**What happens:**
- Fetches actual scores from SportMonks API
- Calculates profit/loss using real market odds
- Updates database with results

### Weekly: Review Performance
```bash
# View PnL report for last 30 days
python3 scripts/get_pnl.py --days 30
```

**Example output:**
```
================================================================================
BETTING PERFORMANCE REPORT
================================================================================
Period: Last 30 days

OVERALL SUMMARY
Total Bets: 45
Wins: 28
Losses: 17
Win Rate: 62.2%
Total Profit/Loss: $12.45
ROI: 27.7%
Average Confidence: 51.3%
Average Odds: 2.15

BY BET TYPE
HOME WINS: 18 bets, 12 wins (66.7%)
DRAW: 12 bets, 8 wins (66.7%)
AWAY WINS: 15 bets, 8 wins (53.3%)
```

## 🤖 Automated Setup (Cron)

Add to crontab (`crontab -e`):

```bash
# Daily predictions at 8 AM
0 8 * * * cd /path/to/pipeline_v4 && python3 scripts/predict_production.py --days-ahead 2

# Daily result updates at 11 PM
0 23 * * * cd /path/to/pipeline_v4 && python3 scripts/update_results.py --days-back 2

# Weekly PnL report (Monday 9 AM)
0 9 * * 1 cd /path/to/pipeline_v4 && python3 scripts/get_pnl.py --days 7 > /tmp/pnl_report.txt

# Weekly model retraining (Sunday 2 AM)
0 2 * * 0 cd /path/to/pipeline_v4 && python3 scripts/weekly_retrain_pipeline.py --weeks-back 4
```

## 📈 What Gets Stored

For each prediction, the database stores:

### Prediction Data
- Match details (fixture_id, teams, league, date)
- Model probabilities (home/draw/away)
- Predicted outcome
- Betting decision (outcome, probability, should_bet)

### Market Odds
- `best_home_odds` - Highest home win odds available
- `best_draw_odds` - Highest draw odds available
- `best_away_odds` - Highest away win odds available
- `avg_home_odds`, `avg_draw_odds`, `avg_away_odds` - Market averages
- `odds_count` - Number of bookmakers

### Features
- `features` - JSONB column with all 162 features used for prediction

### Results (after match)
- `actual_home_score`, `actual_away_score`
- `actual_result` (H/D/A)
- `bet_won` (true/false)
- `bet_profit` - Calculated using best market odds

## 💡 PnL Calculation

```
Profit = (best_market_odds - 1) × stake    (if win)
Profit = -stake                             (if loss)

ROI = (Total Profit / Total Bets) × 100
```

**Example:**
```
Prediction: Home Win (55% confidence)
Best Home Odds: 1.85
Stake: $1.00

If home wins: $0.85 profit
If home loses: -$1.00 loss
```

## 🔍 Advanced: Feature Analysis

Query features to find what correlates with winning bets:

```python
from src.database import SupabaseClient
import pandas as pd

db = SupabaseClient(database_url)

with db.get_connection() as conn:
    # Get all winning bets with features
    df = pd.read_sql("""
        SELECT
            fixture_id,
            bet_outcome,
            bet_probability,
            bet_profit,
            features
        FROM predictions
        WHERE bet_won = TRUE
          AND should_bet = TRUE
    """, conn)

# Extract features from JSONB
features_df = pd.json_normalize(df['features'])

# Find features correlated with profit
correlation = features_df.corrwith(df['bet_profit'])
print("Top 10 features for profitable bets:")
print(correlation.sort_values(ascending=False).head(10))
```

## 📊 Database Queries

### Check pending predictions
```sql
SELECT fixture_id, match_date, home_team_name, away_team_name,
       pred_home_prob, pred_draw_prob, pred_away_prob,
       best_home_odds, best_draw_odds, best_away_odds
FROM predictions
WHERE actual_result IS NULL
  AND should_bet = TRUE
ORDER BY match_date;
```

### Get recent performance
```sql
SELECT
    DATE(match_date) as date,
    COUNT(*) as bets,
    SUM(CASE WHEN bet_won THEN 1 ELSE 0 END) as wins,
    SUM(bet_profit) as daily_profit
FROM predictions
WHERE should_bet = TRUE
  AND actual_result IS NOT NULL
  AND match_date >= NOW() - INTERVAL '30 days'
GROUP BY DATE(match_date)
ORDER BY date DESC;
```

### Best performing odds ranges
```sql
SELECT
    CASE
        WHEN best_home_odds < 1.5 THEN 'Under 1.5'
        WHEN best_home_odds < 2.0 THEN '1.5-2.0'
        WHEN best_home_odds < 2.5 THEN '2.0-2.5'
        ELSE 'Over 2.5'
    END as odds_range,
    COUNT(*) as bets,
    SUM(CASE WHEN bet_won THEN 1 ELSE 0 END) as wins,
    SUM(bet_profit) as profit
FROM predictions
WHERE bet_outcome = 'Home Win'
  AND actual_result IS NOT NULL
GROUP BY odds_range
ORDER BY odds_range;
```

## 🎯 Tips

1. **Run predictions close to match time** - Odds are more accurate
2. **Monitor odds_count** - More bookmakers = better odds reliability
3. **Store features for analysis** - Find what works for your strategy
4. **Adjust thresholds** - Based on PnL performance (config/model_config.py)
5. **Track by league** - Some leagues may perform better
6. **Review weekly** - Identify patterns in winning/losing bets

## 📚 Full Documentation

- `docs/PNL_TRACKING.md` - Complete PnL tracking guide
- `README.md` - Full pipeline documentation
- `docs/FEATURE_DICTIONARY.md` - All 162 features explained

## 🆘 Troubleshooting

**No odds available:**
- Check `odds_count` - might be 0 if match is too far in future
- Verify SportMonks includes odds for your leagues

**Features not stored:**
- Check migration ran successfully
- Verify JSONB column exists in predictions table

**PnL shows 0:**
- Run `update_results.py` after matches complete
- Check actual_result is not NULL in database

**ROI seems wrong:**
- Verify stake is $1 (hardcoded in update_actual_result)
- Check best_*_odds are populated for bets
