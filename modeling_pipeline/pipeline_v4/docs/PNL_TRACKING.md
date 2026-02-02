# PnL Tracking & Feature Storage

Complete guide to tracking betting performance with real market odds and stored features.

## 🎯 Overview

The V4 pipeline now stores:
- **All 162 features** used for each prediction (in JSONB format)
- **Best and average market odds** from bookmakers (home/draw/away)
- **Actual results and PnL** for each bet placed

This allows you to:
- Calculate accurate Profit & Loss using real market odds
- Analyze which features correlate with successful bets
- Track performance by bet type, league, and time period
- Backtest strategy adjustments

## 📊 Database Schema

### New Columns in `predictions` Table

**Odds Data:**
- `best_home_odds` - Highest home win odds available
- `best_draw_odds` - Highest draw odds available
- `best_away_odds` - Highest away win odds available
- `avg_home_odds` - Average home win odds across bookmakers
- `avg_draw_odds` - Average draw odds
- `avg_away_odds` - Average away win odds
- `odds_count` - Number of bookmakers offering odds

**Features:**
- `features` - JSONB column storing all 162 features used for prediction

**PnL Calculation:**
- `bet_profit` - Profit/loss using best market odds (calculated after match)
- `bet_won` - Boolean indicating if bet won

## 🚀 Usage

### Running Predictions with Odds

Predictions automatically fetch and store odds:

```bash
# Set credentials
export SPORTMONKS_API_KEY="your_key"
export DATABASE_URL="postgresql://user:pass@host:5432/db"

# Run predictions (automatically fetches odds)
python3 scripts/predict_production.py --days-ahead 7
```

**What happens:**
1. Fetches upcoming fixtures with odds included
2. Extracts best and average odds for 1X2 market
3. Generates predictions using 162 features
4. Stores predictions with all features and odds in database

### Viewing PnL Reports

Use the `get_pnl.py` script to view performance:

```bash
# All-time PnL
python3 scripts/get_pnl.py

# Last 30 days
python3 scripts/get_pnl.py --days 30

# Specific date range
python3 scripts/get_pnl.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31
```

**Example Output:**
```
================================================================================
BETTING PERFORMANCE REPORT
================================================================================
Period: 2026-01-01 to 2026-01-31

--------------------------------------------------------------------------------
OVERALL SUMMARY
--------------------------------------------------------------------------------
Total Bets: 45
Wins: 28
Losses: 17
Win Rate: 62.2%
Total Profit/Loss: $12.45
ROI: 27.7%
Average Confidence: 51.3%
Average Odds: 2.15

--------------------------------------------------------------------------------
BY BET TYPE
--------------------------------------------------------------------------------

HOME WINS:
  Bets: 18
  Wins: 12
  Win Rate: 66.7%

DRAW:
  Bets: 12
  Wins: 8
  Win Rate: 66.7%

AWAY WINS:
  Bets: 15
  Wins: 8
  Win Rate: 53.3%

--------------------------------------------------------------------------------
MONTHLY BREAKDOWN (Last 12 Months)
--------------------------------------------------------------------------------
Month        Bets     Wins     Win Rate     Profit/Loss
--------------------------------------------------------------------------------
2026-01      45       28       62.2%        $12.45
2025-12      52       31       59.6%        $8.20
...
```

### Updating Results After Matches

After matches complete, update with actual scores:

```python
from src.database import SupabaseClient

db = SupabaseClient(database_url)

# Update with actual result
db.update_actual_result(
    fixture_id=123456,
    home_score=2,
    away_score=1
)
```

**What happens:**
1. Calculates actual result (H/D/A)
2. Determines if bet won
3. Calculates profit using `best_home_odds`, `best_draw_odds`, or `best_away_odds`
4. Updates `bet_won` and `bet_profit` fields

## 📈 PnL Calculation Logic

### Using Real Market Odds

When a bet is placed:
1. System identifies the bet outcome (Home Win, Draw, or Away Win)
2. Uses the **best market odds** for that outcome
3. After match completes, calculates profit:
   - **Win**: `profit = (best_odds - 1) × stake`
   - **Loss**: `profit = -stake`

Example:
```
Prediction: Home Win (62% confidence)
Best Home Odds: 1.85
Stake: $1.00

If home wins: profit = (1.85 - 1) × 1.00 = $0.85
If home loses: profit = -$1.00
```

### ROI Calculation

```
ROI = (Total Profit / Total Bets) × 100
```

Example:
- 45 bets × $1 = $45 staked
- Total profit: $12.45
- ROI = (12.45 / 45) × 100 = 27.7%

## 🔍 Feature Analysis

All features are stored in JSONB format for analysis.

### Querying Features

```sql
-- Get features for a specific prediction
SELECT features FROM predictions WHERE fixture_id = 123456;

-- Analyze winning bets by Elo difference
SELECT
  features->>'elo_diff' as elo_diff,
  COUNT(*) as bets,
  SUM(CASE WHEN bet_won THEN 1 ELSE 0 END) as wins
FROM predictions
WHERE should_bet = TRUE
  AND actual_result IS NOT NULL
GROUP BY features->>'elo_diff'
ORDER BY elo_diff;

-- Find best features for draw predictions
SELECT
  AVG((features->>'home_draws_last_5')::float) as avg_home_draws,
  AVG((features->>'away_draws_last_5')::float) as avg_away_draws,
  AVG((features->>'h2h_draw_rate')::float) as avg_h2h_draw_rate
FROM predictions
WHERE bet_outcome = 'Draw'
  AND bet_won = TRUE;
```

### Python Feature Analysis

```python
from src.database import SupabaseClient
import pandas as pd

db = SupabaseClient(database_url)

# Get all winning bets with features
with db.get_connection() as conn:
    df = pd.read_sql("""
        SELECT
            fixture_id,
            bet_outcome,
            bet_probability,
            best_home_odds,
            best_draw_odds,
            best_away_odds,
            bet_profit,
            features
        FROM predictions
        WHERE bet_won = TRUE
          AND should_bet = TRUE
    """, conn)

# Extract features from JSONB
features_df = pd.json_normalize(df['features'])

# Analyze correlation between features and profit
correlation = features_df.corrwith(df['bet_profit'])
print("Top features correlated with profit:")
print(correlation.sort_values(ascending=False).head(10))
```

## 📊 Performance Metrics Available

### Database Methods

**`get_betting_performance(days=30)`** - Quick summary
```python
performance = db.get_betting_performance(days=30)
# Returns: total_bets, wins, win_rate, total_profit, roi, avg_confidence
```

**`get_detailed_pnl(start_date, end_date)`** - Detailed breakdown
```python
pnl = db.get_detailed_pnl(start_date='2026-01-01', end_date='2026-01-31')
# Returns: summary, by_outcome, monthly breakdown
```

**`get_pending_results()`** - Bets awaiting results
```python
pending = db.get_pending_results()
# Returns: List of predictions without actual results
```

## 🎯 Best Practices

### 1. Regular Result Updates

Set up a daily job to update results:
```bash
# Create update_results.py script
python3 scripts/update_results.py --days-back 2
```

### 2. Stake Sizing

Current implementation assumes $1 stake. To adjust:
```python
# In update_actual_result method, change:
bet_profit = (market_odds - 1) * 10.0  # $10 stake
```

### 3. Odds Freshness

Odds are fetched when predictions are made. For live odds:
- Run predictions closer to match time
- Store odds timestamp for analysis

### 4. Feature Storage

Features are stored as JSONB for:
- Efficient storage (~50KB per prediction)
- Fast querying with PostgreSQL JSON operators
- Easy analysis with Pandas

## 📝 Example Workflow

### Complete Betting Cycle

```bash
# 1. Generate predictions (Monday for weekend matches)
python3 scripts/predict_production.py \
  --start-date 2026-02-08 \
  --end-date 2026-02-09

# 2. Review predictions and decide stakes
python3 scripts/get_pnl.py --days 30

# 3. After matches complete (Monday)
python3 scripts/update_results.py --days-back 2

# 4. Review performance
python3 scripts/get_pnl.py --days 7
```

## 🔧 Troubleshooting

### No Odds Available

If `odds_count = 0`:
- Odds might not be available yet (matches too far in future)
- SportMonks might not have odds for that league
- Check API response includes `odds.bookmaker;odds.markets`

### Feature Storage Size

JSONB is efficient, but with 162 features per prediction:
- ~50KB per prediction
- 1000 predictions = ~50MB
- Consider archiving old predictions after PnL analysis

### Odds Accuracy

SportMonks aggregates from multiple bookmakers:
- `best_*_odds` = highest odds available (best value)
- `avg_*_odds` = market consensus
- Use `best_*_odds` for PnL (represents actual achievable returns)

## 📚 Related Documentation

- `docs/FEATURE_DICTIONARY.md` - Complete list of 162 features
- `docs/BETTING_STRATEGY.md` - Threshold optimization
- `README.md` - Full pipeline documentation
