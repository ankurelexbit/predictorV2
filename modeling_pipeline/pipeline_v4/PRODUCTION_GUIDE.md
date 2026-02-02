# V4 Production Deployment Guide

Complete guide for deploying V4 predictions to production with Supabase storage.

## Quick Start

### 1. Set Environment Variables

```bash
export SPORTMONKS_API_KEY="your_api_key_here"
export DATABASE_URL="postgresql://postgres:password@db.xxx.supabase.co:5432/postgres"
```

### 2. Test Database Connection

```bash
python3 scripts/test_database.py
```

This will:
- Verify database connectivity
- Create the `predictions` table
- Display table structure

### 3. Generate Predictions for Upcoming Matches

```bash
# Predict next 7 days
python3 scripts/predict_and_store.py --days-ahead 7

# Or specific date range
python3 scripts/predict_and_store.py --start-date 2026-02-01 --end-date 2026-02-07
```

This will:
- Fetch upcoming fixtures from SportMonks API
- Generate predictions using V4 model
- Apply threshold strategy (Home>48%, Draw>35%, Away>45%)
- Store predictions in Supabase
- Show recommended bets

### 4. Update Actual Results

After matches are completed:

```bash
# Update last 7 days
python3 scripts/update_results.py --days-back 7

# Or specific date range
python3 scripts/update_results.py --start-date 2026-02-01 --end-date 2026-02-07
```

This will:
- Fetch completed match results
- Update database with actual scores
- Calculate bet wins/losses and profit
- Display performance metrics

## Database Schema

The `predictions` table stores:

### Prediction Data
- `fixture_id` - Unique match identifier
- `match_date` - When the match is scheduled
- `league_id`, `league_name`, `season_id` - Competition info
- `home_team_id`, `home_team_name` - Home team
- `away_team_id`, `away_team_name` - Away team

### Model Predictions
- `pred_home_prob` - Predicted home win probability
- `pred_draw_prob` - Predicted draw probability
- `pred_away_prob` - Predicted away win probability
- `predicted_outcome` - Most likely outcome (H/D/A)

### Betting Strategy
- `bet_outcome` - Recommended bet (Home Win/Draw/Away Win/NULL)
- `bet_probability` - Confidence in the bet
- `bet_odds` - Implied odds for the bet
- `should_bet` - Whether to place bet (boolean)

### Actual Results
- `actual_home_score` - Final home team score
- `actual_away_score` - Final away team score
- `actual_result` - Actual outcome (H/D/A)
- `bet_won` - Did the bet win? (boolean)
- `bet_profit` - Profit/loss from bet ($)

### Metadata
- `model_version` - Model identifier (default: 'v4')
- `prediction_timestamp` - When prediction was made
- `updated_at` - Last update time

## Production Workflow

### Daily Prediction Pipeline

```bash
#!/bin/bash
# daily_predictions.sh

# Load environment
export SPORTMONKS_API_KEY="your_key"
export DATABASE_URL="your_db_url"

cd /path/to/pipeline_v4

# Generate predictions for next 3 days
python3 scripts/predict_and_store.py --days-ahead 3

# Update results from yesterday
python3 scripts/update_results.py --days-back 1
```

### Cron Setup

```bash
# Edit crontab
crontab -e

# Run daily at 6 AM
0 6 * * * /path/to/daily_predictions.sh >> /var/log/predictions.log 2>&1
```

## Querying Predictions

### Get Today's Recommended Bets

```sql
SELECT
    home_team_name,
    away_team_name,
    bet_outcome,
    bet_probability,
    bet_odds,
    match_date
FROM predictions
WHERE should_bet = TRUE
  AND match_date::date = CURRENT_DATE
  AND model_version = 'v4'
ORDER BY bet_probability DESC;
```

### Get Performance Metrics (Last 30 Days)

```sql
SELECT
    COUNT(*) as total_bets,
    SUM(CASE WHEN bet_won THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(CASE WHEN bet_won THEN 1 ELSE 0 END) * 100, 1) as win_rate_pct,
    ROUND(SUM(bet_profit), 2) as total_profit,
    ROUND(SUM(bet_profit) / COUNT(*) * 100, 1) as roi_pct
FROM predictions
WHERE should_bet = TRUE
  AND actual_result IS NOT NULL
  AND match_date >= NOW() - INTERVAL '30 days'
  AND model_version = 'v4';
```

### Get Pending Results

```sql
SELECT
    fixture_id,
    home_team_name,
    away_team_name,
    match_date,
    bet_outcome,
    bet_probability
FROM predictions
WHERE actual_result IS NULL
  AND match_date < NOW()
  AND model_version = 'v4'
ORDER BY match_date DESC;
```

### Performance by Bet Type

```sql
SELECT
    bet_outcome,
    COUNT(*) as bets,
    SUM(CASE WHEN bet_won THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(CASE WHEN bet_won THEN 1 ELSE 0 END) * 100, 1) as win_rate,
    ROUND(SUM(bet_profit), 2) as profit,
    ROUND(SUM(bet_profit) / COUNT(*) * 100, 1) as roi
FROM predictions
WHERE should_bet = TRUE
  AND actual_result IS NOT NULL
  AND model_version = 'v4'
GROUP BY bet_outcome
ORDER BY profit DESC;
```

## Custom Thresholds

You can override the default thresholds:

```bash
python3 scripts/predict_and_store.py \
    --days-ahead 7 \
    --home-threshold 0.50 \
    --draw-threshold 0.30 \
    --away-threshold 0.48
```

Default thresholds:
- Home: 48%
- Draw: 35%
- Away: 45%

## Monitoring

### Check Recent Predictions

```bash
# View predictions stored in last hour
psql $DATABASE_URL -c "
    SELECT home_team_name, away_team_name, bet_outcome, should_bet
    FROM predictions
    WHERE prediction_timestamp > NOW() - INTERVAL '1 hour'
    ORDER BY match_date;
"
```

### Check Performance

```bash
# Quick performance check
python3 -c "
from src.database import SupabaseClient
import os

db = SupabaseClient(os.environ['DATABASE_URL'])
perf = db.get_betting_performance(days=30)

print(f\"Total Bets: {perf['total_bets']}\")
print(f\"Win Rate: {perf['win_rate']:.1%}\")
print(f\"ROI: {perf['roi']:+.1f}%\")
print(f\"Profit: \${perf['total_profit']:.2f}\")
"
```

## Troubleshooting

### Connection Issues

```bash
# Test database connectivity
python3 scripts/test_database.py
```

### Missing Predictions

Check if model file exists:
```bash
ls -lh models/with_draw_features/xgboost_with_draw_features.joblib
```

### API Rate Limits

SportMonks API has rate limits. If you hit them:
- Reduce `--days-ahead` parameter
- Add delays between requests
- Use pagination carefully

### Database Conflicts

The table has a unique constraint on `(fixture_id, model_version)`. If you try to insert the same fixture twice, it will update the existing record instead of creating a duplicate.

## Next Steps

1. **Monitor Performance**: Check the database daily for actual results
2. **Tune Thresholds**: Adjust thresholds based on historical performance
3. **Add Notifications**: Set up alerts for high-confidence bets
4. **Track ROI**: Monitor profitability over time
5. **Add Calibration**: Implement probability calibration (see V2 pipeline) for better predictions

## Support

For issues or questions, check:
- `CLAUDE.md` - Project documentation
- `README.md` - Quick start guide
- Database logs in Supabase dashboard
