# Live Prediction Data Requirements

## How Rolling Statistics Are Calculated

### The Challenge

Many features in the V4 pipeline require **historical match data**:
- Last 5 games form
- Last 10 games xG
- Rolling averages (goals, shots, possession, etc.)
- Recent draw rates
- Momentum indicators

**Critical Point:** These features need RECENT completed matches, not just any historical data.

---

## How It Works

### Point-in-Time Feature Calculation

When you generate features for an upcoming fixture:

```python
features = orchestrator.generate_features_for_fixture(
    fixture_id=12345678,
    as_of_date='2026-02-15'  # Prediction date
)
```

The orchestrator:

1. **Loads all fixtures** from `data/historical/fixtures/`
2. **Filters by date**: Only fixtures BEFORE `as_of_date`
3. **Filters by team**: Gets recent matches for home/away teams
4. **Calculates rolling stats**: Last 5/10 games from filtered matches

### Example

**Scenario:**
- Today: February 15, 2026
- Predicting: Man City vs Liverpool (kickoff: Feb 15, 2026 3:00 PM)
- Need: Man City's last 10 games

**Data Required:**
```
Historical data directory should contain:
  ├─ 2026-02-13_*.json  ← Most recent match
  ├─ 2026-02-10_*.json
  ├─ 2026-02-06_*.json
  ├─ 2026-02-03_*.json
  ├─ 2026-01-30_*.json
  ├─ 2026-01-27_*.json
  ├─ 2026-01-23_*.json
  ├─ 2026-01-20_*.json
  ├─ 2026-01-16_*.json
  └─ 2026-01-13_*.json  ← 10th most recent
```

**If missing recent data:**
```
❌ Historical data only up to 2025-12-31
❌ Last 10 games calculated from December 2025
❌ Missing January/February 2026 matches
❌ Features are STALE and INACCURATE
```

---

## Problem: Stale Historical Data

### How Data Becomes Stale

1. **Initial setup:** You download historical data up to Dec 31, 2024
2. **Time passes:** It's now February 15, 2026
3. **Gap:** 1.5 months of matches missing!
4. **Impact:** Rolling features use old data

### Impact on Feature Accuracy

| Feature | Uses Last N Games | Impact if Stale |
|---------|-------------------|-----------------|
| Form (last 10) | 10 games | ❌ Misses recent wins/losses |
| Goals avg (last 5) | 5 games | ❌ Misses scoring trends |
| xG (last 10) | 10 games | ❌ Misses offensive changes |
| Momentum | Last 3 vs previous 3 | ❌ Completely wrong |
| Draw rate (last 10) | 10 games | ❌ Misses recent draws |
| Position difference | Current standings | ❌ Outdated league table |

**Result:** Predictions will be based on 2-month-old form, leading to poor accuracy!

---

## Solution 1: Update Historical Data Before Prediction (Recommended)

### Option A: Manual Update

```bash
# Step 1: Download recent historical data (last 30 days)
python3 scripts/backfill_historical_data.py \
  --start-date $(date -d "30 days ago" +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --output-dir data/historical

# Step 2: Make predictions
python3 scripts/predict_live_v4.py --date today
```

### Option B: Automated Update (Built-in)

```bash
# Single command - updates historical data first, then predicts
python3 scripts/predict_live_v4.py \
  --date today \
  --update-historical \
  --days-back 30
```

**What this does:**
1. Downloads last 30 days of completed fixtures
2. Saves to `data/historical/fixtures/`
3. Reloads orchestrator with fresh data
4. Generates features using up-to-date rolling stats
5. Makes predictions

### Option C: Shell Script (Convenience)

```bash
# Uses update_and_predict.sh wrapper
bash scripts/update_and_predict.sh --date today --league-id 8
```

---

## Solution 2: Automated Daily Updates

### Cron Job for Daily Historical Updates

Keep historical data fresh automatically:

```bash
# Edit crontab
crontab -e

# Add: Update historical data daily at 5 AM
0 5 * * * cd /path/to/pipeline_v4 && python3 scripts/backfill_historical_data.py --start-date $(date -d "30 days ago" +\%Y-\%m-\%d) --end-date $(date +\%Y-\%m-\%d) >> logs/historical_update.log 2>&1

# Add: Make predictions daily at 6 AM (after update completes)
0 6 * * * cd /path/to/pipeline_v4 && python3 scripts/predict_live_v4.py --date today --output predictions/$(date +\%Y\%m\%d).csv >> logs/predictions.log 2>&1
```

---

## Best Practices

### 1. Always Update Before Live Predictions

**DON'T:**
```bash
# ❌ Uses stale historical data
python3 scripts/predict_live_v4.py --date today
```

**DO:**
```bash
# ✅ Updates historical data first
python3 scripts/predict_live_v4.py --date today --update-historical
```

### 2. How Much Historical Data?

**Minimum:** 30 days (covers ~10 games per team)
```bash
--days-back 30
```

**Recommended:** 60 days (safety margin for injuries, postponements)
```bash
--days-back 60
```

**Maximum:** 90 days (diminishing returns, slower download)
```bash
--days-back 90
```

### 3. Balance Update Frequency vs API Limits

**Daily updates:**
- Updates: ~20-30 fixtures/day × 1 API request = 20-30 requests
- Predictions: ~10 fixtures/day × 1 API request = 10 requests
- **Total: ~30-40 requests/day** (well within 180/hour limit)

**Weekly updates:**
- Updates: ~150 fixtures/week × 1 API request = 150 requests
- Might hit rate limits if run all at once
- **Solution:** Use `--days-back 30` to limit scope

### 4. Monitor Data Freshness

Check when historical data was last updated:

```bash
# Find most recent fixture file
ls -lt data/historical/fixtures/*.json | head -1

# Example output:
# -rw-r--r--  1 user  staff  45678  Feb 14 18:32  2026-02-14_12345678.json
#                                    ↑ Last update date
```

If most recent file is > 7 days old, **update before predictions!**

---

## Workflow Comparison

### ❌ Without Historical Update (Inaccurate)

```
1. Historical data: Last updated Dec 31, 2025
2. Today: Feb 15, 2026
3. Gap: 1.5 months missing
4. Generate features → Uses Dec 2025 data
5. Make predictions → Based on stale form
6. Accuracy: Poor ⚠️
```

### ✅ With Historical Update (Accurate)

```
1. Download recent data: Jan 15 - Feb 15, 2026
2. Merge with historical data
3. Generate features → Uses Feb 2026 data
4. Make predictions → Based on current form
5. Accuracy: Good ✓
```

---

## FAQ

### Q: How do I know if my historical data is stale?

**A:** Check the most recent fixture file:
```bash
ls -lt data/historical/fixtures/*.json | head -1
```

If the date is more than 7 days old, update it.

### Q: Do I need to update every time I predict?

**A:** Depends on prediction frequency:
- **Daily predictions:** Update once per day (morning)
- **Weekly predictions:** Update before prediction run
- **Ad-hoc predictions:** Update if last update was > 3 days ago

### Q: Will updating download all fixtures again?

**A:** No! The backfill script only downloads NEW fixtures. If a fixture file already exists, it skips it.

### Q: How long does an update take?

**A:** Depends on `--days-back`:
- 7 days: ~30-50 fixtures, ~30 seconds
- 30 days: ~150-200 fixtures, ~2-3 minutes
- 60 days: ~300-400 fixtures, ~4-5 minutes

### Q: What if API rate limit is hit during update?

**A:** The backfill script has built-in rate limiting and retry logic. It will slow down automatically if needed.

### Q: Can I skip updating for speed?

**A:** Yes, use `--no-download` flag, but:
- ⚠️ Features will be calculated from existing historical data
- ⚠️ If data is stale, predictions will be inaccurate
- Only skip if you JUST updated (< 1 hour ago)

---

## Production Deployment Checklist

Before deploying live predictions to production:

- [ ] Download at least 60 days of historical data
  ```bash
  python3 scripts/backfill_historical_data.py --start-date 2025-12-15 --end-date 2026-02-15
  ```

- [ ] Set up daily historical update cron job
  ```bash
  0 5 * * * cd /path/to/pipeline_v4 && python3 scripts/backfill_historical_data.py ...
  ```

- [ ] Test prediction with `--update-historical` flag
  ```bash
  python3 scripts/predict_live_v4.py --date today --update-historical
  ```

- [ ] Monitor data freshness (check file dates)
  ```bash
  ls -lt data/historical/fixtures/*.json | head -1
  ```

- [ ] Set up logging for historical updates
  ```bash
  >> logs/historical_update.log 2>&1
  ```

- [ ] Test with different `--days-back` values (30, 60, 90)

---

## Summary

**Key Takeaway:** Live predictions require RECENT historical data for accurate rolling statistics.

**Recommended Approach:**
```bash
# Always update historical data before predictions
python3 scripts/predict_live_v4.py \
  --date today \
  --update-historical \
  --days-back 30 \
  --output predictions.csv
```

**Result:**
- ✅ Up-to-date rolling features
- ✅ Accurate form calculations
- ✅ Current standings
- ✅ Better predictions

**Alternative:** Set up automated daily updates, then predictions run on fresh data automatically.
