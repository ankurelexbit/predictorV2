# Historical Data Backfill Guide

## Quick Start

### 1. Set up environment
```bash
cd pipeline_v3
cp .env.example .env
# Edit .env and add your SPORTMONKS_API_KEY
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run backfill for 2024-2025 season
```bash
python scripts/backfill_historical_data.py \
    --start-date 2024-08-01 \
    --end-date 2025-05-31 \
    --output-dir data/historical
```

### 4. Run backfill for multiple seasons
```bash
# 2022-2023 season
python scripts/backfill_historical_data.py \
    --start-date 2022-08-01 \
    --end-date 2023-05-31

# 2023-2024 season
python scripts/backfill_historical_data.py \
    --start-date 2023-08-01 \
    --end-date 2024-05-31

# 2024-2025 season
python scripts/backfill_historical_data.py \
    --start-date 2024-08-01 \
    --end-date 2025-05-31
```

## Options

- `--start-date`: Start date (YYYY-MM-DD) **required**
- `--end-date`: End date (YYYY-MM-DD) **required**
- `--output-dir`: Output directory (default: data/historical)
- `--skip-lineups`: Skip downloading lineups (faster)
- `--skip-sidelined`: Skip downloading sidelined players (faster)

## What Gets Downloaded

1. **Fixtures** - All matches in date range
   - Match results
   - Teams
   - Dates
   - Scores

2. **Match Statistics** - For each fixture
   - Shots (total, on target, inside/outside box)
   - Big chances
   - Corners
   - Attacks
   - Possession
   - Tackles, interceptions, clearances

3. **Lineups** - For each fixture
   - Starting 11
   - Bench players
   - Formations
   - Jersey numbers

4. **Sidelined Players** - For each team
   - Injuries
   - Suspensions
   - Start/end dates

## Output Structure

```
data/historical/
├── fixtures/
│   ├── league_8_2024-08-01_2025-05-31.json
│   ├── league_564_2024-08-01_2025-05-31.json
│   └── all_fixtures_2024-08-01_2025-05-31.json
├── statistics/
│   ├── fixture_12345.json
│   ├── fixture_12346.json
│   └── ...
├── lineups/
│   ├── fixture_12345.json
│   ├── fixture_12346.json
│   └── ...
└── sidelined/
    ├── team_1.json
    ├── team_2.json
    └── ...
```

## Estimated Time & API Calls

For one full season (~3,800 matches):
- **Fixtures:** ~5 API calls (by league)
- **Statistics:** ~3,800 API calls
- **Lineups:** ~3,800 API calls
- **Sidelined:** ~100 API calls (unique teams)

**Total:** ~7,705 API calls per season

With rate limiting (3,000 calls/min):
- **Time:** ~3-5 minutes per season
- **API usage:** Well within limits

## Monitoring Progress

The script shows progress bars:
```
Leagues: 100%|████████████| 5/5 [00:10<00:00,  2.00s/it]
Match Statistics: 100%|████████████| 3800/3800 [15:00<00:00, 4.22it/s]
Lineups: 100%|████████████| 3800/3800 [15:00<00:00, 4.22it/s]
Sidelined Players: 100%|████████████| 100/100 [00:30<00:00, 3.33it/s]
```

Logs are saved to `backfill.log`

## Troubleshooting

### Rate Limit Errors
The script includes automatic rate limiting and retries. If you still hit limits:
```bash
# Reduce concurrent requests by adding delays
# Edit src/data/sportmonks_client.py and increase sleep times
```

### Missing Data
Some fixtures may not have all data available:
- Lineups may be missing for older matches
- Statistics may be incomplete
- This is normal and handled gracefully

### Resume Interrupted Backfill
The script uses caching, so re-running will skip already downloaded data.

## Next Steps

After backfill completes:
1. Run feature generation script
2. Calculate Elo ratings
3. Build training dataset
4. Train model

See `scripts/generate_training_features.py` (coming next)
