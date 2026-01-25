# Feature Generation Guide

## Overview

The feature generation script processes downloaded historical data and creates training-ready feature vectors.

## Quick Start

```bash
# After downloading historical data
python scripts/generate_training_features.py \
    --data-dir data/historical \
    --output training_features.csv
```

## What It Does

### 1. Loads Historical Data
- Fixtures (matches)
- Match statistics
- Organizes by team for historical lookback

### 2. Calculates Elo Ratings
- Processes matches chronologically
- Updates Elo after each match
- Tracks historical Elo for point-in-time features

### 3. Generates Features
For each match, calculates:
- **Elo features** (10): Current ratings, differentials, momentum
- **Form features** (45): Points, goals, streaks over 3/5/10 matches
- **xG features** (12): Derived xG, xGA, xGD, trends
- **H2H features** (8): Historical matchup patterns
- **Shot features** (12): Volume, location, accuracy
- **Defensive features** (12): PPDA, tackles, possession

**Total:** ~100-140 features per match

### 4. Outputs Training Data
CSV file with:
- All features
- Metadata (fixture_id, teams, date)
- Target variable (H/D/A)
- Actual scores

## Usage

### Basic Usage
```bash
python scripts/generate_training_features.py
```

### Custom Paths
```bash
python scripts/generate_training_features.py \
    --data-dir path/to/historical/data \
    --output path/to/output.csv
```

## Output Format

### CSV Columns

**Metadata:**
- `fixture_id`: Unique match ID
- `match_date`: Match date (ISO format)
- `home_team_id`: Home team ID
- `away_team_id`: Away team ID

**Features:**
- `home_elo`, `away_elo`: Elo ratings
- `elo_diff`, `elo_diff_with_ha`: Elo differentials
- `home_points_last_5`, `away_points_last_5`: Recent form
- `home_derived_xg_per_match`, `away_derived_xg_per_match`: xG
- ... (100-140 total features)

**Target:**
- `target`: Match result ('H', 'D', 'A')
- `home_goals`: Actual home goals
- `away_goals`: Actual away goals

### Example Row
```csv
fixture_id,match_date,home_team_id,away_team_id,home_elo,away_elo,elo_diff,...,target,home_goals,away_goals
12345,2024-08-17T15:00:00,1,2,1650,1580,70,...,H,2,1
```

## Point-in-Time Correctness

The script ensures **no data leakage**:

✅ **Correct:** For a match on 2024-10-15:
- Uses only matches before 2024-10-15
- Elo calculated from prior matches
- Form from last 5 matches before this date
- H2H from historical matchups only

❌ **Incorrect:** Would be using:
- Future match results
- Elo ratings from after this match
- Statistics from this match itself

## Performance

### Processing Time
- **~1,000 matches:** 2-3 minutes
- **~3,800 matches (full season):** 8-10 minutes
- **~11,400 matches (3 seasons):** 25-30 minutes

### Memory Usage
- **Typical:** 500MB - 1GB
- **Peak:** 2GB for 3 seasons

## Minimum Data Requirements

For each match, requires:
- **Minimum 3 prior matches** for each team
- Match statistics (shots, possession, etc.)
- Opponent information

Matches without sufficient history are skipped.

## Troubleshooting

### "No features generated"
**Cause:** Not enough historical data

**Solution:** 
- Ensure you've downloaded data for multiple months
- Check that statistics files exist
- Verify fixtures are sorted chronologically

### "Missing statistics for fixture X"
**Cause:** Statistics file not downloaded

**Solution:**
- Re-run backfill script
- Check `data/historical/statistics/` directory
- Verify API didn't fail for that fixture

### "Elo calculation errors"
**Cause:** Missing match results

**Solution:**
- Check fixture data has scores
- Verify participants are present
- Review logs for specific errors

## Validation

After generation, validate the output:

```python
import pandas as pd

df = pd.read_csv('training_features.csv')

# Check shape
print(f"Matches: {len(df)}")
print(f"Features: {len(df.columns)}")

# Check for missing values
print(f"Missing values:\n{df.isnull().sum().sum()}")

# Check target distribution
print(f"Target distribution:\n{df['target'].value_counts()}")

# Check date range
print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
```

Expected output:
```
Matches: 3500-3800 (per season)
Features: 100-145
Missing values: 0 (or very few)
Target distribution:
H    1600-1700
D    800-900
A    1100-1200
```

## Next Steps

After feature generation:

1. **Explore data:**
   ```python
   import pandas as pd
   df = pd.read_csv('training_features.csv')
   df.describe()
   ```

2. **Train model:**
   ```bash
   python scripts/train_model.py --input training_features.csv
   ```

3. **Evaluate:**
   - Check feature importance
   - Validate predictions
   - Calculate ROI

## Logs

All processing is logged to:
- **Console:** Progress and summary
- **File:** `feature_generation.log` (detailed logs)

Check logs for:
- Number of fixtures processed
- Features generated per match
- Any errors or warnings
- Processing time

## Advanced Usage

### Generate for Specific Date Range

Edit the script to filter fixtures:

```python
# In generate_features() method
for fixture in tqdm(self.fixtures, desc="Generating features"):
    match_date = datetime.fromisoformat(fixture.get('starting_at', '').replace('Z', '+00:00'))
    
    # Add date filter
    if match_date < datetime(2024, 1, 1) or match_date > datetime(2024, 12, 31):
        continue
    
    # ... rest of code
```

### Add Custom Features

Extend the feature pipeline:

```python
# In src/features/feature_pipeline.py
def calculate_features_for_match(self, ...):
    features = {}
    
    # ... existing features
    
    # Add custom features
    features['custom_feature'] = your_calculation()
    
    return features
```

---

**Ready to generate training features!** 🚀
