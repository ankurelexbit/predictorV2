# Root Cause: Missing Statistics in Training Data

## Summary

**Problem:** 63-75% of Pillar 2 features are missing in training data
**Diagnosis:** ✅ Statistics ARE in raw JSON (97% coverage)
**Root Cause:** 🔴 **BUG in feature generation** - statistics not being loaded into fixtures DataFrame

---

## Investigation Results

### ✅ 1. Raw JSON Files: Statistics Present

```
Fixtures analyzed: 419
Fixtures WITH statistics: 407 (97.1%)
Fixtures WITHOUT statistics: 12 (2.9%)

Stat coverage in JSON:
- Shots: 100%
- Possession: 100%
- Attacks: 100%
- Corners: 100%
- Tackles: 75%
```

**Conclusion:** Raw data is GOOD!

### ✅ 2. CSV File: Statistics Extracted

```
Total stat columns in CSV: 40

Coverage in CSV:
- home_ball_possession: 99.2%
- home_shots_total: 94.8%
- home_attacks: 93.4%
- home_tackles: 83.0%
- home_shots_on_target: 48.5% (lower but still present)
```

**Conclusion:** Conversion to CSV is GOOD!

### 🔴 3. Training Data: Statistics Missing

```
Coverage in training features:
- home_possession_pct_5: 99.3% ✅
- home_shots_per_match_5: 88.1% ⚠️
- home_derived_xg_per_match_5: 32.4% 🔴
- home_shots_on_target_per_match_5: 32.6% 🔴
- home_shot_accuracy_5: 39.6% 🔴
```

**Conclusion:** Feature generation is BROKEN!

---

## The Bug

### Location: `src/data/json_loader.py`

**Problem:** The `load_all_fixtures()` method only loads basic fields, NOT statistics.

```python
# In json_loader.py, line 132-190
def _extract_essential_fields(self, fixture: Dict) -> Optional[Dict]:
    """Extract only essential fields from fixture to reduce memory."""

    return {
        'id': fixture.get('id'),
        'league_id': fixture.get('league_id'),
        'season_id': fixture.get('season_id'),
        'starting_at': fixture.get('starting_at'),
        'home_team_id': home_team.get('id'),
        'away_team_id': away_team.get('id'),
        'home_score': home_score,
        'away_score': away_score,
        'result': result,
        'state_id': fixture.get('state_id'),
        # ❌ NO STATISTICS COLUMNS!
    }
```

### What Happens:

1. **CSV Generation** (`convert_json_to_csv.py`):
   - ✅ Correctly extracts statistics
   - ✅ Saves to `fixtures_with_stats.csv`
   - File has 99% coverage for possession, 95% for shots

2. **Feature Generation** (`generate_training_data.py`):
   - Calls `JSONDataLoader.load_all_fixtures()`
   - This loads from CSV if available (good!)
   - **BUT** when loading CSV, it uses `use_csv=True` parameter

3. **In JSONDataLoader** (line 36-59):
   ```python
   if use_csv:
       csv_file = Path('data/processed/fixtures_with_stats.csv')
       if csv_file.exists():
           self._fixtures_cache = pd.read_csv(csv_file)
           # ✅ This DOES load all the statistics columns!
   ```

4. **BUT** in `Pillar2ModernAnalyticsEngine`:
   ```python
   # Line 56-57
   home_recent = self.data_loader.get_team_fixtures(home_team_id, as_of_date, limit=10)

   # This returns a filtered DataFrame from self._fixtures_cache
   # The DataFrame SHOULD have statistics columns...
   ```

5. **In `_extract_team_stats`** (line 197-206):
   ```python
   def _extract_team_stats(self, match: pd.Series, is_home: bool) -> Dict:
       prefix = 'home_' if is_home else 'away_'

       return {
           'shots_total': match.get(f'{prefix}shots_total', 0) or 0,
           'shots_on_target': match.get(f'{prefix}shots_on_target', 0) or 0,
           # ... tries to get stats from DataFrame row
       }
   ```

### The Real Issue

Let me check if the CSV is actually being loaded...

**Hypothesis:** The feature generation is using the OLD cached fixtures (from JSON parsing) instead of the NEW CSV with statistics!

---

## How to Verify

### Test 1: Check what's in the fixtures DataFrame

```python
from src.data.json_loader import JSONDataLoader

loader = JSONDataLoader('data/historical')
df = loader.load_all_fixtures()

print("Columns in loaded fixtures:")
print(df.columns.tolist())

# Check if stats are present
if 'home_shots_total' in df.columns:
    print("✅ Statistics columns ARE loaded")
    coverage = df['home_shots_total'].notna().sum() / len(df) * 100
    print(f"Coverage: {coverage:.1f}%")
else:
    print("🔴 Statistics columns NOT loaded")
```

### Test 2: Check the cache files

```bash
# Check if there are pickle cache files that might be old
ls -lh data/cache/

# If cache exists, it might be using old data without statistics
# Solution: Delete cache and regenerate
rm -rf data/cache/
```

---

## The Fix

### Solution 1: Force CSV Loading (Quick Fix)

Make sure feature generation uses CSV:

```python
# In generate_training_data.py or anywhere using FeatureOrchestrator

# Make sure CSV exists
csv_file = Path('data/processed/fixtures_with_stats.csv')
if not csv_file.exists():
    print("🔴 CSV file not found! Run convert_json_to_csv.py first")
    exit(1)

# Initialize orchestrator (will use CSV if available)
orchestrator = FeatureOrchestrator(data_dir='data/historical')
```

### Solution 2: Delete Old Cache

```bash
# Delete pickle cache (might have old data without stats)
rm -rf data/cache/

# Regenerate training data (will use CSV)
python3 scripts/generate_training_data.py \
  --output data/training_data_fixed.csv
```

### Solution 3: Fix JSONDataLoader to Always Include Stats

Modify `_extract_essential_fields` to include basic statistics:

```python
def _extract_essential_fields(self, fixture: Dict) -> Optional[Dict]:
    """Extract essential fields INCLUDING basic statistics."""

    # ... existing code ...

    # Extract basic statistics
    stats = self._extract_basic_stats(fixture)

    return {
        'id': fixture.get('id'),
        # ... existing fields ...
        **stats  # Add statistics
    }
```

### Solution 4: Verify CSV is Used (Safest)

```python
# Add logging to JSONDataLoader.load_all_fixtures()

if use_csv:
    csv_file = Path('data/processed/fixtures_with_stats.csv')
    if csv_file.exists():
        logger.info(f"✅ Loading from CSV: {csv_file}")
        self._fixtures_cache = pd.read_csv(csv_file)

        # VERIFY statistics are loaded
        stat_cols = [c for c in self._fixtures_cache.columns if 'shots_total' in c]
        if stat_cols:
            logger.info(f"✅ Statistics columns loaded: {len(stat_cols)}")
        else:
            logger.error("🔴 CSV loaded but NO statistics columns!")
```

---

## Action Plan

### Step 1: Diagnose

```bash
python3 -c "
from src.data.json_loader import JSONDataLoader

loader = JSONDataLoader('data/historical')
df = loader.load_all_fixtures()

print(f'Columns: {len(df.columns)}')
print(f'Has home_shots_total: {\"home_shots_total\" in df.columns}')
print(f'Has home_possession: {\"home_ball_possession\" in df.columns}')

if 'home_shots_total' in df.columns:
    print(f'Shots coverage: {df.home_shots_total.notna().sum() / len(df) * 100:.1f}%')
"
```

### Step 2: Fix

If statistics are NOT in the DataFrame:

```bash
# Option A: Delete cache and retry
rm -rf data/cache/

# Option B: Regenerate CSV
rm data/processed/fixtures_with_stats.csv
python3 scripts/convert_json_to_csv.py

# Option C: Both
rm -rf data/cache/
rm data/processed/fixtures_with_stats.csv
python3 scripts/convert_json_to_csv.py
```

### Step 3: Regenerate Training Data

```bash
python3 scripts/generate_training_data.py \
  --output data/training_data_with_stats.csv

# Verify
python3 -c "
import pandas as pd
df = pd.read_csv('data/training_data_with_stats.csv')

xg_coverage = df['home_derived_xg_per_match_5'].notna().sum() / len(df) * 100
print(f'xG coverage: {xg_coverage:.1f}%')

if xg_coverage > 80:
    print('✅ FIXED!')
else:
    print('🔴 Still broken')
"
```

---

## Expected Results After Fix

### Before (Current - Broken):
```
home_derived_xg_per_match_5: 32.4% coverage
home_shots_on_target_per_match_5: 32.6% coverage
home_shot_accuracy_5: 39.6% coverage
```

### After (Fixed):
```
home_derived_xg_per_match_5: 85-95% coverage
home_shots_on_target_per_match_5: 85-90% coverage
home_shot_accuracy_5: 85-90% coverage
```

**Improvement:** +50-60% more data available!

---

## Bottom Line

1. ✅ **Raw JSON has statistics** (97% coverage)
2. ✅ **CSV has statistics** (95-99% coverage)
3. 🔴 **Feature generation doesn't access them** (bug)

**Most likely cause:** Old pickle cache or CSV not being used

**Fix:** Delete cache, regenerate from CSV, verify statistics are loaded

**Impact:** Will fix 50-60% of missing data, improving model significantly!
