# Root Cause Analysis: Data Duplication

## Summary

**Problem:** Every fixture appears exactly 2 times in training data
**Root Cause:** CSV file (`data/processed/fixtures_with_stats.csv`) contains duplicates
**Source:** Either JSON files have duplicates OR convert_json_to_csv.py processes them twice

## Investigation Results

### ✅ What We Know

1. **Training Data: 2x Duplication**
   - Total rows: 35,886
   - Unique fixtures: 17,943
   - Duplication factor: 2.00x

2. **CSV File: 2x Duplication**
   - File: `data/processed/fixtures_with_stats.csv`
   - Total rows: 36,088
   - Unique fixtures: 18,044
   - Duplication factor: 2.00x
   - **This is the source of the problem**

3. **Individual JSON Files: NO Duplication**
   - Example: `all_fixtures_2016-01-01_2016-03-26.json`
   - 585 fixtures, 585 unique IDs
   - No duplicates within single files

4. **Across Multiple JSON Files (first 10): NO Duplication**
   - Checked first 10 JSON files
   - 4,527 fixtures total
   - 4,527 unique IDs
   - No cross-file duplicates (in first 10 files)

## Most Likely Causes

### Hypothesis 1: convert_json_to_csv.py Runs Twice ⭐ **MOST LIKELY**

The script was probably run twice and appended to the same CSV:

```bash
# First run
python3 scripts/convert_json_to_csv.py
# Creates: fixtures_with_stats.csv (18,044 rows)

# Second run (accidentally)
python3 scripts/convert_json_to_csv.py
# Appends: +18,044 rows
# Result: 36,088 rows with perfect 2x duplication
```

**Evidence:**
- Perfect 2x duplication (every fixture exactly twice)
- CSV has 36,088 rows (almost exactly 2x 18,044)
- No duplicates found in JSON files themselves
- convert_json_to_csv.py doesn't check if output file exists

### Hypothesis 2: Some JSON Files Contain Duplicates

Later JSON files (beyond the first 10 checked) might contain duplicates.

**Less likely because:**
- First 10 files are clean
- Individual files are clean
- Would expect irregular duplication pattern, not perfect 2x

### Hypothesis 3: Pandas to_csv Appends Instead of Overwrites

The CSV was created correctly once, then accidentally appended to.

**Evidence needed:**
- Check if there's any code that uses `mode='a'` (append mode)

## How to Verify

### Quick Test:
```bash
# Delete the CSV and regenerate
rm data/processed/fixtures_with_stats.csv

# Regenerate from JSON (run ONCE!)
python3 scripts/convert_json_to_csv.py

# Check for duplicates
python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/fixtures_with_stats.csv')
print(f'Total: {len(df)}, Unique: {df.id.nunique()}, Dups: {df.id.duplicated().sum()}')
"
```

**Expected result:** No duplicates (if hypothesis 1 is correct)

## The Fix

### Immediate Fix (DONE)
✅ Created deduplicated data: `data/training_data_deduped.csv`

### Permanent Fix

**Option 1: Add Duplicate Detection to convert_json_to_csv.py**

```python
# At the end of main(), before saving:
logger.info("\nChecking for duplicates...")
duplicates = df['id'].duplicated().sum()
if duplicates > 0:
    logger.error(f"🚨 Found {duplicates} duplicate fixtures!")
    logger.error("Removing duplicates...")
    df = df.drop_duplicates(subset=['id'], keep='first')
    logger.info(f"✅ Removed duplicates, {len(df)} unique fixtures remain")

# Save to CSV
df.to_csv(csv_file, index=False)
```

**Option 2: Check if File Exists Before Running**

```python
def main():
    csv_file = Path('data/processed/fixtures_with_stats.csv')

    if csv_file.exists():
        logger.warning(f"⚠️  {csv_file} already exists!")
        response = input("Delete and regenerate? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborting.")
            return
        csv_file.unlink()

    # ... rest of the script
```

**Option 3: Add Validation to JSONDataLoader**

```python
def load_all_fixtures(self):
    # ... existing code ...

    # Validation
    duplicates = df['id'].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"⚠️  Found {duplicates} duplicate fixtures in data")
        logger.warning("Removing duplicates...")
        df = df.drop_duplicates(subset=['id'], keep='first')
        logger.info(f"✅ After deduplication: {len(df)} unique fixtures")

    self._fixtures_cache = df
    return df
```

## Recommended Actions

### 1. Regenerate Clean CSV (NOW)

```bash
# Delete duplicated CSV
rm data/processed/fixtures_with_stats.csv

# Regenerate (run ONLY ONCE!)
python3 scripts/convert_json_to_csv.py

# Verify no duplicates
python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/fixtures_with_stats.csv')
dups = df['id'].duplicated().sum()
print(f'Duplicates: {dups}')
if dups == 0:
    print('✅ Clean CSV generated!')
else:
    print(f'🔴 Still has {dups} duplicates')
"
```

### 2. Regenerate Training Data

```bash
# Generate fresh training data from clean CSV
python3 scripts/generate_training_data.py \
  --output data/training_data_clean.csv

# Verify
python3 -c "
import pandas as pd
df = pd.read_csv('data/training_data_clean.csv')
print(f'Rows: {len(df)}')
print(f'Unique fixtures: {df.fixture_id.nunique()}')
print(f'Duplicates: {df.fixture_id.duplicated().sum()}')
"
```

### 3. Add Safeguards

Apply one of the permanent fixes above to prevent future duplication.

## Summary

**Root Cause:** CSV file contains perfect 2x duplication

**Most Likely Reason:** `convert_json_to_csv.py` was run twice

**Solution:**
1. ✅ Use deduplicated data immediately: `data/training_data_deduped.csv`
2. Regenerate clean CSV from JSON (run script only once)
3. Add duplicate detection to prevent future issues
4. Retrain models with clean data

**Impact:**
- Lost 50% of training samples (but they were duplicates anyway)
- Previous metrics were inflated
- New metrics will be more realistic
