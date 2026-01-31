# FINAL DIAGNOSIS: Root Cause of Missing Data

## The Real Problem

**Missing data is caused by CORRUPT STATISTICS in the CSV file!**

### Evidence

```
Sample match data:
home_shots_total:         2.0
home_shots_on_target:    13.0  🔴 WRONG! Should be ≤ total
home_corners:           412.0  🔴 IMPOSSIBLE!
home_offsides:          340.0  🔴 IMPOSSIBLE!
away_passes_percentage: 120.0  🔴 Can't be >100%!
```

### Root Cause

**The type_id mapping in `convert_json_to_csv.py` is INCORRECT!**

```python
# In scripts/convert_json_to_csv.py, lines 72-94
stat_type_map = {
    52: 'shots_total',       # ❌ WRONG TYPE ID
    53: 'shots_on_target',   # ❌ WRONG TYPE ID
    54: 'shots_off_target',  # ❌ WRONG TYPE ID
    ...
}
```

The type IDs don't match SportMonks' actual API structure!

---

## Investigation Timeline

### ✅ Step 1: Raw JSON
- 97% of fixtures have statistics
- Statistics structure is correct in JSON

### ✅ Step 2: CSV Extraction
- Statistics are extracted
- 95%+ coverage in CSV

### 🔴 Step 3: Values are WRONG
- Type IDs mapped incorrectly
- Wrong statistics assigned to wrong columns
- Values make no sense (>100% possession, 400+ corners)

### 🔴 Step 4: Feature Calculation Fails
- Feature calculator sees nonsense values
- Tries to calculate xG from wrong stats
- Returns 0 or None for invalid data
- Result: 60-70% missing features

---

## How to Find Correct Type IDs

### Option 1: Check SportMonks Documentation

Visit: https://docs.sportmonks.com/football/api/endpoints/statistics

Find the correct type_id for each statistic.

### Option 2: Inspect Raw JSON

```python
import json
from pathlib import Path

# Load a sample fixture
fixture_file = Path('data/historical/fixtures').glob('*.json').__next__()
with open(fixture_file) as f:
    data = json.load(f)

# Look at first fixture's statistics
fixture = data[0]
stats = fixture.get('statistics', [])

# Print type_ids and their data
for stat in stats:
    print(f"Type {stat['type_id']}: {stat.get('data', {}).get('value')} ({stat.get('location')})")
```

### Option 3: Use Actual Type ID from Our Sample

From the diagnostic output, we saw type_id in the statistics. Let me check what the actual types are.

---

## The Fix

### Step 1: Find Correct Type IDs

Run this to see what type IDs are actually in your data:

```bash
python3 -c "
import json
from pathlib import Path
from collections import defaultdict

fixtures_dir = Path('data/historical/fixtures')
files = list(fixtures_dir.glob('*.json'))[:1]

type_id_values = defaultdict(list)

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)

    for fixture in data[:5]:  # First 5 fixtures
        stats = fixture.get('statistics', [])
        for stat in stats:
            type_id = stat.get('type_id')
            value = stat.get('data', {}).get('value')
            type_id_values[type_id].append(value)

# Show type IDs and sample values
for type_id in sorted(type_id_values.keys()):
    values = type_id_values[type_id]
    avg = sum(v for v in values if v is not None) / len(values) if values else 0
    print(f'Type {type_id:3d}: avg={avg:6.1f}, samples={values[:3]}')
"
