# Supabase Database Setup Guide

## Prerequisites

- Supabase account (free tier works)
- Database created in Supabase

## Step 1: Create Database Schema

1. Go to your Supabase project
2. Navigate to **SQL Editor**
3. Run the schema from `scripts/create_database.sql`

```sql
-- Copy and paste the entire contents of scripts/create_database.sql
-- This creates all tables, indexes, and views
```

## Step 2: Configure Environment

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` and add your credentials:
```bash
# Get these from Supabase Project Settings > API
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key-here

# Get this from SportMonks dashboard
SPORTMONKS_API_KEY=your-sportmonks-api-key
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `supabase` - Python client for Supabase
- `requests` - HTTP library
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `tqdm` - Progress bars

## Step 4: Test Connection

```python
from src.utils.database import SupabaseDB

db = SupabaseDB()
if db.test_connection():
    print("✅ Database connection successful!")
else:
    print("❌ Connection failed")
```

## Step 5: Download Historical Data

```bash
# Download raw data to JSON files (for caching)
python scripts/backfill_historical_data.py \
    --start-date 2024-08-01 \
    --end-date 2025-05-31
```

## Step 6: Load Data into Database

```bash
# Load JSON files into Supabase
python scripts/load_data_to_database.py \
    --data-dir data/historical
```

This will:
- ✅ Load all fixtures into `matches` table
- ✅ Load all statistics into `match_statistics` table
- ✅ Create indexes for fast queries
- ✅ Handle duplicates gracefully

**Time:** ~2-3 minutes for a full season

## Step 7: Generate Features (from Database)

```bash
# Feature generation now reads from Supabase
python scripts/generate_training_features.py \
    --output training_features.csv
```

## Database Tables

### matches
- `fixture_id` - Unique match ID
- `league_id`, `season_id` - Competition info
- `home_team_id`, `away_team_id` - Teams
- `match_date` - When match was played
- `home_goals`, `away_goals` - Scores
- `result` - 'H', 'D', or 'A'

### match_statistics
- `fixture_id` - Links to matches
- `team_id` - Which team
- Shot statistics (total, on target, inside/outside box)
- Chance creation (big chances)
- Set pieces (corners)
- Attacks (total, dangerous)
- Possession & passing
- Defensive actions (tackles, interceptions, clearances)

### elo_history
- `team_id` - Team
- `match_date` - When
- `elo_rating` - Rating at that time
- `elo_change` - Change from previous match

### training_features
- `fixture_id` - Match
- `match_date` - When
- `features` - JSONB with all 100-140 features
- `target` - Match result

## Querying the Database

### Get matches for a team
```python
from src.utils.database import SupabaseDB

db = SupabaseDB()
matches = db.get_team_matches(
    team_id=1,
    before_date='2024-10-15',
    limit=10
)
```

### Get match statistics
```python
stats = db.get_match_statistics(fixture_id=12345)
```

### Get Elo history
```python
elo_history = db.get_team_elo_history(
    team_id=1,
    before_date='2024-10-15'
)
```

## Advantages of Database Storage

✅ **Fast queries** - Indexed for performance  
✅ **Relational** - Join tables easily  
✅ **Scalable** - Handle millions of records  
✅ **Concurrent** - Multiple processes can read  
✅ **Backup** - Supabase handles backups  
✅ **SQL** - Use SQL for complex queries  

## Data Flow

```
1. Download (API → JSON files)
   ↓
2. Load (JSON → Supabase)
   ↓
3. Generate Features (Supabase → CSV)
   ↓
4. Train Model (CSV → Model)
```

## Troubleshooting

### "Connection failed"
- Check SUPABASE_URL and SUPABASE_KEY in `.env`
- Verify Supabase project is active
- Check network connection

### "Table does not exist"
- Run `create_database.sql` in Supabase SQL Editor
- Verify all tables were created

### "Duplicate key error"
- Normal when re-running load script
- Script uses upsert to handle duplicates
- Safe to ignore

### "Rate limit exceeded"
- Supabase free tier: 500 requests/second
- Script batches inserts to stay within limits
- If needed, add delays between batches

## Monitoring

### Check data in Supabase
1. Go to **Table Editor**
2. View `matches`, `match_statistics` tables
3. Verify data looks correct

### Check row counts
```sql
SELECT COUNT(*) FROM matches;
SELECT COUNT(*) FROM match_statistics;
```

Expected for one season:
- Matches: ~3,800
- Statistics: ~7,600 (2 per match)

---

**Database setup complete!** Ready to generate features from Supabase. 🚀
