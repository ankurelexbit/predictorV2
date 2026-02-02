# V4 Pipeline - Complete Feature Generation System

## 📋 Overview

Complete **3-pillar feature generation pipeline** with **150 features** built from raw SportMonks JSON data.

---

## ⚡ Production Model (v4.1)

**Currently Deployed:** Option 3: Balanced

**Performance (January 2026 backtest on Top 5 leagues):**
- ✅ **28.3% ROI** - Profit: $41.81 on 148 bets
- ✅ **52.0% Overall Win Rate** - 77 wins / 148 bets
- ✅ **39.0% Draw Win Rate** - 32 wins / 82 draw bets (37.8% ROI)
- ✅ **Top 5 Leagues Only** - 3x better than all leagues

**Configuration:**
- Model: `models/weight_experiments/option3_balanced.joblib`
- Class Weights: Away=1.1, Draw=1.4, Home=1.2
- Thresholds: Home=0.65, Draw=0.30, Away=0.42
- Leagues: Premier League, Bundesliga, Serie A, Ligue 1, La Liga
- Full Config: `config/production_config.py`

**Documentation:**
- Setup Guide: `PRODUCTION_DEPLOYMENT_SUMMARY.md`
- Training Guide: `TRAINING_PIPELINE_UPDATED.md`

---

## 🚀 Quick Start

### Step 1: Setup Environment

```bash
cd pipeline_v4

# Create virtual environment
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Historical Data

**IMPORTANT:** This must be done first before feature generation!

```bash
# Set your SportMonks API key
export SPORTMONKS_API_KEY="your_api_key_here"

# Download historical data (example: 2023 season)
python3 scripts/backfill_historical_data.py \
  --start-date 2023-08-01 \
  --end-date 2024-05-31 \
  --output-dir data/historical

# This will download:
# - Fixtures with embedded statistics and lineups
# - Sidelined players (injuries/suspensions)
# - Standings data from participant metadata
```

**Download Options:**
```bash
# Skip lineups (faster)
--skip-lineups

# Skip sidelined players
--skip-sidelined

# More parallel workers (faster, but watch rate limits)
--workers 10

# Custom output directory
--output-dir data/my_data
```

### Step 3: Convert JSON to CSV (Recommended!)

**IMPORTANT:** JSON processing is slow. Convert to CSV once for 100x faster feature generation!

```bash
# One-time conversion (~1 minute)
python3 scripts/convert_json_to_csv.py

# This creates: data/processed/fixtures.csv
# Future feature generation will be instant!
```

### Step 4: Generate Training Data

Once historical data is downloaded, generate features:

```bash
# Generate features for all downloaded data
python3 scripts/generate_training_data.py \
  --output data/training_data.csv

# Or filter by league/date
python3 scripts/generate_training_data.py \
  --league-id 8 \
  --start-date 2023-08-01 \
  --output data/training_pl_2023.csv
```

## ✅ What's Built

Complete **3-pillar feature generation pipeline** with **150 features**:

### Core Components
- ✅ **JSON Loader** (streaming) - Handles large files efficiently
- ✅ **Standings Calculator** - Point-in-time league standings
- ✅ **Elo Calculator** - Team rating system

### Feature Engines (150 features)
- ✅ **Pillar 1: Fundamentals** (50 features)
  - Elo ratings, standings, form, H2H, home advantage
- ✅ **Pillar 2: Modern Analytics** (60 features)  
  - Derived xG, shots, defense, attacks
- ✅ **Pillar 3: Hidden Edges** (40 features)
  - Momentum, fixture-adjusted, player quality, context

### Orchestration
- ✅ **Feature Orchestrator** - Coordinates all 3 pillars
- ✅ **Training Data Generator** - Creates complete datasets

## 📊 Files Created

```
pipeline_v4/
├── src/
│   ├── data/
│   │   └── json_loader.py ✅
│   └── features/
│       ├── standings_calculator.py ✅
│       ├── elo_calculator.py ✅
│       ├── pillar1_fundamentals.py ✅ (50 features)
│       ├── pillar2_modern_analytics.py ✅ (60 features)
│       ├── pillar3_hidden_edges.py ✅ (40 features)
│       └── feature_orchestrator.py ✅
└── scripts/
    ├── test_core_infrastructure.py ✅
    ├── test_feature_orchestrator.py ✅
    ├── generate_training_data.py ✅
    └── build_cache.py ✅
```

## 🚀 Quick Start

### Option 1: Generate Training Data (Recommended)

Generate features for specific leagues/dates to avoid loading all 20K fixtures:

```bash
cd pipeline_v4

# Premier League only (much faster)
python3 scripts/generate_training_data.py \
  --league-id 8 \
  --start-date 2023-01-01 \
  --output data/training_pl_2023.csv

# Multiple leagues
python3 scripts/generate_training_data.py \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --output data/training_2023_2024.csv

# Small sample for testing
python3 scripts/generate_training_data.py \
  --max-fixtures 500 \
  --output data/training_sample.csv
```

### Option 2: Use in Python

```python
from src.features.feature_orchestrator import FeatureOrchestrator

# Initialize (loads all fixtures once)
orchestrator = FeatureOrchestrator('data/historical')

# Generate features for a single fixture
features = orchestrator.generate_features_for_fixture(fixture_id=12345)

# Generate training dataset
df = orchestrator.generate_training_dataset(
    league_id=8,  # Premier League
    start_date='2023-01-01',
    output_file='data/training.csv'
)
```

## ⚠️ Performance Notes

**Initial Load Time:** ~1 minute (loads all 20K+ fixtures from 258 JSON files)
- This happens once per script run
- Necessary for point-in-time features (standings, Elo, form)

**Solutions:**
1. **Filter by league/date** - Reduces fixtures to process
2. **Use cache** (advanced) - Run `build_cache.py` once
3. **Load once, generate multiple** - Reuse orchestrator instance

## 📋 Feature List

### Pillar 1: Fundamentals (50)
- Elo: 10 features
- Standings: 12 features  
- Form: 15 features
- H2H: 8 features
- Home advantage: 5 features

### Pillar 2: Modern Analytics (60)
- Derived xG: 25 features
- Shots: 15 features
- Defense: 12 features
- Attacks: 8 features

### Pillar 3: Hidden Edges (40)
- Momentum: 12 features
- Fixture-adjusted: 10 features
- Player quality: 10 features
- Context: 8 features

## ✅ Key Features

- **Point-in-time correct** - No data leakage
- **Streaming parser** - Handles 1.3GB files
- **Embedded statistics** - Uses data from fixture JSON
- **Modular design** - Easy to extend/modify
- **Production ready** - Comprehensive logging

## 📈 Training Pipeline

### Full Training Workflow

```bash
# 1. Download Historical Data
export SPORTMONKS_API_KEY="your_key"
python3 scripts/backfill_historical_data.py \
  --start-date 2023-08-01 \
  --end-date 2024-05-31

# 2. Convert to CSV (optional but recommended - 100x faster)
python3 scripts/convert_json_to_csv.py

# 3. Generate Training Data (162 features including player features)
python3 scripts/generate_training_data.py \
  --output data/training_data.csv

# 4. Train Model (defaults to Option 3: Balanced - A=1.1, D=1.4, H=1.2)
python3 scripts/train_production_model.py \
  --data data/training_data.csv \
  --output models/production/option3_$(date +%Y%m%d).joblib

# Optional: Custom class weights
python3 scripts/train_production_model.py \
  --data data/training_data.csv \
  --output models/custom_model.joblib \
  --weight-away 1.0 \
  --weight-draw 1.5 \
  --weight-home 1.3
```

### What Gets Generated

**Training CSV Includes:**
- **162 feature columns** across 3 pillars
  - Pillar 1: Elo, standings, form, H2H (50 features)
  - Pillar 2: Derived xG, shots, defense (60 features)
  - Pillar 3: Momentum, player quality, context (52 features)
- **Metadata columns:** fixture_id, team_ids, league_id, season_id, match_date
- **Target columns:** home_score, away_score, result

**Player Features Now Use Real Data:**
- `home/away_lineup_avg_rating_5` - Team quality from recent performance
- `home/away_top_3_players_rating` - Attacking lineup strength
- `home/away_key_players_available` - Position coverage score
- `home/away_players_in_form` - Form from last 3 matches
- `home/away_players_unavailable` - Injury/suspension estimates

---

## 🔮 Live Prediction Pipeline

### Production Prediction Setup

The V4 pipeline includes production-ready scripts for live predictions.

**Production Model (v4.1):**
```
models/weight_experiments/option3_balanced.joblib
```
- **Option 3: Balanced** model with class weights (A=1.1, D=1.4, H=1.2)
- Trained on 162 features including real player data
- Optimized thresholds: H=0.65, D=0.30, A=0.42
- **Top 5 European Leagues Only** (for optimal performance)
- Expected Performance: 28.3% ROI, 52% win rate (January 2026 backtest)

**Basic Usage:**
```bash
# Set credentials
export SPORTMONKS_API_KEY="your_key"
export DATABASE_URL="postgresql://user:pass@host:5432/db"

# Run predictions for next 7 days
python3 scripts/predict_production.py --days-ahead 7

# Or specific date range
python3 scripts/predict_production.py \
  --start-date 2026-02-01 \
  --end-date 2026-02-07
```

**Advanced Options:**
```bash
# Use custom model
python3 scripts/predict_production.py \
  --days-ahead 7 \
  --model-path models/production/my_model.joblib

# Custom historical data range (for Elo/form calculation)
python3 scripts/predict_production.py \
  --days-ahead 7 \
  --history-start-date 2025-08-01 \
  --history-end-date 2026-01-31

# Or just specify history days
python3 scripts/predict_production.py \
  --days-ahead 7 \
  --history-days 180

# Backtest on past fixtures (includes finished matches)
python3 scripts/predict_production.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --include-finished
```

### How Live Prediction Works

**1. Pipeline Initialization (Once per run)**
```python
# Loads 365 days of historical data on startup
pipeline = ProductionLivePipeline(api_key, load_history_days=365)

# Initialization includes:
# - Fetches 7,000+ historical fixtures from API (in 30-day chunks)
# - Calculates Elo ratings for 1,000+ teams
# - Builds statistics DataFrame with derived xG data
# - Initializes all 3 feature pillars
```

**2. Feature Generation (Per match)**
```python
# For each upcoming fixture:
result = pipeline.predict(fixture)

# Features generated point-in-time:
# - Elo ratings from historical matches
# - Form from last 5 matches
# - Derived xG from recent statistics
# - Player quality from recent performance
# - Momentum indicators
# - All 162 features calculated correctly
```

**3. Prediction Output**
```python
{
  'home_prob': 0.42,    # 42% home win
  'draw_prob': 0.31,    # 31% draw
  'away_prob': 0.27     # 27% away win
}

# Applies threshold strategy (Option 3: Balanced):
# - Home bet if prob > 65% (high confidence only)
# - Draw bet if prob > 30% (primary profit source)
# - Away bet if prob > 42% (balanced approach)
```

**4. Database Storage**
Predictions are stored in Supabase with:
- Match details (teams, league, date)
- Probabilities (home/draw/away)
- Predicted outcome
- Betting recommendation (if meets threshold)
- **All 162 features** used for prediction (JSONB)
- **Best and average market odds** from bookmakers
- **Actual results and PnL** (calculated after match)
- Model version

### Live Prediction Components

**Main Scripts:**
- `scripts/predict_production.py` - Full production pipeline with database
- `scripts/predict_live_with_history.py` - Core prediction engine

**Key Features:**
- ✅ **Historical context**: Loads 1 year of data on startup for proper Elo/form calculation
- ✅ **Same feature generation**: Uses identical code as training (no train/test mismatch)
- ✅ **Real player features**: Calculates team quality from recent performance
- ✅ **Option 3: Balanced model**: Class weights A=1.1, D=1.4, H=1.2 (optimal for draws)
- ✅ **Optimized thresholds**: H=0.65, D=0.30, A=0.42 (tested on January 2026)
- ✅ **League filtering**: Top 5 European leagues only (3x better performance)
- ✅ **Database integration**: Stores predictions in Supabase with full feature tracking
- ✅ **Backtesting support**: Can predict on past fixtures with `--include-finished`
- ✅ **Centralized config**: All settings in `config/production_config.py`

---

## 🔬 Backtesting

Test your model on past data to evaluate performance:

```bash
# Generate predictions for January 2026 (past fixtures)
python3 scripts/predict_production.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --include-finished

# Update results to calculate actual PnL
python3 scripts/update_results.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31

# View backtesting performance
python3 scripts/get_pnl.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31
```

**Use Cases:**
- **Model evaluation**: Test new models on historical data before deploying
- **Strategy tuning**: Find optimal betting thresholds
- **Feature analysis**: Identify which features correlate with wins
- **League testing**: See which leagues perform best

**Important Notes:**
- ⚠️ Use `--include-finished` flag for past dates
- ⚠️ Past fixtures already have results, so PnL is immediately calculable
- ⚠️ Odds stored are historical odds from when fixtures were scheduled
- ✅ Point-in-time correctness ensures no data leakage

---

## 💰 PnL Tracking & Feature Storage

### What Gets Stored

Every prediction stores:
- **All 162 features** used for prediction (JSONB format)
- **Market odds** from bookmakers:
  - Best home/draw/away odds (highest available)
  - Average home/draw/away odds (market consensus)
  - Number of bookmakers offering odds
- **Actual results** after match completes
- **Profit/Loss** calculated using real market odds

### Complete Betting Workflow

```bash
# 1. Generate Predictions (fetches odds automatically)
python3 scripts/predict_production.py --days-ahead 7

# 2. Update Results After Matches Complete
python3 scripts/update_results.py --days-back 2

# 3. View PnL Report
python3 scripts/get_pnl.py --days 30
```

### PnL Reports

**All-time performance:**
```bash
python3 scripts/get_pnl.py
```

**Last 30 days:**
```bash
python3 scripts/get_pnl.py --days 30
```

**Specific period:**
```bash
python3 scripts/get_pnl.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31
```

**Example Output:**
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

MONTHLY BREAKDOWN
2026-01: 45 bets, 28 wins, 62.2% win rate, $12.45 profit
2025-12: 52 bets, 31 wins, 59.6% win rate, $8.20 profit
```

### PnL Calculation

Uses **real market odds** for accurate profit calculation:

```
Win:  profit = (best_market_odds - 1) × stake
Loss: profit = -stake

ROI = (Total Profit / Total Bets) × 100
```

**Example:**
- Prediction: Home Win (55% confidence)
- Best Home Odds: 1.85 (from Bet365)
- Stake: $1.00
- If home wins: profit = (1.85 - 1) × 1 = **$0.85**
- If home loses: profit = **-$1.00**

### Feature Analysis

All 162 features are stored and queryable for analysis:

```sql
-- Find features correlated with winning bets
SELECT
  features->>'elo_diff' as elo_diff,
  COUNT(*) as bets,
  AVG(bet_profit) as avg_profit
FROM predictions
WHERE bet_won = TRUE
  AND should_bet = TRUE
GROUP BY features->>'elo_diff'
ORDER BY avg_profit DESC;
```

**Python analysis:**
```python
from src.database import SupabaseClient
import pandas as pd

db = SupabaseClient(database_url)

# Get all predictions with features
with db.get_connection() as conn:
    df = pd.read_sql("""
        SELECT fixture_id, bet_outcome, bet_probability,
               bet_profit, features
        FROM predictions
        WHERE should_bet = TRUE
    """, conn)

# Extract features from JSONB
features_df = pd.json_normalize(df['features'])

# Analyze correlation between features and profit
correlation = features_df.corrwith(df['bet_profit'])
print("Top features for profitable bets:")
print(correlation.sort_values(ascending=False).head(10))
```

See `docs/PNL_TRACKING.md` for complete guide.

---

## 🔄 Weekly Retraining

Automated weekly pipeline to keep model fresh with Option 3 configuration:

```bash
# Retrain with last 4 weeks of new data (uses Option 3 defaults)
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4

# Or use training script directly
python3 scripts/train_production_model.py \
  --data data/training_data_latest.csv \
  --output models/production/option3_$(date +%Y%m%d).joblib

# Retrain specific league only
python3 scripts/weekly_retrain_pipeline.py \
  --weeks-back 4 \
  --league-id 8
```

**What It Does:**
1. Downloads new fixture data from SportMonks API
2. Converts to CSV for fast processing
3. Generates training data with all 162 features (including real player features)
4. Trains new model with **Option 3: Balanced** weights (A=1.1, D=1.4, H=1.2)
5. Saves model with timestamp and metadata

**Setup Automated Retraining:**
```bash
# Add to crontab (runs every Sunday at 2 AM)
crontab -e

# Add this line:
0 2 * * 0 cd /path/to/pipeline_v4 && python3 scripts/weekly_retrain_pipeline.py --weeks-back 4
```

**Monitoring Retrained Models:**
- Check `*_metadata.json` for test set performance
- Compare against baseline: log loss < 1.0, accuracy > 50%, draw accuracy 35-40%
- See `TRAINING_PIPELINE_UPDATED.md` for quality control guidelines

See `docs/WEEKLY_RETRAIN_GUIDE.md` for detailed setup.

---

## 🚀 Production Deployment

### Complete Production Setup

**Current Production Model:** Option 3: Balanced (v4.1)
- **Performance:** 28.3% ROI, 52% win rate on Top 5 leagues (January 2026 backtest)
- **Configuration:** See `config/production_config.py` for all settings
- **Full Guide:** See `PRODUCTION_DEPLOYMENT_SUMMARY.md` for deployment details

**1. Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SPORTMONKS_API_KEY="your_api_key"
export DATABASE_URL="postgresql://user:pass@host:5432/db"

# Run database migration (adds PnL tracking columns)
python3 scripts/migrate_database.py

# Validate production configuration
python3 config/production_config.py
```

**2. Initial Model Training**
```bash
# Download historical data (one-time)
python3 scripts/backfill_historical_data.py \
  --start-date 2023-08-01 \
  --end-date 2024-05-31

# Generate training data
python3 scripts/generate_training_data.py \
  --output data/training_data.csv

# Train initial model with Option 3 defaults (A=1.1, D=1.4, H=1.2)
python3 scripts/train_production_model.py \
  --data data/training_data.csv \
  --output models/production/option3_$(date +%Y%m%d).joblib
```

**3. Daily Predictions**
```bash
# Run predictions for today's matches
python3 scripts/predict_production.py --days-ahead 1

# Or schedule with cron (runs daily at 8 AM)
0 8 * * * cd /path/to/pipeline_v4 && python3 scripts/predict_production.py --days-ahead 1
```

**4. Daily Result Updates**
```bash
# Update results and calculate PnL (runs daily after matches)
python3 scripts/update_results.py --days-back 2

# Schedule with cron (runs daily at 11 PM)
0 23 * * * cd /path/to/pipeline_v4 && python3 scripts/update_results.py --days-back 2
```

**5. Weekly Model Updates**
```bash
# Setup automatic retraining (Sunday 2 AM)
0 2 * * 0 cd /path/to/pipeline_v4 && python3 scripts/weekly_retrain_pipeline.py --weeks-back 4
```

**6. Monitor Performance**
```bash
# View PnL report (run weekly)
python3 scripts/get_pnl.py --days 30
```

---

## 📊 Pipeline Architecture

### Training Flow
```
SportMonks API
      ↓
[backfill_historical_data.py]
      ↓
JSON files (data/historical/)
      ↓
[convert_json_to_csv.py] (optional, 100x faster)
      ↓
CSV file (data/processed/fixtures_with_stats.csv)
      ↓
[generate_training_data.py]
      ↓
FeatureOrchestrator
  ├─→ Pillar1: Elo, form, standings (50 features)
  ├─→ Pillar2: Derived xG, shots (60 features)
  └─→ Pillar3: Momentum, players (52 features)
      ↓
Training CSV (162 features)
      ↓
[train_production_model.py]
      ↓
Trained Model (.joblib)
```

### Live Prediction Flow
```
[predict_production.py]
      ↓
ProductionLivePipeline
  ├─→ Load 365 days history from API
  ├─→ Calculate Elo for all teams
  ├─→ Initialize feature engines
  └─→ Load trained model
      ↓
Fetch upcoming fixtures from API
      ↓
For each fixture:
  ├─→ Generate 162 features (same as training)
  ├─→ Predict probabilities
  └─→ Apply betting strategy
      ↓
Store predictions in Supabase
```

---

## 💡 Tips

### Training
- Start with a small sample (`--max-fixtures 500`) to test
- Use `--league-id` to focus on specific leagues
- Convert JSON to CSV once for 100x faster repeated runs
- Player features now use real team performance data

### Live Prediction
- Initialize once per session (loads historical data)
- 365 days of history ensures accurate Elo/form calculation
- Predictions use exact same feature generation as training
- Threshold strategy reduces bet frequency but increases accuracy

### Performance
- Initial JSON load: ~1 minute for 20K+ fixtures
- CSV conversion: One-time, saves 100x on future runs
- Live prediction init: ~3 minutes (fetches 1 year via API)
- Feature generation per match: ~0.1 seconds

---

## 📚 Documentation

### Production & Deployment
- `PRODUCTION_DEPLOYMENT_SUMMARY.md` - **Production deployment guide and performance metrics**
- `TRAINING_PIPELINE_UPDATED.md` - **Training pipeline with Option 3 configuration**
- `config/production_config.py` - **Centralized production settings**

### Features & Architecture
- `docs/FEATURE_FRAMEWORK.md` - Complete feature philosophy
- `docs/FEATURE_DICTIONARY.md` - All 162 features explained
- `docs/PLAYER_FEATURES_IMPLEMENTATION.md` - Real player features guide
- `docs/TRAINING_DATA_STRATEGY.md` - Point-in-time correctness
- `docs/DERIVED_XG.md` - Expected goals calculation

### Operations
- `docs/PNL_TRACKING.md` - PnL tracking & feature storage guide
- `docs/WEEKLY_RETRAIN_GUIDE.md` - Automated retraining setup

---

## 📝 Future Improvements / TODO

### Enable Real Lineup-Based Player Features

**Current Implementation:**
- Lineups are **deliberately NOT fetched** from SportMonks API for performance
- Player features use **team-level performance estimates** as fallback
- This keeps API calls fast (2-3x faster than with lineups)

**Why Disabled:**
```python
# src/data/sportmonks_client.py:225
# Note: lineups.details, events, formations are very heavy and slow
# Only include what's actually needed for features
includes = ["participants", "scores", "statistics", "state"]
# ❌ 'lineups' excluded
```

**To Enable Real Lineup Features:**

1. **Update API includes** in two places:
   ```python
   # src/data/sportmonks_client.py:227-232
   includes = [
       "participants",
       "scores",
       "statistics",
       "state",
       "lineups"  # ← Add this
   ]

   # scripts/predict_production.py:178
   'include': 'participants;league;state;odds;lineups'  # ← Add lineups
   ```

2. **Re-download ALL historical data** (lineups not in current JSON files)
   ```bash
   python3 scripts/backfill_historical_data.py \
     --start-date 2023-08-01 \
     --end-date 2024-05-31 \
     --output-dir data/historical
   ```

3. **Regenerate CSV** with lineup data
   ```bash
   python3 scripts/convert_json_to_csv.py
   ```

4. **Regenerate training features** (player features will change)
   ```bash
   python3 scripts/generate_training_data.py --output data/training_data.csv
   ```

5. **Retrain model** with new lineup-based features
   ```bash
   python3 scripts/train_production_model.py \
     --data data/training_data.csv \
     --output models/with_lineups/
   ```

**Trade-offs:**
- ✅ **Benefit**: Real player ratings, formation analysis, lineup quality metrics
- ❌ **Cost**: API calls 2-3x slower, full re-download required (hours)
- ⚠️ **Risk**: Lineups may not be available for all fixtures (especially lower leagues)

**Recommendation:**
Test with small date range first (1-2 months) to measure improvement before full re-download.

---

## 🎯 Next Steps

1. ✅ **Generate training data** for your target league/period
2. ✅ **Train model** with XGBoost using 162 features
3. ✅ **Test predictions** on upcoming matches
4. ✅ **Deploy to production** with automated daily predictions
5. ✅ **Setup weekly retraining** to keep model fresh
