# V4 Pipeline - Complete Feature Generation System

## 📋 Overview

Complete **3-pillar feature generation pipeline** with **150 features** built from raw SportMonks JSON data.

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

## 🎯 Next Steps

1. Generate training data for your target league/period
2. Train model with XGBoost
3. Evaluate and iterate on features
4. Deploy for live predictions

## 💡 Tips

- Start with a small sample (`--max-fixtures 500`) to test
- Use `--league-id` to focus on specific leagues
- Statistics are embedded in fixture JSON (no separate files needed)
- All features respect point-in-time correctness
