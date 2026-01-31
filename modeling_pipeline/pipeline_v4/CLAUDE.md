# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **V4 Pipeline**, a complete football match prediction system that generates 150+ features from raw SportMonks JSON data using a 3-pillar feature engineering framework. The pipeline processes historical match data, calculates point-in-time features, and trains XGBoost models to predict match outcomes (Home/Draw/Away).

## Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key for data downloads
export SPORTMONKS_API_KEY="your_api_key_here"
```

### Weekly Retrain (Production)
```bash
# Automated weekly retraining pipeline
# Downloads new data, regenerates CSV, trains model
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4

# For specific league only
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4 --league-id 8

# Setup cron for automatic weekly runs (every Sunday 2am)
# crontab -e
# 0 2 * * 0 cd /path/to/pipeline_v4 && python3 scripts/weekly_retrain_pipeline.py --weeks-back 4

# See docs/WEEKLY_RETRAIN_GUIDE.md for full automation setup
```

### Data Pipeline

**Download Historical Data:**
```bash
# Set API key
export SPORTMONKS_API_KEY="your_api_key_here"

# Download historical data for date range
python3 scripts/backfill_historical_data.py \
  --start-date 2023-08-01 \
  --end-date 2024-05-31 \
  --output-dir data/historical

# Options: --skip-lineups, --skip-sidelined, --workers N
```

**Convert JSON to CSV (Important for Performance):**
```bash
# One-time conversion - makes feature generation 100x faster
python3 scripts/convert_json_to_csv.py
# Creates: data/processed/fixtures.csv
```

**Generate Training Data:**
```bash
# Generate features for all downloaded data
python3 scripts/generate_training_data.py --output data/training_data.csv

# Filter by league (faster)
python3 scripts/generate_training_data.py \
  --league-id 8 \
  --start-date 2023-08-01 \
  --output data/training_pl_2023.csv

# Small sample for testing
python3 scripts/generate_training_data.py \
  --max-fixtures 500 \
  --output data/training_sample.csv
```

### Model Training

**Train Model:**
```bash
# Train with default draw-focused parameters
python3 scripts/train_model.py \
  --data data/training_data.csv \
  --output models/v4_xgboost.joblib

# Train with hyperparameter tuning
python3 scripts/train_model.py \
  --data data/training_data.csv \
  --output models/v4_xgboost.joblib \
  --tune
```

### Analysis & Testing

```bash
# Test core infrastructure
python3 scripts/test_core_infrastructure.py

# Test feature orchestrator
python3 scripts/test_feature_orchestrator.py

# Analyze data quality
python3 scripts/analyze_data_quality.py

# Analyze league distributions
python3 scripts/analyze_league_distributions.py

# Verify missing statistics
python3 scripts/verify_missing_stats.py

# Build cache (for faster loading)
python3 scripts/build_cache.py
```

## Architecture

### Core Philosophy: 3-Pillar Feature Framework

The system generates 150+ features organized into 3 pillars (see `docs/FEATURE_FRAMEWORK.md`):

1. **Pillar 1: Fundamentals (50 features)** - Time-tested metrics
   - Elo ratings, league position, recent form, H2H history, home advantage

2. **Pillar 2: Modern Analytics (60 features)** - Science-backed metrics
   - Derived xG, shot analysis, defensive intensity, attack patterns

3. **Pillar 3: Hidden Edges (40 features)** - Advanced metrics
   - Momentum indicators, fixture-adjusted metrics, player quality, context

### Data Flow

```
SportMonks API
      ↓
[backfill_historical_data.py]
      ↓
JSON files (data/historical/fixtures/*.json)
      ↓
[convert_json_to_csv.py] (optional but recommended)
      ↓
CSV file (data/processed/fixtures.csv)
      ↓
[JSONDataLoader]
      ↓
[FeatureOrchestrator]
      ├─→ [Pillar1FundamentalsEngine]
      ├─→ [Pillar2ModernAnalyticsEngine]
      └─→ [Pillar3HiddenEdgesEngine]
      ↓
Training Dataset (data/training_data.csv)
      ↓
[XGBoostFootballModel]
      ↓
Trained Model (models/v4_xgboost.joblib)
```

### Key Components

**Data Loading (`src/data/`)**
- `json_loader.py`: Streaming JSON parser using ijson for efficient loading of 1.3GB+ files
  - Loads fixtures from JSON files or pre-converted CSV (100x faster)
  - Caches full fixture data for statistics access
  - Provides point-in-time queries (fixtures before date, team fixtures, etc.)
- `sportmonks_client.py`: API client for downloading historical data

**Feature Engineering (`src/features/`)**
- `feature_orchestrator.py`: Coordinates all 3 pillars to generate complete feature sets
  - Initializes data loader, standings calculator, Elo calculator
  - Generates features for single fixtures or entire datasets
  - Handles filtering by league, date range, etc.
- `standings_calculator.py`: Calculates point-in-time league standings
- `elo_calculator.py`: Team rating system (K-factor=32, home advantage=35)
- `pillar1_fundamentals.py`: Elo, standings, form, H2H, home advantage (50 features)
- `pillar2_modern_analytics.py`: Derived xG, shots, defense, attacks (60 features)
- `pillar3_hidden_edges.py`: Momentum, fixture-adjusted, player quality, context (40 features)

**Modeling (`src/models/`)**
- `xgboost_model.py`: Wrapper for XGBoost classifier
  - Multi-class prediction (0=Away, 1=Draw, 2=Home)
  - Calibration support (isotonic/sigmoid)
  - Feature importance analysis

### Critical Design Principles

**Point-in-Time Correctness**
- All features must use only data available BEFORE the match date
- No data leakage from future information
- See `docs/TRAINING_DATA_STRATEGY.md` for details on historical reconstruction

**Performance Optimization**
- Initial JSON load: ~1 minute for 20K+ fixtures from 258 files
- Solution 1: Convert to CSV once (`convert_json_to_csv.py`) for 100x speedup
- Solution 2: Use cache (`build_cache.py`) for instant loading
- Solution 3: Filter by league/date to reduce fixtures processed

**Embedded Statistics**
- Statistics and lineups are embedded in fixture JSON
- No separate API calls needed during feature generation
- Access via `JSONDataLoader.get_statistics(fixture_id)` and `get_lineups(fixture_id)`

### Data Directory Structure

```
data/
├── historical/          # Raw JSON from SportMonks
│   ├── fixtures/       # Fixture JSON files (by date)
│   ├── sidelined/      # Injury/suspension data
│   └── metadata/       # Standings metadata
├── processed/          # Converted CSV files
│   └── fixtures_with_stats.csv  # Pre-processed fixture data
├── cache/              # Pickle caches for instant loading
│   ├── fixtures_df.pkl
│   └── fixtures_dict.pkl
└── training_data.csv   # Generated feature vectors
```

### Model Training Workflow

1. **Data Split**: Chronological 70% train / 15% validation / 15% test
2. **Target Encoding**: {'A': 0, 'D': 1, 'H': 2}
3. **Hyperparameters**: Draw-focused tuning (conservative depth, high min_child_weight)
4. **Calibration**: Isotonic regression on validation set
5. **Evaluation**: Log loss, accuracy, confusion matrix, feature importance

## Development Guidelines

**When Adding Features:**
- Add to appropriate pillar engine (`pillar1_fundamentals.py`, `pillar2_modern_analytics.py`, or `pillar3_hidden_edges.py`)
- Ensure point-in-time correctness (only use data before `as_of_date`)
- Update `docs/FEATURE_DICTIONARY.md` with feature description
- Test with small sample first (`--max-fixtures 500`)

**When Modifying Data Loading:**
- JSONDataLoader uses streaming parser (ijson) to handle large files
- Always preserve point-in-time query capability
- Test with both JSON and CSV sources
- Update cache logic if changing data structure

**When Training Models:**
- Always use chronological splits (never random splits)
- Exclude metadata columns from features: `fixture_id`, `home_team_id`, `away_team_id`, `season_id`, `league_id`, `match_date`, `home_score`, `away_score`, `result`, `target`
- Start with default parameters before tuning
- Monitor draw prediction rate (should be 20-30% of predictions)

**Performance Notes:**
- Start with league-filtered datasets (`--league-id 8`) for faster iteration
- Use CSV conversion for repeated feature generation on same data
- Build cache for production use with frequent orchestrator initialization
- Limit fixture count during testing (`--max-fixtures 500`)

## Common Workflows

**Full Pipeline from Scratch:**
```bash
# 1. Download data
python3 scripts/backfill_historical_data.py --start-date 2023-08-01 --end-date 2024-05-31

# 2. Convert to CSV (optional but recommended)
python3 scripts/convert_json_to_csv.py

# 3. Generate training data
python3 scripts/generate_training_data.py --output data/training_data.csv

# 4. Train model
python3 scripts/train_model.py --data data/training_data.csv --output models/v4_xgboost.joblib
```

**Quick Testing:**
```bash
# Test with small sample
python3 scripts/generate_training_data.py --max-fixtures 500 --output data/test_sample.csv
python3 scripts/train_model.py --data data/test_sample.csv --output models/test_model.joblib
```

**League-Specific Model:**
```bash
# Premier League only
python3 scripts/generate_training_data.py \
  --league-id 8 \
  --start-date 2023-08-01 \
  --output data/training_pl.csv

python3 scripts/train_model.py \
  --data data/training_pl.csv \
  --output models/pl_model.joblib
```

## Important Files

- `README.md`: Quick start guide and feature overview
- `docs/FEATURE_FRAMEWORK.md`: Complete feature philosophy and definitions
- `docs/TRAINING_DATA_STRATEGY.md`: Point-in-time correctness and historical reconstruction
- `docs/DERIVED_XG.md`: Expected goals calculation methodology
- `docs/FEATURE_DICTIONARY.md`: Complete feature catalog
- `docs/BACKFILL_GUIDE.md`: Historical data download guide
