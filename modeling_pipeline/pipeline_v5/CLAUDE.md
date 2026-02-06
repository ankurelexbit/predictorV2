# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Football match prediction system using CatBoost + LightGBM ensemble with 150+ features. Predicts match outcomes (Home/Draw/Away) and makes betting recommendations with real odds integration.

## Common Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export SPORTMONKS_API_KEY="your_key"

# Step 1: Download historical data from SportMonks API
python3 scripts/backfill_historical_data.py --start-date 2022-08-01 --end-date 2026-01-31
python3 scripts/backfill_historical_data.py --start-date 2024-08-01 --end-date 2025-05-31 --skip-sidelined

# Step 2: Convert JSON to lightweight CSV (REQUIRED - avoids OOM)
python3 scripts/build_fixtures_csv.py  # Creates data/processed/fixtures_with_stats.csv (2.4MB)

# Step 3: Generate training data (162 features, ~9 min for 18k fixtures)
python3 scripts/generate_training_data.py --output data/training_data.csv

# Step 4: Train model
python3 scripts/train_model.py --data data/training_data.csv --version 1.0.0
python3 scripts/train_model.py --data data/training_data.csv --tune --trials 50  # with tuning

# Generate live predictions (requires SPORTMONKS_API_KEY)
python3 scripts/predict_live.py --days-ahead 7 --strategy conservative

# Update results and check performance
python3 scripts/update_results.py --days-back 3
python3 scripts/get_pnl.py --days 30

# Recalibrate betting thresholds (every 8 weeks, uses last 120 days)
python3 scripts/recalibrate_strategy.py
python3 scripts/recalibrate_strategy.py --dry-run  # preview only
```

## Architecture

### 3-Pillar Feature Engineering (162 features)
- **Pillar 1 (~109 features)**: Elo ratings, league position, form, H2H, home advantage
- **Pillar 2 (~56 features)**: Derived xG, shot analysis, defensive intensity, attack patterns
- **Pillar 3 (~26 features)**: Momentum, fixture difficulty, player quality, draw parity

### Data Flow
```
SportMonks API (backfill_historical_data.py)
  → Historical JSON (data/historical/fixtures/) [36GB]
  → build_fixtures_csv.py
  → fixtures_with_stats.csv [2.4MB]
  → FeatureOrchestrator (generate_training_data.py)
  → training_data.csv [162 features]
  → train_model.py → CatBoost + LightGBM ensemble
  → predict_live.py → PostgreSQL (predictions table)
```

### Key Modules
- `config/production_config.py`: Central config (thresholds, strategies, DB, leagues)
- `config/api_config.py`: SportMonks API configuration (rate limits, leagues, seasons)
- `src/data/sportmonks_client.py`: SportMonks API client with caching and rate limiting
- `src/data/json_loader.py`: Loads historical data (prefers CSV, falls back to JSON)
- `src/features/feature_orchestrator.py`: Coordinates all 3 pillars for feature generation
- `src/features/standings_calculator.py`: Point-in-time league standings from fixtures
- `src/database/db_client.py`: PostgreSQL client for predictions CRUD
- `scripts/build_fixtures_csv.py`: Converts 36GB JSON → 2.4MB CSV (required step)

### Model
- CatBoost (500 iter, depth=6, lr=0.03) + LightGBM (500 est, depth=6, lr=0.03)
- Simple probability averaging for ensemble
- Output: [P(Away), P(Draw), P(Home)]
- Chronological split: 70% train, 15% val, 15% test

## Configuration

All production settings in `config/production_config.py`:
- `THRESHOLDS`: Probability thresholds for betting (default: H=0.60, A=0.45, D=0.35)
- `ODDS_FILTER`: Odds range filter (default: 1.5-3.5)
- `TOP_5_LEAGUES`: [8, 82, 384, 564, 301] (EPL, Bundesliga, Serie A, Ligue 1, La Liga)
- `STRATEGY_PROFILES`: conservative, selective, high_volume, tight
- Recalibrate thresholds every 8 weeks via `scripts/recalibrate_strategy.py`
- `EloConfig`: K_FACTOR=32, HOME_ADVANTAGE=35, INITIAL_ELO=1500

## Environment Variables

- `SPORTMONKS_API_KEY`: Required for live predictions and result updates
- `DATABASE_URL`: PostgreSQL connection (defaults to config value if not set)

## Gotchas

- **Must run `build_fixtures_csv.py` before `generate_training_data.py`** - Otherwise OOM with 36GB JSON
- Feature generation requires historical JSON files in `data/historical/fixtures/`
- Null features are filled with 0 (missing data for new teams/limited history)
- Trained models are ~100MB+; stored in `models/production/` with LATEST pointer
- Elo/standings calculations require complete chronological history for accuracy
- Point-in-time correctness: all features use only data available BEFORE match date
