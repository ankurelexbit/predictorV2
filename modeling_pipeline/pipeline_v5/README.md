# V5 Pipeline - Football Match Prediction System

CatBoost + LightGBM ensemble predicting match outcomes (1X2) and goals markets (O/U, BTTS) across Europe's top 5 leagues, with automated betting recommendations.

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export SPORTMONKS_API_KEY="your_api_key"

# 1. Download historical data (first time only)
python3 scripts/backfill_historical_data.py --start-date 2022-08-01 --end-date 2026-01-31

# 2. Build fixtures CSV (required — converts 36GB JSON to 2.4MB CSV)
python3 scripts/build_fixtures_csv.py

# 3. Generate training data (162 features)
python3 scripts/generate_training_data.py --output data/training_data.csv

# 4. Train 1X2 model
python3 scripts/train_model.py --data data/training_data.csv --version 1.0.0
python3 scripts/train_model.py --data data/training_data.csv --tune --trials 100  # with tuning

# 5. Train Poisson goals model (O/U, BTTS)
python3 scripts/train_goals_model.py --data data/training_data.csv --version 1.0.0

# 6. Train GBM bet selector
python3 scripts/extract_market_features.py
python3 scripts/train_bet_selector.py --strategy gbm_0.55

# 7. Generate predictions
python3 scripts/predict_live.py --days-ahead 7
python3 scripts/predict_live.py --days-ahead 7 --dry-run  # preview only
```

## Models

### 1X2 Outcome Model
CatBoost + LightGBM classifier ensemble averaging probabilities → [P(Away), P(Draw), P(Home)].
162 features, chronological 70/15/15 split, balanced class weights.

### Poisson Goals Model
CatBoost + LightGBM Poisson regression → lambda_home, lambda_away per match.
Dixon-Coles low-scoring correction (rho ~ -0.046) applied to the score matrix.
Derives: O/U 0.5-3.5, BTTS, handicap, correct score probabilities.

### GBM Bet Selector
LightGBM classifier (max_depth=3, 150 trees) on 28 market features (consensus odds, sharp/soft bookmaker spread, model-vs-market edge). Trained with walk-forward cross-validation on 13,580 fixtures.

## Betting Strategies

### 1X2 — Multi-Strategy System

Each fixture produces up to 3 independent bet decisions. Validated on 13,580 fixtures (2017-2025), 6 walk-forward folds.

| Strategy | How It Works | Bets | ROI | Losing Folds |
|----------|-------------|------|-----|--------------|
| **threshold** | P(outcome) > threshold + odds 1.3-3.5 | 3,890 | +4.1% | 2/6 |
| **hybrid** | P(outcome) > threshold + GBM confirms (no odds filter) | 2,261 | **+8.7%** | 0/6 |
| **selector** | Pure GBM selector on highest-prob outcome | 3,463 | +7.7% | 0/6 |

Default thresholds: H=0.60, D=0.35, A=0.45. Configured in `config/production_config.py` → `BETTING_STRATEGIES`.

### Goals Markets — Value Betting

Bet when model_prob - implied_prob > min_edge. Configured in `GOALS_BETTING_STRATEGIES`.

| Market | Min Edge | Min Prob | Odds Range |
|--------|----------|----------|------------|
| O/U 2.5 | 3% | 52% | 1.50 - 3.00 |
| BTTS | 3% | 52% | 1.50 - 3.00 |

### Recalibration (Every 8 Weeks)

Shifting thresholds based on recent results improves ROI from 6.7% (fixed) to 9.8%, reducing losing months from 26% to 15%. Uses 120-day lookback.

```bash
python3 scripts/recalibrate_strategy.py
python3 scripts/recalibrate_strategy.py --dry-run  # preview only
```

## Operations

### Weekly

```bash
python3 scripts/predict_live.py --days-ahead 7          # generate predictions
python3 scripts/update_results.py --days-back 7          # update results after matches
python3 scripts/get_pnl.py --days 30                     # 1X2 PnL (all strategies)
python3 scripts/get_pnl.py --days 30 --strategy hybrid   # single strategy
python3 scripts/get_market_pnl.py --days 30              # goals market PnL
```

### Every 3-6 Months: Retrain

Retrain at season boundaries (Aug/Sep, Jan/Feb) when 1,000+ new fixtures are available.

```bash
python3 scripts/backfill_historical_data.py --start-date 2026-01-01 --end-date 2026-06-30
python3 scripts/build_fixtures_csv.py
python3 scripts/generate_training_data.py --output data/training_data.csv
python3 scripts/train_model.py --data data/training_data.csv --tune --trials 100 --version 1.1.0
python3 scripts/train_goals_model.py --data data/training_data.csv --tune --trials 100 --version 1.1.0
python3 scripts/extract_market_features.py
python3 scripts/train_bet_selector.py --strategy gbm_0.55
```

## Architecture

### Data Flow

```
SportMonks API → Historical JSON (36GB) → build_fixtures_csv.py → CSV (2.4MB)
  → generate_training_data.py → training_data.csv (162 features)
  → train_model.py        → 1X2 ensemble (models/production/model_v*.joblib)
  → train_goals_model.py  → Poisson model (models/production/goals_model_v*.joblib)
  → predict_live.py       → PostgreSQL (predictions + market_predictions tables)
```

### 3-Pillar Feature Engineering (162 features)

- **Pillar 1 — Fundamentals (~109)**: Elo ratings, league position, form, H2H, home advantage
- **Pillar 2 — Modern Analytics (~56)**: Derived xG, shot analysis, defensive intensity, attack patterns
- **Pillar 3 — Hidden Edges (~26)**: Momentum, fixture difficulty, player quality, draw parity

All features are point-in-time correct (only data available before match date).

### Directory Structure

```
pipeline_v5/
├── config/
│   ├── production_config.py       # Thresholds, strategies, DB, leagues
│   └── api_config.py              # SportMonks API config
├── src/
│   ├── data/                      # Data loading (JSON/CSV, SportMonks API)
│   ├── features/                  # 3-pillar feature engineering
│   ├── goals/                     # Poisson goals model (O/U, BTTS)
│   ├── market/                    # Market features + GBM bet selector
│   └── database/                  # PostgreSQL client
├── scripts/                       # Training, prediction, PnL scripts
├── models/production/             # Trained models + LATEST pointers
└── data/                          # Historical JSON, processed CSV, training data
```

### Database

Two tables — `predictions` (1X2) and `market_predictions` (O/U, BTTS).
Each fixture has up to 3 rows in `predictions` (one per strategy).
Unique constraint: `(fixture_id, strategy, model_version)`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SPORTMONKS_API_KEY` | Yes | SportMonks API key for live data |
| `DATABASE_URL` | No | PostgreSQL connection string (has default) |

## Troubleshooting

- **OOM during training data generation** — Run `build_fixtures_csv.py` first
- **No predictions** — Check `SPORTMONKS_API_KEY` and `TOP_5_LEAGUES` filter
- **Model not found** — Check `models/production/LATEST` pointer
- **Recalibration fails** — Need 30+ predictions with results; run `update_results.py` first
