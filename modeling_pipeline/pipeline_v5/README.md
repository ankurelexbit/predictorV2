# V5 Pipeline - Football Match Prediction System

Complete football match prediction system using a CatBoost + LightGBM ensemble model with 162 features from 3-pillar feature engineering framework.

## Key Changes from V4

| Aspect | V4 | V5 |
|--------|----|----|
| Model | XGBoost | CatBoost + LightGBM Ensemble |
| Thresholds | H=0.55, A=0.50, D=0.40 | H=0.60, A=0.45, D=0.35 (recalibrated) |
| Odds Filter | None | 1.5-3.5 |
| Outcomes | Primarily H/A | All 3 (H/D/A) viable |
| Database | Supabase | Local PostgreSQL |
| Recalibration | None | Every 8 weeks on 120-day window |

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export SPORTMONKS_API_KEY="your_api_key"
export DATABASE_URL="postgresql://ankurgupta@localhost/football_predictions"
```

### 2. Download Historical Data (First Time Only)

```bash
python3 scripts/backfill_historical_data.py --start-date 2022-08-01 --end-date 2026-01-31
python3 scripts/backfill_historical_data.py --start-date 2024-08-01 --end-date 2025-05-31 --skip-sidelined
```

### 3. Build Fixtures CSV (Required Before Training)

```bash
python3 scripts/build_fixtures_csv.py
```

Creates `data/processed/fixtures_with_stats.csv` (~2.4MB vs 36GB JSON).

### 4. Generate Training Data

```bash
python3 scripts/generate_training_data.py --output data/training_data.csv
```

### 5. Train Model

```bash
python3 scripts/train_model.py --data data/training_data.csv --version 1.0.0
python3 scripts/train_model.py --data data/training_data.csv --tune --trials 50
```

### 6. Train Bet Selector (One-Time, After Step 5)

```bash
# Extract market features from historical JSON
python3 scripts/extract_market_features.py

# Train GBM bet selector with walk-forward backtest
python3 scripts/train_bet_selector.py
python3 scripts/train_bet_selector.py --dry-run  # backtest only
```

### 7. Generate Live Predictions (3 Strategies Per Fixture)

```bash
python3 scripts/predict_live.py --days-ahead 7
python3 scripts/predict_live.py --days-ahead 7 --dry-run

# Backtest mode
python3 scripts/predict_live.py --start-date 2026-01-01 --end-date 2026-01-31
```

### 8. Update Results & Get PnL

```bash
python3 scripts/update_results.py --days-back 3
python3 scripts/get_pnl.py --days 30                 # all strategies combined
python3 scripts/get_pnl.py --days 30 --strategy hybrid  # specific strategy
```

### 8. Recalibrate Strategy (Every 8 Weeks)

```bash
python3 scripts/recalibrate_strategy.py
python3 scripts/recalibrate_strategy.py --dry-run          # preview only
python3 scripts/recalibrate_strategy.py --lookback-days 90  # custom window
```

## Betting Strategy

### Multi-Strategy System

Each fixture produces up to 3 independent bet decisions. Users can choose which strategy to follow.
Validated on 13,580 fixtures (2017-2025) with 6 walk-forward folds.

| Strategy | Description | Bets | ROI% | Losing Folds |
|----------|-------------|------|------|--------------|
| **threshold** | Probability thresholds (H60/D35/A45) + odds filter (1.5-3.5) | 3,890 | +4.1% | 2/6 |
| **hybrid** | Thresholds (no odds filter) + GBM selector at 0.55 confidence | 2,261 | +8.7% | 0/6 |
| **selector** | Pure GBM bet selector at 0.55 confidence | 3,463 | +7.7% | 0/6 |

### How It Works

The model outputs probabilities for all 3 outcomes: P(Home), P(Draw), P(Away).

- **threshold**: Bet when probability exceeds threshold AND odds are in 1.5-3.5 range
- **hybrid**: Bet when probability exceeds threshold AND GBM selector confirms (no odds filter — GBM handles it)
- **selector**: Bet when GBM selector says yes, on highest-probability outcome

### Configuration

All strategies are configured in `config/production_config.py` → `BETTING_STRATEGIES`:

```python
BETTING_STRATEGIES = {
    'threshold': {
        'enabled': True,
        'thresholds': {'home': 0.60, 'draw': 0.35, 'away': 0.45},
        'odds_filter': {'min': 1.5, 'max': 3.5, 'enabled': True},
    },
    'hybrid': {
        'enabled': True,
        'thresholds': {'home': 0.60, 'draw': 0.35, 'away': 0.45},
        'odds_filter': {'enabled': False},
        'selector': {'model_path': 'models/production/bet_selector.joblib', 'min_confidence': 0.55},
    },
    'selector': {
        'enabled': True,
        'selector': {'model_path': 'models/production/bet_selector.joblib', 'min_confidence': 0.55},
    },
}
```

Set `'enabled': False` on any strategy to disable it.

### GBM Bet Selector

A LightGBM classifier (max_depth=3, 150 trees) trained on 28 market features:
- Consensus odds, implied probabilities, market overround
- Sharp vs soft bookmaker spread (sharp: {2, 5, 34}, soft: {1, 3, 16, 20, 23, 29, 35})
- Model-vs-market edge, cross-market signals
- Profit-weighted training (winning bets weighted by payout)

### Recalibration Strategy

Optimal thresholds shift over time as league dynamics change. Recalibrating every 8 weeks
using the last 120 days of results improves ROI from 6.7% (fixed) to 9.8% (recalibrated)
while reducing losing months from 26% to 15%.

**Validated on 3 years of walk-forward testing (2023-2025, 5,322 predictions):**

| Approach | Bets | WR% | Profit | ROI% | Losing Months |
|----------|------|-----|--------|------|---------------|
| Fixed thresholds | 1,757 | 46.0% | $118 | 6.7% | 8/31 (26%) |
| Recalibrate every 8 weeks | 1,052 | 52.6% | $103 | 9.8% | 4/27 (15%) |

**How to recalibrate:**

```bash
# 1. Make sure recent results are updated
python3 scripts/update_results.py --days-back 60

# 2. Run recalibration (updates production_config.py automatically)
python3 scripts/recalibrate_strategy.py

# 3. Future predict_live.py runs will use the new thresholds
python3 scripts/predict_live.py --days-ahead 7
```

**Key findings from recalibration analysis:**

- Training window matters more than frequency. Use 120-180 days of lookback. Shorter windows (30-45d) overfit to noise.
- The odds filter (1.5-3.5) is stable across all analyses and should not be recalibrated.
- Expect ~30-40% of individual months to be losing months. This is normal for a profitable strategy.
- Realistic long-term ROI expectation: 6-10%.

### What NOT to Do

- Do not recalibrate more frequently than every 4 weeks (overfits to noise)
- Do not use less than 90 days of lookback data
- Do not change the odds filter range (1.5-3.5 is consistently optimal)
- Do not expect every month to be profitable (15-40% losing months is normal)

## Architecture

### Directory Structure

```
pipeline_v5/
├── config/
│   ├── production_config.py    # Betting thresholds, strategies, DB config
│   └── api_config.py           # SportMonks API configuration
├── src/
│   ├── data/
│   │   ├── json_loader.py      # Historical data loading (CSV/JSON)
│   │   └── sportmonks_client.py # SportMonks API client
│   ├── features/
│   │   ├── elo_calculator.py           # Elo ratings
│   │   ├── standings_calculator.py     # Point-in-time league standings
│   │   ├── pillar1_fundamentals.py     # ~109 fundamental features
│   │   ├── pillar2_modern_analytics.py # ~56 modern analytics features
│   │   ├── pillar3_hidden_edges.py     # ~26 hidden edge features
│   │   ├── player_features.py          # Player/lineup quality features
│   │   └── feature_orchestrator.py     # Coordinates all pillars
│   ├── market/
│   │   ├── market_feature_extractor.py  # 28 market features from odds
│   │   └── bet_selector.py              # GBM bet selector model
│   └── database/
│       └── db_client.py        # PostgreSQL client (multi-strategy)
├── scripts/
│   ├── backfill_historical_data.py # Download data from SportMonks API
│   ├── build_fixtures_csv.py       # Convert JSON to lightweight CSV
│   ├── generate_training_data.py   # Feature generation (162 features)
│   ├── train_model.py              # Model training
│   ├── extract_market_features.py   # Extract market features from historical JSON
│   ├── train_bet_selector.py        # Train GBM bet selector
│   ├── predict_live.py              # Live predictions (3 strategies per fixture)
│   ├── update_results.py            # Update actual results
│   ├── get_pnl.py                   # PnL reports (per-strategy filtering)
│   └── recalibrate_strategy.py      # Threshold recalibration (every 8 weeks)
├── models/
│   └── production/             # Trained models (.joblib)
├── data/
│   ├── historical/             # Raw JSON from API
│   ├── processed/
│   │   └── fixtures_with_stats.csv  # Lightweight fixture CSV (2.4MB)
│   └── training_data.csv       # Generated features (162 columns)
└── README.md
```

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
  → update_results.py → PnL tracking
  → recalibrate_strategy.py (every 8 weeks) → updated thresholds
```

### Feature Engineering (3-Pillar Framework) - 162 Total Features

**Pillar 1: Fundamentals (~109 features)**
- Elo ratings (10): current Elo, diff, momentum over 5/10 games
- League position & points (12): position, points, PPG, top 6/bottom 3 flags
- Recent form (15): points, goals, wins in last 3/5/10 games
- Head-to-head (8): historical H2H record, goals, wins
- Home advantage (5): home win rate, goals at home

**Pillar 2: Modern Analytics (~56 features)**
- Derived xG (25): expected goals from shot data, inside-box ratio, corner xG
- Shot analysis (15): total shots, on target, accuracy, big chance conversion
- Defensive intensity (12): tackles, interceptions, PPDA, tackle success rate
- Attack patterns (8): possession, corners, dangerous attacks

**Pillar 3: Hidden Edges (~26 features)**
- Momentum & trajectory (12): winning streaks, goal trends
- Fixture difficulty adjusted (10): opponent-strength weighted stats
- Player quality (10): lineup ratings, key players available, injuries
- Situational context (8): rest days, fixture congestion
- Draw parity (12): indicators for predicting draws

**Point-in-Time Correctness**: All features use only data available BEFORE the match date.

### Model Architecture

```
CatBoost Classifier (500 iter, depth=6, lr=0.03, balanced weights)
  + LightGBM Classifier (500 est, depth=6, lr=0.03, balanced weights)
  → Ensemble (Simple Average)
  → [P(Away), P(Draw), P(Home)]
```

Chronological split: 70% train, 15% val, 15% test.

## Operations Playbook

### Weekly

```bash
# 1. Generate predictions for upcoming week (all 3 strategies)
python3 scripts/predict_live.py --days-ahead 7

# 2. After matches finish, update results (all strategies updated automatically)
python3 scripts/update_results.py --days-back 7

# 3. Check PnL
python3 scripts/get_pnl.py --days 30                 # all strategies + comparison
python3 scripts/get_pnl.py --days 30 --strategy hybrid  # single strategy
```

### Every 8 Weeks: Recalibrate Thresholds

Lightweight — just shifts probability cutoffs based on recent results.

```bash
python3 scripts/recalibrate_strategy.py
```

### Every 3-6 Months: Retrain Main Model

Retrain at season boundaries when you have 1,000+ new fixtures. Monthly is overkill
(~200 fixtures = ~1% of training set, won't meaningfully shift weights). Features like
Elo, form, and standings already adapt in real-time.

| When | Why |
|------|-----|
| Aug/Sep (season start) | New promoted teams, summer transfers |
| Jan/Feb (mid-season) | Winter transfer window, half-season of new data |
| May/Jun (optional) | Full season complete, largest new data increment |

```bash
# 1. Backfill new data
python3 scripts/backfill_historical_data.py --start-date 2026-01-01 --end-date 2026-06-30
python3 scripts/build_fixtures_csv.py

# 2. Regenerate training data and retrain
python3 scripts/generate_training_data.py --output data/training_data.csv
python3 scripts/train_model.py --data data/training_data.csv --version 1.1.0

# 3. Retrain bet selector (depends on main model's CV probs)
python3 scripts/extract_market_features.py
python3 scripts/train_bet_selector.py --strategy gbm_0.55
```

Retrain sooner if PnL degrades noticeably or you add new features.

### When NOT to Retrain

- Don't retrain the main model monthly (too little new data for meaningful improvement)
- Don't retrain the bet selector on its own without retraining the main model first
  (selector features include model probabilities — they must stay aligned)
- Don't retrain the bet selector every 8 weeks alongside threshold recalibration
  (different cadence — selector needs more data to justify a retrain)

## Database Schema

Each fixture has up to 3 rows (one per strategy). Unique constraint on `(fixture_id, strategy, model_version)`.

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER NOT NULL,
    match_date TIMESTAMP NOT NULL,
    league_id INTEGER,
    home_team_id INTEGER NOT NULL,
    home_team_name VARCHAR(255),
    away_team_id INTEGER NOT NULL,
    away_team_name VARCHAR(255),
    pred_home_prob FLOAT NOT NULL,
    pred_draw_prob FLOAT NOT NULL,
    pred_away_prob FLOAT NOT NULL,
    predicted_outcome VARCHAR(10),
    strategy VARCHAR(50) DEFAULT 'threshold',  -- threshold, hybrid, selector
    bet_outcome VARCHAR(20),
    bet_probability FLOAT,
    bet_odds FLOAT,
    should_bet BOOLEAN DEFAULT FALSE,
    best_home_odds FLOAT,
    best_draw_odds FLOAT,
    best_away_odds FLOAT,
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    actual_result VARCHAR(10),
    bet_won BOOLEAN,
    bet_profit FLOAT,
    model_version VARCHAR(50),
    prediction_timestamp TIMESTAMP DEFAULT NOW()
);
-- UNIQUE INDEX on (fixture_id, strategy, model_version)
```

## Troubleshooting

1. **OOM during training data generation** - Run `build_fixtures_csv.py` first (36GB JSON to 2.4MB CSV)
2. **No predictions generated** - Check SPORTMONKS_API_KEY and TOP_5_LEAGUES filter
3. **Database connection failed** - Verify DATABASE_URL, ensure PostgreSQL is running
4. **Model not found** - Train with `train_model.py`, check `models/production/LATEST`
5. **Recalibration finds no valid config** - Need at least 30 predictions with results. Run `update_results.py` first.

## License

Internal use only.
