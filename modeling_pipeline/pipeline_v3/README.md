# Football Prediction Pipeline V3 - Complete Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Data Pipeline](#data-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Model Training](#model-training)
8. [Live Prediction](#live-prediction)
9. [Monitoring & Maintenance](#monitoring--maintenance)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This is a complete football match prediction system that:
- Downloads historical data from SportMonks API
- Engineers 296 features across 8 categories
- Trains ensemble models (XGBoost + LightGBM + Neural Network)
- Provides live predictions for upcoming matches
- Achieves **0.915-0.923 log loss** (target)

### Key Features
- ✅ **296 engineered features** (optimized to ~200)
- ✅ **Injury-aware predictions** (handles missing key players)
- ✅ **No player database needed** (team-level aggregates)
- ✅ **Live prediction compatible** (all features available)
- ✅ **Data leakage prevention** (temporal cutoffs enforced)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                             │
├─────────────────────────────────────────────────────────────┤
│  SportMonks API → SQLite Database → CSV Export              │
│  (fixtures, lineups, events, formations, odds, injuries,    │
│   standings from participants.meta)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                          │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Player Statistics (49 features)                   │
│  Phase 2: Match Events (32 features)                        │
│  Phase 3: Formations (12 features)                          │
│  Phase 4: Injuries (16 features) ⭐                         │
│  Phase 5: Betting Odds (6 features)                         │
│  Phase 6: Temporal (8 features)                             │
│  Phase 7: Feature Selection (~200 features)                 │
│  + V3 Baseline (173 features)                               │
│  = ~296 features → ~200 after selection                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING                             │
├─────────────────────────────────────────────────────────────┤
│  Phase 8: Ensemble Model                                    │
│  - XGBoost (60% weight)                                     │
│  - LightGBM (30% weight)                                    │
│  - Neural Network (10% weight)                              │
│  - Probability Calibration                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  LIVE PREDICTION                             │
├─────────────────────────────────────────────────────────────┤
│  Fetch upcoming fixtures → Generate features → Predict      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

### Required Software
- **Python:** 3.8+
- **SQLite:** 3.x (usually pre-installed)
- **Git:** For version control

### Required API Keys
- **SportMonks API Key:** Enterprise subscription
  - Get from: https://www.sportmonks.com/
  - Set in `.env` file

### Python Packages
```bash
# Core
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0

# ML Models
xgboost>=1.7.0
lightgbm>=3.3.0

# Database
sqlite3 (built-in)

# API
requests>=2.28.0
python-dotenv>=0.21.0

# Utilities
joblib>=1.2.0
tqdm>=4.64.0
```

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline
```

### Step 2: Set Up Environment

```bash
cd pipeline_v3

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create `.env` file:

```bash
# .env
SPORTMONKS_API_KEY=your_api_key_here
SPORTMONKS_BASE_URL=https://api.sportmonks.com/v3/football

# Database
DATABASE_PATH=data/football.db

# Model settings
MODEL_VERSION=v3
RANDOM_SEED=42
```

### Step 4: Verify Installation

```bash
python3 -c "
from src.features.enhanced_csv_feature_engine import EnhancedCSVFeatureEngine
print('✅ Installation successful!')
"
```

---

## 📊 Data Pipeline

### Overview

The data pipeline downloads historical data from SportMonks API and stores it in SQLite database, then exports to CSV for feature engineering.

### Step 1: Initialize Database

```bash
# Create database schema
python3 scripts/init_database.py
```

**Output:**
- `data/football.db` - SQLite database with tables:
  - `fixtures`
  - `lineups`
  - `events`
  - `formations`
  - `odds`
  - `sidelined`
  - `standings`

### Step 2: Download Historical Data

```bash
# Download data for specific leagues and seasons
python3 scripts/backfill_historical_data.py \
    --leagues 8 \
    --seasons 2016-2024 \
    --batch-size 100
```

**Parameters:**
- `--leagues`: League IDs (8 = Premier League, 384 = La Liga, etc.)
- `--seasons`: Season range (YYYY-YYYY)
- `--batch-size`: API requests per batch (default: 100)

**What it downloads:**
1. **Fixtures:** Match results, dates, teams
2. **Lineups:** Player lineups with detail_* stats (ratings, touches, passes)
3. **Events:** Goals, cards, substitutions with timestamps
4. **Formations:** Team formations (4-3-3, 4-4-2, etc.)
5. **Odds:** Pre-match betting odds (1x2)
6. **Sidelined:** Injuries and suspensions
7. **Standings:** League positions and points (from participants.meta) ⭐ **NEW**

**Progress tracking:**
```
Downloading fixtures: 100%|████████████| 3800/3800 [15:23<00:00]
Downloading lineups: 100%|█████████████| 3800/3800 [42:15<00:00]
Downloading events: 100%|██████████████| 3800/3800 [28:45<00:00]
...
✅ Download complete: 3800 fixtures
```

**Estimated time:**
- Premier League (1 season): ~2 hours
- Premier League (8 seasons): ~12-16 hours
- Multiple leagues: Scale accordingly

### Step 3: Export to CSV

```bash
# Export JSON to CSV files (V2 - Robust with validation)
python3 scripts/convert_json_to_csv_v2.py
```

**Features:**
- ✅ Comprehensive error handling and logging
- ✅ Data validation (numeric parsing, range checks)
- ✅ Automatic standings extraction from `participants.meta`
- ✅ Detailed conversion statistics
- ✅ Log file: `json_to_csv_conversion.log`


**Output:** `data/csv/`
- `fixtures.csv` (3800 rows)
- `lineups.csv` (83,600 rows - 22 players × 3800 matches)
- `events.csv` (152,000 rows - ~40 events × 3800 matches)
- `formations.csv` (7,600 rows - 2 teams × 3800 matches)
- `odds.csv` (11,400 rows - 3 outcomes × 3800 matches)
- `sidelined.csv` (varies)
- `standings.csv` (7,600 rows - 2 teams × 3800 matches) ⭐ **NEW**

> **Note:** The V2 converter automatically extracts standings from the API's `participants.meta` field. This provides **official league standings** (position, points, wins, draws, losses, GF, GA, GD) which are more accurate than calculating from match results. No separate conversion step needed!

### Step 4: Verify Data Quality

```bash
# Comprehensive data validation with random sampling
python3 scripts/validate_data.py
```

**What it validates:**
- ✅ **Fixtures:** Team IDs, scores, results match JSON
- ✅ **Standings:** Position, points, W/D/L, GF/GA/GD match `participants.meta` ⭐
- ✅ **Lineups:** Player count, IDs, starter status match JSON
- ✅ **Statistics:** Team count, possession totals are sane
- ✅ **Data Quality:** No missing columns, acceptable null rates, no duplicates

**Method:** Random sampling (20 fixtures) + cross-validation against original JSON files

---

## 🔧 Feature Engineering

### Overview

The feature engineering pipeline generates 296 features across 8 phases, then optimizes to ~200 features.

### Architecture

```python
EnhancedCSVFeatureEngine
├── ComprehensiveCSVFeatureEngine (V3 Baseline - 173 features)
├── PlayerStatisticsExtractor (Phase 1 - 49 features)
├── EventFeatureExtractor (Phase 2 - 32 features)
├── FormationFeatureExtractor (Phase 3 - 12 features)
├── InjuryFeatureExtractor (Phase 4 - 16 features) ⭐
├── OddsFeatureExtractor (Phase 5 - 6 features)
└── TemporalFeatureExtractor (Phase 6 - 8 features)
```

### Feature Categories

#### Phase 0: V3 Baseline (173 features)
- Elo ratings
- League position & points (from API standings) ⭐
- Form metrics (L3, L5, L10)
- Head-to-head statistics
- xG (expected goals)
- Shots, possession, passing
- Defense metrics
- Momentum & trajectory

**Note:** League position features now use **official standings from SportMonks API** (`participants.meta`) instead of calculating from match results. This provides:
- ✅ More accurate positions (handles all tiebreaker rules)
- ✅ Additional features: wins, draws, losses, GF, GA, GD
- ✅ 100x faster lookup vs calculation

#### Phase 1: Player Statistics (49 features)
```python
# Core stats (per team, L5 window):
- rating_avg_5, rating_max_5, rating_min_5, rating_std_5
- touches_avg_5, touches_max_5, touches_min_5
- passes_total_avg_5, pass_accuracy_avg_5
- duels_won_avg_5, duels_total_avg_5
- shots_avg_5, shots_on_target_avg_5
- key_passes_avg_5

# Derived metrics:
- top_3_rating_avg_5 (best 3 players)
- team_pass_completion_5
- team_duel_success_5

# Relative features:
- player_rating_diff_5 (home - away)
```

**Data source:** `lineups.csv` (detail_118, detail_80, etc.)

#### Phase 2: Match Events (32 features)
```python
# Goal timing (per team):
- goals_first_15min (early pressure)
- goals_15_45min (first half)
- goals_second_half
- goals_last_15min (late goals)
- late_goals_conceded (defensive lapses)
- early_pressure (binary)

# Discipline:
- yellow_cards, red_cards
- cards_per_match
- early_cards (0-30 min)
- discipline_score
- has_discipline_issue (binary)

# Substitutions:
- subs_before_60min (tactical)
- subs_after_75min (time wasting)
- avg_subs_per_match
- tactical_flexibility (binary)
```

**Data source:** `events.csv` (type_id: 14=goal, 15=yellow, 16=red, 18=sub)

#### Phase 3: Formations (12 features)
```python
# Per team:
- formation_consistency (% same formation)
- formation_changes (number of changes)
- primary_formation_encoded (most used)
- defensive_formation_rate (% 5-4-1, 4-5-1)
- attacking_formation_rate (% 4-3-3, 3-4-3)
- balanced_formation_rate (% 4-4-2)
```

**Data source:** `formations.csv`

#### Phase 4: Injuries (16 features) ⭐ CRITICAL
```python
# Per team:
- injuries_count (total unavailable)
- injuries_long_term (out > 4 weeks)
- injury_crisis (>= 3 injuries)
- key_players_missing (star players out)
- lineup_strength_ratio (available / full)
- attack_strength_impact (missing forwards)
- defense_strength_impact (missing defenders)
- midfield_strength_impact (missing midfielders)
```

**Key player identification:**
```python
# From historical lineups:
key_players = players with:
    - avg_rating > 7.5 (top performers)
    - games_played >= 10 (regulars)
    - avg_minutes > 70 (starters)
```

**Data source:** `sidelined.csv` + `lineups.csv`

#### Phase 5: Betting Odds (6 features)
```python
# Market probabilities:
- bookmaker_home_win_prob
- bookmaker_draw_prob
- bookmaker_away_win_prob
- bookmaker_favorite (1=home, 0=draw, -1=away)
- bookmaker_confidence (margin)
- market_efficiency (bookmaker margin)
```

**Temporal cutoff:** 24+ hours before match (prevents data leakage)

**Data source:** `odds.csv`

#### Phase 6: Temporal (8 features)
```python
# Per team:
- days_since_last_match (rest days)
- matches_in_7_days (fixture density)
- fixture_congestion_score (fatigue)
- season_phase (0=start, 1=end)
```

**Data source:** `fixtures.csv`

### Generate Training Data

```bash
# Generate features for all fixtures
python3 scripts/regenerate_training_data.py \
    --start-date 2016-08-01 \
    --end-date 2024-05-31 \
    --output data/processed/training_data.csv
```

**Process:**
1. Load CSV data
2. For each fixture:
   - Extract fixture details
   - Generate all 296 features
   - Validate for data leakage
   - Save to output
3. Progress tracking

**Output:**
```csv
fixture_id,date,home_team_id,away_team_id,result,home_elo,away_elo,...,home_injuries_count,away_injuries_count
5590,2016-03-12,52,30,1,1542.3,1489.7,...,2,0
...
```

**Estimated time:**
- 3800 fixtures: ~30-45 minutes
- Progress: ~2-3 fixtures/second

### Validate Feature Quality ⭐ **NEW**

```bash
# Comprehensive feature validation
python3 scripts/validate_features_data.py
```

**What it validates:**
- ✅ **Feature Completeness:** All expected features present (200-300 features)
- ✅ **Data Quality:** No nulls, infinities, or extreme outliers
- ✅ **Target Distribution:** Balanced classes (H/D/A)
- ✅ **Data Leakage:** No future data in features
- ✅ **Feature Correlations:** No redundant features (>0.95 correlation)

**Checks performed:**
1. Feature count in expected range (200-300)
2. Null values <10% per feature
3. No infinite values
4. No constant features (zero variance)
5. Extreme outliers <1% per feature
6. Target variable has valid values (H/D/A)
7. No perfect correlations with target (>0.99)
8. Highly correlated feature pairs identified

**Output:**
```
🎉 ALL VALIDATIONS PASSED!
Training data is ready for model training
```

---


## 🤖 Model Training

### Phase 7: Feature Selection

```bash
# Optimize features (296 → ~200)
python3 scripts/train_with_feature_selection.py
```

**Process:**
1. **Correlation Analysis**
   - Remove features with correlation > 0.95
   - Keeps first feature from each pair

2. **Feature Importance**
   - Train XGBoost model
   - Remove features with importance < 0.001

3. **Recursive Feature Elimination**
   - Cross-validation (5-fold)
   - Find optimal subset
   - Target: ~200 features

**Output:**
```
Original features: 296
After correlation filter: 267 features
After importance filter: 223 features
After RFE: 198 features

✅ Final feature count: 198
```

**Files created:**
- `models/feature_selector.pkl` - Selector for production
- `models/feature_importance.csv` - Importance report
- `models/xgboost_selected.pkl` - Optimized model

**Top features (example):**
```
Feature                                          Importance
home_elo                                         0.082341
away_elo                                         0.078923
home_rating_avg_5                                0.045123
home_key_players_missing                         0.038456
bookmaker_home_win_prob                          0.035789
home_form_5                                      0.032145
...
```

### Phase 8: Ensemble Model

```bash
# Train ensemble (XGBoost + LightGBM + NN)
python3 scripts/train_ensemble_model.py
```

**Models:**
1. **XGBoost (60% weight)**
   ```python
   XGBClassifier(
       n_estimators=500,
       max_depth=6,
       learning_rate=0.05,
       subsample=0.8,
       colsample_bytree=0.8
   )
   ```

2. **LightGBM (30% weight)**
   ```python
   LGBMClassifier(
       n_estimators=500,
       max_depth=6,
       learning_rate=0.05,
       num_leaves=31
   )
   ```

3. **Neural Network (10% weight)**
   ```python
   MLPClassifier(
       hidden_layer_sizes=(128, 64, 32),
       activation='relu',
       solver='adam',
       max_iter=500
   )
   ```

**Ensemble prediction:**
```python
final_pred = (
    0.60 * xgb_pred +
    0.30 * lgb_pred +
    0.10 * nn_pred
)
```

**Calibration:**
```python
# Isotonic calibration on validation set
calibrated = CalibratedClassifierCV(
    ensemble, method='isotonic', cv='prefit'
)
```

**Output:**
```
Training Ensemble Models
========================
1. Training XGBoost...
   Train Log Loss: 0.8234
   Val Log Loss: 0.9123

2. Training LightGBM...
   Train Log Loss: 0.8456
   Val Log Loss: 0.9234

3. Training Neural Network...
   Train Log Loss: 0.8789
   Val Log Loss: 0.9456

4. Calibrating probabilities...

FINAL RESULTS
=============
XGBoost Log Loss:     0.9123
LightGBM Log Loss:    0.9234
Neural Net Log Loss:  0.9456
Ensemble Log Loss:    0.9087

Ensemble Improvement: 0.0036

✅ Ensemble training complete!
```

**Files created:**
- `models/ensemble_model.pkl` - Production model

**Estimated training time:**
- Feature selection: ~15-30 minutes
- Ensemble training: ~20-40 minutes
- Total: ~45-70 minutes

---

## 🔮 Live Prediction

### Setup

```bash
# Ensure models are trained
ls models/
# Should see:
# - ensemble_model.pkl
# - feature_selector.pkl
```

### Predict Upcoming Matches

```bash
# Predict matches for tomorrow
python3 scripts/predict_live.py --date tomorrow
```

**Process:**
1. Fetch upcoming fixtures from API
2. For each fixture:
   - Generate all 296 features
   - Apply feature selection
   - Predict with ensemble
3. Output predictions

**Output:**
```
LIVE MATCH PREDICTIONS FOR 2026-01-29
======================================

Match: Liverpool vs Manchester United
  Home Win: 58.3%
  Draw:     23.4%
  Away Win: 18.3%
  Recommendation: BET HOME (confidence: HIGH)

Match: Arsenal vs Chelsea
  Home Win: 45.2%
  Draw:     28.9%
  Away Win: 25.9%
  Recommendation: NO BET (confidence: LOW)

...

Completed 10 predictions
```

### API Integration

```python
# Use in your application
from src.models.ensemble_model import load_ensemble
from src.features.enhanced_csv_feature_engine import EnhancedCSVFeatureEngine

# Load model
ensemble = load_ensemble('models/ensemble_model.pkl')
engine = EnhancedCSVFeatureEngine('data/csv')

# Generate features for upcoming match
features = engine.generate_features_for_fixture(
    fixture_id=upcoming_fixture_id,
    as_of_date='2026-01-29'
)

# Predict
prediction = ensemble.predict_proba([list(features.values())])

print(f"Home Win: {prediction[0][2]:.1%}")
print(f"Draw: {prediction[0][1]:.1%}")
print(f"Away Win: {prediction[0][0]:.1%}")
```

---

## 📈 Monitoring & Maintenance

### Performance Tracking

```bash
# Evaluate on recent matches
python3 scripts/evaluate_recent.py --days 30
```

**Metrics:**
- Log loss
- Accuracy
- Calibration
- ROI (if betting)

### Model Retraining

**When to retrain:**
- Every 3-6 months (seasonal changes)
- When performance degrades (log loss > 0.95)
- After major rule changes
- New data available

**Process:**
```bash
# 1. Download new data
python3 scripts/backfill_historical_data.py --seasons 2024-2025

# 2. Regenerate training data
python3 scripts/regenerate_training_data.py

# 3. Retrain models
python3 scripts/train_with_feature_selection.py
python3 scripts/train_ensemble_model.py

# 4. Validate
python3 scripts/validate_model.py
```

### Data Updates

```bash
# Daily: Update recent fixtures
python3 scripts/update_recent_data.py --days 7

# Weekly: Full sync
python3 scripts/sync_all_data.py
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. API Rate Limiting
**Error:** `429 Too Many Requests`

**Solution:**
```python
# In backfill_historical_data.py
RATE_LIMIT_DELAY = 2  # Increase delay between requests
```

#### 2. Missing Features
**Error:** `KeyError: 'home_rating_avg_5'`

**Solution:**
```bash
# Check if lineups data exists
python3 -c "
import pandas as pd
lineups = pd.read_csv('data/csv/lineups.csv')
print(f'Lineups: {len(lineups)} rows')
print(f'Columns: {lineups.columns.tolist()}')
"
```

#### 3. Memory Issues
**Error:** `MemoryError`

**Solution:**
```python
# Process in batches
python3 scripts/regenerate_training_data.py --batch-size 500
```

#### 4. Model Not Loading
**Error:** `FileNotFoundError: models/ensemble_model.pkl`

**Solution:**
```bash
# Verify models exist
ls -lh models/

# Retrain if missing
python3 scripts/train_ensemble_model.py
```

### Logs

```bash
# Check logs
tail -f logs/pipeline.log

# Debug mode
python3 scripts/regenerate_training_data.py --debug
```

---

## 📚 Additional Resources

### Documentation
- [SportMonks API Docs](https://docs.sportmonks.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/)

### Support
- GitHub Issues: [Create issue](https://github.com/your-repo/issues)
- Email: support@example.com

---

## 🎉 Quick Start Summary

```bash
# 1. Setup
cd pipeline_v3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your API key

# 3. Download data (12-16 hours)
python3 scripts/backfill_historical_data.py --leagues 8 --seasons 2016-2024

# 4. Export to CSV (5 minutes)
python3 scripts/export_to_csv.py

# 5. Generate features (30-45 minutes)
python3 scripts/regenerate_training_data.py

# 6. Train models (45-70 minutes)
python3 scripts/train_with_feature_selection.py
python3 scripts/train_ensemble_model.py

# 7. Predict!
python3 scripts/predict_live.py --date tomorrow
```

**Total time:** ~14-18 hours (mostly data download)

---

## ✅ Success Criteria

- ✅ Log loss < 0.923 (target: 0.915-0.923)
- ✅ Exceeds parent pipeline (0.9478)
- ✅ All 296 features generated successfully
- ✅ Feature selection reduces to ~200 features
- ✅ Ensemble improves over individual models
- ✅ Live predictions work without errors
- ✅ No data leakage detected

---

**Version:** 3.0  
**Last Updated:** 2026-01-28  
**Status:** Production Ready ✅
