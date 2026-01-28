# Pipeline V3 - Quick Start Guide

## 🚀 Getting Started

### 1. Setup Environment

```bash
cd pipeline_v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your SPORTMONKS_API_KEY
```

### 2. Download Historical Data
```bash
# Download missing data (e.g., 2020-2025)
python scripts/backfill_historical_data.py \
    --start-date 2020-01-01 \
    --end-date 2025-05-31
```
**Output:** `data/historical/` (JSON files)

### 3. Process Data (CSV Pipeline) 🚀
Convert raw JSON to optimized CSVs for 1000x faster processing:

```bash
# 1. Convert JSON to CSV
python scripts/convert_to_csv.py

# 2. Validate Data Integrity
python scripts/validate_json_to_csv.py
```
**Output:** `data/csv/` (fixtures.csv, statistics.csv, lineups.csv)

### 4. Generate Training Data
Generate the complete 150-feature dataset using the comprehensive engine:

```bash
python scripts/generate_complete_training_data.py \
    --output data/csv/training_data_complete.csv
```

This will:
- ✅ Load CSV data (instantly)
- ✅ Calculate season-aware standings
- ✅ Generate 150 advanced features per match (Fundamentals, Modern Analytics, Hidden Edges)
- ✅ Save training dataset

**Time:** ~21 minutes for 18K fixtures  
**Output:** `data/csv/training_data_complete.csv`

### 5. Validate Features
Perform comprehensive sanity checks on the generated features:

```bash
python scripts/validate_features.py \
    --input data/csv/training_data_complete.csv
```

### 6. Train Model (Next Step)

```bash
# Coming soon: scripts/train_model.py
```

---

## 📁 Project Structure

```
pipeline_v3/
├── config/              # Configuration files
│   ├── api_config.py   # SportMonks API settings
│   ├── feature_config.py  # Feature engineering parameters
│   └── database_config.py # Database settings
│
├── src/
│   ├── data/           # Data ingestion
│   │   └── sportmonks_client.py  # API client
│   │
│   ├── pipeline/       # Feature Generation Pipeline
│   │   ├── data_loader.py          # Historical data loader
│   │   ├── standings_calculator.py # Season-aware standings
│   │   ├── elo_tracker.py          # Chronological Elo
│   │   ├── pillar1_fundamentals.py # Fundamental features
│   │   ├── pillar2_modern_analytics.py # xG & analytics
│   │   ├── pillar3_hidden_edges.py # Advanced features
│   │   └── feature_orchestrator.py # Orchestrator
│   │
│   ├── models/         # Model training (coming soon)
│   ├── betting/        # Betting strategy (coming soon)
│   └── utils/          # Utilities
│
├── scripts/            # Executable scripts
│   ├── backfill_historical_data.py  # Download data
│   ├── convert_to_csv.py            # JSON to CSV
│   ├── generate_complete_training_data.py # Main generator
│   ├── validate_features.py         # Feature validation
│   └── ...
│
├── docs/               # Documentation
│   ├── README.md
│   ├── FEATURE_FRAMEWORK.md
│   ├── FEATURE_DICTIONARY.md
│   ├── FEATURE_VALIDATION_GUIDE.md
│   └── ...
```

---

## 🎯 What's Implemented

### ✅ Complete
- Configuration system
- SportMonks API client & Historical data backfill
- JSON to CSV conversion pipeline
- **Complete Feature Generation Pipeline**:
    - HistoricalDataLoader
    - SeasonAwareStandingsCalculator
    - EloTracker (Chronological)
    - Pillar 1 Engine (Fundamentals)
    - Pillar 2 Engine (Modern Analytics)
    - Pillar 3 Engine (Hidden Edges)
    - Feature Orchestrator
- **Feature Validation Suite**:
    - Comprehensive sanity checks
    - Detailed reporting
- Database schema

### 🚧 In Progress
- Model training pipeline (Baseline, Dixon-Coles, XGBoost, Stacking)

### 📋 Planned
- Betting strategy module
- Live prediction system
- Monitoring & logging
- Web dashboard

---

## 📊 Features Overview

**Total Features:** 150

### Pillar 1: Fundamentals (50 features)
- Elo Ratings (10)
- League Position & Points (12)
- Recent Form (15)
- Head-to-Head (8)
- Home Advantage (5)

### Pillar 2: Modern Analytics (60 features)
- Derived xG (25)
- Shot Analysis (15)
- Defensive Intensity (12)
- Attack Patterns (8)

### Pillar 3: Hidden Edges (40 features)
- Momentum & Trajectory (12)
- Fixture-Adjusted (10)
- Player Quality Proxies (10)
- Situational Context (8)

---

## ⚙️ Configuration

### API Settings (`config/api_config.py`)
- API key, rate limiting, retry logic

### Feature Settings (`config/feature_config.py`)
- Elo parameters (K=32, HA=35)
- xG coefficients (Inside Box=0.12, Big Chance=0.35, etc.)
- Rolling window sizes (5, 10 matches)

### Database Settings (`config/database_config.py`)
- Connection string, table names

---

## 📚 Documentation

- **[FEATURE_FRAMEWORK.md](docs/FEATURE_FRAMEWORK.md)** - Complete feature specifications
- **[FEATURE_VALIDATION_GUIDE.md](docs/FEATURE_VALIDATION_GUIDE.md)** - Validation script usage
- **src/pipeline/README.md** - Detailed pipeline documentation
- **validation_report.md** - Data quality report

---

## 🚀 Next Steps

1. ✅ Download historical data
2. ✅ Generate training features (150 features)
3. ✅ Validate feature quality
4. ⏳ Train Models:
    - Baseline Elo
    - Dixon-Coles
    - XGBoost
    - Stacking Ensemble
5. ⏳ Implement betting strategy
6. ⏳ Deploy live predictions

---

## 📞 Support

- Check logs: `backfill.log`, `feature_generation.log`, `feature_validation.log`
- Review documentation in `docs/`

---

**Version:** 3.1
**Status:** Development
**Last Updated:** January 28, 2026
