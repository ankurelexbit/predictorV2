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
# Download 2024-2025 season data
python scripts/backfill_historical_data.py \
    --start-date 2024-08-01 \
    --end-date 2025-05-31
```

This will download:
- ✅ All fixtures
- ✅ Match statistics
- ✅ Lineups
- ✅ Injury/suspension data

**Time:** ~3-5 minutes per season  
**Output:** `data/historical/`

### 3. Set Up Database (Optional)

```bash
# If using PostgreSQL/Supabase
psql -U your_user -d your_database -f scripts/create_database.sql
```

### 3. Generate Training Features

```bash
# Process downloaded data and create feature vectors
python scripts/generate_training_features.py \
    --data-dir data/historical \
    --output training_features.csv
```

This will:
- ✅ Load all historical data
- ✅ Calculate Elo ratings chronologically
- ✅ Generate 100-140 features per match
- ✅ Create training-ready CSV

**Time:** ~8-10 minutes per season  
**Output:** `training_features.csv`

### 4. Train Model (Coming Soon)

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
│   ├── features/       # Feature engineering
│   │   ├── elo_calculator.py     # Elo ratings
│   │   ├── derived_xg.py         # xG calculation
│   │   ├── form_calculator.py    # Form metrics
│   │   ├── h2h_calculator.py     # H2H analysis
│   │   ├── shot_analyzer.py      # Shot patterns
│   │   ├── defensive_metrics.py  # Defensive stats
│   │   └── feature_pipeline.py   # Orchestrator
│   │
│   ├── models/         # Model training (coming soon)
│   ├── betting/        # Betting strategy (coming soon)
│   └── utils/          # Utilities
│
├── scripts/            # Executable scripts
│   ├── backfill_historical_data.py  # Download data
│   └── create_database.sql          # Database schema
│
├── docs/               # Documentation
│   ├── README.md
│   ├── FEATURE_FRAMEWORK.md
│   ├── FEATURE_DICTIONARY.md
│   ├── DERIVED_XG.md
│   ├── BACKFILL_GUIDE.md
│   └── ...
│
├── notebooks/          # Jupyter notebooks
├── tests/              # Unit tests
├── requirements.txt    # Dependencies
└── .env.example        # Environment template
```

---

## 🎯 What's Implemented

### ✅ Complete
- Configuration system
- SportMonks API client (rate limiting, caching, retries)
- Elo rating calculator
- Derived xG calculator
- Form calculator
- H2H calculator
- Shot analyzer
- Defensive metrics calculator
- Feature pipeline orchestrator
- Database schema
- Historical data backfill script

### 🚧 In Progress
- Training feature generation
- Database integration
- Model training pipeline

### 📋 Planned
- Betting strategy module
- Live prediction system
- Monitoring & logging
- Web dashboard

---

## � Features Overview

**Total Features:** 140-190

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

### Pillar 3: Hidden Edges (40-80 features)
- Momentum & Trajectory (12)
- Fixture-Adjusted (10)
- Player Quality (25)
- Situational Context (8)

---

## � Configuration

### API Settings (`config/api_config.py`)
- API key and base URL
- Rate limiting (3,000 req/min)
- Retry logic
- Caching settings

### Feature Settings (`config/feature_config.py`)
- Elo parameters (K=32, HA=35)
- xG coefficients
- Rolling window sizes
- Feature groups

### Database Settings (`config/database_config.py`)
- Connection string
- Table names
- Batch sizes

---

## � Documentation

- **[FEATURE_FRAMEWORK.md](docs/FEATURE_FRAMEWORK.md)** - Complete feature specifications
- **[FEATURE_DICTIONARY.md](docs/FEATURE_DICTIONARY.md)** - Feature data sources
- **[DERIVED_XG.md](docs/DERIVED_XG.md)** - xG calculation methodology
- **[BACKFILL_GUIDE.md](docs/BACKFILL_GUIDE.md)** - Data download guide
- **[HISTORICAL_DATA_RESEARCH.md](docs/HISTORICAL_DATA_RESEARCH.md)** - API research
- **[API_DATA_MAPPING.md](docs/API_DATA_MAPPING.md)** - API endpoints

---

## 🎓 Key Design Decisions

1. **Complete Independence** - No external AI models or paid add-ons
2. **Derived xG** - Calculate from base statistics (saves $1,800-3,600/year)
3. **Home Advantage** - 35 points (calibrated for modern football)
4. **Historical Data** - All 165-190 features available from API
5. **Modular Architecture** - Easy to test, maintain, and extend

---

## 🚀 Next Steps

1. ✅ Download historical data (2022-2025)
2. ⏳ Generate training features
3. ⏳ Train XGBoost model
4. ⏳ Implement betting strategy
5. ⏳ Deploy live predictions

---

## 📞 Support

- Check logs: `backfill.log`
- Review documentation in `docs/`
- See examples in `notebooks/`

---

**Version:** 3.0  
**Status:** Development  
**Last Updated:** January 25, 2026
