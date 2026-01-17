# Football Prediction ML Pipeline

Complete ML pipeline for 1X2 football match predictions.

## Project Structure

```
football_prediction/
├── 01_data_collection.py      # Fetch historical data from multiple sources
├── 02_data_storage.py         # Database setup and data ingestion
├── 03_feature_engineering.py  # Build features (Elo, form, etc.)
├── 04_model_baseline_elo.py   # Elo-based probability model
├── 05_model_dixon_coles.py    # Dixon-Coles goal-based model
├── 06_model_xgboost.py        # Gradient boosting classifier
├── 07_model_ensemble.py       # Ensemble and calibration
├── 08_evaluation.py           # Backtesting and metrics
├── 09_prediction_pipeline.py  # Production inference pipeline
├── config.py                  # Configuration and API keys
├── utils.py                   # Shared utilities
├── requirements.txt           # Dependencies
└── data/                      # Local data storage
    ├── raw/                   # Raw downloaded data
    ├── processed/             # Cleaned data
    └── football.db            # SQLite database
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
cd football_prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Edit `config.py` and add your API keys:
- Football-Data.org (free tier available)
- API-Football (free tier: 100 requests/day)

### 4. Run Notebooks in Order
```bash
python 01_data_collection.py
python 02_data_storage.py
python 03_feature_engineering.py
python 04_model_baseline_elo.py
python 05_model_dixon_coles.py
python 06_model_xgboost.py
python 07_model_ensemble.py
python 08_evaluation.py
```

## Data Sources

| Source | Data Type | Cost | Rate Limit |
|--------|-----------|------|------------|
| football-data.co.uk | Historical CSV | Free | None |
| Football-Data.org API | Fixtures, Results | Free tier | 10/min |
| API-Football | Live odds, Lineups | Free: 100/day | 10/min |
| The Odds API | Betting odds | Free: 500/mo | None |

## Supported Leagues (MVP)

| League | Code | Country |
|--------|------|---------|
| Premier League | E0 / PL | England |
| Championship | E1 | England |
| La Liga | SP1 | Spain |
| Bundesliga | D1 | Germany |
| Serie A | I1 | Italy |
| Ligue 1 | F1 | France |

## Models Implemented

1. **Elo Rating System** - Team strength baseline
2. **Dixon-Coles** - Bivariate Poisson with low-score correction
3. **XGBoost Multiclass** - Feature-rich gradient boosting
4. **Calibrated Ensemble** - Combined predictions with isotonic calibration

## Key Metrics

- **Log Loss** (primary) - Measures probability quality
- **Brier Score** - Probabilistic accuracy
- **Calibration Error** - Reliability of probabilities
- **ROI** - Simulated betting returns vs market
