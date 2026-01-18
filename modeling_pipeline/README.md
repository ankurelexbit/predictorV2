# Football Match Prediction Pipeline

A comprehensive machine learning pipeline for predicting football match outcomes using historical data, Elo ratings, and advanced feature engineering.

## Overview

This pipeline predicts match outcomes (Home/Draw/Away) and probabilities using:
- Historical match data from 6 major European leagues
- Elo rating system for team strength
- Data-driven feature engineering based on actual patterns
- Multiple models: Elo baseline, Dixon-Coles, XGBoost, and Ensemble

## Quick Start

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the complete pipeline
python 01_data_collection.py        # Download match data
python 02_process_raw_data.py       # Create matches.csv
python 03_feature_engineering.py    # Generate features
python 03d_data_driven_features.py  # Add advanced features
python 04_model_baseline_elo.py     # Train Elo model
python 05_model_dixon_coles.py      # Train Dixon-Coles model
python 06_model_xgboost.py          # Train XGBoost model
python 07_model_ensemble.py         # Create ensemble
python 08_evaluation.py             # Evaluate all models
```

## Project Structure

```
modeling_pipeline/
├── 01_data_collection.py      # Download historical match data
├── 02_process_raw_data.py     # Process raw CSV files into unified format
├── 03_feature_engineering.py  # Create base features (Elo, form, H2H)
├── 03d_data_driven_features.py # Add advanced features from data patterns
├── 04_model_baseline_elo.py  # Elo rating baseline model
├── 05_model_dixon_coles.py   # Dixon-Coles Poisson model
├── 06_model_xgboost.py       # XGBoost multiclass model
├── 07_model_ensemble.py      # Weighted ensemble of all models
├── 08_evaluation.py          # Model evaluation and comparison
├── 09_prediction_pipeline.py # Make predictions on new matches
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                 # Downloaded raw data
│   ├── processed/           # Processed CSV files
│   └── predictions/         # Model predictions
├── models/                  # Saved model files
├── logs/                    # Log files
└── docs/                    # Documentation

```

## Features

### Base Features (53 total)
- **Elo Ratings**: Team strength ratings with K=32
- **Form**: Points, goals for/against over last 3/5/10 games
- **Head-to-Head**: Historical results between teams
- **Rest Days**: Days since last match
- **League Position**: Current standings

### Advanced Features (20 additional)
Based on analysis of 14,019 matches, we discovered:
- **Optimal Activity**: Teams perform best with 2 games in 10 days
- **Team Variance**: High-variance teams (Newcastle, Atalanta) vs consistent teams (Juventus, Sevilla)
- **Tactical Flexibility**: Goal variance as a positive indicator
- **Momentum**: Activity + Form combination

## Models

1. **Elo Baseline** - Simple probability model using Elo ratings
2. **Dixon-Coles** - Bivariate Poisson with low-score correction
3. **XGBoost** - Gradient boosting with all features
4. **Ensemble** - Weighted combination optimized for log loss

## Data Sources

- **football-data.co.uk**: Historical results and odds (2019-2025)
- **Football-Data.org API**: Recent fixtures and results
- **The Odds API**: Current betting odds

### Supported Leagues
- Premier League (England)
- Championship (England)
- La Liga (Spain)
- Bundesliga (Germany)
- Serie A (Italy)
- Ligue 1 (France)

## Key Insights from Data

Our analysis of 14,019 matches revealed:
- Rotation effect is minimal (-0.059 goals after midweek games)
- Moderate activity (2 games/10 days) is optimal (+0.048 pts/game)
- High variance teams are identifiable and predictable
- "Unstable" teams actually perform better (+0.223 pts/game)

## CSV-Only Workflow

This pipeline works entirely with CSV files - no database required:
- `data/processed/matches.csv` - All historical matches
- `data/processed/features.csv` - Base engineered features
- `data/processed/features_data_driven.csv` - Enhanced features
- `data/processed/elo_ratings.csv` - Current team ratings

## Making Predictions

```python
# For upcoming matches
python 09_prediction_pipeline.py --date 2024-01-20

# For specific teams
python 09_prediction_pipeline.py --home "Liverpool" --away "Man City"
```

## Model Performance

Typical performance metrics:
- **Log Loss**: 0.95-1.05 (lower is better)
- **Accuracy**: 50-55%
- **ROI**: +2-5% on betting simulation

## Configuration

Edit `config.py` to:
- Set data directories
- Configure model parameters
- Add API keys (optional)
- Adjust Elo K-factor

## Tips for Best Results

1. **Regular Updates**: Run data collection weekly for latest results
2. **Feature Selection**: Not all features improve all models
3. **Ensemble Weights**: Optimize based on recent performance
4. **Betting Strategy**: Kelly Criterion with 0.25 fraction recommended

## Troubleshooting

**Missing data/processed/matches.csv**
- Run `python 02_process_raw_data.py` first

**Low model accuracy**
- This is normal - football is inherently unpredictable
- Focus on probability calibration, not just accuracy

**Slow feature engineering**
- The data-driven features script processes 14k matches
- Run once and reuse the output file

## Future Enhancements

- [ ] Live match data integration
- [ ] Weather data features
- [ ] Referee statistics
- [ ] Market inefficiency detection
- [ ] Deep learning models

## License

This project is for educational purposes. Please gamble responsibly.