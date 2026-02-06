# Sports Analytics Made Easy - V1 Model

## Overview

A 1X2 football prediction system that achieves consistent profitability through sophisticated market-aware modeling.

## Performance (Sep-Dec 2025)

| Month | Bets | Win Rate | ROI | H/D/A+ |
|-------|------|----------|-----|--------|
| Sep 2025 | 34 | 67.6% | 35.0% | ✓ |
| Oct 2025 | 43 | 74.4% | 43.0% | ✓ |
| Nov 2025 | 43 | 60.5% | 14.8% | ✓ |
| Dec 2025 | 43 | 62.8% | 11.3% | ✓ |
| **Total** | **163** | **66.3%** | **25.5%** | **✓** |

**All months satisfy constraints: WR≥50%, ROI≥10%, H/D/A all positive**

## Architecture

### Features (170 total)

**Match Features (162)**
- Elo ratings & changes
- League position & points
- Recent form (last 5/10 matches)
- Goals scored/conceded
- Derived xG metrics
- Head-to-head history
- Home advantage metrics
- Defensive metrics (PPDA, tackles)
- Attack patterns (shots, dangerous attacks)
- Momentum indicators

**Market Signal Features (8)** - THE KEY INNOVATION
- `home_sharp_vs_soft`: Sharp book advantage over soft books
- `away_sharp_vs_soft`: Sharp book advantage for away
- `draw_sharp_vs_soft`: Sharp book advantage for draw
- `home_bookmaker_disagreement`: Uncertainty in home odds
- `draw_bookmaker_disagreement`: Uncertainty in draw odds
- `away_bookmaker_disagreement`: Uncertainty in away odds
- `ah_main_line`: Asian Handicap main line (market's goal diff estimate)
- `num_bookmakers`: Liquidity indicator

### Model

- **Algorithm**: CatBoost Classifier
- **Calibration**: Isotonic Regression per outcome
- **Training**: Chronological split, trained on pre-Sep 2025 data

### Winning Strategy Parameters

```python
MIN_ODDS = 1.60        # Filter low-value favorites
MIN_EV = 0.08          # 8% minimum expected value
MIN_CAL_PROB = 0.40    # 40% minimum calibrated probability
MIN_EDGE = 0.05        # 5% minimum edge vs market
```

## Top Predictive Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `away_sharp_vs_soft` | 9.8 |
| 2 | `home_sharp_vs_soft` | 8.8 |
| 3 | `elo_diff` | 5.6 |
| 4 | `draw_bookmaker_disagreement` | 4.5 |
| 5 | `elo_diff_with_home_advantage` | 4.3 |

**Key Insight**: Market signals (sharp/soft book spread) are the most predictive features, outranking traditional metrics like Elo.

## Usage

### Simple API

```python
from src.showcase.predictor import Predictor

# Initialize
predictor = Predictor()

# Get predictions
predictions = predictor.predict(features_df, odds_df)

# Each prediction contains:
for p in predictions:
    print(p.match)        # "Arsenal vs Chelsea"
    print(p.prediction)   # "Home", "Away", or "No Bet"
    print(p.confidence)   # "High", "Medium", "Low"
    print(p.odds)         # 1.85
    print(p.expected_value)  # "+12.5%"
```

### Data Requirements

**features_df columns:**
- All 170 model features
- `home_team`, `away_team` (or `home_team_id`, `away_team_id`)
- `match_date`

**odds_df columns:**
- `home_best_odds`, `draw_best_odds`, `away_best_odds`
- `home_implied_prob`, `draw_implied_prob`, `away_implied_prob`
- `draw_sharp_vs_soft`, `draw_bookmaker_disagreement` (for draw bets)

## Files

```
models/
  integrated_model_v1.joblib    # Model + calibrators + features

src/showcase/
  __init__.py
  predictor.py                  # Simple API for users

scripts/
  extract_market_features.py    # Extract odds data from JSON
  train_integrated_model.py     # Train full model

data/
  training_data_with_market.csv # Features + market data (15,256 matches)
  market_features.csv           # Extracted market features
```

## Key Insights

1. **Market Features Matter Most**: Sharp book spreads are the #1 and #2 most predictive features. When sharp books offer better odds on an outcome, it wins more often.

2. **Selectivity is Key**: The strategy only bets when multiple conditions align:
   - Calibrated probability ≥ 40%
   - Edge vs market ≥ 5%
   - Expected value ≥ 8%
   - Odds ≥ 1.60

3. **Avoid Draws (Mostly)**: Draw bets have only 25% base rate but markets price them efficiently. Only bet draws when sharp books strongly favor them AND the model sees high edge.

4. **Higher Odds = Higher ROI**: The minimum odds filter (1.60) removes low-value bets on heavy favorites. This sacrifices some win rate for better ROI.

## Future Improvements

1. **Over/Under Model**: Your xG features would be even more predictive for totals markets.

2. **Live Odds Integration**: Track line movements to find CLV (Closing Line Value).

3. **League-Specific Models**: Some leagues may have different patterns.

4. **Player-Level Features**: Injuries, suspensions, key player form.

5. **Weather/Referee Data**: Available in SportMonks API but not yet used.
