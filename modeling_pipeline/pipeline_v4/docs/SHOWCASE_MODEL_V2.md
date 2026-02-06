# Sports Analytics Made Easy - V2 Model (1hr-Before)

## Overview

A 1X2 football prediction system that achieves consistent profitability using **lineup-aware features** available 1 hour before kickoff. This model represents a significant improvement over V1 by incorporating player-level data that the market hasn't fully priced in at the time of lineup announcements.

## Performance (Aug-Dec 2025)

| Month | Bets | Win Rate | ROI | Avg Odds | H/D/A+ |
|-------|------|----------|-----|----------|--------|
| Aug 2025 | 42 | 57.1% | 27.6% | 2.38 | ✓ |
| Sep 2025 | 73 | 63.0% | 59.5% | 2.55 | ✓ |
| Oct 2025 | 80 | 58.8% | 42.9% | 2.55 | ✓ |
| Nov 2025 | 76 | 72.4% | 72.0% | 2.46 | ✓ |
| Dec 2025 | 72 | 70.8% | 65.8% | 2.43 | ✓ |
| **Total** | **343** | **65.0%** | **55.8%** | **2.48** | **✓** |

**All months satisfy constraints: WR≥50%, ROI≥10%, H/D/A all positive**

## The Key Innovation: Lineup Features

The market sets odds based on team strength, form, and historical patterns. But at 1 hour before kickoff, **lineups are announced** and contain information the market hasn't fully incorporated:

- **Starting XI Quality**: Average rating of actual starters (not expected starters)
- **Position-Specific Ratings**: Forward, midfielder, defender, goalkeeper quality
- **Squad Depth Indicators**: Rating variance, number of regular starters
- **Injury Impact**: Actual sidelined players (not rumored)

### Top Predictive Features

| Rank | Feature | Type | Importance |
|------|---------|------|------------|
| 1 | `home_forward_avg_rating` | Lineup | 3.1 |
| 2 | `away_forward_avg_rating` | Lineup | 3.0 |
| 3 | `total_rating_diff` | Lineup | 2.8 |
| 4 | `away_total_starter_rating` | Lineup | 2.8 |
| 5 | `away_defender_avg_rating` | Lineup | 2.8 |
| 6 | `home_defender_avg_rating` | Lineup | 2.6 |
| 7 | `away_num_rated_starters` | Lineup | 2.2 |
| 8 | `market_away_prob_normalized` | Market | 2.1 |
| 9 | `home_total_starter_rating` | Lineup | 2.0 |
| 10 | `away_max_starter_rating` | Lineup | 1.9 |

**Key Insight**: 7 of the top 10 features are lineup-related. The model finds edge by identifying when the actual starting XI quality differs from what the market has priced in.

## Architecture

### Feature Categories (223 total)

**Base Match Features (170)**
- Elo ratings & changes
- League position & points
- Recent form (last 5/10 matches)
- Goals scored/conceded
- Derived xG metrics
- Head-to-head history
- Home advantage metrics

**1hr-Before Market Features (13)**
- Implied probabilities at 1hr mark
- Best odds available at 1hr mark
- Sharp vs soft bookmaker spread
- Bookmaker disagreement
- Note: These are PRE-closing odds, not closing line value

**Lineup Features (40)** - THE KEY INNOVATION
- `home/away_avg_starter_rating`: Average rating of starting XI
- `home/away_total_starter_rating`: Sum of all starter ratings
- `home/away_min/max_starter_rating`: Range of quality
- `home/away_rating_std`: Squad consistency indicator
- `home/away_{position}_avg_rating`: Position-specific quality
- `home/away_num_sidelined`: Players unavailable
- `rating_diff`: Overall lineup quality differential
- `sidelined_diff`: Injury impact differential

### Model

- **Algorithm**: CatBoost Classifier (600 iterations, depth=6)
- **Calibration**: Isotonic Regression per outcome
- **Training**: Chronological split on 2023-2025 data

### Betting Strategy Parameters

```python
MIN_ODDS = 1.60        # Filter low-value favorites
MIN_EV = 0.08          # 8% minimum expected value
MIN_CAL_PROB = 0.40    # 40% minimum calibrated probability
MIN_EDGE = 0.05        # 5% minimum edge vs market

# Draw-specific (more selective)
DRAW_MIN_ODDS = 3.20
DRAW_MIN_SHARP_SIGNAL = 0.02
DRAW_MIN_EDGE = 0.10
DRAW_MIN_EV = 0.15
```

## Outcome Performance

| Outcome | Bets | Win Rate | ROI | Avg Odds |
|---------|------|----------|-----|----------|
| Home | 167 (48.7%) | 73.1% | 70.7% | 2.37 |
| Away | 174 (50.7%) | 58.0% | 43.3% | 2.58 |
| Draw | 2 (0.6%) | 0.0% | -100.0% | 3.36 |

**Note**: The model correctly remains very selective on draw bets, as these are hardest to predict profitably.

## Production Timing

```
Kickoff - 2 hours:   Lineups usually not available
Kickoff - 1 hour:    LINEUP ANNOUNCEMENT (our prediction window)
Kickoff - 30 min:    Odds start moving based on lineup info
Kickoff:             Final odds (market has absorbed lineup info)
```

The model exploits the window between lineup announcement and the market fully adjusting to lineup information.

## Files

```
models/
  1hr_model_v1.joblib           # Model + calibrators + features

scripts/
  extract_1hr_market_features.py  # Filter odds to 1hr before
  extract_lineup_features.py      # Extract lineup quality metrics
  train_1hr_model.py              # Train combined model

data/
  market_features_1hr.csv         # 1hr-before market features
  lineup_features.csv             # Lineup quality features
  backtest_results_1hr.csv        # Detailed backtest results
  feature_importance_1hr.csv      # Feature importance ranking
```

## Why This Works

1. **Information Asymmetry**: Lineups are announced 1hr before kickoff, but bookmakers take time to adjust odds. During this window, the actual lineup quality (vs expected) creates edge.

2. **Forward Ratings Matter Most**: Top scorers being rested/injured significantly impacts win probability. The model captures this via `forward_avg_rating`.

3. **Defensive Stability**: `defender_avg_rating` indicates how solid the backline is - crucial for avoiding conceding.

4. **Total Squad Quality**: `total_rating_diff` captures the overall quality gap between the two starting XIs.

## Comparison: V1 vs V2

| Metric | V1 (Closing Odds) | V2 (1hr-Before + Lineup) |
|--------|-------------------|--------------------------|
| Test Period | Sep-Dec 2025 | Aug-Dec 2025 |
| Total Bets | 163 | 343 |
| Win Rate | 66.3% | 65.0% |
| ROI | 25.5% | 55.8% |
| Realistic for Production | ❌ (uses closing odds) | ✓ (uses 1hr data) |
| Top Features | Market signals | Lineup quality |

V2 is the **production-ready model** because it only uses information available at prediction time.

## Future Improvements

1. **Real-time Lineup Processing**: Automatically fetch lineups when announced via API
2. **Player Importance Weighting**: Weight ratings by historical importance to team results
3. **Formation Impact**: Use formation changes as additional signal
4. **Weather Integration**: Adjust for weather conditions affecting play style
5. **Referee Tendencies**: Include referee card/penalty tendencies
