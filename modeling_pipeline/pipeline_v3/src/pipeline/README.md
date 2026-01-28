# Complete Feature Generation Pipeline

## 🎯 Overview

A production-ready feature generation pipeline that processes raw historical football data and generates **150 comprehensive features** for machine learning model training.

**Key Features:**
- ✅ **Point-in-time correctness** - No data leakage
- ✅ **Season-aware standings** - League-wise and season-wise calculation
- ✅ **Chronological Elo tracking** - Historical ratings at any date
- ✅ **150 features across 3 pillars** - Fundamentals, Modern Analytics, Hidden Edges
- ✅ **Fast processing** - 14.4 fixtures/second (~21 min for 18K fixtures)

---

## 🚀 Quick Start

### 1. Generate Training Data

```bash
# Basic usage (all data)
python3 scripts/generate_complete_training_data.py

# Custom date range
python3 scripts/generate_complete_training_data.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --output data/csv/training_data_2020_2024.csv
```

### 2. Output

Creates a CSV file with:
- **150 features** per fixture
- **9 metadata columns** (fixture_id, teams, date, scores, result)
- **Point-in-time correct** (no future data used)

---

## 📊 Feature Breakdown

### Pillar 1: Fundamentals (50 features)

**Elo Ratings (10)**
- `home_elo`, `away_elo`, `elo_diff`, `elo_diff_with_ha`
- `home_elo_change_5`, `away_elo_change_5`
- `home_elo_change_10`, `away_elo_change_10`
- `home_elo_vs_league_avg`, `away_elo_vs_league_avg`

**League Position & Points (12)**
- `home_league_position`, `away_league_position`, `position_diff`
- `home_points`, `away_points`, `points_diff`
- `home_points_per_game`, `away_points_per_game`
- `home_in_top_6`, `away_in_top_6`
- `home_in_bottom_3`, `away_in_bottom_3`

**Recent Form (15)**
- `home_points_last_3/5/10`, `away_points_last_3/5/10`
- `home_wins_last_5`, `away_wins_last_5`
- `home_draws_last_5`, `away_draws_last_5`
- `home_goals_scored_last_5`, `away_goals_scored_last_5`
- `home_goals_conceded_last_5`, `away_goals_conceded_last_5`
- `home_goal_diff_last_5`

**Head-to-Head (8)**
- `h2h_home_wins_last_5`, `h2h_draws_last_5`, `h2h_away_wins_last_5`
- `h2h_home_goals_avg`, `h2h_away_goals_avg`
- `h2h_home_win_pct`, `h2h_btts_pct`, `h2h_over_2_5_pct`

**Home Advantage (5)**
- `home_points_at_home`, `away_points_away`
- `home_home_win_pct`, `away_away_win_pct`
- `home_advantage_strength`

### Pillar 2: Modern Analytics (60 features)

**Derived xG (25)**
- `home_derived_xg_per_match_5/10`, `away_derived_xg_per_match_5/10`
- `home_derived_xga_per_match_5/10`, `away_derived_xga_per_match_5/10`
- `home_derived_xgd_5`, `away_derived_xgd_5`, `derived_xgd_matchup`
- `home_goals_vs_xg_5`, `away_goals_vs_xg_5`
- `home_xg_per_shot_5`, `away_xg_per_shot_5`
- `home_big_chances_per_match_5`, `away_big_chances_per_match_5`
- `home_xg_trend_10`, `away_xg_trend_10`, `home_xga_trend_10`, `away_xga_trend_10`

**Shot Analysis (15)**
- `home_shots_per_match_5`, `away_shots_per_match_5`
- `home_shots_on_target_per_match_5`, `away_shots_on_target_per_match_5`
- `home_inside_box_shot_pct_5`, `away_inside_box_shot_pct_5`
- `home_shot_accuracy_5`, `away_shot_accuracy_5`
- `home_shots_per_goal_5`, `away_shots_per_goal_5`
- `home_shots_conceded_per_match_5`, `away_shots_conceded_per_match_5`

**Defensive Intensity (12)**
- `home_ppda_5`, `away_ppda_5` (Passes Per Defensive Action)
- `home_tackles_per_90`, `away_tackles_per_90`
- `home_interceptions_per_90`, `away_interceptions_per_90`
- `home_defensive_actions_per_90`, `away_defensive_actions_per_90`
- `home_possession_pct_5`, `away_possession_pct_5`

**Attack Patterns (8)**
- `home_attacks_per_match_5`, `away_attacks_per_match_5`
- `home_dangerous_attacks_per_match_5`, `away_dangerous_attacks_per_match_5`
- `home_dangerous_attack_ratio_5`, `away_dangerous_attack_ratio_5`
- `home_shots_per_attack_5`, `away_shots_per_attack_5`

### Pillar 3: Hidden Edges (40 features)

**Momentum & Trajectory (12)**
- `home_points_trend_10`, `away_points_trend_10` (linear regression slope)
- `home_weighted_form_5`, `away_weighted_form_5` (exponential weighting)
- `home_win_streak`, `away_win_streak`
- `home_unbeaten_streak`, `away_unbeaten_streak`
- `home_clean_sheet_streak`, `away_clean_sheet_streak`
- `home_goals_trend_10`, `away_goals_trend_10`

**Fixture Difficulty Adjusted (10)**
- `home_avg_opponent_elo_5`, `away_avg_opponent_elo_5`
- `home_points_vs_strong_5`, `away_points_vs_strong_5`
- `home_points_vs_weak_5`, `away_points_vs_weak_5`
- `home_goals_vs_strong_5`, `away_goals_vs_strong_5`
- `home_goals_vs_weak_5`, `away_goals_vs_weak_5`

**Player Quality (10 - simplified)**
- `home_lineup_quality_proxy`, `away_lineup_quality_proxy`
- `home_squad_depth_proxy`, `away_squad_depth_proxy`
- `home_consistency_rating`, `away_consistency_rating`
- `home_recent_performance`, `away_recent_performance`
- `home_goal_threat`, `away_goal_threat`

**Situational Context (8)**
- `home_days_since_last_match`, `away_days_since_last_match`
- `rest_advantage`
- `is_derby_match`
- `home_elo_pressure`, `away_elo_pressure`
- `home_underdog`, `away_underdog`

---

## 🏗️ Architecture

```
src/pipeline/
├── data_loader.py              # Load and organize CSV data
├── standings_calculator.py     # Season-aware standings
├── elo_tracker.py              # Chronological Elo ratings
├── pillar1_fundamentals.py     # 50 fundamental features
├── pillar2_modern_analytics.py # 60 modern analytics features
├── pillar3_hidden_edges.py     # 40 hidden edge features
└── feature_orchestrator.py     # Coordinate all components
```

---

## 📝 Usage Examples

### Generate Full Dataset

```bash
python3 scripts/generate_complete_training_data.py \
    --data-dir data/csv \
    --output data/csv/training_data_full.csv
```

### Generate for Specific Seasons

```bash
# 2022-2024 seasons
python3 scripts/generate_complete_training_data.py \
    --start-date 2022-08-01 \
    --end-date 2024-05-31 \
    --output data/csv/training_data_2022_2024.csv
```

### Custom Minimum Matches

```bash
# Require 10 matches history per team
python3 scripts/generate_complete_training_data.py \
    --min-matches 10 \
    --output data/csv/training_data_min10.csv
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Processing Speed** | 14.4 fixtures/second |
| **Full Dataset Time** | ~21 minutes (18K fixtures) |
| **Memory Usage** | ~1.5 GB peak |
| **Output Size** | ~15 MB (18K rows × 159 columns) |

---

## ✅ Data Quality

### Point-in-Time Correctness

All features use only data from **before** the fixture date:

```python
# For a match on 2024-01-15 20:00:00
as_of_date = "2024-01-15 19:00:00"  # 1 hour before

# Features calculated using only:
# - Fixtures before 2024-01-15 19:00:00
# - Elo ratings as of that date
# - Standings as of that date
```

### Season-Aware Standings

Standings properly separated by **league_id AND season_id**:

```python
standings = standings_calc.calculate_standings_at_date(
    league_id=8,      # Premier League
    season_id=19735,  # 2023-2024 season
    as_of_date="2024-01-15"
)
```

---

## 🔧 Configuration

### Elo Parameters

- **K-factor**: 32 (standard)
- **Home Advantage**: 35 points (calibrated for modern football)
- **Initial Elo**: 1500
- **Season Regression**: 0.5 (regression to mean between seasons)

### Derived xG Coefficients

- **Inside Box**: 0.12
- **Outside Box**: 0.03
- **Big Chance**: 0.35
- **Corner**: 0.03
- **Accuracy Multiplier**: Max 1.3

---

## 📖 Documentation

- **[Implementation Plan](file:///Users/ankurgupta/.gemini/antigravity/brain/b17befe7-0b46-48c7-8e29-6cb1b85b637c/implementation_plan.md)** - Detailed architecture and design
- **[Walkthrough](file:///Users/ankurgupta/.gemini/antigravity/brain/b17befe7-0b46-48c7-8e29-6cb1b85b637c/walkthrough.md)** - Implementation summary and test results
- **[FEATURE_FRAMEWORK.md](file:///Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3/docs/FEATURE_FRAMEWORK.md)** - Complete feature specifications
- **[TRAINING_DATA_STRATEGY.md](file:///Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v3/docs/TRAINING_DATA_STRATEGY.md)** - Point-in-time strategy

---

## 🎯 Next Steps

1. **Generate Full Dataset**
   ```bash
   python3 scripts/generate_complete_training_data.py
   ```

2. **Analyze Features**
   ```python
   import pandas as pd
   df = pd.read_csv('data/csv/training_data_complete.csv')
   df.describe()
   ```

3. **Train Model**
   ```bash
   python scripts/train_model.py --input data/csv/training_data_complete.csv
   ```

---

## 🐛 Troubleshooting

### Missing Statistics

Some matches may have missing statistics (attacks, dangerous attacks):
- **Impact**: ~2,900 missing values
- **Mitigation**: Features default to 0.0

### Memory Issues

If running out of memory:
- Process in smaller date ranges
- Increase system swap space
- Use `--min-matches` to reduce dataset size

---

## 📄 License

Part of Pipeline V3 - Football Prediction System

---

**Ready to generate production-quality training data!** 🚀
