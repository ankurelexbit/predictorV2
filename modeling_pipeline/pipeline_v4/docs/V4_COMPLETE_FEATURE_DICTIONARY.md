# V4 Complete Feature Dictionary

**Last Updated:** 2026-02-03
**Pipeline Version:** V4
**Total Features:** 162

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Summary](#feature-summary)
3. [Pillar 1: Fundamentals (50 Features)](#pillar-1-fundamentals-50-features)
   - [Elo Ratings (10 features)](#elo-ratings-10-features)
   - [League Position & Points (12 features)](#league-position--points-12-features)
   - [Recent Form (15 features)](#recent-form-15-features)
   - [Head-to-Head (8 features)](#head-to-head-8-features)
   - [Home Advantage (5 features)](#home-advantage-5-features)
4. [Pillar 2: Modern Analytics (60 Features)](#pillar-2-modern-analytics-60-features)
   - [Derived xG (25 features)](#derived-xg-25-features)
   - [Shot Analysis (15 features)](#shot-analysis-15-features)
   - [Defensive Intensity (12 features)](#defensive-intensity-12-features)
   - [Attack Patterns (8 features)](#attack-patterns-8-features)
5. [Pillar 3: Hidden Edges (52 Features)](#pillar-3-hidden-edges-52-features)
   - [Momentum & Trajectory (12 features)](#momentum--trajectory-12-features)
   - [Fixture Difficulty Adjusted (10 features)](#fixture-difficulty-adjusted-10-features)
   - [Player Quality (10 features)](#player-quality-10-features)
   - [Situational Context (8 features)](#situational-context-8-features)
   - [Draw Parity Indicators (12 features)](#draw-parity-indicators-12-features)
6. [Calculation Details](#calculation-details)
7. [Data Source Mapping](#data-source-mapping)

---

## Overview

The V4 pipeline generates **162 features** across 3 pillars using a comprehensive feature engineering framework. All features are calculated using **point-in-time data** to prevent data leakage.

### Feature Philosophy

- **Pillar 1 (Fundamentals):** Time-tested metrics that have always worked
- **Pillar 2 (Modern Analytics):** Science-backed metrics from modern football analytics
- **Pillar 3 (Hidden Edges):** Advanced metrics that others might miss

---

## Feature Summary

| Pillar | Category | Count | Importance |
|--------|----------|-------|------------|
| **Pillar 1** | Elo Ratings | 10 | High |
| **Pillar 1** | League Position & Points | 12 | High |
| **Pillar 1** | Recent Form | 15 | High |
| **Pillar 1** | Head-to-Head | 8 | Medium |
| **Pillar 1** | Home Advantage | 5 | Medium |
| **Pillar 2** | Derived xG | 25 | High |
| **Pillar 2** | Shot Analysis | 15 | Medium |
| **Pillar 2** | Defensive Intensity | 12 | Medium |
| **Pillar 2** | Attack Patterns | 8 | Low-Medium |
| **Pillar 3** | Momentum & Trajectory | 12 | Medium |
| **Pillar 3** | Fixture Difficulty Adjusted | 10 | Medium |
| **Pillar 3** | Player Quality | 10 | High |
| **Pillar 3** | Situational Context | 8 | Low-Medium |
| **Pillar 3** | Draw Parity Indicators | 12 | Medium-High |
| **TOTAL** | | **162** | |

---

## Pillar 1: Fundamentals (50 Features)

### Elo Ratings (10 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_elo` | Current Elo rating at match date | Elo Calculator | Float | Team strength rating | High |
| `away_elo` | Current Elo rating at match date | Elo Calculator | Float | Team strength rating | High |
| `elo_diff` | `home_elo - away_elo` | Elo Calculator | Float | Strength difference | High |
| `elo_diff_with_home_advantage` | `home_elo - away_elo + 35` | Elo Calculator | Float | Adjusted strength difference | High |
| `home_elo_change_5` | Elo change over last 5 matches | Elo Calculator | Float | Recent rating momentum | Medium |
| `away_elo_change_5` | Elo change over last 5 matches | Elo Calculator | Float | Recent rating momentum | Medium |
| `home_elo_change_10` | Elo change over last 10 matches | Elo Calculator | Float | Medium-term momentum | Medium |
| `away_elo_change_10` | Elo change over last 10 matches | Elo Calculator | Float | Medium-term momentum | Medium |
| `home_elo_vs_league_avg` | `home_elo - 1500` | Elo Calculator | Float | Relative to league average | Medium |
| `away_elo_vs_league_avg` | `away_elo - 1500` | Elo Calculator | Float | Relative to league average | Medium |

**Elo Formula:**
- **Starting Elo:** 1500
- **K-Factor:** 32
- **Home Advantage:** +35 points
- **Update:** `new_elo = old_elo + K * (actual - expected)`
- **Expected:** `1 / (1 + 10^((opponent_elo - team_elo) / 400))`

---

### League Position & Points (12 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_position` | League position at match date | Standings Calculator | Int | Table position | High |
| `away_position` | League position at match date | Standings Calculator | Int | Table position | High |
| `position_diff` | `home_position - away_position` | Standings Calculator | Int | Position gap | Medium |
| `home_points` | Total points at match date | Standings Calculator | Int | Season points | High |
| `away_points` | Total points at match date | Standings Calculator | Int | Season points | High |
| `points_diff` | `home_points - away_points` | Standings Calculator | Int | Points gap | High |
| `home_ppg` | Points per game (season) | Standings Calculator | Float | Average form | High |
| `away_ppg` | Points per game (season) | Standings Calculator | Float | Average form | High |
| `home_goal_diff` | Goals scored - conceded (season) | Standings Calculator | Int | Net goals | Medium |
| `away_goal_diff` | Goals scored - conceded (season) | Standings Calculator | Int | Net goals | Medium |
| `home_wins` | Total wins (season) | Standings Calculator | Int | Win count | Medium |
| `away_wins` | Total wins (season) | Standings Calculator | Int | Win count | Medium |

---

### Recent Form (15 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_points_last_3` | Sum of points from last 3 matches | Historical Fixtures | Int | Very recent form | High |
| `away_points_last_3` | Sum of points from last 3 matches | Historical Fixtures | Int | Very recent form | High |
| `home_points_last_5` | Sum of points from last 5 matches | Historical Fixtures | Int | Recent form | High |
| `away_points_last_5` | Sum of points from last 5 matches | Historical Fixtures | Int | Recent form | High |
| `home_points_last_10` | Sum of points from last 10 matches | Historical Fixtures | Int | Medium-term form | Medium |
| `away_points_last_10` | Sum of points from last 10 matches | Historical Fixtures | Int | Medium-term form | Medium |
| `home_wins_last_5` | Win count in last 5 matches | Historical Fixtures | Int | Recent wins | Medium |
| `away_wins_last_5` | Win count in last 5 matches | Historical Fixtures | Int | Recent wins | Medium |
| `home_draws_last_5` | Draw count in last 5 matches | Historical Fixtures | Int | Recent draws | Medium |
| `away_draws_last_5` | Draw count in last 5 matches | Historical Fixtures | Int | Recent draws | Medium |
| `home_goals_scored_last_5` | Total goals scored in last 5 | Historical Fixtures | Int | Attacking form | High |
| `away_goals_scored_last_5` | Total goals scored in last 5 | Historical Fixtures | Int | Attacking form | High |
| `home_goals_conceded_last_5` | Total goals conceded in last 5 | Historical Fixtures | Int | Defensive form | High |
| `away_goals_conceded_last_5` | Total goals conceded in last 5 | Historical Fixtures | Int | Defensive form | High |
| `home_goal_diff_last_5` | Goal difference in last 5 | Historical Fixtures | Int | Net recent form | High |

---

### Head-to-Head (8 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `h2h_home_wins_last_5` | Home wins in last 5 H2H | Historical H2H | Int | Recent H2H dominance | Medium |
| `h2h_draws_last_5` | Draws in last 5 H2H | Historical H2H | Int | H2H draw tendency | Medium |
| `h2h_away_wins_last_5` | Away wins in last 5 H2H | Historical H2H | Int | Recent H2H form | Medium |
| `h2h_home_goals_avg` | Average home goals in all H2H | Historical H2H | Float | Historical scoring | Low-Medium |
| `h2h_away_goals_avg` | Average away goals in all H2H | Historical H2H | Float | Historical scoring | Low-Medium |
| `h2h_home_win_pct` | Home win percentage (all-time) | Historical H2H | Float | Overall H2H record | Medium |
| `h2h_btts_pct` | Both teams scored percentage | Historical H2H | Float | Goal tendency | Low |
| `h2h_over_2_5_pct` | Over 2.5 goals percentage | Historical H2H | Float | High-scoring tendency | Low |

---

### Home Advantage (5 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_points_at_home` | Total home points (season) | Season Fixtures | Int | Home form strength | High |
| `away_points_away` | Total away points (season) | Season Fixtures | Int | Away form strength | High |
| `home_home_win_pct` | Win rate at home (season) | Season Fixtures | Float | Home dominance | Medium |
| `away_away_win_pct` | Win rate away (season) | Season Fixtures | Float | Away quality | Medium |
| `home_advantage_strength` | `home_ppg_at_home - 1.5` | Season Fixtures | Float | Home boost vs average | Medium |

---

## Pillar 2: Modern Analytics (60 Features)

### Derived xG (25 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_derived_xg_per_match_5` | Average xG from last 5 matches | Statistics (derived) | Float | Expected goals created | High |
| `away_derived_xg_per_match_5` | Average xG from last 5 matches | Statistics (derived) | Float | Expected goals created | High |
| `home_derived_xga_per_match_5` | Average xG against from last 5 | Statistics (derived) | Float | Expected goals conceded | High |
| `away_derived_xga_per_match_5` | Average xG against from last 5 | Statistics (derived) | Float | Expected goals conceded | High |
| `home_derived_xgd_5` | `xG_per_match - xGA_per_match` | Statistics (derived) | Float | Net expected goal difference | High |
| `away_derived_xgd_5` | `xG_per_match - xGA_per_match` | Statistics (derived) | Float | Net expected goal difference | High |
| `derived_xgd_matchup` | `home_xgd - away_xgd` | Statistics (derived) | Float | xG advantage in matchup | High |
| `home_goals_vs_xg_5` | Actual goals - expected goals | Statistics (derived) | Float | Finishing quality | Medium |
| `away_goals_vs_xg_5` | Actual goals - expected goals | Statistics (derived) | Float | Finishing quality | Medium |
| `home_ga_vs_xga_5` | Goals against - xG against | Statistics (derived) | Float | Defensive over/underperformance | Medium |
| `away_ga_vs_xga_5` | Goals against - xG against | Statistics (derived) | Float | Defensive over/underperformance | Medium |
| `home_xg_per_shot_5` | `total_xG / total_shots` | Statistics (derived) | Float | Shot quality | Medium |
| `away_xg_per_shot_5` | `total_xG / total_shots` | Statistics (derived) | Float | Shot quality | Medium |
| `home_inside_box_xg_ratio` | Inside box xG / total xG | Statistics (derived) | Float | Shot location quality | Medium |
| `away_inside_box_xg_ratio` | Inside box xG / total xG | Statistics (derived) | Float | Shot location quality | Medium |
| `home_big_chances_per_match_5` | Average big chances created | Statistics | Float | High-quality chances | Medium |
| `away_big_chances_per_match_5` | Average big chances created | Statistics | Float | High-quality chances | Medium |
| `home_big_chance_conversion_5` | Big chances scored / created | Statistics | Float | Clinical finishing | Medium |
| `away_big_chance_conversion_5` | Big chances scored / created | Statistics | Float | Clinical finishing | Medium |
| `home_xg_from_corners_5` | xG from corner kicks | Statistics (derived) | Float | Set piece threat | Low-Medium |
| `away_xg_from_corners_5` | xG from corner kicks | Statistics (derived) | Float | Set piece threat | Low-Medium |
| `home_xg_trend_10` | Linear trend of xG over 10 matches | Statistics (derived) | Float | xG momentum direction | Medium |
| `away_xg_trend_10` | Linear trend of xG over 10 matches | Statistics (derived) | Float | xG momentum direction | Medium |
| `home_xga_trend_10` | Linear trend of xGA over 10 matches | Statistics (derived) | Float | Defensive trend | Medium |
| `away_xga_trend_10` | Linear trend of xGA over 10 matches | Statistics (derived) | Float | Defensive trend | Medium |

**Derived xG Formula:**
```
xG = (shots_on_target * 0.35) + (shots_inside_box * 0.15)
```

This is a simplified formula when direct xG data is unavailable. Real xG models use shot location, angle, distance, and game state.

---

### Shot Analysis (15 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_shots_per_match_5` | Average total shots (last 5) | Statistics CSV | Float | Shooting volume | Medium |
| `away_shots_per_match_5` | Average total shots (last 5) | Statistics CSV | Float | Shooting volume | Medium |
| `home_shots_on_target_per_match_5` | Average shots on target (last 5) | Statistics CSV | Float | Shot accuracy | Medium |
| `away_shots_on_target_per_match_5` | Average shots on target (last 5) | Statistics CSV | Float | Shot accuracy | Medium |
| `home_inside_box_shot_pct_5` | Inside box / total shots | Statistics CSV | Float | Shot location quality | Medium |
| `away_inside_box_shot_pct_5` | Inside box / total shots | Statistics CSV | Float | Shot location quality | Medium |
| `home_outside_box_shot_pct_5` | Outside box / total shots | Statistics CSV | Float | Long-range tendency | Low |
| `away_outside_box_shot_pct_5` | Outside box / total shots | Statistics CSV | Float | Long-range tendency | Low |
| `home_shot_accuracy_5` | Shots on target / total shots | Statistics CSV | Float | Shooting precision | Medium |
| `away_shot_accuracy_5` | Shots on target / total shots | Statistics CSV | Float | Shooting precision | Medium |
| `home_shots_per_goal_5` | Total shots / goals scored | Statistics CSV | Float | Conversion efficiency | Medium |
| `away_shots_per_goal_5` | Total shots / goals scored | Statistics CSV | Float | Conversion efficiency | Medium |
| `home_shots_conceded_per_match_5` | Average opponent shots (last 5) | Statistics CSV | Float | Defensive pressure allowed | Medium |
| `away_shots_conceded_per_match_5` | Average opponent shots (last 5) | Statistics CSV | Float | Defensive pressure allowed | Medium |
| `home_shots_on_target_conceded_5` | Average opponent SOT (last 5) | Statistics CSV | Float | Defensive vulnerability | Medium |

---

### Defensive Intensity (12 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_ppda_5` | Passes allowed per defensive action | Statistics (derived) | Float | Pressing intensity (lower = more) | Medium |
| `away_ppda_5` | Passes allowed per defensive action | Statistics (derived) | Float | Pressing intensity | Medium |
| `home_tackles_per_90` | Average tackles per match | Statistics CSV | Float | Defensive activity | Medium |
| `away_tackles_per_90` | Average tackles per match | Statistics CSV | Float | Defensive activity | Medium |
| `home_interceptions_per_90` | Average interceptions per match | Statistics CSV | Float | Reading of play | Medium |
| `away_interceptions_per_90` | Average interceptions per match | Statistics CSV | Float | Reading of play | Medium |
| `home_tackle_success_rate_5` | Successful tackles / attempted | Statistics | Float | Tackling quality | Low-Medium |
| `away_tackle_success_rate_5` | Successful tackles / attempted | Statistics | Float | Tackling quality | Low-Medium |
| `home_defensive_actions_per_90` | Tackles + interceptions | Statistics CSV | Float | Total defensive work | Medium |
| `away_defensive_actions_per_90` | Tackles + interceptions | Statistics CSV | Float | Total defensive work | Medium |
| `home_possession_pct_5` | Average possession percentage | Statistics CSV | Float | Ball control | Medium |
| `away_possession_pct_5` | Average possession percentage | Statistics CSV | Float | Ball control | Medium |

---

### Attack Patterns (8 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_attacks_per_match_5` | Average total attacks | Statistics CSV | Float | Attacking frequency | Low-Medium |
| `away_attacks_per_match_5` | Average total attacks | Statistics CSV | Float | Attacking frequency | Low-Medium |
| `home_dangerous_attacks_per_match_5` | Average dangerous attacks | Statistics CSV | Float | High-threat attacks | Medium |
| `away_dangerous_attacks_per_match_5` | Average dangerous attacks | Statistics CSV | Float | High-threat attacks | Medium |
| `home_dangerous_attack_ratio_5` | Dangerous / total attacks | Statistics CSV | Float | Attack quality | Low-Medium |
| `away_dangerous_attack_ratio_5` | Dangerous / total attacks | Statistics CSV | Float | Attack quality | Low-Medium |
| `home_shots_per_attack_5` | Total shots / total attacks | Statistics CSV | Float | Attacking efficiency | Low-Medium |
| `away_shots_per_attack_5` | Total shots / total attacks | Statistics CSV | Float | Attacking efficiency | Low-Medium |

---

## Pillar 3: Hidden Edges (52 Features)

### Momentum & Trajectory (12 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_points_trend_10` | Linear slope of points over 10 matches | Historical Fixtures | Float | Form direction (+ = improving) | Medium |
| `away_points_trend_10` | Linear slope of points over 10 matches | Historical Fixtures | Float | Form direction | Medium |
| `home_xg_trend_10` | Linear slope of xG over 10 matches | Statistics (derived) | Float | xG trajectory | Medium |
| `away_xg_trend_10` | Linear slope of xG over 10 matches | Statistics (derived) | Float | xG trajectory | Medium |
| `home_weighted_form_5` | Weighted average (recent = higher) | Historical Fixtures | Float | Recent-weighted form | Medium |
| `away_weighted_form_5` | Weighted average (recent = higher) | Historical Fixtures | Float | Recent-weighted form | Medium |
| `home_win_streak` | Current consecutive wins | Historical Fixtures | Int | Winning momentum | Medium |
| `away_win_streak` | Current consecutive wins | Historical Fixtures | Int | Winning momentum | Medium |
| `home_unbeaten_streak` | Current consecutive W/D | Historical Fixtures | Int | Form stability | Medium |
| `away_unbeaten_streak` | Current consecutive W/D | Historical Fixtures | Int | Form stability | Medium |
| `home_clean_sheet_streak` | Current consecutive clean sheets | Historical Fixtures | Int | Defensive momentum | Low-Medium |
| `away_clean_sheet_streak` | Current consecutive clean sheets | Historical Fixtures | Int | Defensive momentum | Low-Medium |

**Weighted Form Formula:**
```
weighted_form_5 = (match1 * 0.30) + (match2 * 0.25) + (match3 * 0.20) + (match4 * 0.15) + (match5 * 0.10)
```

**Trend Calculation:**
```
trend = slope of linear regression (numpy.polyfit(x, y, 1)[0])
```

---

### Fixture Difficulty Adjusted (10 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_avg_opponent_elo_5` | Average Elo of last 5 opponents | Elo Calculator | Float | Schedule difficulty | Medium |
| `away_avg_opponent_elo_5` | Average Elo of last 5 opponents | Elo Calculator | Float | Schedule difficulty | Medium |
| `home_points_vs_top_6` | Points earned vs top 6 teams | Historical Fixtures | Int | Big match performance | Medium |
| `away_points_vs_top_6` | Points earned vs top 6 teams | Historical Fixtures | Int | Big match performance | Medium |
| `home_points_vs_bottom_6` | Points earned vs bottom 6 | Historical Fixtures | Int | Performance vs weak teams | Low-Medium |
| `away_points_vs_bottom_6` | Points earned vs bottom 6 | Historical Fixtures | Int | Performance vs weak teams | Low-Medium |
| `home_xg_vs_top_half` | xG vs top half opponents | Statistics (derived) | Float | Quality vs strong teams | Medium |
| `away_xg_vs_top_half` | xG vs top half opponents | Statistics (derived) | Float | Quality vs strong teams | Medium |
| `home_xga_vs_bottom_half` | xGA vs bottom half opponents | Statistics (derived) | Float | Defense vs weaker teams | Low-Medium |
| `away_xga_vs_bottom_half` | xGA vs bottom half opponents | Statistics (derived) | Float | Defense vs weaker teams | Low-Medium |

---

### Player Quality (10 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_avg_player_value` | Average market value of lineup | Lineup data | Float | Squad financial value | High |
| `away_avg_player_value` | Average market value of lineup | Lineup data | Float | Squad financial value | High |
| `home_total_minutes` | Total career minutes of lineup | Lineup data | Int | Experience level | Medium |
| `away_total_minutes` | Total career minutes of lineup | Lineup data | Int | Experience level | Medium |
| `home_goals_season` | Total goals scored by lineup | Lineup data | Int | Attacking talent | Medium |
| `away_goals_season` | Total goals scored by lineup | Lineup data | Int | Attacking talent | Medium |
| `home_assists_season` | Total assists by lineup | Lineup data | Int | Creative talent | Medium |
| `away_assists_season` | Total assists by lineup | Lineup data | Int | Creative talent | Medium |
| `home_missing_starters` | Count of absent key players | Lineup/Sidelined data | Int | Squad availability | High |
| `away_missing_starters` | Count of absent key players | Lineup/Sidelined data | Int | Squad availability | High |

**Note:** Player features use lineup data when available, otherwise fall back to team-level estimates.

---

### Situational Context (8 features)

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `home_points_from_relegation` | Points above relegation zone | Standings | Int | Safety cushion (- = in danger) | Medium |
| `away_points_from_relegation` | Points above relegation zone | Standings | Int | Safety cushion | Medium |
| `home_points_from_top` | Points behind 1st place | Standings | Int | Title race position | Low-Medium |
| `away_points_from_top` | Points behind 1st place | Standings | Int | Title race position | Low-Medium |
| `home_days_since_last_match` | Days of rest before match | Fixture dates | Int | Recovery time | Low-Medium |
| `away_days_since_last_match` | Days of rest before match | Fixture dates | Int | Recovery time | Low-Medium |
| `rest_advantage` | `home_rest - away_rest` | Fixture dates | Int | Fatigue differential | Low-Medium |
| `is_derby_match` | Geographic rivalry indicator | Team metadata | Binary | Local derby | Low |

---

### Draw Parity Indicators (12 features)

**NEW in V4:** Features specifically designed to identify evenly-matched teams (draw likelihood).

| Feature Name | Calculation | Data Source | Type | Meaning | Importance |
|--------------|-------------|-------------|------|---------|------------|
| `elo_difference` | `abs(home_elo - away_elo)` | Elo Calculator | Float | Strength parity (lower = more even) | High |
| `form_difference_10` | `abs(home_points_10 - away_points_10)` | Historical Fixtures | Float | Recent form parity | Medium-High |
| `position_difference` | `abs(home_pos - away_pos)` | Standings | Int | Table position gap | Medium |
| `h2h_draw_rate` | Historical draws / total H2H | Historical H2H | Float | H2H draw tendency | Medium |
| `home_draw_rate_10` | Draws / 10 matches | Historical Fixtures | Float | Team draw tendency | Medium-High |
| `away_draw_rate_10` | Draws / 10 matches | Historical Fixtures | Float | Team draw tendency | Medium-High |
| `combined_draw_tendency` | `(home_rate + away_rate) / 2` | Historical Fixtures | Float | Combined draw likelihood | High |
| `league_draw_rate` | League draws / total matches | Season Fixtures | Float | League draw baseline | Medium |
| `both_midtable` | Both in positions 7-14 | Standings | Binary | Midtable clash (1=yes) | Medium |
| `both_low_scoring` | Both avg < 1.2 goals/game | Historical Fixtures | Binary | Low-scoring teams | Medium-High |
| `both_defensive` | Both concede < 1.0/game | Historical Fixtures | Binary | Defensive teams | Medium |
| `either_coming_from_draw` | Last result was draw | Historical Fixtures | Binary | Draw momentum | Low-Medium |

**Draw Detection Logic:**
- Low `elo_difference` + high `combined_draw_tendency` = high draw probability
- `both_midtable` + `both_defensive` = draw indicator
- High `league_draw_rate` + low scoring = contextual draw signal

---

## Calculation Details

### Elo Rating System

**Parameters:**
- Starting Elo: 1500
- K-factor: 32
- Home advantage: +35 points

**Update Formula:**
```python
expected_score = 1 / (1 + 10**((opponent_elo - team_elo) / 400))
new_elo = old_elo + K * (actual_score - expected_score)

# actual_score: 1.0 (win), 0.5 (draw), 0.0 (loss)
```

### Derived xG Formula

When direct xG is unavailable:
```python
xG = (shots_on_target * 0.35) + (shots_inside_box * 0.15)
```

Coefficients are empirically derived from shot conversion data.

### Trend Calculation

Linear regression slope using numpy:
```python
x = np.arange(len(values))  # Match index
slope, intercept = np.polyfit(x, values, 1)
trend = slope
```
- Positive slope = improving
- Negative slope = declining

### Weighted Form

Recent matches weighted more heavily:
```python
weights = [0.30, 0.25, 0.20, 0.15, 0.10]  # Most recent to oldest
weighted_form = sum(points[i] * weights[i] for i in range(5))
```

---

## Data Source Mapping

### SportMonks API Endpoints

| Data Type | API Endpoint | Features Generated |
|-----------|--------------|-------------------|
| Fixtures | `/fixtures` | All form, H2H, goal features |
| Statistics | `/fixtures/{id}/statistics` | xG, shots, tackles, possession |
| Lineups | `/fixtures/{id}/lineups` | Player quality features |
| Sidelined | `/teams/{id}/sidelined` | Missing starters |
| Standings | `/standings/season/{id}` | Position, points features (metadata only) |

### Calculated Internally

| Feature Category | Source | Notes |
|------------------|--------|-------|
| Elo Ratings | `EloCalculator` | Calculated from match history |
| Standings | `StandingsCalculator` | Point-in-time reconstruction |
| Derived xG | Statistics + Formula | When xG unavailable |
| Trends | Historical data + Linear regression | 10-match window |

### CSV Pre-processing

For performance, statistics are embedded in CSV:
```
data/processed/fixtures_with_stats.csv
```

Columns include:
- `home_shots_total`, `away_shots_total`
- `home_shots_on_target`, `away_shots_on_target`
- `home_shots_inside_box`, `away_shots_inside_box`
- `home_tackles`, `away_tackles`
- `home_interceptions`, `away_interceptions`
- `home_ball_possession`, `away_ball_possession`
- `home_attacks`, `away_attacks`
- `home_dangerous_attacks`, `away_dangerous_attacks`

---

## Feature Importance (General Rankings)

### Tier 1 (Critical - Top 20%)
1. `elo_diff_with_home_advantage`
2. `home_elo`, `away_elo`
3. `home_derived_xg_per_match_5`, `away_derived_xg_per_match_5`
4. `home_points_last_5`, `away_points_last_5`
5. `home_goals_scored_last_5`, `away_goals_scored_last_5`
6. `home_position`, `away_position`
7. `combined_draw_tendency`
8. `home_avg_player_value`, `away_avg_player_value`

### Tier 2 (Important - Top 50%)
- Elo changes, form trends
- xG differential features
- Position and points differences
- Shot analysis features
- Draw parity indicators
- Fixture-adjusted metrics

### Tier 3 (Supporting - Top 80%)
- H2H statistics
- Attack patterns
- Defensive intensity
- Momentum indicators
- Context features

### Tier 4 (Contextual - Bottom 20%)
- Derby indicators
- Set piece features
- Long-term historical stats
- Placeholder features

---

## Usage Notes

### Point-in-Time Correctness

All features use **only data available before the match date**:
- Elo calculated up to (not including) match date
- Form calculated from previous matches only
- Standings exclude current match result
- Statistics from completed matches only

### Missing Data Handling

| Scenario | Handling |
|----------|----------|
| No fixtures available | Default to baseline (1500 Elo, 0 points, etc.) |
| Statistics unavailable | Use derived formulas or placeholders |
| Lineup data missing | Fall back to team-level estimates |
| Insufficient history | Use available data with reduced window |

### Feature Scaling

Features have different scales:
- Binary: 0 or 1
- Small integers: 0-20 (positions, streaks)
- Medium integers: 0-100 (points, goals)
- Large integers: 1000-2000 (Elo)
- Floats: 0.0-3.0 (xG, rates)

**Recommendation:** Use XGBoost (scale-invariant) or normalize for other models.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| V4.0 | 2026-02-03 | Added 12 draw parity features, increased total to 162 |
| V3.0 | 2026-01-30 | Added player quality features (10), total 150 |
| V2.0 | 2026-01-25 | Added modern analytics pillar (60 features) |
| V1.0 | 2026-01-20 | Initial fundamentals pillar (50 features) |

---

## Related Documentation

- `FEATURE_FRAMEWORK.md` - Feature philosophy and design principles
- `FEATURE_GENERATION_GUIDE.md` - Technical implementation guide
- `PLAYER_FEATURES_IMPLEMENTATION.md` - Player quality feature details
- `TRAINING_DATA_STRATEGY.md` - Point-in-time correctness methodology

---

**Maintained by:** V4 Pipeline Development Team
**Contact:** See README.md for support information
