# Feature Dictionary - Pipeline V3
## Complete Reference for All 150-180 Features

**Last Updated:** January 25, 2026  
**Total Features:** 150-180

---

## 📖 How to Use This Dictionary

Each feature includes:
- **Name:** Feature identifier
- **Source:** Where data comes from (API endpoint/calculation)
- **Calculation:** How it's computed
- **Type:** Numeric/Boolean/Categorical
- **Meaning:** What it represents
- **Importance:** Expected predictive value (⭐-⭐⭐⭐⭐⭐)

---

## PILLAR 1: FUNDAMENTALS (50 features)

### 1.1 Elo Ratings (10 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_elo` | Calculated | Historical Elo tracking | Float | Current team strength rating | ⭐⭐⭐⭐⭐ |
| `away_elo` | Calculated | Historical Elo tracking | Float | Current team strength rating | ⭐⭐⭐⭐⭐ |
| `elo_diff` | Calculated | `home_elo - away_elo` | Float | Strength differential | ⭐⭐⭐⭐⭐ |
| `elo_diff_with_ha` | Calculated | `elo_diff + 50` | Float | Strength diff + home advantage | ⭐⭐⭐⭐⭐ |
| `home_elo_change_5` | Calculated | Elo change last 5 matches | Float | Recent Elo momentum | ⭐⭐⭐⭐ |
| `away_elo_change_5` | Calculated | Elo change last 5 matches | Float | Recent Elo momentum | ⭐⭐⭐⭐ |
| `home_elo_change_10` | Calculated | Elo change last 10 matches | Float | Extended Elo trend | ⭐⭐⭐ |
| `away_elo_change_10` | Calculated | Elo change last 10 matches | Float | Extended Elo trend | ⭐⭐⭐ |
| `home_elo_vs_league_avg` | Calculated | `home_elo - league_avg_elo` | Float | Relative strength in league | ⭐⭐⭐ |
| `away_elo_vs_league_avg` | Calculated | `away_elo - league_avg_elo` | Float | Relative strength in league | ⭐⭐⭐ |

**Elo Calculation:**
```python
expected = 1 / (1 + 10**((opp_elo - team_elo - 50) / 400))
new_elo = old_elo + 32 * (result - expected)
```

### 1.2 League Position & Points (12 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_league_position` | API: `/standings` | Current position | Int | League standing (1-20) | ⭐⭐⭐⭐⭐ |
| `away_league_position` | API: `/standings` | Current position | Int | League standing (1-20) | ⭐⭐⭐⭐⭐ |
| `position_diff` | Calculated | `home_pos - away_pos` | Int | Position differential | ⭐⭐⭐⭐ |
| `home_points` | API: `/standings` | Total season points | Int | Season points accumulated | ⭐⭐⭐⭐⭐ |
| `away_points` | API: `/standings` | Total season points | Int | Season points accumulated | ⭐⭐⭐⭐⭐ |
| `points_diff` | Calculated | `home_pts - away_pts` | Int | Points differential | ⭐⭐⭐⭐ |
| `home_points_per_game` | Calculated | `points / matches_played` | Float | Average PPG this season | ⭐⭐⭐⭐⭐ |
| `away_points_per_game` | Calculated | `points / matches_played` | Float | Average PPG this season | ⭐⭐⭐⭐⭐ |
| `home_in_top_6` | Calculated | `position <= 6` | Bool | Title contender | ⭐⭐⭐ |
| `away_in_top_6` | Calculated | `position <= 6` | Bool | Title contender | ⭐⭐⭐ |
| `home_in_bottom_3` | Calculated | `position >= 18` | Bool | Relegation zone | ⭐⭐⭐ |
| `away_in_bottom_3` | Calculated | `position >= 18` | Bool | Relegation zone | ⭐⭐⭐ |

### 1.3 Recent Form (15 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_points_last_3` | API: `/fixtures` | Sum points last 3 | Int | Very recent form (0-9) | ⭐⭐⭐⭐⭐ |
| `away_points_last_3` | API: `/fixtures` | Sum points last 3 | Int | Very recent form (0-9) | ⭐⭐⭐⭐⭐ |
| `home_points_last_5` | API: `/fixtures` | Sum points last 5 | Int | Recent form (0-15) | ⭐⭐⭐⭐⭐ |
| `away_points_last_5` | API: `/fixtures` | Sum points last 5 | Int | Recent form (0-15) | ⭐⭐⭐⭐⭐ |
| `home_points_last_10` | API: `/fixtures` | Sum points last 10 | Int | Extended form (0-30) | ⭐⭐⭐⭐ |
| `away_points_last_10` | API: `/fixtures` | Sum points last 10 | Int | Extended form (0-30) | ⭐⭐⭐⭐ |
| `home_wins_last_5` | API: `/fixtures` | Count wins | Int | Winning form (0-5) | ⭐⭐⭐⭐ |
| `away_wins_last_5` | API: `/fixtures` | Count wins | Int | Winning form (0-5) | ⭐⭐⭐⭐ |
| `home_draws_last_5` | API: `/fixtures` | Count draws | Int | Draw tendency (0-5) | ⭐⭐⭐⭐ |
| `away_draws_last_5` | API: `/fixtures` | Count draws | Int | Draw tendency (0-5) | ⭐⭐⭐⭐ |
| `home_goals_scored_last_5` | API: `/fixtures` | Sum goals scored | Int | Attacking form | ⭐⭐⭐⭐⭐ |
| `away_goals_scored_last_5` | API: `/fixtures` | Sum goals scored | Int | Attacking form | ⭐⭐⭐⭐⭐ |
| `home_goals_conceded_last_5` | API: `/fixtures` | Sum goals conceded | Int | Defensive form | ⭐⭐⭐⭐⭐ |
| `away_goals_conceded_last_5` | API: `/fixtures` | Sum goals conceded | Int | Defensive form | ⭐⭐⭐⭐⭐ |
| `home_goal_diff_last_5` | Calculated | `GF - GA last 5` | Int | Goal differential | ⭐⭐⭐⭐⭐ |

---

## PILLAR 2: MODERN ANALYTICS (60 features)

### 2.1 Derived xG (25 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_derived_xg_per_match_5` | Calculated | Avg derived xG last 5 | Float | Attack quality | ⭐⭐⭐⭐⭐ |
| `away_derived_xg_per_match_5` | Calculated | Avg derived xG last 5 | Float | Attack quality | ⭐⭐⭐⭐⭐ |
| `home_derived_xga_per_match_5` | Calculated | Avg derived xGA last 5 | Float | Defensive quality | ⭐⭐⭐⭐⭐ |
| `away_derived_xga_per_match_5` | Calculated | Avg derived xGA last 5 | Float | Defensive quality | ⭐⭐⭐⭐⭐ |
| `home_derived_xgd_5` | Calculated | `xG - xGA` | Float | xG differential | ⭐⭐⭐⭐⭐ |
| `away_derived_xgd_5` | Calculated | `xG - xGA` | Float | xG differential | ⭐⭐⭐⭐⭐ |
| `derived_xgd_matchup` | Calculated | `home_xgd - away_xgd` | Float | Matchup quality diff | ⭐⭐⭐⭐⭐ |
| `home_goals_vs_xg_5` | Calculated | `goals - xG` | Float | Luck/finishing quality | ⭐⭐⭐⭐ |
| `away_goals_vs_xg_5` | Calculated | `goals - xG` | Float | Luck/finishing quality | ⭐⭐⭐⭐ |

**Derived xG Formula:**
```python
xG = (shots_inside_box × 0.12 + shots_outside_box × 0.03 + 
      big_chances × 0.35 + corners × 0.03) × accuracy_multiplier
```

**Data Sources:**
- `shots_insidebox`: API `/statistics` type_id=49
- `shots_outsidebox`: API `/statistics` type_id=50
- `big_chances_created`: API `/statistics` type_id=580
- `corners`: API `/statistics` type_id=34

### 2.2 Shot Analysis (15 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_shots_per_match_5` | API: `/statistics` (42) | Avg shots last 5 | Float | Shot volume | ⭐⭐⭐⭐ |
| `home_shots_on_target_per_match_5` | API: `/statistics` | Avg SoT last 5 | Float | Shot accuracy | ⭐⭐⭐⭐ |
| `home_inside_box_shot_pct_5` | Calculated | `inside / total` | Float | Shot location quality | ⭐⭐⭐⭐ |
| `home_shot_accuracy_5` | Calculated | `SoT / total` | Float | Accuracy rate | ⭐⭐⭐⭐ |
| `home_shots_per_goal_5` | Calculated | `shots / goals` | Float | Conversion efficiency | ⭐⭐⭐⭐ |

---

## PILLAR 3: HIDDEN EDGES (40 features)

### 3.1 Momentum & Trajectory (12 features)

| Feature | Source | Calculation | Type | Meaning | Importance |
|---------|--------|-------------|------|---------|------------|
| `home_points_trend_10` | Calculated | Linear regression slope | Float | Form improving/declining | ⭐⭐⭐⭐⭐ |
| `home_weighted_form_5` | Calculated | Exponential weighted avg | Float | Recent-weighted form | ⭐⭐⭐⭐ |
| `home_win_streak` | Calculated | Consecutive wins | Int | Current winning streak | ⭐⭐⭐⭐ |
| `home_unbeaten_streak` | Calculated | Matches without loss | Int | Unbeaten run | ⭐⭐⭐ |

---

## 📊 Feature Importance Rankings

### Top 20 Expected Features:

1. **elo_diff** (8-12%) - Team strength gap
2. **home_derived_xgd_5** (6-9%) - xG differential  
3. **home_points_5** (5-8%) - Recent form
4. **away_points_5** (5-8%) - Away form
5. **home_derived_xg_per_match_5** (4-6%) - Attack quality
6. **home_points_trend_10** (3-5%) - Momentum
7. **home_league_position** (3-4%) - Season standing
8. **h2h_home_wins_last_5** (3-5%) - H2H history
9. **home_ppda_5** (2-4%) - Pressing intensity
10. **home_big_chances_per_match_5** (2-4%) - Chance creation

---

## 🔧 Implementation Guide

### Feature Calculation Order:

1. **Base Statistics** (from API)
2. **Elo Ratings** (historical calculation)
3. **Derived xG** (from base stats)
4. **Form Features** (rolling windows)
5. **Momentum** (trends, streaks)
6. **Fixture-Adjusted** (opponent strength)
7. **Final Features** (interactions, ratios)

---

**See full specifications in FEATURE_FRAMEWORK.md**
