# Complete Feature List - Football Match Prediction

**Total Features:** 465  
**Dataset:** sportmonks_features.csv  
**Last Updated:** 2026-01-18

---

## Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| [Metadata & Identifiers](#1-metadata--identifiers) | 7 | Match identifiers and basic info |
| [Target Variables](#2-target-variables) | 4 | Prediction targets |
| [Elo Ratings](#3-elo-ratings) | 3 | Team strength ratings |
| [Form Features](#4-form-features-3510-games) | 18 | Recent performance metrics |
| [Rolling Statistics - 3 Games](#5-rolling-statistics---3-games) | 130 | Team & player stats over 3 games |
| [Rolling Statistics - 5 Games](#6-rolling-statistics---5-games) | 130 | Team & player stats over 5 games |
| [Rolling Statistics - 10 Games](#7-rolling-statistics---10-games) | 130 | Team & player stats over 10 games |
| [Attack/Defense Strength](#8-attackdefense-strength) | 8 | Derived strength metrics |
| [Standings Features](#9-standings-features) | 6 | League position and points |
| [Head-to-Head](#10-head-to-head-h2h) | 5 | Historical matchup stats |
| [Injuries](#11-injuries) | 3 | Sidelined player counts |
| [Market/Odds](#12-marketodds) | 8 | Betting market data |
| [Contextual](#13-contextual) | 5 | Match context features |

---

## 1. Metadata & Identifiers

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `fixture_id` | int | Unique match identifier |
| 2 | `date` | datetime | Match date and time |
| 3 | `season_id` | int | Season identifier |
| 4 | `home_team_id` | int | Home team ID |
| 5 | `away_team_id` | int | Away team ID |
| 6 | `home_team_name` | str | Home team name |
| 7 | `away_team_name` | str | Away team name |

---

## 2. Target Variables

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 8 | `target` | int | Match outcome (0=away, 1=draw, 2=home) |
| 9 | `home_win` | bool | Whether home team won |
| 10 | `draw` | bool | Whether match was a draw |
| 11 | `away_win` | bool | Whether away team won |

**Ground Truth:**
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 12 | `home_goals` | int | Actual goals scored by home team |
| 13 | `away_goals` | int | Actual goals scored by away team |

---

## 3. Elo Ratings

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 14 | `home_elo` | float | Home team Elo rating |
| 15 | `away_elo` | float | Away team Elo rating |
| 16 | `elo_diff` | float | home_elo - away_elo |

**K-factor:** 32 (standard chess Elo)

---

## 4. Form Features (3/5/10 games)

### Form - 3 Games
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 17-22 | `{home/away}_wins_3` | int | Wins in last 3 games |
| | `{home/away}_draws_3` | int | Draws in last 3 games |
| | `{home/away}_losses_3` | int | Losses in last 3 games |
| 23-25 | `{home/away}_form_3` | float | Points avg (3pts/win, 1pt/draw) |
| | `form_diff_3` | float | home_form_3 - away_form_3 |

### Form - 5 Games
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 26-31 | `{home/away}_wins_5` | int | Wins in last 5 games |
| | `{home/away}_draws_5` | int | Draws in last 5 games |
| | `{home/away}_losses_5` | int | Losses in last 5 games |
| 32-34 | `{home/away}_form_5` | float | Points avg (3pts/win, 1pt/draw) |
| | `form_diff_5` | float | home_form_5 - away_form_5 |

---

## 5. Rolling Statistics - 3 Games

### Goals & xG (3-game)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 35-38 | `{home/away}_goals_3` | float | Goals scored (avg) |
| | `{home/away}_goals_conceded_3` | float | Goals conceded (avg) |
| 39-42 | `{home/away}_xg_3` | float | Expected goals (avg) |
| | `{home/away}_xg_conceded_3` | float | xG conceded (avg) |

### Shooting (3-game)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 43-46 | `{home/away}_shots_total_3` | float | Total shots (avg) |
| | `{home/away}_shots_total_conceded_3` | float | Shots conceded (avg) |
| 47-50 | `{home/away}_shots_on_target_3` | float | Shots on target (avg) |
| | `{home/away}_shots_on_target_conceded_3` | float | SoT conceded (avg) |

### Possession & Passing (3-game)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 51-54 | `{home/away}_possession_pct_3` | float | Ball possession % (avg) |
| | `{home/away}_possession_pct_conceded_3` | float | Opp. possession % (avg) |
| 63-66 | `{home/away}_passes_3` | float | Total passes (avg) |
| | `{home/away}_passes_conceded_3` | float | Opp. passes (avg) |
| 67-70 | `{home/away}_successful_passes_pct_3` | float | Pass accuracy % (avg) |
| | `{home/away}_successful_passes_pct_conceded_3` | float | Opp. pass accuracy (avg) |

### Attacking (3-game)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 55-58 | `{home/away}_dangerous_attacks_3` | float | Dangerous attacks (avg) |
| | `{home/away}_dangerous_attacks_conceded_3` | float | Dangerous attacks conceded |
| 71-74 | `{home/away}_big_chances_created_3` | float | Big chances (avg) |
| | `{home/away}_big_chances_created_conceded_3` | float | Big chances conceded |

### Set Pieces (3-game)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 59-62 | `{home/away}_corners_3` | float | Corner kicks (avg) |
| | `{home/away}_corners_conceded_3` | float | Corners conceded (avg) |

### Defensive (3-game)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 75-78 | `{home/away}_tackles_3` | float | Tackles (avg) |
| | `{home/away}_tackles_conceded_3` | float | Tackles conceded (avg) |
| 79-82 | `{home/away}_interceptions_3` | float | Interceptions (avg) |
| | `{home/away}_interceptions_conceded_3` | float | Interceptions conceded |

### Player-Level Stats (3-game)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 83-86 | `{home/away}_player_clearances_3` | float | Player clearances (avg) |
| 87-90 | `{home/away}_player_aerials_won_3` | float | Aerial duels won (avg) |
| 91-94 | `{home/away}_player_touches_3` | float | Player touches (avg) |
| 95-98 | `{home/away}_player_rating_3` | float | Average player rating |
| 99-102 | `{home/away}_player_total_duels_3` | float | Total duels (avg) |
| 103-106 | `{home/away}_player_duels_won_3` | float | Duels won (avg) |
| 107-110 | `{home/away}_player_possession_lost_3` | float | Possession lost (avg) |
| 111-114 | `{home/away}_player_accurate_passes_3` | float | Accurate passes (avg) |
| 115-118 | `{home/away}_player_dribble_attempts_3` | float | Dribble attempts (avg) |
| 119-122 | `{home/away}_player_successful_dribbles_3` | float | Successful dribbles |
| 123-126 | `{home/away}_player_tackles_won_3` | float | Player tackles won |
| 127-130 | `{home/away}_player_long_balls_won_3` | float | Long balls won |
| 131-134 | `{home/away}_player_dispossessed_3` | float | Dispossessed (avg) |
| 135-138 | `{home/away}_player_fouls_drawn_3` | float | Fouls drawn (avg) |
| 139-142 | `{home/away}_player_blocked_shots_3` | float | Shots blocked (avg) |
| 143-146 | `{home/away}_player_duels_lost_3` | float | Duels lost (avg) |
| 147-150 | `{home/away}_player_aerials_lost_3` | float | Aerial duels lost |
| 151-154 | `{home/away}_player_aerials_total_3` | float | Total aerial duels |
| 155-158 | `{home/away}_player_saves_3` | float | GK saves (avg) |
| 159-162 | `{home/away}_player_goals_conceded_3` | float | GK goals conceded |
| 163-166 | `{home/away}_player_saves_insidebox_3` | float | GK saves inside box |

**Note:** Each metric has both "for" and "conceded" versions  
**Total 3-game rolling features:** 130

---

## 6. Rolling Statistics - 5 Games

Same structure as 3-game rolling stats, covering last 5 matches:

| Feature Range | Description |
|---------------|-------------|
| 167-170 | Goals & xG (5-game avg) |
| 171-206 | Shooting, possession, passing (5-game) |
| 207-214 | Attacking & defensive (5-game) |
| 215-298 | Player-level stats (5-game) |

**Total 5-game rolling features:** 130

---

## 7. Rolling Statistics - 10 Games

Same structure as 3/5-game rolling stats, covering last 10 matches:

| Feature Range | Description |
|---------------|-------------|
| 299-302 | Goals & xG (10-game avg) |
| 303-338 | Shooting, possession, passing (10-game) |
| 339-346 | Attacking & defensive (10-game) |
| 347-430 | Player-level stats (10-game) |

**Total 10-game rolling features:** 130

---

## 8. Attack/Defense Strength

Derived metrics from goals scored/conceded:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 431 | `home_attack_strength_5` | float | Goals scored / league avg (5 games) |
| 432 | `away_attack_strength_5` | float | Goals scored / league avg (5 games) |
| 433 | `home_defense_strength_5` | float | Goals conceded / league avg (5 games) |
| 434 | `away_defense_strength_5` | float | Goals conceded / league avg (5 games) |
| 435 | `home_attack_strength_10` | float | Goals scored / league avg (10 games) |
| 436 | `away_attack_strength_10` | float | Goals scored / league avg (10 games) |
| 437 | `home_defense_strength_10` | float | Goals conceded / league avg (10 games) |
| 438 | `away_defense_strength_10` | float | Goals conceded / league avg (10 games) |

**Total:** 8 features

---

## 9. Standings Features

Current league position and points:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 439 | `home_position` | int | Current league position (1=best) |
| 440 | `away_position` | int | Current league position |
| 441 | `position_diff` | int | home_position - away_position |
| 442 | `home_points` | int | Current season points |
| 443 | `away_points` | int | Current season points |
| 444 | `points_diff` | int | home_points - away_points |

**Note:** Missing for ~7.3% of matches (early-season or unavailable)

**Total:** 6 features

---

## 10. Head-to-Head (H2H)

Historical matchup statistics:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 445 | `h2h_home_wins` | int | Home team H2H wins |
| 446 | `h2h_away_wins` | int | Away team H2H wins |
| 447 | `h2h_draws` | int | H2H draws |
| 448 | `h2h_home_goals_avg` | float | Avg goals scored by home in H2H |
| 449 | `h2h_away_goals_avg` | float | Avg goals scored by away in H2H |

**Lookback:** Last 10 H2H matches

**Total:** 5 features

---

## 11. Injuries

Sidelined player counts:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 450 | `home_injuries` | int | Injured/suspended home players |
| 451 | `away_injuries` | int | Injured/suspended away players |
| 452 | `injury_diff` | int | home_injuries - away_injuries |

**Total:** 3 features

---

## 12. Market/Odds

Betting market data:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 453 | `odds_home` | float | Decimal odds for home win |
| 454 | `odds_draw` | float | Decimal odds for draw |
| 455 | `odds_away` | float | Decimal odds for away win |
| 456 | `market_prob_home` | float | Implied probability (1/odds) |
| 457 | `market_prob_draw` | float | Implied probability |
| 458 | `market_prob_away` | float | Implied probability |
| 459 | `market_home_away_ratio` | float | odds_away / odds_home |
| 460 | `market_favorite` | int | Favorite team (0=away, 1=none, 2=home) |

**Total:** 8 features

---

## 13. Contextual

Match context features:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 461 | `round_num` | int | Match round number (100% missing - not calculated) |
| 462 | `season_progress` | float | % through season (100% missing - not calculated) |
| 463 | `is_early_season` | bool | First 10 games of season (zero variance - all False) |
| 464 | `day_of_week` | int | Day of week (0=Monday, 6=Sunday) |
| 465 | `is_weekend` | bool | Saturday or Sunday match |

**Total:** 5 features

---

## Feature Summary by Type

### Core Features (Always Available)
- Elo ratings: 3
- Form metrics: 18
- Attack/defense strength: 8
- H2H: 5
- Contextual: 5
- **Subtotal: 39 features**

### Rolling Window Features (May be NaN for early matches)
- 3-game rolling: 130
- 5-game rolling: 130
- 10-game rolling: 130
- **Subtotal: 390 features**

### External Data (May be missing)
- Standings: 6 (7.3% missing)
- Injuries: 3
- Market/odds: 8
- **Subtotal: 17 features**

### Metadata
- IDs and names: 7
- Targets: 6
- **Subtotal: 13 features**

### Not Calculated
- `round_num`, `season_progress`: 2 (100% missing)
- `is_early_season`: 1 (zero variance)
- **Subtotal: 3 features (can be removed)**

**Total: 465 features**

---

## Most Important Features (by XGBoost)

Based on previous model training:

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `position_diff` | 10.03 | Standings |
| 2 | `points_diff` | 7.84 | Standings |
| 3 | `elo_diff` | 2.61 | Elo |
| 4 | `home_points` | 2.53 | Standings |
| 5 | `home_player_touches_5` | 2.19 | Player stats (5-game) |
| 6 | `home_position` | 2.43 | Standings |
| 7 | `home_wins_5` | 2.37 | Form |
| 8 | `away_points` | 2.34 | Standings |
| 9 | `home_attack_strength_5` | 2.29 | Derived |
| 10 | `h2h_draws` | 2.22 | H2H |

---

## Feature Engineering Details

### Rolling Windows
- **Windows:** 3, 5, 10 games
- **Direction:** Backward-looking only (no data leakage)
- **Early matches:** NaN if insufficient history
- **Both sides:** Team stats + opponent conceded stats

### xG Approximation
Calculated from shot data:
```
xG = (shots_inside_box * 0.12) + 
     (shots_outside_box * 0.03) + 
     (big_chances * 0.23)
```

### Attack/Defense Strength
Normalized by league average:
```
attack_strength = goals_scored / league_avg_goals
defense_strength = goals_conceded / league_avg_goals_conceded
```

### Market Probabilities
Implied from decimal odds:
```
market_prob = 1 / odds
# Normalized to sum to 1.0
```

---

## Data Quality Notes

### Missing Values
- **Player stats (~75% missing):** Expected for early-season matches
- **Standings (7.3% missing):** Early-season or unavailable data
- **round_num/season_progress (100% missing):** Not implemented

### Zero Variance
- `is_early_season`: All False (can be removed)

### High Correlations (>0.95)
- `home_wins_3` ↔ `home_form_3`: 0.961
- `home_player_total_duels_3` ↔ `home_player_total_duels_conceded_3`: 0.998

**Impact:** Minimal for tree-based models

---

## Usage

### Load Features
```python
import pandas as pd
df = pd.read_csv('data/processed/sportmonks_features.csv')
print(f"Features: {df.shape[1]}")
print(f"Samples: {df.shape[0]}")
```

### Get Feature Groups
```python
# Elo features
elo_features = [c for c in df.columns if 'elo' in c]

# Rolling features
rolling_3 = [c for c in df.columns if '_3' in c]
rolling_5 = [c for c in df.columns if '_5' in c]
rolling_10 = [c for c in df.columns if '_10' in c]

# Player-level features
player_features = [c for c in df.columns if 'player_' in c]

# Standings features
standings = ['home_position', 'away_position', 'position_diff',
             'home_points', 'away_points', 'points_diff']
```

### Clean Features
```python
# Remove zero-variance and 100% missing
drop_cols = ['is_early_season', 'round_num', 'season_progress']
df = df.drop(columns=drop_cols)

# Impute standings features
df['points_diff'] = df['points_diff'].fillna(0)
df['position_diff'] = df['position_diff'].fillna(0)
```

---

**Last Updated:** 2026-01-18  
**Validation Status:** ✅ PASSED  
**Model Ready:** Yes
