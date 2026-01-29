# Sportmonks Feature Engineering - Implementation Guide

## Quick Start Code Examples

This document provides ready-to-use Python code for implementing the feature engineering roadmap.

---

## 1. DATA EXTRACTION & PARSING

### 1.1 Complete Data Extraction Class

```python
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class SportmonksParser:
    """Parse Sportmonks API data into structured dataframes"""
    
    def __init__(self, json_file_path: str):
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        self.fixtures = self.data.get('data', [])
    
    def extract_fixtures_base(self) -> pd.DataFrame:
        """Extract base fixture information"""
        fixtures = []
        
        for fixture in self.fixtures:
            base_data = {
                'fixture_id': fixture['id'],
                'league_id': fixture['league_id'],
                'season_id': fixture['season_id'],
                'round_id': fixture['round_id'],
                'venue_id': fixture['venue_id'],
                'starting_at': fixture['starting_at'],
                'starting_at_timestamp': fixture['starting_at_timestamp'],
                'match_name': fixture['name'],
                'result_info': fixture.get('result_info'),
                'length': fixture['length'],
                'has_odds': fixture['has_odds'],
                'state_id': fixture['state_id'],
            }
            
            # Extract state info
            state = fixture.get('state', {})
            base_data['state'] = state.get('state')
            base_data['state_name'] = state.get('name')
            
            fixtures.append(base_data)
        
        df = pd.DataFrame(fixtures)
        df['starting_at'] = pd.to_datetime(df['starting_at'])
        return df
    
    def extract_participants(self) -> pd.DataFrame:
        """Extract team/participant information"""
        participants = []
        
        for fixture in self.fixtures:
            fixture_id = fixture['id']
            
            for team in fixture.get('participants', []):
                meta = team.get('meta', {})
                
                team_data = {
                    'fixture_id': fixture_id,
                    'team_id': team['id'],
                    'team_name': team['name'],
                    'short_code': team['short_code'],
                    'country_id': team['country_id'],
                    'founded': team.get('founded'),
                    'location': meta.get('location'),  # home/away
                    'winner': meta.get('winner'),
                    'league_position': meta.get('position'),
                }
                
                participants.append(team_data)
        
        return pd.DataFrame(participants)
    
    def extract_scores(self) -> pd.DataFrame:
        """Extract all score types"""
        scores = []
        
        for fixture in self.fixtures:
            fixture_id = fixture['id']
            
            for score in fixture.get('scores', []):
                score_data = {
                    'fixture_id': fixture_id,
                    'participant_id': score['participant_id'],
                    'type_id': score['type_id'],
                    'description': score['description'],
                    'goals': score['score']['goals'],
                    'participant_location': score['score']['participant'],
                }
                
                scores.append(score_data)
        
        return pd.DataFrame(scores)
    
    def extract_statistics(self) -> pd.DataFrame:
        """Extract match statistics"""
        statistics = []
        
        for fixture in self.fixtures:
            fixture_id = fixture['id']
            
            for stat in fixture.get('statistics', []):
                stat_data = {
                    'fixture_id': fixture_id,
                    'participant_id': stat['participant_id'],
                    'type_id': stat['type_id'],
                    'value': stat['data'].get('value'),
                    'location': stat.get('location'),
                }
                
                statistics.append(stat_data)
        
        return pd.DataFrame(statistics)
    
    def extract_lineups(self) -> pd.DataFrame:
        """Extract lineup information"""
        lineups = []
        
        for fixture in self.fixtures:
            fixture_id = fixture['id']
            
            for lineup in fixture.get('lineups', []):
                details = lineup.get('details', {})
                
                lineup_data = {
                    'fixture_id': fixture_id,
                    'player_id': lineup['player_id'],
                    'team_id': lineup['team_id'],
                    'position_id': lineup.get('position_id'),
                    'jersey_number': lineup.get('jersey_number'),
                    'formation_position': lineup.get('formation_position'),
                    'type_id': lineup.get('type_id'),  # 11=starter, 12=sub
                    'player_name': details.get('common_name', details.get('display_name')),
                    'height': details.get('height'),
                    'weight': details.get('weight'),
                    'date_of_birth': details.get('date_of_birth'),
                    'nationality_id': details.get('nationality_id'),
                    'detailed_position': details.get('type', {}).get('name'),
                }
                
                lineups.append(lineup_data)
        
        return pd.DataFrame(lineups)
    
    def extract_events(self) -> pd.DataFrame:
        """Extract match events (goals, cards, substitutions)"""
        events = []
        
        for fixture in self.fixtures:
            fixture_id = fixture['id']
            
            for event in fixture.get('events', []):
                event_data = {
                    'fixture_id': fixture_id,
                    'event_id': event['id'],
                    'type_id': event['type_id'],
                    'section': event.get('section'),
                    'player_id': event.get('player_id'),
                    'related_player_id': event.get('related_player_id'),
                    'participant_id': event['participant_id'],
                    'period_id': event.get('period_id'),
                    'minute': event.get('minute'),
                    'extra_minute': event.get('extra_minute', 0),
                    'injured': event.get('injured', False),
                }
                
                events.append(event_data)
        
        return pd.DataFrame(events)
    
    def extract_formations(self) -> pd.DataFrame:
        """Extract team formations"""
        formations = []
        
        for fixture in self.fixtures:
            fixture_id = fixture['id']
            
            for formation in fixture.get('formations', []):
                formation_data = {
                    'fixture_id': fixture_id,
                    'participant_id': formation['participant_id'],
                    'formation': formation['formation'],
                    'location': formation['location'],
                }
                
                formations.append(formation_data)
        
        return pd.DataFrame(formations)
    
    def extract_odds(self) -> pd.DataFrame:
        """Extract betting odds"""
        odds = []
        
        for fixture in self.fixtures:
            fixture_id = fixture['id']
            
            for odd in fixture.get('odds', []):
                odd_data = {
                    'fixture_id': fixture_id,
                    'odd_id': odd['id'],
                    'market_id': odd['market_id'],
                    'bookmaker_id': odd['bookmaker_id'],
                    'label': odd.get('label'),
                    'value': float(odd['value']),
                    'name': odd.get('name'),
                    'market_description': odd.get('market_description'),
                    'probability': odd.get('probability'),
                    'winning': odd.get('winning'),
                    'total': odd.get('total'),
                    'handicap': odd.get('handicap'),
                    'created_at': odd.get('created_at'),
                    'latest_update': odd.get('latest_bookmaker_update'),
                }
                
                odds.append(odd_data)
        
        df = pd.DataFrame(odds)
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['latest_update'] = pd.to_datetime(df['latest_update'])
        
        return df
    
    def parse_all(self) -> Dict[str, pd.DataFrame]:
        """Parse all data and return dictionary of dataframes"""
        return {
            'fixtures': self.extract_fixtures_base(),
            'participants': self.extract_participants(),
            'scores': self.extract_scores(),
            'statistics': self.extract_statistics(),
            'lineups': self.extract_lineups(),
            'events': self.extract_events(),
            'formations': self.extract_formations(),
            'odds': self.extract_odds(),
        }


# Usage
parser = SportmonksParser('sportmonks_sample_formatted.json')
dataframes = parser.parse_all()

# Save to CSV
for name, df in dataframes.items():
    df.to_csv(f'{name}.csv', index=False)
    print(f"Saved {name}.csv with {len(df)} rows")
```

---

## 2. FEATURE ENGINEERING PIPELINE

### 2.1 Score Feature Extractor

```python
class ScoreFeatureExtractor:
    """Extract features from score data"""
    
    def __init__(self, fixtures_df: pd.DataFrame, scores_df: pd.DataFrame, 
                 participants_df: pd.DataFrame):
        self.fixtures_df = fixtures_df
        self.scores_df = scores_df
        self.participants_df = participants_df
    
    def extract_match_scores(self) -> pd.DataFrame:
        """Extract final and half-time scores for each match"""
        features = []
        
        for fixture_id in self.fixtures_df['fixture_id'].unique():
            fixture_scores = self.scores_df[
                self.scores_df['fixture_id'] == fixture_id
            ]
            
            fixture_data = {'fixture_id': fixture_id}
            
            # Get home and away team IDs
            participants = self.participants_df[
                self.participants_df['fixture_id'] == fixture_id
            ]
            home_id = participants[participants['location'] == 'home']['team_id'].values[0]
            away_id = participants[participants['location'] == 'away']['team_id'].values[0]
            
            # Full-time scores (CURRENT)
            ft_scores = fixture_scores[fixture_scores['description'] == 'CURRENT']
            home_ft = ft_scores[ft_scores['participant_id'] == home_id]['goals'].values
            away_ft = ft_scores[ft_scores['participant_id'] == away_id]['goals'].values
            
            fixture_data['home_goals_ft'] = home_ft[0] if len(home_ft) > 0 else None
            fixture_data['away_goals_ft'] = away_ft[0] if len(away_ft) > 0 else None
            
            # Half-time scores (1ST_HALF)
            ht_scores = fixture_scores[fixture_scores['description'] == '1ST_HALF']
            home_ht = ht_scores[ht_scores['participant_id'] == home_id]['goals'].values
            away_ht = ht_scores[ht_scores['participant_id'] == away_id]['goals'].values
            
            fixture_data['home_goals_ht'] = home_ht[0] if len(home_ht) > 0 else None
            fixture_data['away_goals_ht'] = away_ht[0] if len(away_ht) > 0 else None
            
            # 2nd half scores
            if fixture_data['home_goals_ft'] is not None and fixture_data['home_goals_ht'] is not None:
                fixture_data['home_goals_2h'] = fixture_data['home_goals_ft'] - fixture_data['home_goals_ht']
                fixture_data['away_goals_2h'] = fixture_data['away_goals_ft'] - fixture_data['away_goals_ht']
            
            # Derived features
            if fixture_data['home_goals_ft'] is not None:
                fixture_data['total_goals'] = fixture_data['home_goals_ft'] + fixture_data['away_goals_ft']
                fixture_data['goal_difference'] = fixture_data['home_goals_ft'] - fixture_data['away_goals_ft']
                
                # Match result
                if fixture_data['goal_difference'] > 0:
                    fixture_data['result'] = 'home_win'
                elif fixture_data['goal_difference'] < 0:
                    fixture_data['result'] = 'away_win'
                else:
                    fixture_data['result'] = 'draw'
                
                # Binary targets
                fixture_data['home_win'] = int(fixture_data['result'] == 'home_win')
                fixture_data['away_win'] = int(fixture_data['result'] == 'away_win')
                fixture_data['draw'] = int(fixture_data['result'] == 'draw')
                
                # Goal thresholds
                fixture_data['over_1_5'] = int(fixture_data['total_goals'] > 1.5)
                fixture_data['over_2_5'] = int(fixture_data['total_goals'] > 2.5)
                fixture_data['over_3_5'] = int(fixture_data['total_goals'] > 3.5)
                
                # Both teams to score
                fixture_data['btts'] = int(
                    fixture_data['home_goals_ft'] > 0 and fixture_data['away_goals_ft'] > 0
                )
                
                # Clean sheet
                fixture_data['home_clean_sheet'] = int(fixture_data['away_goals_ft'] == 0)
                fixture_data['away_clean_sheet'] = int(fixture_data['home_goals_ft'] == 0)
                
                # Half-time / Full-time
                if fixture_data['home_goals_ht'] is not None:
                    ht_result = 'H' if fixture_data['home_goals_ht'] > fixture_data['away_goals_ht'] else \
                                'A' if fixture_data['home_goals_ht'] < fixture_data['away_goals_ht'] else 'D'
                    ft_result = fixture_data['result'][0].upper()
                    fixture_data['ht_ft'] = ht_result + ft_result
                    
                    # Comeback
                    fixture_data['comeback'] = int(ht_result != ft_result and ft_result != 'D')
            
            features.append(fixture_data)
        
        return pd.DataFrame(features)


# Usage
score_extractor = ScoreFeatureExtractor(
    fixtures_df=dataframes['fixtures'],
    scores_df=dataframes['scores'],
    participants_df=dataframes['participants']
)
score_features = score_extractor.extract_match_scores()
```

### 2.2 Statistics Feature Extractor

```python
class StatisticsFeatureExtractor:
    """Extract and compute features from match statistics"""
    
    # Mapping of type_ids to feature names
    STAT_MAPPING = {
        45: 'possession',
        52: 'shots_total',
        53: 'shots_on_target',
        54: 'shots_off_target',
        55: 'corners',
        56: 'shots_blocked',
        60: 'attacks',
        62: 'dangerous_attacks',
        73: 'tackles',
        74: 'fouls',
        75: 'yellow_cards',
        76: 'red_cards',
        79: 'interceptions',
        80: 'passes_total',
        81: 'passes_accurate',
        82: 'passes_accuracy',
        83: 'throw_ins',
        84: 'blocks',
        86: 'shots_inside_box',
        87: 'shots_outside_box',
        88: 'saves',
        89: 'saves_inside_box',
        90: 'hit_woodwork',
        106: 'yellow_red_cards',
        1605: 'pass_accuracy_pct',
    }
    
    def __init__(self, fixtures_df: pd.DataFrame, statistics_df: pd.DataFrame,
                 participants_df: pd.DataFrame):
        self.fixtures_df = fixtures_df
        self.statistics_df = statistics_df
        self.participants_df = participants_df
    
    def pivot_statistics(self) -> pd.DataFrame:
        """Pivot statistics into wide format"""
        features = []
        
        for fixture_id in self.fixtures_df['fixture_id'].unique():
            fixture_stats = self.statistics_df[
                self.statistics_df['fixture_id'] == fixture_id
            ]
            
            participants = self.participants_df[
                self.participants_df['fixture_id'] == fixture_id
            ]
            home_id = participants[participants['location'] == 'home']['team_id'].values[0]
            away_id = participants[participants['location'] == 'away']['team_id'].values[0]
            
            fixture_data = {'fixture_id': fixture_id}
            
            # Extract stats for home and away
            for team_id, prefix in [(home_id, 'home'), (away_id, 'away')]:
                team_stats = fixture_stats[fixture_stats['participant_id'] == team_id]
                
                for type_id, stat_name in self.STAT_MAPPING.items():
                    stat_row = team_stats[team_stats['type_id'] == type_id]
                    value = stat_row['value'].values[0] if len(stat_row) > 0 else None
                    fixture_data[f'{prefix}_{stat_name}'] = value
            
            features.append(fixture_data)
        
        return pd.DataFrame(features)
    
    def compute_derived_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived statistics"""
        
        # Shot accuracy
        df['home_shot_accuracy'] = np.where(
            df['home_shots_total'] > 0,
            df['home_shots_on_target'] / df['home_shots_total'],
            0
        )
        df['away_shot_accuracy'] = np.where(
            df['away_shots_total'] > 0,
            df['away_shots_on_target'] / df['away_shots_total'],
            0
        )
        
        # Attack efficiency
        df['home_attack_efficiency'] = np.where(
            df['home_attacks'] > 0,
            df['home_dangerous_attacks'] / df['home_attacks'],
            0
        )
        df['away_attack_efficiency'] = np.where(
            df['away_attacks'] > 0,
            df['away_dangerous_attacks'] / df['away_attacks'],
            0
        )
        
        # Pass completion
        df['home_pass_completion'] = np.where(
            df['home_passes_total'] > 0,
            df['home_passes_accurate'] / df['home_passes_total'],
            0
        )
        df['away_pass_completion'] = np.where(
            df['away_passes_total'] > 0,
            df['away_passes_accurate'] / df['away_passes_total'],
            0
        )
        
        # Relative statistics (home vs away)
        df['possession_difference'] = df['home_possession'] - df['away_possession']
        df['shot_difference'] = df['home_shots_total'] - df['away_shots_total']
        df['corner_difference'] = df['home_corners'] - df['away_corners']
        df['attack_difference'] = df['home_attacks'] - df['away_attacks']
        
        # Total cards
        df['home_total_cards'] = df['home_yellow_cards'].fillna(0) + df['home_red_cards'].fillna(0)
        df['away_total_cards'] = df['away_yellow_cards'].fillna(0) + df['away_red_cards'].fillna(0)
        df['total_cards'] = df['home_total_cards'] + df['away_total_cards']
        
        return df
    
    def extract_all(self) -> pd.DataFrame:
        """Extract and compute all statistics features"""
        stats_df = self.pivot_statistics()
        stats_df = self.compute_derived_stats(stats_df)
        return stats_df


# Usage
stats_extractor = StatisticsFeatureExtractor(
    fixtures_df=dataframes['fixtures'],
    statistics_df=dataframes['statistics'],
    participants_df=dataframes['participants']
)
stats_features = stats_extractor.extract_all()
```

### 2.3 Event Features Extractor

```python
class EventFeatureExtractor:
    """Extract features from match events"""
    
    EVENT_TYPES = {
        14: 'goal',
        17: 'penalty',
        18: 'own_goal',
        79: 'red_card',
        80: 'yellow_red_card',
        83: 'yellow_card',
        87: 'substitution',
    }
    
    def __init__(self, fixtures_df: pd.DataFrame, events_df: pd.DataFrame,
                 participants_df: pd.DataFrame):
        self.fixtures_df = fixtures_df
        self.events_df = events_df
        self.participants_df = participants_df
    
    def extract_event_features(self) -> pd.DataFrame:
        """Extract event-based features"""
        features = []
        
        for fixture_id in self.fixtures_df['fixture_id'].unique():
            fixture_events = self.events_df[
                self.events_df['fixture_id'] == fixture_id
            ]
            
            participants = self.participants_df[
                self.participants_df['fixture_id'] == fixture_id
            ]
            home_id = participants[participants['location'] == 'home']['team_id'].values[0]
            away_id = participants[participants['location'] == 'away']['team_id'].values[0]
            
            fixture_data = {'fixture_id': fixture_id}
            
            # Goal events
            goal_events = fixture_events[fixture_events['type_id'].isin([14, 17, 18])]
            
            if len(goal_events) > 0:
                goal_minutes = (goal_events['minute'] + goal_events['extra_minute']).tolist()
                fixture_data['first_goal_minute'] = min(goal_minutes)
                fixture_data['last_goal_minute'] = max(goal_minutes)
                
                # Goals in different periods
                fixture_data['goals_before_30'] = len(goal_events[goal_events['minute'] < 30])
                fixture_data['goals_30_to_60'] = len(goal_events[
                    (goal_events['minute'] >= 30) & (goal_events['minute'] < 60)
                ])
                fixture_data['goals_after_60'] = len(goal_events[goal_events['minute'] >= 60])
                fixture_data['goals_after_75'] = len(goal_events[goal_events['minute'] >= 75])
                
                # Penalties and own goals
                fixture_data['penalty_count'] = len(goal_events[goal_events['type_id'] == 17])
                fixture_data['own_goal_count'] = len(goal_events[goal_events['type_id'] == 18])
            else:
                fixture_data['first_goal_minute'] = None
                fixture_data['last_goal_minute'] = None
                fixture_data['goals_before_30'] = 0
                fixture_data['goals_30_to_60'] = 0
                fixture_data['goals_after_60'] = 0
                fixture_data['goals_after_75'] = 0
                fixture_data['penalty_count'] = 0
                fixture_data['own_goal_count'] = 0
            
            # Card events
            for team_id, prefix in [(home_id, 'home'), (away_id, 'away')]:
                team_events = fixture_events[fixture_events['participant_id'] == team_id]
                
                yellow_cards = team_events[team_events['type_id'] == 83]
                red_cards = team_events[team_events['type_id'].isin([79, 80])]
                
                fixture_data[f'{prefix}_yellow_cards_events'] = len(yellow_cards)
                fixture_data[f'{prefix}_red_cards_events'] = len(red_cards)
                
                if len(red_cards) > 0:
                    fixture_data[f'{prefix}_first_red_minute'] = \
                        (red_cards['minute'] + red_cards['extra_minute']).min()
                else:
                    fixture_data[f'{prefix}_first_red_minute'] = None
                
                # Substitutions
                subs = team_events[team_events['type_id'] == 87]
                fixture_data[f'{prefix}_substitutions'] = len(subs)
                
                if len(subs) > 0:
                    sub_minutes = (subs['minute'] + subs['extra_minute']).tolist()
                    fixture_data[f'{prefix}_avg_sub_minute'] = np.mean(sub_minutes)
                    fixture_data[f'{prefix}_first_sub_minute'] = min(sub_minutes)
                else:
                    fixture_data[f'{prefix}_avg_sub_minute'] = None
                    fixture_data[f'{prefix}_first_sub_minute'] = None
            
            features.append(fixture_data)
        
        return pd.DataFrame(features)


# Usage
event_extractor = EventFeatureExtractor(
    fixtures_df=dataframes['fixtures'],
    events_df=dataframes['events'],
    participants_df=dataframes['participants']
)
event_features = event_extractor.extract_event_features()
```

### 2.4 Lineup Features Extractor

```python
class LineupFeatureExtractor:
    """Extract features from lineup data"""
    
    def __init__(self, fixtures_df: pd.DataFrame, lineups_df: pd.DataFrame,
                 participants_df: pd.DataFrame):
        self.fixtures_df = fixtures_df
        self.lineups_df = lineups_df
        self.participants_df = participants_df
    
    def extract_lineup_features(self) -> pd.DataFrame:
        """Extract lineup-based features"""
        features = []
        
        for fixture_id in self.fixtures_df['fixture_id'].unique():
            fixture_lineups = self.lineups_df[
                self.lineups_df['fixture_id'] == fixture_id
            ]
            
            participants = self.participants_df[
                self.participants_df['fixture_id'] == fixture_id
            ]
            home_id = participants[participants['location'] == 'home']['team_id'].values[0]
            away_id = participants[participants['location'] == 'away']['team_id'].values[0]
            
            fixture_data = {'fixture_id': fixture_id}
            
            for team_id, prefix in [(home_id, 'home'), (away_id, 'away')]:
                team_lineup = fixture_lineups[fixture_lineups['team_id'] == team_id]
                
                # Starter vs substitutes
                starters = team_lineup[team_lineup['type_id'] == 11]
                subs = team_lineup[team_lineup['type_id'] == 12]
                
                fixture_data[f'{prefix}_starters'] = len(starters)
                fixture_data[f'{prefix}_substitutes'] = len(subs)
                
                # Age statistics
                if 'date_of_birth' in team_lineup.columns:
                    dob = pd.to_datetime(team_lineup['date_of_birth'], errors='coerce')
                    match_date = self.fixtures_df[
                        self.fixtures_df['fixture_id'] == fixture_id
                    ]['starting_at'].values[0]
                    match_date = pd.to_datetime(match_date)
                    
                    ages = (match_date - dob).dt.days / 365.25
                    fixture_data[f'{prefix}_avg_age'] = ages.mean()
                    fixture_data[f'{prefix}_min_age'] = ages.min()
                    fixture_data[f'{prefix}_max_age'] = ages.max()
                
                # Physical attributes
                if 'height' in team_lineup.columns:
                    heights = pd.to_numeric(team_lineup['height'], errors='coerce')
                    fixture_data[f'{prefix}_avg_height'] = heights.mean()
                
                if 'weight' in team_lineup.columns:
                    weights = pd.to_numeric(team_lineup['weight'], errors='coerce')
                    fixture_data[f'{prefix}_avg_weight'] = weights.mean()
                
                # Nationality diversity
                if 'nationality_id' in team_lineup.columns:
                    unique_nationalities = team_lineup['nationality_id'].nunique()
                    fixture_data[f'{prefix}_nationality_diversity'] = unique_nationalities / len(team_lineup)
            
            features.append(fixture_data)
        
        return pd.DataFrame(features)


# Usage
lineup_extractor = LineupFeatureExtractor(
    fixtures_df=dataframes['fixtures'],
    lineups_df=dataframes['lineups'],
    participants_df=dataframes['participants']
)
lineup_features = lineup_extractor.extract_lineup_features()
```

---

## 3. ROLLING WINDOW FEATURES

### 3.1 Rolling Statistics Calculator

```python
class RollingFeatureCalculator:
    """Calculate rolling window features for teams"""
    
    def __init__(self, fixtures_df: pd.DataFrame, score_features: pd.DataFrame,
                 stats_features: pd.DataFrame, participants_df: pd.DataFrame):
        self.fixtures_df = fixtures_df.sort_values('starting_at')
        self.score_features = score_features
        self.stats_features = stats_features
        self.participants_df = participants_df
        
        # Merge all features
        self.full_df = self.fixtures_df.merge(
            score_features, on='fixture_id', how='left'
        ).merge(
            stats_features, on='fixture_id', how='left'
        )
    
    def create_team_level_data(self) -> pd.DataFrame:
        """Convert fixture-level data to team-level data"""
        rows = []
        
        for _, row in self.full_df.iterrows():
            fixture_id = row['fixture_id']
            participants = self.participants_df[
                self.participants_df['fixture_id'] == fixture_id
            ]
            
            # Home team row
            home = participants[participants['location'] == 'home'].iloc[0]
            home_row = {
                'team_id': home['team_id'],
                'fixture_id': fixture_id,
                'date': row['starting_at'],
                'location': 'home',
                'opponent_id': participants[participants['location'] == 'away']['team_id'].values[0],
                'goals_for': row.get('home_goals_ft'),
                'goals_against': row.get('away_goals_ft'),
                'result': 'W' if row.get('home_win') == 1 else 'D' if row.get('draw') == 1 else 'L',
                'points': 3 if row.get('home_win') == 1 else 1 if row.get('draw') == 1 else 0,
                'clean_sheet': row.get('home_clean_sheet', 0),
                'shots': row.get('home_shots_total'),
                'shots_on_target': row.get('home_shots_on_target'),
                'possession': row.get('home_possession'),
                'corners': row.get('home_corners'),
                'fouls': row.get('home_fouls'),
                'yellow_cards': row.get('home_yellow_cards'),
                'red_cards': row.get('home_red_cards'),
            }
            rows.append(home_row)
            
            # Away team row
            away = participants[participants['location'] == 'away'].iloc[0]
            away_row = {
                'team_id': away['team_id'],
                'fixture_id': fixture_id,
                'date': row['starting_at'],
                'location': 'away',
                'opponent_id': participants[participants['location'] == 'home']['team_id'].values[0],
                'goals_for': row.get('away_goals_ft'),
                'goals_against': row.get('home_goals_ft'),
                'result': 'W' if row.get('away_win') == 1 else 'D' if row.get('draw') == 1 else 'L',
                'points': 3 if row.get('away_win') == 1 else 1 if row.get('draw') == 1 else 0,
                'clean_sheet': row.get('away_clean_sheet', 0),
                'shots': row.get('away_shots_total'),
                'shots_on_target': row.get('away_shots_on_target'),
                'possession': row.get('away_possession'),
                'corners': row.get('away_corners'),
                'fouls': row.get('away_fouls'),
                'yellow_cards': row.get('away_yellow_cards'),
                'red_cards': row.get('away_red_cards'),
            }
            rows.append(away_row)
        
        return pd.DataFrame(rows).sort_values(['team_id', 'date'])
    
    def calculate_rolling_features(self, team_df: pd.DataFrame, 
                                   windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Calculate rolling window features for a team"""
        
        for window in windows:
            # Results
            team_df[f'win_rate_L{window}'] = (
                team_df['result'].eq('W').rolling(window, min_periods=1).mean()
            )
            team_df[f'draw_rate_L{window}'] = (
                team_df['result'].eq('D').rolling(window, min_periods=1).mean()
            )
            
            # Points
            team_df[f'points_L{window}'] = (
                team_df['points'].rolling(window, min_periods=1).sum()
            )
            team_df[f'ppg_L{window}'] = (  # points per game
                team_df['points'].rolling(window, min_periods=1).mean()
            )
            
            # Goals
            team_df[f'goals_scored_avg_L{window}'] = (
                team_df['goals_for'].rolling(window, min_periods=1).mean()
            )
            team_df[f'goals_conceded_avg_L{window}'] = (
                team_df['goals_against'].rolling(window, min_periods=1).mean()
            )
            team_df[f'goal_difference_L{window}'] = (
                team_df[f'goals_scored_avg_L{window}'] - team_df[f'goals_conceded_avg_L{window}']
            )
            
            # Clean sheets
            team_df[f'clean_sheets_L{window}'] = (
                team_df['clean_sheet'].rolling(window, min_periods=1).sum()
            )
            
            # BTTS
            team_df[f'btts_L{window}'] = (
                ((team_df['goals_for'] > 0) & (team_df['goals_against'] > 0))
                .rolling(window, min_periods=1).mean()
            )
            
            # Over 2.5
            team_df[f'over_2_5_L{window}'] = (
                ((team_df['goals_for'] + team_df['goals_against']) > 2.5)
                .rolling(window, min_periods=1).mean()
            )
            
            # Stats (if available)
            for col in ['shots', 'shots_on_target', 'possession', 'corners']:
                if col in team_df.columns:
                    team_df[f'{col}_avg_L{window}'] = (
                        team_df[col].rolling(window, min_periods=1).mean()
                    )
        
        return team_df
    
    def calculate_home_away_rolling(self, team_df: pd.DataFrame,
                                    windows: List[int] = [3, 5]) -> pd.DataFrame:
        """Calculate home/away specific rolling features"""
        
        for window in windows:
            for location in ['home', 'away']:
                location_mask = team_df['location'] == location
                location_df = team_df[location_mask].copy()
                
                # Calculate features for this location
                location_df[f'{location}_win_rate_L{window}'] = (
                    location_df['result'].eq('W').rolling(window, min_periods=1).mean()
                )
                location_df[f'{location}_goals_avg_L{window}'] = (
                    location_df['goals_for'].rolling(window, min_periods=1).mean()
                )
                location_df[f'{location}_goals_conceded_avg_L{window}'] = (
                    location_df['goals_against'].rolling(window, min_periods=1).mean()
                )
                
                # Merge back to main dataframe
                for col in location_df.columns:
                    if col.startswith(location):
                        team_df.loc[location_mask, col] = location_df[col]
                        # Fill NaN for opposite location
                        team_df[col] = team_df[col].fillna(0)
        
        return team_df
    
    def compute_all_rolling_features(self) -> pd.DataFrame:
        """Compute rolling features for all teams"""
        team_level = self.create_team_level_data()
        
        # Calculate features per team
        teams = []
        for team_id in team_level['team_id'].unique():
            team_df = team_level[team_level['team_id'] == team_id].copy()
            team_df = self.calculate_rolling_features(team_df)
            team_df = self.calculate_home_away_rolling(team_df)
            teams.append(team_df)
        
        return pd.concat(teams, ignore_index=True)


# Usage
rolling_calculator = RollingFeatureCalculator(
    fixtures_df=dataframes['fixtures'],
    score_features=score_features,
    stats_features=stats_features,
    participants_df=dataframes['participants']
)
rolling_features = rolling_calculator.compute_all_rolling_features()

# Merge back to fixture level
def merge_rolling_to_fixture(fixtures_df, rolling_features, participants_df):
    """Merge team-level rolling features back to fixture level"""
    fixture_features = []
    
    for fixture_id in fixtures_df['fixture_id'].unique():
        participants = participants_df[participants_df['fixture_id'] == fixture_id]
        rolling_fixture = rolling_features[rolling_features['fixture_id'] == fixture_id]
        
        home_id = participants[participants['location'] == 'home']['team_id'].values[0]
        away_id = participants[participants['location'] == 'away']['team_id'].values[0]
        
        home_features = rolling_fixture[rolling_fixture['team_id'] == home_id]
        away_features = rolling_fixture[rolling_fixture['team_id'] == away_id]
        
        fixture_row = {'fixture_id': fixture_id}
        
        # Add home features with prefix
        for col in home_features.columns:
            if col not in ['team_id', 'fixture_id', 'date', 'opponent_id']:
                fixture_row[f'home_{col}'] = home_features[col].values[0] if len(home_features) > 0 else None
        
        # Add away features with prefix
        for col in away_features.columns:
            if col not in ['team_id', 'fixture_id', 'date', 'opponent_id']:
                fixture_row[f'away_{col}'] = away_features[col].values[0] if len(away_features) > 0 else None
        
        fixture_features.append(fixture_row)
    
    return pd.DataFrame(fixture_features)

fixture_rolling_features = merge_rolling_to_fixture(
    dataframes['fixtures'], rolling_features, dataframes['participants']
)
```

---

## 4. ODDS FEATURES

### 4.1 Odds Feature Extractor

```python
class OddsFeatureExtractor:
    """Extract betting odds features"""
    
    # Market mappings
    MARKET_IDS = {
        1: '1x2',  # 3-way result
        2: 'double_chance',
        3: 'over_under',
        6: 'asian_handicap',
        31: 'half_time',
        44: 'odd_even',
        57: 'correct_score',
        80: 'total_goals',
    }
    
    def __init__(self, fixtures_df: pd.DataFrame, odds_df: pd.DataFrame):
        self.fixtures_df = fixtures_df
        self.odds_df = odds_df
    
    def extract_market_odds(self, fixture_id: int, market_id: int) -> Dict:
        """Extract odds for a specific market"""
        market_odds = self.odds_df[
            (self.odds_df['fixture_id'] == fixture_id) &
            (self.odds_df['market_id'] == market_id)
        ]
        
        if len(market_odds) == 0:
            return {}
        
        features = {}
        
        # Get unique labels (e.g., Home, Draw, Away)
        labels = market_odds['label'].unique()
        
        for label in labels:
            label_odds = market_odds[market_odds['label'] == label]
            
            # Average odds across bookmakers
            avg_odds = label_odds['value'].mean()
            features[f'odds_{label.lower()}'] = avg_odds
            
            # Implied probability
            features[f'prob_{label.lower()}'] = 1 / avg_odds if avg_odds > 0 else None
            
            # Odds variance (market disagreement)
            if len(label_odds) > 1:
                features[f'odds_var_{label.lower()}'] = label_odds['value'].var()
            
            # Best and worst odds
            features[f'odds_best_{label.lower()}'] = label_odds['value'].max()
            features[f'odds_worst_{label.lower()}'] = label_odds['value'].min()
        
        # Market margin
        if all(f'prob_{label.lower()}' in features for label in labels):
            total_prob = sum(features[f'prob_{label.lower()}'] for label in labels)
            features['market_margin'] = total_prob - 1
        
        return features
    
    def extract_1x2_odds(self) -> pd.DataFrame:
        """Extract 1X2 (match result) odds"""
        features = []
        
        for fixture_id in self.fixtures_df['fixture_id'].unique():
            fixture_data = {'fixture_id': fixture_id}
            
            # Get 1X2 odds
            odds_1x2 = self.extract_market_odds(fixture_id, market_id=1)
            fixture_data.update(odds_1x2)
            
            # Favorite/underdog
            if 'odds_home' in odds_1x2 and 'odds_away' in odds_1x2:
                fixture_data['favorite'] = 'home' if odds_1x2['odds_home'] < odds_1x2['odds_away'] else 'away'
                fixture_data['odds_favorite'] = min(odds_1x2['odds_home'], odds_1x2['odds_away'])
                fixture_data['odds_underdog'] = max(odds_1x2['odds_home'], odds_1x2['odds_away'])
                fixture_data['odds_spread'] = fixture_data['odds_underdog'] - fixture_data['odds_favorite']
            
            features.append(fixture_data)
        
        return pd.DataFrame(features)
    
    def extract_over_under_odds(self) -> pd.DataFrame:
        """Extract over/under goal odds"""
        features = []
        
        for fixture_id in self.fixtures_df['fixture_id'].unique():
            fixture_data = {'fixture_id': fixture_id}
            
            # Common lines: 2.5, 3.5
            for total_line in [2.5, 3.5]:
                market_odds = self.odds_df[
                    (self.odds_df['fixture_id'] == fixture_id) &
                    (self.odds_df['market_id'].isin([3, 80])) &
                    (self.odds_df['total'] == total_line)
                ]
                
                if len(market_odds) > 0:
                    over_odds = market_odds[market_odds['label'] == 'Over']['value'].mean()
                    under_odds = market_odds[market_odds['label'] == 'Under']['value'].mean()
                    
                    fixture_data[f'odds_over_{total_line}'] = over_odds
                    fixture_data[f'odds_under_{total_line}'] = under_odds
                    fixture_data[f'prob_over_{total_line}'] = 1 / over_odds if over_odds > 0 else None
            
            features.append(fixture_data)
        
        return pd.DataFrame(features)
    
    def extract_odds_movement(self) -> pd.DataFrame:
        """Extract odds movement features"""
        features = []
        
        for fixture_id in self.fixtures_df['fixture_id'].unique():
            fixture_data = {'fixture_id': fixture_id}
            
            fixture_odds = self.odds_df[
                (self.odds_df['fixture_id'] == fixture_id) &
                (self.odds_df['market_id'] == 1)  # 1X2 market
            ]
            
            for label in ['Home', 'Draw', 'Away']:
                label_odds = fixture_odds[fixture_odds['label'] == label]
                
                if len(label_odds) > 0:
                    # Opening odds (earliest created_at)
                    opening = label_odds.sort_values('created_at').iloc[0]['value']
                    # Closing odds (latest updated)
                    closing = label_odds.sort_values('latest_update').iloc[-1]['value']
                    
                    fixture_data[f'odds_opening_{label.lower()}'] = opening
                    fixture_data[f'odds_closing_{label.lower()}'] = closing
                    fixture_data[f'odds_movement_{label.lower()}'] = closing - opening
                    fixture_data[f'odds_movement_pct_{label.lower()}'] = \
                        ((closing - opening) / opening) * 100 if opening > 0 else 0
            
            features.append(fixture_data)
        
        return pd.DataFrame(features)
    
    def extract_all_odds_features(self) -> pd.DataFrame:
        """Extract all odds-related features"""
        odds_1x2 = self.extract_1x2_odds()
        odds_ou = self.extract_over_under_odds()
        odds_movement = self.extract_odds_movement()
        
        # Merge all odds features
        result = odds_1x2
        for df in [odds_ou, odds_movement]:
            result = result.merge(df, on='fixture_id', how='left')
        
        return result


# Usage
odds_extractor = OddsFeatureExtractor(
    fixtures_df=dataframes['fixtures'],
    odds_df=dataframes['odds']
)
odds_features = odds_extractor.extract_all_odds_features()
```

---

## 5. COMPLETE PIPELINE

### 5.1 Master Feature Engineering Pipeline

```python
class FootballFeaturesPipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self, json_file_path: str):
        # Parse data
        parser = SportmonksParser(json_file_path)
        self.dataframes = parser.parse_all()
        
        # Initialize extractors
        self.score_extractor = ScoreFeatureExtractor(
            self.dataframes['fixtures'],
            self.dataframes['scores'],
            self.dataframes['participants']
        )
        
        self.stats_extractor = StatisticsFeatureExtractor(
            self.dataframes['fixtures'],
            self.dataframes['statistics'],
            self.dataframes['participants']
        )
        
        self.event_extractor = EventFeatureExtractor(
            self.dataframes['fixtures'],
            self.dataframes['events'],
            self.dataframes['participants']
        )
        
        self.lineup_extractor = LineupFeatureExtractor(
            self.dataframes['fixtures'],
            self.dataframes['lineups'],
            self.dataframes['participants']
        )
        
        self.odds_extractor = OddsFeatureExtractor(
            self.dataframes['fixtures'],
            self.dataframes['odds']
        )
    
    def extract_all_features(self) -> pd.DataFrame:
        """Extract all features and merge into single dataframe"""
        print("Extracting score features...")
        score_features = self.score_extractor.extract_match_scores()
        
        print("Extracting statistics features...")
        stats_features = self.stats_extractor.extract_all()
        
        print("Extracting event features...")
        event_features = self.event_extractor.extract_event_features()
        
        print("Extracting lineup features...")
        lineup_features = self.lineup_extractor.extract_lineup_features()
        
        print("Extracting odds features...")
        odds_features = self.odds_extractor.extract_all_odds_features()
        
        print("Calculating rolling features...")
        rolling_calculator = RollingFeatureCalculator(
            self.dataframes['fixtures'],
            score_features,
            stats_features,
            self.dataframes['participants']
        )
        rolling_features = rolling_calculator.compute_all_rolling_features()
        fixture_rolling = merge_rolling_to_fixture(
            self.dataframes['fixtures'],
            rolling_features,
            self.dataframes['participants']
        )
        
        print("Merging all features...")
        # Start with base fixtures
        final_df = self.dataframes['fixtures'].copy()
        
        # Merge all feature sets
        for df in [score_features, stats_features, event_features, 
                   lineup_features, odds_features, fixture_rolling]:
            final_df = final_df.merge(df, on='fixture_id', how='left')
        
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Total features: {final_df.shape[1]}")
        
        return final_df
    
    def save_features(self, output_path: str = 'features_complete.csv'):
        """Extract and save all features"""
        features_df = self.extract_all_features()
        features_df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
        return features_df


# Usage - Complete Pipeline
pipeline = FootballFeaturesPipeline('sportmonks_sample_formatted.json')
complete_features = pipeline.save_features()

# View feature summary
print("\nFeature Summary:")
print(f"Total fixtures: {len(complete_features)}")
print(f"Total features: {complete_features.shape[1]}")
print(f"\nFeature categories:")
print(f"- Basic info: {len([c for c in complete_features.columns if c.startswith('fixture')])}")
print(f"- Score features: {len([c for c in complete_features.columns if 'goal' in c.lower()])}")
print(f"- Statistics: {len([c for c in complete_features.columns if any(s in c for s in ['shot', 'pass', 'tackle'])])}")
print(f"- Odds features: {len([c for c in complete_features.columns if 'odds' in c])}")
print(f"- Rolling features: {len([c for c in complete_features.columns if '_L' in c])}")
```

---

## 6. USAGE EXAMPLE

```python
# Complete workflow
if __name__ == '__main__':
    # Initialize pipeline
    pipeline = FootballFeaturesPipeline('sportmonks_sample_formatted.json')
    
    # Extract all features
    features = pipeline.extract_all_features()
    
    # Save to CSV
    features.to_csv('football_features_complete.csv', index=False)
    
    # Quick analysis
    print("\n=== FEATURE ENGINEERING COMPLETE ===")
    print(f"Dataset shape: {features.shape}")
    print(f"\nSample features:")
    print(features[['fixture_id', 'match_name', 'result', 
                   'total_goals', 'odds_home', 'home_win_rate_L5']].head())
    
    # Check for missing values
    print(f"\nMissing values per feature (top 10):")
    missing = features.isnull().sum().sort_values(ascending=False).head(10)
    print(missing)
    
    # Export feature list
    with open('feature_list.txt', 'w') as f:
        for col in sorted(features.columns):
            f.write(f"{col}\n")
    
    print("\n✅ Feature engineering pipeline completed successfully!")
```

---

## NOTES

- This implementation guide provides production-ready code for all major feature engineering tasks
- The code is modular and can be extended with additional features
- Remember to handle data leakage by only using historical data for predictions
- Always validate on a time-based split, never random split
- The pipeline can process any number of fixtures from the Sportmonks API

**Next Steps:**
1. Run the pipeline on your data
2. Perform exploratory data analysis on the features
3. Train machine learning models
4. Evaluate and iterate
