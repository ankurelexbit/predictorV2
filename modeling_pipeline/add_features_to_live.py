#!/usr/bin/env python3
"""
Add EMA and rest days features to predict_live.py
"""

from pathlib import Path

# Read the file
file_path = Path("predict_live.py")
with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the line with "Built {len(features)} features"
insert_line = None
for i, line in enumerate(lines):
    if 'Built {len(features)} features' in line:
        insert_line = i
        break

if insert_line is None:
    print("Could not find insertion point!")
    exit(1)

print(f"Found insertion point at line {insert_line + 1}")

# Code to insert
new_code = '''
        # Calculate EMA features
        logger.info("Calculating EMA features...")
        home_ema = self.calculate_ema_features(home_matches)
        away_ema = self.calculate_ema_features(away_matches)
        
        # Add EMA features to features dict
        features.update({
            'home_goals_ema': home_ema['goals_ema'],
            'away_goals_ema': away_ema['goals_ema'],
            'home_goals_conceded_ema': home_ema['goals_conceded_ema'],
            'away_goals_conceded_ema': away_ema['goals_conceded_ema'],
            'home_xg_ema': home_ema['xg_ema'],
            'away_xg_ema': away_ema['xg_ema'],
            'home_xg_conceded_ema': home_ema['xg_conceded_ema'],
            'away_xg_conceded_ema': away_ema['xg_conceded_ema'],
            'home_shots_total_ema': home_ema['shots_total_ema'],
            'away_shots_total_ema': away_ema['shots_total_ema'],
            'home_shots_total_conceded_ema': home_ema['shots_total_conceded_ema'],
            'away_shots_total_conceded_ema': away_ema['shots_total_conceded_ema'],
            'home_shots_on_target_ema': home_ema['shots_on_target_ema'],
            'away_shots_on_target_ema': away_ema['shots_on_target_ema'],
            'home_shots_on_target_conceded_ema': home_ema['shots_on_target_conceded_ema'],
            'away_shots_on_target_conceded_ema': away_ema['shots_on_target_conceded_ema'],
            'home_possession_pct_ema': home_ema['possession_pct_ema'],
            'away_possession_pct_ema': away_ema['possession_pct_ema'],
            'home_possession_pct_conceded_ema': home_ema['possession_pct_conceded_ema'],
            'away_possession_pct_conceded_ema': away_ema['possession_pct_conceded_ema'],
        })

        # Calculate rest days features
        logger.info("Calculating rest days features...")
        home_rest = self.calculate_rest_days(home_matches, fixture_date)
        away_rest = self.calculate_rest_days(away_matches, fixture_date)
        
        # Add rest days features
        features.update({
            'days_rest_home': home_rest['days_rest'],
            'days_rest_away': away_rest['days_rest'],
            'home_short_rest': home_rest['short_rest'],
            'away_short_rest': away_rest['short_rest'],
            'rest_diff': home_rest['days_rest'] - away_rest['days_rest'],
        })

'''

# Insert the code
lines.insert(insert_line, new_code)

# Write back
with open(file_path, 'w') as f:
    f.writelines(lines)

print("✅ Successfully added EMA and rest days features!")
print(f"   Inserted {len(new_code.split(chr(10)))} lines before line {insert_line + 1}")
