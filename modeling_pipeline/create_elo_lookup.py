"""Create a lookup file with latest Elo ratings for each team."""
import pandas as pd
import json

# Load training features
features = pd.read_csv('data/processed/sportmonks_features.csv')

# Get most recent Elo rating for each team
home_elo = features.groupby('home_team_id').agg({
    'home_elo': 'last',
    'date': 'last'
}).reset_index()
home_elo.columns = ['team_id', 'elo', 'last_date']

away_elo = features.groupby('away_team_id').agg({
    'away_elo': 'last',
    'date': 'last'
}).reset_index()
away_elo.columns = ['team_id', 'elo', 'last_date']

# Combine and get the most recent for each team
all_elo = pd.concat([home_elo, away_elo])
latest_elo = all_elo.sort_values('last_date').groupby('team_id').last().reset_index()

# Convert to dictionary
elo_lookup = {}
for _, row in latest_elo.iterrows():
    elo_lookup[int(row['team_id'])] = float(row['elo'])

# Save to JSON
with open('data/processed/team_elo_ratings.json', 'w') as f:
    json.dump(elo_lookup, f, indent=2)

print(f"Created Elo lookup for {len(elo_lookup)} teams")
print(f"\nSample Elo ratings:")
for team_id in list(elo_lookup.keys())[:10]:
    print(f"  Team {team_id}: {elo_lookup[team_id]:.1f}")

print(f"\nSaved to: data/processed/team_elo_ratings.json")

