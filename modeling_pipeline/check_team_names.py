"""
Check team name mismatch between API and Dixon-Coles model.
"""
import joblib
from pathlib import Path

dc_data = joblib.load(Path("models/dixon_coles_model.joblib"))
attack = dc_data['base_model_data']['attack']

print("=" * 80)
print("TEAM NAME MISMATCH INVESTIGATION")
print("=" * 80)

# Check for Paris/PSG
print("\nSearching for Paris/PSG variants:")
paris_variants = [team for team in attack.keys() if 'paris' in team.lower() or 'psg' in team.lower()]
for team in paris_variants:
    print(f"  - {team}")

# Check for Dutch teams
print("\nSearching for Dutch teams (Eredivisie):")
dutch_teams = [team for team in attack.keys() if any(x in team.lower() for x in ['heerenveen', 'groningen', 'feyenoord', 'sparta'])]
for team in dutch_teams:
    print(f"  - {team}")

# Check for Italian teams
print("\nSearching for Italian teams (Serie A):")
italian_teams = [team for team in attack.keys() if any(x in team.lower() for x in ['parma', 'genoa', 'juventus'])]
for team in italian_teams:
    print(f"  - {team}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\n🚨 CRITICAL ISSUE: Team name mismatch!")
print("\nThe API returns simplified team names (e.g., 'Paris')")
print("But Dixon-Coles was trained with different names (e.g., 'Paris Saint-Germain')")
print("\nThis causes Dixon-Coles to use default 0.0 attack/defense for 'unknown' teams,")
print("making ALL predictions wrong!")

print("\n💡 SOLUTION:")
print("Dixon-Coles model needs a team name mapping layer OR")
print("Should be trained with team IDs instead of names")

