#!/usr/bin/env python3
"""Update fetch_standings.py with better team name matching."""

from pathlib import Path

file_path = Path("fetch_standings.py")
with open(file_path, 'r') as f:
    content = f.read()

# Add Greek league
old_swiss = "    'Swiss Super League': 'sui.1',"
new_swiss = "    'Swiss Super League': 'sui.1',\n    'Greek Super League': 'gre.1',  # Added"

content = content.replace(old_swiss, new_swiss)

# Add team name variations
old_west_ham = "    'West Ham United': ['West Ham', 'West Ham United'],\n}"

new_variations = """    'West Ham United': ['West Ham', 'West Ham United'],
    # German teams
    'Bayer 04 Leverkusen': ['Bayer Leverkusen', 'Leverkusen'],
    'FC Bayern München': ['Bayern Munich', 'Bayern'],
    'Borussia Dortmund': ['Dortmund'],
    'RB Leipzig': ['Leipzig'],
    # Greek teams
    'Olympiacos F.C.': ['Olympiacos', 'Olympiakos'],
    'Panathinaikos': ['Panathinaikos Athens'],
    'AEK Athens': ['AEK Athens F.C.'],
    # Italian teams
    'Inter': ['Inter Milan', 'Internazionale'],
    'AC Milan': ['Milan', 'AC Milan'],
    # Spanish teams
    'Atlético Madrid': ['Atletico Madrid', 'Atlético de Madrid'],
    'Athletic Bilbao': ['Athletic Club'],
}"""

content = content.replace(old_west_ham, new_variations)

# Write back
with open(file_path, 'w') as f:
    f.write(content)

print("✅ Updated fetch_standings.py")
print("   - Added Greek Super League (gre.1)")
print("   - Added 12 team name variations")
