from predict_live import LiveFeatureCalculator, get_upcoming_fixtures
import pandas as pd

calculator = LiveFeatureCalculator()
fixtures = get_upcoming_fixtures('2026-01-18')

feyenoord_match = fixtures[fixtures['home_team_name'].str.contains('Feyenoord', case=False, na=False)]

if not feyenoord_match.empty:
    match = feyenoord_match.iloc[0]
    print(f"Match: {match['home_team_name']} vs {match['away_team_name']}")
    print(f"Actual result: Feyenoord lost 3-4 (Away Win!)")
    print(f"Model predicted: 69.2% home win\n")
    
    features = calculator.build_features_for_match(
        match['home_team_id'],
        match['away_team_id'],
        pd.to_datetime(match['date'])
    )
    
    if features:
        print(f"Key Features:")
        print(f"  Home Elo (Feyenoord): {features.get('home_elo', 'N/A'):.1f}")
        print(f"  Away Elo (Sparta): {features.get('away_elo', 'N/A'):.1f}")
        print(f"  Elo Diff: {features.get('elo_diff', 'N/A'):.1f}")
        print(f"  Home Form: {features.get('home_form_5', 'N/A'):.1f}")
        print(f"  Away Form: {features.get('away_form_5', 'N/A'):.1f}")
