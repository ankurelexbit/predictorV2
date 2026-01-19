"""Compare old vs new predictions to see the improvement."""
import pandas as pd

print("=" * 80)
print("COMPARING OLD VS NEW PREDICTIONS")
print("=" * 80)

old = pd.read_csv('predictions_jan_18_stacking.csv')
new = pd.read_csv('predictions_jan_18_fixed.csv')

print("\n1. PREDICTION DISTRIBUTION")
print("-" * 80)

# Count predictions
old_home = (old['predicted_outcome'] == 'Home Win').sum()
old_draw = (old['predicted_outcome'] == 'Draw').sum()
old_away = (old['predicted_outcome'] == 'Away Win').sum()

new_home = (new['predicted_outcome'] == 'Home Win').sum()
new_draw = (new['predicted_outcome'] == 'Draw').sum()
new_away = (new['predicted_outcome'] == 'Away Win').sum()

print(f"\nOLD MODEL (before fixes):")
print(f"  Home: {old_home}, Draw: {old_draw}, Away: {old_away}")

print(f"\nNEW MODEL (after fixes):")
print(f"  Home: {new_home}, Draw: {new_draw}, Away: {new_away}")

print("\n2. PROBABILITY COMPARISON - SELECTED MATCHES")
print("-" * 80)

matches_to_compare = [
    'Feyenoord',
    'Paris',
    'Gent'
]

for match_name in matches_to_compare:
    old_match = old[old['home_team'].str.contains(match_name, na=False) | 
                    old['away_team'].str.contains(match_name, na=False)]
    new_match = new[new['home_team'].str.contains(match_name, na=False) | 
                    new['away_team'].str.contains(match_name, na=False)]
    
    if not old_match.empty and not new_match.empty:
        old_row = old_match.iloc[0]
        new_row = new_match.iloc[0]
        
        print(f"\n{old_row['home_team']} vs {old_row['away_team']}:")
        print(f"  OLD: H:{old_row['home_win_prob']:.1%} D:{old_row['draw_prob']:.1%} A:{old_row['away_win_prob']:.1%}")
        print(f"  NEW: H:{new_row['home_win_prob']:.1%} D:{new_row['draw_prob']:.1%} A:{new_row['away_win_prob']:.1%}")
        
        # Check which has more balanced probabilities
        old_max_margin = max(old_row['home_win_prob'], old_row['draw_prob'], old_row['away_win_prob']) - \
                        min(old_row['home_win_prob'], old_row['draw_prob'], old_row['away_win_prob'])
        new_max_margin = max(new_row['home_win_prob'], new_row['draw_prob'], new_row['away_win_prob']) - \
                        min(new_row['home_win_prob'], new_row['draw_prob'], new_row['away_win_prob'])
        
        if new_max_margin < old_max_margin:
            print(f"  ✓ More balanced (margin reduced from {old_max_margin:.1%} to {new_max_margin:.1%})")
        else:
            print(f"  ✗ Less balanced (margin increased from {old_max_margin:.1%} to {new_max_margin:.1%})")

print("\n3. AVERAGE PROBABILITIES")
print("-" * 80)

print(f"\nOLD MODEL averages:")
print(f"  Home: {old['home_win_prob'].mean():.1%}")
print(f"  Draw: {old['draw_prob'].mean():.1%}")
print(f"  Away: {old['away_win_prob'].mean():.1%}")

print(f"\nNEW MODEL averages:")
print(f"  Home: {new['home_win_prob'].mean():.1%}")
print(f"  Draw: {new['draw_prob'].mean():.1%}")
print(f"  Away: {new['away_win_prob'].mean():.1%}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if new_home < old_home:
    print("\n✓ Improvement: Reduced home win bias")
else:
    print("\n✗ No improvement: Still predicting all home wins")

if new['away_win_prob'].mean() > old['away_win_prob'].mean():
    print("✓ Improvement: Higher average away win probability")
else:
    print("✗ No improvement: Away probabilities didn't increase")

