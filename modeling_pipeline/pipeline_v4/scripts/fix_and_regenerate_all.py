"""
Fix and Regenerate All Data with Correct Type Mappings.

This script:
1. Deletes old CSV with wrong statistics
2. Regenerates CSV with correct type_id mapping
3. Regenerates training data
4. Verifies the fix worked
"""
import sys
from pathlib import Path
import pandas as pd
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

def run_command(cmd, description):
    """Run a shell command and show progress."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}")
    print(f"Running: {cmd}")
    print()

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False
    else:
        print(f"\n✅ SUCCESS: {description}")
        return True

def verify_csv_data():
    """Verify CSV has sensible statistics."""
    print(f"\n{'=' * 80}")
    print("VERIFYING CSV DATA QUALITY")
    print(f"{'=' * 80}\n")

    csv_file = Path('data/processed/fixtures_with_stats.csv')

    if not csv_file.exists():
        print("❌ CSV file not found!")
        return False

    df = pd.read_csv(csv_file)

    # Check sample values
    print("Sample match statistics:")
    sample_idx = 100

    shots_total = df['home_shots_total'].iloc[sample_idx]
    shots_on_target = df['home_shots_on_target'].iloc[sample_idx]
    possession = df['home_ball_possession'].iloc[sample_idx]
    corners = df['home_corners'].iloc[sample_idx]
    passes = df['home_passes_total'].iloc[sample_idx]

    print(f"  Shots total: {shots_total}")
    print(f"  Shots on target: {shots_on_target}")
    print(f"  Possession: {possession}%")
    print(f"  Corners: {corners}")
    print(f"  Passes: {passes}")
    print()

    # Validate values are sensible
    issues = []

    if pd.notna(shots_total) and pd.notna(shots_on_target):
        if shots_on_target > shots_total:
            issues.append("❌ Shots on target > total shots")

    if pd.notna(possession) and (possession < 0 or possession > 100):
        issues.append(f"❌ Possession out of range: {possession}%")

    if pd.notna(corners) and corners > 20:
        issues.append(f"❌ Corners too high: {corners}")

    if issues:
        print("Data quality issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ All values look sensible!")
        return True

def verify_training_data():
    """Verify training data has improved coverage."""
    print(f"\n{'=' * 80}")
    print("VERIFYING TRAINING DATA COVERAGE")
    print(f"{'=' * 80}\n")

    train_file = Path('data/training_data_fixed.csv')

    if not train_file.exists():
        print("❌ Training data file not found!")
        return False

    df = pd.read_csv(train_file)

    # Remove duplicates for analysis
    df = df.drop_duplicates(subset=['fixture_id'], keep='first')

    print(f"Total unique fixtures: {len(df):,}")
    print()

    # Check key features
    features_to_check = [
        ('home_derived_xg_per_match_5', 'Derived xG'),
        ('home_shots_per_match_5', 'Shots per match'),
        ('home_shots_on_target_per_match_5', 'Shots on target'),
        ('home_possession_pct_5', 'Possession'),
    ]

    print("Feature coverage:")
    all_good = True

    for feat, desc in features_to_check:
        if feat in df.columns:
            coverage = df[feat].notna().sum() / len(df) * 100
            status = '✅' if coverage > 80 else '⚠️' if coverage > 50 else '❌'
            print(f"  {status} {desc:30s}: {coverage:5.1f}%")

            if coverage < 80:
                all_good = False
        else:
            print(f"  ❌ {desc:30s}: NOT FOUND")
            all_good = False

    return all_good

def main():
    print("\n" + "=" * 80)
    print("FIXING AND REGENERATING ALL DATA")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Delete old CSV with wrong statistics")
    print("  2. Regenerate CSV with CORRECT type_id mappings")
    print("  3. Regenerate training data")
    print("  4. Verify everything is fixed")
    print()

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Step 1: Delete old CSV
    print("\n" + "=" * 80)
    print("STEP 1: Removing old CSV with wrong statistics")
    print("=" * 80)

    csv_file = Path('data/processed/fixtures_with_stats.csv')
    if csv_file.exists():
        csv_file.unlink()
        print(f"✅ Deleted {csv_file}")
    else:
        print("ℹ️  CSV file doesn't exist, nothing to delete")

    # Step 2: Regenerate CSV
    success = run_command(
        "python3 scripts/convert_json_to_csv.py",
        "STEP 2: Regenerating CSV with correct type_id mappings"
    )

    if not success:
        print("\n❌ CSV regeneration failed. Stopping.")
        return

    # Step 3: Verify CSV
    if not verify_csv_data():
        print("\n⚠️  CSV data quality check failed!")
        print("Values still look wrong. Check the type_id mapping.")
        return

    # Step 4: Regenerate training data
    success = run_command(
        "python3 scripts/generate_training_data.py --output data/training_data_fixed.csv",
        "STEP 3: Regenerating training data"
    )

    if not success:
        print("\n❌ Training data regeneration failed. Stopping.")
        return

    # Step 5: Verify training data
    if not verify_training_data():
        print("\n⚠️  Training data coverage still low!")
        print("Some features may still have issues.")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ CSV regenerated with correct type_id mappings")
    print("✅ Training data regenerated")
    print()
    print("Files created:")
    print("  - data/processed/fixtures_with_stats.csv (corrected)")
    print("  - data/training_data_fixed.csv (with better coverage)")
    print()
    print("Next steps:")
    print("  1. Compare old vs new training data:")
    print("     python3 -c \"")
    print("     import pandas as pd")
    print("     old = pd.read_csv('data/training_data.csv').drop_duplicates(subset=['fixture_id'])")
    print("     new = pd.read_csv('data/training_data_fixed.csv').drop_duplicates(subset=['fixture_id'])")
    print("     print('Old xG coverage:', old['home_derived_xg_per_match_5'].notna().sum() / len(old) * 100)")
    print("     print('New xG coverage:', new['home_derived_xg_per_match_5'].notna().sum() / len(new) * 100)")
    print("     \"")
    print()
    print("  2. Train model with fixed data:")
    print("     python3 scripts/train_improved_model.py --data data/training_data_fixed.csv")
    print()

if __name__ == '__main__':
    main()
