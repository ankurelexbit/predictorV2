"""
Analyze Raw JSON Files to Check Statistics Availability.

This script checks if statistics are actually missing in the raw JSON
or if there's a conversion bug causing missing data.
"""
import sys
from pathlib import Path
import ijson
import json
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_fixture_statistics(fixture: dict) -> dict:
    """Analyze what statistics are available in a fixture."""
    analysis = {
        'fixture_id': fixture.get('id'),
        'has_statistics': False,
        'statistics_count': 0,
        'stat_types_found': set(),
        'has_shots': False,
        'has_possession': False,
        'has_tackles': False,
        'has_attacks': False,
        'has_corners': False,
        'statistics_structure': None,
    }

    # Check if statistics exist
    statistics = fixture.get('statistics', [])

    if statistics and len(statistics) > 0:
        analysis['has_statistics'] = True
        analysis['statistics_count'] = len(statistics)

        # Analyze what types of stats we have
        for stat in statistics:
            type_id = stat.get('type_id')
            if type_id:
                analysis['stat_types_found'].add(type_id)

                # Check for specific stats
                if type_id in [52, 53, 54, 55, 56, 57]:  # Shot types
                    analysis['has_shots'] = True
                elif type_id == 82:  # Possession
                    analysis['has_possession'] = True
                elif type_id == 78:  # Tackles
                    analysis['has_tackles'] = True
                elif type_id in [44, 45]:  # Attacks/dangerous attacks
                    analysis['has_attacks'] = True
                elif type_id == 80:  # Corners
                    analysis['has_corners'] = True

        # Store a sample for structure inspection
        if len(statistics) > 0:
            analysis['statistics_structure'] = statistics[0]

    return analysis

def main():
    print("=" * 80)
    print("ANALYZING RAW JSON STATISTICS AVAILABILITY")
    print("=" * 80)
    print()

    fixtures_dir = Path('data/historical/fixtures')
    fixture_files = sorted(fixtures_dir.glob('*.json'))

    print(f"Found {len(fixture_files)} JSON files")
    print()

    # Sample files to check (check a few from different time periods)
    sample_size = min(10, len(fixture_files))
    sample_indices = [0, len(fixture_files)//4, len(fixture_files)//2,
                     3*len(fixture_files)//4, len(fixture_files)-1]
    sample_files = [fixture_files[i] for i in sample_indices if i < len(fixture_files)]

    print(f"Sampling {len(sample_files)} files from different time periods:")
    for f in sample_files:
        print(f"  - {f.name}")
    print()

    # Statistics
    total_fixtures = 0
    fixtures_with_stats = 0
    fixtures_without_stats = 0

    stat_type_counts = defaultdict(int)

    fixtures_by_availability = {
        'with_stats': [],
        'without_stats': [],
    }

    print("Processing files...")
    print()

    for file_path in sample_files:
        print(f"Processing {file_path.name}...")

        try:
            with open(file_path, 'rb') as f:
                parser = ijson.items(f, 'item')

                for fixture in parser:
                    total_fixtures += 1
                    analysis = analyze_fixture_statistics(fixture)

                    if analysis['has_statistics']:
                        fixtures_with_stats += 1
                        fixtures_by_availability['with_stats'].append(analysis)

                        # Count stat types
                        for stat_type in analysis['stat_types_found']:
                            stat_type_counts[stat_type] += 1
                    else:
                        fixtures_without_stats += 1
                        fixtures_by_availability['without_stats'].append(analysis)

        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
            continue

        print(f"  ✓ Processed")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Overall statistics
    print(f"Total fixtures analyzed: {total_fixtures:,}")
    print(f"Fixtures WITH statistics: {fixtures_with_stats:,} ({fixtures_with_stats/total_fixtures*100:.1f}%)")
    print(f"Fixtures WITHOUT statistics: {fixtures_without_stats:,} ({fixtures_without_stats/total_fixtures*100:.1f}%)")
    print()

    if fixtures_without_stats > 0:
        pct = fixtures_without_stats / total_fixtures * 100
        print(f"🔴 {pct:.1f}% of fixtures have NO statistics in raw JSON!")
        print("This explains the missing data in your training features.")
    else:
        print("✅ All fixtures have statistics in raw JSON")
        print("🔴 The missing data is a conversion/processing bug!")

    print()

    # Detailed breakdown
    if fixtures_with_stats > 0:
        print("STATISTICS BREAKDOWN (for fixtures that have stats):")
        print()

        # Check specific stat types
        fixtures_with_stat_types = {
            'shots': sum(1 for a in fixtures_by_availability['with_stats'] if a['has_shots']),
            'possession': sum(1 for a in fixtures_by_availability['with_stats'] if a['has_possession']),
            'tackles': sum(1 for a in fixtures_by_availability['with_stats'] if a['has_tackles']),
            'attacks': sum(1 for a in fixtures_by_availability['with_stats'] if a['has_attacks']),
            'corners': sum(1 for a in fixtures_by_availability['with_stats'] if a['has_corners']),
        }

        for stat_name, count in fixtures_with_stat_types.items():
            pct = count / fixtures_with_stats * 100
            print(f"  {stat_name.capitalize()}: {count}/{fixtures_with_stats} ({pct:.1f}%)")

        print()
        print("Most common stat type IDs:")
        for stat_type, count in sorted(stat_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = count / fixtures_with_stats * 100
            print(f"  Type {stat_type}: {count} occurrences ({pct:.1f}%)")

    print()

    # Show example structure
    if fixtures_by_availability['with_stats']:
        print("EXAMPLE STATISTICS STRUCTURE:")
        print()
        example = fixtures_by_availability['with_stats'][0]
        print(f"Fixture ID: {example['fixture_id']}")
        print(f"Number of stats: {example['statistics_count']}")
        print(f"Stat types found: {sorted(example['stat_types_found'])}")
        print()
        if example['statistics_structure']:
            print("Sample statistic:")
            print(json.dumps(example['statistics_structure'], indent=2))

    if fixtures_by_availability['without_stats']:
        print()
        print("EXAMPLE FIXTURE WITHOUT STATISTICS:")
        print()
        example_no_stats = fixtures_by_availability['without_stats'][0]
        print(f"Fixture ID: {example_no_stats['fixture_id']}")
        print("This fixture has no 'statistics' array or it's empty")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    if fixtures_without_stats / total_fixtures > 0.5:
        print("🔴 ISSUE: Over 50% of fixtures missing statistics in RAW JSON")
        print()
        print("Possible reasons:")
        print("  1. SportMonks doesn't have statistics for these matches")
        print("  2. API download didn't include statistics (missing include parameter)")
        print("  3. These are old matches where stats weren't recorded")
        print()
        print("Solutions:")
        print("  1. Check backfill_historical_data.py for correct API includes")
        print("  2. Re-download with statistics included")
        print("  3. Focus on more recent matches that have stats")
        print("  4. Remove or impute features that depend on missing stats")
    elif fixtures_without_stats / total_fixtures > 0.2:
        print("⚠️  MODERATE ISSUE: 20-50% of fixtures missing statistics")
        print()
        print("This is expected for:")
        print("  - Very old matches (pre-2015)")
        print("  - Lower league matches")
        print("  - Cup/friendly matches")
        print()
        print("Solutions:")
        print("  1. Filter to leagues/dates with good statistics coverage")
        print("  2. Impute missing statistics features")
    else:
        print("✅ GOOD: Most fixtures have statistics in raw JSON")
        print()
        print("🔴 The missing data in training is a BUG in conversion/feature generation!")
        print()
        print("Next steps:")
        print("  1. Check convert_json_to_csv.py extraction logic")
        print("  2. Check pillar2_modern_analytics.py feature calculation")
        print("  3. Verify statistics are being properly extracted")

    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
