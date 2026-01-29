import pandas as pd
from pathlib import Path

def main():
    # Path adjustment to find modeling_pipeline/data/processed
    # Current: pipeline_v3/scripts/detailed_feature_diff.py
    # Root: modeling_pipeline/
    root_dir = Path(__file__).parent.parent.parent 
    parent_path = root_dir / 'data/processed/sportmonks_features.csv'
    v3_path = Path('data/csv/training_data_complete_v2.csv')
    output_path = Path('/Users/ankurgupta/.gemini/antigravity/brain/b17befe7-0b46-48c7-8e29-6cb1b85b637c/feature_diff_analysis.md')
    
    # Load headers
    parent_cols = set(pd.read_csv(parent_path, nrows=0).columns)
    v3_cols = set(pd.read_csv(v3_path, nrows=0).columns)
    
    # Exclude targets/IDs
    exclude = {'date', 'target', 'fixture_id', 'home_team_id', 'away_team_id', 'league_id', 'season_id', 'home_score', 'away_score', 'result', 'target_home_win', 'target_draw', 'target_away_win', 'home_goals', 'away_goals', 'starting_at', 'match_date', 'is_derby_match'}
    
    parent_feats = sorted([c for c in parent_cols if c not in exclude])
    v3_feats = sorted([c for c in v3_cols if c not in exclude])
    
    # Diff
    common = sorted(list(set(parent_feats) & set(v3_feats)))
    parent_only = sorted(list(set(parent_feats) - set(v3_feats)))
    v3_only = sorted(list(set(v3_feats) - set(parent_feats)))
    
    # Write Report
    with open(output_path, 'w') as f:
        f.write("# Feature Set Diff Analysis\n\n")
        f.write(f"## Summary\n")
        f.write(f"- **Parent Features**: {len(parent_feats)}\n")
        f.write(f"- **V3 Features**: {len(v3_feats)}\n")
        f.write(f"- **Common**: {len(common)}\n")
        f.write(f"- **Parent Only (Dropped)**: {len(parent_only)}\n")
        f.write(f"- **V3 Only (New)**: {len(v3_only)}\n\n")
        
        f.write("## 1. Parent Features (Dropped in V3)\n")
        f.write("These features were used in the parent model but are missing in V3 (likely due to 'No Calculation' rule).\n\n")
        for ft in parent_only:
            f.write(f"- `{ft}`\n")
            
        f.write("\n## 2. V3 Only Features (New Premium Stats)\n")
        f.write("These are the new/enhanced features added in V3.\n\n")
        for ft in v3_only:
            f.write(f"- `{ft}`\n")
            
        f.write("\n## 3. Common Features\n")
        for ft in common:
            f.write(f"- `{ft}`\n")
            
    print(f"Report generated at {output_path}")
    print(f"Parent Only: {len(parent_only)}")
    print(f"V3 Only: {len(v3_only)}")

if __name__ == "__main__":
    main()
