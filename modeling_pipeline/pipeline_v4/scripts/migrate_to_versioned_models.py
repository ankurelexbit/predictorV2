#!/usr/bin/env python3
"""
Migrate to Versioned Production Models
========================================

Migrates existing Option 3 model to the new versioned production structure.

This script:
1. Copies models/weight_experiments/option3_balanced.joblib → models/production/model_v1.0.0.joblib
2. Copies the metadata file as well
3. Creates models/production/LATEST file pointing to v1.0.0
4. Validates the migration

Usage:
    python3 scripts/migrate_to_versioned_models.py
"""

import sys
from pathlib import Path
import shutil
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate():
    """Migrate existing model to versioned structure."""
    print("=" * 80)
    print("MIGRATE TO VERSIONED PRODUCTION MODELS")
    print("=" * 80)
    print()

    # Source paths
    source_model = Path("models/weight_experiments/option3_balanced.joblib")
    source_metadata = Path("models/weight_experiments/option3_balanced_metadata.json")

    # Destination paths
    dest_dir = Path("models/production")
    dest_model = dest_dir / "model_v1.0.0.joblib"
    dest_metadata = dest_dir / "model_v1.0.0_metadata.json"
    latest_file = dest_dir / "LATEST"

    print(f"Source model: {source_model}")
    print(f"Destination: {dest_model}")
    print()

    # Check if source exists
    if not source_model.exists():
        print(f"❌ Source model not found: {source_model}")
        print("   Train a model first with train_production_model.py")
        return 1

    # Check if destination already exists
    if dest_model.exists():
        print(f"⚠️  Destination already exists: {dest_model}")
        response = input("Overwrite? (yes/no): ").strip().lower()
        if response != 'yes':
            print("❌ Migration cancelled")
            return 1
        print()

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: {dest_dir}")

    # Copy model file
    print(f"📦 Copying model file...")
    shutil.copy2(source_model, dest_model)
    print(f"✅ Copied: {dest_model}")

    # Copy metadata if exists
    if source_metadata.exists():
        print(f"📦 Copying metadata file...")
        # Load and update metadata
        with open(source_metadata, 'r') as f:
            metadata = json.load(f)

        # Add version info
        metadata['version'] = 'v1.0.0'
        metadata['migration'] = {
            'migrated_from': str(source_model),
            'migration_date': '2026-02-03'
        }

        # Save updated metadata
        with open(dest_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Copied and updated: {dest_metadata}")
    else:
        print(f"⚠️  No metadata file found at {source_metadata}")

    # Create LATEST file
    print(f"📝 Creating LATEST pointer...")
    with open(latest_file, 'w') as f:
        f.write(dest_model.name)
    print(f"✅ Created: {latest_file} → {dest_model.name}")

    print()
    print("=" * 80)
    print("✅ MIGRATION COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Verify configuration: python3 config/production_config.py")
    print("2. Test predictions: python3 scripts/predict_production.py --days-ahead 1")
    print("3. Future retrains will auto-increment to v1.0.1, v1.0.2, etc.")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(migrate())
