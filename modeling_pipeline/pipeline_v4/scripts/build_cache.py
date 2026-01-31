"""
Build fixture cache for fast loading.

Run this once to create a pickle cache of all fixtures.
Subsequent loads will be instant.
"""
import sys
from pathlib import Path
import logging
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.json_loader import JSONDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build fixture cache."""
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING FIXTURE CACHE")
    logger.info("=" * 80 + "\n")
    
    # Initialize loader
    loader = JSONDataLoader(data_dir='data/historical')
    
    # Load all fixtures (this will take ~1 minute)
    logger.info("Loading all fixtures from JSON files...")
    logger.info("This will take about 1 minute (one-time operation)...\n")
    
    start = time.time()
    df = loader.load_all_fixtures(cache_full_data=True)
    elapsed = time.time() - start
    
    logger.info(f"\n✅ Loaded {len(df)} fixtures in {elapsed:.1f} seconds")
    logger.info(f"   Date range: {df['starting_at'].min()} to {df['starting_at'].max()}")
    logger.info(f"   Cached full data for {len(loader._fixtures_dict)} fixtures")
    
    # Save to pickle
    cache_dir = Path('data/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    fixtures_cache_file = cache_dir / 'fixtures_df.pkl'
    fixtures_dict_cache_file = cache_dir / 'fixtures_dict.pkl'
    
    logger.info(f"\n💾 Saving cache files...")
    df.to_pickle(fixtures_cache_file)
    logger.info(f"   ✓ Saved fixtures DataFrame to {fixtures_cache_file}")
    
    import pickle
    with open(fixtures_dict_cache_file, 'wb') as f:
        pickle.dump(loader._fixtures_dict, f)
    logger.info(f"   ✓ Saved fixtures dict to {fixtures_dict_cache_file}")
    
    # Show file sizes
    df_size = fixtures_cache_file.stat().st_size / (1024 * 1024)
    dict_size = fixtures_dict_cache_file.stat().st_size / (1024 * 1024)
    
    logger.info(f"\n📊 Cache file sizes:")
    logger.info(f"   fixtures_df.pkl: {df_size:.1f} MB")
    logger.info(f"   fixtures_dict.pkl: {dict_size:.1f} MB")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ CACHE BUILT SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nNext time you run feature generation, it will load instantly!")
    logger.info("\n")


if __name__ == '__main__':
    main()
