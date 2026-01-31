"""
Weekly Retrain Pipeline - Production Script for Regular Model Updates.

This script should be run weekly (via cron or scheduler) to:
1. Download latest fixture data
2. Convert JSON to CSV
3. Generate training data
4. Train model
5. Evaluate and save

Usage:
    python3 scripts/weekly_retrain_pipeline.py --weeks-back 4

Cron example (run every Sunday at 2am):
    0 2 * * 0 cd /path/to/pipeline_v4 && python3 scripts/weekly_retrain_pipeline.py --weeks-back 4
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/weekly_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WeeklyRetrainPipeline:
    """Weekly retraining pipeline."""

    def __init__(self, weeks_back: int = 4, league_id: int = None):
        """
        Initialize pipeline.

        Args:
            weeks_back: Number of weeks of new data to download
            league_id: Optional league filter
        """
        self.weeks_back = weeks_back
        self.league_id = league_id
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def run_command(self, cmd: str, description: str) -> bool:
        """Run shell command and log results."""
        logger.info("=" * 80)
        logger.info(description)
        logger.info("=" * 80)
        logger.info(f"Command: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"FAILED: {description}")
                logger.error(f"Error: {result.stderr}")
                return False

            logger.info(f"SUCCESS: {description}")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"TIMEOUT: {description}")
            return False
        except Exception as e:
            logger.error(f"ERROR: {description} - {e}")
            return False

    def step1_download_new_data(self) -> bool:
        """Download latest fixture data."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DOWNLOADING NEW DATA")
        logger.info("=" * 80)

        # Calculate date range (last N weeks)
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.weeks_back)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        logger.info(f"Downloading fixtures from {start_str} to {end_str}")

        cmd = f"""python3 scripts/backfill_historical_data.py \
            --start-date {start_str} \
            --end-date {end_str} \
            --output-dir data/historical"""

        if self.league_id:
            cmd += f" --league-id {self.league_id}"

        return self.run_command(cmd, "Download new fixture data")

    def step2_convert_to_csv(self) -> bool:
        """Convert JSON to CSV (incremental update)."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: CONVERTING JSON TO CSV")
        logger.info("=" * 80)

        # For production, you might want to append new data instead of regenerating all
        # For now, regenerate the full CSV (includes old + new data)
        cmd = "python3 scripts/convert_json_to_csv.py"

        return self.run_command(cmd, "Convert JSON to CSV")

    def step3_generate_training_data(self) -> bool:
        """Generate training dataset."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: GENERATING TRAINING DATA")
        logger.info("=" * 80)

        output_file = f"data/training_data_{self.timestamp}.csv"

        cmd = f"python3 scripts/generate_training_data.py --output {output_file}"

        if self.league_id:
            cmd += f" --league-id {self.league_id}"

        success = self.run_command(cmd, "Generate training data")

        if success:
            # Create symlink to latest
            latest_link = Path('data/training_data_latest.csv')
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(Path(output_file).name)
            logger.info(f"Created symlink: {latest_link} -> {output_file}")

        return success

    def step4_train_model(self) -> bool:
        """Train model on latest data."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: TRAINING MODEL")
        logger.info("=" * 80)

        training_file = "data/training_data_latest.csv"
        output_model = f"models/v4_model_{self.timestamp}.joblib"

        cmd = f"""python3 scripts/train_improved_model.py \
            --data {training_file} \
            --output {output_model} \
            --model stacking"""

        success = self.run_command(cmd, "Train model")

        if success:
            # Create symlink to latest model
            latest_link = Path('models/v4_model_latest.joblib')
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(Path(output_model).name)
            logger.info(f"Created symlink: {latest_link} -> {output_model}")

        return success

    def step5_cleanup_old_files(self, keep_last_n: int = 3):
        """Clean up old training data and model files."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: CLEANUP OLD FILES")
        logger.info("=" * 80)

        # Clean old training data
        data_dir = Path('data')
        training_files = sorted(data_dir.glob('training_data_*.csv'), reverse=True)

        if len(training_files) > keep_last_n:
            for old_file in training_files[keep_last_n:]:
                if 'latest' not in old_file.name:
                    logger.info(f"Deleting old training data: {old_file}")
                    old_file.unlink()

        # Clean old models
        models_dir = Path('models')
        model_files = sorted(models_dir.glob('v4_model_*.joblib'), reverse=True)

        if len(model_files) > keep_last_n:
            for old_model in model_files[keep_last_n:]:
                if 'latest' not in old_model.name:
                    logger.info(f"Deleting old model: {old_model}")
                    old_model.unlink()

        logger.info("Cleanup complete")

    def run(self):
        """Run complete pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("WEEKLY RETRAIN PIPELINE STARTED")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"Weeks back: {self.weeks_back}")
        logger.info(f"League ID: {self.league_id or 'All leagues'}")

        steps = [
            ("Download new data", self.step1_download_new_data),
            ("Convert to CSV", self.step2_convert_to_csv),
            ("Generate training data", self.step3_generate_training_data),
            ("Train model", self.step4_train_model),
        ]

        for step_name, step_func in steps:
            logger.info(f"\nExecuting: {step_name}")

            success = step_func()

            if not success:
                logger.error(f"Pipeline FAILED at: {step_name}")
                logger.error("Aborting pipeline")
                return False

        # Cleanup (optional, doesn't stop pipeline if it fails)
        try:
            self.step5_cleanup_old_files()
        except Exception as e:
            logger.warning(f"Cleanup failed (non-critical): {e}")

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Latest model: models/v4_model_latest.joblib")
        logger.info(f"Latest training data: data/training_data_latest.csv")

        return True


def main():
    parser = argparse.ArgumentParser(description='Weekly retrain pipeline')
    parser.add_argument('--weeks-back', type=int, default=4,
                       help='Number of weeks of data to download (default: 4)')
    parser.add_argument('--league-id', type=int, default=None,
                       help='Optional league ID filter')
    parser.add_argument('--dry-run', action='store_true',
                       help='Log what would be done without executing')

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)

    if args.dry_run:
        logger.info("DRY RUN MODE - Commands will be logged but not executed")

    pipeline = WeeklyRetrainPipeline(
        weeks_back=args.weeks_back,
        league_id=args.league_id
    )

    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
