#!/bin/bash
# Setup Cron Jobs for Production Pipeline
# Run this once to set up automated tasks

echo "Setting up production pipeline cron jobs..."

# Get the absolute path to the project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data/predictions"

# Create cron entries
CRON_FILE="/tmp/football_prediction_cron.txt"

cat > "$CRON_FILE" << EOF
# Football Prediction Pipeline - Automated Tasks
# Generated on $(date)

# Live predictions every 30 minutes
*/30 * * * * cd $PROJECT_DIR && venv/bin/python run_live_predictions.py >> logs/live_predictions.log 2>&1

# Weekly model retraining (Sunday 2 AM)
0 2 * * 0 cd $PROJECT_DIR && bash scripts/weekly_model_retraining.sh >> logs/weekly_training.log 2>&1

# Daily performance summary (Every day at 11 PM)
0 23 * * * cd $PROJECT_DIR && venv/bin/python scripts/daily_summary.py >> logs/daily_summary.log 2>&1

EOF

echo "Cron entries created in $CRON_FILE"
echo ""
echo "To install, run:"
echo "  crontab $CRON_FILE"
echo ""
echo "To view current crontab:"
echo "  crontab -l"
echo ""
echo "To edit crontab manually:"
echo "  crontab -e"
echo ""

# Make scripts executable
chmod +x "$PROJECT_DIR/run_live_predictions.py"
chmod +x "$PROJECT_DIR/scripts/"*.sh 2>/dev/null || true

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review the cron entries in $CRON_FILE"
echo "2. Install with: crontab $CRON_FILE"
echo "3. Test manually: venv/bin/python run_live_predictions.py"
