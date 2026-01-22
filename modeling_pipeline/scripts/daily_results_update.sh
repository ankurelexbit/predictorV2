#!/bin/bash
# Daily Results Update Script
# Run this daily to update predictions with actual results

set -e

echo "================================================================================"
echo "DAILY RESULTS UPDATE"
echo "================================================================================"
echo "Time: $(date)"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Create log file
LOG_FILE="logs/results_update_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Log file: $LOG_FILE"
echo ""

# Update results for last 7 days
echo "Updating predictions with actual results..." | tee -a "$LOG_FILE"
venv/bin/python update_prediction_results.py --days 7 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Results update failed!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "✅ Results update complete" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
