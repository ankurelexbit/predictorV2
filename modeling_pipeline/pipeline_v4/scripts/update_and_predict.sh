#!/bin/bash
"""
Update Historical Data & Make Live Predictions
===============================================

This script ensures historical data is up-to-date before making predictions.

Usage:
    bash scripts/update_and_predict.sh --date today
    bash scripts/update_and_predict.sh --date 2026-02-15 --league-id 8
"""

set -e  # Exit on error

# Default parameters
DATE="today"
LEAGUE_ID=""
OUTPUT=""
DAYS_BACK=30  # Download last 30 days of historical data

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --date)
            DATE="$2"
            shift 2
            ;;
        --league-id)
            LEAGUE_ID="--league-id $2"
            shift 2
            ;;
        --output)
            OUTPUT="--output $2"
            shift 2
            ;;
        --days-back)
            DAYS_BACK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "UPDATE & PREDICT PIPELINE"
echo "================================================================================"
echo "Step 1: Updating historical data (last $DAYS_BACK days)..."
echo "Step 2: Making predictions for $DATE"
echo "================================================================================"

# Calculate date range for historical update
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    START_DATE=$(date -v-${DAYS_BACK}d +%Y-%m-%d)
    END_DATE=$(date +%Y-%m-%d)
else
    # Linux
    START_DATE=$(date -d "$DAYS_BACK days ago" +%Y-%m-%d)
    END_DATE=$(date +%Y-%m-%d)
fi

echo ""
echo "📥 Downloading recent historical data ($START_DATE to $END_DATE)..."
python3 scripts/backfill_historical_data.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --output-dir data/historical

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to download historical data"
    exit 1
fi

echo ""
echo "📊 Making live predictions..."
python3 scripts/predict_live_v4.py \
    --date "$DATE" \
    $LEAGUE_ID \
    $OUTPUT

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to make predictions"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ UPDATE & PREDICT COMPLETE"
echo "================================================================================"
