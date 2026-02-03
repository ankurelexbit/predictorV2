#!/bin/bash
# Cleanup Script for Pipeline V4
# Removes temporary and experimental files
# Run with: bash cleanup.sh

set -e  # Exit on error

echo "======================================================================================================"
echo "PIPELINE V4 CLEANUP SCRIPT"
echo "======================================================================================================"
echo ""
echo "This script will delete temporary files from experiments."
echo "See CLEANUP_GUIDE.md for details on what will be deleted."
echo ""
read -p "Continue with cleanup? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]
then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo "Starting cleanup..."
echo ""

# Track space saved
INITIAL_SIZE=$(du -sh . | cut -f1)

# 1. Delete experimental models
echo "🗑️  Deleting experimental models..."
rm -rf models/calibrated/ 2>/dev/null && echo "   ✅ Deleted models/calibrated/" || echo "   ⏭️  models/calibrated/ not found"
rm -rf models/unbiased/ 2>/dev/null && echo "   ✅ Deleted models/unbiased/" || echo "   ⏭️  models/unbiased/ not found"
rm -rf models/final/ 2>/dev/null && echo "   ✅ Deleted models/final/" || echo "   ⏭️  models/final/ not found"
rm -rf models/moderate_weights/ 2>/dev/null && echo "   ✅ Deleted models/moderate_weights/" || echo "   ⏭️  models/moderate_weights/ not found"
rm -rf models/with_draw_features/ 2>/dev/null && echo "   ✅ Deleted models/with_draw_features/" || echo "   ⏭️  models/with_draw_features/ not found"
rm -f models/v4_*.joblib models/v4_*.json 2>/dev/null && echo "   ✅ Deleted root-level old models" || echo "   ⏭️  No root-level old models"

# 2. Delete old production model (superseded)
echo ""
echo "🗑️  Deleting old production model..."
rm -rf models/production/ 2>/dev/null && echo "   ✅ Deleted models/production/" || echo "   ⏭️  models/production/ not found"

# 3. Delete log files
echo ""
echo "🗑️  Deleting log files..."
rm -f logs/*.log 2>/dev/null && echo "   ✅ Deleted all log files" || echo "   ⏭️  No log files found"

# 4. Delete old result files
echo ""
echo "🗑️  Deleting old result files..."
rm -f results/backtest*.csv results/backtest*.log 2>/dev/null && echo "   ✅ Deleted backtest files"
rm -f results/calibrated_ev*.txt 2>/dev/null && echo "   ✅ Deleted calibration test results"
rm -f results/COMPREHENSIVE_MODEL_REPORT.md 2>/dev/null && echo "   ✅ Deleted old comprehensive report"
rm -f results/model_comparison.csv results/model_comparison.json 2>/dev/null && echo "   ✅ Deleted old comparison files"
rm -f results/logloss_optimization_full.csv results/threshold_optimization.csv 2>/dev/null && echo "   ✅ Deleted old optimization files"

# 5. Delete old documentation
echo ""
echo "🗑️  Deleting superseded documentation..."
rm -f CLASS_WEIGHT_EXPERIMENT.md 2>/dev/null && echo "   ✅ Deleted CLASS_WEIGHT_EXPERIMENT.md"
rm -f FEATURE_VALIDATION_REPORT.md 2>/dev/null && echo "   ✅ Deleted FEATURE_VALIDATION_REPORT.md"
rm -f HOME_PREDICTION_IMPROVEMENT_PLAN.md 2>/dev/null && echo "   ✅ Deleted HOME_PREDICTION_IMPROVEMENT_PLAN.md"
rm -f LIVE_PREDICTION_GUIDE.md 2>/dev/null && echo "   ✅ Deleted LIVE_PREDICTION_GUIDE.md"
rm -f MODEL_COMPARISON_FINAL_REPORT.md 2>/dev/null && echo "   ✅ Deleted MODEL_COMPARISON_FINAL_REPORT.md"
rm -f MODEL_IMPROVEMENT_PLAN.md 2>/dev/null && echo "   ✅ Deleted MODEL_IMPROVEMENT_PLAN.md"
rm -f PRODUCTION_FILES.md 2>/dev/null && echo "   ✅ Deleted PRODUCTION_FILES.md"
rm -f PRODUCTION_GUIDE.md 2>/dev/null && echo "   ✅ Deleted PRODUCTION_GUIDE.md"
rm -f QUICK_START_PNL.md 2>/dev/null && echo "   ✅ Deleted QUICK_START_PNL.md"

# 6. Delete old/duplicate scripts (optional - uncomment if desired)
# echo ""
# echo "🗑️  Deleting old/duplicate scripts..."
# rm -f scripts/analyze_thresholds_no_odds.py
# rm -f scripts/analyze_thresholds.py
# rm -f scripts/backtest_january_2026.py
# rm -f scripts/backtest_multioutcome_january_2026.py
# ... (add more as needed)

echo ""
echo "======================================================================================================"
echo "✅ CLEANUP COMPLETE!"
echo "======================================================================================================"
echo ""

FINAL_SIZE=$(du -sh . | cut -f1)
echo "Initial size: $INITIAL_SIZE"
echo "Final size:   $FINAL_SIZE"
echo ""
echo "Remaining essential files:"
echo "  ✅ src/ - All source code"
echo "  ✅ config/production_config.py - Production configuration"
echo "  ✅ models/weight_experiments/option3_balanced.joblib - Production model"
echo "  ✅ scripts/ - Essential scripts (9 core scripts + optional analysis scripts)"
echo "  ✅ README.md, PRODUCTION_DEPLOYMENT_SUMMARY.md - Documentation"
echo ""
echo "See CLEANUP_GUIDE.md for complete list of what was kept/deleted."
echo ""
echo "Run this to verify production pipeline still works:"
echo "  python3 config/production_config.py"
echo ""
