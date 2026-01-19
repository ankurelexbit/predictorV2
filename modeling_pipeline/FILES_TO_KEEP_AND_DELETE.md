# 📁 Essential Files vs Temporary Files

**Purpose**: Guide for cleanup - what to keep for production vs what can be deleted

---

## ✅ ESSENTIAL FILES - KEEP THESE

### 🔧 Core Pipeline Scripts (01-11)

These are the main pipeline - **KEEP ALL**:

```
01_sportmonks_data_collection.py    # Fetch data from SportMonks API
02_sportmonks_feature_engineering.py # Generate features from raw data
04_model_baseline_elo.py            # Train Elo model
05_model_dixon_coles.py             # Train Dixon-Coles model
06_model_xgboost.py                 # Train XGBoost model
07_model_ensemble.py                # Train ensemble (stacking)
08_evaluation.py                    # Evaluate model performance
09_prediction_pipeline.py           # Make predictions on new matches
11_smart_betting_strategy.py        # Betting strategy with optimized thresholds
```

### ⚙️ Configuration & Utilities

**KEEP**:
```
config.py                           # All settings (leagues, paths, hyperparameters)
utils.py                            # Shared utility functions
requirements.txt                    # Python dependencies
run_pipeline.sh                     # Script to run full pipeline
```

**DELETE**:
```
config.py.backup                    # Backup copy - not needed
```

### 🎯 Live Prediction & Deployment

**KEEP**:
```
predict_live.py                     # Generate predictions for upcoming matches
fetch_standings.py                  # Fetch current standings from ESPN
live_testing_system.py              # Run live testing and validation
backtest_live_system.py             # Backtest live prediction system
optimize_betting_thresholds.py      # Optimize betting thresholds
generate_180day_predictions.py      # Generate predictions for calibration
```

**DELETE** (one-time scripts):
```
predict_upcoming.py                 # Superseded by predict_live.py
predict_demo.py                     # Demo/test script
predict_with_bets.py               # Old version
```

### 📊 Data Files

**KEEP**:
```
data/processed/sportmonks_features.csv    # Main feature file (60MB) - ESSENTIAL
```

**DELETE** (temporary/intermediate):
```
data/processed/features.csv              # Old feature file (if exists)
data/processed/features_data_driven.csv  # Duplicate/old version
```

### 🧠 Trained Models

**KEEP ALL in models/ directory**:
```
models/elo_model.joblib
models/dixon_coles_model.joblib
models/xgboost_model.joblib
models/stacking_ensemble.joblib
models/ensemble_model.joblib
```

**DELETE**:
```
models/xgboost_optimized.joblib    # Old version (if xgboost_model.joblib exists)
```

### 📚 Key Documentation

**KEEP**:
```
README.md                          # Project overview
CLAUDE.md                          # Instructions for Claude Code
CALIBRATION_180_DAYS_COMPLETE.md  # Final calibration results
FINAL_LIVE_TEST_RESULTS.md        # Live testing validation
LIVE_TESTING_COMPLETE.md          # Live system documentation
```

**DELETE** (outdated/interim):
```
CALIBRATION_TIMEFRAME_SUMMARY.md          # Superseded by 180-day version
BETTING_STRATEGY_OPTIMIZATION_COMPLETE.md # Old version with bug
START_HERE.md                             # Outdated
QUICKSTART.md                             # Outdated
QUICK_START_OPTIMIZED.md                  # Outdated
DOCUMENTATION_INDEX.md                    # Can regenerate
```

---

## 🗑️ TEMPORARY FILES - SAFE TO DELETE

### 🧪 Testing & Debugging Scripts

All these were created for one-time testing/debugging:

```
analyze_features.py
analyze_training_features.py
check_data_quality.py
check_feature_mismatch.py
check_feyenoord.py
check_home_advantage.py
check_prediction_distribution.py
check_probabilities.py
check_results.py
check_team_names.py
compare_feature_calculation.py
compare_predictions.py
debug_features.py
deep_diagnostic.py
diagnose_dixon_coles.py
test_base_models.py
test_combined_strategy.py
test_draw_strategies.py
test_elo_loading.py
test_model_directly.py
test_optimizations.py
test_paper_trading.csv
test_strategy_thresholds.py
test_two_months.py
validate_features.py
verify_retraining.py
```

### 📊 Temporary Output Files

Prediction files (keep most recent, delete old ones):

```
# KEEP MOST RECENT:
historical_predictions_180days_20260119_193325.csv  # Latest 180-day calibration

# DELETE OLDER VERSIONS:
recent_predictions_20260119_164742.csv              # Old 90-day
10_day_predictions_20260119_173145.csv             # Old 10-day
demo_predictions.csv                                # Demo only
predictions_2025_*.csv                              # All old prediction files
predictions_2026_*.csv                              # Old prediction files
predictions_jan_*.csv                               # Old format
```

Bet history files (keep if needed for analysis, otherwise delete):

```
# KEEP IF DOING HISTORICAL ANALYSIS:
complete_bet_history.csv              # Historical bets

# DELETE (duplicates or old):
10_day_all_bets.csv
10_day_bets_20260119_173145.csv
10_day_daily_summary.csv
two_month_performance.csv
paper_trading_log.csv                  # Test file
```

Threshold optimization results (keep latest, delete old):

```
# KEEP LATEST:
threshold_optimization_20260119_180247.txt         # Latest with correct bug fix
threshold_optimization_results_20260119_180247.json

# DELETE OLD:
threshold_optimization_20260119_175639.txt         # Has bug
threshold_optimization_results_20260119_175639.json
```

### 📝 Temporary Documentation/Reports

```
10_DAY_PERFORMANCE_REPORT.md          # Old
10_day_analysis_output.log            # Log file
LIVE_10_DAY_BACKTEST_*.txt           # Old test report
LIVE_PERFORMANCE_REPORT.md           # Superseded
strategy_comparison_report.json       # Old
strategy_comparison.csv              # Old
two_month_report.md                   # Old
weekly_accuracy_report.md            # Old
xgboost_optimization.log             # Log file
last_10_days.txt                      # Temporary list
```

### 📋 Interim Status/Issue Files

```
FINAL_DATA_SOURCES.md                 # Interim document
FINAL_STATUS.md                       # Superseded
IMPLEMENTATION_COMPLETE.md            # Interim
IMPLEMENTATION_SUMMARY.md             # Interim
ISSUES_FOUND.md                       # Old issues
LIVE_PREDICTION_DATA_SOURCES.md      # Covered in other docs
FINAL_OPTIMIZATION_SUMMARY.md        # Old
```

### 🔧 One-Time Utility Scripts

```
bet_selector.py                       # Old version
build_complete_bet_history.py         # One-time script
build_complete_bet_history_v2.py     # One-time script
calculate_pnl.py                      # One-time script
create_elo_lookup.py                  # One-time script
run_10_day_analysis.py               # One-time analysis
simple_pnl_calculator.py             # One-time analysis
```

### 🎨 Backtest Variations

Keep `backtest_live_system.py`, delete the rest:

```
# KEEP:
backtest_live_system.py              # Main live backtest

# DELETE:
backtest_betting_strategy.py         # Old version
backtest_relaxed.py                  # Experimental
backtest_simple.py                   # Simple version
backtest_strategies.py               # Old version
```

### 📦 Old/Unused Pipeline Scripts

```
01_data_collection.py                # Old CSV-based version (now using SportMonks)
02_data_storage.py                   # Old storage method
02_process_raw_data.py              # Old processing
03_feature_engineering.py            # Old feature engineering (now using sportmonks version)
optimize_xgboost.py                  # One-time optimization
evaluate_recent_predictions.py       # One-time evaluation
10_hyperparameter_tuning.py         # Never fully implemented
```

### 🔄 Generated Prediction Files (Very Old)

All the old daily prediction files - you can keep last 7-14 days, delete older:

```
predictions_2025_11_*.csv            # November 2025 - DELETE
predictions_2025_12_*.csv            # December 2025 - DELETE
predictions_2026_01_01.csv           # January 1 - DELETE
predictions_2026_01_04.csv           # January 4 - DELETE
... (all old dates)
```

Keep only recent ones if you need them for validation:
```
predictions_2026_01_17.csv           # Last week - KEEP if needed
predictions_2026_01_18.csv           # Last week - KEEP if needed
```

### 🗂️ Miscellaneous

```
COMMANDS.txt                         # Old commands list
CHECKLIST.md                         # Old checklist
DATA_SOURCES_SUMMARY.txt            # Covered in other docs
DATA_AVAILABILITY_GUIDE.md          # Covered in other docs
FEATURE_LIST.md                     # Covered in other docs
FEATURE_VALIDATION_GUIDE.md         # One-time guide
PRE_MATCH_DATA_CHECKLIST.md         # One-time checklist
generate_historical_predictions.py   # Failed attempt (use generate_180day_predictions.py)
```

---

## 📋 CLEAN PIPELINE STRUCTURE

Here's what your directory should look like after cleanup:

```
modeling_pipeline/
│
├── Core Pipeline Scripts (KEEP ALL)
│   ├── 01_sportmonks_data_collection.py
│   ├── 02_sportmonks_feature_engineering.py
│   ├── 04_model_baseline_elo.py
│   ├── 05_model_dixon_coles.py
│   ├── 06_model_xgboost.py
│   ├── 07_model_ensemble.py
│   ├── 08_evaluation.py
│   ├── 09_prediction_pipeline.py
│   └── 11_smart_betting_strategy.py
│
├── Configuration (KEEP)
│   ├── config.py
│   ├── utils.py
│   ├── requirements.txt
│   └── run_pipeline.sh
│
├── Live Prediction (KEEP)
│   ├── predict_live.py
│   ├── fetch_standings.py
│   ├── live_testing_system.py
│   ├── backtest_live_system.py
│   ├── optimize_betting_thresholds.py
│   └── generate_180day_predictions.py
│
├── Data (KEEP)
│   └── processed/
│       └── sportmonks_features.csv (60MB)
│
├── Models (KEEP ALL)
│   └── models/
│       ├── elo_model.joblib
│       ├── dixon_coles_model.joblib
│       ├── xgboost_model.joblib
│       ├── stacking_ensemble.joblib
│       └── ensemble_model.joblib
│
├── Documentation (KEEP KEY DOCS)
│   ├── README.md
│   ├── CLAUDE.md
│   ├── CALIBRATION_180_DAYS_COMPLETE.md
│   ├── FINAL_LIVE_TEST_RESULTS.md
│   └── LIVE_TESTING_COMPLETE.md
│
└── Recent Outputs (KEEP RECENT ONLY)
    └── historical_predictions_180days_20260119_193325.csv
```

---

## 🚀 CLEANUP COMMANDS

### Safe Deletion (Step by Step)

```bash
# 1. Delete test/debug scripts
rm -f analyze_*.py check_*.py compare_*.py debug_*.py deep_diagnostic.py
rm -f diagnose_*.py test_*.py validate_*.py verify_*.py

# 2. Delete old pipeline scripts
rm -f 01_data_collection.py 02_data_storage.py 02_process_raw_data.py
rm -f 03_feature_engineering.py 10_hyperparameter_tuning.py
rm -f optimize_xgboost.py evaluate_recent_predictions.py

# 3. Delete temporary prediction files (keep last 7 days if needed)
rm -f predictions_2025_*.csv
rm -f predictions_2026_01_0*.csv  # Delete Jan 1-9
rm -f predictions_jan_*.csv
rm -f demo_predictions.csv
rm -f recent_predictions_20260119_164742.csv

# 4. Delete old bet history files
rm -f 10_day_*.csv two_month_*.csv complete_bet_history.csv
rm -f paper_trading_log.csv strategy_comparison.*

# 5. Delete old threshold optimization files
rm -f threshold_optimization_20260119_175639.*

# 6. Delete temporary docs
rm -f *_SUMMARY.md *_COMPLETE.md IMPLEMENTATION_*.md ISSUES_*.md
rm -f START_HERE.md QUICKSTART.md QUICK_START_*.md
rm -f DATA_*.md FEATURE_*.md PRE_MATCH_*.md
rm -f weekly_*.md two_month_*.md
rm -f BETTING_STRATEGY_OPTIMIZATION_COMPLETE.md
rm -f CALIBRATION_TIMEFRAME_SUMMARY.md
# KEEP: CALIBRATION_180_DAYS_COMPLETE.md, FINAL_LIVE_TEST_RESULTS.md, LIVE_TESTING_COMPLETE.md

# 7. Delete misc files
rm -f COMMANDS.txt CHECKLIST.md last_10_days.txt
rm -f config.py.backup
rm -f *.log

# 8. Delete old backtest scripts
rm -f backtest_betting_strategy.py backtest_relaxed.py
rm -f backtest_simple.py backtest_strategies.py

# 9. Delete one-time utility scripts
rm -f bet_selector.py build_complete_bet_history*.py
rm -f calculate_pnl.py create_elo_lookup.py
rm -f run_10_day_analysis.py simple_pnl_calculator.py
rm -f generate_historical_predictions.py

# 10. Delete old predict scripts
rm -f predict_upcoming.py predict_demo.py predict_with_bets.py
```

### Verify What's Left

```bash
# Count essential files (should be ~25-30)
ls -1 *.py *.sh *.md | wc -l

# List what remains
ls -1 *.py *.sh *.md *.txt *.csv 2>/dev/null | head -40
```

---

## 📦 BACKUP BEFORE DELETE

Before deleting, create a backup of deletable files:

```bash
# Create backup directory
mkdir -p ../backup_temp_files_$(date +%Y%m%d)

# Move (not delete) temporary files there
mv analyze_*.py check_*.py compare_*.py debug_*.py ../backup_temp_files_$(date +%Y%m%d)/ 2>/dev/null
mv test_*.py validate_*.py verify_*.py ../backup_temp_files_$(date +%Y%m%d)/ 2>/dev/null
mv predictions_2025_*.csv predictions_jan_*.csv ../backup_temp_files_$(date +%Y%m%d)/ 2>/dev/null

# After 1 week, if everything works, delete backup:
# rm -rf ../backup_temp_files_20260119
```

---

## ✅ ESSENTIAL FILES SUMMARY

### Minimum Required for Full Pipeline

**Total: ~20-25 core files**

**Pipeline (9 files)**:
1. 01_sportmonks_data_collection.py
2. 02_sportmonks_feature_engineering.py
3. 04_model_baseline_elo.py
4. 05_model_dixon_coles.py
5. 06_model_xgboost.py
6. 07_model_ensemble.py
7. 08_evaluation.py
8. 09_prediction_pipeline.py
9. 11_smart_betting_strategy.py

**Live Prediction (6 files)**:
10. predict_live.py
11. fetch_standings.py
12. live_testing_system.py
13. backtest_live_system.py
14. optimize_betting_thresholds.py
15. generate_180day_predictions.py

**Config (4 files)**:
16. config.py
17. utils.py
18. requirements.txt
19. run_pipeline.sh

**Data (1 file)**:
20. data/processed/sportmonks_features.csv

**Models (5 files)**:
21-25. models/*.joblib (all 5 models)

**Docs (3 files)**:
26. README.md
27. CLAUDE.md
28. CALIBRATION_180_DAYS_COMPLETE.md

---

## 🎯 QUICK CLEANUP SCRIPT

Save this as `cleanup.sh`:

```bash
#!/bin/bash

echo "🧹 Cleaning up temporary files..."

# Backup directory
BACKUP_DIR="../backup_temp_files_$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Move temporary files to backup
mv analyze_*.py check_*.py compare_*.py debug_*.py "$BACKUP_DIR/" 2>/dev/null
mv diagnose_*.py test_*.py validate_*.py verify_*.py "$BACKUP_DIR/" 2>/dev/null
mv predictions_2025_*.csv predictions_jan_*.csv "$BACKUP_DIR/" 2>/dev/null
mv 10_day_*.csv two_month_*.csv "$BACKUP_DIR/" 2>/dev/null
mv *_SUMMARY.md IMPLEMENTATION_*.md ISSUES_*.md "$BACKUP_DIR/" 2>/dev/null
mv bet_selector.py build_complete_bet*.py calculate_pnl.py "$BACKUP_DIR/" 2>/dev/null
mv backtest_betting_strategy.py backtest_relaxed.py backtest_simple.py "$BACKUP_DIR/" 2>/dev/null

# Delete old versions
rm -f config.py.backup
rm -f 01_data_collection.py 02_data_storage.py 02_process_raw_data.py 03_feature_engineering.py
rm -f threshold_optimization_20260119_175639.*
rm -f *.log

echo "✅ Cleanup complete!"
echo "📦 Temporary files backed up to: $BACKUP_DIR"
echo "🗑️  After verifying everything works, delete backup with:"
echo "    rm -rf $BACKUP_DIR"
```

Run with: `chmod +x cleanup.sh && ./cleanup.sh`

---

## 📊 FILE COUNT SUMMARY

| Category | Essential Files | Temp Files | Total |
|----------|----------------|------------|-------|
| **Pipeline Scripts** | 9 | 15+ | 24+ |
| **Live/Deploy Scripts** | 6 | 8+ | 14+ |
| **Config/Utils** | 4 | 1 | 5 |
| **Data Files** | 1 | 5+ | 6+ |
| **Models** | 5 | 1 | 6 |
| **Documentation** | 3-5 | 15+ | 20+ |
| **Predictions** | 1-2 | 30+ | 32+ |
| **Reports** | 1-2 | 10+ | 12+ |
| **TOTAL** | **29-34** | **85+** | **119+** |

**Cleanup saves**: ~85 files, ~30-50MB disk space

---

**Last Updated**: 2026-01-19
**Recommended Action**: Run cleanup script to move temporary files to backup, verify for 1 week, then delete backup
