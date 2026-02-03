# Scripts Directory Cleanup Guide

**Date:** February 3, 2026

This guide identifies essential vs. temporary scripts in the `scripts/` directory.

---

## ✅ ESSENTIAL SCRIPTS - KEEP (11 scripts)

### Core Production Pipeline
```
1. backfill_historical_data.py       (20 KB)  - Download historical data from API
2. convert_json_to_csv.py            (8.8 KB) - Convert JSON to CSV (100x speedup)
3. generate_training_data.py         (2.8 KB) - Generate feature vectors
4. train_production_model.py         (14 KB)  - Train models with auto-versioning
5. predict_production.py             (20 KB)  - Live predictions + database storage
6. predict_live_with_history.py      (21 KB)  - Core prediction engine library
7. update_results.py                 (8.8 KB) - Update PnL tracking
8. get_pnl.py                        (4.1 KB) - Generate PnL reports
9. weekly_retrain_pipeline.py        (8.7 KB) - Automated weekly retraining
10. migrate_database.py              (4.0 KB) - Database schema setup
11. migrate_to_versioned_models.py   (3.3 KB) - Model versioning migration
```

**Total Essential:** ~115 KB, 11 scripts

---

## ❌ TEMPORARY/EXPERIMENTAL SCRIPTS - SAFE TO DELETE (39 scripts)

### Analysis Scripts (Experiments - No Longer Needed)
```
analyze_calibration_failure.py       (9.8 KB)  - Calibration experiment analysis
analyze_ev_detailed.py               (8.5 KB)  - EV strategy analysis
analyze_ev_strategy.py               (15 KB)   - EV strategy testing
analyze_home_threshold_option3.py    (8.5 KB)  - Home threshold analysis
analyze_thresholds_no_odds.py        (4.5 KB)  - Old threshold analysis
analyze_thresholds.py                (6.4 KB)  - Old threshold analysis
```
**Subtotal:** ~53 KB, 6 scripts

### Backtest Scripts (Old - No Longer Needed)
```
backtest_january_2026.py             (18 KB)   - Old backtest
backtest_multioutcome_january_2026.py (16 KB)  - Old backtest
backtest_threshold_strategy.py       (14 KB)   - Old backtest
```
**Subtotal:** ~48 KB, 3 scripts

### Model Comparison Scripts (Experiments)
```
compare_all_models_live.py           (24 KB)   - Live model comparison
compare_model_distributions.py       (11 KB)   - Distribution analysis
compare_models_from_db.py            (15 KB)   - Database model comparison
compare_models_live.py               (18 KB)   - Live comparison
compare_models_simple.py             (8.6 KB)  - Simple comparison
compare_optimization_strategies.py   (11 KB)   - Optimization comparison
```
**Subtotal:** ~88 KB, 6 scripts

### Test Scripts (Experiments)
```
test_all_models_with_thresholds.py   (15 KB)   - Model threshold testing
test_calibrated_ev_strategy.py       (13 KB)   - Calibration EV test
test_core_infrastructure.py          (5.0 KB)  - Infrastructure test
test_database.py                     (2.6 KB)  - Database test
test_feature_orchestrator.py         (2.9 KB)  - Feature orchestrator test
test_unbiased_models.py              (12 KB)   - Unbiased model test
```
**Subtotal:** ~51 KB, 6 scripts

### Experimental Training Scripts
```
train_calibrated_model.py            (13 KB)   - Failed calibration experiment
train_unbiased_calibrated_model.py   (9.9 KB)  - Unbiased model experiment
train_production_model_backup.py     (11 KB)   - Backup (can delete after verifying new version)
quick_retrain.py                     (3.6 KB)  - Old retrain script
retrain_with_draw_features.py        (9.3 KB)  - Old retrain script
```
**Subtotal:** ~47 KB, 5 scripts

### Debug/Investigation Scripts
```
check_available_fixtures.py          (4.2 KB)  - Debug script
check_fixture_structure.py           (6.2 KB)  - Debug script
debug_predictions.py                 (3.2 KB)  - Debug script
diagnose_feature_mismatch.py         (6.8 KB)  - Debug script
investigate_production_issues.py     (12 KB)   - Investigation script
query_predictions.py                 (2.5 KB)  - Query script
validate_historical_data.py          (5.2 KB)  - Validation script
validate_live_features.py            (16 KB)   - Validation script
```
**Subtotal:** ~56 KB, 8 scripts

### Old/Superseded Prediction Scripts
```
predict_live_standalone.py           (28 KB)   - Old version (superseded by predict_production.py)
predict_live_v4.py                   (20 KB)   - Old version (superseded)
predict_and_store.py                 (12 KB)   - Old version (superseded)
```
**Subtotal:** ~60 KB, 3 scripts

### Old Data Download Scripts
```
download_historical_data.py          (19 KB)   - Old version (use backfill_historical_data.py)
```
**Subtotal:** ~19 KB, 1 script

### Build Scripts (Optional)
```
build_cache.py                       (2.4 KB)  - Cache builder (optional, can keep for optimization)
```
**Subtotal:** ~2.4 KB, 1 script

---

## 📊 Summary

**Essential Scripts:** 11 scripts (~115 KB)
**Temporary Scripts:** 39 scripts (~422 KB)

**Total Space to Reclaim:** ~422 KB

---

## 🗑️ Cleanup Commands

### Safe Deletion (Recommended)

```bash
# Navigate to scripts directory
cd scripts/

# Delete analysis scripts
rm -f analyze_calibration_failure.py
rm -f analyze_ev_detailed.py
rm -f analyze_ev_strategy.py
rm -f analyze_home_threshold_option3.py
rm -f analyze_thresholds_no_odds.py
rm -f analyze_thresholds.py

# Delete backtest scripts
rm -f backtest_january_2026.py
rm -f backtest_multioutcome_january_2026.py
rm -f backtest_threshold_strategy.py

# Delete comparison scripts
rm -f compare_all_models_live.py
rm -f compare_model_distributions.py
rm -f compare_models_from_db.py
rm -f compare_models_live.py
rm -f compare_models_simple.py
rm -f compare_optimization_strategies.py

# Delete test scripts
rm -f test_all_models_with_thresholds.py
rm -f test_calibrated_ev_strategy.py
rm -f test_core_infrastructure.py
rm -f test_database.py
rm -f test_feature_orchestrator.py
rm -f test_unbiased_models.py

# Delete experimental training scripts
rm -f train_calibrated_model.py
rm -f train_unbiased_calibrated_model.py
rm -f train_production_model_backup.py
rm -f quick_retrain.py
rm -f retrain_with_draw_features.py

# Delete debug/investigation scripts
rm -f check_available_fixtures.py
rm -f check_fixture_structure.py
rm -f debug_predictions.py
rm -f diagnose_feature_mismatch.py
rm -f investigate_production_issues.py
rm -f query_predictions.py
rm -f validate_historical_data.py
rm -f validate_live_features.py

# Delete old prediction scripts
rm -f predict_live_standalone.py
rm -f predict_live_v4.py
rm -f predict_and_store.py

# Delete old data download script
rm -f download_historical_data.py

# Return to root
cd ..
```

### Or Use Automated Script

```bash
bash cleanup_scripts.sh
```

---

## ✅ Final Essential Script List

After cleanup, your `scripts/` directory will contain only:

1. `backfill_historical_data.py` - Download data
2. `convert_json_to_csv.py` - Speed optimization
3. `generate_training_data.py` - Feature generation
4. `train_production_model.py` - Model training
5. `predict_production.py` - Live predictions
6. `predict_live_with_history.py` - Prediction engine
7. `update_results.py` - PnL tracking
8. `get_pnl.py` - Reports
9. `weekly_retrain_pipeline.py` - Automation
10. `migrate_database.py` - Database setup
11. `migrate_to_versioned_models.py` - Version migration
12. `build_cache.py` - Optional (for caching optimization)

**Total: 11-12 essential scripts**

---

## 🔄 Workflow After Cleanup

Your complete production workflow will be:

```bash
# 1. Download data (once or weekly)
python3 scripts/backfill_historical_data.py --start-date 2023-08-01 --end-date 2024-05-31

# 2. Convert to CSV (once, or after new data)
python3 scripts/convert_json_to_csv.py

# 3. Generate features
python3 scripts/generate_training_data.py --output data/training_data.csv

# 4. Train model (auto-versioning)
python3 scripts/train_production_model.py --data data/training_data.csv

# 5. Make predictions
python3 scripts/predict_production.py --days-ahead 7

# 6. Check PnL
python3 scripts/get_pnl.py --days 30

# 7. Weekly retrain (automated via cron)
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4
```

All temporary/experimental scripts are no longer needed! ✅
