# Pipeline Cleanup Guide

**Date:** February 3, 2026
**Purpose:** Identify essential vs. temporary files for production pipeline

---

## ✅ ESSENTIAL FILES - KEEP THESE

### Core Source Code (src/)
```
src/                                    # ALL files - core pipeline code
├── data/
│   ├── json_loader.py                 # Data loading
│   └── sportmonks_client.py           # API client
├── features/
│   ├── elo_calculator.py              # Elo ratings
│   ├── feature_orchestrator.py        # Main feature generation
│   ├── pillar1_fundamentals.py        # 50 features
│   ├── pillar2_modern_analytics.py    # 60 features
│   ├── pillar3_hidden_edges.py        # 52 features
│   └── standings_calculator.py        # League standings
└── models/
    └── xgboost_model.py               # Model wrapper
```
**Action:** KEEP ALL

---

### Production Model & Config
```
models/weight_experiments/
├── option3_balanced.joblib            # ✅ PRODUCTION MODEL (12 MB)
└── option3_balanced_metadata.json     # ✅ Model metadata

config/
└── production_config.py               # ✅ Central configuration
```
**Action:** KEEP

---

### Essential Scripts
```
scripts/
├── backfill_historical_data.py        # ✅ Download historical data
├── convert_json_to_csv.py             # ✅ Speed optimization
├── generate_training_data.py          # ✅ Feature generation
├── train_production_model.py          # ✅ Training pipeline
├── predict_production.py              # ✅ Live predictions
├── update_results.py                  # ✅ PnL tracking
├── get_pnl.py                         # ✅ Performance reports
├── weekly_retrain_pipeline.py         # ✅ Automated retraining
└── migrate_database.py                # ✅ Database setup
```
**Action:** KEEP

---

### Documentation
```
README.md                              # ✅ Main documentation
CLAUDE.md                              # ✅ Claude Code instructions
PRODUCTION_DEPLOYMENT_SUMMARY.md       # ✅ Deployment guide
TRAINING_PIPELINE_UPDATED.md           # ✅ Training guide
WHY_CALIBRATION_FAILED.md              # ✅ Technical analysis (useful reference)
EV_STRATEGY_ANALYSIS_RESULTS.md        # ✅ EV analysis (useful reference)
```
**Action:** KEEP

---

### Training Data
```
data/
├── training_data.csv                  # ✅ Latest training dataset (23 MB)
├── processed/
│   └── fixtures.csv                   # ✅ Converted fixtures (if exists)
└── historical/                        # ✅ Raw JSON data (keep for retraining)
```
**Action:** KEEP

---

## ❌ TEMPORARY FILES - SAFE TO DELETE

### Experimental Models (77.6 MB total to delete)
```
models/calibrated/                     # ❌ Failed calibration experiment
├── option3_calibrated_for_ev.joblib   # 21 MB
├── option3_calibrated_for_ev_metadata.json
└── calibration_curves.png

models/unbiased/                       # ❌ Unbiased model experiment
├── unbiased_base.joblib               # ~12 MB
├── unbiased_calibrated.joblib         # ~12 MB
└── unbiased_metadata.json

models/final/                          # ❌ Old experiments
├── model_conservative_calibrated.joblib
├── model_conservative_uncalibrated.joblib
├── model_no_weights_calibrated.joblib
├── model_no_weights_uncalibrated.joblib
└── training_results.json

models/moderate_weights/               # ❌ Old experiments
├── model_moderate_calibrated.joblib
├── model_moderate_uncalibrated.joblib
└── training_results.json

models/production/                     # ❌ Old production model
├── model_v4_162feat.joblib            # Superseded by option3
└── model_v4_162feat_metadata.json

models/with_draw_features/             # ❌ Old experiments
├── conservative_with_draw_features.joblib
├── xgboost_fixed.joblib
├── xgboost_with_draw_features.joblib
├── results.json
└── xgboost_results.json

models/                                # ❌ Root-level old models
├── v4_model.joblib
├── v4_model_metadata.json
├── v4_optimized_model.joblib
├── v4_optimized_weighted_model.joblib
├── v4_xgboost_tuned.joblib
└── v4_xgboost.joblib
```

**KEEP from models/:**
```
models/weight_experiments/             # ✅ KEEP ALL (these are your 4 tested options)
├── option1_conservative.joblib        # Backup option
├── option1_conservative_metadata.json
├── option2_aggressive.joblib          # Backup option
├── option2_aggressive_metadata.json
├── option3_balanced.joblib            # ✅ PRODUCTION
├── option3_balanced_metadata.json     # ✅ PRODUCTION
├── option4_original.joblib            # Backup option
└── option4_original_metadata.json
```

---

### Analysis Scripts (keep for reference or delete if space is tight)
```
scripts/
├── analyze_calibration_failure.py     # 📊 Analysis (can delete or keep for reference)
├── analyze_ev_detailed.py             # 📊 Analysis (can delete)
├── analyze_ev_strategy.py             # 📊 Analysis (can delete)
├── analyze_home_threshold_option3.py  # 📊 Analysis (can delete)
├── compare_models_from_db.py          # 📊 Analysis (can delete)
├── compare_optimization_strategies.py # 📊 Analysis (can delete)
├── find_marketable_thresholds.py      # 📊 Analysis (can delete)
├── test_calibrated_ev_strategy.py     # 📊 Testing (can delete)
├── test_unbiased_models.py            # 📊 Testing (can delete)
├── train_calibrated_model.py          # 📊 Experimental (can delete)
└── train_unbiased_calibrated_model.py # 📊 Experimental (can delete)
```

**Old/Duplicate Scripts (safe to delete):**
```
scripts/
├── analyze_thresholds_no_odds.py      # ❌ Old version
├── analyze_thresholds.py              # ❌ Old version
├── backtest_january_2026.py           # ❌ Duplicate backtest
├── backtest_multioutcome_january_2026.py # ❌ Old backtest
├── backtest_threshold_strategy.py     # ❌ Old backtest
├── check_available_fixtures.py        # ❌ Debug script
├── check_fixture_structure.py         # ❌ Debug script
├── compare_all_models_live.py         # ❌ Old comparison
├── compare_model_distributions.py     # ❌ Old comparison
├── compare_models_live.py             # ❌ Old comparison
├── compare_models_simple.py           # ❌ Old comparison
├── debug_predictions.py               # ❌ Debug script
├── diagnose_feature_mismatch.py       # ❌ Debug script
├── download_historical_data.py        # ❌ Use backfill instead
├── investigate_production_issues.py   # ❌ Debug script
├── predict_and_store.py               # ❌ Old version
├── predict_live_standalone.py         # ❌ Old version
├── predict_live_v4.py                 # ❌ Old version
├── predict_live_with_history.py       # ❌ Old version
├── query_predictions.py               # ❌ Debug script
├── quick_retrain.py                   # ❌ Use train_production_model.py
├── retrain_with_draw_features.py      # ❌ Old version
├── test_all_models_with_thresholds.py # ❌ Old testing
├── test_core_infrastructure.py        # ❌ Debug/testing
├── test_database.py                   # ❌ Debug/testing
├── test_feature_orchestrator.py       # ❌ Debug/testing
├── validate_historical_data.py        # ❌ Debug script
└── validate_live_features.py          # ❌ Debug script
```

---

### Log Files
```
logs/
├── model_optimization.log             # ❌ Delete (old)
├── train_calibrated_model.log         # ❌ Delete (experiment)
├── train_calibrated.log               # ❌ Delete (experiment)
└── train_unbiased.log                 # ❌ Delete (experiment)
```
**Action:** DELETE ALL (or keep train_calibrated*.log for reference)

---

### Results Files
```
results/
├── backtest_complete_output.log       # ❌ Delete (old)
├── backtest_full_january_2026.csv     # ❌ Delete (old)
├── backtest_january_2026_complete.csv # ❌ Delete (old)
├── backtest_january_2026.csv          # ❌ Delete (old)
├── backtest_output.log                # ❌ Delete (old)
├── backtest_threshold_output.log      # ❌ Delete (old)
├── backtest_threshold_strategy.csv    # ❌ Delete (old)
├── calibrated_ev_test_results.txt     # ❌ Delete (experiment)
├── calibrated_ev_test.txt             # ❌ Delete (experiment)
├── class_weights_optimization.csv     # 📊 Keep for reference
├── COMPREHENSIVE_MODEL_REPORT.md      # ❌ Delete (superseded)
├── ev_strategy_analysis_jan2026.txt   # 📊 Keep for reference
├── logloss_optimization_full.csv      # ❌ Delete (old)
├── model_comparison_weighted.csv      # 📊 Keep for reference
├── model_comparison_weighted.json     # 📊 Keep for reference
├── model_comparison.csv               # ❌ Delete (old)
├── model_comparison.json              # ❌ Delete (old)
└── threshold_optimization.csv         # ❌ Delete (old)
```

---

### Old Documentation
```
CLASS_WEIGHT_EXPERIMENT.md             # ❌ Delete (superseded by TRAINING_PIPELINE_UPDATED.md)
FEATURE_VALIDATION_REPORT.md           # ❌ Delete (old)
HOME_PREDICTION_IMPROVEMENT_PLAN.md    # ❌ Delete (not implemented)
LIVE_PREDICTION_GUIDE.md               # ❌ Delete (superseded by PRODUCTION_DEPLOYMENT_SUMMARY.md)
MODEL_COMPARISON_FINAL_REPORT.md       # ❌ Delete (superseded)
MODEL_IMPROVEMENT_PLAN.md              # ❌ Delete (old)
PRODUCTION_FILES.md                    # ❌ Delete (superseded by this file)
PRODUCTION_GUIDE.md                    # ❌ Delete (superseded by PRODUCTION_DEPLOYMENT_SUMMARY.md)
QUICK_START_PNL.md                     # ❌ Delete (info now in README)
```

---

## 📋 Cleanup Commands

### Safe Deletion (Recommended)
```bash
# Delete experimental models (~60 MB)
rm -rf models/calibrated/
rm -rf models/unbiased/
rm -rf models/final/
rm -rf models/moderate_weights/
rm -rf models/with_draw_features/
rm models/v4_*.joblib
rm models/v4_*.json

# Delete old production model (superseded by option3)
rm -rf models/production/

# Delete log files
rm logs/*.log

# Delete old result files
rm results/backtest*.csv
rm results/backtest*.log
rm results/calibrated_ev*.txt
rm results/COMPREHENSIVE_MODEL_REPORT.md
rm results/model_comparison.csv
rm results/model_comparison.json
rm results/logloss_optimization_full.csv
rm results/threshold_optimization.csv

# Delete old documentation
rm CLASS_WEIGHT_EXPERIMENT.md
rm FEATURE_VALIDATION_REPORT.md
rm HOME_PREDICTION_IMPROVEMENT_PLAN.md
rm LIVE_PREDICTION_GUIDE.md
rm MODEL_COMPARISON_FINAL_REPORT.md
rm MODEL_IMPROVEMENT_PLAN.md
rm PRODUCTION_FILES.md
rm PRODUCTION_GUIDE.md
rm QUICK_START_PNL.md
```

### Delete Analysis Scripts (Optional - saves ~100 KB)
```bash
# Only if you don't need these for future reference
rm scripts/analyze_calibration_failure.py
rm scripts/analyze_ev_detailed.py
rm scripts/analyze_ev_strategy.py
rm scripts/analyze_home_threshold_option3.py
rm scripts/compare_models_from_db.py
rm scripts/compare_optimization_strategies.py
rm scripts/find_marketable_thresholds.py
rm scripts/test_calibrated_ev_strategy.py
rm scripts/test_unbiased_models.py
rm scripts/train_calibrated_model.py
rm scripts/train_unbiased_calibrated_model.py
```

### Delete Old/Duplicate Scripts (Recommended)
```bash
rm scripts/analyze_thresholds_no_odds.py
rm scripts/analyze_thresholds.py
rm scripts/backtest_january_2026.py
rm scripts/backtest_multioutcome_january_2026.py
rm scripts/backtest_threshold_strategy.py
rm scripts/check_available_fixtures.py
rm scripts/check_fixture_structure.py
rm scripts/compare_all_models_live.py
rm scripts/compare_model_distributions.py
rm scripts/compare_models_live.py
rm scripts/compare_models_simple.py
rm scripts/debug_predictions.py
rm scripts/diagnose_feature_mismatch.py
rm scripts/download_historical_data.py
rm scripts/investigate_production_issues.py
rm scripts/predict_and_store.py
rm scripts/predict_live_standalone.py
rm scripts/predict_live_v4.py
rm scripts/predict_live_with_history.py
rm scripts/query_predictions.py
rm scripts/quick_retrain.py
rm scripts/retrain_with_draw_features.py
rm scripts/test_all_models_with_thresholds.py
rm scripts/test_core_infrastructure.py
rm scripts/test_database.py
rm scripts/test_feature_orchestrator.py
rm scripts/validate_historical_data.py
rm scripts/validate_live_features.py
```

---

## 💾 Storage Savings

**Estimated space to reclaim:**
- Experimental models: ~60 MB
- Old models: ~20 MB
- Log files: ~5 MB
- Old scripts: ~0.5 MB
- Results files: ~2 MB
- **Total: ~87 MB**

---

## ✅ Final Essential File List

After cleanup, your production pipeline only needs:

**Code:**
- `src/` - All source files
- `config/production_config.py`

**Models:**
- `models/weight_experiments/option3_balanced.joblib` (12 MB)
- `models/weight_experiments/option3_balanced_metadata.json`
- `models/weight_experiments/option{1,2,4}*` (backups)

**Scripts (9 essential):**
- `backfill_historical_data.py`
- `convert_json_to_csv.py`
- `generate_training_data.py`
- `train_production_model.py`
- `predict_production.py`
- `update_results.py`
- `get_pnl.py`
- `weekly_retrain_pipeline.py`
- `migrate_database.py`

**Documentation (5 files):**
- `README.md`
- `CLAUDE.md`
- `PRODUCTION_DEPLOYMENT_SUMMARY.md`
- `TRAINING_PIPELINE_UPDATED.md`
- `WHY_CALIBRATION_FAILED.md` (optional - good reference)
- `EV_STRATEGY_ANALYSIS_RESULTS.md` (optional - good reference)

**Data:**
- `data/training_data.csv`
- `data/historical/` (for retraining)
- `data/processed/fixtures.csv` (if exists)

**Total: ~100 files, ~50 MB** (vs ~200+ files, ~140 MB currently)

---

## 🚀 Production Deployment Checklist

After cleanup, verify these essential components work:

```bash
# 1. Validate configuration
python3 config/production_config.py

# 2. Test prediction pipeline
python3 scripts/predict_production.py --days-ahead 1

# 3. Verify model loads
python3 -c "import joblib; m = joblib.load('models/weight_experiments/option3_balanced.joblib'); print('✅ Model loaded')"

# 4. Check PnL tracking
python3 scripts/get_pnl.py --days 7
```

All should work without any deleted files!

---

**Last Updated:** February 3, 2026
