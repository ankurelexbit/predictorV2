# Pre-Execution Checklist

Before running the pipeline, verify these requirements:

## ✓ Setup Checklist

### 1. Python Environment
- [ ] Python 3.8+ installed
  ```bash
  python --version  # Should show 3.8 or higher
  ```

- [ ] Virtual environment created
  ```bash
  ls venv/  # Should exist
  ```

- [ ] Virtual environment activated
  ```bash
  echo $VIRTUAL_ENV  # Should show path to venv
  ```

- [ ] Dependencies installed
  ```bash
  pip list | grep -E "(pandas|xgboost|scikit-learn|requests)"
  ```

### 2. API Configuration
- [ ] Sportmonks account created (https://www.sportmonks.com/)
- [ ] API key obtained from dashboard
- [ ] API key added to `config.py`
  ```bash
  grep "SPORTMONKS_API_KEY" config.py | grep -v "your_api_key_here"
  ```

### 3. Directory Structure
- [ ] Project directory exists
  ```bash
  pwd  # Should be in modeling_pipeline/
  ```

- [ ] Data directories exist
  ```bash
  mkdir -p data/raw/sportmonks data/processed models/evaluation
  ```

### 4. Disk Space
- [ ] At least 100 MB free space
  ```bash
  df -h .  # Check available space
  ```

## ✓ Verification Tests

Run these commands to verify setup:

```bash
# 1. Test Python imports
python -c "import pandas, xgboost, sklearn, requests; print('✓ All imports successful')"

# 2. Test config file
python -c "from config import SPORTMONKS_API_KEY; print('✓ API key loaded')"

# 3. Check directory permissions
touch data/test.txt && rm data/test.txt && echo "✓ Write permissions OK"

# 4. Test API connection (optional)
python -c "
from config import SPORTMONKS_API_KEY, SPORTMONKS_BASE_URL
import requests
headers = {'Authorization': SPORTMONKS_API_KEY}
r = requests.get(f'{SPORTMONKS_BASE_URL}/leagues', headers=headers)
print(f'✓ API connection OK (status: {r.status_code})')
"
```

## ✓ Ready to Run

If all checks pass, you're ready to execute the pipeline:

### Option 1: Automated Script
```bash
source venv/bin/activate
./run_pipeline.sh
```

### Option 2: Manual Commands
```bash
source venv/bin/activate
python 01_sportmonks_data_collection.py
python 02_sportmonks_feature_engineering.py
python 04_model_baseline_elo.py
python 05_model_dixon_coles.py
python 06_model_xgboost.py
python 07_model_ensemble.py
python 08_evaluation.py
```

### Option 3: One-Liner
```bash
source venv/bin/activate && \
python 01_sportmonks_data_collection.py && \
python 02_sportmonks_feature_engineering.py && \
python 04_model_baseline_elo.py && \
python 05_model_dixon_coles.py && \
python 06_model_xgboost.py && \
python 07_model_ensemble.py && \
python 08_evaluation.py
```

## ✓ Post-Execution Verification

After running the pipeline, verify outputs:

```bash
# Check raw data files
ls -lh data/raw/sportmonks/
# Expected: fixtures.csv, lineups.csv, events.csv, sidelined.csv, standings.csv

# Check processed features
ls -lh data/processed/
# Expected: sportmonks_features.csv (~15 MB)

# Check trained models
ls -lh models/*.joblib
# Expected: 4 model files

# Check evaluation results
ls -lh models/evaluation/
# Expected: Multiple analysis files and plots

# Verify model performance
grep "Log Loss" models/xgboost_model.log 2>/dev/null || \
python -c "
import joblib
model = joblib.load('models/xgboost_model.joblib')
print('✓ XGBoost model loaded successfully')
"
```

## Common Issues

### Issue: Virtual environment not activating
```bash
# Solution: Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: API key not working
```bash
# Solution: Verify API key format
# Should be a long alphanumeric string, not "your_api_key_here"
grep SPORTMONKS_API_KEY config.py
```

### Issue: Permission denied on run_pipeline.sh
```bash
# Solution: Make script executable
chmod +x run_pipeline.sh
```

### Issue: Module not found errors
```bash
# Solution: Reinstall dependencies in venv
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

## Estimated Timeline

| Phase | Time | Can Run Overnight |
|-------|------|-------------------|
| Data Collection | 25 min | Yes |
| Feature Engineering | 30 sec | No (too fast) |
| Model Training | 20 sec | No (too fast) |
| Evaluation | 15 sec | No (too fast) |
| **Total** | **~26 min** | **Yes** |

## Success Indicators

You'll know the pipeline succeeded if:

1. ✓ No error messages in terminal
2. ✓ All 5 raw CSV files created
3. ✓ sportmonks_features.csv has 2,877 rows × 465 columns
4. ✓ 4 model .joblib files created
5. ✓ XGBoost test log loss ~0.998
6. ✓ Evaluation results in models/evaluation/

## Next Steps After Success

1. Review model performance in evaluation results
2. Examine feature importance rankings
3. Analyze predictions on test set
4. Consider collecting data for additional leagues
5. Update models regularly with new match data

---

**Still having issues?** See detailed troubleshooting in [README.md](README.md)
