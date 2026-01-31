# Weekly Retrain Guide

## Overview

This guide explains how to set up **automated weekly model retraining** for production use.

---

## Two Approaches

### Approach 1: Full Pipeline (Recommended)

**Use:** `scripts/weekly_retrain_pipeline.py`

Downloads new data, regenerates everything, trains model.

```bash
# Run manually
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4

# Run for specific league only
python3 scripts/weekly_retrain_pipeline.py --weeks-back 4 --league-id 8
```

**Pros:**
- All-in-one script
- Handles everything automatically
- Keeps versioned backups
- Creates symlinks to latest files

**Cons:**
- Downloads data every time
- Regenerates full CSV (slower)

### Approach 2: Incremental Updates (Advanced)

Only download new fixtures, append to CSV, retrain.

See "Incremental Pipeline" section below.

---

## Setup for Weekly Automation

### Option 1: Cron (Linux/Mac)

```bash
# 1. Edit crontab
crontab -e

# 2. Add this line (run every Sunday at 2am)
0 2 * * 0 cd /Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v4 && /usr/bin/python3 scripts/weekly_retrain_pipeline.py --weeks-back 4 >> logs/cron_retrain.log 2>&1

# 3. Save and exit

# Cron schedule examples:
# Every Sunday at 2am:     0 2 * * 0
# Every Monday at 3am:     0 3 * * 1
# Every day at midnight:   0 0 * * *
# Every 3 days at 1am:     0 1 */3 * *
```

### Option 2: macOS launchd

Create file: `~/Library/LaunchAgents/com.predictor.weekly_retrain.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.predictor.weekly_retrain</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v4/scripts/weekly_retrain_pipeline.py</string>
        <string>--weeks-back</string>
        <string>4</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>  <!-- Sunday -->
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v4/logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/ankurgupta/code/predictorV2/modeling_pipeline/pipeline_v4/logs/launchd_stderr.log</string>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.predictor.weekly_retrain.plist
```

### Option 3: systemd Timer (Linux)

Create `/etc/systemd/system/predictor-retrain.service`:

```ini
[Unit]
Description=Predictor Weekly Retrain
After=network.target

[Service]
Type=oneshot
User=ankurgupta
WorkingDirectory=/path/to/pipeline_v4
ExecStart=/usr/bin/python3 scripts/weekly_retrain_pipeline.py --weeks-back 4
StandardOutput=append:/path/to/pipeline_v4/logs/systemd_retrain.log
StandardError=append:/path/to/pipeline_v4/logs/systemd_retrain.log
```

Create `/etc/systemd/system/predictor-retrain.timer`:

```ini
[Unit]
Description=Run Predictor Retrain Weekly

[Timer]
OnCalendar=Sun *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable predictor-retrain.timer
sudo systemctl start predictor-retrain.timer
```

### Option 4: GitHub Actions (Cloud)

Create `.github/workflows/weekly_retrain.yml`:

```yaml
name: Weekly Model Retrain

on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday at 2am UTC
  workflow_dispatch:  # Allow manual trigger

jobs:
  retrain:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          cd modeling_pipeline/pipeline_v4
          pip install -r requirements.txt

      - name: Run retrain pipeline
        env:
          SPORTMONKS_API_KEY: ${{ secrets.SPORTMONKS_API_KEY }}
        run: |
          cd modeling_pipeline/pipeline_v4
          python3 scripts/weekly_retrain_pipeline.py --weeks-back 4

      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: modeling_pipeline/pipeline_v4/models/v4_model_latest.joblib
```

---

## What the Pipeline Does

### Step-by-Step Process

1. **Download New Data** (last N weeks)
   - Calls `backfill_historical_data.py`
   - Downloads fixtures with statistics
   - Saves to `data/historical/`

2. **Convert to CSV**
   - Calls `convert_json_to_csv.py`
   - Uses CORRECT type_id mappings (already fixed!)
   - Saves to `data/processed/fixtures_with_stats.csv`

3. **Generate Training Data**
   - Calls `generate_training_data.py`
   - Creates features for all fixtures
   - Saves to `data/training_data_YYYYMMDD_HHMMSS.csv`
   - Creates symlink: `data/training_data_latest.csv`

4. **Train Model**
   - Calls `train_improved_model.py`
   - Trains stacking ensemble (XGB+LGB+Cat)
   - Saves to `models/v4_model_YYYYMMDD_HHMMSS.joblib`
   - Creates symlink: `models/v4_model_latest.joblib`

5. **Cleanup**
   - Keeps last 3 versions
   - Deletes older files to save space

### Files Created

```
data/
├── training_data_20260131_140000.csv  # Versioned
├── training_data_20260207_140000.csv
├── training_data_20260214_140000.csv
└── training_data_latest.csv -> training_data_20260214_140000.csv  # Symlink

models/
├── v4_model_20260131_140000.joblib
├── v4_model_20260207_140000.joblib
├── v4_model_20260214_140000.joblib
└── v4_model_latest.joblib -> v4_model_20260214_140000.joblib  # Symlink

logs/
├── weekly_retrain.log  # Pipeline execution log
└── cron_retrain.log    # Cron output
```

---

## Monitoring

### Check if Pipeline is Running

```bash
# View logs
tail -f logs/weekly_retrain.log

# Check latest model
ls -lh models/v4_model_latest.joblib

# Check when last retrained
stat models/v4_model_latest.joblib
```

### Verify Pipeline Success

```python
# Check log for errors
grep -i "error\|failed" logs/weekly_retrain.log

# Check model was created today
find models -name "v4_model_*.joblib" -mtime -1

# Verify training data coverage
python3 -c "
import pandas as pd
df = pd.read_csv('data/training_data_latest.csv')
df = df.drop_duplicates(subset=['fixture_id'])
xg_cov = df['home_derived_xg_per_match_5'].notna().sum() / len(df) * 100
print(f'xG coverage: {xg_cov:.1f}%')
print('Status:', '✅ Good' if xg_cov > 80 else '❌ Bad')
"
```

### Email Notifications (Optional)

Add to end of `weekly_retrain_pipeline.py`:

```python
def send_notification(success: bool, log_file: str):
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg['Subject'] = f"Model Retrain {'Success' if success else 'FAILED'}"
    msg['From'] = 'your-email@example.com'
    msg['To'] = 'your-email@example.com'

    with open(log_file) as f:
        msg.set_content(f.read())

    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.starttls()
        smtp.login('your-email@example.com', 'your-app-password')
        smtp.send_message(msg)
```

---

## Incremental Pipeline (Advanced)

For faster updates, only download NEW fixtures:

### Create `scripts/incremental_update.py`:

```python
"""
Incremental update - only download and process new fixtures.
"""
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

def get_last_update_date():
    """Get date of last update from training data."""
    latest = Path('data/training_data_latest.csv')
    if not latest.exists():
        return None

    df = pd.read_csv(latest)
    df['match_date'] = pd.to_datetime(df['match_date'])
    return df['match_date'].max()

def incremental_update():
    """Update only new data."""

    last_date = get_last_update_date()

    if last_date:
        start_date = last_date + timedelta(days=1)
    else:
        # First run, download last 90 days
        start_date = datetime.now() - timedelta(days=90)

    end_date = datetime.now()

    print(f"Downloading from {start_date} to {end_date}")

    # Download only new fixtures
    # ... rest of pipeline
```

---

## Best Practices

### 1. Test Before Automating

```bash
# Test the pipeline manually first
python3 scripts/weekly_retrain_pipeline.py --weeks-back 1 --dry-run

# Then run for real
python3 scripts/weekly_retrain_pipeline.py --weeks-back 1
```

### 2. Set Reasonable Update Frequency

- **Weekly:** Good for most leagues (stable, not too frequent)
- **Bi-weekly:** For slower leagues or limited API quota
- **Daily:** Only if you need very fresh models (uses more API calls)

### 3. Monitor API Usage

SportMonks has rate limits. Track your usage:

```python
# Add to pipeline
def check_api_quota():
    import requests
    response = requests.get(
        'https://api.sportmonks.com/v3/my/quota',
        headers={'Authorization': f'Bearer {API_KEY}'}
    )
    print(f"API quota remaining: {response.json()}")
```

### 4. Version Control Models

Keep the last 3-5 model versions in case you need to rollback:

```bash
# The pipeline already does this!
ls -lht models/v4_model_*.joblib | head -5
```

### 5. Alert on Failures

Add to cron:

```bash
0 2 * * 0 cd /path && python3 scripts/weekly_retrain_pipeline.py || echo "Pipeline failed!" | mail -s "Retrain Alert" your@email.com
```

---

## Production Checklist

Before automating:

- [ ] API key is set: `echo $SPORTMONKS_API_KEY`
- [ ] Logs directory exists: `mkdir -p logs`
- [ ] Pipeline runs successfully manually
- [ ] Type mappings are correct (already fixed!)
- [ ] Cron/scheduler is configured
- [ ] Monitoring is set up
- [ ] Disk space is sufficient (models are ~100MB each)

---

## Troubleshooting

### Pipeline Fails

```bash
# Check logs
tail -100 logs/weekly_retrain.log

# Common issues:
# 1. API key not set
export SPORTMONKS_API_KEY="your_key"

# 2. No internet/API down
curl https://api.sportmonks.com/v3/football/leagues

# 3. Disk space full
df -h
```

### Model Quality Degrades

```bash
# Compare models
python3 -c "
import joblib
old = joblib.load('models/v4_model_20260131_140000.joblib')
new = joblib.load('models/v4_model_latest.joblib')
# Compare metrics...
"

# Rollback if needed
rm models/v4_model_latest.joblib
ln -s v4_model_20260131_140000.joblib models/v4_model_latest.joblib
```

---

## Summary

**For weekly retraining:**

1. Use `scripts/weekly_retrain_pipeline.py` (already created)
2. Set up cron/scheduler (see options above)
3. Monitor logs and model performance
4. The pipeline uses the FIXED type_id mappings automatically

**Key files:**
- Pipeline: `scripts/weekly_retrain_pipeline.py`
- Latest model: `models/v4_model_latest.joblib` (symlink)
- Latest data: `data/training_data_latest.csv` (symlink)
- Logs: `logs/weekly_retrain.log`

**The fix in `convert_json_to_csv.py` is permanent - all future runs will use correct type mappings!** ✅
