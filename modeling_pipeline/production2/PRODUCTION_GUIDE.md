# Production Deployment Guide

Complete guide to deploying the football prediction system in production.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRODUCTION SYSTEM                        │
└─────────────────────────────────────────────────────────────────┘

Weekly Training (Sundays 2 AM)
├─> Fetch new data
├─> Rebuild player database
├─> Retrain all models
├─> Validate performance
└─> Deploy if passing thresholds

Hourly Predictions (Every hour)
├─> Fetch upcoming fixtures (48h ahead)
├─> Generate predictions
├─> Apply betting strategy
├─> Store to database
└─> Check lineup updates

Lineup Updates (Every 15 min)
├─> Check matches <2h away
├─> Fetch lineups if available
├─> Re-predict with real player data
└─> Update database

Results Settlement (Every 30 min)
├─> Fetch finished match results
├─> Settle pending bets
├─> Calculate P/L
└─> Update performance metrics
```

## Prerequisites

### 1. System Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.8+
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 50GB minimum
- **CPU**: 4+ cores recommended

### 2. Software Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system tools
sudo apt-get install sqlite3 cron supervisor
```

### 3. API Keys

Required API keys (set as environment variables):

```bash
export SPORTMONKS_API_KEY="your_key_here"
export DATABASE_URL="sqlite:///predictions.db"  # or PostgreSQL URL
```

## Installation

### Step 1: Clone and Setup

```bash
# Navigate to project
cd /Users/ankurgupta/code/predictorV2/modeling_pipeline

# Create production directory
mkdir -p production
mkdir -p /var/log/football-predictions

# Set permissions
chmod +x production/*.py
```

### Step 2: Initial Model Training

```bash
# Run initial training to create baseline models
cd production
python weekly_training.py --dry-run  # Test first
python weekly_training.py            # Actual training
```

This will:
- Fetch historical data
- Build player database
- Train all models
- Create baseline deployment

**Expected time**: 2-4 hours

### Step 3: Database Setup

```bash
# Initialize prediction database
cd production
python -c "from hourly_predictions import PredictionDatabase; PredictionDatabase()"
```

This creates `predictions.db` with tables:
- `predictions` - All predictions made
- `bets` - Betting recommendations
- `results` - Actual match results
- `lineup_updates` - Lineup availability tracking

### Step 4: Configure Cron Jobs

```bash
# Edit crontab
crontab -e

# Add these lines (adjust paths):
0 2 * * 0 cd /path/to/production && python weekly_training.py >> /var/log/football-predictions/weekly.log 2>&1
0 * * * * cd /path/to/production && python hourly_predictions.py >> /var/log/football-predictions/hourly.log 2>&1
```

See `crontab.example` for complete configuration.

## Monitoring

### 1. Log Files

```bash
# View logs
tail -f /var/log/football-predictions/weekly_training.log
tail -f /var/log/football-predictions/hourly_predictions.log

# Check for errors
grep ERROR /var/log/football-predictions/*.log
```

### 2. Database Queries

```bash
# Check recent predictions
sqlite3 production/predictions.db "SELECT * FROM predictions ORDER BY prediction_time DESC LIMIT 10;"

# Check betting performance
sqlite3 production/predictions.db "
SELECT
  bet_outcome,
  COUNT(*) as total_bets,
  SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
  SUM(stake) as total_staked,
  SUM(profit_loss) as net_profit,
  (SUM(profit_loss) / SUM(stake) * 100) as roi
FROM bets
WHERE status = 'settled'
GROUP BY bet_outcome;
"
```

### 3. Health Checks

```bash
# Manual health check
cd production
python health_check.py

# Expected output:
# ✅ Models loaded successfully
# ✅ Database accessible
# ✅ API connection working
# ✅ Latest prediction: 2026-01-20 01:00:00
```

## Operational Procedures

### Weekly Training Procedure

**When**: Every Sunday 2 AM
**Duration**: 2-4 hours
**What it does**:

1. Fetches new match data from the past week
2. Rebuilds player database with latest stats
3. Retrains all 4 models (Elo, Dixon-Coles, XGBoost, Ensemble)
4. Validates performance against thresholds
5. Deploys if passing, rolls back if failing

**Performance Thresholds**:
- Log Loss < 1.05
- Accuracy > 50%
- Brier Score < 0.60

**Manual trigger** (if needed):
```bash
cd production
python weekly_training.py --force-deploy  # Skip validation
```

**Rollback** (if deployment fails):
```bash
# Models are automatically backed up to models/backup_TIMESTAMP/
# Restore manually if needed:
cp models/backup_20260120_020000/*.joblib models/
```

### Hourly Prediction Procedure

**When**: Every hour (top of the hour)
**Duration**: 5-10 minutes
**What it does**:

1. Fetches fixtures in next 48 hours
2. Generates predictions for each match
3. Applies betting strategy
4. Saves to database
5. Checks for lineup updates (if match <2h away)

**Manual trigger**:
```bash
cd production
python hourly_predictions.py --force
```

### Lineup Update Procedure

**When**: Every 15 minutes
**Duration**: 1-2 minutes
**What it does**:

1. Checks matches starting within 2 hours
2. Fetches lineups from API
3. Re-predicts with real player data if lineups available
4. Updates database with improved predictions

**Why important**: Predictions with real lineups are 1-3% more accurate

### Results Settlement

**When**: Every 30 minutes
**Duration**: 1-2 minutes
**What it does**:

1. Fetches results for finished matches
2. Settles all pending bets
3. Calculates profit/loss
4. Updates performance metrics

## Troubleshooting

### Issue: Weekly training fails

**Symptoms**: Models not updating, old predictions being used

**Check**:
```bash
# Check training log
tail -100 /var/log/football-predictions/weekly_training.log

# Check if models exist
ls -lh models/*.joblib

# Check last modified time
stat models/stacking_ensemble.joblib
```

**Fix**:
```bash
# Re-run manually with verbose logging
cd production
python weekly_training.py --force-deploy
```

### Issue: Hourly predictions not running

**Symptoms**: No new predictions in database

**Check**:
```bash
# Check cron is running
systemctl status cron

# Check cron logs
grep hourly /var/log/syslog

# Check last prediction
sqlite3 production/predictions.db "SELECT MAX(prediction_time) FROM predictions;"
```

**Fix**:
```bash
# Run manually to test
cd production
python hourly_predictions.py

# Check crontab is configured
crontab -l | grep hourly
```

### Issue: API rate limits

**Symptoms**: "429 Too Many Requests" errors

**Fix**:
```bash
# Reduce prediction frequency temporarily
# Edit crontab to run every 2 hours instead of hourly
0 */2 * * * cd /path/to/production && python hourly_predictions.py

# Or implement request throttling in code
```

### Issue: Database locked

**Symptoms**: "database is locked" errors

**Fix**:
```bash
# Check for long-running queries
sqlite3 production/predictions.db ".timeout 5000"

# Or migrate to PostgreSQL for better concurrency
export DATABASE_URL="postgresql://user:pass@localhost/predictions"
```

## Performance Optimization

### 1. Speed up predictions

```bash
# Use parallel processing for multiple fixtures
# Modify hourly_predictions.py to use multiprocessing

# Cache API responses
# Add Redis caching layer
```

### 2. Reduce API calls

```bash
# Batch fixture fetches
# Cache team stats for 1 hour
# Only fetch lineups for matches <2h away
```

### 3. Database optimization

```bash
# Add indexes
sqlite3 production/predictions.db "
CREATE INDEX idx_fixture_id ON predictions(fixture_id);
CREATE INDEX idx_prediction_time ON predictions(prediction_time);
CREATE INDEX idx_bets_status ON bets(status);
"

# Vacuum database monthly
sqlite3 production/predictions.db "VACUUM;"
```

## Scaling for Production

### Option 1: Single Server (Current Setup)

- **Cost**: $20-50/month (VPS)
- **Capacity**: ~1000 predictions/day
- **Use case**: Personal use, small betting operation

### Option 2: Containerized (Docker)

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "production/hourly_predictions.py"]
```

```bash
# Build and run
docker build -t football-predictor .
docker run -d --name predictor football-predictor
```

### Option 3: Cloud Deployment (AWS/GCP)

- **Weekly training**: AWS Batch / GCP Cloud Run Jobs
- **Hourly predictions**: Lambda / Cloud Functions
- **Database**: RDS PostgreSQL / Cloud SQL
- **Monitoring**: CloudWatch / Stackdriver

**Estimated cost**: $100-200/month

### Option 4: Kubernetes (High Scale)

- For handling 10,000+ predictions/day
- Auto-scaling based on fixture volume
- Multi-region deployment

## Backup and Disaster Recovery

### Daily Backups

```bash
# Add to crontab
0 4 * * * cd /path/to/production && python backup.py

# Backup script backs up:
# - prediction database
# - model files
# - configuration files
```

### Model Rollback

```bash
# List available backups
ls -lt models/backup_*/

# Restore specific backup
cp models/backup_20260120_020000/*.joblib models/
```

### Database Restore

```bash
# Restore from backup
sqlite3 predictions.db < backups/predictions_20260120.sql
```

## Cost Breakdown (Monthly)

| Component | Cost |
|-----------|------|
| VPS Server (4 vCPU, 16GB RAM) | $40 |
| Sportmonks API (Pro plan) | $40 |
| Database (Managed PostgreSQL) | $20 |
| Monitoring (Datadog/NewRelic) | $15 |
| Backups (S3 storage) | $5 |
| **Total** | **~$120/month** |

## Next Steps

1. ✅ Set up cron jobs
2. ✅ Test weekly training (dry-run first)
3. ✅ Test hourly predictions
4. ⏳ Monitor for 1 week
5. ⏳ Validate betting performance
6. ⏳ Scale based on usage

## Support

For issues or questions:
- Check logs first
- Review this guide
- Check GitHub issues
- Contact maintainers

---

**Last Updated**: 2026-01-20
**Version**: 1.0.0
