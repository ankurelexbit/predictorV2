# Production Deployment

This directory contains all scripts and configuration for running the football prediction system in production.

## Quick Start

### 1. Local Deployment (Cron-based)

```bash
# Setup
cd production
python weekly_training.py  # Initial training (2-4 hours)

# Configure cron
crontab -e
# Add lines from crontab.example

# Test
python hourly_predictions.py
```

### 2. Docker Deployment

```bash
# Set environment variables
export SPORTMONKS_API_KEY="your_key"
export DB_PASSWORD="secure_password"

# Start services
docker-compose up -d

# View logs
docker-compose logs -f hourly-predictor
```

### 3. Cloud Deployment (AWS/GCP)

See `PRODUCTION_GUIDE.md` for detailed instructions.

## Files

### Core Scripts

| File | Purpose | Schedule |
|------|---------|----------|
| `weekly_training.py` | Retrain models | Weekly (Sun 2 AM) |
| `hourly_predictions.py` | Generate predictions | Hourly |
| `settle_results.py` | Update bet outcomes | Every 30 min |

### Configuration

| File | Purpose |
|------|---------|
| `crontab.example` | Cron schedule template |
| `docker-compose.yml` | Docker orchestration |
| `PRODUCTION_GUIDE.md` | Complete deployment guide |

### Monitoring

| File | Purpose |
|------|---------|
| `health_check.py` | System health monitoring |
| `dashboard.py` | Web-based analytics |

## Architecture

```
Weekly Training (Sundays 2 AM)
└─> Fetches data → Trains models → Validates → Deploys

Hourly Predictions (Every hour)
└─> Fetches fixtures → Predicts → Applies strategy → Stores

Lineup Updates (Every 15 min)
└─> Checks matches <2h → Fetches lineups → Re-predicts

Results Settlement (Every 30 min)
└─> Fetches results → Settles bets → Calculates P/L
```

## Database Schema

**predictions**
- fixture_id, prediction_time, match_date
- home_team, away_team, league, venue
- home_win_prob, draw_prob, away_win_prob
- predicted_outcome, lineup_available
- model_version

**bets**
- fixture_id, prediction_id, bet_time
- bet_outcome, stake, odds, expected_value
- rule_applied, status, actual_outcome
- result, profit_loss

**results**
- fixture_id, match_date
- home_team, away_team
- home_score, away_score, actual_outcome

**lineup_updates**
- fixture_id, update_time
- lineup_available, home_players_found, away_players_found

## Performance Thresholds

Models must meet these criteria to deploy:

- **Log Loss**: < 1.05
- **Accuracy**: > 50%
- **Brier Score**: < 0.60

If validation fails, system automatically rolls back to previous models.

## Monitoring

### Check Logs

```bash
tail -f /var/log/football-predictions/hourly_predictions.log
```

### Query Database

```bash
# Recent predictions
sqlite3 predictions.db "SELECT * FROM predictions ORDER BY prediction_time DESC LIMIT 10"

# Betting ROI
sqlite3 predictions.db "
SELECT
  COUNT(*) as total_bets,
  SUM(profit_loss) as net_profit,
  SUM(stake) as total_staked,
  (SUM(profit_loss) / SUM(stake) * 100) as roi
FROM bets WHERE status = 'settled';
"
```

### Health Check

```bash
python health_check.py
```

Expected output:
```
✅ Models loaded
✅ Database accessible
✅ API connection OK
✅ Latest prediction: 2026-01-20 01:00:00
```

## Scaling

| Setup | Capacity | Cost/month |
|-------|----------|------------|
| **Local** | 1K predictions/day | $0 |
| **VPS** | 5K predictions/day | $40 |
| **Docker** | 10K predictions/day | $80 |
| **Cloud** | 50K+ predictions/day | $200+ |

## Troubleshooting

**Training fails**
```bash
python weekly_training.py --force-deploy
```

**Predictions not updating**
```bash
python hourly_predictions.py --force
```

**Database locked**
```bash
# Migrate to PostgreSQL
export DATABASE_URL="postgresql://user:pass@localhost/predictions"
```

## Backup

Models and database backed up automatically to `backups/` directory.

Restore:
```bash
cp models/backup_20260120/*.joblib models/
sqlite3 predictions.db < backups/predictions_20260120.sql
```

## Next Steps

1. Review `PRODUCTION_GUIDE.md`
2. Set up cron jobs or Docker
3. Monitor for 1 week
4. Validate betting performance
5. Scale as needed

## Support

Questions? Check:
1. `PRODUCTION_GUIDE.md` (comprehensive guide)
2. Logs in `/var/log/football-predictions/`
3. GitHub issues

---

**Version**: 1.0.0
**Last Updated**: 2026-01-20
