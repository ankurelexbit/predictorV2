# Live Prediction Approaches - Comparison Guide

## Two Approaches for Live Prediction

We provide two live prediction scripts for different deployment scenarios:

1. **`predict_live_v4.py`** - Uses historical CSV data (faster, requires setup)
2. **`predict_live_standalone.py`** ⭐ - Fetches everything from API (slower, production-ready)

---

## Approach 1: predict_live_v4.py (CSV-Based)

### How It Works

```
1. Relies on pre-downloaded historical data (data/historical/fixtures/)
2. Uses FeatureOrchestrator with existing data
3. Optionally downloads new fixtures
4. Fast feature generation (data already loaded)
```

### Pros ✅
- **Fast:** Data already loaded from CSV
- **Uses exact training pipeline:** Same FeatureOrchestrator
- **Consistent:** Same feature calculation as training
- **Efficient:** No repeated API calls for same data

### Cons ❌
- **Requires historical data:** Must maintain `data/historical/` directory
- **Not standalone:** Coupled with training pipeline
- **Storage:** Needs disk space for historical fixtures
- **Stale data risk:** Must update historical data regularly

### Use Cases
- Development/testing
- When you have training pipeline deployed
- When you want exact feature parity with training
- Local predictions with pre-downloaded data

### Usage
```bash
# Requires historical data directory
python3 scripts/predict_live_v4.py \
  --date today \
  --update-historical \
  --days-back 30
```

---

## Approach 2: predict_live_standalone.py (API-Only) ⭐

### How It Works

```
1. Fetches upcoming fixtures from API
2. For each fixture:
   - Fetches last 15 games for home team via API
   - Fetches last 15 games for away team via API
   - Calculates features on-the-fly
3. Makes predictions
4. Completely self-contained
```

### Pros ✅
- **Standalone:** No dependency on historical data directory
- **Always fresh:** Fetches latest data every run
- **Production-ready:** Deploy in container without data files
- **Scalable:** Each prediction is independent
- **Clean separation:** Training and prediction pipelines separate

### Cons ❌
- **Slower:** ~2-3 API calls per fixture (more for date ranges)
- **API limits:** Can hit rate limits with many fixtures
- **Network dependency:** Requires API access each run
- **Cost:** More API calls = higher usage

### Use Cases
- **Production deployment** ⭐
- Container/Docker deployment
- Lambda/serverless functions
- When training and prediction pipelines are separate
- When you don't want to maintain historical data

### Usage
```bash
# No setup needed - just API key
export SPORTMONKS_API_KEY="your_key"

python3 scripts/predict_live_standalone.py --date today
```

---

## Feature Comparison

| Feature | CSV-Based (v4) | API-Only (Standalone) |
|---------|----------------|----------------------|
| **Setup required** | ✅ Historical data download | ❌ None |
| **Speed** | ⚡ Fast | 🐢 Slower |
| **API calls per prediction** | 1 | 2-3 |
| **Storage** | 📦 Needs disk space | 💨 No storage |
| **Data freshness** | ⚠️ Must update manually | ✅ Always fresh |
| **Deployment** | 🔧 Complex | 🚀 Simple |
| **Feature accuracy** | ✅ Exact (same as training) | ~95% (some approximations) |
| **Dependency on training** | ✅ Coupled | ❌ Independent |
| **Container-friendly** | ❌ Needs data volume | ✅ Fully standalone |
| **Serverless-ready** | ❌ Not suitable | ✅ Perfect fit |

---

## Performance Comparison

### predict_live_v4.py (CSV-Based)

**For 10 fixtures:**
- Historical data load: 5 seconds (first time)
- Feature generation: 0.5 seconds per fixture
- **Total: ~10 seconds**

**API calls:** 10 (just upcoming fixtures)

### predict_live_standalone.py (API-Only)

**For 10 fixtures:**
- Fetch upcoming fixtures: 1 second
- Per fixture:
  - Fetch home team matches: 2 seconds
  - Fetch away team matches: 2 seconds
  - Calculate features: 0.1 seconds
- **Total: ~45 seconds**

**API calls:** 21 (1 for fixtures + 2 per fixture)

---

## API Usage Comparison

### For 10 upcoming fixtures:

**CSV-Based:**
```
Upcoming fixtures: 10 requests
Update historical data (optional): 150 requests (for 30 days)
────────────────────────────────────────
Total: 10 requests (without update)
       160 requests (with update)

Rate: ~10 requests if data is fresh
```

**API-Only:**
```
Upcoming fixtures: 1 request
Team matches (home): 10 requests
Team matches (away): 10 requests
────────────────────────────────────────
Total: 21 requests

Rate: Always 21 requests
```

**Winner:** API-Only (if you already have recent historical data)

---

## Which Should You Use?

### Use `predict_live_v4.py` If:
- ✅ You're also running the training pipeline
- ✅ You have storage for historical data
- ✅ You want exact feature parity with training
- ✅ You make predictions frequently (daily)
- ✅ You need maximum speed
- ✅ You're running on a server with disk storage

**Example:** Monolithic application where training and prediction run on same server

### Use `predict_live_standalone.py` If: ⭐
- ✅ Training and prediction are separate systems
- ✅ You want zero setup/dependencies
- ✅ You're deploying in containers/Docker
- ✅ You're using serverless (Lambda, Cloud Functions)
- ✅ You make predictions infrequently
- ✅ You want clean separation of concerns

**Example:** Microservices architecture where prediction is a standalone service

---

## Production Recommendations

### Scenario 1: Monolithic Deployment

**Use:** `predict_live_v4.py`

```bash
# Setup: Download historical data once
python3 scripts/backfill_historical_data.py --start-date 2024-01-01 --end-date 2024-12-31

# Daily cron: Update data + predict
0 6 * * * python3 scripts/predict_live_v4.py --date today --update-historical --days-back 7
```

**Why:** Faster, shares data with training pipeline

### Scenario 2: Microservices / Containers

**Use:** `predict_live_standalone.py` ⭐

```dockerfile
# Dockerfile
FROM python:3.9-slim
COPY scripts/predict_live_standalone.py /app/
COPY models/with_draw_features/conservative_with_draw_features.joblib /app/models/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
CMD ["python3", "predict_live_standalone.py", "--date", "today"]
```

**Why:** No data volumes needed, fully self-contained

### Scenario 3: Serverless (AWS Lambda)

**Use:** `predict_live_standalone.py` ⭐

```python
# lambda_handler.py
from predict_live_standalone import StandaloneLivePipeline

def lambda_handler(event, context):
    api_key = os.environ['SPORTMONKS_API_KEY']
    pipeline = StandaloneLivePipeline(api_key)
    pipeline.load_model('/opt/model.joblib')

    date = event['date']
    fixtures = pipeline.fetch_upcoming_fixtures(date, date)
    predictions = pipeline.make_predictions(fixtures)

    return predictions.to_dict('records')
```

**Why:** Lambda doesn't persist data, API-only is perfect

---

## Migration Path

### From CSV-Based to API-Only

**Step 1:** Test standalone version
```bash
python3 scripts/predict_live_standalone.py --date today --output test_predictions.csv
```

**Step 2:** Compare results with CSV-based version
```bash
python3 scripts/predict_live_v4.py --date today --output csv_predictions.csv

# Compare
python3 -c "
import pandas as pd
test = pd.read_csv('test_predictions.csv')
csv = pd.read_csv('csv_predictions.csv')
print((test['home_win_prob'] - csv['home_win_prob']).abs().mean())
"
```

**Step 3:** Deploy standalone version to production

**Step 4:** Remove dependency on historical data directory

---

## Feature Differences

### Features Identical in Both:
- Elo ratings
- Rolling averages (goals, xG, shots, possession)
- Form calculations (last 3/5/10 games)
- Draw parity features
- Attack/defense strength

### Features Approximated in Standalone:
- **Standings (position, points):** Estimated from Elo/form (API-only doesn't call standings endpoint)
- **H2H stats:** Default to 0 (would need additional API call)
- **Injuries:** Default to 0 (would need additional API call)
- **Odds:** Default values (would need betting odds API)
- **Player stats:** Approximated from team stats

**Impact:** ~1-2% difference in predictions due to approximations

**Solution if needed:** Add optional API calls for standings, H2H, injuries in standalone version

---

## Cost Analysis

### API Costs (SportMonks pricing)

**Assumptions:**
- 10 fixtures/day
- 30 days/month

**CSV-Based (with daily updates):**
```
Daily: 10 fixtures + 50 historical updates = 60 requests/day
Monthly: 60 × 30 = 1,800 requests/month
```

**API-Only:**
```
Daily: 1 + (10 × 2) = 21 requests/day
Monthly: 21 × 30 = 630 requests/month
```

**Winner:** API-Only (65% fewer requests!)

---

## Summary

### Use CSV-Based If:
- Speed is critical
- You have historical data anyway (for training)
- You want exact feature parity

### Use API-Only (Standalone) If: ⭐
- **Recommended for production**
- Clean separation from training pipeline
- Container/serverless deployment
- Don't want to maintain historical data
- Lower API usage
- Simpler deployment

---

## Final Recommendation

For **production deployment with separate training/prediction pipelines:**

✅ **Use `predict_live_standalone.py`**

**Why:**
1. Zero dependencies on training data
2. Simple Docker deployment
3. Fewer API calls than CSV approach (with updates)
4. Always fresh data
5. Scales horizontally
6. Perfect for microservices

**Example production setup:**
```bash
# Docker container
docker build -t prediction-service .
docker run -e SPORTMONKS_API_KEY=$API_KEY prediction-service --date today

# Kubernetes deployment
kubectl apply -f prediction-deployment.yaml

# AWS Lambda
sam deploy --template-file template.yaml
```
