# How to Integrate Odds Fetcher into predict_live.py

## Quick Integration Guide

The `odds_fetcher.py` module has been created to fetch real-time betting odds from SportMonks API.

### Step 1: Import the Module

Add this import at the top of `predict_live.py`:

```python
from odds_fetcher import OddsFetcher
```

### Step 2: Initialize in __init__

Add to the `LiveFeatureCalculator.__init__` method:

```python
def __init__(self):
    # ... existing code ...
    self.odds_fetcher = OddsFetcher()  # Add this line
```

### Step 3: Replace Hardcoded Odds

Find this section in `build_features_for_match` (around line 1141):

```python
# OLD CODE (lines 1141-1149):
features.update({
    # Market features (betting odds - using neutral placeholders)
    'odds_home': 2.5,  # Neutral odds
    'odds_draw': 3.3,
    'odds_away': 2.5,
    'market_prob_home': 0.33,
    'market_prob_draw': 0.33,
    'market_prob_away': 0.33,
    'market_favorite': 0,
    'market_home_away_ratio': 1.0,
})
```

Replace with:

```python
# NEW CODE:
# Fetch real-time betting odds
logger.info("Fetching real-time betting odds...")
odds_data = self.odds_fetcher.get_odds(fixture_id) if fixture_id else self.odds_fetcher._get_neutral_odds()

features.update({
    # Market features (real-time betting odds)
    'odds_home': odds_data['odds_home'],
    'odds_draw': odds_data['odds_draw'],
    'odds_away': odds_data['odds_away'],
    'odds_total': odds_data['odds_total'],
    'odds_home_draw_ratio': odds_data['odds_home_draw_ratio'],
    'odds_home_away_ratio': odds_data['odds_home_away_ratio'],
    'market_home_away_ratio': odds_data['market_home_away_ratio'],
})
```

### Step 4: Test the Integration

```bash
# Test odds fetcher standalone
venv/bin/python odds_fetcher.py 19191683

# Test live predictions with real odds
venv/bin/python run_live_predictions.py
```

Look for log messages like:
```
✅ Fetched odds for fixture 12345: H=1.75, D=3.50, A=4.20
```

---

## Module Features

### OddsFetcher Class

```python
from odds_fetcher import OddsFetcher

fetcher = OddsFetcher()
odds = fetcher.get_odds(fixture_id=12345)
```

### Convenience Function

```python
from odds_fetcher import fetch_odds

odds = fetch_odds(12345)
print(odds['odds_home'])  # 1.75
```

### Returned Data

```python
{
    'odds_home': 1.75,
    'odds_draw': 3.50,
    'odds_away': 4.20,
    'odds_total': 9.45,
    'odds_home_draw_ratio': 0.50,
    'odds_home_away_ratio': 0.42,
    'market_home_away_ratio': 0.42
}
```

### Graceful Fallback

- If API key missing → neutral odds (2.5, 3.3, 2.5)
- If API fails → neutral odds
- If no odds found → neutral odds
- Logs warnings but continues

---

## Benefits

✅ **Fixes feature mismatch** - Training and live now use real odds  
✅ **Modular design** - Easy to test and maintain  
✅ **Graceful fallback** - Never breaks predictions  
✅ **Reusable** - Can be used in other scripts  

---

## Testing

```bash
# Test with a real fixture ID
venv/bin/python odds_fetcher.py 19191683

# Expected output:
# ✅ Fetched odds for fixture 19191683: H=2.10, D=3.40, A=3.50
# Odds:
#   Home: 2.10
#   Draw: 3.40
#   Away: 3.50
```

---

## Next Steps

1. Make the 3 code changes above in `predict_live.py`
2. Test with `venv/bin/python run_live_predictions.py`
3. Verify odds are being fetched in logs
4. Deploy!
