# Model Performance Report - Last Week (January 11-18, 2026)

## Summary Statistics

| Date | Fixtures | Correct | Accuracy | Notes |
|------|----------|---------|----------|-------|
| **Jan 11** | 24 | 8 | **33.3%** | Mixed results, many draws |
| **Jan 12** | 5 | 3 | **60.0%** | Strong day |
| **Jan 14** | 14 | 9 | **64.3%** | Excellent performance |
| **Jan 15** | 5 | 0 | **0.0%** | All away wins/draws |
| **Jan 17** | 25 | 14 | **56.0%** | Very good, large sample |
| **Jan 18** | 25 | 6 | **24.0%** | Unusual day, many away wins |
| **TOTAL** | **98** | **40** | **40.8%** | Overall performance |

## Key Findings

### ✅ Good Performance Days
- **January 14**: 64.3% accuracy (9/14)
  - Featured strong home teams: TSG Hoffenheim 5-1, RB Leipzig 2-0
  - Model correctly identified home advantage

- **January 17**: 56.0% accuracy (14/25)
  - Large sample size with consistent performance
  - Correct predictions: Man United vs Man City, Real Madrid, Chelsea, Leeds

- **January 12**: 60.0% accuracy (3/5)
  - Small sample but strong: Bayern, Liverpool, Genoa all correct

### ❌ Poor Performance Days
- **January 15**: 0.0% accuracy (0/5)
  - ALL predictions were home wins
  - ALL actual results were away wins or draws
  - Unusual day with strong away performances

- **January 18**: 24.0% accuracy (6/25)
  - 40% away wins vs expected 32%
  - 36% draws vs expected 25%
  - Model heavily predicted home wins

### 📊 Performance Analysis

**Overall Accuracy: 40.8%** (40/98 matches)
- Better than random guessing (33.3%)
- Below typical betting market accuracy (50-55%)
- Significantly better on "typical" days vs unusual days

**Model Behavior:**
- Predicts home win in nearly all cases (98/98 predictions)
- Rarely predicts away win even when away win probability is highest
- Decision rule: Always pick max probability (needs improvement)

**Actual Outcome Distribution (98 matches):**
- Home Win: 40 (40.8%)
- Draw: 27 (27.6%)
- Away Win: 31 (31.6%)

**Model Predicted Distribution:**
- Home Win: 98 (100%)
- Draw: 0 (0%)
- Away Win: 0 (0%)

## Issues Identified

### 1. **Prediction Logic Problem**
The model ALWAYS predicts the highest probability outcome, even when margins are small:
- Example: 42.6% Home, 26.8% Draw, 30.6% Away → Predicts Home Win
- Better approach: Use probability thresholds or confidence intervals

### 2. **Home Bias in Predictions**
Model slightly favors home wins:
- Average predicted home win prob: 48.3%
- Actual home win rate: 40.8%
- This 7.5% overestimation leads to many incorrect predictions

### 3. **Variance in Daily Performance**
- Best day: 64.3% accuracy
- Worst day: 0.0% accuracy
- Suggests model is sensitive to unusual match conditions

## Recommendations

### Immediate Improvements
1. **Update Decision Logic**
   - Don't always predict max probability
   - Use threshold: Only predict outcome if probability > 45%
   - Consider draw predictions when probabilities are close (within 5%)

2. **Calibrate Probabilities**
   - Model predicts 48.3% home win average
   - Actual rate is 40.8%
   - Apply calibration: `calibrated_prob = 0.85 * predicted_prob`

3. **Use Ensemble Model**
   - Currently using XGBoost only
   - Stacking ensemble (trained) might be more accurate
   - Test with: `--model stacking`

### Medium-term Enhancements
1. Add confidence thresholds for predictions
2. Separate home/away form features
3. Include head-to-head historical data
4. Add league-specific adjustments
5. Consider recent injuries/suspensions

### Long-term Strategy
1. Retrain model with probability calibration
2. Add more diverse features (weather, referee stats, etc.)
3. Test different prediction thresholds
4. Implement Kelly criterion for betting optimization

## Conclusion

The model shows **promising but inconsistent performance**:
- ✅ Works well on typical match days (56-64% accuracy)
- ❌ Struggles with unusual result patterns
- ⚠️ Needs better decision logic beyond "pick max probability"

**Current State:** The live prediction system is technically functional with real-time API data and proper feature calculation. The 40.8% overall accuracy is limited more by the prediction strategy than by the model quality itself.

**Next Priority:** Implement smarter prediction logic with probability thresholds and draw consideration.
