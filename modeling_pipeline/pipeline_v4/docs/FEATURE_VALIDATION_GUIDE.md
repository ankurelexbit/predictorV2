# Feature Validation Script - Usage Guide

## Overview

The `validate_features.py` script performs comprehensive sanity checks on all 150 features in your training data.

## Quick Start

```bash
# Validate your training data
python3 scripts/validate_features.py --input data/csv/training_data_complete.csv
```

## What It Checks

### 1. **Missing Values** (18% weight)
- Counts missing values per feature
- Flags features with >10% missing (⚡) or >50% missing (⚠️)

### 2. **Range Validation** (33% weight)
- Validates features are within expected ranges
- Checks for impossible values (e.g., negative Elo, position > 25)

### 3. **Distribution Analysis** (10% weight)
- Detects zero variance features
- Identifies extreme skewness or kurtosis
- Flags constant-value features

### 4. **Logical Consistency** (67% weight - most critical)
- `elo_diff` should equal `home_elo - away_elo`
- `points_last_5` should equal `wins*3 + draws`
- `position_diff` should equal `home_position - away_position`
- `xgd` should equal `xg - xga`
- Binary features should only be 0 or 1

### 5. **Outlier Detection** (7% weight)
- Uses IQR method (3× IQR from Q1/Q3)
- Reports features with >5% outliers

### 6. **Correlation Analysis**
- Identifies highly correlated pairs (>0.95)
- Helps detect redundant features

## Output

### Console Output

```
================================================================================
VALIDATION SUMMARY
================================================================================
Total samples: 135
Total features: 150

Issues found:
  Features with missing values: 28
  Features with range violations: 15
  Features with distribution issues: 1
  Logical inconsistencies: 0
  Features with outliers: 10
  Highly correlated pairs: 64

🎯 Overall Health Score: 79.1/100
================================================================================
```

### Health Score Interpretation

| Score | Status | Meaning |
|-------|--------|---------|
| 90-100 | ✅ Excellent | Data quality is excellent, ready for production |
| 70-89 | ⚠️ Acceptable | Data quality is good with minor issues |
| 0-69 | ❌ Poor | Significant data quality issues detected |

### JSON Report

Detailed report saved to `feature_validation_report.json`:

```json
{
  "missing_values": {
    "home_attacks_per_match_5": {
      "count": 23,
      "percentage": 17.0
    }
  },
  "range_violations": {
    "home_ppda_5": {
      "expected": [0, 100],
      "actual": [0.5, 88.8],
      "violation_type": []
    }
  },
  "logical_inconsistencies": [],
  "outliers": {
    "home_clean_sheet_streak": {
      "count": 30,
      "percentage": 22.2,
      "bounds": [-3.0, 3.0],
      "extreme_values": {
        "min": 0.0,
        "max": 7.0
      }
    }
  },
  "summary": {
    "total_samples": 135,
    "total_features": 150,
    "health_score": 79.1
  }
}
```

## Command-Line Options

```bash
python3 scripts/validate_features.py \
    --input data/csv/training_data_complete.csv \
    --output validation_report.json \
    --log-file validation.log
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input CSV file with training data |
| `--output` | `feature_validation_report.json` | Output JSON report file |
| `--log-file` | `feature_validation.log` | Log file path |

## Common Issues & Fixes

### Missing Values

**Issue**: Some features have 10-20% missing values

**Cause**: Not all matches have complete statistics (e.g., attacks, dangerous attacks)

**Fix**: This is expected. Features default to 0.0 when data unavailable. Consider:
- Imputation strategies (mean, median, forward fill)
- Feature engineering to create "data_available" flags
- Using models that handle missing values (XGBoost, LightGBM)

### Range Violations

**Issue**: Some features exceed expected ranges

**Cause**: Conservative range definitions or unusual match statistics

**Fix**: 
- Review the specific violations in the JSON report
- Update expected ranges if violations are legitimate
- Investigate extreme values for data quality issues

### High Correlations

**Issue**: 64 highly correlated pairs detected

**Cause**: Some features are mathematically related:
- `home_elo` ≈ `home_lineup_quality_proxy` (both use Elo)
- `home_elo` ≈ `home_elo_vs_league_avg` (derived from same value)
- `home_points` ≈ `home_points_per_game` (mathematically related)

**Fix**:
- Use feature selection (e.g., remove one from each pair)
- Use PCA or feature extraction
- Let tree-based models handle it (they're robust to multicollinearity)

### Outliers

**Issue**: Binary features flagged as having outliers

**Cause**: IQR method not suitable for binary features (0/1)

**Fix**: This is expected for binary features. Focus on outliers in continuous features.

## Integration with Training Pipeline

### 1. Validate After Generation

```bash
# Generate training data
python3 scripts/generate_complete_training_data.py

# Validate it
python3 scripts/validate_features.py \
    --input data/csv/training_data_complete.csv
```

### 2. Automated Validation

```bash
#!/bin/bash
# generate_and_validate.sh

# Generate features
python3 scripts/generate_complete_training_data.py || exit 1

# Validate features
python3 scripts/validate_features.py \
    --input data/csv/training_data_complete.csv || exit 1

# If validation passes, proceed with training
python3 scripts/train_model.py
```

### 3. CI/CD Integration

```yaml
# .github/workflows/validate.yml
- name: Validate Features
  run: |
    python3 scripts/validate_features.py \
      --input data/csv/training_data_complete.csv
  # Exit code 0 = passed, 1 = failed
```

## Expected Results

For the test dataset (135 fixtures, January 2024):

| Metric | Value | Status |
|--------|-------|--------|
| **Missing Values** | 28 features | ⚡ Expected (some stats unavailable) |
| **Range Violations** | 15 features | ✅ Within tolerance |
| **Distribution Issues** | 1 feature | ✅ Minimal |
| **Logical Inconsistencies** | 0 | ✅ Perfect |
| **Outliers** | 10 features | ✅ Expected for binary features |
| **Health Score** | 79.1/100 | ⚠️ Acceptable |

## Tips

1. **Focus on Logical Inconsistencies**: These are the most critical issues
2. **Missing Values Are Expected**: Not all matches have complete statistics
3. **Outliers in Binary Features**: Ignore these (IQR method limitation)
4. **High Correlations**: Consider feature selection before training
5. **Health Score**: Aim for >70 for production use

## Troubleshooting

### Script Fails to Run

```bash
# Check Python version (requires 3.7+)
python3 --version

# Install dependencies
pip install pandas numpy scipy
```

### Memory Issues

```bash
# Process in chunks (modify script)
# Or increase available memory
```

### Unexpected Violations

```bash
# Review detailed JSON report
cat feature_validation_report.json | jq '.range_violations'

# Check specific feature
cat feature_validation_report.json | jq '.range_violations.home_ppda_5'
```

---

**Ready to validate your features!** 🎯
