"""
Maximum Win Rate Model - All Outcomes (H/D/A)

Goal: Maximize prediction accuracy for Home, Draw, and Away
- No edge/ROI considerations
- Focus purely on when to predict each outcome
- Find optimal confidence thresholds per outcome
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and merge data."""
    print("Loading data...")

    base_df = pd.read_csv('data/training_data_with_market.csv')
    base_df['match_date'] = pd.to_datetime(base_df['match_date'])

    lineup_df = pd.read_csv('data/lineup_features_v2.csv')
    lineup_df['match_date'] = pd.to_datetime(lineup_df['match_date'])

    lineup_cols = [c for c in lineup_df.columns
                   if c not in ['match_date', 'home_team_id', 'away_team_id']]

    df = base_df.merge(lineup_df[lineup_cols], on='fixture_id', how='inner')
    df = df[df['match_date'] >= '2019-01-01'].copy()
    df = df.sort_values('match_date').reset_index(drop=True)

    print(f"  Loaded {len(df):,} fixtures")
    return df


def get_features(df):
    """Get features - exclude odds since we're not using them."""
    exclude_patterns = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result',
        'odds', 'implied', 'bookmaker', 'sharp', 'soft', 'disagreement',
        'market_home', 'market_away', 'market_draw', 'market_overround',
        'ah_', 'ou_', 'num_bookmakers', 'over_2_5', 'under_2_5',
    ]

    def should_exclude(col):
        return any(p.lower() in col.lower() for p in exclude_patterns)

    return [c for c in df.columns
            if not should_exclude(c) and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]


def engineer_features(df, feature_cols):
    """Add engineered features."""
    df = df.copy()
    new_features = []

    # Elo features
    if 'elo_diff' in df.columns:
        df['elo_diff_sq'] = df['elo_diff'] ** 2 * np.sign(df['elo_diff'])
        new_features.append('elo_diff_sq')

        df['elo_diff_abs'] = abs(df['elo_diff'])
        new_features.append('elo_diff_abs')

        # Closeness indicator (for draws)
        df['elo_closeness'] = 1 / (1 + abs(df['elo_diff']) / 50)
        new_features.append('elo_closeness')

        if 'lineup_rating_diff' in df.columns:
            df['elo_x_lineup'] = df['elo_diff'] * df['lineup_rating_diff']
            new_features.append('elo_x_lineup')

    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
        new_features.append('elo_ratio')

        df['away_elo_advantage'] = df['away_elo'] - df['home_elo']
        new_features.append('away_elo_advantage')

    # Position closeness
    if 'position_diff' in df.columns:
        df['position_closeness'] = 1 / (1 + abs(df['position_diff']))
        new_features.append('position_closeness')

        df['position_diff_abs'] = abs(df['position_diff'])
        new_features.append('position_diff_abs')

    # Points closeness
    if 'points_diff' in df.columns:
        df['points_closeness'] = 1 / (1 + abs(df['points_diff']) / 5)
        new_features.append('points_closeness')

    # Lineup features
    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_closeness'] = 1 / (1 + abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating']) * 10)
        new_features.append('lineup_closeness')

        df['lineup_diff_abs'] = abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating'])
        new_features.append('lineup_diff_abs')

    # Combined balance score (higher = more likely draw)
    balance_cols = ['elo_closeness', 'position_closeness', 'lineup_closeness']
    available = [c for c in balance_cols if c in df.columns]
    if available:
        df['match_balance'] = df[available].mean(axis=1)
        new_features.append('match_balance')

    return df, feature_cols + new_features


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model."""
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        eval_metric='Accuracy',  # Optimize for accuracy
        auto_class_weights='Balanced'  # Help with class imbalance
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def train_lgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model."""
    model = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lambda env: None])  # Suppress output
    return model


def calibrate_model(model, X_val, y_val):
    """Calibrate model probabilities."""
    raw_probs = model.predict_proba(X_val)

    calibrators = {}
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
        calibrators[outcome].fit(raw_probs[:, idx], (y_val == idx).astype(int))

    return calibrators


def apply_calibration(probs, calibrators):
    """Apply calibration."""
    cal_probs = np.zeros_like(probs)
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(probs[:, idx])

    row_sums = cal_probs.sum(axis=1, keepdims=True)
    return cal_probs / np.where(row_sums == 0, 1, row_sums)


def find_optimal_thresholds(y_true, probs, outcome_name, outcome_idx):
    """Find optimal probability threshold for maximum win rate."""
    best_wr = 0
    best_thresh = 0.33
    best_count = 0

    for thresh in np.arange(0.30, 0.75, 0.02):
        # Predictions where this outcome has prob >= threshold AND is highest
        mask = (probs[:, outcome_idx] >= thresh) & (probs.argmax(axis=1) == outcome_idx)

        if mask.sum() >= 20:  # Minimum sample
            preds_correct = (y_true[mask] == outcome_idx).sum()
            wr = preds_correct / mask.sum() * 100

            if wr > best_wr:
                best_wr = wr
                best_thresh = thresh
                best_count = mask.sum()

    return {'threshold': best_thresh, 'win_rate': best_wr, 'count': best_count}


def predict_with_thresholds(probs, thresholds):
    """Make predictions using per-outcome thresholds."""
    predictions = []
    confidences = []

    for i in range(len(probs)):
        p_away, p_draw, p_home = probs[i]

        # Check which outcomes meet their threshold
        candidates = []

        if p_home >= thresholds['home']['threshold']:
            candidates.append(('Home', 2, p_home))
        if p_away >= thresholds['away']['threshold']:
            candidates.append(('Away', 0, p_away))
        if p_draw >= thresholds['draw']['threshold']:
            candidates.append(('Draw', 1, p_draw))

        if candidates:
            # Pick highest probability among qualifying outcomes
            best = max(candidates, key=lambda x: x[2])
            predictions.append(best[1])
            confidences.append(best[2])
        else:
            # No outcome meets threshold - pick highest prob anyway
            idx = probs[i].argmax()
            predictions.append(idx)
            confidences.append(probs[i, idx])

    return np.array(predictions), np.array(confidences)


def evaluate_predictions(y_true, predictions, probs, title):
    """Evaluate prediction accuracy."""
    print(f"\n{'='*70}")
    print(title)
    print('='*70)

    # Overall accuracy
    acc = accuracy_score(y_true, predictions)
    print(f"\nOverall Accuracy: {acc*100:.1f}%")

    # Per-outcome accuracy
    outcome_names = ['Away', 'Draw', 'Home']
    print("\nPer-Outcome Performance:")
    print(f"{'Outcome':<10} {'Predicted':>10} {'Correct':>10} {'Win Rate':>10}")
    print("-"*45)

    results = {}
    for idx, name in enumerate(outcome_names):
        mask = predictions == idx
        if mask.sum() > 0:
            correct = (y_true[mask] == idx).sum()
            wr = correct / mask.sum() * 100
            results[name] = {'predicted': mask.sum(), 'correct': correct, 'win_rate': wr}
            print(f"{name:<10} {mask.sum():>10} {correct:>10} {wr:>9.1f}%")
        else:
            results[name] = {'predicted': 0, 'correct': 0, 'win_rate': 0}
            print(f"{name:<10} {0:>10} {0:>10} {'N/A':>10}")

    # Actual distribution
    print(f"\nActual Result Distribution:")
    for idx, name in enumerate(outcome_names):
        actual_count = (y_true == idx).sum()
        print(f"  {name}: {actual_count} ({actual_count/len(y_true)*100:.1f}%)")

    return results


def main():
    # Load data
    df = load_data()

    # Features
    feature_cols = get_features(df)
    df, all_features = engineer_features(df, feature_cols)
    print(f"Total features: {len(all_features)}")

    # Target
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y = df['result'].map(result_map)
    X = df[all_features].fillna(0)

    # Split
    train_end = int(len(df) * 0.65)
    val_end = int(len(df) * 0.80)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    df_test = df.iloc[val_end:].copy()

    print(f"\nSplits:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,} ({df_test['match_date'].min().date()} to {df_test['match_date'].max().date()})")

    # Train models
    print("\nTraining CatBoost...")
    model_cb = train_catboost(X_train, y_train, X_val, y_val)

    print("Training LightGBM...")
    model_lgbm = train_lgbm(X_train, y_train, X_val, y_val)

    # Calibrate
    print("Calibrating...")
    cal_cb = calibrate_model(model_cb, X_val, y_val)
    cal_lgbm = calibrate_model(model_lgbm, X_val, y_val)

    # Get probabilities
    probs_cb_val = apply_calibration(model_cb.predict_proba(X_val), cal_cb)
    probs_lgbm_val = apply_calibration(model_lgbm.predict_proba(X_val), cal_lgbm)

    probs_cb_test = apply_calibration(model_cb.predict_proba(X_test), cal_cb)
    probs_lgbm_test = apply_calibration(model_lgbm.predict_proba(X_test), cal_lgbm)

    # Ensemble: average probabilities
    probs_ensemble_val = (probs_cb_val + probs_lgbm_val) / 2
    probs_ensemble_test = (probs_cb_test + probs_lgbm_test) / 2

    # Find optimal thresholds on validation set
    print("\n" + "="*70)
    print("FINDING OPTIMAL THRESHOLDS (on validation set)")
    print("="*70)

    thresholds = {}
    for name, idx in [('home', 2), ('away', 0), ('draw', 1)]:
        opt = find_optimal_thresholds(y_val.values, probs_ensemble_val, name, idx)
        thresholds[name] = opt
        print(f"  {name.upper()}: threshold={opt['threshold']:.2f}, val_WR={opt['win_rate']:.1f}%, count={opt['count']}")

    # ========================================
    # TEST SET EVALUATION
    # ========================================

    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)

    # Method 1: Simple argmax (baseline)
    preds_simple = probs_ensemble_test.argmax(axis=1)
    results_simple = evaluate_predictions(y_test.values, preds_simple, probs_ensemble_test,
                                          "METHOD 1: Simple Argmax (Baseline)")

    # Method 2: With optimized thresholds
    preds_thresh, confs_thresh = predict_with_thresholds(probs_ensemble_test, thresholds)
    results_thresh = evaluate_predictions(y_test.values, preds_thresh, probs_ensemble_test,
                                          "METHOD 2: Optimized Thresholds")

    # Method 3: High confidence only
    high_conf_thresholds = {
        'home': {'threshold': 0.55},
        'away': {'threshold': 0.50},
        'draw': {'threshold': 0.38}
    }
    preds_high, confs_high = predict_with_thresholds(probs_ensemble_test, high_conf_thresholds)
    results_high = evaluate_predictions(y_test.values, preds_high, probs_ensemble_test,
                                        "METHOD 3: High Confidence Thresholds")

    # ========================================
    # SELECTIVE PREDICTIONS (Only predict when confident)
    # ========================================

    print("\n" + "="*70)
    print("SELECTIVE PREDICTIONS (Skip uncertain matches)")
    print("="*70)

    for min_conf in [0.45, 0.50, 0.55, 0.60]:
        # Only predict when max probability >= min_conf
        mask = probs_ensemble_test.max(axis=1) >= min_conf

        if mask.sum() > 100:
            preds_sel = probs_ensemble_test[mask].argmax(axis=1)
            y_sel = y_test.values[mask]

            acc = accuracy_score(y_sel, preds_sel)

            # Per outcome
            home_mask = preds_sel == 2
            away_mask = preds_sel == 0
            draw_mask = preds_sel == 1

            home_wr = (y_sel[home_mask] == 2).mean() * 100 if home_mask.sum() > 0 else 0
            away_wr = (y_sel[away_mask] == 0).mean() * 100 if away_mask.sum() > 0 else 0
            draw_wr = (y_sel[draw_mask] == 1).mean() * 100 if draw_mask.sum() > 0 else 0

            print(f"\nMin confidence >= {min_conf*100:.0f}%:")
            print(f"  Predictions: {mask.sum()} / {len(mask)} ({mask.sum()/len(mask)*100:.1f}%)")
            print(f"  Overall accuracy: {acc*100:.1f}%")
            print(f"  Home: {home_mask.sum()} preds, {home_wr:.1f}% WR")
            print(f"  Away: {away_mask.sum()} preds, {away_wr:.1f}% WR")
            print(f"  Draw: {draw_mask.sum()} preds, {draw_wr:.1f}% WR")

    # ========================================
    # MONTHLY BREAKDOWN
    # ========================================

    print("\n" + "="*70)
    print("MONTHLY WIN RATES (Optimized Thresholds)")
    print("="*70)

    df_test_copy = df_test.copy()
    df_test_copy['prediction'] = preds_thresh
    df_test_copy['actual'] = y_test.values
    df_test_copy['correct'] = (df_test_copy['prediction'] == df_test_copy['actual']).astype(int)
    df_test_copy['month'] = df_test_copy['match_date'].dt.strftime('%Y-%m')

    monthly = df_test_copy.groupby('month').agg({
        'correct': ['sum', 'count']
    })
    monthly.columns = ['correct', 'total']
    monthly['accuracy'] = monthly['correct'] / monthly['total'] * 100

    print(f"\n{'Month':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-"*40)

    acc50_months = 0
    for month, row in monthly.iterrows():
        status = "✓" if row['accuracy'] >= 50 else "✗"
        if row['accuracy'] >= 50:
            acc50_months += 1
        print(f"{month:<10} {row['correct']:>8.0f} {row['total']:>8.0f} {row['accuracy']:>9.1f}% {status}")

    print(f"\nMonths with accuracy >= 50%: {acc50_months}/{len(monthly)} ({acc50_months/len(monthly)*100:.0f}%)")

    # ========================================
    # SUMMARY
    # ========================================

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"""
Best Configuration: Ensemble (CatBoost + LightGBM) with Optimized Thresholds

Thresholds:
  Home: {thresholds['home']['threshold']:.2f}
  Away: {thresholds['away']['threshold']:.2f}
  Draw: {thresholds['draw']['threshold']:.2f}

Test Set Performance:
  Overall Accuracy: {accuracy_score(y_test.values, preds_thresh)*100:.1f}%

  Home Predictions: {results_thresh['Home']['predicted']} ({results_thresh['Home']['win_rate']:.1f}% WR)
  Away Predictions: {results_thresh['Away']['predicted']} ({results_thresh['Away']['win_rate']:.1f}% WR)
  Draw Predictions: {results_thresh['Draw']['predicted']} ({results_thresh['Draw']['win_rate']:.1f}% WR)
""")

    # Save model
    model_data = {
        'model_cb': model_cb,
        'model_lgbm': model_lgbm,
        'calibrators_cb': cal_cb,
        'calibrators_lgbm': cal_lgbm,
        'features': all_features,
        'thresholds': thresholds,
        'trained_date': datetime.now().isoformat()
    }

    Path('models').mkdir(exist_ok=True)
    joblib.dump(model_data, 'models/max_winrate_model.joblib')
    print(f"✓ Model saved to: models/max_winrate_model.joblib")

    # Save predictions for analysis
    pred_df = pd.DataFrame({
        'date': df_test['match_date'].values,
        'actual': y_test.values,
        'predicted': preds_thresh,
        'prob_away': probs_ensemble_test[:, 0],
        'prob_draw': probs_ensemble_test[:, 1],
        'prob_home': probs_ensemble_test[:, 2],
        'correct': (preds_thresh == y_test.values).astype(int)
    })
    pred_df.to_csv('data/predictions_max_winrate.csv', index=False)
    print(f"✓ Predictions saved to: data/predictions_max_winrate.csv")

    return model_cb, model_lgbm, thresholds, probs_ensemble_test


if __name__ == '__main__':
    model_cb, model_lgbm, thresholds, probs = main()
