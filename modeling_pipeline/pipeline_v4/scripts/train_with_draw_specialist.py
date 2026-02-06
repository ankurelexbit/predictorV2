"""
Maximum Win Rate with Draw Specialist

Approach:
1. Main model for Home vs Away (binary)
2. Specialized Draw detector (binary)
3. Combine: If draw detector fires AND match is balanced, predict Draw
   Otherwise use H/A model
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    """Get features."""
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
    """Add features with focus on draw-relevant ones."""
    df = df.copy()
    new_features = []

    # === BALANCE/CLOSENESS FEATURES (Draw indicators) ===

    if 'elo_diff' in df.columns:
        df['elo_diff_sq'] = df['elo_diff'] ** 2 * np.sign(df['elo_diff'])
        df['elo_diff_abs'] = abs(df['elo_diff'])
        df['elo_closeness'] = 1 / (1 + abs(df['elo_diff']) / 30)  # More sensitive
        df['elo_very_close'] = (abs(df['elo_diff']) < 40).astype(int)
        df['elo_close'] = (abs(df['elo_diff']) < 80).astype(int)
        new_features.extend(['elo_diff_sq', 'elo_diff_abs', 'elo_closeness', 'elo_very_close', 'elo_close'])

    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
        df['away_elo_advantage'] = df['away_elo'] - df['home_elo']
        df['elo_sum'] = df['home_elo'] + df['away_elo']  # Match quality
        new_features.extend(['elo_ratio', 'away_elo_advantage', 'elo_sum'])

    if 'position_diff' in df.columns:
        df['position_closeness'] = 1 / (1 + abs(df['position_diff']))
        df['position_diff_abs'] = abs(df['position_diff'])
        df['position_very_close'] = (abs(df['position_diff']) <= 2).astype(int)
        new_features.extend(['position_closeness', 'position_diff_abs', 'position_very_close'])

    if 'points_diff' in df.columns:
        df['points_closeness'] = 1 / (1 + abs(df['points_diff']) / 3)
        df['points_diff_abs'] = abs(df['points_diff'])
        df['points_very_close'] = (abs(df['points_diff']) <= 3).astype(int)
        new_features.extend(['points_closeness', 'points_diff_abs', 'points_very_close'])

    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_closeness'] = 1 / (1 + abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating']) * 5)
        df['lineup_diff_abs'] = abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating'])
        df['lineup_very_close'] = (abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating']) < 0.3).astype(int)
        new_features.extend(['lineup_closeness', 'lineup_diff_abs', 'lineup_very_close'])

    # Combined balance score
    balance_cols = ['elo_closeness', 'position_closeness', 'lineup_closeness']
    available = [c for c in balance_cols if c in df.columns]
    if available:
        df['match_balance'] = df[available].mean(axis=1)
        new_features.append('match_balance')

    # Very balanced indicator
    very_close_cols = ['elo_very_close', 'position_very_close', 'lineup_very_close']
    available_vc = [c for c in very_close_cols if c in df.columns]
    if available_vc:
        df['very_balanced'] = df[available_vc].sum(axis=1)
        new_features.append('very_balanced')

    # === FORM SIMILARITY (Draw indicator) ===
    if 'home_win_rate_5' in df.columns and 'away_win_rate_5' in df.columns:
        df['form_similarity'] = 1 - abs(df['home_win_rate_5'] - df['away_win_rate_5'])
        new_features.append('form_similarity')

    # === Elo x Lineup interaction ===
    if 'elo_diff' in df.columns and 'lineup_rating_diff' in df.columns:
        df['elo_x_lineup'] = df['elo_diff'] * df['lineup_rating_diff']
        new_features.append('elo_x_lineup')

    return df, feature_cols + new_features


def train_ha_model(X_train, y_train, X_val, y_val):
    """Train Home vs Away model (excluding draws from training)."""
    # Filter to only H and A
    mask_train = y_train != 1  # Not draw
    mask_val = y_val != 1

    X_tr = X_train[mask_train]
    y_tr = (y_train[mask_train] == 2).astype(int)  # 1=Home, 0=Away

    X_v = X_val[mask_val]
    y_v = (y_val[mask_val] == 2).astype(int)

    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.03,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )
    model.fit(X_tr, y_tr, eval_set=(X_v, y_v), verbose=False)

    return model


def train_draw_detector(X_train, y_train, X_val, y_val):
    """Train binary Draw vs Not-Draw classifier."""
    y_train_draw = (y_train == 1).astype(int)
    y_val_draw = (y_val == 1).astype(int)

    # Use balanced class weights since draws are minority
    model = CatBoostClassifier(
        iterations=400,
        depth=5,
        learning_rate=0.02,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        auto_class_weights='Balanced'
    )
    model.fit(X_train, y_train_draw, eval_set=(X_val, y_val_draw), verbose=False)

    return model


def train_3way_model(X_train, y_train, X_val, y_val):
    """Train standard 3-way classifier."""
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        auto_class_weights='Balanced'
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    return model


def calibrate_binary(model, X_val, y_val_binary):
    """Calibrate binary model."""
    raw_probs = model.predict_proba(X_val)[:, 1]
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(raw_probs, y_val_binary)
    return cal


def calibrate_3way(model, X_val, y_val):
    """Calibrate 3-way model."""
    raw_probs = model.predict_proba(X_val)

    calibrators = {}
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
        calibrators[outcome].fit(raw_probs[:, idx], (y_val == idx).astype(int))

    return calibrators


def apply_3way_calibration(probs, calibrators):
    """Apply 3-way calibration."""
    cal_probs = np.zeros_like(probs)
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(probs[:, idx])

    row_sums = cal_probs.sum(axis=1, keepdims=True)
    return cal_probs / np.where(row_sums == 0, 1, row_sums)


def find_draw_threshold(draw_probs, y_val, min_wr=50):
    """Find threshold for draw detector to achieve target WR."""
    best_thresh = 0.5
    best_wr = 0
    best_count = 0

    for thresh in np.arange(0.25, 0.80, 0.02):
        mask = draw_probs >= thresh
        if mask.sum() >= 20:
            correct = ((y_val[mask] == 1)).sum()
            wr = correct / mask.sum() * 100

            if wr >= min_wr and mask.sum() > best_count:
                best_thresh = thresh
                best_wr = wr
                best_count = mask.sum()

    return best_thresh, best_wr, best_count


def combined_prediction(ha_probs, draw_probs, draw_threshold, balance_scores, balance_threshold=0.6):
    """
    Combine predictions:
    1. If draw detector prob >= threshold AND match is balanced, predict Draw
    2. Otherwise, use H/A model
    """
    predictions = []

    for i in range(len(ha_probs)):
        draw_p = draw_probs[i]
        home_p = ha_probs[i]
        balance = balance_scores[i] if balance_scores is not None else 0.5

        # Predict draw if: high draw probability AND balanced match
        if draw_p >= draw_threshold and balance >= balance_threshold:
            predictions.append(1)  # Draw
        elif home_p >= 0.5:
            predictions.append(2)  # Home
        else:
            predictions.append(0)  # Away

    return np.array(predictions)


def evaluate_predictions(y_true, predictions, title):
    """Evaluate predictions."""
    print(f"\n{'='*70}")
    print(title)
    print('='*70)

    acc = accuracy_score(y_true, predictions)
    print(f"\nOverall Accuracy: {acc*100:.1f}%")

    outcome_names = {0: 'Away', 1: 'Draw', 2: 'Home'}
    print(f"\n{'Outcome':<8} {'Predicted':>10} {'Correct':>10} {'Win Rate':>10} {'Actual':>10}")
    print("-"*55)

    results = {}
    for idx in [2, 0, 1]:  # Home, Away, Draw
        name = outcome_names[idx]
        pred_mask = predictions == idx
        actual_count = (y_true == idx).sum()

        if pred_mask.sum() > 0:
            correct = (y_true[pred_mask] == idx).sum()
            wr = correct / pred_mask.sum() * 100
        else:
            correct = 0
            wr = 0

        results[name] = {'predicted': pred_mask.sum(), 'correct': correct, 'win_rate': wr}
        print(f"{name:<8} {pred_mask.sum():>10} {correct:>10} {wr:>9.1f}% {actual_count:>10}")

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

    # Get balance score column
    balance_col = 'match_balance' if 'match_balance' in df.columns else None

    # Split
    train_end = int(len(df) * 0.65)
    val_end = int(len(df) * 0.80)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    df_test = df.iloc[val_end:].copy()

    balance_val = df.iloc[train_end:val_end][balance_col].values if balance_col else None
    balance_test = df.iloc[val_end:][balance_col].values if balance_col else None

    print(f"\nSplits: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    print(f"Test period: {df_test['match_date'].min().date()} to {df_test['match_date'].max().date()}")

    # ========================================
    # APPROACH 1: Standard 3-way model
    # ========================================
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)

    print("\n1. Training 3-way classifier...")
    model_3way = train_3way_model(X_train, y_train, X_val, y_val)
    cal_3way = calibrate_3way(model_3way, X_val, y_val)

    probs_3way_test = apply_3way_calibration(model_3way.predict_proba(X_test), cal_3way)
    preds_3way = probs_3way_test.argmax(axis=1)

    results_3way = evaluate_predictions(y_test.values, preds_3way, "APPROACH 1: Standard 3-Way Model")

    # ========================================
    # APPROACH 2: H/A model + Draw detector
    # ========================================
    print("\n2. Training Home/Away model...")
    model_ha = train_ha_model(X_train, y_train, X_val, y_val)

    print("3. Training Draw detector...")
    model_draw = train_draw_detector(X_train, y_train, X_val, y_val)

    # Calibrate
    y_val_draw = (y_val == 1).astype(int)
    cal_draw = calibrate_binary(model_draw, X_val, y_val_draw)

    # Get predictions on validation to tune threshold
    draw_probs_val = cal_draw.predict(model_draw.predict_proba(X_val)[:, 1])

    # Find draw threshold
    print("\n4. Finding optimal draw threshold...")
    for target_wr in [50, 45, 40, 35]:
        thresh, wr, count = find_draw_threshold(draw_probs_val, y_val.values, min_wr=target_wr)
        print(f"   Target WR>={target_wr}%: threshold={thresh:.2f}, achieved WR={wr:.1f}%, count={count}")

    # Use threshold targeting 40% WR (realistic)
    draw_threshold, _, _ = find_draw_threshold(draw_probs_val, y_val.values, min_wr=40)

    # Test predictions
    ha_probs_test = model_ha.predict_proba(X_test)[:, 1]  # P(Home)
    draw_probs_test = cal_draw.predict(model_draw.predict_proba(X_test)[:, 1])

    # Combined predictions
    preds_combined = combined_prediction(ha_probs_test, draw_probs_test, draw_threshold, balance_test)

    results_combined = evaluate_predictions(y_test.values, preds_combined,
                                            f"APPROACH 2: H/A Model + Draw Detector (thresh={draw_threshold:.2f})")

    # ========================================
    # APPROACH 3: Selective predictions
    # ========================================
    print("\n" + "="*70)
    print("APPROACH 3: SELECTIVE PREDICTIONS (High Confidence Only)")
    print("="*70)

    for conf_thresh in [0.50, 0.55, 0.60]:
        # Only predict when 3-way model is confident
        max_prob = probs_3way_test.max(axis=1)
        mask = max_prob >= conf_thresh

        if mask.sum() > 100:
            preds_sel = probs_3way_test[mask].argmax(axis=1)
            y_sel = y_test.values[mask]

            acc = accuracy_score(y_sel, preds_sel)

            print(f"\nConfidence >= {conf_thresh*100:.0f}%:")
            print(f"  Coverage: {mask.sum()} / {len(mask)} ({mask.sum()/len(mask)*100:.1f}%)")
            print(f"  Overall accuracy: {acc*100:.1f}%")

            for idx, name in [(2, 'Home'), (0, 'Away'), (1, 'Draw')]:
                pred_mask = preds_sel == idx
                if pred_mask.sum() > 0:
                    correct = (y_sel[pred_mask] == idx).sum()
                    wr = correct / pred_mask.sum() * 100
                    print(f"  {name}: {pred_mask.sum()} predictions, {wr:.1f}% WR")
                else:
                    print(f"  {name}: 0 predictions")

    # ========================================
    # APPROACH 4: Confidence-based with Draw override
    # ========================================
    print("\n" + "="*70)
    print("APPROACH 4: CONFIDENCE-BASED + DRAW OVERRIDE")
    print("="*70)

    # Strategy:
    # - For H/A: require high confidence (55%+)
    # - For Draw: use draw detector when match is very balanced

    very_balanced_test = df_test['very_balanced'].values if 'very_balanced' in df_test.columns else np.zeros(len(df_test))

    final_preds = []
    for i in range(len(X_test)):
        p_away, p_draw, p_home = probs_3way_test[i]
        draw_p = draw_probs_test[i]
        is_balanced = very_balanced_test[i] >= 2  # At least 2 "very close" indicators

        # Check for confident H/A prediction first
        if p_home >= 0.55:
            final_preds.append(2)
        elif p_away >= 0.55:
            final_preds.append(0)
        # If not confident on H/A, check draw
        elif draw_p >= 0.45 and is_balanced:
            final_preds.append(1)
        # Fall back to highest prob
        else:
            final_preds.append(probs_3way_test[i].argmax())

    final_preds = np.array(final_preds)
    results_final = evaluate_predictions(y_test.values, final_preds,
                                         "APPROACH 4: Confidence-based + Draw Override")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    print(f"\n{'Approach':<40} {'Overall':>10} {'Home WR':>10} {'Away WR':>10} {'Draw WR':>10}")
    print("-"*85)

    approaches = [
        ("1. Standard 3-Way", results_3way),
        ("2. H/A + Draw Detector", results_combined),
        ("4. Confidence + Draw Override", results_final)
    ]

    for name, res in approaches:
        overall = sum(r['correct'] for r in res.values()) / sum(r['predicted'] for r in res.values()) * 100
        home_wr = res['Home']['win_rate']
        away_wr = res['Away']['win_rate']
        draw_wr = res['Draw']['win_rate']
        print(f"{name:<40} {overall:>9.1f}% {home_wr:>9.1f}% {away_wr:>9.1f}% {draw_wr:>9.1f}%")

    # Save best model
    best_model_data = {
        'model_3way': model_3way,
        'calibrators_3way': cal_3way,
        'model_draw': model_draw,
        'calibrator_draw': cal_draw,
        'model_ha': model_ha,
        'features': all_features,
        'draw_threshold': draw_threshold,
        'trained_date': datetime.now().isoformat()
    }

    Path('models').mkdir(exist_ok=True)
    joblib.dump(best_model_data, 'models/draw_specialist_model.joblib')
    print(f"\n✓ Model saved to: models/draw_specialist_model.joblib")

    return best_model_data


if __name__ == '__main__':
    model_data = main()
