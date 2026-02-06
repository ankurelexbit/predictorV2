#!/usr/bin/env python3
"""
Compare Weighted vs Unweighted CatBoost Models
================================================

Trains two models on the same data/split:
  A) Current production weights  (Home=1.2, Draw=1.4, Away=1.1)
  B) No weights                  (all 1.0)

Compares on the shared test split:
  1. Log loss & accuracy
  2. Per-class precision / recall (especially Draw)
  3. Raw probability calibration error (binned MAE vs actual outcome rates)

Does NOT save anything to models/production/.  Results printed to stdout.

Usage:
    python3 scripts/compare_weight_vs_no_weight.py --data data/training_data.csv
    python3 scripts/compare_weight_vs_no_weight.py --data data/training_data.csv --n-trials 50
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import catboost as cb
import optuna
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix

# Silence optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_TO_EXCLUDE = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target',
    'home_team_name', 'away_team_name', 'state_id'
]

TRAIN_RATIO = 0.70
VAL_RATIO  = 0.15
SEED = 42


def load_and_split(data_path):
    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target']).sort_values('match_date').reset_index(drop=True)
    df['target'] = df['target'].astype(int)

    feature_cols = [c for c in df.columns if c not in FEATURES_TO_EXCLUDE]

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = df.iloc[:train_end][feature_cols],  df.iloc[:train_end]['target']
    X_val,   y_val   = df.iloc[train_end:val_end][feature_cols], df.iloc[train_end:val_end]['target']
    X_test,  y_test  = df.iloc[val_end:][feature_cols],  df.iloc[val_end:]['target']

    logger.info(f"Data: {len(df)} rows, {len(feature_cols)} features")
    logger.info(f"  Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    logger.info(f"  Test class dist — A:{(y_test==0).sum()} D:{(y_test==1).sum()} H:{(y_test==2).sum()}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_with_optuna(X_train, y_train, X_val, y_val, class_weights, n_trials):
    """Run Optuna search, return best model."""

    def objective(trial):
        params = {
            'iterations':           trial.suggest_int('iterations', 200, 800),
            'depth':                trial.suggest_int('depth', 4, 10),
            'learning_rate':        trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg':          trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count':         trial.suggest_int('border_count', 32, 255),
            'bagging_temperature':  trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength':      trial.suggest_float('random_strength', 0, 10),
            'loss_function':        'MultiClass',
            'class_weights':        class_weights,
            'random_seed':          SEED,
            'verbose':              False,
            'thread_count':         -1
        }
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        return log_loss(y_val, model.predict_proba(X_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Re-train best
    best = study.best_params
    best['class_weights']  = class_weights
    best['loss_function']  = 'MultiClass'
    best['random_seed']    = SEED
    best['verbose']        = False
    best['thread_count']   = -1

    model = cb.CatBoostClassifier(**best)
    model.fit(X_train, y_train, verbose=False)

    logger.info(f"  Best val log_loss = {study.best_value:.4f}  (iterations={best['iterations']}, depth={best['depth']})")
    return model


def calibration_mae(probs, actuals, bins=np.arange(0.10, 0.95, 0.05), min_per_bin=5):
    """Binned calibration MAE: mean |predicted_prob - actual_rate| per bin."""
    errors = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() < min_per_bin:
            continue
        errors.append(abs(probs[mask].mean() - actuals[mask].mean()))
    return np.mean(errors) if errors else float('nan')


def evaluate(label, model, X_test, y_test):
    """Full evaluation suite. Returns dict of results."""
    proba = model.predict_proba(X_test)   # columns: [Away=0, Draw=1, Home=2]
    preds = np.argmax(proba, axis=1)
    y     = y_test.values

    # --- 1. Aggregate metrics ---
    ll  = log_loss(y, proba)
    acc = accuracy_score(y, preds)

    # --- 2. Per-class precision / recall ---
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y, preds, average=None, labels=[0,1,2])  # A, D, H
    recall    = recall_score(y, preds, average=None, labels=[0,1,2])

    # --- 3. Calibration MAE per outcome ---
    cal_away = calibration_mae(proba[:, 0], (y == 0).astype(float))
    cal_draw = calibration_mae(proba[:, 1], (y == 1).astype(float))
    cal_home = calibration_mae(proba[:, 2], (y == 2).astype(float))

    # --- 4. Mean predicted prob vs actual rate (overall bias) ---
    bias_away = proba[:, 0].mean() - (y == 0).mean()
    bias_draw = proba[:, 1].mean() - (y == 1).mean()
    bias_home = proba[:, 2].mean() - (y == 2).mean()

    # --- 5. Prediction distribution ---
    dist = {cls: (preds == cls).sum() for cls in [0, 1, 2]}

    results = {
        'label': label,
        'log_loss': ll,
        'accuracy': acc,
        'precision': {'away': precision[0], 'draw': precision[1], 'home': precision[2]},
        'recall':    {'away': recall[0],    'draw': recall[1],    'home': recall[2]},
        'cal_mae':   {'away': cal_away,     'draw': cal_draw,     'home': cal_home},
        'bias':      {'away': bias_away,    'draw': bias_draw,    'home': bias_home},
        'pred_dist': dist,
    }
    return results


def print_comparison(weighted, unweighted, y_test):
    """Pretty-print side-by-side comparison."""
    sep = "=" * 78

    print(f"\n{sep}")
    print(f"  HEAD-TO-HEAD: WEIGHTED (H=1.2 D=1.4 A=1.1)  vs  UNWEIGHTED (all 1.0)")
    print(sep)

    # Actual test-set class distribution
    y = y_test.values
    print(f"\n  Actual test distribution:  Away={( y==0).sum()}  Draw={(y==1).sum()}  Home={(y==2).sum()}")

    # --- Log loss & accuracy ---
    print(f"\n  {'Metric':<22} {'Weighted':>12} {'Unweighted':>12} {'Winner':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")

    for key, fmt, lower_is_better in [('log_loss', '.4f', True), ('accuracy', '.4f', False)]:
        w, u = weighted[key], unweighted[key]
        if lower_is_better:
            winner = 'Weighted' if w < u else 'Unweighted' if u < w else 'Tie'
        else:
            winner = 'Weighted' if w > u else 'Unweighted' if u > w else 'Tie'
        print(f"  {key:<22} {w:>12{fmt}} {u:>12{fmt}} {winner:>10}")

    # --- Per-class table ---
    for metric_name, metric_key in [('Precision', 'precision'), ('Recall', 'recall')]:
        print(f"\n  {metric_name}:")
        print(f"    {'Outcome':<8} {'Weighted':>10} {'Unweighted':>10} {'Winner':>10}")
        print(f"    {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for cls in ['away', 'draw', 'home']:
            w = weighted[metric_key][cls]
            u = unweighted[metric_key][cls]
            winner = 'Weighted' if w > u else 'Unweighted' if u > w else 'Tie'
            print(f"    {cls:<8} {w:>10.3f} {u:>10.3f} {winner:>10}")

    # --- Calibration MAE (lower = better) ---
    print(f"\n  Calibration MAE (lower = better calibrated):")
    print(f"    {'Outcome':<8} {'Weighted':>10} {'Unweighted':>10} {'Winner':>10}")
    print(f"    {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for cls in ['away', 'draw', 'home']:
        w = weighted['cal_mae'][cls]
        u = unweighted['cal_mae'][cls]
        winner = 'Weighted' if w < u else 'Unweighted' if u < w else 'Tie'
        print(f"    {cls:<8} {w:>9.1%} {u:>9.1%} {winner:>10}")
    # Average cal MAE
    w_avg = np.mean([weighted['cal_mae'][c] for c in ['away','draw','home']])
    u_avg = np.mean([unweighted['cal_mae'][c] for c in ['away','draw','home']])
    winner = 'Weighted' if w_avg < u_avg else 'Unweighted' if u_avg < w_avg else 'Tie'
    print(f"    {'AVG':<8} {w_avg:>9.1%} {u_avg:>9.1%} {winner:>10}")

    # --- Probability bias ---
    print(f"\n  Mean-prob bias (mean predicted − actual rate; 0.0 = perfect):")
    print(f"    {'Outcome':<8} {'Weighted':>10} {'Unweighted':>10} {'Winner':>10}")
    print(f"    {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for cls in ['away', 'draw', 'home']:
        w = weighted['bias'][cls]
        u = unweighted['bias'][cls]
        winner = 'Weighted' if abs(w) < abs(u) else 'Unweighted' if abs(u) < abs(w) else 'Tie'
        print(f"    {cls:<8} {w:>+9.1%} {u:>+9.1%} {winner:>10}")

    # --- Prediction distribution ---
    print(f"\n  Prediction count distribution:")
    print(f"    {'Outcome':<8} {'Weighted':>10} {'Unweighted':>10}")
    print(f"    {'-'*8} {'-'*10} {'-'*10}")
    for cls, name in [(0,'Away'),(1,'Draw'),(2,'Home')]:
        print(f"    {name:<8} {weighted['pred_dist'][cls]:>10} {unweighted['pred_dist'][cls]:>10}")

    # --- Verdict ---
    print(f"\n{sep}")
    print("  VERDICT")
    print(sep)

    # Simple scoring: count wins across the key metrics
    # Key metrics: log_loss, draw recall, avg cal MAE, draw bias
    scores = {'Weighted': 0, 'Unweighted': 0}

    # log_loss (lower wins)
    if weighted['log_loss'] < unweighted['log_loss']:   scores['Weighted'] += 1
    elif unweighted['log_loss'] < weighted['log_loss']: scores['Unweighted'] += 1

    # draw recall (higher wins)
    if weighted['recall']['draw'] > unweighted['recall']['draw']:   scores['Weighted'] += 1
    elif unweighted['recall']['draw'] > weighted['recall']['draw']: scores['Unweighted'] += 1

    # avg cal MAE (lower wins)
    if w_avg < u_avg:   scores['Weighted'] += 1
    elif u_avg < w_avg: scores['Unweighted'] += 1

    # draw bias (lower abs wins)
    if abs(weighted['bias']['draw']) < abs(unweighted['bias']['draw']):   scores['Weighted'] += 1
    elif abs(unweighted['bias']['draw']) < abs(weighted['bias']['draw']): scores['Unweighted'] += 1

    print(f"\n  Scorecard (log_loss, draw recall, avg cal MAE, draw bias):")
    print(f"    Weighted:   {scores['Weighted']} / 4")
    print(f"    Unweighted: {scores['Unweighted']} / 4")

    if scores['Weighted'] > scores['Unweighted']:
        print(f"\n  → Keep class weights.  Calibration layer on top remains the right approach.")
    elif scores['Unweighted'] > scores['Weighted']:
        print(f"\n  → Drop class weights.  Retrain production with all weights = 1.0, then re-fit calibrators.")
    else:
        print(f"\n  → Tie — review per-metric detail above.  Draw recall is the tiebreaker for this use case.")

    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare weighted vs unweighted CatBoost')
    parser.add_argument('--data', type=str, required=True, help='Path to training_data.csv')
    parser.add_argument('--n-trials', type=int, default=30, help='Optuna trials per model (default 30)')
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split(Path(args.data))

    # --- Model A: weighted (production) ---
    logger.info("\n>>> Training WEIGHTED model (H=1.2, D=1.4, A=1.1) ...")
    weighted_model = train_with_optuna(X_train, y_train, X_val, y_val,
                                       class_weights=[1.1, 1.4, 1.2],  # [Away, Draw, Home]
                                       n_trials=args.n_trials)

    # --- Model B: unweighted ---
    logger.info("\n>>> Training UNWEIGHTED model (all 1.0) ...")
    unweighted_model = train_with_optuna(X_train, y_train, X_val, y_val,
                                         class_weights=[1.0, 1.0, 1.0],
                                         n_trials=args.n_trials)

    # --- Evaluate both ---
    logger.info("\n>>> Evaluating on test set ...")
    w_results  = evaluate("Weighted",   weighted_model,   X_test, y_test)
    u_results  = evaluate("Unweighted", unweighted_model, X_test, y_test)

    # --- Print comparison ---
    print_comparison(w_results, u_results, y_test)


if __name__ == '__main__':
    main()
