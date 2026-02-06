"""
Advanced Model Training: Multiple Approaches

1. Hyperparameter Tuning with Optuna
2. Goal Difference Regression
3. Ensemble of Outcome-Specific Models

Each approach is tested on Nov-Dec 2025 data with the same strategy.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import norm
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and prepare data"""
    df = pd.read_csv('data/training_data_with_market.csv')
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    # Targets
    df['target'] = df['result'].map({'H': 2, 'D': 1, 'A': 0})
    df['goal_diff'] = df['home_score'] - df['away_score']

    return df


def get_features(df, include_market_signals=False):
    """Get feature columns"""
    METADATA = ['fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
                'match_date', 'home_score', 'away_score', 'result', 'target', 'goal_diff']
    ODDS = ['home_best_odds', 'home_avg_odds', 'home_implied_prob',
            'draw_best_odds', 'draw_avg_odds', 'draw_implied_prob',
            'away_best_odds', 'away_avg_odds', 'away_implied_prob',
            'market_overround', 'market_home_prob_normalized',
            'market_draw_prob_normalized', 'market_away_prob_normalized',
            'ah_home_odds', 'ah_away_odds', 'over_2_5_best', 'over_2_5_avg',
            'under_2_5_best', 'under_2_5_avg']
    MARKET_SIGNALS = ['home_bookmaker_disagreement', 'home_sharp_vs_soft',
                      'draw_bookmaker_disagreement', 'draw_sharp_vs_soft',
                      'away_bookmaker_disagreement', 'away_sharp_vs_soft',
                      'ah_main_line', 'num_bookmakers']

    exclude = set(METADATA + ODDS)
    if not include_market_signals:
        exclude.update(MARKET_SIGNALS)

    return [c for c in df.columns if c not in exclude]


def split_data(df):
    """Chronological split"""
    train_end = df[df['match_date'] < '2025-10-01'].index.max()
    val_end = df[df['match_date'] < '2025-11-01'].index.max()

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ============================================================================
# APPROACH 1: HYPERPARAMETER TUNING
# ============================================================================

def train_tuned_model(train_df, val_df, features, n_trials=50):
    """Train model with Optuna hyperparameter tuning"""

    X_train = train_df[features].fillna(0)
    y_train = train_df['target']
    X_val = val_df[features].fillna(0)
    y_val = val_df['target']

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'bootstrap_type': 'Bernoulli',
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'MultiClass',
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)

        # Evaluate on validation
        probs = model.predict_proba(X_val)
        preds = probs.argmax(axis=1)
        accuracy = (preds == y_val).mean()

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best params: {study.best_params}")
    print(f"Best val accuracy: {study.best_value:.1%}")

    # Train final model with best params
    best_params = study.best_params
    best_params['random_seed'] = 42
    best_params['verbose'] = False
    best_params['loss_function'] = 'MultiClass'
    best_params['bootstrap_type'] = 'Bernoulli'

    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)

    return model, study.best_params


# ============================================================================
# APPROACH 2: GOAL DIFFERENCE REGRESSION
# ============================================================================

class GoalDifferenceModel:
    """
    Predict goal difference (continuous), derive H/D/A probabilities
    """

    def __init__(self):
        self.model = None
        self.residual_std = 1.3  # Default, will be learned

    def fit(self, X_train, y_train, X_val, y_val, tune=False):
        """
        y_train/y_val = goal difference (home - away)
        """
        if tune:
            # Quick tuning
            def objective(trial):
                params = {
                    'iterations': trial.suggest_int('iterations', 300, 800),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_seed': 42,
                    'verbose': False,
                    'loss_function': 'RMSE',
                }
                model = CatBoostRegressor(**params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)
                preds = model.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - preds) ** 2))
                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=30, show_progress_bar=True)

            best_params = study.best_params
            best_params['random_seed'] = 42
            best_params['verbose'] = False
            best_params['loss_function'] = 'RMSE'
        else:
            best_params = {
                'iterations': 500,
                'depth': 6,
                'learning_rate': 0.03,
                'random_seed': 42,
                'verbose': False,
                'loss_function': 'RMSE',
            }

        self.model = CatBoostRegressor(**best_params)
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        # Learn residual distribution
        val_preds = self.model.predict(X_val)
        residuals = y_val - val_preds
        self.residual_std = np.std(residuals)
        print(f"Residual std: {self.residual_std:.2f}")

    def predict_proba(self, X):
        """
        Derive P(Home), P(Draw), P(Away) from goal difference distribution
        """
        pred_mean = self.model.predict(X)
        probs = []

        for mean in pred_mean:
            # Using normal distribution
            # P(Home) = P(goal_diff > 0.5)
            # P(Away) = P(goal_diff < -0.5)
            # P(Draw) = P(-0.5 <= goal_diff <= 0.5)

            p_home = 1 - norm.cdf(0.5, loc=mean, scale=self.residual_std)
            p_away = norm.cdf(-0.5, loc=mean, scale=self.residual_std)
            p_draw = norm.cdf(0.5, loc=mean, scale=self.residual_std) - norm.cdf(-0.5, loc=mean, scale=self.residual_std)

            # Ensure valid probabilities
            p_home = max(0.01, min(0.98, p_home))
            p_away = max(0.01, min(0.98, p_away))
            p_draw = max(0.01, min(0.98, p_draw))

            # Normalize
            total = p_home + p_draw + p_away
            probs.append([p_away / total, p_draw / total, p_home / total])

        return np.array(probs)


# ============================================================================
# APPROACH 3: ENSEMBLE OF SPECIALISTS
# ============================================================================

class EnsembleSpecialists:
    """
    Three separate binary classifiers, each specialized for one outcome
    """

    def __init__(self):
        self.home_model = None
        self.draw_model = None
        self.away_model = None

    def fit(self, X_train, y_train, X_val, y_val, tune=False):
        """
        Train three binary classifiers
        """
        # Binary targets
        y_home_train = (y_train == 2).astype(int)
        y_draw_train = (y_train == 1).astype(int)
        y_away_train = (y_train == 0).astype(int)

        y_home_val = (y_val == 2).astype(int)
        y_draw_val = (y_val == 1).astype(int)
        y_away_val = (y_val == 0).astype(int)

        # Home model
        print("Training Home Specialist...")
        self.home_model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.03,
            scale_pos_weight=1.3,  # Home is ~44%
            random_seed=42, verbose=False
        )
        self.home_model.fit(X_train, y_home_train, eval_set=(X_val, y_home_val), verbose=False)

        # Draw model - needs more care
        print("Training Draw Specialist...")
        self.draw_model = CatBoostClassifier(
            iterations=800, depth=5, learning_rate=0.02,
            scale_pos_weight=3.0,  # Draw is ~25%
            random_seed=42, verbose=False
        )
        self.draw_model.fit(X_train, y_draw_train, eval_set=(X_val, y_draw_val), verbose=False)

        # Away model
        print("Training Away Specialist...")
        self.away_model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.03,
            scale_pos_weight=2.0,  # Away is ~31%
            random_seed=42, verbose=False
        )
        self.away_model.fit(X_train, y_away_train, eval_set=(X_val, y_away_val), verbose=False)

    def predict_proba(self, X):
        """
        Combine predictions from all three specialists
        """
        p_home = self.home_model.predict_proba(X)[:, 1]
        p_draw = self.draw_model.predict_proba(X)[:, 1]
        p_away = self.away_model.predict_proba(X)[:, 1]

        # Normalize
        total = p_home + p_draw + p_away
        probs = np.column_stack([
            p_away / total,
            p_draw / total,
            p_home / total
        ])

        return probs


# ============================================================================
# CALIBRATION
# ============================================================================

def calibrate(probs, y_val):
    """Train isotonic calibrators"""
    calibrators = {}
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal = IsotonicRegression(out_of_bounds='clip')
        y_binary = (y_val == idx).astype(int)
        cal.fit(probs[:, idx], y_binary)
        calibrators[outcome] = cal
    return calibrators


def apply_calibration(probs, calibrators):
    """Apply calibration"""
    cal_probs = np.zeros_like(probs)
    for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
        cal_probs[:, idx] = calibrators[outcome].predict(probs[:, idx])

    # Normalize
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    cal_probs = cal_probs / np.where(row_sums == 0, 1, row_sums)
    return cal_probs


# ============================================================================
# BACKTESTING
# ============================================================================

def backtest(cal_probs, test_df, name, min_odds=1.60, min_cal=0.40, min_ev=0.08):
    """Backtest with consistent strategy"""
    bets = []

    for i in range(len(test_df)):
        row = test_df.iloc[i]

        best_bet = None
        best_ev = -999

        for outcome, idx, result_code, odds_col, implied_col in [
            ('Home', 2, 'H', 'home_best_odds', 'home_implied_prob'),
            ('Away', 0, 'A', 'away_best_odds', 'away_implied_prob'),
            ('Draw', 1, 'D', 'draw_best_odds', 'draw_implied_prob'),
        ]:
            cal_prob = cal_probs[i, idx]
            odds = row[odds_col] if pd.notna(row[odds_col]) else 0
            implied = row[implied_col] if pd.notna(row[implied_col]) else 0.33

            # Draw has stricter requirements
            if outcome == 'Draw':
                min_odds_outcome = 3.0
                min_cal_outcome = 0.30
                min_ev_outcome = 0.15
            else:
                min_odds_outcome = min_odds
                min_cal_outcome = min_cal
                min_ev_outcome = min_ev

            if odds >= min_odds_outcome and cal_prob >= min_cal_outcome:
                edge = cal_prob - implied
                ev = cal_prob * odds - 1

                if edge >= 0.05 and ev >= min_ev_outcome:
                    if ev > best_ev:
                        best_ev = ev
                        best_bet = (outcome, result_code, odds, cal_prob, ev)

        if best_bet:
            outcome, result_code, odds, cal_prob, ev = best_bet
            won = row['result'] == result_code
            pnl = odds - 1 if won else -1
            bets.append({
                'month': row['match_date'].strftime('%Y-%m'),
                'outcome': outcome,
                'won': won,
                'pnl': pnl,
                'cal_prob': cal_prob,
                'ev': ev
            })

    if not bets:
        return None

    return pd.DataFrame(bets)


def print_results(bets_df, name):
    """Print formatted results"""
    if bets_df is None or len(bets_df) == 0:
        print(f"{name}: No bets")
        return

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    print(f"Total: {len(bets_df)} bets, {bets_df['won'].mean():.1%} WR, "
          f"${bets_df['pnl'].sum():.2f}, {bets_df['pnl'].sum()/len(bets_df):.1%} ROI")

    # By outcome
    for o in ['Home', 'Draw', 'Away']:
        om = bets_df[bets_df['outcome'] == o]
        if len(om) > 0:
            print(f"  {o}: {len(om)} bets, {om['won'].mean():.1%} WR, ${om['pnl'].sum():.2f}")

    # Monthly
    print("Monthly:")
    all_pass = True
    for month in sorted(bets_df['month'].unique()):
        mm = bets_df[bets_df['month'] == month]
        wr = mm['won'].mean()
        roi = mm['pnl'].sum() / len(mm)
        wr_ok = '✓' if wr >= 0.50 else '✗'
        roi_ok = '✓' if roi >= 0.10 else '✗'

        # Check H/D/A
        hda_ok = True
        hda_status = []
        for o in ['Home', 'Draw', 'Away']:
            om = mm[mm['outcome'] == o]
            if len(om) > 0:
                o_pnl = om['pnl'].sum()
                if o_pnl < 0:
                    hda_ok = False
                hda_status.append(f"{o[0]}:{'+' if o_pnl >= 0 else ''}{o_pnl:.1f}")

        if not (wr >= 0.50 and roi >= 0.10 and hda_ok):
            all_pass = False

        print(f"  {month}: {len(mm)} bets, {wr:.1%} WR {wr_ok}, "
              f"{roi:.1%} ROI {roi_ok}, {' '.join(hda_status)}")

    if all_pass:
        print("*** ALL CONSTRAINTS PASSED ***")

    return all_pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ADVANCED MODEL TRAINING")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data()
    features = get_features(df, include_market_signals=False)
    train_df, val_df, test_df = split_data(df)

    print(f"Features: {len(features)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train = train_df[features].fillna(0)
    y_train = train_df['target']
    y_train_gd = train_df['goal_diff']

    X_val = val_df[features].fillna(0)
    y_val = val_df['target']
    y_val_gd = val_df['goal_diff']

    X_test = test_df[features].fillna(0)

    results = {}

    # =========================================================================
    # APPROACH 1: Hyperparameter Tuned Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 1: HYPERPARAMETER TUNED MODEL")
    print("=" * 70)

    model_tuned, best_params = train_tuned_model(train_df, val_df, features, n_trials=30)

    probs_val = model_tuned.predict_proba(X_val)
    calibrators_tuned = calibrate(probs_val, y_val.values)

    probs_test = model_tuned.predict_proba(X_test)
    cal_probs_tuned = apply_calibration(probs_test, calibrators_tuned)

    bets_tuned = backtest(cal_probs_tuned, test_df, "Tuned Model")
    results['tuned'] = print_results(bets_tuned, "TUNED MODEL")

    # =========================================================================
    # APPROACH 2: Goal Difference Regression
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 2: GOAL DIFFERENCE REGRESSION")
    print("=" * 70)

    gd_model = GoalDifferenceModel()
    gd_model.fit(X_train, y_train_gd.values, X_val, y_val_gd.values, tune=True)

    probs_val_gd = gd_model.predict_proba(X_val)
    calibrators_gd = calibrate(probs_val_gd, y_val.values)

    probs_test_gd = gd_model.predict_proba(X_test)
    cal_probs_gd = apply_calibration(probs_test_gd, calibrators_gd)

    bets_gd = backtest(cal_probs_gd, test_df, "Goal Diff Model")
    results['goal_diff'] = print_results(bets_gd, "GOAL DIFFERENCE REGRESSION")

    # =========================================================================
    # APPROACH 3: Ensemble of Specialists
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 3: ENSEMBLE OF SPECIALISTS")
    print("=" * 70)

    ensemble = EnsembleSpecialists()
    ensemble.fit(X_train, y_train.values, X_val, y_val.values)

    probs_val_ens = ensemble.predict_proba(X_val)
    calibrators_ens = calibrate(probs_val_ens, y_val.values)

    probs_test_ens = ensemble.predict_proba(X_test)
    cal_probs_ens = apply_calibration(probs_test_ens, calibrators_ens)

    bets_ens = backtest(cal_probs_ens, test_df, "Ensemble")
    results['ensemble'] = print_results(bets_ens, "ENSEMBLE OF SPECIALISTS")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n| Approach | Passes All Constraints? |")
    print("|----------|------------------------|")
    for name, passed in results.items():
        status = "✓ YES" if passed else "✗ NO"
        print(f"| {name:20s} | {status:22s} |")

    # Save best model
    print("\nSaving models...")
    joblib.dump({
        'tuned_model': model_tuned,
        'tuned_calibrators': calibrators_tuned,
        'tuned_params': best_params,
        'gd_model': gd_model,
        'gd_calibrators': calibrators_gd,
        'ensemble': ensemble,
        'ensemble_calibrators': calibrators_ens,
        'features': features,
    }, 'models/advanced_models.joblib')
    print("Saved to: models/advanced_models.joblib")


if __name__ == '__main__':
    main()
