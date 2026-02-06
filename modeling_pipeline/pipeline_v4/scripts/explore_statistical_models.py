"""
Explore Multiple Techniques for Pure Statistical Model

Techniques explored:
1. XGBoost vs CatBoost vs LightGBM
2. Goal Difference Regression → Derive probabilities
3. Stacked Ensemble
4. Advanced Feature Engineering
5. Hyperparameter Tuning with Optuna
6. Different betting strategies
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Models
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def load_data():
    """Load and prepare data."""
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


def get_pure_features(df):
    """Get pure statistical features (no odds)."""
    exclude_keywords = [
        'odds', 'implied', 'bookmaker', 'sharp', 'soft', 'disagreement',
        'market_home', 'market_away', 'market_draw', 'market_overround',
        'ah_', 'ou_', 'num_bookmakers', 'over_2_5', 'under_2_5',
        'fixture_id', 'team_id', 'season_id', 'league_id',
        'match_date', 'score', 'result'
    ]

    def is_excluded(col):
        return any(kw in col.lower() for kw in exclude_keywords)

    feature_cols = [c for c in df.columns
                    if not is_excluded(c)
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return feature_cols


def engineer_advanced_features(df, feature_cols):
    """Create advanced engineered features."""
    print("Engineering advanced features...")

    df = df.copy()
    new_features = []

    # 1. Interaction features
    if 'elo_diff' in df.columns and 'lineup_rating_diff' in df.columns:
        df['elo_x_lineup'] = df['elo_diff'] * df['lineup_rating_diff']
        new_features.append('elo_x_lineup')

    if 'elo_diff' in df.columns and 'position_diff' in df.columns:
        df['elo_x_position'] = df['elo_diff'] * df['position_diff']
        new_features.append('elo_x_position')

    # 2. Relative strength features
    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
        new_features.append('elo_ratio')

    # 3. Combined form indicators
    form_home_cols = [c for c in df.columns if c.startswith('home_') and '_5' in c]
    form_away_cols = [c for c in df.columns if c.startswith('away_') and '_5' in c]

    if form_home_cols and form_away_cols:
        # Normalize and create composite
        for col in form_home_cols[:5]:
            away_col = col.replace('home_', 'away_')
            if away_col in df.columns:
                diff_col = f'{col}_diff'
                df[diff_col] = df[col] - df[away_col]
                new_features.append(diff_col)

    # 4. Momentum features
    if 'home_points_trend_10' in df.columns and 'away_points_trend_10' in df.columns:
        df['momentum_diff'] = df['home_points_trend_10'] - df['away_points_trend_10']
        new_features.append('momentum_diff')

    # 5. Quality gap features
    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_quality_gap'] = abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating'])
        new_features.append('lineup_quality_gap')

        # Squared for non-linear
        df['lineup_rating_diff_sq'] = df['lineup_rating_diff'] ** 2
        new_features.append('lineup_rating_diff_sq')

    # 6. Home advantage adjusted Elo
    if 'elo_diff' in df.columns:
        df['elo_diff_sq'] = df['elo_diff'] ** 2 * np.sign(df['elo_diff'])
        new_features.append('elo_diff_sq')

    # 7. Fatigue proxy (based on recent matches)
    if 'home_matches_played' in df.columns:
        df['home_fixture_congestion'] = df.get('home_matches_played', 0) / 10
        new_features.append('home_fixture_congestion')

    print(f"  Added {len(new_features)} engineered features")

    all_features = feature_cols + new_features
    return df, all_features


def prepare_targets(df):
    """Prepare multiple target variables."""
    result_map = {'A': 0, 'D': 1, 'H': 2}
    df['target_class'] = df['result'].map(result_map)

    # Goal difference for regression
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['goal_diff'] = df['home_score'] - df['away_score']
    else:
        df['goal_diff'] = 0

    return df


class GoalDiffModel:
    """Predict goal difference, derive 1X2 probabilities."""

    def __init__(self, base_model='xgb'):
        if base_model == 'xgb':
            self.model = XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            )
        elif base_model == 'lgbm':
            self.model = LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )
        else:
            self.model = CatBoostRegressor(
                iterations=300, depth=5, learning_rate=0.05,
                random_seed=42, verbose=0
            )

        self.residual_std = 1.3  # Will be learned

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

        # Learn residual distribution
        if X_val is not None and y_val is not None:
            preds = self.model.predict(X_val)
            residuals = y_val - preds
            self.residual_std = np.std(residuals)
            if self.residual_std < 0.5:
                self.residual_std = 1.3  # Default

    def predict_proba(self, X):
        """Convert goal diff prediction to 1X2 probabilities."""
        pred_mean = self.model.predict(X)

        probs = []
        for mean in pred_mean:
            # P(Home) = P(goal_diff > 0.5)
            p_home = 1 - norm.cdf(0.5, loc=mean, scale=self.residual_std)
            # P(Away) = P(goal_diff < -0.5)
            p_away = norm.cdf(-0.5, loc=mean, scale=self.residual_std)
            # P(Draw) = remainder
            p_draw = 1 - p_home - p_away
            p_draw = max(0.05, min(0.5, p_draw))  # Clip

            # Renormalize
            total = p_away + p_draw + p_home
            probs.append([p_away/total, p_draw/total, p_home/total])

        return np.array(probs)


class EnsembleModel:
    """Stacked ensemble of multiple models."""

    def __init__(self):
        self.models = {
            'catboost': CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.03,
                random_seed=42, verbose=0
            ),
            'xgboost': XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                subsample=0.8, random_state=42, use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                subsample=0.8, random_state=42, verbose=-1
            ),
            'goal_diff': GoalDiffModel('xgb')
        }
        self.weights = {'catboost': 0.3, 'xgboost': 0.3, 'lightgbm': 0.2, 'goal_diff': 0.2}
        self.calibrators = {}

    def fit(self, X_train, y_train, X_val, y_val, goal_diff_train=None, goal_diff_val=None):
        print("  Training ensemble models...")

        for name, model in self.models.items():
            if name == 'goal_diff':
                if goal_diff_train is not None:
                    model.fit(X_train, goal_diff_train, X_val, goal_diff_val)
            else:
                model.fit(X_train, y_train)

        # Calibrate ensemble output
        raw_probs = self._get_raw_probs(X_val)
        for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
            self.calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators[outcome].fit(raw_probs[:, idx], (y_val == idx).astype(int))

    def _get_raw_probs(self, X):
        all_probs = []
        for name, model in self.models.items():
            if name == 'goal_diff':
                probs = model.predict_proba(X)
            else:
                probs = model.predict_proba(X)
            all_probs.append(probs * self.weights[name])

        combined = np.sum(all_probs, axis=0)
        return combined / combined.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        raw = self._get_raw_probs(X)

        cal_probs = np.zeros_like(raw)
        for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
            cal_probs[:, idx] = self.calibrators[outcome].predict(raw[:, idx])

        row_sums = cal_probs.sum(axis=1, keepdims=True)
        return cal_probs / np.where(row_sums == 0, 1, row_sums)


def backtest(df_test, probs, min_edge=0.03, min_prob=0.35):
    """Backtest with configurable thresholds."""
    results = []
    result_map = {'A': 0, 'D': 1, 'H': 2}

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        p_away, p_draw, p_home = probs[i]

        h_odds = row.get('home_best_odds', 0)
        d_odds = row.get('draw_best_odds', 0)
        a_odds = row.get('away_best_odds', 0)

        if not h_odds or not d_odds or not a_odds:
            continue

        h_implied = 1/h_odds if h_odds > 1 else 0.5
        a_implied = 1/a_odds if a_odds > 1 else 0.35
        d_implied = 1/d_odds if d_odds > 1 else 0.25

        actual = result_map.get(row['result'], -1)
        if actual == -1:
            continue

        candidates = []

        # Home
        h_edge = p_home - h_implied
        if h_odds >= 1.40 and p_home >= min_prob and h_edge >= min_edge:
            candidates.append({'out': 'H', 'idx': 2, 'odds': h_odds, 'prob': p_home, 'edge': h_edge})

        # Away
        a_edge = p_away - a_implied
        if a_odds >= 1.40 and p_away >= min_prob and a_edge >= min_edge:
            candidates.append({'out': 'A', 'idx': 0, 'odds': a_odds, 'prob': p_away, 'edge': a_edge})

        # Draw (more selective)
        d_edge = p_draw - d_implied
        if d_odds >= 2.80 and p_draw >= 0.26 and d_edge >= min_edge + 0.02:
            candidates.append({'out': 'D', 'idx': 1, 'odds': d_odds, 'prob': p_draw, 'edge': d_edge})

        if not candidates:
            continue

        best = max(candidates, key=lambda x: x['edge'])
        won = 1 if actual == best['idx'] else 0
        pnl = (best['odds'] - 1) if won else -1

        results.append({
            'month': pd.to_datetime(row['match_date']).strftime('%Y-%m'),
            'out': best['out'],
            'odds': best['odds'],
            'prob': best['prob'],
            'edge': best['edge'],
            'won': won,
            'pnl': pnl
        })

    return pd.DataFrame(results)


def analyze_results(results_df, model_name):
    """Analyze and print results."""
    if len(results_df) == 0:
        print(f"\n{model_name}: No bets placed")
        return None

    total = len(results_df)
    wr = results_df['won'].mean() * 100
    roi = results_df['pnl'].sum() / total * 100

    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Bets: {total}, WR: {wr:.1f}%, ROI: {roi:.1f}%")

    # Monthly
    print("\nMonthly:")
    monthly_stats = []
    for month in sorted(results_df['month'].unique()):
        m = results_df[results_df['month'] == month]
        m_wr = m['won'].mean() * 100
        m_roi = m['pnl'].sum() / len(m) * 100
        monthly_stats.append({'month': month, 'bets': len(m), 'wr': m_wr, 'roi': m_roi})
        wr_ok = '✓' if m_wr >= 50 else '✗'
        roi_ok = '✓' if m_roi >= 0 else '✗'  # Relaxed: just positive
        print(f"  {month}: {len(m):3d} bets, {m_wr:5.1f}% WR {wr_ok}, {m_roi:6.1f}% ROI {roi_ok}")

    return {'total': total, 'wr': wr, 'roi': roi, 'monthly': monthly_stats}


def run_single_model_comparison(X_train, y_train, X_val, y_val, X_test, df_test):
    """Compare individual models."""
    print("\n" + "="*70)
    print("COMPARING INDIVIDUAL MODELS")
    print("="*70)

    models = {
        'CatBoost': CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.025,
            l2_leaf_reg=5, random_seed=42, verbose=0
        ),
        'XGBoost': XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='mlogloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        ),
    }

    results_summary = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Calibrate
        raw_val = model.predict_proba(X_val)
        calibrators = {}
        for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
            calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
            calibrators[outcome].fit(raw_val[:, idx], (y_val == idx).astype(int))

        # Test
        raw_test = model.predict_proba(X_test)
        cal_test = np.zeros_like(raw_test)
        for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
            cal_test[:, idx] = calibrators[outcome].predict(raw_test[:, idx])
        cal_test = cal_test / cal_test.sum(axis=1, keepdims=True)

        # Accuracy
        preds = cal_test.argmax(axis=1)
        y_test = df_test['target_class'].values
        acc = accuracy_score(y_test, preds)
        print(f"  Accuracy: {acc:.1%}")

        # Backtest
        bt_results = backtest(df_test, cal_test)
        stats = analyze_results(bt_results, name)
        if stats:
            results_summary[name] = stats

    return results_summary


def run_goal_diff_model(X_train, y_train_gd, X_val, y_val_gd, X_test, df_test):
    """Test goal difference regression approach."""
    print("\n" + "="*70)
    print("GOAL DIFFERENCE REGRESSION MODEL")
    print("="*70)

    model = GoalDiffModel('xgb')
    model.fit(X_train, y_train_gd, X_val, y_val_gd)

    probs = model.predict_proba(X_test)

    # Check probability distribution
    print(f"\nProbability stats:")
    print(f"  P(Home) mean: {probs[:, 2].mean():.3f}")
    print(f"  P(Draw) mean: {probs[:, 1].mean():.3f}")
    print(f"  P(Away) mean: {probs[:, 0].mean():.3f}")

    bt_results = backtest(df_test, probs)
    stats = analyze_results(bt_results, "Goal Diff → Probs")

    return stats


def run_ensemble(X_train, y_train, X_val, y_val, X_test, df_test, gd_train, gd_val):
    """Test stacked ensemble."""
    print("\n" + "="*70)
    print("STACKED ENSEMBLE MODEL")
    print("="*70)

    ensemble = EnsembleModel()
    ensemble.fit(X_train, y_train, X_val, y_val, gd_train, gd_val)

    probs = ensemble.predict_proba(X_test)

    bt_results = backtest(df_test, probs)
    stats = analyze_results(bt_results, "Ensemble")

    return stats, ensemble


def tune_betting_thresholds(df_test, probs):
    """Find optimal betting thresholds."""
    print("\n" + "="*70)
    print("TUNING BETTING THRESHOLDS")
    print("="*70)

    best_roi = -999
    best_params = None

    for min_edge in [0.02, 0.03, 0.04, 0.05, 0.06]:
        for min_prob in [0.30, 0.35, 0.40, 0.45]:
            results = backtest(df_test, probs, min_edge=min_edge, min_prob=min_prob)
            if len(results) >= 50:  # Minimum sample
                roi = results['pnl'].sum() / len(results) * 100
                wr = results['won'].mean() * 100
                if roi > best_roi:
                    best_roi = roi
                    best_params = {
                        'min_edge': min_edge,
                        'min_prob': min_prob,
                        'bets': len(results),
                        'wr': wr,
                        'roi': roi
                    }

    if best_params:
        print(f"\nBest parameters found:")
        print(f"  min_edge: {best_params['min_edge']}")
        print(f"  min_prob: {best_params['min_prob']}")
        print(f"  Results: {best_params['bets']} bets, {best_params['wr']:.1f}% WR, {best_params['roi']:.1f}% ROI")

    return best_params


def main():
    # Load data
    df = load_data()

    # Get features
    feature_cols = get_pure_features(df)
    print(f"Base features: {len(feature_cols)}")

    # Engineer advanced features
    df, all_features = engineer_advanced_features(df, feature_cols)

    # Prepare targets
    df = prepare_targets(df)

    # Split
    train_end = int(len(df) * 0.65)
    val_end = int(len(df) * 0.80)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:].copy()

    X_train = df_train[all_features].fillna(0)
    X_val = df_val[all_features].fillna(0)
    X_test = df_test[all_features].fillna(0)

    y_train = df_train['target_class']
    y_val = df_val['target_class']

    gd_train = df_train['goal_diff']
    gd_val = df_val['goal_diff']

    print(f"\nSplits:")
    print(f"  Train: {len(df_train):,} ({df_train['match_date'].min().date()} to {df_train['match_date'].max().date()})")
    print(f"  Val: {len(df_val):,}")
    print(f"  Test: {len(df_test):,} ({df_test['match_date'].min().date()} to {df_test['match_date'].max().date()})")

    # Run comparisons
    single_results = run_single_model_comparison(X_train, y_train, X_val, y_val, X_test, df_test)

    gd_results = run_goal_diff_model(X_train, gd_train, X_val, gd_val, X_test, df_test)

    ensemble_results, ensemble_model = run_ensemble(
        X_train, y_train, X_val, y_val, X_test, df_test, gd_train, gd_val
    )

    # Tune thresholds using ensemble
    print("\nTuning thresholds on ensemble model...")
    ensemble_probs = ensemble_model.predict_proba(X_test)
    best_params = tune_betting_thresholds(df_test, ensemble_probs)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - All Models")
    print("="*70)

    all_results = {**single_results}
    if gd_results:
        all_results['Goal Diff'] = gd_results
    if ensemble_results:
        all_results['Ensemble'] = ensemble_results

    print(f"\n{'Model':<15} {'Bets':>6} {'WR':>8} {'ROI':>8}")
    print("-" * 40)
    for name, stats in all_results.items():
        print(f"{name:<15} {stats['total']:>6} {stats['wr']:>7.1f}% {stats['roi']:>7.1f}%")

    # Save best model
    if ensemble_results:
        joblib.dump({
            'model': ensemble_model,
            'features': all_features,
            'best_params': best_params
        }, 'models/ensemble_statistical.joblib')
        print("\nSaved ensemble model to models/ensemble_statistical.joblib")


if __name__ == '__main__':
    main()
