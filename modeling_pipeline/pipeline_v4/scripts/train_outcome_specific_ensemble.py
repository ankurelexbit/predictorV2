"""
Ensemble of Outcome-Specific Models

Train 3 separate binary classifiers:
1. Home Model: Predicts P(Home win) vs P(Not Home)
2. Draw Model: Predicts P(Draw) vs P(Not Draw)
3. Away Model: Predicts P(Away win) vs P(Not Away)

Each model can learn outcome-specific patterns:
- Home model focuses on home advantage factors
- Draw model focuses on match balance indicators
- Away model focuses on away team strength signals

Final prediction: Ensemble the 3 models and select highest confidence.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and merge feature sets."""
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
    """Get pure statistical features (no odds)."""
    exclude_patterns = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result',
        'odds', 'implied', 'bookmaker', 'sharp', 'soft', 'disagreement',
        'market_home', 'market_away', 'market_draw', 'market_overround',
        'ah_', 'ou_', 'num_bookmakers', 'over_2_5', 'under_2_5',
    ]

    def should_exclude(col):
        return any(p.lower() in col.lower() for p in exclude_patterns)

    feature_cols = [c for c in df.columns
                    if not should_exclude(c)
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return feature_cols


def engineer_features(df, feature_cols):
    """Add engineered features."""
    df = df.copy()
    new_features = []

    # Elo interactions
    if 'elo_diff' in df.columns:
        df['elo_diff_sq'] = df['elo_diff'] ** 2 * np.sign(df['elo_diff'])
        new_features.append('elo_diff_sq')

        if 'lineup_rating_diff' in df.columns:
            df['elo_x_lineup'] = df['elo_diff'] * df['lineup_rating_diff']
            new_features.append('elo_x_lineup')

        if 'position_diff' in df.columns:
            df['elo_x_position'] = df['elo_diff'] * df['position_diff']
            new_features.append('elo_x_position')

    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
        new_features.append('elo_ratio')

    # Lineup quality
    if 'home_lineup_avg_rating' in df.columns and 'away_lineup_avg_rating' in df.columns:
        df['lineup_quality_gap'] = abs(df['home_lineup_avg_rating'] - df['away_lineup_avg_rating'])
        new_features.append('lineup_quality_gap')

        df['lineup_quality_sum'] = df['home_lineup_avg_rating'] + df['away_lineup_avg_rating']
        new_features.append('lineup_quality_sum')

    # Draw-specific features (balanced matches)
    if 'elo_diff' in df.columns:
        df['elo_closeness'] = 1 / (1 + abs(df['elo_diff']) / 100)  # Higher when teams are close
        new_features.append('elo_closeness')

    if 'position_diff' in df.columns:
        df['position_closeness'] = 1 / (1 + abs(df['position_diff']))
        new_features.append('position_closeness')

    print(f"  Added {len(new_features)} engineered features")
    return df, feature_cols + new_features


def select_features_for_outcome(X_train, y_train, feature_cols, outcome_name, n_features=100):
    """Select most relevant features for a specific outcome using mutual information."""
    print(f"  Selecting features for {outcome_name}...")

    # Calculate mutual information
    mi_scores = mutual_info_classif(X_train.fillna(0), y_train, random_state=42)

    # Create feature importance df
    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    # Select top features
    selected = mi_df.head(n_features)['feature'].tolist()

    print(f"    Top 5 features: {selected[:5]}")

    return selected


class OutcomeSpecificModel:
    """Binary classifier for a specific outcome."""

    def __init__(self, outcome_name, outcome_value):
        self.outcome_name = outcome_name
        self.outcome_value = outcome_value  # 0=Away, 1=Draw, 2=Home
        self.model = None
        self.calibrator = None
        self.features = None

    def train(self, X_train, y_train_multi, X_val, y_val_multi, feature_cols):
        """Train binary classifier for this outcome."""
        # Create binary target
        y_train = (y_train_multi == self.outcome_value).astype(int)
        y_val = (y_val_multi == self.outcome_value).astype(int)

        print(f"\n  Training {self.outcome_name} model...")
        print(f"    Positive class rate: {y_train.mean()*100:.1f}%")

        # Select features specific to this outcome
        self.features = select_features_for_outcome(
            X_train[feature_cols], y_train, feature_cols,
            self.outcome_name, n_features=80
        )

        X_train_sel = X_train[self.features].fillna(0)
        X_val_sel = X_val[self.features].fillna(0)

        # Train model with class weights for imbalanced data
        pos_weight = (1 - y_train.mean()) / y_train.mean()

        self.model = CatBoostClassifier(
            iterations=400,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=5,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=50,
            scale_pos_weight=pos_weight if self.outcome_name == 'Draw' else 1.0
        )

        self.model.fit(X_train_sel, y_train, eval_set=(X_val_sel, y_val), verbose=False)

        # Calibrate
        raw_probs = self.model.predict_proba(X_val_sel)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_probs, y_val)

        # Evaluate
        cal_probs = self.calibrator.predict(raw_probs)
        auc = roc_auc_score(y_val, cal_probs)

        # Find optimal threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.3, 0.7, 0.05):
            preds = (cal_probs >= thresh).astype(int)
            if preds.sum() > 0:
                prec = precision_score(y_val, preds)
                rec = recall_score(y_val, preds)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

        print(f"    Val AUC: {auc:.3f}, Best threshold: {best_thresh:.2f}")

        return auc

    def predict_proba(self, X):
        """Get calibrated probability for this outcome."""
        X_sel = X[self.features].fillna(0)
        raw_probs = self.model.predict_proba(X_sel)[:, 1]
        return self.calibrator.predict(raw_probs)


class OutcomeEnsemble:
    """Ensemble of outcome-specific models."""

    def __init__(self):
        self.models = {
            'away': OutcomeSpecificModel('Away', 0),
            'draw': OutcomeSpecificModel('Draw', 1),
            'home': OutcomeSpecificModel('Home', 2)
        }
        self.calibrators = {}  # For final ensemble calibration

    def train(self, X_train, y_train, X_val, y_val, feature_cols):
        """Train all outcome models."""
        print("\n" + "="*60)
        print("TRAINING OUTCOME-SPECIFIC MODELS")
        print("="*60)

        aucs = {}
        for name, model in self.models.items():
            auc = model.train(X_train, y_train, X_val, y_val, feature_cols)
            aucs[name] = auc

        # Ensemble calibration on validation set
        print("\n  Calibrating ensemble...")
        ensemble_probs = self._get_raw_ensemble_probs(X_val)

        for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
            self.calibrators[outcome] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators[outcome].fit(ensemble_probs[:, idx], (y_val == idx).astype(int))

        return aucs

    def _get_raw_ensemble_probs(self, X):
        """Get raw ensemble probabilities."""
        probs = np.zeros((len(X), 3))
        probs[:, 0] = self.models['away'].predict_proba(X)
        probs[:, 1] = self.models['draw'].predict_proba(X)
        probs[:, 2] = self.models['home'].predict_proba(X)

        # Normalize to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(row_sums == 0, 1, row_sums)

        return probs

    def predict_proba(self, X):
        """Get final calibrated ensemble probabilities."""
        raw_probs = self._get_raw_ensemble_probs(X)

        cal_probs = np.zeros_like(raw_probs)
        for outcome, idx in [('away', 0), ('draw', 1), ('home', 2)]:
            cal_probs[:, idx] = self.calibrators[outcome].predict(raw_probs[:, idx])

        # Renormalize
        row_sums = cal_probs.sum(axis=1, keepdims=True)
        cal_probs = cal_probs / np.where(row_sums == 0, 1, row_sums)

        return cal_probs


def backtest(df_test, probs, betting_params):
    """Backtest with given parameters."""
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

        h_implied = 1 / h_odds if h_odds > 1 else 0.5
        d_implied = 1 / d_odds if d_odds > 1 else 0.25
        a_implied = 1 / a_odds if a_odds > 1 else 0.35

        actual = result_map.get(row['result'], -1)
        if actual == -1:
            continue

        candidates = []

        # Home
        hp = betting_params.get('home', {})
        if hp.get('enabled', False):
            h_edge = p_home - h_implied
            if (hp['min_odds'] <= h_odds <= hp['max_odds'] and
                p_home >= hp['min_prob'] and h_edge >= hp['min_edge']):
                candidates.append({
                    'outcome': 'Home', 'idx': 2,
                    'odds': h_odds, 'prob': p_home, 'edge': h_edge
                })

        # Away
        ap = betting_params.get('away', {})
        if ap.get('enabled', False):
            a_edge = p_away - a_implied
            if (ap['min_odds'] <= a_odds <= ap['max_odds'] and
                p_away >= ap['min_prob'] and a_edge >= ap['min_edge']):
                candidates.append({
                    'outcome': 'Away', 'idx': 0,
                    'odds': a_odds, 'prob': p_away, 'edge': a_edge
                })

        # Draw
        dp = betting_params.get('draw', {})
        if dp.get('enabled', False):
            d_edge = p_draw - d_implied
            if (dp['min_odds'] <= d_odds <= dp['max_odds'] and
                p_draw >= dp['min_prob'] and d_edge >= dp['min_edge']):
                candidates.append({
                    'outcome': 'Draw', 'idx': 1,
                    'odds': d_odds, 'prob': p_draw, 'edge': d_edge
                })

        if not candidates:
            continue

        best = max(candidates, key=lambda x: x['edge'])
        won = 1 if actual == best['idx'] else 0
        pnl = (best['odds'] - 1) if won else -1

        results.append({
            'date': row['match_date'],
            'outcome': best['outcome'],
            'odds': best['odds'],
            'prob': best['prob'],
            'edge': best['edge'],
            'won': won,
            'pnl': pnl,
            'month': pd.to_datetime(row['match_date']).strftime('%Y-%m')
        })

    return pd.DataFrame(results)


def analyze_results(results_df, title):
    """Analyze backtest results."""
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

    if len(results_df) == 0:
        print("No bets placed!")
        return None

    total = len(results_df)
    wins = results_df['won'].sum()
    wr = wins / total * 100
    pnl = results_df['pnl'].sum()
    roi = pnl / total * 100

    print(f"\nOVERALL: {total} bets, {wr:.1f}% WR, {pnl:.2f} PnL, {roi:.1f}% ROI")

    # Monthly
    print("\nMONTHLY:")
    monthly = results_df.groupby('month').agg({
        'won': ['sum', 'count'],
        'pnl': 'sum'
    })
    monthly.columns = ['wins', 'bets', 'pnl']
    monthly['wr'] = monthly['wins'] / monthly['bets'] * 100
    monthly['roi'] = monthly['pnl'] / monthly['bets'] * 100

    wr50 = 0
    profitable = 0
    for month, row in monthly.iterrows():
        wr_ok = "✓" if row['wr'] >= 50 else "✗"
        roi_ok = "✓" if row['roi'] >= 0 else "✗"
        if row['wr'] >= 50: wr50 += 1
        if row['roi'] >= 0: profitable += 1
        print(f"  {month}: {row['bets']:.0f} bets, {row['wr']:.1f}% WR {wr_ok}, {row['roi']:.1f}% ROI {roi_ok}")

    print(f"\nMonths WR>=50%: {wr50}/{len(monthly)} ({wr50/len(monthly)*100:.0f}%)")
    print(f"Profitable months: {profitable}/{len(monthly)} ({profitable/len(monthly)*100:.0f}%)")

    # By outcome
    print("\nBY OUTCOME:")
    for outcome in ['Home', 'Away', 'Draw']:
        sub = results_df[results_df['outcome'] == outcome]
        if len(sub) > 0:
            o_wr = sub['won'].mean() * 100
            o_roi = sub['pnl'].sum() / len(sub) * 100
            print(f"  {outcome}: {len(sub)} bets, {o_wr:.1f}% WR, {o_roi:.1f}% ROI")

    return {'total': total, 'wr': wr, 'roi': roi, 'wr50_pct': wr50/len(monthly)*100}


def main():
    # Load data
    df = load_data()

    # Get and engineer features
    feature_cols = get_features(df)
    df, all_features = engineer_features(df, feature_cols)

    print(f"\nTotal features: {len(all_features)}")

    # Prepare target
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y = df['result'].map(result_map)
    X = df[all_features]

    # Chronological split
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

    # Train ensemble
    ensemble = OutcomeEnsemble()
    aucs = ensemble.train(X_train, y_train, X_val, y_val, all_features)

    # Get test predictions
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    probs = ensemble.predict_proba(X_test)

    # Overall accuracy
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"\nOverall accuracy: {acc:.1%}")

    # Per-outcome accuracy
    for outcome, idx in [('Away', 0), ('Draw', 1), ('Home', 2)]:
        mask = preds == idx
        if mask.sum() > 0:
            outcome_acc = (preds[mask] == y_test.values[mask]).mean()
            print(f"  {outcome} predictions: {mask.sum()} ({outcome_acc*100:.1f}% correct)")

    # Probability distribution
    print(f"\nProbability distribution:")
    print(f"  P(Home) mean: {probs[:, 2].mean():.3f}")
    print(f"  P(Draw) mean: {probs[:, 1].mean():.3f}")
    print(f"  P(Away) mean: {probs[:, 0].mean():.3f}")

    # ========================================
    # BACKTEST VARIOUS STRATEGIES
    # ========================================

    print("\n" + "="*70)
    print("BACKTESTING STRATEGIES")
    print("="*70)

    # Strategy 1: Home only (baseline comparison)
    params_home = {
        'home': {'enabled': True, 'min_prob': 0.52, 'min_edge': 0.06, 'min_odds': 1.40, 'max_odds': 3.50},
        'away': {'enabled': False},
        'draw': {'enabled': False}
    }
    results_home = backtest(df_test, probs, params_home)
    stats_home = analyze_results(results_home, "STRATEGY 1: HOME ONLY")

    # Strategy 2: Home + Away
    params_ha = {
        'home': {'enabled': True, 'min_prob': 0.52, 'min_edge': 0.06, 'min_odds': 1.40, 'max_odds': 3.50},
        'away': {'enabled': True, 'min_prob': 0.50, 'min_edge': 0.08, 'min_odds': 2.00, 'max_odds': 4.50},
        'draw': {'enabled': False}
    }
    results_ha = backtest(df_test, probs, params_ha)
    stats_ha = analyze_results(results_ha, "STRATEGY 2: HOME + AWAY")

    # Strategy 3: All outcomes with specialized thresholds
    params_all = {
        'home': {'enabled': True, 'min_prob': 0.52, 'min_edge': 0.06, 'min_odds': 1.40, 'max_odds': 3.50},
        'away': {'enabled': True, 'min_prob': 0.48, 'min_edge': 0.08, 'min_odds': 2.20, 'max_odds': 5.00},
        'draw': {'enabled': True, 'min_prob': 0.32, 'min_edge': 0.08, 'min_odds': 3.20, 'max_odds': 4.50}
    }
    results_all = backtest(df_test, probs, params_all)
    stats_all = analyze_results(results_all, "STRATEGY 3: ALL OUTCOMES")

    # Strategy 4: High confidence only
    params_hc = {
        'home': {'enabled': True, 'min_prob': 0.58, 'min_edge': 0.10, 'min_odds': 1.50, 'max_odds': 3.00},
        'away': {'enabled': True, 'min_prob': 0.55, 'min_edge': 0.12, 'min_odds': 2.50, 'max_odds': 4.50},
        'draw': {'enabled': False}
    }
    results_hc = backtest(df_test, probs, params_hc)
    stats_hc = analyze_results(results_hc, "STRATEGY 4: HIGH CONFIDENCE")

    # Summary comparison
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    print(f"{'Strategy':<25} {'Bets':>6} {'WR':>8} {'ROI':>8} {'WR50%':>8}")
    print("-"*60)

    for name, stats in [("Home Only", stats_home), ("Home + Away", stats_ha),
                        ("All Outcomes", stats_all), ("High Confidence", stats_hc)]:
        if stats:
            print(f"{name:<25} {stats['total']:>6} {stats['wr']:>7.1f}% {stats['roi']:>7.1f}% {stats['wr50_pct']:>7.0f}%")

    # Save best model
    best_strategy = max(
        [("Home Only", stats_home, params_home, results_home),
         ("Home + Away", stats_ha, params_ha, results_ha),
         ("All Outcomes", stats_all, params_all, results_all)],
        key=lambda x: x[1]['wr'] if x[1] else 0
    )

    print(f"\nBest strategy by WR: {best_strategy[0]}")

    # Save model
    model_data = {
        'ensemble': ensemble,
        'features': all_features,
        'betting_params': best_strategy[2],
        'trained_date': datetime.now().isoformat(),
        'performance': best_strategy[1]
    }

    Path('models').mkdir(exist_ok=True)
    joblib.dump(model_data, 'models/outcome_specific_ensemble.joblib')
    print(f"\n✓ Model saved to: models/outcome_specific_ensemble.joblib")

    best_strategy[3].to_csv('data/backtest_outcome_ensemble.csv', index=False)
    print(f"✓ Backtest results saved to: data/backtest_outcome_ensemble.csv")

    return ensemble, probs, df_test


if __name__ == '__main__':
    ensemble, probs, df_test = main()
