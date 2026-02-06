"""
Bet Selector Model
==================

ML model that decides whether to bet based on model probabilities + market features.
Supports LR (Logistic Regression) and GBM (LightGBM) model types.

Walk-forward validated on 13,500+ fixtures across 2017-2025.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .market_feature_extractor import MarketFeatureExtractor

logger = logging.getLogger(__name__)


class BetSelector:
    """ML-based bet selector using market features."""

    def __init__(self, min_confidence: float = 0.55, model_type: str = 'gbm'):
        self.min_confidence = min_confidence
        self.model_type = model_type
        self.model = None
        self.scaler = None  # Only used for LR
        self.feature_names = MarketFeatureExtractor.get_feature_names()

    def train(self, X: pd.DataFrame, y: np.ndarray, odds: np.ndarray = None):
        """Train the selector model with profit-weighted samples.

        Args:
            X: Feature DataFrame with columns matching get_feature_names().
            y: Binary target (1 = bet would have won, 0 = bet would have lost).
            odds: Odds for each sample (used for profit-based sample weighting).
        """
        X_features = X[self.feature_names].fillna(0)

        sample_weight = None
        if odds is not None:
            sample_weight = np.where(y == 1, np.clip(odds - 1, 0.1, 10.0), 1.0)

        if self.model_type == 'gbm':
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0, verbose=-1, random_state=42,
            )
            self.model.fit(X_features, y, sample_weight=sample_weight)
            train_acc = self.model.score(X_features, y)
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_features)
            self.model = LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1,
                class_weight='balanced', max_iter=2000, random_state=42,
            )
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
            train_acc = self.model.score(X_scaled, y)

        logger.info(f"Training accuracy ({self.model_type}): {train_acc:.3f}")

    def _prepare_input(self, features: dict) -> np.ndarray:
        """Prepare input for prediction."""
        X = pd.DataFrame([features])[self.feature_names].fillna(0)
        if self.model_type == 'gbm':
            return X
        return self.scaler.transform(X)

    def predict(self, features: dict) -> bool:
        """Predict whether to bet."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = self._prepare_input(features)
        prob = self.model.predict_proba(X)[0, 1]
        return prob >= self.min_confidence

    def predict_proba(self, features: dict) -> float:
        """Return P(bet wins)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = self._prepare_input(features)
        return float(self.model.predict_proba(X)[0, 1])

    def save(self, path: str):
        """Save model + scaler to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'min_confidence': self.min_confidence,
        }, path)
        logger.info(f"Saved bet selector ({self.model_type}, conf={self.min_confidence}) to {path}")

    def load(self, path: str):
        """Load model + scaler from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data.get('scaler')
        self.model_type = data.get('model_type', 'lr')
        self.feature_names = data['feature_names']
        self.min_confidence = data.get('min_confidence', 0.5)
        logger.info(f"Loaded bet selector ({self.model_type}, conf={self.min_confidence}) from {path}")

    @staticmethod
    def backtest(df: pd.DataFrame, n_folds: int = 7) -> dict:
        """Walk-forward backtest comparing multiple strategies.

        Strategies tested:
        - Threshold only (baseline)
        - LR selector at various confidence levels
        - Hybrid: threshold + LR selector as extra filter
        - GBM selector (shallow gradient boosting)

        Args:
            df: DataFrame with market features, model probs, actual_result,
                best_*_odds columns. Must be sorted by match_date.
            n_folds: Number of chronological folds.

        Returns:
            Dict with per-fold data, strategy comparison, and top features.
        """
        from lightgbm import LGBMClassifier

        feature_names = MarketFeatureExtractor.get_feature_names()

        # Determine predicted outcome and whether it won for each row
        df = df.copy()
        df['pred_outcome'] = df.apply(
            lambda r: _get_predicted_outcome(r['pred_home_prob'], r['pred_draw_prob'], r['pred_away_prob']),
            axis=1
        )
        df['pred_outcome_code'] = df['pred_outcome'].map({'home': 'H', 'draw': 'D', 'away': 'A'})
        df['bet_won'] = (df['pred_outcome_code'] == df['actual_result']).astype(int)
        df['pred_odds'] = df.apply(
            lambda r: r[f"best_{r['pred_outcome']}_odds"],
            axis=1
        )

        # Pre-compute threshold masks for all rows
        threshold_mask_all = _apply_thresholds(df)
        threshold_h55_mask_all = _apply_thresholds(df, thresholds={'home': 0.55, 'draw': 0.35, 'away': 0.45})
        # No-odds variants: thresholds only, GBM handles odds selection
        threshold_no_odds_mask_all = _apply_thresholds(df, odds_range=(0, 999))
        threshold_h55_no_odds_mask_all = _apply_thresholds(df, thresholds={'home': 0.55, 'draw': 0.35, 'away': 0.45}, odds_range=(0, 999))

        # Split into folds, collect per-row data
        fold_size = len(df) // n_folds
        all_rows = []  # List of dicts: {bet_won, pred_odds, lr_prob, gbm_prob, threshold_pass, fold}

        last_lr_model = None
        last_lr_scaler = None

        for fold_idx in range(1, n_folds):
            train_end = fold_idx * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, len(df))

            if test_end <= test_start:
                continue

            train_data = df.iloc[:train_end]
            test_data = df.iloc[test_start:test_end]

            X_train = train_data[feature_names].fillna(0)
            y_train = train_data['bet_won'].values
            odds_train = train_data['pred_odds'].values

            # Profit weighting
            sample_weight = np.where(y_train == 1, np.clip(odds_train - 1, 0.1, 10.0), 1.0)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            X_test = test_data[feature_names].fillna(0)
            X_test_scaled = scaler.transform(X_test)

            # --- LR model ---
            lr_model = LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1,
                class_weight='balanced', max_iter=2000, random_state=42,
            )
            lr_model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
            lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

            last_lr_model = lr_model
            last_lr_scaler = scaler

            # --- GBM model (shallow, conservative) ---
            gbm_model = LGBMClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0, verbose=-1, random_state=42,
            )
            gbm_model.fit(X_train, y_train, sample_weight=sample_weight)
            gbm_probs = gbm_model.predict_proba(X_test)[:, 1]

            # Store per-row data
            test_indices = test_data.index
            for i, idx in enumerate(test_indices):
                all_rows.append({
                    'bet_won': test_data.loc[idx, 'bet_won'],
                    'pred_odds': test_data.loc[idx, 'pred_odds'],
                    'lr_prob': lr_probs[i],
                    'gbm_prob': gbm_probs[i],
                    'threshold_pass': threshold_mask_all.loc[idx],
                    'threshold_h55_pass': threshold_h55_mask_all.loc[idx],
                    'threshold_no_odds_pass': threshold_no_odds_mask_all.loc[idx],
                    'threshold_h55_no_odds_pass': threshold_h55_no_odds_mask_all.loc[idx],
                    'fold': fold_idx,
                    'match_date': test_data.loc[idx, 'match_date'],
                })

        if not all_rows:
            return {'strategies': {}, 'top_features': [], 'fold_details': []}

        rows_df = pd.DataFrame(all_rows)

        # --- Compute strategy results ---
        def compute_pnl(mask):
            subset = rows_df[mask]
            if len(subset) == 0:
                return {'bets': 0, 'wins': 0, 'wr': 0, 'profit': 0, 'roi': 0}
            wins = 0
            profit = 0.0
            for _, r in subset.iterrows():
                if r['pred_odds'] <= 0:
                    continue
                if r['bet_won'] == 1:
                    profit += r['pred_odds'] - 1
                    wins += 1
                else:
                    profit -= 1
            n = len(subset)
            return {
                'bets': n, 'wins': wins,
                'wr': wins / n * 100 if n > 0 else 0,
                'profit': round(profit, 2),
                'roi': profit / n * 100 if n > 0 else 0,
            }

        strategies = {}

        # 1. Threshold only baselines
        strategies['threshold_H60'] = compute_pnl(rows_df['threshold_pass'])
        strategies['threshold_H55'] = compute_pnl(rows_df['threshold_h55_pass'])

        # 2. GBM selector standalone
        for conf in [0.45, 0.50, 0.55, 0.60, 0.65]:
            strategies[f'gbm_{conf:.2f}'] = compute_pnl(rows_df['gbm_prob'] >= conf)

        # 3. Hybrid H60: threshold(H60/D35/A45) + GBM
        for conf in [0.40, 0.45, 0.50, 0.55, 0.60]:
            mask = rows_df['threshold_pass'] & (rows_df['gbm_prob'] >= conf)
            strategies[f'H60+gbm_{conf:.2f}'] = compute_pnl(mask)

        # 4. Hybrid H55: threshold(H55/D35/A45) + GBM (with odds filter)
        for conf in [0.40, 0.45, 0.50, 0.55, 0.60]:
            mask = rows_df['threshold_h55_pass'] & (rows_df['gbm_prob'] >= conf)
            strategies[f'H55+gbm_{conf:.2f}'] = compute_pnl(mask)

        # 5. No odds filter: threshold + GBM (GBM decides on odds)
        for conf in [0.45, 0.50, 0.55, 0.60]:
            mask = rows_df['threshold_no_odds_pass'] & (rows_df['gbm_prob'] >= conf)
            strategies[f'H60_noOF+gbm_{conf:.2f}'] = compute_pnl(mask)

        # 6. H55 + no odds filter + GBM
        for conf in [0.45, 0.50, 0.55, 0.60]:
            mask = rows_df['threshold_h55_no_odds_pass'] & (rows_df['gbm_prob'] >= conf)
            strategies[f'H55_noOF+gbm_{conf:.2f}'] = compute_pnl(mask)

        # Feature importance (LR coefficients from last fold)
        top_features = []
        if last_lr_model is not None:
            coefs = dict(zip(feature_names, last_lr_model.coef_[0]))
            top_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        # Per-fold breakdown for key strategies
        key_strategies = ['threshold_H60', 'gbm_0.55',
                          'H60+gbm_0.55', 'H55+gbm_0.55',
                          'H60_noOF+gbm_0.55', 'H55_noOF+gbm_0.55',
                          'H55_noOF+gbm_0.50', 'H55_noOF+gbm_0.60']
        fold_details = {}

        for strat_name in key_strategies:
            fold_details[strat_name] = []
            for fold_idx in sorted(rows_df['fold'].unique()):
                fold_data = rows_df[rows_df['fold'] == fold_idx]

                mask = _resolve_strategy_mask(strat_name, fold_data)
                if mask is None:
                    continue

                subset = fold_data[mask]
                if len(subset) == 0:
                    result = {'bets': 0, 'wins': 0, 'wr': 0, 'profit': 0, 'roi': 0}
                else:
                    wins = 0
                    profit = 0.0
                    for _, r in subset.iterrows():
                        if r['pred_odds'] <= 0:
                            continue
                        if r['bet_won'] == 1:
                            profit += r['pred_odds'] - 1
                            wins += 1
                        else:
                            profit -= 1
                    n = len(subset)
                    result = {'bets': n, 'wins': wins, 'wr': wins/n*100, 'profit': round(profit, 2), 'roi': profit/n*100}

                result['fold'] = fold_idx
                result['test_start'] = str(fold_data['match_date'].iloc[0])[:10]
                result['test_end'] = str(fold_data['match_date'].iloc[-1])[:10]
                fold_details[strat_name].append(result)

        return {
            'strategies': strategies,
            'top_features': top_features,
            'fold_details': fold_details,
        }


def _get_predicted_outcome(p_home, p_draw, p_away) -> str:
    """Return predicted outcome as 'home', 'draw', or 'away'."""
    probs = {'home': p_home, 'draw': p_draw, 'away': p_away}
    return max(probs, key=probs.get)


def _apply_thresholds(df: pd.DataFrame, thresholds: dict = None, odds_range: tuple = (1.5, 3.5)) -> pd.Series:
    """Apply threshold strategy to get mask of bets."""
    if thresholds is None:
        thresholds = {'home': 0.60, 'draw': 0.35, 'away': 0.45}
    odds_min, odds_max = odds_range

    mask = pd.Series(False, index=df.index)

    for _, row in df.iterrows():
        pred = _get_predicted_outcome(row['pred_home_prob'], row['pred_draw_prob'], row['pred_away_prob'])
        prob = row[f'pred_{pred}_prob']
        odds = row[f'best_{pred}_odds']

        if prob >= thresholds[pred] and odds_min <= odds <= odds_max:
            mask.at[row.name] = True

    return mask


def _resolve_strategy_mask(strat_name: str, df: pd.DataFrame):
    """Resolve a strategy name into a boolean mask over df rows."""
    import re as re_mod

    if strat_name == 'threshold_H60':
        return df['threshold_pass']
    elif strat_name == 'threshold_H55':
        return df['threshold_h55_pass']
    elif strat_name.startswith('gbm_'):
        conf = float(strat_name.split('_')[1])
        return df['gbm_prob'] >= conf
    elif strat_name.startswith('lr_'):
        conf = float(strat_name.split('_')[1])
        return df['lr_prob'] >= conf

    # Hybrid: H60+gbm_0.55, H55+gbm_0.55, H60_noOF+gbm_0.55, H55_noOF+gbm_0.55
    m = re_mod.match(r'(H\d+(?:_noOF)?)\+gbm_(\d+\.\d+)', strat_name)
    if m:
        thr_key = m.group(1)
        conf = float(m.group(2))
        col_map = {
            'H60': 'threshold_pass',
            'H55': 'threshold_h55_pass',
            'H60_noOF': 'threshold_no_odds_pass',
            'H55_noOF': 'threshold_h55_no_odds_pass',
        }
        thr_col = col_map.get(thr_key)
        if thr_col and thr_col in df.columns:
            return df[thr_col] & (df['gbm_prob'] >= conf)

    return None
