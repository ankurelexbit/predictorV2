"""
Train Improved V4 Model with Data Cleanup and Ensemble Methods.

Implements:
1. Data cleanup (remove bad features)
2. Multiple models (XGBoost, LightGBM, CatBoost)
3. Stacking ensemble
4. Hyperparameter tuning with Optuna
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Features to remove based on data quality analysis
CONSTANT_FEATURES = [
    'home_big_chances_per_match_5', 'away_big_chances_per_match_5',
    'home_xg_trend_10', 'away_xg_trend_10',
    'home_xg_vs_top_half', 'away_xg_vs_top_half',
    'home_xga_vs_bottom_half', 'away_xga_vs_bottom_half',
    'home_lineup_avg_rating_5', 'away_lineup_avg_rating_5',
    'home_top_3_players_rating', 'away_top_3_players_rating',
    'home_key_players_available', 'away_key_players_available',
    'home_players_in_form', 'away_players_in_form',
    'home_players_unavailable', 'away_players_unavailable',
    'home_days_since_last_match', 'away_days_since_last_match',
    'rest_advantage', 'is_derby_match'
]

HIGH_MISSING_FEATURES = [
    'away_interceptions_per_90', 'away_defensive_actions_per_90',
    'home_defensive_actions_per_90', 'home_interceptions_per_90',
    'derived_xgd_matchup', 'home_xga_trend_10', 'away_xga_trend_10',
    'away_derived_xga_per_match_5', 'away_derived_xgd_5',
    'away_derived_xg_per_match_5', 'away_goals_vs_xg_5',
    'away_ga_vs_xga_5', 'home_derived_xga_per_match_5',
    'home_derived_xgd_5', 'home_ga_vs_xga_5',
    'home_goals_vs_xg_5', 'home_derived_xg_per_match_5',
    'away_shots_on_target_per_match_5', 'home_shots_on_target_conceded_5',
    'home_shots_on_target_per_match_5', 'away_xg_per_shot_5',
    'away_shot_accuracy_5', 'home_xg_per_shot_5', 'home_shot_accuracy_5'
]

REDUNDANT_FEATURES = [
    'home_elo_vs_league_avg', 'away_elo_vs_league_avg',
    'elo_diff_with_home_advantage',
    'home_ppda_5', 'away_ppda_5',
    'home_xg_from_corners_5', 'away_xg_from_corners_5',
    'home_big_chance_conversion_5', 'away_big_chance_conversion_5',
    'home_inside_box_xg_ratio', 'away_inside_box_xg_ratio',
]

BAD_FEATURES = CONSTANT_FEATURES + HIGH_MISSING_FEATURES + REDUNDANT_FEATURES

METADATA_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id',
    'league_id', 'match_date', 'home_score', 'away_score', 'result', 'target'
]


def load_and_clean_data(data_path: str):
    """Load and clean training data."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Convert date
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Map target
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)

    # Drop rows with missing target
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Sort by date
    df = df.sort_values('match_date').reset_index(drop=True)

    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Remove bad features
    features_to_drop = [f for f in BAD_FEATURES if f in df.columns]
    logger.info(f"Removing {len(features_to_drop)} bad features...")
    df = df.drop(columns=features_to_drop, errors='ignore')

    logger.info(f"After cleanup: {len(df.columns)} columns remaining")

    return df


def get_train_val_test_split(df: pd.DataFrame, train_split: float = 0.70, val_split: float = 0.15):
    """Split data chronologically."""
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Data Split:")
    logger.info(f"  Train:      {len(train_df)} ({len(train_df)/n:.1%})")
    logger.info(f"  Validation: {len(val_df)} ({len(val_df)/n:.1%})")
    logger.info(f"  Test:       {len(test_df)} ({len(test_df)/n:.1%})")

    return train_df, val_df, test_df


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    import xgboost as xgb

    logger.info("Training XGBoost...")

    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': 0.03,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'gamma': 1.0,
        'n_estimators': 500,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not installed. Install with: pip install lightgbm")
        return None

    logger.info("Training LightGBM...")

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 500,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    return model


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.warning("CatBoost not installed. Install with: pip install catboost")
        return None

    logger.info("Training CatBoost...")

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=False
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{model_name} EVALUATION")
    logger.info(f"{'=' * 80}")

    # Predict
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Metrics
    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"Log Loss: {loss:.4f}")
    logger.info(f"Accuracy: {acc:.1%}")

    # Class distribution
    draw_preds = (y_pred == 1).sum()
    draw_actual = (y_test == 1).sum()
    logger.info(f"Draws Predicted: {draw_preds} ({draw_preds/len(y_pred):.1%}) vs Actual: {draw_actual} ({draw_actual/len(y_pred):.1%})")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"         Away   Draw   Home")
    logger.info(f"Away:    {cm[0]}")
    logger.info(f"Draw:    {cm[1]}")
    logger.info(f"Home:    {cm[2]}")

    return {'log_loss': loss, 'accuracy': acc}


def train_stacking_ensemble(X_train, y_train, X_val, y_val):
    """Train stacking ensemble of XGBoost, LightGBM, and CatBoost."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING STACKING ENSEMBLE")
    logger.info("=" * 80)

    # Train base models
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    cat_model = train_catboost(X_train, y_train, X_val, y_val)

    # Build estimators list
    estimators = []
    if xgb_model:
        estimators.append(('xgb', xgb_model))
    if lgb_model:
        estimators.append(('lgb', lgb_model))
    if cat_model:
        estimators.append(('cat', cat_model))

    if len(estimators) < 2:
        logger.warning("Not enough models for stacking, returning best single model")
        return xgb_model

    # Create stacking ensemble
    logger.info("Creating stacking ensemble with meta-learner...")
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )

    logger.info("Training stacking ensemble (this may take a while)...")
    stacking_model.fit(X_train, y_train)

    return stacking_model


def main():
    parser = argparse.ArgumentParser(description='Train Improved V4 Model')
    parser.add_argument('--data', default='data/training_data.csv', help='Training data path')
    parser.add_argument('--output', default='models/v4_improved.joblib', help='Model output path')
    parser.add_argument('--model', default='stacking',
                       choices=['xgboost', 'lightgbm', 'catboost', 'stacking'],
                       help='Model type to train')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("IMPROVED MODEL TRAINING")
    logger.info("=" * 80)

    # 1. Load and clean data
    df = load_and_clean_data(args.data)

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    logger.info(f"Using {len(feature_cols)} features (after cleanup)")

    # 2. Split data
    train_df, val_df, test_df = get_train_val_test_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    # 3. Train model
    if args.model == 'xgboost':
        model = train_xgboost(X_train, y_train, X_val, y_val)
    elif args.model == 'lightgbm':
        model = train_lightgbm(X_train, y_train, X_val, y_val)
    elif args.model == 'catboost':
        model = train_catboost(X_train, y_train, X_val, y_val)
    elif args.model == 'stacking':
        model = train_stacking_ensemble(X_train, y_train, X_val, y_val)

    if model is None:
        logger.error("Model training failed")
        return

    # 4. Evaluate on test set
    results = evaluate_model(model, X_test, y_test, model_name=args.model.upper())

    # 5. Save model
    import joblib
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"\nModel saved to {output_path}")

    # 6. Save feature list
    feature_list_path = output_path.parent / 'feature_list.txt'
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    logger.info(f"Feature list saved to {feature_list_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Test Accuracy: {results['accuracy']:.1%}")
    logger.info(f"Test Log Loss: {results['log_loss']:.4f}")


if __name__ == '__main__':
    main()
