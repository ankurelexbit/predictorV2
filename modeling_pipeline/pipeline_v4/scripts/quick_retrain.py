#!/usr/bin/env python3
"""Quick retrain without class weights."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading training data...")
df = pd.read_csv('data/training_data_with_draw_features.csv')

# Drop non-numeric and metadata columns
meta_cols = ['fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
             'match_date', 'home_score', 'away_score', 'target']
X = df.drop(columns=[c for c in meta_cols if c in df.columns] + ['result'])
y = df['result'].map({'A': 0, 'D': 1, 'H': 2})

logger.info(f"Features: {X.shape[1]}")
logger.info(f"Samples: {len(X)}")

# Convert boolean columns to int
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    logger.info(f"Converting boolean columns to int: {bool_cols}")
    for col in bool_cols:
        X[col] = X[col].astype(int)

# Check for remaining non-numeric columns
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    logger.warning(f"Dropping non-numeric columns: {non_numeric}")
    X = X.select_dtypes(include=[np.number])

logger.info(f"Final features: {X.shape[1]}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info("\nTraining XGBoost WITHOUT class weights...")
logger.info("(This will fix the draw over-prediction issue)")

model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=6,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
logger.info("\nEvaluating model...")
probs = model.predict_proba(X_test)
preds = np.argmax(probs, axis=1)

acc = accuracy_score(y_test, preds)
logloss = log_loss(y_test, probs)

logger.info("\n" + "="*80)
logger.info("TEST SET RESULTS")
logger.info("="*80)
logger.info(f"Accuracy: {acc:.4f}")
logger.info(f"Log Loss: {logloss:.4f}")

# Prediction distribution
pred_counts = pd.Series(preds).value_counts().sort_index()
pred_pcts = pred_counts / len(preds) * 100

logger.info("\nPrediction Distribution:")
logger.info(f"  Away (0): {pred_pcts.get(0, 0):.1f}% ({pred_counts.get(0, 0)} predictions)")
logger.info(f"  Draw (1): {pred_pcts.get(1, 0):.1f}% ({pred_counts.get(1, 0)} predictions)")
logger.info(f"  Home (2): {pred_pcts.get(2, 0):.1f}% ({pred_counts.get(2, 0)} predictions)")

# Average probabilities
avg_probs = probs.mean(axis=0)
logger.info("\nAverage Probabilities:")
logger.info(f"  Away: {avg_probs[0]:.1%}")
logger.info(f"  Draw: {avg_probs[1]:.1%}")
logger.info(f"  Home: {avg_probs[2]:.1%}")

# Sample high-confidence predictions
logger.info("\nSample High-Confidence Predictions:")
for i in range(min(5, len(probs))):
    max_prob = probs[i].max()
    pred_class = ['Away', 'Draw', 'Home'][preds[i]]
    actual_class = ['Away', 'Draw', 'Home'][y_test.iloc[i]]
    correct = "✅" if preds[i] == y_test.iloc[i] else "❌"
    logger.info(f"  {correct} Pred: {pred_class} ({max_prob:.1%}) | Actual: {actual_class}")

# Save model
output_path = 'models/with_draw_features/xgboost_fixed.joblib'
joblib.dump(model, output_path)
logger.info(f"\n✅ Model saved to: {output_path}")

logger.info("\nTo use this model:")
logger.info("  1. Update predict_live_standalone.py line 53 to use 'xgboost_fixed.joblib'")
logger.info("  2. Run predictions again")
logger.info("="*80)
