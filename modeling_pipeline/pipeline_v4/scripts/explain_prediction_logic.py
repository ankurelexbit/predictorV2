#!/usr/bin/env python3
"""
Detailed Explanation of Prediction Logic
=========================================

This script demonstrates exactly how predictions are made for a match.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"

META_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target'
]


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    result_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(result_map)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values

    # Train models (simplified - no hypertuning shown here)
    print("Training CatBoost model...")
    catboost = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.03,
        auto_class_weights='Balanced', random_seed=42, verbose=False
    )
    catboost.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

    print("Training LightGBM model...")
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        class_weight='balanced', random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # =========================================================================
    # PREDICTION LOGIC EXPLANATION
    # =========================================================================
    print("\n" + "="*80)
    print("HOW PREDICTION WORKS - STEP BY STEP")
    print("="*80)

    # Take 5 example matches from test set
    examples = test_df.iloc[100:105].copy()

    for idx, row in examples.iterrows():
        print(f"\n{'='*80}")
        print(f"MATCH EXAMPLE: {row['match_date'].date()}")
        print(f"Actual Result: {row['result']} (Home score: {row['home_score']}, Away score: {row['away_score']})")
        print("="*80)

        # Step 1: Extract features for this match
        X_match = row[feature_cols].values.reshape(1, -1)

        print("\nSTEP 1: Key features for this match")
        print("-" * 40)
        key_features = ['home_elo', 'away_elo', 'elo_diff', 'home_league_position',
                        'away_league_position', 'position_diff']
        for f in key_features:
            if f in feature_cols:
                print(f"  {f}: {row[f]:.2f}")

        # Step 2: Each model outputs PROBABILITIES for each outcome
        print("\nSTEP 2: Model outputs probabilities for EACH outcome")
        print("-" * 40)

        probs_catboost = catboost.predict_proba(X_match)[0]
        probs_lgbm = lgbm.predict_proba(X_match)[0]

        print("  CatBoost probabilities:")
        print(f"    Away (A): {probs_catboost[0]*100:.1f}%")
        print(f"    Draw (D): {probs_catboost[1]*100:.1f}%")
        print(f"    Home (H): {probs_catboost[2]*100:.1f}%")

        print("\n  LightGBM probabilities:")
        print(f"    Away (A): {probs_lgbm[0]*100:.1f}%")
        print(f"    Draw (D): {probs_lgbm[1]*100:.1f}%")
        print(f"    Home (H): {probs_lgbm[2]*100:.1f}%")

        # Step 3: Ensemble (average) the probabilities
        print("\nSTEP 3: Ensemble (average) the probabilities")
        print("-" * 40)
        probs_ensemble = (probs_catboost + probs_lgbm) / 2
        print(f"  Ensemble probabilities:")
        print(f"    Away (A): {probs_ensemble[0]*100:.1f}%")
        print(f"    Draw (D): {probs_ensemble[1]*100:.1f}%")
        print(f"    Home (H): {probs_ensemble[2]*100:.1f}%")

        # Step 4: Apply confidence thresholds
        print("\nSTEP 4: Apply confidence thresholds to decide prediction")
        print("-" * 40)
        print("  Thresholds used:")
        print("    Home: >= 60% confidence")
        print("    Away: >= 55% confidence")
        print("    Draw: >= 45% confidence + conditions (elo close, etc.)")

        # Check thresholds
        home_thresh = 0.60
        away_thresh = 0.55
        draw_thresh = 0.45

        candidates = []
        if probs_ensemble[2] >= home_thresh:
            candidates.append(('Home', probs_ensemble[2]))
        if probs_ensemble[0] >= away_thresh:
            candidates.append(('Away', probs_ensemble[0]))
        if probs_ensemble[1] >= draw_thresh:
            # Additional draw conditions would be checked here
            elo_close = abs(row.get('home_elo', 0) - row.get('away_elo', 0)) < 50
            if elo_close:
                candidates.append(('Draw', probs_ensemble[1]))

        print(f"\n  Candidates meeting threshold: {candidates if candidates else 'None'}")

        # Step 5: Final prediction
        print("\nSTEP 5: Final prediction")
        print("-" * 40)
        if candidates:
            # Pick highest confidence among candidates
            final_pred = max(candidates, key=lambda x: x[1])
            print(f"  PREDICTION: {final_pred[0]} (confidence: {final_pred[1]*100:.1f}%)")
        else:
            print("  PREDICTION: NO BET (no outcome meets confidence threshold)")

        # Step 6: Evaluate
        print("\nSTEP 6: Evaluation")
        print("-" * 40)
        if candidates:
            pred_outcome = final_pred[0][0]  # 'H', 'A', or 'D'
            actual = row['result']
            correct = (pred_outcome == actual)
            print(f"  Predicted: {pred_outcome}, Actual: {actual}")
            print(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")
        else:
            print("  No prediction made (skipped this match)")

    # =========================================================================
    # SUMMARY OF LOGIC
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: HOW THE SYSTEM WORKS")
    print("="*80)
    print("""
1. MODEL TRAINING:
   - Train CatBoost and LightGBM on historical data (2016-2022)
   - Models learn to predict probability of each outcome (H/D/A)
   - Class balancing ensures draws aren't ignored

2. FOR EACH NEW MATCH:
   a) Extract 162 features (Elo, position, form, xG, shots, etc.)
   b) Each model outputs 3 probabilities: P(Away), P(Draw), P(Home)
   c) Ensemble: Average the probabilities from both models

3. PREDICTION DECISION:
   - Check if any outcome exceeds its confidence threshold:
     * Home: >= 60-70% probability
     * Away: >= 55-60% probability
     * Draw: >= 45% probability AND additional conditions met

   - If multiple outcomes qualify, pick the one with highest probability
   - If NO outcome qualifies, skip the match (no bet)

4. WHY THIS SELECTIVE APPROACH?
   - Higher thresholds = fewer predictions but higher win rate
   - Lower thresholds = more predictions but lower win rate
   - The key insight: We don't HAVE to predict every match
   - Only predict when the model is confident

5. DRAW SPECIAL HANDLING:
   - Draws have extra conditions (elo close, positions close, etc.)
   - Because draws are inherently harder to predict
   - Even with conditions, draw WR maxes out at ~35%
""")


if __name__ == '__main__':
    main()
