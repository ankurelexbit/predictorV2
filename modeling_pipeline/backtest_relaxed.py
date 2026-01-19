"""Quick backtest with relaxed thresholds"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import joblib
import importlib.util

# Import betting strategy
spec = importlib.util.spec_from_file_location("betting", "11_smart_betting_strategy.py")
betting_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(betting_module)

# Load data
df = pd.read_csv('data/processed/sportmonks_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['target'].notna() & df['season_name'].isin(['2023/2024'])].tail(500)

# Load models
spec_elo = importlib.util.spec_from_file_location("elo", "04_model_baseline_elo.py")
elo_mod = importlib.util.module_from_spec(spec_elo)
spec_elo.loader.exec_module(elo_mod)

spec_xgb = importlib.util.spec_from_file_location("xgb", "06_model_xgboost.py")
xgb_mod = importlib.util.module_from_spec(spec_xgb)
spec_xgb.loader.exec_module(xgb_mod)

spec_ens = importlib.util.spec_from_file_location("ens", "07_model_ensemble.py")
ens_mod = importlib.util.module_from_spec(spec_ens)
spec_ens.loader.exec_module(ens_mod)

# Build ensemble
ensemble = ens_mod.EnsembleModel()
elo_model = elo_mod.EloProbabilityModel()
elo_model.load('models/elo_model.joblib')
ensemble.add_model('elo', elo_model, 0.2)

class DCWrapper:
    def predict_proba(self, df, calibrated=True):
        return np.ones((len(df), 3)) / 3
ensemble.add_model('dc', DCWrapper(), 0.3)

xgb_model = xgb_mod.XGBoostFootballModel()
xgb_model.load('models/xgboost_model.joblib')
ensemble.add_model('xgb', xgb_model, 0.5)

ens_data = joblib.load('models/ensemble_model.joblib')
if 'calibrators' in ens_data:
    ensemble.calibrators = ens_data['calibrators']
    ensemble.is_calibrated = True
if 'stacking_model' in ens_data:
    ensemble.stacking_model = ens_data['stacking_model']

# Predict
preds = ensemble.predict_proba(df, calibrated=True)
df['away_prob'] = preds[:, 0]
df['draw_prob'] = preds[:, 1]
df['home_prob'] = preds[:, 2]
df['actual'] = df['target'].map({0: 'Away Win', 1: 'Draw', 2: 'Home Win'})

# Strategy with RELAXED thresholds
strategy = betting_module.SmartMultiOutcomeStrategy(
    away_win_min_prob=0.35,  # Was 0.50
    draw_close_threshold=0.15,  # Was 0.10
    home_win_min_prob=0.55,  # Was 0.65
    bankroll=1000.0
)

# Run backtest
bets = []
for _, row in df.iterrows():
    match = {
        'home_team': row.get('home_team_name', 'Home'),
        'away_team': row.get('away_team_name', 'Away'),
        'home_prob': row['home_prob'],
        'draw_prob': row['draw_prob'],
        'away_prob': row['away_prob']
    }
    recs = strategy.evaluate_match(match)
    
    for bet in recs:
        profit = bet.stake * (bet.fair_odds - 1) if bet.bet_outcome == row['actual'] else -bet.stake
        bets.append({
            'bet': bet.bet_outcome,
            'actual': row['actual'],
            'stake': bet.stake,
            'odds': bet.fair_odds,
            'profit': profit,
            'rule': bet.rule_applied
        })
        if bet.bet_outcome == row['actual']:
            strategy.bankroll += profit
        else:
            strategy.bankroll -= bet.stake

# Results
if len(bets) > 0:
    bets_df = pd.DataFrame(bets)
    total_stake = bets_df['stake'].sum()
    total_profit = bets_df['profit'].sum()
    wins = len(bets_df[bets_df['profit'] > 0])
    
    print(f"\n{'='*60}")
    print(f"BACKTEST WITH RELAXED THRESHOLDS")
    print(f"{'='*60}")
    print(f"Away min prob: 35% (was 50%)")
    print(f"Draw threshold: 15% (was 10%)")
    print(f"Home min prob: 55% (was 65%)")
    print(f"\nMatches analyzed: {len(df)}")
    print(f"Bets placed: {len(bets)}")
    print(f"Winning bets: {wins} ({wins/len(bets)*100:.1f}%)")
    print(f"Total staked: ${total_stake:.2f}")
    print(f"Net profit: ${total_profit:+.2f}")
    print(f"ROI: {total_profit/total_stake*100:+.2f}%")
    print(f"Final bankroll: ${strategy.bankroll:.2f}")
    print(f"\nBy outcome:")
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        outcome_bets = bets_df[bets_df['bet'] == outcome]
        if len(outcome_bets) > 0:
            profit = outcome_bets['profit'].sum()
            wins_pct = len(outcome_bets[outcome_bets['profit'] > 0]) / len(outcome_bets) * 100
            print(f"  {outcome}: {len(outcome_bets)} bets, {wins_pct:.1f}% win rate, ${profit:+.2f}")
else:
    print("No bets placed")
