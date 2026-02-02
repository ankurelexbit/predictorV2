-- Quick PnL Analysis for January 2026
-- Run this in your Supabase SQL editor or via psql

-- Current Performance
SELECT
    model_version,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE actual_result IS NOT NULL) as finished,
    COUNT(*) FILTER (WHERE should_bet = true) as bets_placed,
    COUNT(*) FILTER (WHERE bet_won = true) as bets_won,
    ROUND(100.0 * COUNT(*) FILTER (WHERE bet_won = true) / NULLIF(COUNT(*) FILTER (WHERE should_bet = true), 0), 1) as win_rate_pct,
    ROUND(SUM(COALESCE(bet_profit, 0))::numeric, 2) as total_profit,
    ROUND(100.0 * SUM(COALESCE(bet_profit, 0)) / NULLIF(COUNT(*) FILTER (WHERE should_bet = true), 0), 1) as roi_pct
FROM predictions
WHERE match_date >= '2026-01-01'
  AND match_date <= '2026-01-31'
GROUP BY model_version;

-- Breakdown by bet type
SELECT
    bet_outcome,
    COUNT(*) as bets,
    COUNT(*) FILTER (WHERE bet_won = true) as wins,
    ROUND(100.0 * COUNT(*) FILTER (WHERE bet_won = true) / COUNT(*), 1) as win_rate_pct,
    ROUND(SUM(COALESCE(bet_profit, 0))::numeric, 2) as profit
FROM predictions
WHERE match_date >= '2026-01-01'
  AND match_date <= '2026-01-31'
  AND should_bet = true
  AND actual_result IS NOT NULL
GROUP BY bet_outcome
ORDER BY profit DESC;

-- Export data for threshold analysis
\copy (SELECT pred_home_prob, pred_draw_prob, pred_away_prob, actual_result, best_home_odds, best_draw_odds, best_away_odds FROM predictions WHERE match_date >= '2026-01-01' AND match_date <= '2026-01-31' AND actual_result IS NOT NULL) TO '/tmp/predictions_jan2026.csv' CSV HEADER;
