-- Enhanced Predictions Schema Migration
-- Adds comprehensive metadata storage for lineups, injuries, odds, and timing
-- Run this in Supabase SQL Editor

-- ============================================================================
-- TIMING DATA
-- ============================================================================

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS kickoff_time TIMESTAMP WITH TIME ZONE;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS prediction_time TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS hours_before_kickoff DECIMAL(5,2);

COMMENT ON COLUMN predictions.kickoff_time IS 'Actual match kickoff time';
COMMENT ON COLUMN predictions.prediction_time IS 'When this prediction was generated';
COMMENT ON COLUMN predictions.hours_before_kickoff IS 'Hours between prediction and kickoff';

-- ============================================================================
-- LINEUP DATA
-- ============================================================================

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_lineup JSONB;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_lineup JSONB;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS lineup_available BOOLEAN DEFAULT FALSE;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS lineup_coverage_home DECIMAL(5,2);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS lineup_coverage_away DECIMAL(5,2);

COMMENT ON COLUMN predictions.home_lineup IS 'Home team starting XI with player details (JSONB)';
COMMENT ON COLUMN predictions.away_lineup IS 'Away team starting XI with player details (JSONB)';
COMMENT ON COLUMN predictions.lineup_available IS 'Whether lineups were available at prediction time';
COMMENT ON COLUMN predictions.lineup_coverage_home IS 'Percentage of home players found in database';
COMMENT ON COLUMN predictions.lineup_coverage_away IS 'Percentage of away players found in database';

-- ============================================================================
-- INJURY DATA
-- ============================================================================

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_injuries_count INTEGER DEFAULT 0;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_injuries_count INTEGER DEFAULT 0;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_injured_players JSONB;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_injured_players JSONB;

COMMENT ON COLUMN predictions.home_injuries_count IS 'Number of injured/suspended home players';
COMMENT ON COLUMN predictions.away_injuries_count IS 'Number of injured/suspended away players';
COMMENT ON COLUMN predictions.home_injured_players IS 'List of home team injured/suspended players (JSONB)';
COMMENT ON COLUMN predictions.away_injured_players IS 'List of away team injured/suspended players (JSONB)';

-- ============================================================================
-- MULTIPLE BOOKMAKER ODDS
-- ============================================================================

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS bookmaker_odds JSONB;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS best_odds_home DECIMAL(6,2);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS best_odds_draw DECIMAL(6,2);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS best_odds_away DECIMAL(6,2);

COMMENT ON COLUMN predictions.bookmaker_odds IS 'Odds from all available bookmakers (JSONB)';
COMMENT ON COLUMN predictions.best_odds_home IS 'Best available odds for home win';
COMMENT ON COLUMN predictions.best_odds_draw IS 'Best available odds for draw';
COMMENT ON COLUMN predictions.best_odds_away IS 'Best available odds for away win';

-- ============================================================================
-- OUR CALCULATED FAIR ODDS
-- ============================================================================

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS our_odds_home DECIMAL(6,2);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS our_odds_draw DECIMAL(6,2);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS our_odds_away DECIMAL(6,2);

COMMENT ON COLUMN predictions.our_odds_home IS 'Our calculated fair odds for home win (1/prob_home)';
COMMENT ON COLUMN predictions.our_odds_draw IS 'Our calculated fair odds for draw (1/prob_draw)';
COMMENT ON COLUMN predictions.our_odds_away IS 'Our calculated fair odds for away win (1/prob_away)';

-- ============================================================================
-- DATA QUALITY FLAGS
-- ============================================================================

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS used_lineup_data BOOLEAN DEFAULT FALSE;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS used_injury_data BOOLEAN DEFAULT FALSE;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS data_quality_score DECIMAL(3,2);

COMMENT ON COLUMN predictions.used_lineup_data IS 'Whether lineup data was used in feature calculation';
COMMENT ON COLUMN predictions.used_injury_data IS 'Whether injury data was used in feature calculation';
COMMENT ON COLUMN predictions.data_quality_score IS 'Overall data quality score (0-1)';

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_predictions_kickoff_time ON predictions(kickoff_time);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_time ON predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_predictions_lineup_available ON predictions(lineup_available);
CREATE INDEX IF NOT EXISTS idx_predictions_used_lineup_data ON predictions(used_lineup_data);
CREATE INDEX IF NOT EXISTS idx_predictions_hours_before_kickoff ON predictions(hours_before_kickoff);

-- ============================================================================
-- USEFUL VIEWS
-- ============================================================================

-- View: Lineup availability analysis
CREATE OR REPLACE VIEW lineup_availability_stats AS
SELECT 
    DATE(kickoff_time) as date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN lineup_available THEN 1 ELSE 0 END) as with_lineups,
    ROUND(AVG(CASE WHEN lineup_available THEN 1.0 ELSE 0.0 END) * 100, 1) as lineup_pct,
    ROUND(AVG(hours_before_kickoff), 1) as avg_hours_before,
    ROUND(AVG(lineup_coverage_home), 1) as avg_home_coverage,
    ROUND(AVG(lineup_coverage_away), 1) as avg_away_coverage
FROM predictions
WHERE kickoff_time IS NOT NULL
GROUP BY DATE(kickoff_time)
ORDER BY date DESC;

-- View: Value bets (our odds vs market)
CREATE OR REPLACE VIEW value_bets AS
SELECT 
    fixture_id,
    kickoff_time,
    home_team,
    away_team,
    recommended_bet,
    CASE 
        WHEN recommended_bet = 'HOME' THEN our_odds_home
        WHEN recommended_bet = 'DRAW' THEN our_odds_draw
        WHEN recommended_bet = 'AWAY' THEN our_odds_away
    END as our_odds,
    CASE 
        WHEN recommended_bet = 'HOME' THEN best_odds_home
        WHEN recommended_bet = 'DRAW' THEN best_odds_draw
        WHEN recommended_bet = 'AWAY' THEN best_odds_away
    END as market_odds,
    CASE 
        WHEN recommended_bet = 'HOME' THEN ROUND((best_odds_home - our_odds_home) / our_odds_home * 100, 1)
        WHEN recommended_bet = 'DRAW' THEN ROUND((best_odds_draw - our_odds_draw) / our_odds_draw * 100, 1)
        WHEN recommended_bet = 'AWAY' THEN ROUND((best_odds_away - our_odds_away) / our_odds_away * 100, 1)
    END as edge_pct
FROM predictions
WHERE recommended_bet IN ('HOME', 'DRAW', 'AWAY')
  AND our_odds_home IS NOT NULL
  AND best_odds_home IS NOT NULL
ORDER BY edge_pct DESC;

-- View: Injury impact analysis
CREATE OR REPLACE VIEW injury_impact_stats AS
SELECT 
    CASE 
        WHEN home_injuries_count > away_injuries_count + 2 THEN 'Home heavily injured'
        WHEN home_injuries_count > away_injuries_count THEN 'Home more injured'
        WHEN away_injuries_count > home_injuries_count + 2 THEN 'Away heavily injured'
        WHEN away_injuries_count > home_injuries_count THEN 'Away more injured'
        ELSE 'Equal injuries'
    END as injury_situation,
    COUNT(*) as total_bets,
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_bets,
    ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate_pct,
    ROUND(SUM(profit_loss), 2) as total_pnl,
    ROUND(AVG(profit_loss) * 100, 1) as roi_pct
FROM predictions
WHERE recommended_bet != 'NO_BET'
  AND used_injury_data = TRUE
  AND actual_result IS NOT NULL
GROUP BY injury_situation
ORDER BY total_bets DESC;

-- View: Performance with vs without lineups
CREATE OR REPLACE VIEW lineup_performance_comparison AS
SELECT 
    used_lineup_data,
    COUNT(*) as total_bets,
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_bets,
    ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate_pct,
    ROUND(SUM(profit_loss), 2) as total_pnl,
    ROUND(AVG(profit_loss) * 100, 1) as roi_pct,
    ROUND(AVG(data_quality_score), 2) as avg_quality_score
FROM predictions
WHERE recommended_bet != 'NO_BET'
  AND actual_result IS NOT NULL
GROUP BY used_lineup_data;

-- Grant access to views
GRANT SELECT ON lineup_availability_stats TO authenticated;
GRANT SELECT ON value_bets TO authenticated;
GRANT SELECT ON injury_impact_stats TO authenticated;
GRANT SELECT ON lineup_performance_comparison TO authenticated;

-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

-- Find predictions with high value (our odds significantly better than market)
-- SELECT * FROM value_bets WHERE edge_pct > 10 ORDER BY kickoff_time DESC LIMIT 20;

-- Check lineup availability over time
-- SELECT * FROM lineup_availability_stats ORDER BY date DESC LIMIT 30;

-- Compare performance with vs without lineup data
-- SELECT * FROM lineup_performance_comparison;

-- Analyze injury impact on predictions
-- SELECT * FROM injury_impact_stats;

-- Find recent predictions with full metadata
-- SELECT 
--     home_team, away_team, kickoff_time, hours_before_kickoff,
--     lineup_available, home_injuries_count, away_injuries_count,
--     our_odds_home, best_odds_home, recommended_bet
-- FROM predictions 
-- WHERE kickoff_time > NOW() - INTERVAL '7 days'
-- ORDER BY kickoff_time DESC;
