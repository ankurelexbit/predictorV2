-- Supabase Schema for Football Predictions
-- Run this in your Supabase SQL Editor

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Match Information
    fixture_id BIGINT,
    match_date TIMESTAMP WITH TIME ZONE,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    league VARCHAR(100),
    league_id INTEGER,
    
    -- Model Predictions
    prob_home DECIMAL(5,4),
    prob_draw DECIMAL(5,4),
    prob_away DECIMAL(5,4),
    
    -- Recommendation
    recommended_bet VARCHAR(10),  -- 'HOME', 'DRAW', 'AWAY', 'NO_BET'
    confidence DECIMAL(5,4),
    
    -- Betting Odds
    odds_home DECIMAL(6,2),
    odds_draw DECIMAL(6,2),
    odds_away DECIMAL(6,2),
    best_odds DECIMAL(6,2),
    
    -- Features
    features_count INTEGER,
    
    -- Actual Results (to be updated later)
    actual_result VARCHAR(10),  -- 'HOME', 'DRAW', 'AWAY'
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    
    -- Performance Tracking
    is_correct BOOLEAN,
    profit_loss DECIMAL(10,2),
    
    -- Metadata
    model_version VARCHAR(50) DEFAULT 'xgboost_draw_tuned',
    thresholds JSONB,
    
    -- Ensure unique predictions per fixture
    UNIQUE(fixture_id, created_at)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_match_date ON predictions(match_date);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions(league);
CREATE INDEX IF NOT EXISTS idx_predictions_recommended_bet ON predictions(recommended_bet);
CREATE INDEX IF NOT EXISTS idx_predictions_fixture_id ON predictions(fixture_id);

-- Enable Row Level Security (RLS)
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (adjust based on your security needs)
CREATE POLICY "Enable all operations for authenticated users" ON predictions
    FOR ALL
    USING (auth.role() = 'authenticated');

-- Or for public access (if needed):
-- CREATE POLICY "Enable read access for all users" ON predictions
--     FOR SELECT
--     USING (true);

-- Create a view for easy performance analysis
CREATE OR REPLACE VIEW prediction_performance AS
SELECT 
    DATE(match_date) as date,
    league,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN recommended_bet != 'NO_BET' THEN 1 ELSE 0 END) as total_bets,
    SUM(CASE WHEN is_correct = true THEN 1 ELSE 0 END) as correct_bets,
    ROUND(AVG(CASE WHEN is_correct = true THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate_pct,
    SUM(profit_loss) as total_profit,
    ROUND(SUM(profit_loss) / NULLIF(SUM(CASE WHEN recommended_bet != 'NO_BET' THEN 1 ELSE 0 END), 0) * 100, 2) as roi_pct
FROM predictions
WHERE recommended_bet != 'NO_BET'
GROUP BY DATE(match_date), league
ORDER BY date DESC, league;

-- Grant access to the view
GRANT SELECT ON prediction_performance TO authenticated;

COMMENT ON TABLE predictions IS 'Stores all live football match predictions with probabilities, recommendations, and actual results';
COMMENT ON VIEW prediction_performance IS 'Aggregated performance metrics by date and league';
