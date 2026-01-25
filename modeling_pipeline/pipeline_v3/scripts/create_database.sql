-- Database Schema for Pipeline V3
-- PostgreSQL / Supabase

-- ============================================================================
-- MATCHES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER UNIQUE NOT NULL,
    league_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    match_date TIMESTAMP NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    result VARCHAR(10), -- 'H', 'D', 'A'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_matches_fixture_id ON matches(fixture_id);
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_teams ON matches(home_team_id, away_team_id);
CREATE INDEX idx_matches_league_season ON matches(league_id, season_id);

-- ============================================================================
-- MATCH STATISTICS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS match_statistics (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER REFERENCES matches(fixture_id),
    team_id INTEGER NOT NULL,
    
    -- Shot statistics
    shots_total INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    shots_insidebox INTEGER DEFAULT 0,
    shots_outsidebox INTEGER DEFAULT 0,
    shots_off_target INTEGER DEFAULT 0,
    
    -- Chance creation
    big_chances_created INTEGER DEFAULT 0,
    big_chances_missed INTEGER DEFAULT 0,
    
    -- Set pieces
    corners INTEGER DEFAULT 0,
    
    -- Attacks
    attacks INTEGER DEFAULT 0,
    dangerous_attacks INTEGER DEFAULT 0,
    
    -- Possession & passing
    possession FLOAT DEFAULT 0,
    passes INTEGER DEFAULT 0,
    accurate_passes INTEGER DEFAULT 0,
    
    -- Defensive actions
    tackles INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    clearances INTEGER DEFAULT 0,
    
    -- Discipline
    fouls INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_match_stats_fixture ON match_statistics(fixture_id);
CREATE INDEX idx_match_stats_team ON match_statistics(team_id);

-- ============================================================================
-- ELO HISTORY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS elo_history (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    match_date TIMESTAMP NOT NULL,
    elo_rating FLOAT NOT NULL,
    elo_change FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_elo_team_date ON elo_history(team_id, match_date);
CREATE INDEX idx_elo_season ON elo_history(season_id);

-- ============================================================================
-- STANDINGS HISTORY TABLE (calculated)
-- ============================================================================
CREATE TABLE IF NOT EXISTS standings_history (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    league_position INTEGER,
    points INTEGER DEFAULT 0,
    matches_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    goals_for INTEGER DEFAULT 0,
    goals_against INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_standings_team_date ON standings_history(team_id, as_of_date);
CREATE INDEX idx_standings_season ON standings_history(season_id);

-- ============================================================================
-- XG HISTORY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS xg_history (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER REFERENCES matches(fixture_id),
    team_id INTEGER NOT NULL,
    match_date TIMESTAMP NOT NULL,
    derived_xg FLOAT DEFAULT 0,
    derived_xga FLOAT DEFAULT 0,
    derived_xgd FLOAT DEFAULT 0,
    shots_insidebox INTEGER DEFAULT 0,
    shots_outsidebox INTEGER DEFAULT 0,
    big_chances INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_xg_team_date ON xg_history(team_id, match_date);
CREATE INDEX idx_xg_fixture ON xg_history(fixture_id);

-- ============================================================================
-- TRAINING FEATURES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS training_features (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER REFERENCES matches(fixture_id),
    match_date TIMESTAMP NOT NULL,
    
    -- Store all features as JSONB for flexibility
    features JSONB NOT NULL,
    
    -- Target variable
    target VARCHAR(10), -- 'H', 'D', 'A'
    home_goals INTEGER,
    away_goals INTEGER,
    
    -- Metadata
    feature_version VARCHAR(20) DEFAULT 'v3.0',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_training_fixture ON training_features(fixture_id);
CREATE INDEX idx_training_date ON training_features(match_date);
CREATE INDEX idx_training_features_gin ON training_features USING GIN (features);

-- ============================================================================
-- PLAYER AVAILABILITY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_availability (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER REFERENCES matches(fixture_id),
    team_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    is_available BOOLEAN DEFAULT TRUE,
    reason VARCHAR(100), -- 'injured', 'suspended', 'rested'
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_player_avail_fixture ON player_availability(fixture_id);
CREATE INDEX idx_player_avail_team ON player_availability(team_id);
CREATE INDEX idx_player_avail_player ON player_availability(player_id);

-- ============================================================================
-- KEY PLAYERS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS key_players (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(255),
    position VARCHAR(50),
    avg_rating FLOAT DEFAULT 0,
    key_score FLOAT DEFAULT 0,
    is_key_player BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_key_players_team_season ON key_players(team_id, season_id);
CREATE INDEX idx_key_players_player ON key_players(player_id);

-- ============================================================================
-- VIEWS FOR CONVENIENCE
-- ============================================================================

-- Latest Elo ratings
CREATE OR REPLACE VIEW latest_elo_ratings AS
SELECT DISTINCT ON (team_id)
    team_id,
    elo_rating,
    match_date
FROM elo_history
ORDER BY team_id, match_date DESC;

-- Latest standings
CREATE OR REPLACE VIEW latest_standings AS
SELECT DISTINCT ON (team_id, season_id)
    team_id,
    season_id,
    league_position,
    points,
    matches_played
FROM standings_history
ORDER BY team_id, season_id, as_of_date DESC;
