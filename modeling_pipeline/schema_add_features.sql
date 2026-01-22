-- Add features column to predictions table
-- Run this in Supabase SQL Editor

ALTER TABLE predictions 
ADD COLUMN IF NOT EXISTS features JSONB;

-- Add index for features column (for faster queries)
CREATE INDEX IF NOT EXISTS idx_predictions_features ON predictions USING GIN (features);

-- Add comment
COMMENT ON COLUMN predictions.features IS 'All features used to make the prediction (JSONB)';
