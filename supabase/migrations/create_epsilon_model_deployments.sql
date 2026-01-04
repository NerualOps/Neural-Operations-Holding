-- Migration: Create epsilon_model_deployments table
-- This table stores model deployments that require owner approval before going live
-- Created: 2025-01-01

-- Ensure is_owner() function exists (required for RLS policies)
-- This function checks if the current user is an owner
CREATE OR REPLACE FUNCTION is_owner()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
STABLE
AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM profiles 
        WHERE id = auth.uid() AND role = 'owner'
    );
END;
$$;

-- Drop if exists (for re-running migration)
DROP TABLE IF EXISTS epsilon_model_deployments CASCADE;

-- Create epsilon_model_deployments table
CREATE TABLE epsilon_model_deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id TEXT UNIQUE NOT NULL,
    model_data JSONB,
    storage_path TEXT,
    stats JSONB NOT NULL,
    temperature FLOAT NOT NULL,
    version TEXT DEFAULT '1.0.0',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    quality_score FLOAT,
    improvement FLOAT,
    training_samples INTEGER,
    learning_description TEXT,
    deployed_at TIMESTAMPTZ,
    deployed_by TEXT,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT DEFAULT 'epsilon_automatic_training',
    CONSTRAINT model_data_or_storage CHECK (
        (status IN ('pending', 'rejected')) OR ((model_data IS NOT NULL) OR (storage_path IS NOT NULL))
    )
);

-- Create indexes for performance
CREATE INDEX idx_epsilon_model_deployments_status ON epsilon_model_deployments(status);
CREATE INDEX idx_epsilon_model_deployments_created_at ON epsilon_model_deployments(created_at DESC);
CREATE INDEX idx_epsilon_model_deployments_model_id ON epsilon_model_deployments(model_id);

-- Create trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_epsilon_model_deployments_updated_at 
    BEFORE UPDATE ON epsilon_model_deployments 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security
ALTER TABLE epsilon_model_deployments ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- Note: is_owner() function must exist (defined in complete_schema.sql)
-- Owners can manage all deployments
CREATE POLICY "Owners manage model deployments" 
    ON epsilon_model_deployments 
    FOR ALL 
    USING (is_owner());

-- Service role can manage all deployments (for backend operations)
CREATE POLICY "Service role manages model deployments" 
    ON epsilon_model_deployments 
    FOR ALL 
    USING (auth.role() = 'service_role');

-- Grant necessary permissions
GRANT ALL ON epsilon_model_deployments TO authenticated;
GRANT ALL ON epsilon_model_deployments TO service_role;

COMMENT ON TABLE epsilon_model_deployments IS 'Stores Epsilon AI model deployments that require owner approval before going live';
COMMENT ON COLUMN epsilon_model_deployments.status IS 'pending: awaiting approval, approved: deployed to production, rejected: deployment denied';
COMMENT ON COLUMN epsilon_model_deployments.model_data IS 'JSONB containing model weights and configuration (required for approved status)';
COMMENT ON COLUMN epsilon_model_deployments.storage_path IS 'Alternative storage path for large models (if model_data is too large)';
COMMENT ON COLUMN epsilon_model_deployments.quality_score IS 'Model quality score (0-1), used to determine if deployment is good enough';
COMMENT ON COLUMN epsilon_model_deployments.improvement IS 'Improvement percentage over previous model, must be positive for approval';

