-- Database initialization script for Causal UI Gym
-- This script runs when the PostgreSQL container starts for the first time

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create main database if it doesn't exist
-- (Note: This is already created by POSTGRES_DB environment variable)

-- Create application user with limited privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'causal_ui_app') THEN
        CREATE ROLE causal_ui_app WITH LOGIN PASSWORD 'app_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE causal_ui_gym TO causal_ui_app;
GRANT USAGE ON SCHEMA public TO causal_ui_app;
GRANT CREATE ON SCHEMA public TO causal_ui_app;

-- Create schemas for better organization
CREATE SCHEMA IF NOT EXISTS experiments;
CREATE SCHEMA IF NOT EXISTS users;
CREATE SCHEMA IF NOT EXISTS metrics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Grant permissions on schemas
GRANT USAGE ON SCHEMA experiments TO causal_ui_app;
GRANT USAGE ON SCHEMA users TO causal_ui_app;
GRANT USAGE ON SCHEMA metrics TO causal_ui_app;
GRANT USAGE ON SCHEMA audit TO causal_ui_app;

GRANT CREATE ON SCHEMA experiments TO causal_ui_app;
GRANT CREATE ON SCHEMA users TO causal_ui_app;
GRANT CREATE ON SCHEMA metrics TO causal_ui_app;
GRANT CREATE ON SCHEMA audit TO causal_ui_app;

-- Create core tables

-- Users table
CREATE TABLE IF NOT EXISTS users.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- User sessions table
CREATE TABLE IF NOT EXISTS users.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_by UUID NOT NULL REFERENCES users.users(id),
    causal_model JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'draft',
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Interventions table
CREATE TABLE IF NOT EXISTS experiments.interventions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments.experiments(id) ON DELETE CASCADE,
    variable_name VARCHAR(255) NOT NULL,
    intervention_value JSONB NOT NULL,
    performed_by UUID NOT NULL REFERENCES users.users(id),
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    result JSONB,
    computation_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

-- LLM responses table
CREATE TABLE IF NOT EXISTS experiments.llm_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments.experiments(id) ON DELETE CASCADE,
    intervention_id UUID REFERENCES experiments.interventions(id) ON DELETE SET NULL,
    question TEXT NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    response TEXT NOT NULL,
    confidence DECIMAL(3,2),
    reasoning TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Experiment metrics table
CREATE TABLE IF NOT EXISTS metrics.experiment_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments.experiments(id) ON DELETE CASCADE,
    intervention_id UUID REFERENCES experiments.interventions(id) ON DELETE SET NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- System metrics table for monitoring
CREATE TABLE IF NOT EXISTS metrics.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    tags JSONB DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users.users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(255) NOT NULL,
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users.users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users.users(username);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users.users(created_at);

-- Sessions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON users.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON users.sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON users.sessions(expires_at);

-- Experiments indexes
CREATE INDEX IF NOT EXISTS idx_experiments_created_by ON experiments.experiments(created_by);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments.experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments.experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_is_public ON experiments.experiments(is_public);

-- Interventions indexes
CREATE INDEX IF NOT EXISTS idx_interventions_experiment_id ON experiments.interventions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_interventions_performed_by ON experiments.interventions(performed_by);
CREATE INDEX IF NOT EXISTS idx_interventions_performed_at ON experiments.interventions(performed_at);
CREATE INDEX IF NOT EXISTS idx_interventions_variable_name ON experiments.interventions(variable_name);

-- LLM responses indexes
CREATE INDEX IF NOT EXISTS idx_llm_responses_experiment_id ON experiments.llm_responses(experiment_id);
CREATE INDEX IF NOT EXISTS idx_llm_responses_intervention_id ON experiments.llm_responses(intervention_id);
CREATE INDEX IF NOT EXISTS idx_llm_responses_model_name ON experiments.llm_responses(model_name);
CREATE INDEX IF NOT EXISTS idx_llm_responses_created_at ON experiments.llm_responses(created_at);

-- Metrics indexes
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_experiment_id ON metrics.experiment_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_metric_name ON metrics.experiment_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_calculated_at ON metrics.experiment_metrics(calculated_at);

CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_name ON metrics.system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON metrics.system_metrics(recorded_at);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit.audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit.audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource_type ON audit.audit_log(resource_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit.audit_log(created_at);

-- Create update trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic updated_at
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users.users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_experiments_updated_at 
    BEFORE UPDATE ON experiments.experiments 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to clean old sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM users.sessions WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Create function to archive old metrics
CREATE OR REPLACE FUNCTION archive_old_metrics(retention_days INTEGER DEFAULT 90)
RETURNS void AS $$
BEGIN
    -- Archive system metrics older than retention period
    DELETE FROM metrics.system_metrics 
    WHERE recorded_at < NOW() - INTERVAL '1 day' * retention_days;
    
    -- Could add logic to move to archive table instead of delete
END;
$$ LANGUAGE plpgsql;

-- Grant permissions on tables and functions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA users TO causal_ui_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA experiments TO causal_ui_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA metrics TO causal_ui_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA audit TO causal_ui_app;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA users TO causal_ui_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA experiments TO causal_ui_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA metrics TO causal_ui_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO causal_ui_app;

GRANT EXECUTE ON FUNCTION cleanup_expired_sessions() TO causal_ui_app;
GRANT EXECUTE ON FUNCTION archive_old_metrics(INTEGER) TO causal_ui_app;

-- Insert some initial data for development/testing
INSERT INTO users.users (username, email, password_hash, first_name, last_name, is_verified)
VALUES 
    ('admin', 'admin@causal-ui-gym.dev', crypt('admin123', gen_salt('bf')), 'Admin', 'User', TRUE),
    ('demo', 'demo@causal-ui-gym.dev', crypt('demo123', gen_salt('bf')), 'Demo', 'User', TRUE)
ON CONFLICT (username) DO NOTHING;

-- Create a sample experiment
DO $$
DECLARE
    admin_id UUID;
    experiment_id UUID;
BEGIN
    SELECT id INTO admin_id FROM users.users WHERE username = 'admin';
    
    INSERT INTO experiments.experiments (name, description, created_by, causal_model, status, is_public)
    VALUES (
        'Sample Pricing Experiment',
        'A demonstration experiment showing price-demand-revenue relationships',
        admin_id,
        '{
            "nodes": [
                {"id": "price", "label": "Price", "type": "continuous"},
                {"id": "demand", "label": "Demand", "type": "continuous"},
                {"id": "revenue", "label": "Revenue", "type": "continuous"}
            ],
            "edges": [
                {"from": "price", "to": "demand", "relationship": "negative"},
                {"from": "price", "to": "revenue", "relationship": "positive"},
                {"from": "demand", "to": "revenue", "relationship": "positive"}
            ]
        }'::jsonb,
        'active',
        TRUE
    )
    RETURNING id INTO experiment_id;
    
    -- Add a sample intervention
    INSERT INTO experiments.interventions (experiment_id, variable_name, intervention_value, performed_by, result, computation_time_ms)
    VALUES (
        experiment_id,
        'price',
        '50'::jsonb,
        admin_id,
        '{"demand": 75, "revenue": 3750}'::jsonb,
        234
    );
END $$;

-- Create database maintenance scheduled job function (requires pg_cron extension)
-- This would need to be enabled separately in production
-- SELECT cron.schedule('cleanup-sessions', '0 2 * * *', 'SELECT cleanup_expired_sessions();');
-- SELECT cron.schedule('archive-metrics', '0 3 * * 0', 'SELECT archive_old_metrics(90);');

-- Log successful initialization
INSERT INTO audit.audit_log (action, resource_type, new_values)
VALUES ('database_initialized', 'system', '{"version": "1.0", "timestamp": "' || NOW() || '"}'::jsonb);