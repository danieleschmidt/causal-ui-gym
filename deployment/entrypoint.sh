#!/bin/bash
# Causal UI Gym Production Entrypoint Script
# Handles initialization, health checks, and graceful startup

set -euo pipefail

# Configuration
export PYTHONPATH="${PYTHONPATH:-/app}"
export ENVIRONMENT="${ENVIRONMENT:-production}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ENTRYPOINT] $1" >&2
}

log "Starting Causal UI Gym Production Container..."

# Check if running as root and switch to causalapp user if needed
if [ "$(id -u)" = "0" ]; then
    log "WARNING: Running as root, switching to causalapp user"
    exec gosu causalapp "$0" "$@"
fi

# Verify required environment variables
required_vars=("PYTHONPATH" "ENVIRONMENT")
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        log "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Create necessary directories
log "Creating directory structure..."
mkdir -p /app/logs /app/tmp /app/data /app/cache

# Set proper permissions
chmod 755 /app/logs /app/tmp /app/data /app/cache

# Load environment variables from template if .env doesn't exist
if [ ! -f /app/.env ] && [ -f /app/.env.template ]; then
    log "Creating .env from template..."
    cp /app/.env.template /app/.env
fi

# Source environment variables
if [ -f /app/.env ]; then
    log "Loading environment variables from .env file..."
    set -a
    source /app/.env
    set +a
fi

# Database migration check (if using a database)
if [ "${DATABASE_URL:-}" ]; then
    log "Checking database connectivity..."
    python -c "
import sys
sys.path.insert(0, '/app')
try:
    from backend.database import check_connection
    if check_connection():
        print('Database connection successful')
    else:
        print('Database connection failed')
        sys.exit(1)
except ImportError:
    print('No database module found, skipping check')
except Exception as e:
    print(f'Database check failed: {e}')
    sys.exit(1)
" || {
        log "ERROR: Database connectivity check failed"
        exit 1
    }
fi

# Health check for dependencies
log "Performing dependency health checks..."

# Check Python backend
python -c "
import sys
sys.path.insert(0, '/app')
try:
    from backend.api.server import app
    print('Backend API module loaded successfully')
except Exception as e:
    print(f'Backend API module load failed: {e}')
    sys.exit(1)
" || {
    log "ERROR: Backend API health check failed"
    exit 1
}

# Check frontend build
if [ ! -d "/app/frontend/dist" ] || [ ! -f "/app/frontend/dist/index.html" ]; then
    log "ERROR: Frontend build not found or incomplete"
    exit 1
fi

# Redis health check (if Redis is enabled)
if [ "${REDIS_ENABLED:-false}" = "true" ]; then
    log "Checking Redis availability..."
    timeout 5 bash -c "</dev/tcp/127.0.0.1/6379" || {
        log "WARNING: Redis not available, starting without cache"
        export REDIS_ENABLED=false
    }
fi

# Initialize application data
log "Initializing application data..."
python -c "
import sys
sys.path.insert(0, '/app')
try:
    from backend.initialization import initialize_application
    initialize_application()
    print('Application initialization completed')
except ImportError:
    print('No initialization module found, skipping')
except Exception as e:
    print(f'Application initialization failed: {e}')
    # Don't exit on initialization failure, just log it
" || log "WARNING: Application initialization had issues"

# Security hardening
log "Applying security hardening..."

# Set secure file permissions
find /app -type f -name "*.py" -exec chmod 644 {} \;
find /app -type f -name "*.sh" -exec chmod 755 {} \;
find /app -type d -exec chmod 755 {} \;

# Remove any temporary files
find /app -name "*.pyc" -delete 2>/dev/null || true
find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Set up signal handlers for graceful shutdown
trap 'log "Received SIGTERM, initiating graceful shutdown..."; kill -TERM $PID; wait $PID' TERM
trap 'log "Received SIGINT, initiating graceful shutdown..."; kill -INT $PID; wait $PID' INT

# Pre-start health checks
log "Running pre-start health checks..."

# Check disk space
available_space=$(df /app | awk 'NR==2 {print $4}')
if [ "$available_space" -lt 1000000 ]; then  # Less than ~1GB
    log "WARNING: Low disk space available: ${available_space}KB"
fi

# Check memory
available_memory=$(free -m | awk 'NR==2{print $7}')
if [ "$available_memory" -lt 512 ]; then  # Less than 512MB
    log "WARNING: Low memory available: ${available_memory}MB"
fi

# Validate configuration files
log "Validating configuration files..."
nginx -t || {
    log "ERROR: Nginx configuration validation failed"
    exit 1
}

supervisord -c /etc/supervisor/conf.d/supervisord.conf -t || {
    log "ERROR: Supervisor configuration validation failed"
    exit 1
}

# Performance tuning
log "Applying performance optimizations..."

# Set Python optimizations
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Set system limits (if running as root or with capabilities)
if [ -w /proc/sys ]; then
    echo 8192 > /proc/sys/net/core/somaxconn 2>/dev/null || true
    echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse 2>/dev/null || true
fi

# Warmup phase
log "Starting application warmup..."
timeout 30 python -c "
import sys
sys.path.insert(0, '/app')
try:
    from backend.warmup import warmup_application
    warmup_application()
    print('Application warmup completed successfully')
except ImportError:
    print('No warmup module found, skipping')
except Exception as e:
    print(f'Warmup failed: {e}')
" || log "WARNING: Application warmup had issues"

# Final health check before starting services
log "Final health check before service startup..."
python -c "
import sys, os
sys.path.insert(0, '/app')

# Check all critical components
checks = {
    'Python Backend': lambda: __import__('backend.api.server'),
    'Configuration': lambda: os.path.exists('/app/.env') or os.path.exists('/app/.env.template'),
    'Frontend Assets': lambda: os.path.exists('/app/frontend/dist/index.html'),
    'Log Directory': lambda: os.path.isdir('/app/logs') and os.access('/app/logs', os.W_OK),
}

failed_checks = []
for name, check in checks.items():
    try:
        if not check():
            failed_checks.append(name)
    except Exception as e:
        failed_checks.append(f'{name}: {e}')

if failed_checks:
    print(f'Health check failures: {failed_checks}')
    sys.exit(1)
else:
    print('All health checks passed')
" || {
    log "ERROR: Final health check failed"
    exit 1
}

# Start the main application
log "Starting services with supervisor..."
log "Configuration: Environment=$ENVIRONMENT, LogLevel=$LOG_LEVEL"
log "Services will be available on ports 80 (HTTP) and 8000 (API)"

# Execute the main command
exec "$@" &
PID=$!

# Wait for the process to complete
wait $PID
EXIT_CODE=$?

log "Application exited with code $EXIT_CODE"
exit $EXIT_CODE