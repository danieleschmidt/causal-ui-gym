"""
Application Initialization Module for Causal UI Gym

Handles startup tasks, data seeding, and system preparation for production deployment.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_application():
    """Initialize the application with required data and configurations."""
    logger.info("Starting application initialization...")
    
    try:
        # Create necessary directories
        _create_directories()
        
        # Initialize logging configuration
        _setup_logging()
        
        # Set up metrics collection
        _setup_metrics()
        
        # Initialize cache if enabled
        _initialize_cache()
        
        # Run database setup if configured
        _setup_database()
        
        # Load initial configuration
        _load_configuration()
        
        logger.info("Application initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        raise

def _create_directories():
    """Create necessary directories for the application."""
    directories = [
        '/app/logs',
        '/app/tmp',
        '/app/data',
        '/app/cache',
        '/app/uploads'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def _setup_logging():
    """Set up logging configuration."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('LOG_FORMAT', 'standard')
    
    if log_format == 'json':
        # JSON structured logging for production
        import json
        import structlog
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    logger.info(f"Logging configured: level={log_level}, format={log_format}")

def _setup_metrics():
    """Initialize metrics collection."""
    if os.getenv('METRICS_ENABLED', 'true').lower() == 'true':
        try:
            from prometheus_client import CollectorRegistry, multiprocess, generate_latest
            
            # Set up multiprocess metrics directory
            metrics_dir = os.getenv('PROMETHEUS_MULTIPROC_DIR', '/tmp/prometheus')
            os.makedirs(metrics_dir, exist_ok=True)
            
            logger.info("Metrics collection initialized")
        except ImportError:
            logger.warning("Prometheus client not available, metrics disabled")

def _initialize_cache():
    """Initialize cache system."""
    if os.getenv('REDIS_ENABLED', 'false').lower() == 'true':
        try:
            import redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            
            # Test Redis connection
            r = redis.from_url(redis_url)
            r.ping()
            
            logger.info("Redis cache initialized successfully")
        except ImportError:
            logger.warning("Redis client not available")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")

def _setup_database():
    """Set up database if configured."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        logger.info("No database URL configured, skipping database setup")
        return
    
    try:
        import psycopg2
        
        # Test database connection
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        cursor.execute('SELECT version()')
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        
        logger.info(f"Database connection successful: {version[0]}")
        
    except ImportError:
        logger.warning("psycopg2 not available, skipping database setup")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        # Don't raise here as database might not be ready yet

def _load_configuration():
    """Load and validate application configuration."""
    config = {
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'debug': os.getenv('DEBUG', 'false').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'info'),
        'api_workers': int(os.getenv('WORKER_COUNT', '4')),
        'max_request_size': int(os.getenv('MAX_REQUEST_SIZE', '10485760')),  # 10MB
        'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '30')),
        'cors_origins': os.getenv('CORS_ORIGINS', '').split(',') if os.getenv('CORS_ORIGINS') else [],
        'rate_limit_enabled': os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true',
        'metrics_enabled': os.getenv('METRICS_ENABLED', 'true').lower() == 'true',
    }
    
    # Validate critical configuration
    if config['environment'] == 'production':
        required_vars = ['SECRET_KEY', 'JWT_SECRET']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info(f"Configuration loaded: environment={config['environment']}")
    
    # Save configuration for runtime access
    config_path = '/app/runtime_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Runtime configuration saved to {config_path}")

if __name__ == "__main__":
    initialize_application()