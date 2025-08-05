"""
Application Warmup Module for Causal UI Gym

Pre-loads models, caches, and performs initial computations to reduce cold start latency.
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, List, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def warmup_application():
    """Perform application warmup tasks."""
    logger.info("Starting application warmup...")
    start_time = time.time()
    
    try:
        # Warmup JAX computation engine
        _warmup_jax_engine()
        
        # Pre-load models and algorithms
        _preload_causal_algorithms()
        
        # Warmup cache connections
        _warmup_cache()
        
        # Pre-compute common calculations
        _precompute_common_calculations()
        
        # Warmup HTTP client pools
        _warmup_http_clients()
        
        duration = time.time() - start_time
        logger.info(f"Application warmup completed in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Application warmup failed: {e}")
        # Don't raise - warmup failures shouldn't prevent startup

def _warmup_jax_engine():
    """Warmup JAX computation engine with sample operations."""
    try:
        import jax
        import jax.numpy as jnp
        
        logger.info("Warming up JAX computation engine...")
        
        # Simple matrix operations to trigger JIT compilation
        key = jax.random.PRNGKey(42)
        
        # Generate sample data
        X = jax.random.normal(key, (100, 5))
        y = jax.random.normal(key, (100,))
        
        # Perform common operations
        _ = jnp.dot(X.T, X)
        _ = jnp.linalg.inv(jnp.dot(X.T, X) + jnp.eye(5) * 0.01)
        _ = jnp.dot(X, jnp.ones(5))
        
        # Trigger random number generation
        _ = jax.random.normal(key, (1000, 10))
        
        logger.info("JAX engine warmup completed")
        
    except ImportError:
        logger.warning("JAX not available, skipping JAX warmup")
    except Exception as e:
        logger.warning(f"JAX warmup failed: {e}")

def _preload_causal_algorithms():
    """Pre-load and warmup causal inference algorithms."""
    try:
        logger.info("Pre-loading causal inference algorithms...")
        
        # Import and initialize causal engine
        sys.path.insert(0, '/app')
        from backend.engine.causal_engine import CausalEngine
        
        # Create engine instance
        engine = CausalEngine()
        
        # Create sample DAG for warmup
        sample_dag = {
            'nodes': [
                {'id': 'X', 'label': 'Treatment'},
                {'id': 'Y', 'label': 'Outcome'},
                {'id': 'Z', 'label': 'Confounder'}
            ],
            'edges': [
                {'source': 'X', 'target': 'Y', 'weight': 1.0},
                {'source': 'Z', 'target': 'X', 'weight': 0.5},
                {'source': 'Z', 'target': 'Y', 'weight': 0.3}
            ]
        }
        
        # Warmup common algorithms with small sample sizes
        if hasattr(engine, 'compute_do_intervention'):
            _ = engine.compute_do_intervention(
                sample_dag, 'X', 'Y', intervention_value=1.0, n_samples=100
            )
        
        if hasattr(engine, 'compute_ate'):
            _ = engine.compute_ate(
                sample_dag, 'X', 'Y', treatment_values=[0.0, 1.0], n_samples=100
            )
        
        logger.info("Causal algorithms pre-loaded successfully")
        
    except ImportError as e:
        logger.warning(f"Causal engine not available: {e}")
    except Exception as e:
        logger.warning(f"Causal algorithms warmup failed: {e}")

def _warmup_cache():
    """Warmup cache connections and test operations."""
    if os.getenv('REDIS_ENABLED', 'false').lower() != 'true':
        logger.info("Cache not enabled, skipping cache warmup")
        return
    
    try:
        import redis
        logger.info("Warming up cache connections...")
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        
        # Test basic operations
        test_key = 'warmup:test'
        r.set(test_key, 'warmup_value', ex=60)
        _ = r.get(test_key)
        r.delete(test_key)
        
        # Test pipeline operations
        pipe = r.pipeline()
        pipe.set('warmup:pipe1', 'value1', ex=60)
        pipe.set('warmup:pipe2', 'value2', ex=60)
        pipe.execute()
        
        # Cleanup
        r.delete('warmup:pipe1', 'warmup:pipe2')
        
        logger.info("Cache warmup completed")
        
    except ImportError:
        logger.warning("Redis client not available")
    except Exception as e:
        logger.warning(f"Cache warmup failed: {e}")

def _precompute_common_calculations():
    """Pre-compute common statistical calculations."""
    try:
        logger.info("Pre-computing common statistical calculations...")
        
        # Import statistical utilities
        sys.path.insert(0, '/app')
        from backend.utils.statistics import (
            compute_confidence_interval,
            compute_effect_size,
            compute_p_value
        )
        
        # Generate sample data for warmup
        np.random.seed(42)
        sample_data_1 = np.random.normal(0, 1, 100)
        sample_data_2 = np.random.normal(0.5, 1, 100)
        
        # Warmup statistical functions
        _ = compute_confidence_interval(sample_data_1)
        _ = compute_effect_size(sample_data_1, sample_data_2)
        _ = compute_p_value(sample_data_1, sample_data_2)
        
        logger.info("Statistical calculations pre-computed")
        
    except ImportError:
        logger.warning("Statistical utilities not available")
    except Exception as e:
        logger.warning(f"Statistical calculations warmup failed: {e}")

def _warmup_http_clients():
    """Warmup HTTP client connection pools."""
    try:
        import httpx
        import asyncio
        
        logger.info("Warming up HTTP client pools...")
        
        async def warmup_clients():
            # Create client with connection pooling
            async with httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                timeout=httpx.Timeout(10.0)
            ) as client:
                # Test internal health check
                try:
                    response = await client.get('http://localhost:8000/health')
                    logger.info(f"Internal health check: {response.status_code}")
                except Exception:
                    pass  # Service might not be ready yet
        
        # Run async warmup
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop:
            loop.run_until_complete(warmup_clients())
        
        logger.info("HTTP client warmup completed")
        
    except ImportError:
        logger.warning("HTTP client libraries not available")
    except Exception as e:
        logger.warning(f"HTTP client warmup failed: {e}")

if __name__ == "__main__":
    warmup_application()