"""
Backend Metrics Collection Framework for Causal UI Gym

Provides comprehensive metrics collection using Prometheus client library.
Integrates with FastAPI for HTTP metrics and custom application metrics.
"""

import time
import functools
import logging
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, multiprocess, generate_latest,
        CONTENT_TYPE_LATEST, REGISTRY
    )
    from prometheus_client.multiprocess import MultiProcessCollector
except ImportError:
    # Fallback for development without prometheus_client
    print("Warning: prometheus_client not installed. Metrics will be logged only.")
    Counter = Histogram = Gauge = Summary = Info = None
    CollectorRegistry = multiprocess = generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None

import jax
import jax.numpy as jnp
from datetime import datetime, timezone
import psutil
import os


class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    name: str
    help: str
    labels: List[str]
    metric_type: MetricType


class ApplicationMetrics:
    """Centralized metrics collection for the Causal UI Gym backend."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self.logger = logging.getLogger(__name__)
        self._metrics: Dict[str, Any] = {}
        self._setup_default_metrics()
        self._setup_system_metrics()
        
    def _setup_default_metrics(self):
        """Initialize standard application metrics."""
        if not Counter:
            return
            
        # HTTP Request Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Causal Analysis Metrics
        self.causal_graph_operations = Counter(
            'causal_graph_operations_total',
            'Total causal graph operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.causal_graph_size = Histogram(
            'causal_graph_size',
            'Size of causal graphs (nodes + edges)',
            ['operation'],
            buckets=[10, 50, 100, 500, 1000, 5000],
            registry=self.registry
        )
        
        self.causal_inference_duration = Histogram(
            'causal_inference_duration_seconds',
            'Duration of causal inference computations',
            ['algorithm', 'dataset_size_bucket'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # JAX/ML Metrics
        self.jax_computation_duration = Histogram(
            'jax_computation_duration_seconds',
            'JAX computation duration',
            ['computation_type', 'device'],
            registry=self.registry
        )
        
        self.jax_memory_usage = Gauge(
            'jax_memory_usage_bytes',
            'JAX memory usage in bytes',
            ['device'],
            registry=self.registry
        )
        
        # Experiment Metrics
        self.experiments_total = Counter(
            'experiments_total',
            'Total experiments executed',
            ['experiment_type', 'status'],
            registry=self.registry
        )
        
        self.experiment_duration = Histogram(
            'experiment_duration_seconds',
            'Experiment execution duration',
            ['experiment_type'],
            registry=self.registry
        )
        
        self.experiment_sample_size = Histogram(
            'experiment_sample_size',
            'Number of samples in experiments',
            ['experiment_type'],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=self.registry
        )
        
        # Database Metrics
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['query_type', 'table'],
            registry=self.registry
        )
        
        # Error Metrics
        self.application_errors = Counter(
            'application_errors_total',
            'Total application errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Performance Metrics
        self.active_users = Gauge(
            'active_users',
            'Current number of active users',
            registry=self.registry
        )
        
        self.response_size = Histogram(
            'response_size_bytes',
            'HTTP response size in bytes',
            ['endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=self.registry
        )

    def _setup_system_metrics(self):
        """Initialize system-level metrics."""
        if not Gauge:
            return
            
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        # Start background system metrics collection
        import threading
        self._system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        self._system_metrics_thread.start()

    def _collect_system_metrics(self):
        """Background thread for system metrics collection."""
        while True:
            try:
                if self.system_cpu_usage:
                    self.system_cpu_usage.set(psutil.cpu_percent(interval=1))
                
                if self.system_memory_usage:
                    memory = psutil.virtual_memory()
                    self.system_memory_usage.labels(type='used').set(memory.used)
                    self.system_memory_usage.labels(type='available').set(memory.available)
                    self.system_memory_usage.labels(type='total').set(memory.total)
                
                if self.system_disk_usage:
                    disk = psutil.disk_usage('/')
                    self.system_disk_usage.labels(type='used').set(disk.used)
                    self.system_disk_usage.labels(type='free').set(disk.free)
                    self.system_disk_usage.labels(type='total').set(disk.total)
                
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.warning(f"System metrics collection failed: {e}")
                time.sleep(60)  # Wait longer before retry

    # HTTP Metrics Methods
    def track_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Track HTTP request metrics."""
        if self.http_requests_total:
            self.http_requests_total.labels(
                method=method.upper(),
                endpoint=self._sanitize_endpoint(endpoint),
                status_code=str(status_code)
            ).inc()
        
        if self.http_request_duration:
            self.http_request_duration.labels(
                method=method.upper(),
                endpoint=self._sanitize_endpoint(endpoint)
            ).observe(duration)

    def track_response_size(self, endpoint: str, size_bytes: int):
        """Track HTTP response size."""
        if self.response_size:
            self.response_size.labels(
                endpoint=self._sanitize_endpoint(endpoint)
            ).observe(size_bytes)

    # Causal Analysis Metrics
    def track_causal_graph_operation(self, operation: str, node_count: int, edge_count: int, success: bool):
        """Track causal graph operations."""
        status = "success" if success else "error"
        
        if self.causal_graph_operations:
            self.causal_graph_operations.labels(
                operation=operation,
                status=status
            ).inc()
        
        if self.causal_graph_size:
            self.causal_graph_size.labels(operation=operation).observe(node_count + edge_count)

    @contextmanager
    def track_causal_inference(self, algorithm: str, dataset_size: int):
        """Context manager for tracking causal inference computations."""
        start_time = time.time()
        dataset_bucket = self._get_size_bucket(dataset_size, [100, 1000, 10000, 100000])
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            self.track_error("causal_inference_error", str(e), f"algorithm_{algorithm}")
            raise
        finally:
            duration = time.time() - start_time
            if self.causal_inference_duration:
                self.causal_inference_duration.labels(
                    algorithm=algorithm,
                    dataset_size_bucket=dataset_bucket
                ).observe(duration)

    # JAX/ML Metrics
    @contextmanager
    def track_jax_computation(self, computation_type: str):
        """Context manager for tracking JAX computations."""
        start_time = time.time()
        device = str(jax.devices()[0]).split(':')[0] if jax.devices() else "unknown"
        
        # Track memory before computation
        try:
            if hasattr(jax, 'profiler'):
                memory_before = self._get_jax_memory_usage()
        except:
            memory_before = 0
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            self.track_error("jax_computation_error", str(e), computation_type)
            raise
        finally:
            duration = time.time() - start_time
            if self.jax_computation_duration:
                self.jax_computation_duration.labels(
                    computation_type=computation_type,
                    device=device
                ).observe(duration)
            
            # Track memory after computation
            try:
                memory_after = self._get_jax_memory_usage()
                if self.jax_memory_usage and memory_after > 0:
                    self.jax_memory_usage.labels(device=device).set(memory_after)
            except:
                pass

    def _get_jax_memory_usage(self) -> int:
        """Get current JAX memory usage."""
        try:
            # This is a simplified version - actual implementation would depend on JAX version
            devices = jax.devices()
            if devices and hasattr(devices[0], 'memory_stats'):
                return devices[0].memory_stats()['bytes_in_use']
        except:
            pass
        return 0

    # Experiment Metrics
    @contextmanager
    def track_experiment(self, experiment_type: str, sample_size: int):
        """Context manager for tracking experiments."""
        start_time = time.time()
        
        try:
            yield
            status = "success"
        except Exception as e:
            status = "error"
            self.track_error("experiment_error", str(e), experiment_type)
            raise
        finally:
            duration = time.time() - start_time
            
            if self.experiments_total:
                self.experiments_total.labels(
                    experiment_type=experiment_type,
                    status=status
                ).inc()
            
            if self.experiment_duration:
                self.experiment_duration.labels(
                    experiment_type=experiment_type
                ).observe(duration)
            
            if self.experiment_sample_size:
                self.experiment_sample_size.labels(
                    experiment_type=experiment_type
                ).observe(sample_size)

    # Database Metrics
    @contextmanager
    def track_database_query(self, query_type: str, table: str):
        """Context manager for tracking database queries."""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            if self.database_query_duration:
                self.database_query_duration.labels(
                    query_type=query_type,
                    table=table
                ).observe(duration)

    def set_database_connections(self, count: int):
        """Set current database connection count."""
        if self.database_connections:
            self.database_connections.set(count)

    # Error Tracking
    def track_error(self, error_type: str, error_message: str, component: str):
        """Track application errors."""
        if self.application_errors:
            self.application_errors.labels(
                error_type=error_type,
                component=component
            ).inc()
        
        # Log error details
        self.logger.error(f"Application Error: {error_type} in {component}: {error_message}")

    # User Activity
    def set_active_users(self, count: int):
        """Set current active user count."""
        if self.active_users:
            self.active_users.set(count)

    # Utility Methods
    def _sanitize_endpoint(self, endpoint: str) -> str:
        """Sanitize endpoint to avoid high cardinality."""
        # Replace IDs with placeholders
        import re
        endpoint = re.sub(r'/\d+', '/:id', endpoint)
        endpoint = re.sub(r'/[a-f0-9-]{36}', '/:uuid', endpoint)
        endpoint = re.sub(r'/[a-zA-Z0-9]{20,}', '/:hash', endpoint)
        return endpoint

    def _get_size_bucket(self, size: int, buckets: List[int]) -> str:
        """Get size bucket for histogram labels."""
        for bucket in buckets:
            if size <= bucket:
                return f"le_{bucket}"
        return f"gt_{buckets[-1]}"

    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if not generate_latest:
            return "# Prometheus client not available\n"
        
        return generate_latest(self.registry).decode('utf-8')


# Global metrics instance
metrics = ApplicationMetrics()


# Decorators for easy metrics integration
def track_duration(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track function execution duration."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # This would integrate with the metrics system
                logging.info(f"{metric_name} duration: {duration:.3f}s")
        return wrapper
    return decorator


def track_errors(component: str):
    """Decorator to track function errors."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                metrics.track_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    component=component
                )
                raise
        return wrapper
    return decorator


# FastAPI middleware for automatic HTTP metrics
class MetricsMiddleware:
    """FastAPI middleware for automatic HTTP metrics collection."""
    
    def __init__(self, app, metrics_instance: ApplicationMetrics):
        self.app = app
        self.metrics = metrics_instance

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        method = scope["method"]
        path = scope["path"]

        # Wrap send to capture response
        response_size = 0
        status_code = 500  # Default to 500 in case of error

        async def send_with_metrics(message):
            nonlocal response_size, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_size += len(message.get("body", b""))
            
            await send(message)

        try:
            await self.app(scope, receive, send_with_metrics)
        finally:
            duration = time.time() - start_time
            self.metrics.track_http_request(method, path, status_code, duration)
            self.metrics.track_response_size(path, response_size)


# Example usage functions
def setup_fastapi_metrics(app):
    """Setup FastAPI with metrics middleware."""
    app.add_middleware(MetricsMiddleware, metrics_instance=metrics)
    
    @app.get("/metrics")
    async def get_metrics():
        """Endpoint to expose Prometheus metrics."""
        return Response(
            content=metrics.get_metrics(),
            media_type=CONTENT_TYPE_LATEST
        )


# Testing utilities
class MockMetrics(ApplicationMetrics):
    """Mock metrics class for testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recorded_metrics = []
    
    def _record_metric(self, metric_type: str, name: str, value: float, labels: Dict[str, str]):
        """Record metric for testing."""
        self.recorded_metrics.append({
            'type': metric_type,
            'name': name,
            'value': value,
            'labels': labels,
            'timestamp': datetime.now(timezone.utc)
        })
    
    def get_recorded_metrics(self) -> List[Dict]:
        """Get all recorded metrics for testing."""
        return self.recorded_metrics.copy()
    
    def clear_metrics(self):
        """Clear recorded metrics."""
        self.recorded_metrics.clear()


if __name__ == "__main__":
    # Example usage
    print("Metrics framework initialized")
    print(f"Available metrics: {len(metrics._metrics)}")
    
    # Example causal inference tracking
    with metrics.track_causal_inference("pc_algorithm", 1000):
        time.sleep(0.1)  # Simulate computation
    
    # Example JAX computation tracking
    with metrics.track_jax_computation("matrix_multiplication"):
        # Simulate JAX computation
        x = jnp.array([1, 2, 3])
        y = jnp.dot(x, x)
    
    print("Example metrics recorded successfully")