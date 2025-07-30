# Monitoring and Observability

## Overview

This document describes the monitoring and observability setup for Causal UI Gym, including metrics collection, logging, tracing, and alerting strategies.

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Application   │───▶│  Prometheus  │───▶│    Grafana      │
│   (Metrics)     │    │  (Storage)   │    │ (Visualization) │
└─────────────────┘    └──────────────┘    └─────────────────┘

┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Application   │───▶│   Jaeger     │───▶│   Trace UI      │
│   (Traces)      │    │  (Storage)   │    │   (Analysis)    │
└─────────────────┘    └──────────────┘    └─────────────────┘

┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Application   │───▶│ Elasticsearch│───▶│    Kibana       │
│    (Logs)       │    │   (Storage)  │    │ (Log Analysis)  │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

## Metrics Collection

### Application Metrics

#### Frontend Metrics (React/TypeScript)
- **User Interactions**: Button clicks, form submissions, navigation
- **Performance**: Page load times, component render times
- **Errors**: JavaScript errors, API failures
- **Business Logic**: Causal graph interactions, intervention tracking

#### Backend Metrics (Python/JAX)
- **API Performance**: Request duration, throughput, error rates
- **Computation Metrics**: Causal inference execution time, memory usage
- **LLM Integration**: API call latency, token usage, costs
- **Database**: Query performance, connection pool usage

### Infrastructure Metrics
- **System Resources**: CPU, memory, disk usage
- **Network**: Bandwidth, latency, packet loss
- **Container Metrics**: Docker container performance
- **Database**: PostgreSQL performance, Redis cache hit rates

### Custom Metrics Implementation

#### Frontend (TypeScript)
```typescript
// metrics/client.ts
import { createMetricsClient } from '@prometheus/client';

class CausalUIMetrics {
  private static instance: CausalUIMetrics;
  private metricsClient: any;

  constructor() {
    this.metricsClient = createMetricsClient({
      endpoint: '/api/metrics',
      labels: {
        app: 'causal-ui-gym',
        version: process.env.VITE_APP_VERSION
      }
    });
  }

  // Track causal graph interactions
  trackCausalGraphInteraction(action: string, nodeCount: number) {
    this.metricsClient.increment('causal_graph_interactions_total', {
      action,
      node_count: nodeCount.toString()
    });
  }

  // Track intervention performance
  trackInterventionTime(duration: number, successful: boolean) {
    this.metricsClient.histogram('intervention_duration_seconds', duration, {
      status: successful ? 'success' : 'failure'
    });
  }

  // Track LLM belief updates
  trackLLMBeliefUpdate(agent: string, beliefType: string) {
    this.metricsClient.increment('llm_belief_updates_total', {
      agent,
      belief_type: beliefType
    });
  }
}

export const metrics = new CausalUIMetrics();
```

#### Backend (Python)
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

CAUSAL_COMPUTATION_TIME = Histogram(
    'causal_computation_duration_seconds',
    'Time spent on causal computations',
    ['computation_type', 'node_count']
)

LLM_API_CALLS = Counter(
    'llm_api_calls_total',
    'Total LLM API calls',
    ['provider', 'model', 'status']
)

ACTIVE_EXPERIMENTS = Gauge(
    'active_experiments',
    'Number of active causal experiments'
)

def track_request_metrics(func):
    """Decorator to track HTTP request metrics."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        status = 'success'
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            status = 'error'
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.endpoint
            ).observe(duration)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint,
                status=status
            ).inc()
    
    return wrapper

def track_causal_computation(computation_type: str):
    """Decorator to track causal computation performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract node count from arguments
            node_count = len(kwargs.get('nodes', []))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                CAUSAL_COMPUTATION_TIME.labels(
                    computation_type=computation_type,
                    node_count=str(node_count)
                ).observe(duration)
        
        return wrapper
    
    return decorator
```

## Logging Strategy

### Structured Logging
- **Format**: JSON for machine parsing, human-readable for development
- **Levels**: DEBUG, INFO, WARN, ERROR, CRITICAL
- **Context**: Request IDs, user IDs, session information
- **Sampling**: High-volume logs with intelligent sampling

### Log Aggregation
```python
# logging/config.py
import logging
import json
from pythonjsonlogger import jsonlogger

class CausalUILogFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['service'] = 'causal-ui-gym'
        log_record['version'] = os.getenv('APP_VERSION', 'unknown')
        log_record['environment'] = os.getenv('ENVIRONMENT', 'development')

def setup_logging():
    """Configure structured logging."""
    handler = logging.StreamHandler()
    formatter = CausalUILogFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
```

## Distributed Tracing

### OpenTelemetry Integration
```python
# tracing/setup.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing():
    """Initialize distributed tracing."""
    trace.set_tracer_provider(TracerProvider())
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

# Usage in causal computation
@trace.get_tracer(__name__).start_as_current_span("causal_inference")
def compute_causal_effect(dag, intervention):
    """Compute causal effect with tracing."""
    with trace.get_tracer(__name__).start_as_current_span("mutilate_graph"):
        mutilated_dag = mutilate_graph(dag, intervention)
    
    with trace.get_tracer(__name__).start_as_current_span("inference"):
        result = perform_inference(mutilated_dag)
    
    return result
```

## Alerting Rules

### Critical Alerts
- **Service Down**: Application unavailable for >2 minutes
- **High Error Rate**: >5% error rate for >5 minutes
- **Response Time**: P95 response time >2 seconds for >5 minutes
- **Memory Usage**: >90% memory usage for >10 minutes

### Warning Alerts
- **Elevated Error Rate**: >2% error rate for >10 minutes
- **Slow Responses**: P95 response time >1 second for >10 minutes
- **LLM API Failures**: >10% LLM API failure rate
- **Database Slow Queries**: Query time >1 second

### Business Logic Alerts
- **Causal Computation Failures**: >5% computation failures
- **Abnormal User Behavior**: Unusual experiment patterns
- **Resource Exhaustion**: JAX computation resource limits

## Dashboard Configuration

### Application Dashboard
- **Request Rate**: Requests per second over time
- **Response Time**: P50, P95, P99 response times
- **Error Rate**: Error percentage and count
- **Active Users**: Current active experiments

### Infrastructure Dashboard
- **System Resources**: CPU, memory, disk usage
- **Network Metrics**: Bandwidth, latency
- **Database Performance**: Query performance, connections
- **Container Health**: Docker container status

### Business Metrics Dashboard
- **Experiment Metrics**: Active experiments, completion rates
- **Causal Analysis**: Computation success rates, performance
- **LLM Usage**: API calls, costs, response quality
- **User Engagement**: Feature usage, session duration

## Health Checks

### Application Health
```python
# health/checks.py
from fastapi import APIRouter
import asyncio

router = APIRouter()

@router.get("/health")
async def health_check():
    """Comprehensive health check."""
    checks = {
        'database': await check_database(),
        'redis': await check_redis(),
        'llm_apis': await check_llm_apis(),
        'jax_backend': await check_jax_backend()
    }
    
    healthy = all(checks.values())
    status_code = 200 if healthy else 503
    
    return {
        'status': 'healthy' if healthy else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }

async def check_database():
    """Check database connectivity."""
    try:
        # Simple query to verify database
        result = await db.execute("SELECT 1")
        return True
    except Exception:
        return False

async def check_jax_backend():
    """Check JAX computational backend."""
    try:
        import jax.numpy as jnp
        # Simple computation test
        result = jnp.array([1, 2, 3]).sum()
        return result == 6
    except Exception:
        return False
```

## Performance Monitoring

### Frontend Performance
- **Core Web Vitals**: LCP, FID, CLS tracking
- **Bundle Size**: JavaScript bundle analysis
- **Runtime Performance**: Component render times
- **Network Performance**: API call latencies

### Backend Performance
- **Causal Computation**: JAX performance profiling
- **Database Performance**: Query optimization tracking
- **Memory Profiling**: Python memory usage analysis
- **CPU Profiling**: Computational bottleneck identification

## Data Retention

### Metrics Retention
- **High Resolution**: 30 days (1-minute intervals)
- **Medium Resolution**: 90 days (5-minute intervals)
- **Low Resolution**: 1 year (1-hour intervals)

### Log Retention
- **Application Logs**: 30 days
- **Error Logs**: 90 days
- **Audit Logs**: 1 year
- **Debug Logs**: 7 days

### Trace Retention
- **All Traces**: 7 days
- **Error Traces**: 30 days
- **Sampled Traces**: 90 days

## Deployment

### Docker Compose Setup
```yaml
# monitoring/docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
```

## Monitoring Checklist

### Initial Setup
- [ ] Prometheus metrics collection
- [ ] Grafana dashboard configuration
- [ ] Jaeger tracing setup
- [ ] Log aggregation pipeline
- [ ] Alert rule configuration

### Ongoing Maintenance
- [ ] Regular dashboard review
- [ ] Alert rule tuning
- [ ] Performance baseline updates
- [ ] Retention policy adjustments
- [ ] Cost optimization review

---

*Last Updated: January 2025*  
*Review Schedule: Monthly*