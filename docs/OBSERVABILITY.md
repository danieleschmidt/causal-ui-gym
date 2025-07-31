# Observability and Monitoring

This document outlines the comprehensive observability setup for Causal UI Gym, including monitoring, logging, tracing, and alerting.

## Overview

Observability is implemented using the "Three Pillars":
- **Metrics**: Numerical data about system performance
- **Logs**: Textual records of events
- **Traces**: Request flow through distributed systems

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   OpenTelemetry │───▶│   Observability │
│   (Frontend +   │    │   Collector     │    │   Backend       │
│    Backend)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                              ┌───────┴───────┐
                                              ▼               ▼
                                       ┌─────────────┐ ┌─────────────┐
                                       │  Metrics    │ │  Logs &     │
                                       │  (Prometheus│ │  Traces     │
                                       │   + Grafana)│ │  (Jaeger)   │
                                       └─────────────┘ └─────────────┘
```

## Frontend Monitoring (React/TypeScript)

### Performance Monitoring

```typescript
// src/utils/monitoring.ts
import { trace, metrics, SpanStatusCode } from '@opentelemetry/api'
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web'
import { getWebAutoInstrumentations } from '@opentelemetry/auto-instrumentations-web'
import { Resource } from '@opentelemetry/resources'
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions'

const provider = new WebTracerProvider({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'causal-ui-gym-frontend',
    [SemanticResourceAttributes.SERVICE_VERSION]: process.env.VITE_APP_VERSION || '0.1.0',
  }),
})

// Initialize tracing
provider.addSpanProcessor(new BatchSpanProcessor(new OTLPTraceExporter({
  url: 'http://localhost:4318/v1/traces',
})))

provider.register({
  instrumentations: [getWebAutoInstrumentations()],
})

// Custom metrics
const meter = metrics.getMeter('causal-ui-gym-frontend')

export const causalExperimentCounter = meter.createCounter('causal_experiments_total', {
  description: 'Total number of causal experiments performed',
})

export const interventionLatencyHistogram = meter.createHistogram('intervention_latency_ms', {
  description: 'Latency of causal interventions in milliseconds',
})

export const llmRequestCounter = meter.createCounter('llm_requests_total', {
  description: 'Total number of LLM API requests',
})

export const errorCounter = meter.createCounter('frontend_errors_total', {
  description: 'Total number of frontend errors',
})
```

### React Component Monitoring

```typescript
// src/components/monitoring/PerformanceMonitor.tsx
import React, { useEffect } from 'react'
import { trace } from '@opentelemetry/api'

const tracer = trace.getTracer('causal-ui-gym-components')

export function withPerformanceMonitoring<T extends object>(
  Component: React.ComponentType<T>,
  componentName: string
) {
  return function MonitoredComponent(props: T) {
    useEffect(() => {
      const span = tracer.startSpan(`component.${componentName}.mount`)
      
      return () => {
        span.end()
      }
    }, [])

    return <Component {...props} />
  }
}

// Usage example
export const MonitoredCausalGraph = withPerformanceMonitoring(CausalGraph, 'CausalGraph')
```

### Error Monitoring

```typescript
// src/utils/errorMonitoring.ts
import { errorCounter } from './monitoring'

export class ErrorBoundaryWithTelemetry extends React.Component {
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    const span = tracer.startSpan('error.component')
    span.recordException(error)
    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message })
    
    errorCounter.add(1, {
      component: errorInfo.componentStack?.split('\n')[1] || 'unknown',
      error_type: error.name,
    })
    
    span.end()
  }
}
```

## Backend Monitoring (Python/JAX)

### FastAPI Instrumentation

```python
# backend/monitoring/instrumentation.py
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.jinja2 import Jinja2Instrumentor
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
trace.get_tracer_provider().add_span_processor(span_processor)

# Metrics
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="http://localhost:4317"),
    export_interval_millis=5000,
)
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
meter = metrics.get_meter(__name__)

# Custom metrics
causal_computation_counter = meter.create_counter(
    "causal_computations_total",
    description="Total number of causal computations performed"
)

intervention_latency = meter.create_histogram(
    "intervention_computation_duration_ms",
    description="Duration of causal intervention computations"
)

llm_api_calls = meter.create_counter(
    "llm_api_calls_total",
    description="Total number of LLM API calls"
)

jax_compilation_time = meter.create_histogram(
    "jax_compilation_duration_ms",
    description="JAX JIT compilation time"
)

# Prometheus metrics for compatibility
CAUSAL_OPERATIONS = Counter('causal_operations_total', 'Total causal operations', ['operation_type', 'status'])
INTERVENTION_DURATION = Histogram('intervention_duration_seconds', 'Intervention computation duration')
ACTIVE_EXPERIMENTS = Gauge('active_experiments', 'Number of active experiments')
LLM_RESPONSE_TIME = Histogram('llm_response_time_seconds', 'LLM API response time', ['provider', 'model'])

def setup_monitoring(app):
    """Setup monitoring for FastAPI application"""
    # Auto-instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    
    # Start Prometheus metrics server
    start_http_server(8001)  # Metrics available at :8001/metrics
    
    return app
```

### JAX Computation Monitoring

```python
# backend/causal/monitored_engine.py
import jax
import jax.numpy as jnp
from functools import wraps
import time
from .instrumentation import (
    causal_computation_counter, 
    intervention_latency, 
    jax_compilation_time,
    CAUSAL_OPERATIONS,
    INTERVENTION_DURATION,
    tracer,
    logger
)

def monitor_jax_computation(operation_name: str):
    """Decorator to monitor JAX computations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"jax.{operation_name}") as span:
                start_time = time.time()
                
                try:
                    # Check if function needs compilation
                    if not hasattr(func, '_compiled'):
                        compile_start = time.time()
                        result = func(*args, **kwargs)
                        compile_duration = (time.time() - compile_start) * 1000
                        
                        jax_compilation_time.record(compile_duration)
                        span.set_attribute("jax.compiled", True)
                        span.set_attribute("jax.compilation_time_ms", compile_duration)
                        
                        func._compiled = True
                    else:
                        result = func(*args, **kwargs)
                        span.set_attribute("jax.compiled", False)
                    
                    duration = (time.time() - start_time) * 1000
                    
                    # Record metrics
                    causal_computation_counter.add(1, {"operation": operation_name})
                    intervention_latency.record(duration)
                    
                    # Prometheus metrics
                    CAUSAL_OPERATIONS.labels(operation_type=operation_name, status='success').inc()
                    INTERVENTION_DURATION.observe(duration / 1000)
                    
                    # Structured logging
                    logger.info(
                        "JAX computation completed",
                        operation=operation_name,
                        duration_ms=duration,
                        compiled=hasattr(func, '_compiled')
                    )
                    
                    span.set_attribute("operation.duration_ms", duration)
                    span.set_attribute("operation.status", "success")
                    
                    return result
                    
                except Exception as e:
                    CAUSAL_OPERATIONS.labels(operation_type=operation_name, status='error').inc()
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    
                    logger.error(
                        "JAX computation failed",
                        operation=operation_name,
                        error=str(e),
                        duration_ms=(time.time() - start_time) * 1000
                    )
                    
                    raise
                    
        return wrapper
    return decorator

# Apply monitoring to causal computations
@jax.jit
@monitor_jax_computation("intervention")
def compute_intervention(dag, intervention, evidence):
    """Monitored causal intervention computation"""
    # Implementation here
    pass

@jax.jit
@monitor_jax_computation("ate")
def compute_ate(dag, treatment, outcome, covariates):
    """Monitored ATE computation"""
    # Implementation here
    pass
```

## Infrastructure Monitoring

### Docker Compose with Observability Stack

```yaml
# docker-compose.observability.yml
version: '3.8'

services:
  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yml"]
    volumes:
      - ./observability/otel-collector-config.yml:/etc/otel-collector-config.yml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8889:8889"   # Prometheus metrics
    depends_on:
      - prometheus
      - jaeger

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./observability/grafana/dashboards:/var/lib/grafana/dashboards
      - ./observability/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana

  # Jaeger
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  # Loki for logs
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./observability/loki-config.yml:/etc/loki/local-config.yaml

  # Promtail for log collection
  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./observability/promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml

volumes:
  prometheus_data:
  grafana_data:
```

### OpenTelemetry Collector Configuration

```yaml
# observability/otel-collector-config.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
  
  prometheus:
    config:
      scrape_configs:
        - job_name: 'causal-ui-gym-backend'
          static_configs:
            - targets: ['host.docker.internal:8001']

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  
  memory_limiter:
    limit_mib: 512
  
  resource:
    attributes:
      - key: service.namespace
        value: causal-ui-gym
        action: upsert

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
    
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
      
  loki:
    endpoint: http://loki:3100/loki/api/v1/push

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [jaeger]
      
    metrics:
      receivers: [otlp, prometheus]
      processors: [memory_limiter, resource, batch]
      exporters: [prometheus]
      
    logs:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [loki]
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Causal UI Gym - System Overview",
    "panels": [
      {
        "title": "Causal Computations Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(causal_computations_total[5m])",
            "legendFormat": "Computations/sec"
          }
        ]
      },
      {
        "title": "Intervention Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(intervention_computation_duration_ms_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(intervention_computation_duration_ms_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "LLM API Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_response_time_seconds_bucket[5m])) by (provider)",
            "legendFormat": "{{provider}} - 95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(frontend_errors_total[5m])",
            "legendFormat": "Frontend Errors"
          },
          {
            "expr": "rate(causal_operations_total{status=\"error\"}[5m])",
            "legendFormat": "Backend Errors"
          }
        ]
      }
    ]
  }
}
```

## Alerting Rules

### Prometheus Alerting Rules

```yaml
# observability/alert-rules.yml
groups:
  - name: causal-ui-gym-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(causal_operations_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in causal computations"
          description: "Error rate is {{ $value }} per second for the last 5 minutes"

      - alert: HighInterventionLatency
        expr: histogram_quantile(0.95, rate(intervention_computation_duration_ms_bucket[5m])) > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High intervention computation latency"
          description: "95th percentile latency is {{ $value }}ms"

      - alert: LLMAPIFailure
        expr: rate(llm_api_calls_total{status="error"}[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM API failures detected"
          description: "LLM API error rate is {{ $value }} per second"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"
```

## Health Checks and SLOs

### Application Health Endpoints

```python
# backend/api/health.py
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import time
import jax
from ..monitoring.instrumentation import ACTIVE_EXPERIMENTS

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/health/ready")
async def readiness_check():
    """Readiness check for K8s"""
    try:
        # Test JAX functionality
        test_array = jax.numpy.array([1, 2, 3])
        _ = jax.numpy.sum(test_array)
        
        # Test database connection (if applicable)
        # db_healthy = await check_database()
        
        return {
            "status": "ready",
            "jax": "healthy",
            "timestamp": time.time()
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "error": str(e)}
        )

@router.get("/health/metrics")
async def metrics_summary():
    """Metrics summary for monitoring"""
    return {
        "active_experiments": ACTIVE_EXPERIMENTS._value._value,
        "timestamp": time.time(),
        "version": "0.1.0"
    }
```

### Service Level Objectives (SLOs)

```yaml
# observability/slos.yml
slos:
  - name: intervention_latency
    description: "95% of intervention computations complete within 500ms"
    query: "histogram_quantile(0.95, rate(intervention_computation_duration_ms_bucket[5m])) < 500"
    target: 0.95
    
  - name: api_availability
    description: "99% of API requests succeed"
    query: "rate(http_requests_total{status!~'5..'}[5m]) / rate(http_requests_total[5m])"
    target: 0.99
    
  - name: llm_api_success
    description: "95% of LLM API calls succeed"
    query: "rate(llm_api_calls_total{status='success'}[5m]) / rate(llm_api_calls_total[5m])"
    target: 0.95
```

## Deployment and Configuration

### Monitoring Setup Script

```bash
#!/bin/bash
# scripts/setup-monitoring.sh

set -e

echo "🔍 Setting up observability stack..."

# Create directories
mkdir -p observability/{grafana/{dashboards,provisioning},prometheus,loki}

# Deploy observability stack
docker-compose -f docker-compose.observability.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Create Grafana dashboards
curl -X POST http://admin:admin@localhost:3001/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @observability/grafana/causal-ui-gym-dashboard.json

# Set up Prometheus alerts
curl -X POST http://localhost:9090/-/reload

echo "✅ Observability stack deployed!"
echo "📊 Grafana: http://localhost:3001 (admin/admin)"
echo "🔍 Prometheus: http://localhost:9090"
echo "📈 Jaeger: http://localhost:16686"
```

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)
- [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)
- [Structured Logging with Python](https://structlog.org/)

---

*This observability setup provides comprehensive monitoring for the Causal UI Gym application across all components and infrastructure layers.*