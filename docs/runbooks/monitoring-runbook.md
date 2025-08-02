# Monitoring Runbook for Causal UI Gym

## Overview

This runbook provides operational procedures for monitoring, alerting, and troubleshooting the Causal UI Gym application.

## Monitoring Stack Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Applications  │───▶│  OpenTelemetry │───▶│   Prometheus    │
│                 │    │   Collector    │    │   (Metrics)     │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                      │                      │
         │                      ▼                      ▼
         │              ┌──────────────┐    ┌─────────────────┐
         │              │    Loki      │    │    Grafana      │
         │              │   (Logs)     │    │ (Visualization) │
         │              └──────────────┘    └─────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌──────────────┐
│     Jaeger      │    │  Promtail    │
│   (Traces)      │    │ (Log Shipper)│
└─────────────────┘    └──────────────┘
```

## Key Metrics and SLIs

### Application Performance
- **Request Rate**: `rate(http_requests_total[5m])`
- **Response Time P95**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])`
- **Availability**: `up{job="causal-ui-backend"}`

### Causal Computation Metrics
- **Computation Rate**: `rate(causal_computations_total[5m])`
- **Computation Latency**: `histogram_quantile(0.95, rate(causal_computation_duration_seconds_bucket[5m]))`
- **Accuracy**: `causal_inference_accuracy_ratio`
- **Memory Usage**: `process_resident_memory_bytes / 1024 / 1024`

### Infrastructure Metrics
- **CPU Usage**: `100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`
- **Memory Usage**: `(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100`
- **Disk Usage**: `(node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100`

## Alert Severity Levels

### Critical (P1)
- Service completely down
- Database unavailable
- Disk space < 10%
- Error rate > 50%

**Response Time**: Immediate (within 15 minutes)

### Warning (P2)
- High latency (P95 > 2s)
- High error rate (5-50%)
- High CPU/Memory usage (>80%)
- Causal computation accuracy < 70%

**Response Time**: Within 1 hour

### Info (P3)
- Bundle size increases
- Performance degradation
- Unusual traffic patterns

**Response Time**: Next business day

## Runbook Procedures

### 🚨 Service Down (Critical)

#### Symptoms
- Alert: `JAXBackendDown` or `FrontendDown`
- Status: `up{job="causal-ui-backend"} == 0`

#### Investigation Steps
1. **Check service status**:
   ```bash
   docker-compose ps
   kubectl get pods -n causal-ui-gym
   ```

2. **Check logs**:
   ```bash
   docker-compose logs backend
   kubectl logs -f deployment/causal-ui-backend
   ```

3. **Check resource usage**:
   ```bash
   docker stats
   kubectl top pods
   ```

#### Resolution Steps
1. **Restart service**:
   ```bash
   docker-compose restart backend
   kubectl rollout restart deployment/causal-ui-backend
   ```

2. **Scale up if needed**:
   ```bash
   docker-compose up --scale backend=3
   kubectl scale deployment causal-ui-backend --replicas=3
   ```

3. **Check dependencies**:
   ```bash
   docker-compose ps postgres redis
   kubectl get pods -l app=postgres
   ```

#### Escalation
If service doesn't recover within 15 minutes, escalate to senior engineer.

### ⚠️ High Latency (Warning)

#### Symptoms
- Alert: `CausalComputationLatencyHigh`
- Metric: P95 latency > 2 seconds

#### Investigation Steps
1. **Check current latency**:
   ```promql
   histogram_quantile(0.95, rate(causal_computation_duration_seconds_bucket[5m]))
   ```

2. **Identify slow requests**:
   ```promql
   topk(10, rate(http_request_duration_seconds_bucket{le="+Inf"}[5m]))
   ```

3. **Check JAX performance**:
   ```bash
   curl http://localhost:8000/metrics | grep jax_
   ```

#### Resolution Steps
1. **Check memory usage**:
   ```promql
   process_resident_memory_bytes{job="causal-ui-backend"}
   ```

2. **Scale backend if needed**:
   ```bash
   docker-compose up --scale backend=2
   ```

3. **Optimize computation**:
   - Review large DAG processing
   - Check JAX compilation cache
   - Monitor GPU/CPU utilization

### 📊 High Error Rate (Warning)

#### Symptoms
- Alert: `FrontendErrorRateHigh`
- Metric: Error rate > 5%

#### Investigation Steps
1. **Check error distribution**:
   ```promql
   sum by (status) (rate(http_requests_total{status=~"[45].."}[5m]))
   ```

2. **Review error logs**:
   ```bash
   docker-compose logs backend | grep ERROR
   ```

3. **Check specific endpoints**:
   ```promql
   rate(http_requests_total{status="500"}[5m]) by (path)
   ```

#### Resolution Steps
1. **Identify error patterns**:
   - 4xx errors: Client-side issues
   - 5xx errors: Server-side issues
   - Timeout errors: Performance issues

2. **Common fixes**:
   - Restart services for memory leaks
   - Increase timeout values
   - Validate input data quality

3. **Database issues**:
   ```bash
   docker-compose exec postgres psql -U user -d causal_ui_gym -c "SELECT * FROM pg_stat_activity;"
   ```

### 🧠 Causal Inference Issues

#### Symptoms
- Alert: `CausalInferenceAccuracyLow`
- Metric: Accuracy < 70%

#### Investigation Steps
1. **Check data quality**:
   ```promql
   missing_data_percentage
   data_validation_failures_total
   ```

2. **Review experiment parameters**:
   ```bash
   curl http://localhost:8000/api/experiments/stats
   ```

3. **Check LLM performance**:
   ```promql
   llm_response_accuracy_ratio by (model)
   ```

#### Resolution Steps
1. **Data validation**:
   - Check for missing or corrupted data
   - Validate causal model structure
   - Review intervention parameters

2. **Model optimization**:
   - Adjust JAX computation parameters
   - Review algorithm selection
   - Check for bias in training data

### 💾 Database Performance Issues

#### Symptoms
- Slow query performance
- Connection pool exhaustion
- Lock contention

#### Investigation Steps
1. **Check connection pool**:
   ```sql
   SELECT * FROM pg_stat_activity WHERE state = 'active';
   ```

2. **Review slow queries**:
   ```sql
   SELECT query, mean_exec_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_exec_time DESC LIMIT 10;
   ```

3. **Check locks**:
   ```sql
   SELECT * FROM pg_locks WHERE NOT granted;
   ```

#### Resolution Steps
1. **Optimize queries**:
   - Add missing indexes
   - Review query execution plans
   - Update table statistics

2. **Scale database**:
   - Increase connection pool size
   - Add read replicas
   - Optimize PostgreSQL configuration

### 🔧 Memory Leaks

#### Symptoms
- Alert: `MemoryLeakDetected`
- Continuously increasing memory usage

#### Investigation Steps
1. **Monitor memory growth**:
   ```promql
   increase(process_resident_memory_bytes[1h])
   ```

2. **Check garbage collection**:
   ```bash
   curl http://localhost:8000/debug/gc
   ```

3. **Profile memory usage**:
   ```bash
   docker-compose exec backend python -m memory_profiler app.py
   ```

#### Resolution Steps
1. **Immediate mitigation**:
   ```bash
   docker-compose restart backend
   ```

2. **Investigation**:
   - Review large object creation
   - Check for circular references
   - Monitor JAX memory allocation

3. **Long-term fixes**:
   - Implement memory monitoring
   - Add garbage collection tuning
   - Review caching strategies

## Performance Tuning

### JAX Optimization
```python
# Enable XLA compilation
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

# Memory management
import jax
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', True)
```

### Database Tuning
```sql
-- Optimize PostgreSQL for causal workloads
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
SELECT pg_reload_conf();
```

### Caching Strategies
```python
# Redis caching for causal computations
import redis
r = redis.Redis(host='redis', port=6379, db=0)

def cache_computation(dag_hash, result, ttl=3600):
    r.setex(f"causal:{dag_hash}", ttl, json.dumps(result))
```

## Monitoring Tools Access

### Grafana Dashboards
- **URL**: http://localhost:3000
- **Default Login**: admin/admin
- **Key Dashboards**:
  - System Overview
  - Application Performance
  - Causal Computation Metrics
  - Infrastructure Health

### Prometheus
- **URL**: http://localhost:9090
- **Query Examples**:
  ```promql
  # Current error rate
  rate(http_requests_total{status=~"5.."}[5m])
  
  # Memory usage trend
  process_resident_memory_bytes
  
  # Request rate by endpoint
  sum by (path) (rate(http_requests_total[5m]))
  ```

### Jaeger Tracing
- **URL**: http://localhost:16686
- **Usage**:
  - Search for traces by service
  - Analyze latency breakdown
  - Identify bottlenecks

### Loki Logs
- **Access via Grafana**: Explore → Loki
- **Query Examples**:
  ```logql
  # Error logs
  {job="causal-ui-backend"} |= "ERROR"
  
  # Slow requests
  {job="nginx-access"} | json | request_time > 2
  
  # Authentication failures
  {job="causal-ui-backend"} |= "authentication failed"
  ```

## Emergency Contacts

### On-Call Rotation
- **Primary**: Senior Backend Engineer
- **Secondary**: DevOps Engineer
- **Escalation**: Engineering Manager

### External Dependencies
- **Cloud Provider**: AWS/GCP Support
- **LLM APIs**: OpenAI/Anthropic Support
- **Database**: PostgreSQL Expert

## Maintenance Procedures

### Daily Checks
- [ ] Review alerts and resolve warnings
- [ ] Check system resource usage
- [ ] Verify backup completion
- [ ] Monitor causal computation accuracy

### Weekly Tasks
- [ ] Review performance trends
- [ ] Update monitoring thresholds
- [ ] Analyze slow query logs
- [ ] Clean up old metrics data

### Monthly Reviews
- [ ] Capacity planning assessment
- [ ] Alert fatigue analysis
- [ ] Monitoring infrastructure updates
- [ ] Runbook procedure updates

## References

- [Prometheus Alerting Rules](../monitoring/alerts.yml)
- [Grafana Dashboard Configs](../monitoring/grafana/dashboards/)
- [OpenTelemetry Configuration](../monitoring/otel-collector.yml)
- [Development Guide](../DEVELOPMENT.md)
- [Architecture Documentation](../ARCHITECTURE.md)