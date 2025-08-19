# Production Deployment Guide - Causal UI Gym

## Overview

This guide covers the complete production deployment process for Causal UI Gym, including infrastructure setup, security configurations, monitoring, and scaling strategies.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │     Frontend    │    │     Backend     │
│   (Ingress)     │───▶│   (React/TS)    │───▶│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │   PostgreSQL    │
                       │    (Cache)      │    │   (Metadata)    │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster** (v1.21+)
2. **Docker** (v20.10+)
3. **kubectl** configured with cluster access
4. **Helm** (v3.0+) for package management
5. **Git** for version control

### One-Command Deployment

```bash
# Clone and deploy
git clone <repository-url>
cd causal-ui-gym
./scripts/deploy-production.sh
```

## 🔧 Detailed Setup

### 1. Infrastructure Prerequisites

#### Kubernetes Cluster Setup

```bash
# For local development with kind
kind create cluster --config deployment/kind-config.yml

# For cloud providers
# AWS: eksctl create cluster --name causal-ui-gym --region us-west-2
# GCP: gcloud container clusters create causal-ui-gym --zone us-central1-a
# Azure: az aks create --resource-group myResourceGroup --name causal-ui-gym
```

#### Required Kubernetes Add-ons

```bash
# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager for TLS
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Install Prometheus monitoring
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

### 2. Environment Configuration

#### Create Environment Secrets

```bash
# Create namespace
kubectl create namespace causal-ui-gym

# Create secrets
kubectl create secret generic causal-ui-gym-secrets \
  --from-literal=DATABASE_URL="postgresql://user:password@postgres:5432/causalui" \
  --from-literal=REDIS_PASSWORD="secure-redis-password" \
  --from-literal=JWT_SECRET="your-jwt-secret-key" \
  --from-literal=SENTRY_DSN="https://your-sentry-dsn.ingest.sentry.io" \
  -n causal-ui-gym
```

#### Configuration Files

**deployment/production-config.yml**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: causal-ui-gym-config
  namespace: causal-ui-gym
data:
  # Application settings
  NODE_ENV: "production"
  LOG_LEVEL: "info"
  
  # Performance settings
  CACHE_TTL: "3600"
  MAX_CONCURRENT_REQUESTS: "100"
  
  # Feature flags
  ENABLE_RESEARCH_MODE: "true"
  ENABLE_ADVANCED_ANALYTICS: "true"
  ENABLE_AUTO_SCALING: "true"
```

### 3. Security Configuration

#### Network Policies

```yaml
# Restrict pod-to-pod communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: causal-ui-gym-network-policy
  namespace: causal-ui-gym
spec:
  podSelector:
    matchLabels:
      app: causal-ui-gym
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis-cache
  - to:
    - podSelector:
        matchLabels:
          app: postgres-db
```

#### TLS Configuration

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: causal-ui-gym-tls
  namespace: causal-ui-gym
spec:
  secretName: causal-ui-gym-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - causal-ui-gym.yourcompany.com
```

### 4. Deployment Process

#### Standard Deployment

```bash
# 1. Run quality gates
python3 scripts/quality-gates.py

# 2. Build container images
docker build -t causal-ui-gym/frontend:latest .
docker build -t causal-ui-gym/backend:latest backend/

# 3. Deploy to Kubernetes
kubectl apply -f deployment/production-orchestrator.yml

# 4. Wait for rollout
kubectl rollout status deployment/causal-ui-gym-frontend -n causal-ui-gym
kubectl rollout status deployment/causal-ui-gym-backend -n causal-ui-gym
```

#### Zero-Downtime Deployment

```bash
# Using the automated script
./scripts/deploy-production.sh \
  --environment production \
  --namespace causal-ui-gym \
  --config deployment/production-orchestrator.yml
```

### 5. Monitoring and Observability

#### Prometheus Metrics

The application exposes metrics at `/metrics` endpoint:

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `causal_computation_duration_seconds` - Time spent on causal computations
- `cache_hit_rate` - Cache hit rate percentage
- `memory_usage_bytes` - Memory consumption
- `concurrent_experiments` - Number of running experiments

#### Grafana Dashboards

```bash
# Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring

# Import dashboard from deployment/grafana-dashboard.json
```

#### Log Aggregation

```bash
# Install Fluentd for log collection
helm repo add fluent https://fluent.github.io/helm-charts
helm install fluentd fluent/fluentd -n logging --create-namespace
```

### 6. Auto-Scaling Configuration

#### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: causal-ui-gym-frontend-hpa
  namespace: causal-ui-gym
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: causal-ui-gym-frontend
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Vertical Pod Autoscaler

```bash
# Install VPA
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/download/vertical-pod-autoscaler-0.13.0/vpa-release-0.13.0.yaml
```

### 7. Backup and Disaster Recovery

#### Database Backup

```bash
# Automated backup script
#!/bin/bash
kubectl exec postgres-db-0 -n causal-ui-gym -- pg_dump -U user causalui > backup-$(date +%Y%m%d).sql
aws s3 cp backup-$(date +%Y%m%d).sql s3://your-backup-bucket/
```

#### Configuration Backup

```bash
# Backup all Kubernetes resources
kubectl get all,configmap,secret,pvc -n causal-ui-gym -o yaml > causal-ui-gym-backup.yml
```

### 8. Performance Optimization

#### Resource Limits

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

#### Cache Optimization

```yaml
# Redis configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
data:
  redis.conf: |
    maxmemory 1gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
```

### 9. Health Checks and Readiness

#### Liveness Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 3000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

#### Readiness Probes

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 3000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

### 10. Troubleshooting

#### Common Issues

1. **Pod Startup Failures**
   ```bash
   kubectl describe pod <pod-name> -n causal-ui-gym
   kubectl logs <pod-name> -n causal-ui-gym
   ```

2. **Service Discovery Issues**
   ```bash
   kubectl get endpoints -n causal-ui-gym
   kubectl get services -n causal-ui-gym
   ```

3. **Performance Issues**
   ```bash
   kubectl top pods -n causal-ui-gym
   kubectl top nodes
   ```

#### Debug Commands

```bash
# Check cluster status
kubectl cluster-info
kubectl get nodes

# Check application status
kubectl get pods -n causal-ui-gym -o wide
kubectl get services -n causal-ui-gym
kubectl get ingress -n causal-ui-gym

# View logs
kubectl logs -f deployment/causal-ui-gym-frontend -n causal-ui-gym
kubectl logs -f deployment/causal-ui-gym-backend -n causal-ui-gym

# Execute commands in pods
kubectl exec -it deployment/causal-ui-gym-frontend -n causal-ui-gym -- /bin/bash
```

## 🔒 Security Best Practices

### 1. Container Security

- Use non-root users in containers
- Scan images for vulnerabilities
- Use minimal base images (Alpine Linux)
- Keep containers stateless

### 2. Network Security

- Implement network policies
- Use TLS for all communications
- Restrict egress traffic
- Use service mesh for advanced security

### 3. Secrets Management

- Use Kubernetes secrets
- Rotate secrets regularly
- Never commit secrets to version control
- Use external secret management (HashiCorp Vault, AWS Secrets Manager)

### 4. Access Control

- Implement RBAC
- Use service accounts
- Limit pod privileges
- Regular security audits

## 📊 Performance Benchmarks

### Expected Performance Metrics

- **Response Time**: < 200ms (95th percentile)
- **Throughput**: > 1000 requests/second
- **Memory Usage**: < 2GB per pod
- **CPU Usage**: < 70% average
- **Cache Hit Rate**: > 85%

### Load Testing

```bash
# Apache Bench
ab -n 10000 -c 100 http://causal-ui-gym.yourcompany.com/

# Artillery.js
npm install -g artillery
artillery run deployment/load-test.yml
```

## 🚨 Alerting and Monitoring

### Critical Alerts

1. **Pod Down**: Any pod restart or failure
2. **High Memory**: Memory usage > 90%
3. **High CPU**: CPU usage > 85%
4. **High Error Rate**: Error rate > 5%
5. **Certificate Expiry**: TLS certificate expires in < 30 days

### Monitoring Stack

- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **AlertManager**: Alert routing
- **Jaeger**: Distributed tracing
- **Fluentd**: Log aggregation

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run Quality Gates
      run: python3 scripts/quality-gates.py
    - name: Deploy to Production
      run: ./scripts/deploy-production.sh
      env:
        KUBECONFIG: ${{ secrets.KUBECONFIG }}
```

### GitLab CI

```yaml
stages:
  - test
  - build
  - deploy

deploy_production:
  stage: deploy
  script:
    - python3 scripts/quality-gates.py
    - ./scripts/deploy-production.sh
  only:
    - main
```

## 📋 Maintenance Procedures

### Regular Maintenance

1. **Weekly**: Update dependencies, security patches
2. **Monthly**: Performance review, capacity planning
3. **Quarterly**: Security audit, disaster recovery testing
4. **Annually**: Architecture review, technology updates

### Update Procedures

```bash
# Rolling update
kubectl set image deployment/causal-ui-gym-frontend \
  frontend=causal-ui-gym/frontend:v1.2.0 \
  -n causal-ui-gym

# Rollback if needed
kubectl rollout undo deployment/causal-ui-gym-frontend -n causal-ui-gym
```

## 🆘 Emergency Procedures

### Incident Response

1. **Identify**: Monitor alerts and metrics
2. **Assess**: Determine impact and severity
3. **Respond**: Implement immediate fixes
4. **Communicate**: Update stakeholders
5. **Resolve**: Apply permanent fixes
6. **Review**: Post-incident analysis

### Emergency Contacts

- **On-call Engineer**: [Phone/Slack]
- **DevOps Lead**: [Phone/Slack]
- **Product Owner**: [Phone/Slack]

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)
- [Security Best Practices](https://kubernetes.io/docs/concepts/security/)

---

For questions or support, please contact the DevOps team or create an issue in the repository.