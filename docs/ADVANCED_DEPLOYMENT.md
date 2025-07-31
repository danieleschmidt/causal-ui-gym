# Advanced Deployment and Scaling

This document outlines comprehensive deployment strategies, scaling approaches, and infrastructure management for Causal UI Gym in production environments.

## Deployment Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Global Load Balancer                         │
│                      (CloudFlare/AWS)                           │
├─────────────────────────────────────────────────────────────────┤
│  Region 1 (us-west-2)     │     Region 2 (eu-west-1)          │
│  ┌───────────────────────┐ │ ┌───────────────────────────────┐ │
│  │   Frontend (CDN)      │ │ │   Frontend (CDN)              │ │
│  │   - Static Assets     │ │ │   - Static Assets             │ │
│  │   - Edge Caching      │ │ │   - Edge Caching              │ │
│  └───────────────────────┘ │ └───────────────────────────────┘ │
│  ┌───────────────────────┐ │ ┌───────────────────────────────┐ │
│  │   Kubernetes Cluster  │ │ │   Kubernetes Cluster          │ │
│  │   - Backend Services  │ │ │   - Backend Services          │ │
│  │   - JAX Computation   │ │ │   - JAX Computation           │ │
│  │   - LLM Integration   │ │ │   - LLM Integration           │ │
│  └───────────────────────┘ │ └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Kubernetes Deployment Manifests

### Production Namespace and RBAC

```yaml
# k8s/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: causal-ui-gym
  labels:
    name: causal-ui-gym
    tier: production
    monitoring: enabled
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: causal-ui-gym-sa
  namespace: causal-ui-gym
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/causal-ui-gym-role
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: causal-ui-gym-role
  namespace: causal-ui-gym
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: causal-ui-gym-binding
  namespace: causal-ui-gym
subjects:
- kind: ServiceAccount
  name: causal-ui-gym-sa
  namespace: causal-ui-gym
roleRef:
  kind: Role
  name: causal-ui-gym-role
  apiGroup: rbac.authorization.k8s.io
```

### ConfigMap and Secrets

```yaml
# k8s/configmap.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: causal-ui-gym-config
  namespace: causal-ui-gym
data:
  app.env: "production"
  log.level: "info"
  server.port: "8000"
  server.workers: "4"
  jax.enable_x64: "true"
  jax.platform_name: "gpu"
  cors.origins: "https://causal-ui-gym.dev,https://app.causal-ui-gym.dev"
  monitoring.enabled: "true"
  monitoring.endpoint: "http://otel-collector:4318"
---
apiVersion: v1
kind: Secret
metadata:
  name: causal-ui-gym-secrets
  namespace: causal-ui-gym
type: Opaque
data:
  # Base64 encoded secrets - managed by external-secrets-operator
  openai-api-key: ""
  anthropic-api-key: ""
  database-url: ""
  redis-url: ""
  jwt-secret: ""
```

### Deployment with Advanced Configuration

```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: causal-ui-gym
  namespace: causal-ui-gym
  labels:
    app: causal-ui-gym
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 50%
      maxUnavailable: 25%
  selector:
    matchLabels:
      app: causal-ui-gym
  template:
    metadata:
      labels:
        app: causal-ui-gym
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
        fluentd.io/parser: "json"
    spec:
      serviceAccountName: causal-ui-gym-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
        seccompProfile:
          type: RuntimeDefault
      nodeSelector:
        instance-type: "gpu-optimized"
        availability-zone: "us-west-2a"
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values: ["causal-ui-gym"]
              topologyKey: kubernetes.io/hostname
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/arch
                operator: In
                values: ["amd64"]
      containers:
      - name: causal-ui-gym
        image: ghcr.io/yourusername/causal-ui-gym:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 8001
          name: metrics
          protocol: TCP
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        envFrom:
        - configMapRef:
            name: causal-ui-gym-config
        - secretRef:
            name: causal-ui-gym-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
            nvidia.com/gpu: "1"
          limits:
            memory: "2Gi"
            cpu: "2000m"
            nvidia.com/gpu: "1"
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1001
          runAsGroup: 1001
          capabilities:
            drop:
            - ALL
            add:
            - NET_BIND_SERVICE
          seccompProfile:
            type: RuntimeDefault
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: app-logs
          mountPath: /app/logs
        - name: model-cache
          mountPath: /app/model-cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
          successThreshold: 1
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          successThreshold: 1
        startupProbe:
          httpGet:
            path: /health
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
          successThreshold: 1
      volumes:
      - name: tmp
        emptyDir:
          sizeLimit: 500Mi
      - name: app-logs
        emptyDir:
          sizeLimit: 200Mi
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      terminationGracePeriodSeconds: 60
      dnsPolicy: ClusterFirst
      restartPolicy: Always
```

### Service and Ingress

```yaml
# k8s/service.yml
apiVersion: v1
kind: Service
metadata:
  name: causal-ui-gym-service
  namespace: causal-ui-gym
  labels:
    app: causal-ui-gym
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "http"
spec:
  type: LoadBalancer
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 8001
    targetPort: 8001
    protocol: TCP
  selector:
    app: causal-ui-gym
  sessionAffinity: None
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: causal-ui-gym-ingress
  namespace: causal-ui-gym
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.causal-ui-gym.dev
    secretName: causal-ui-gym-tls
  rules:
  - host: api.causal-ui-gym.dev
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: causal-ui-gym-service
            port:
              number: 80
```

## Horizontal Pod Autoscaler (HPA)

```yaml
# k8s/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: causal-ui-gym-hpa
  namespace: causal-ui-gym
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: causal-ui-gym
  minReplicas: 3
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
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  - type: External
    external:
      metric:
        name: sqs_queue_length
        selector:
          matchLabels:
            queue_name: causal-computations
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max
```

## Vertical Pod Autoscaler (VPA)

```yaml
# k8s/vpa.yml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: causal-ui-gym-vpa
  namespace: causal-ui-gym
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: causal-ui-gym
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: causal-ui-gym
      minAllowed:
        cpu: 100m
        memory: 512Mi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
```

## Multi-Region Deployment Strategy

### GitOps with ArgoCD

```yaml
# argocd/applications/causal-ui-gym-prod.yml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: causal-ui-gym-prod
  namespace: argocd
  labels:
    environment: production
spec:
  project: causal-ui-gym
  source:
    repoURL: https://github.com/yourusername/causal-ui-gym-k8s
    path: environments/production
    targetRevision: main
    helm:
      valueFiles:
      - values-production.yaml
      parameters:
      - name: image.tag
        value: v1.0.0
      - name: replicaCount
        value: "5"
      - name: autoscaling.enabled
        value: "true"
  destination:
    server: https://prod-cluster.us-west-2.causal-ui-gym.dev
    namespace: causal-ui-gym
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  revisionHistoryLimit: 10
```

### Istio Service Mesh Configuration

```yaml
# istio/virtual-service.yml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: causal-ui-gym-vs
  namespace: causal-ui-gym
spec:
  hosts:
  - api.causal-ui-gym.dev
  gateways:
  - causal-ui-gym-gateway
  http:
  - match:
    - uri:
        prefix: /api/v1/experiments
    route:
    - destination:
        host: causal-ui-gym-service
        port:
          number: 80
        subset: stable
      weight: 90
    - destination:
        host: causal-ui-gym-service
        port:
          number: 80
        subset: canary
      weight: 10
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s
    retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,reset,connect-failure,refused-stream
    timeout: 30s
  - match:
    - uri:
        prefix: /
    route:
    - destination:
        host: causal-ui-gym-service
        port:
          number: 80
        subset: stable
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: causal-ui-gym-dr
  namespace: causal-ui-gym
spec:
  host: causal-ui-gym-service
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
        maxRetries: 3
        consecutiveGatewayErrors: 5
        interval: 30s
        baseEjectionTime: 30s
        maxEjectionPercent: 50
    outlierDetection:
      consecutiveGatewayErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 50
  subsets:
  - name: stable
    labels:
      version: stable
  - name: canary
    labels:
      version: canary
```

## Database Scaling and Management

### PostgreSQL with PGBouncer

```yaml
# k8s/postgres-cluster.yml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: causal-ui-gym
spec:
  instances: 3
  imageName: postgres:15
  
  postgresql:
    parameters:
      max_connections: "500"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      maintenance_work_mem: "64MB"
      checkpoint_completion_target: "0.7"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
      work_mem: "4MB"
      min_wal_size: "1GB"
      max_wal_size: "4GB"

  bootstrap:
    initdb:
      database: causal_ui_gym
      owner: app_user
      secret:
        name: postgres-credentials

  storage:
    size: 100Gi
    storageClass: fast-ssd

  monitoring:
    enabled: true

  backup:
    target: prefer-standby
    retentionPolicy: "30d"
    data:
      compression: gzip
    wal:
      retention: "7d"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: causal-ui-gym
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: pgbouncer/pgbouncer:latest
        ports:
        - containerPort: 5432
        env:
        - name: DATABASES_HOST
          value: postgres-cluster-rw
        - name: DATABASES_PORT
          value: "5432"
        - name: DATABASES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: username
        - name: DATABASES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password
        - name: DATABASES_DBNAME
          value: causal_ui_gym
        - name: POOL_MODE
          value: transaction
        - name: MAX_CLIENT_CONN
          value: "1000"
        - name: DEFAULT_POOL_SIZE
          value: "50"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
```

### Redis Cluster for Caching

```yaml
# k8s/redis-cluster.yml
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: redis-cluster
  namespace: causal-ui-gym
spec:
  clusterSize: 6
  kubernetesConfig:
    image: redis:7-alpine
    imagePullPolicy: IfNotPresent
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 500m
        memory: 512Mi
    serviceType: ClusterIP
  redisExporter:
    enabled: true
    image: oliver006/redis_exporter:latest
  redisConfig:
    additionalRedisConfig: |
      maxmemory 256mb
      maxmemory-policy allkeys-lru
      save 900 1
      save 300 10
      save 60 10000
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
        storageClassName: fast-ssd
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
```

## Advanced Monitoring and Alerting

### Prometheus ServiceMonitor

```yaml
# k8s/service-monitor.yml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: causal-ui-gym-metrics
  namespace: causal-ui-gym
  labels:
    app: causal-ui-gym
spec:
  selector:
    matchLabels:
      app: causal-ui-gym
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    honorLabels: true
    scrapeTimeout: 10s
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: causal-ui-gym-alerts
  namespace: causal-ui-gym
spec:
  groups:
  - name: causal-ui-gym.rules
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: warning
        service: causal-ui-gym
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} req/sec for {{ $labels.instance }}"

    - alert: HighLatency
      expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
      for: 5m
      labels:
        severity: warning
        service: causal-ui-gym
      annotations:
        summary: "High latency detected"
        description: "95th percentile latency is {{ $value }}s for {{ $labels.instance }}"

    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
      for: 5m
      labels:
        severity: critical
        service: causal-ui-gym
      annotations:
        summary: "Pod is crash looping"
        description: "Pod {{ $labels.pod }} is restarting frequently"

    - alert: GPUUtilizationHigh
      expr: DCGM_FI_DEV_GPU_UTIL > 90
      for: 10m
      labels:
        severity: warning
        service: causal-ui-gym
      annotations:
        summary: "High GPU utilization"
        description: "GPU utilization is {{ $value }}% for {{ $labels.gpu }}"
```

## Disaster Recovery and Backup

### Velero Backup Configuration

```yaml
# velero/backup-schedule.yml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: causal-ui-gym-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  template:
    includedNamespaces:
    - causal-ui-gym
    excludedResources:
    - pods
    - replicasets
    - events
    - events.events.k8s.io
    storageLocation: default
    volumeSnapshotLocations:
    - default
    ttl: 720h  # 30 days
    hooks:
      resources:
      - name: postgres-backup
        includedNamespaces:
        - causal-ui-gym
        labelSelector:
          matchLabels:
            app: postgres-cluster
        pre:
        - exec:
            container: postgres
            command: ["pg_dump", "-h", "localhost", "-U", "app_user", "-d", "causal_ui_gym", "-f", "/tmp/backup.sql"]
        post:
        - exec:
            container: postgres
            command: ["rm", "/tmp/backup.sql"]
```

### Multi-Region Replication

```yaml
# k8s/cross-region-service.yml
apiVersion: v1
kind: Service
metadata:
  name: causal-ui-gym-global
  namespace: causal-ui-gym
  annotations:
    external-dns.alpha.kubernetes.io/hostname: api.causal-ui-gym.dev
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 8000
    protocol: TCP
    name: https
  selector:
    app: causal-ui-gym
  externalTrafficPolicy: Local
---
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: causal-ui-gym-global-gateway
  namespace: causal-ui-gym
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: causal-ui-gym-tls
    hosts:
    - api.causal-ui-gym.dev
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - api.causal-ui-gym.dev
    redirect:
      httpsRedirect: true
```

## Cost Optimization

### Cluster Autoscaler Configuration

```yaml
# k8s/cluster-autoscaler.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/causal-ui-gym
        - --balance-similar-node-groups
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --scale-down-utilization-threshold=0.5
        - --max-node-provision-time=15m
        - --max-empty-bulk-delete=10
        - --max-graceful-termination-sec=600
```

### Spot Instance Configuration

```yaml
# k8s/spot-node-group.yml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: causal-ui-gym-cluster
  region: us-west-2
nodeGroups:
- name: spot-workers
  instanceTypes: 
  - g4dn.xlarge
  - g4dn.2xlarge
  - p3.2xlarge
  spot: true
  minSize: 0
  maxSize: 20
  desiredCapacity: 3
  volumeSize: 100
  volumeType: gp3
  labels:
    node-type: spot
    workload: gpu-compute
  taints:
  - key: spot-instance
    value: "true"
    effect: NoSchedule
  tags:
    Environment: production
    Application: causal-ui-gym
    NodeGroup: spot-workers
  iam:
    withAddonPolicies:
      autoScaler: true
      ebs: true
      fsx: true
      efs: true
```

## Performance Optimization

### KEDA for Event-Driven Scaling

```yaml
# k8s/keda-scaler.yml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: causal-ui-gym-scaler
  namespace: causal-ui-gym
spec:
  scaleTargetRef:
    name: causal-ui-gym
  pollingInterval: 15
  cooldownPeriod: 300
  idleReplicaCount: 2
  minReplicaCount: 2
  maxReplicaCount: 50
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: http_requests_per_second
      threshold: '100'
      query: sum(rate(http_requests_total[1m]))
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.us-west-2.amazonaws.com/123456789/causal-computation-queue
      queueLength: '10'
      awsRegion: us-west-2
  - type: cpu
    metadata:
      type: Utilization
      value: '70'
  - type: memory
    metadata:
      type: Utilization
      value: '80'
```

## Deployment Scripts

### Blue-Green Deployment Script

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

set -e

NAMESPACE="causal-ui-gym"
NEW_VERSION="$1"
CURRENT_COLOR="$2"

if [ -z "$NEW_VERSION" ] || [ -z "$CURRENT_COLOR" ]; then
    echo "Usage: $0 <new-version> <current-color>"
    echo "Example: $0 v1.2.0 blue"
    exit 1
fi

# Determine target color
if [ "$CURRENT_COLOR" == "blue" ]; then
    TARGET_COLOR="green"
else
    TARGET_COLOR="blue"
fi

echo "🚀 Starting blue-green deployment..."
echo "Current: $CURRENT_COLOR -> Target: $TARGET_COLOR"
echo "Version: $NEW_VERSION"

# Deploy to target environment
echo "📦 Deploying to $TARGET_COLOR environment..."
kubectl set image deployment/causal-ui-gym-$TARGET_COLOR \
    causal-ui-gym=ghcr.io/yourusername/causal-ui-gym:$NEW_VERSION \
    -n $NAMESPACE

# Wait for rollout
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/causal-ui-gym-$TARGET_COLOR -n $NAMESPACE --timeout=600s

# Health check
echo "🏥 Running health checks..."
HEALTH_URL="http://causal-ui-gym-$TARGET_COLOR-service.$NAMESPACE.svc.cluster.local/health"

for i in {1..30}; do
    if kubectl run health-check-$TARGET_COLOR --image=curlimages/curl --rm -i --restart=Never -- \
        curl -f $HEALTH_URL; then
        echo "✅ Health check passed"
        break
    fi
    echo "❌ Health check failed, attempt $i/30"
    sleep 10
done

# Run smoke tests
echo "🧪 Running smoke tests..."
kubectl run smoke-test-$TARGET_COLOR --image=curlimages/curl --rm -i --restart=Never -- \
    curl -f $HEALTH_URL/ready

# Switch traffic
echo "🔄 Switching traffic to $TARGET_COLOR..."
kubectl patch service causal-ui-gym-service -n $NAMESPACE -p \
    "{\"spec\":{\"selector\":{\"color\":\"$TARGET_COLOR\"}}}"

# Verify traffic switch
echo "✅ Verifying traffic switch..."
sleep 30
kubectl run traffic-check --image=curlimages/curl --rm -i --restart=Never -- \
    curl -f http://causal-ui-gym-service.$NAMESPACE.svc.cluster.local/health

echo "🎉 Blue-green deployment completed successfully!"
echo "New active environment: $TARGET_COLOR"
echo "Previous environment: $CURRENT_COLOR (kept for rollback)"
```

---

*This comprehensive deployment configuration provides production-ready, scalable, and resilient infrastructure for the Causal UI Gym application across multiple environments and regions.*