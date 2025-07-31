# Container Security Scanning and Hardening

This document outlines comprehensive container security practices for Causal UI Gym, including scanning, hardening, and runtime security.

## Overview

Container security encompasses:
- **Image Security**: Vulnerability scanning and minimal base images
- **Build Security**: Secure build processes and dependency management
- **Runtime Security**: Runtime protection and monitoring
- **Compliance**: Security standards and policy enforcement

## Multi-Stage Docker Security

### Hardened Production Dockerfile

```dockerfile
# Dockerfile.secure
# Multi-stage build with security best practices

# ================================
# Build Stage - Frontend
# ================================
FROM node:20-alpine3.18 AS frontend-builder

# Security: Create non-root user early
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Security: Install only necessary packages
RUN apk add --no-cache \
    dumb-init \
    curl \
    && rm -rf /var/cache/apk/*

# Security: Set working directory and permissions
WORKDIR /app
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Copy package files first (layer caching)
COPY --chown=appuser:appgroup package*.json ./

# Security: Audit dependencies before install
RUN npm audit --audit-level=moderate

# Install dependencies with exact versions
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY --chown=appuser:appgroup . .

# Build application
RUN npm run build

# ================================
# Build Stage - Backend
# ================================
FROM python:3.11-slim-bullseye AS backend-builder

# Security: System updates and minimal packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Security: Audit Python dependencies
RUN pip install --no-cache-dir pip-audit && \
    pip-audit --requirement requirements.txt --format=json --output=audit-report.json

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ================================
# Production Stage
# ================================
FROM python:3.11-slim-bullseye AS production

# Security: Install security updates and minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Security: Create non-root user and group
RUN groupadd -r -g 1001 appgroup && \
    useradd -r -u 1001 -g appgroup -d /app -s /sbin/nologin -c "Application User" appuser

# Security: Create app directory with proper permissions
RUN mkdir -p /app /app/static /app/logs && \
    chown -R appuser:appgroup /app

# Security: Copy Python packages from builder
COPY --from=backend-builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Security: Copy frontend build from builder
COPY --from=frontend-builder --chown=appuser:appgroup /app/dist /app/static

# Copy application code
COPY --chown=appuser:appgroup backend/ /app/backend/

# Security: Set PATH to include user packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Security: Switch to non-root user
USER appuser

WORKDIR /app

# Security: Use dumb-init as PID 1
ENTRYPOINT ["dumb-init", "--"]

# Security: Run application
CMD ["python", "-m", "backend.main"]

# Security: Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Security: Labels for metadata
LABEL \
    org.opencontainers.image.title="Causal UI Gym" \
    org.opencontainers.image.description="Secure React + JAX framework for causal reasoning" \
    org.opencontainers.image.vendor="Causal UI Gym Team" \
    org.opencontainers.image.version="0.1.0" \
    org.opencontainers.image.created="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    org.opencontainers.image.source="https://github.com/yourusername/causal-ui-gym" \
    org.opencontainers.image.licenses="MIT"

# Security: Expose only necessary port
EXPOSE 8000

# Security: Run as non-root
USER 1001:1001
```

### Distroless Alternative

```dockerfile
# Dockerfile.distroless
FROM node:20-alpine AS frontend-builder
# ... frontend build steps ...

FROM python:3.11-slim AS backend-builder
# ... backend build steps ...

# Use Google Distroless for minimal attack surface
FROM gcr.io/distroless/python3-debian11:latest

# Copy from builders
COPY --from=backend-builder /root/.local /root/.local
COPY --from=frontend-builder /app/dist /app/static
COPY backend/ /app/backend/

WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["python", "-m", "backend.main"]
```

## Security Scanning Tools

### Trivy Integration

```yaml
# .github/workflows/container-security.yml
name: Container Security Scanning

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * *'  # Daily scan

jobs:
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Build Docker image
        run: |
          docker build -f Dockerfile.secure -t causal-ui-gym:${{ github.sha }} .
          
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: causal-ui-gym:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH,MEDIUM'
          exit-code: '1'  # Fail on vulnerabilities
          
      - name: Run Trivy config scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'config'
          scan-ref: 'Dockerfile.secure'
          format: 'sarif'
          output: 'trivy-config.sarif'
          
      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
          
      - name: Generate Trivy HTML report
        if: always()
        run: |
          docker run --rm -v "$(pwd):/workspace" \
            aquasec/trivy:latest image \
            --format template --template "@contrib/html.tpl" \
            --output /workspace/trivy-report.html \
            causal-ui-gym:${{ github.sha }}
            
      - name: Upload scan results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-scan-results
          path: |
            trivy-results.sarif
            trivy-config.sarif
            trivy-report.html
```

### Snyk Container Scanning

```yaml
      - name: Snyk Container Scan
        uses: snyk/actions/docker@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          image: causal-ui-gym:${{ github.sha }}
          args: --severity-threshold=high --file=Dockerfile.secure
          
      - name: Upload Snyk results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: snyk.sarif
```

### Clair Scanning

```yaml
      - name: Clair Scanner
        run: |
          # Run Clair database
          docker run -d --name clair-db arminc/clair-db:latest
          docker run -p 6060:6060 --link clair-db:postgres -d --name clair arminc/clair-local-scan:latest
          
          # Wait for Clair to be ready
          sleep 30
          
          # Scan image
          docker run --rm --network container:clair \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -v $(pwd):/output \
            arminc/clair-scanner:latest \
            --clair="http://localhost:6060" \
            --ip="$(hostname -i)" \
            --report="/output/clair-report.json" \
            --log="/output/clair.log" \
            --whitelist="/output/.clair-whitelist.yml" \
            causal-ui-gym:${{ github.sha }}
```

## Runtime Security

### Docker Compose Security Configuration

```yaml
# docker-compose.secure.yml
version: '3.8'

services:
  causal-ui-gym:
    build:
      context: .
      dockerfile: Dockerfile.secure
    security_opt:
      - no-new-privileges:true  # Prevent privilege escalation
      - apparmor:docker-default # Use AppArmor profile
    cap_drop:
      - ALL  # Drop all capabilities
    cap_add:
      - NET_BIND_SERVICE  # Only add necessary capabilities
    read_only: true  # Read-only root filesystem
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /app/logs:noexec,nosuid,size=50m
    user: "1001:1001"  # Run as non-root user
    networks:
      - app-network
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=production
      - PYTHONPATH=/app
    volumes:
      - type: bind
        source: ./config
        target: /app/config
        read_only: true
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'

networks:
  app-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Kubernetes Security Context

```yaml
# k8s/deployment-secure.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: causal-ui-gym
  namespace: causal-ui-gym
spec:
  replicas: 3
  selector:
    matchLabels:
      app: causal-ui-gym
  template:
    metadata:
      labels:
        app: causal-ui-gym
      annotations:
        container.apparmor.security.beta.kubernetes.io/causal-ui-gym: runtime/default
    spec:
      serviceAccountName: causal-ui-gym-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: causal-ui-gym
        image: causal-ui-gym:latest
        imagePullPolicy: Always
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
        ports:
        - containerPort: 8000
          protocol: TCP
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /app/logs
        - name: config
          mountPath: /app/config
          readOnly: true
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
          requests:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: tmp
        emptyDir:
          sizeLimit: 100Mi
      - name: logs
        emptyDir:
          sizeLimit: 50Mi
      - name: config
        configMap:
          name: causal-ui-gym-config
      automountServiceAccountToken: false
```

## Security Policies

### Pod Security Policy

```yaml
# k8s/pod-security-policy.yml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: causal-ui-gym-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  allowedCapabilities:
    - NET_BIND_SERVICE
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  runAsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1001
        max: 1001
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### Network Policy

```yaml
# k8s/network-policy.yml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: causal-ui-gym-netpol
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
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []  # Allow all egress (restrict as needed)
    ports:
    - protocol: TCP
      port: 53  # DNS
    - protocol: UDP
      port: 53  # DNS
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 80   # HTTP
```

## Security Scanning Scripts

### Comprehensive Scanning Script

```bash
#!/bin/bash
# scripts/security-scan.sh

set -e

IMAGE_NAME="causal-ui-gym"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "🔒 Running comprehensive security scan for ${FULL_IMAGE}"

# Create output directory
mkdir -p security-reports

# Build image if not exists
if ! docker image inspect "${FULL_IMAGE}" >/dev/null 2>&1; then
    echo "📦 Building image ${FULL_IMAGE}..."
    docker build -f Dockerfile.secure -t "${FULL_IMAGE}" .
fi

echo "🔍 Running Trivy vulnerability scan..."
docker run --rm -v "$(pwd):/workspace" \
    aquasec/trivy:latest image \
    --format json \
    --output /workspace/security-reports/trivy-vuln.json \
    --severity HIGH,CRITICAL \
    "${FULL_IMAGE}"

echo "🔍 Running Trivy configuration scan..."
docker run --rm -v "$(pwd):/workspace" \
    aquasec/trivy:latest config \
    --format json \
    --output /workspace/security-reports/trivy-config.json \
    /workspace/Dockerfile.secure

echo "🔍 Running Docker Bench Security..."
docker run --rm --net host --pid host --userns host --cap-add audit_control \
    -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
    -v /etc:/etc:ro \
    -v /usr/bin/containerd:/usr/bin/containerd:ro \
    -v /usr/bin/runc:/usr/bin/runc:ro \
    -v /usr/lib/systemd:/usr/lib/systemd:ro \
    -v /var/lib:/var/lib:ro \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    --label docker_bench_security \
    docker/docker-bench-security > security-reports/docker-bench.log

echo "🔍 Running Hadolint on Dockerfile..."
docker run --rm -i hadolint/hadolint < Dockerfile.secure > security-reports/hadolint.log

echo "🔍 Running Dockle for image security..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    goodwithtech/dockle:latest \
    --format json \
    --output security-reports/dockle.json \
    "${FULL_IMAGE}"

echo "🔍 Scanning for secrets with TruffleHog..."
docker run --rm -v "$(pwd)":/workspace \
    trufflesecurity/trufflehog:latest \
    filesystem /workspace \
    --json > security-reports/trufflehog.json

echo "📊 Generating security report summary..."
cat > security-reports/summary.md << EOF
# Security Scan Summary

**Image**: ${FULL_IMAGE}
**Scan Date**: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Scan Results

- **Trivy Vulnerabilities**: security-reports/trivy-vuln.json
- **Trivy Configuration**: security-reports/trivy-config.json
- **Docker Bench Security**: security-reports/docker-bench.log
- **Hadolint (Dockerfile)**: security-reports/hadolint.log
- **Dockle (Image)**: security-reports/dockle.json
- **TruffleHog (Secrets)**: security-reports/trufflehog.json

## Next Steps

1. Review all high and critical vulnerabilities
2. Update base images and dependencies
3. Fix Dockerfile issues identified by Hadolint
4. Address any secrets found by TruffleHog
5. Implement recommendations from Docker Bench Security

EOF

echo "✅ Security scan complete! Results in security-reports/"
echo "📋 View summary: cat security-reports/summary.md"

# Check for critical issues
CRITICAL_VULNS=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | length' security-reports/trivy-vuln.json 2>/dev/null | wc -l)

if [ "$CRITICAL_VULNS" -gt 0 ]; then
    echo "❌ CRITICAL vulnerabilities found! Review security-reports/trivy-vuln.json"
    exit 1
else
    echo "✅ No critical vulnerabilities found"
fi
```

### Continuous Security Monitoring

```bash
#!/bin/bash
# scripts/security-monitor.sh

set -e

SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"
REGISTRY="your-registry.com"
IMAGE_NAME="causal-ui-gym"

# Monitor for new vulnerabilities
echo "🔍 Monitoring container security..."

# Pull latest vulnerability database
docker run --rm aquasec/trivy:latest image --download-db-only

# Scan all tagged images
for tag in $(docker images "${REGISTRY}/${IMAGE_NAME}" --format "{{.Tag}}"); do
    echo "Scanning ${REGISTRY}/${IMAGE_NAME}:${tag}..."
    
    # Run Trivy scan
    CRITICAL_COUNT=$(docker run --rm \
        aquasec/trivy:latest image \
        --format json \
        --severity CRITICAL \
        "${REGISTRY}/${IMAGE_NAME}:${tag}" | \
        jq '.Results[]?.Vulnerabilities | length // 0')
    
    if [ "$CRITICAL_COUNT" -gt 0 ]; then
        echo "⚠️  Found $CRITICAL_COUNT critical vulnerabilities in ${tag}"
        
        # Send Slack notification
        if [ -n "$SLACK_WEBHOOK" ]; then
            curl -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"🚨 Security Alert: $CRITICAL_COUNT critical vulnerabilities found in ${REGISTRY}/${IMAGE_NAME}:${tag}\"}" \
                "$SLACK_WEBHOOK"
        fi
    fi
done

echo "✅ Security monitoring complete"
```

## Security Best Practices Checklist

### Build Time
- [ ] Use minimal base images (Alpine, Distroless)
- [ ] Multi-stage builds to reduce final image size
- [ ] Scan dependencies before adding to image
- [ ] Use specific version tags, not `latest`
- [ ] Run as non-root user
- [ ] Use `.dockerignore` to exclude sensitive files
- [ ] Sign images with Docker Content Trust

### Runtime
- [ ] Read-only root filesystem
- [ ] Drop all capabilities, add only necessary ones
- [ ] Use security contexts in Kubernetes
- [ ] Network policies to restrict traffic
- [ ] Resource limits to prevent DoS
- [ ] Regular security updates
- [ ] Monitor runtime behavior

### Operations
- [ ] Automated vulnerability scanning
- [ ] Image signing and verification
- [ ] SBOM generation and tracking
- [ ] Incident response procedures
- [ ] Security monitoring and alerting
- [ ] Regular security audits
- [ ] Staff security training

## References

- [NIST Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Container Security](https://owasp.org/www-project-container-security/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Docker Security Documentation](https://docs.docker.com/engine/security/)

---

*This security configuration provides comprehensive protection for containerized applications throughout the development and deployment lifecycle.*