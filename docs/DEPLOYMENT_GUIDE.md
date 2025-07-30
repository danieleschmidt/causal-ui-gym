# Deployment Guide

## Overview

This guide covers deployment strategies for the Causal UI Gym application across different environments, from local development to production clusters.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Load Balancer │───▶│    Nginx     │───▶│  React Frontend │
│     (Public)    │    │  (Reverse    │    │   (Static)      │
└─────────────────┘    │   Proxy)     │    └─────────────────┘
                       └──────────────┘             │
                              │                     │
                              ▼                     │
                       ┌──────────────┐             │
                       │  FastAPI     │◀────────────┘
                       │  Backend     │
                       │  (Python)    │
                       └──────────────┘
                              │
                              ▼
                  ┌─────────────────────────────┐
                  │     Data Layer              │
                  │  ┌─────────┐ ┌────────────┐ │
                  │  │PostgreSQL│ │   Redis    │ │
                  │  │ (Primary) │ │  (Cache)   │ │
                  │  └─────────┘ └────────────┘ │
                  └─────────────────────────────┘
```

## Environment Configurations

### Local Development
- **Frontend**: Vite dev server (port 5173)
- **Backend**: FastAPI with hot reload (port 8000)
- **Database**: Docker PostgreSQL (port 5432)
- **Cache**: Docker Redis (port 6379)

### Staging
- **Environment**: Docker Compose with production builds
- **Database**: Managed PostgreSQL instance
- **Cache**: Managed Redis instance
- **Domain**: staging.causal-ui-gym.com
- **TLS**: Let's Encrypt certificates

### Production
- **Environment**: Kubernetes cluster or Docker Swarm
- **Database**: High-availability PostgreSQL cluster
- **Cache**: Redis cluster with persistence
- **CDN**: CloudFront for static assets
- **Domain**: causal-ui-gym.com
- **TLS**: SSL/TLS certificates

## Docker Deployment

### Build Images

```bash
# Build production image
docker build -t causal-ui-gym:latest .

# Build with specific tag
docker build -t causal-ui-gym:v1.0.0 .

# Multi-architecture build
docker buildx build --platform linux/amd64,linux/arm64 \
  -t causal-ui-gym:latest --push .
```

### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    image: causal-ui-gym:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - JWT_SECRET=${JWT_SECRET}
    volumes:
      - app_data:/app/data
      - app_logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - static_files:/usr/share/nginx/html
    depends_on:
      - app
    restart: unless-stopped

volumes:
  app_data:
  app_logs:
  static_files:
```

### Run Production Stack

```bash
# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale application instances
docker-compose -f docker-compose.prod.yml up -d --scale app=3

# Monitor deployment
docker-compose logs -f app
```

## Kubernetes Deployment

### Namespace and Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: causal-ui-gym
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: causal-ui-gym
data:
  PYTHON_ENV: "production"
  JAX_PLATFORM_NAME: "cpu"
  LOG_LEVEL: "INFO"
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: causal-ui-gym
type: Opaque
data:
  # Base64 encoded values
  DATABASE_URL: <base64-encoded-url>
  OPENAI_API_KEY: <base64-encoded-key>
  ANTHROPIC_API_KEY: <base64-encoded-key>
  JWT_SECRET: <base64-encoded-secret>
```

### Application Deployment

```yaml
# k8s/deployment.yaml
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
    spec:
      containers:
      - name: app
        image: causal-ui-gym:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: app-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-storage
          mountPath: /app/data
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: app-data-pvc
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: causal-ui-gym-service
  namespace: causal-ui-gym
spec:
  selector:
    app: causal-ui-gym
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: causal-ui-gym-ingress
  namespace: causal-ui-gym
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - causal-ui-gym.com
    secretName: causal-ui-gym-tls
  rules:
  - host: causal-ui-gym.com
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

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get pods -n causal-ui-gym
kubectl describe deployment causal-ui-gym -n causal-ui-gym

# View logs
kubectl logs -f deployment/causal-ui-gym -n causal-ui-gym

# Scale deployment
kubectl scale deployment causal-ui-gym --replicas=5 -n causal-ui-gym
```

## Cloud Provider Deployments

### AWS ECS Deployment

```json
{
  "family": "causal-ui-gym",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "causal-ui-gym",
      "image": "your-account.dkr.ecr.region.amazonaws.com/causal-ui-gym:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PYTHON_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:prod/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/causal-ui-gym",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### Google Cloud Run Deployment

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/causal-ui-gym:$COMMIT_SHA', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/causal-ui-gym:$COMMIT_SHA']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'causal-ui-gym'
      - '--image'
      - 'gcr.io/$PROJECT_ID/causal-ui-gym:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--set-env-vars'
      - 'PYTHON_ENV=production'
      - '--memory'
      - '1Gi'
      - '--cpu'
      - '1'
      - '--max-instances'
      - '10'
```

### Azure Container Instances

```bash
# Create resource group
az group create --name causal-ui-gym-rg --location eastus

# Deploy container
az container create \
  --resource-group causal-ui-gym-rg \
  --name causal-ui-gym \
  --image causal-ui-gym:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8000 \
  --dns-name-label causal-ui-gym \
  --environment-variables \
    PYTHON_ENV=production \
  --secure-environment-variables \
    DATABASE_URL=$DATABASE_URL \
    OPENAI_API_KEY=$OPENAI_API_KEY
```

## Database Setup

### PostgreSQL Configuration

```sql
-- Create production database
CREATE DATABASE causal_ui_gym_prod;
CREATE USER causal_app WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE causal_ui_gym_prod TO causal_app;

-- Create indexes for performance
CREATE INDEX idx_experiments_user_id ON experiments(user_id);
CREATE INDEX idx_causal_graphs_created_at ON causal_graphs(created_at);
CREATE INDEX idx_interventions_experiment_id ON interventions(experiment_id);

-- Set up connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

### Redis Configuration

```bash
# Redis configuration for production
# /etc/redis/redis.conf

# Security
requirepass your_secure_password
bind 127.0.0.1 ::1

# Memory management
maxmemory 512mb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
```

## Environment Variables

### Required Environment Variables

```bash
# Application
PYTHON_ENV=production
NODE_ENV=production
APP_VERSION=1.0.0
DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=ant-...

# Security
JWT_SECRET=your-secret-key
CORS_ORIGINS=https://causal-ui-gym.com

# Monitoring
PROMETHEUS_ENDPOINT=http://prometheus:9090
JAEGER_ENDPOINT=http://jaeger:14268

# Performance
JAX_PLATFORM_NAME=cpu
MAX_WORKERS=4
```

### Environment-Specific Variables

```bash
# Development
DATABASE_URL=postgresql://user:password@localhost:5432/causal_ui_gym_dev
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
DEBUG=true

# Staging
DATABASE_URL=postgresql://user:password@staging-db:5432/causal_ui_gym_staging
CORS_ORIGINS=https://staging.causal-ui-gym.com
DEBUG=false

# Production
DATABASE_URL=postgresql://user:password@prod-db:5432/causal_ui_gym_prod
CORS_ORIGINS=https://causal-ui-gym.com
DEBUG=false
```

## SSL/TLS Configuration

### Nginx SSL Configuration

```nginx
# /etc/nginx/sites-available/causal-ui-gym
server {
    listen 80;
    server_name causal-ui-gym.com www.causal-ui-gym.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name causal-ui-gym.com www.causal-ui-gym.com;

    ssl_certificate /etc/ssl/certs/causal-ui-gym.crt;
    ssl_certificate_key /etc/ssl/private/causal-ui-gym.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000" always;

    # Frontend static files
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
        
        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # API proxy
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring and Logging

### Application Monitoring

```bash
# Health check endpoint
curl -f https://causal-ui-gym.com/health

# Metrics endpoint
curl https://causal-ui-gym.com/metrics

# Application logs
docker logs causal-ui-gym-app

# System metrics
docker stats causal-ui-gym-app
```

### Log Aggregation

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup-database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="causal_ui_gym_prod"

# Create backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
  --no-password --format=custom \
  --file="$BACKUP_DIR/db_backup_$DATE.dump"

# Upload to S3
aws s3 cp "$BACKUP_DIR/db_backup_$DATE.dump" \
  "s3://your-backup-bucket/database/"

# Clean old local backups (keep last 7 days)
find $BACKUP_DIR -name "db_backup_*.dump" -mtime +7 -delete
```

### Application Data Backup

```bash
#!/bin/bash
# backup-data.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup application data
tar -czf "$BACKUP_DIR/app_data_$DATE.tar.gz" /app/data

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/app_data_$DATE.tar.gz" \
  "s3://your-backup-bucket/app-data/"

# Clean old backups
find $BACKUP_DIR -name "app_data_*.tar.gz" -mtime +30 -delete
```

## Disaster Recovery

### Recovery Procedures

1. **Database Recovery**
   ```bash
   # Restore from backup
   pg_restore -h $DB_HOST -U $DB_USER -d $DB_NAME \
     --clean --if-exists backup_file.dump
   ```

2. **Application Recovery**
   ```bash
   # Restore application data
   tar -xzf app_data_backup.tar.gz -C /

   # Restart services
   docker-compose restart
   ```

3. **DNS Failover**
   - Update DNS records to point to backup infrastructure
   - Verify SSL certificates are valid
   - Test all critical functionality

## Performance Optimization

### Application Optimization
- **Code Optimization**: Profile and optimize hot paths
- **Database Queries**: Optimize slow queries and add indexes
- **Caching**: Implement Redis caching for frequent operations
- **CDN**: Use CloudFront or similar for static assets

### Infrastructure Optimization
- **Auto Scaling**: Configure horizontal pod autoscaling
- **Load Balancing**: Distribute traffic across multiple instances
- **Resource Limits**: Set appropriate CPU and memory limits
- **Monitoring**: Set up alerts for performance degradation

## Security Considerations

### Application Security
- **HTTPS Only**: Force HTTPS for all connections
- **Input Validation**: Sanitize all user inputs
- **Authentication**: Implement proper JWT token handling
- **Rate Limiting**: Prevent abuse with rate limiting

### Infrastructure Security
- **Network Security**: Use VPCs and security groups
- **Secrets Management**: Use proper secret storage
- **Image Scanning**: Scan container images for vulnerabilities
- **Access Control**: Implement least privilege access

## Troubleshooting

### Common Issues

1. **Application Won't Start**
   - Check environment variables
   - Verify database connectivity
   - Review application logs

2. **Database Connection Issues**
   - Check database credentials
   - Verify network connectivity
   - Check connection pool settings

3. **High Memory Usage**
   - Profile JAX computations
   - Check for memory leaks
   - Optimize data structures

4. **Slow Response Times**
   - Check database query performance
   - Profile API endpoints
   - Verify caching is working

### Debug Commands

```bash
# Check container status
docker ps -a

# View application logs
docker logs -f causal-ui-gym-app

# Connect to application container
docker exec -it causal-ui-gym-app bash

# Check database connectivity
docker exec causal-ui-gym-app pg_isready -h db

# Test API endpoints
curl -v https://causal-ui-gym.com/health
```

---

*Last Updated: January 2025*  
*Version: 1.0*  
*Review Schedule: Quarterly*