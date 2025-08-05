# Causal UI Gym - Production Deployment Guide

This directory contains production-ready deployment configurations and scripts for the Causal UI Gym application.

## 📁 Directory Structure

```
deployment/
├── README.md                       # This file
├── Dockerfile                      # Production Docker image
├── docker-compose.production.yml   # Docker Compose for production
├── kubernetes.yml                  # Kubernetes deployment manifests
├── deploy.sh                      # Automated deployment script
├── entrypoint.sh                  # Container entrypoint script
├── nginx.conf                     # Nginx reverse proxy config
├── supervisord.conf               # Process supervisor config
└── .env.production.template       # Environment variables template
```

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 8GB RAM and 20GB disk space
- Network access for downloading dependencies

### 1. Environment Setup

```bash
# Copy environment template
cp .env.production.template .env.production

# Edit configuration (required!)
nano .env.production
```

**Important**: Update these critical values in `.env.production`:
- `SECRET_KEY` - Generate a secure random key
- `JWT_SECRET` - Generate a secure random key  
- `OPENAI_API_KEY` - Your OpenAI API key
- `ANTHROPIC_API_KEY` - Your Anthropic API key
- `DB_PASSWORD` - Secure database password

### 2. Deploy with Docker Compose

```bash
# Make deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh deploy
```

### 3. Verify Deployment

The script will automatically perform health checks. You can also manually verify:

```bash
# Check deployment status
./deploy.sh status

# Run health checks
./deploy.sh health

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

## 🌐 Access Points

After successful deployment:

- **Main Application**: http://localhost
- **API Documentation**: http://localhost/docs
- **API Health Check**: http://localhost/health
- **Prometheus Metrics**: http://localhost:9090
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Kibana Logs**: http://localhost:5601

## 🎛️ Deployment Options

### Docker Compose (Recommended)

Best for single-server deployments or development environments.

```bash
# Standard deployment
./deploy.sh deploy

# Update existing deployment
./deploy.sh update

# Rollback to previous version
./deploy.sh rollback
```

### Kubernetes

For production clusters with auto-scaling and high availability.

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes.yml

# Check deployment status
kubectl get pods -n causal-ui-gym

# View logs
kubectl logs -f deployment/causal-ui-gym -n causal-ui-gym
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env.production`:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `SECRET_KEY` | Application secret key | *Required* |
| `OPENAI_API_KEY` | OpenAI API key | *Required* |
| `REDIS_ENABLED` | Enable Redis caching | `true` |
| `METRICS_ENABLED` | Enable Prometheus metrics | `true` |
| `WORKER_COUNT` | Number of API workers | `4` |

### Resource Requirements

**Minimum System Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Disk: 20GB
- Network: 1 Mbps

**Recommended Production:**
- CPU: 4+ cores
- RAM: 8GB+
- Disk: 50GB+ SSD
- Network: 10+ Mbps

### Scaling Configuration

#### Docker Compose Scaling
```bash
# Scale API workers
docker-compose -f docker-compose.production.yml up -d --scale causal-ui-gym=3
```

#### Kubernetes Auto-scaling
Auto-scaling is configured via HorizontalPodAutoscaler:
- Min replicas: 3
- Max replicas: 10
- CPU threshold: 70%
- Memory threshold: 80%

## 🔒 Security Features

### Container Security
- Non-root user execution
- Read-only file systems where possible
- Minimal base images (Alpine Linux)
- Regular security updates
- Secret management via environment variables

### Network Security
- Nginx reverse proxy with rate limiting
- HTTPS/TLS termination ready
- Internal network isolation
- Security headers configured

### Application Security
- Input validation and sanitization
- SQL injection protection
- XSS protection headers
- CORS configuration
- Authentication and authorization ready

## 📊 Monitoring & Observability

### Metrics Collection
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **Application metrics**: Custom business metrics

### Logging
- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Structured logging**: JSON format
- **Log aggregation**: Centralized log collection
- **Log retention**: Configurable retention policies

### Health Checks
- **Liveness probes**: Application health
- **Readiness probes**: Service availability
- **Startup probes**: Container initialization
- **Custom health endpoints**: `/health`, `/ready`

## 🔄 Backup & Recovery

### Automated Backups
```bash
# Create manual backup
./deploy.sh backup

# Backups are automatically created before deployments
# Location: /tmp/causal-ui-gym-backup-YYYYMMDD_HHMMSS
```

### Backup Strategy
- **Database**: PostgreSQL dump every 24 hours
- **Application data**: Persistent volume snapshots
- **Configuration**: Environment files and secrets
- **Retention**: 30 days (configurable)

### Disaster Recovery
```bash
# Rollback to previous version
./deploy.sh rollback

# Restore from specific backup
# 1. Stop services
docker-compose -f docker-compose.production.yml down

# 2. Restore database
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U causal_user -d causal_ui_gym < /path/to/backup.sql

# 3. Restart services
docker-compose -f docker-compose.production.yml up -d
```

## 🚀 Performance Optimization

### Application Performance
- **Multi-worker API**: Uvicorn with 4 workers
- **Connection pooling**: Database connection management
- **Caching**: Redis for API responses and computations
- **Static asset optimization**: Nginx serving with compression

### Database Performance
- **Connection pooling**: PostgreSQL connection limits
- **Indexing**: Optimized database queries
- **Query optimization**: Efficient causal computations

### Frontend Performance
- **Static asset caching**: Long-term browser caching
- **Compression**: Gzip/Brotli compression
- **CDN ready**: Static assets can be served from CDN

## 🔧 Troubleshooting

### Common Issues

#### 1. Health Check Failures
```bash
# Check service logs
docker-compose -f docker-compose.production.yml logs causal-ui-gym

# Check specific health endpoint
curl -v http://localhost/health
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose -f docker-compose.production.yml exec postgres pg_isready

# Check database logs
docker-compose -f docker-compose.production.yml logs postgres
```

#### 3. Memory Issues
```bash
# Check container resource usage
docker stats

# Check system resources
free -h
df -h
```

#### 4. Performance Issues
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost/api/status

# Monitor with Grafana dashboards
# Visit: http://localhost:3000
```

### Debug Mode

Enable debug logging:
```bash
# Set in .env.production
LOG_LEVEL=debug

# Restart services
./deploy.sh update
```

### Log Analysis
```bash
# View all logs
docker-compose -f docker-compose.production.yml logs

# View specific service logs
docker-compose -f docker-compose.production.yml logs causal-ui-gym

# Follow logs in real-time
docker-compose -f docker-compose.production.yml logs -f --tail=100
```

## 🚀 Advanced Deployment

### Multi-Environment Setup
```bash
# Development
DEPLOYMENT_ENV=development ./deploy.sh deploy

# Staging
DEPLOYMENT_ENV=staging ./deploy.sh deploy

# Production
DEPLOYMENT_ENV=production ./deploy.sh deploy
```

### Custom Configuration
```bash
# Use custom compose file
docker-compose -f docker-compose.custom.yml up -d

# Override specific services
docker-compose -f docker-compose.production.yml \
               -f docker-compose.override.yml up -d
```

### SSL/TLS Setup
1. Obtain SSL certificates (Let's Encrypt recommended)
2. Update `nginx.conf` with SSL configuration
3. Update `.env.production` with SSL settings
4. Redeploy services

## 📞 Support

### Getting Help
- **Documentation**: Check `/docs` directory
- **Logs**: Always check logs first
- **Health checks**: Use built-in health endpoints
- **Monitoring**: Use Grafana dashboards

### Reporting Issues
When reporting issues, include:
1. Deployment method (Docker Compose/Kubernetes)
2. Error messages and logs
3. System specifications
4. Configuration (without secrets)
5. Steps to reproduce

## 🔄 Maintenance

### Regular Maintenance Tasks
```bash
# Update container images
docker-compose -f docker-compose.production.yml pull
./deploy.sh update

# Clean up unused resources
docker system prune -f

# Rotate logs (automatic with logrotate)
docker-compose -f docker-compose.production.yml exec causal-ui-gym \
  python -m backend.logging.rotate

# Check security updates
docker scan causal-ui-gym:latest
```

### Monitoring Checklist
- [ ] Check application health endpoints
- [ ] Verify database connectivity
- [ ] Monitor disk space usage
- [ ] Check memory usage
- [ ] Verify backup completion
- [ ] Review security logs
- [ ] Check SSL certificate expiration

---

For additional support or questions, please refer to the main project documentation or create an issue in the project repository.