#!/bin/bash
# Production Deployment Script for Causal UI Gym
# Handles zero-downtime deployment with health checks and rollback capability

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if running as root or with docker permissions
    if ! docker info &> /dev/null; then
        error "Cannot connect to Docker daemon. Please run as root or add user to docker group"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$SCRIPT_DIR/.env.production" ]; then
        warn ".env.production not found, creating from template..."
        cp "$SCRIPT_DIR/.env.production.template" "$SCRIPT_DIR/.env.production"
        warn "Please edit .env.production with your configuration before continuing"
        read -p "Press Enter to continue after editing .env.production..."
    fi
    
    # Check disk space (need at least 10GB)
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        error "Insufficient disk space. Need at least 10GB available"
        exit 1
    fi
    
    success "All prerequisites checked"
}

# Function to backup current deployment
backup_current_deployment() {
    if [ "$BACKUP_ENABLED" = "true" ]; then
        log "Creating backup of current deployment..."
        
        local backup_dir="/tmp/causal-ui-gym-backup-$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup database if running
        if docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" ps postgres | grep -q "Up"; then
            log "Backing up database..."
            docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" exec -T postgres \
                pg_dump -U causal_user causal_ui_gym > "$backup_dir/database_backup.sql"
        fi
        
        # Backup volumes
        log "Backing up persistent volumes..."
        docker run --rm -v causal-ui-gym_app_data:/data -v "$backup_dir:/backup" alpine \
            tar czf /backup/app_data_backup.tar.gz -C /data .
        
        log "Backup created at $backup_dir"
        echo "$backup_dir" > /tmp/causal-ui-gym-last-backup
    else
        log "Backup disabled, skipping..."
    fi
}

# Function to perform health check
health_check() {
    local url="${1:-http://localhost/health}"
    local timeout="${2:-$HEALTH_CHECK_TIMEOUT}"
    local interval=5
    local elapsed=0
    
    log "Performing health check on $url (timeout: ${timeout}s)..."
    
    while [ $elapsed -lt $timeout ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            success "Health check passed"
            return 0
        fi
        
        log "Health check failed, retrying in ${interval}s... (${elapsed}/${timeout}s elapsed)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    error "Health check failed after ${timeout}s"
    return 1
}

# Function to rollback deployment
rollback_deployment() {
    error "Deployment failed, initiating rollback..."
    
    # Get the last backup location
    if [ -f /tmp/causal-ui-gym-last-backup ]; then
        local backup_dir=$(cat /tmp/causal-ui-gym-last-backup)
        
        if [ -d "$backup_dir" ]; then
            log "Rolling back to backup: $backup_dir"
            
            # Stop current services
            docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" down --remove-orphans
            
            # Restore database
            if [ -f "$backup_dir/database_backup.sql" ]; then
                log "Restoring database..."
                docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" up -d postgres
                sleep 10
                docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" exec -T postgres \
                    psql -U causal_user -d causal_ui_gym < "$backup_dir/database_backup.sql"
            fi
            
            # Restore volumes
            if [ -f "$backup_dir/app_data_backup.tar.gz" ]; then
                log "Restoring application data..."
                docker run --rm -v causal-ui-gym_app_data:/data -v "$backup_dir:/backup" alpine \
                    sh -c "cd /data && tar xzf /backup/app_data_backup.tar.gz"
            fi
            
            # Start services with old image
            docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" up -d
            
            if health_check; then
                success "Rollback completed successfully"
            else
                error "Rollback failed - manual intervention required"
                exit 1
            fi
        else
            error "Backup directory not found: $backup_dir"
            exit 1
        fi
    else
        error "No backup information found for rollback"
        exit 1
    fi
}

# Function to deploy the application
deploy_application() {
    log "Starting deployment process..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Build new images
    log "Building application images..."
    docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" build --no-cache
    
    # Check if services are already running
    if docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" ps | grep -q "Up"; then
        log "Services are running, performing rolling update..."
        
        # Update services one by one for zero-downtime deployment
        services=("postgres" "redis" "causal-ui-gym" "prometheus" "grafana")
        for service in "${services[@]}"; do
            log "Updating service: $service"
            docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" up -d --no-deps "$service"
            
            # Wait for service to be healthy
            sleep 10
            
            # Check specific service health
            case $service in
                "causal-ui-gym")
                    if ! health_check; then
                        rollback_deployment
                        exit 1
                    fi
                    ;;
                "postgres")
                    if ! docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" exec postgres pg_isready -U causal_user; then
                        error "PostgreSQL health check failed"
                        rollback_deployment
                        exit 1
                    fi
                    ;;
                "redis")
                    if ! docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" exec redis redis-cli ping; then
                        error "Redis health check failed"
                        rollback_deployment
                        exit 1
                    fi
                    ;;
            esac
        done
    else
        log "No services running, performing fresh deployment..."
        docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" up -d
    fi
    
    # Final comprehensive health check
    log "Performing comprehensive health checks..."
    
    # Wait for all services to be ready
    sleep 30
    
    # Check main application
    if ! health_check; then
        rollback_deployment
        exit 1
    fi
    
    # Check API endpoints
    if ! health_check "http://localhost/api/status"; then
        rollback_deployment
        exit 1
    fi
    
    # Check WebSocket endpoint
    if ! curl -f -s -H "Connection: Upgrade" -H "Upgrade: websocket" "http://localhost/ws" > /dev/null 2>&1; then
        warn "WebSocket health check failed, but continuing deployment"
    fi
    
    success "Application deployed successfully!"
}

# Function to post-deployment tasks
post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    # Database migrations (if any)
    log "Running database migrations..."
    docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" exec causal-ui-gym \
        python -c "
from backend.database import run_migrations
run_migrations()
print('Database migrations completed')
" || warn "Database migrations failed or not available"
    
    # Cache warmup
    log "Warming up application cache..."
    docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" exec causal-ui-gym \
        python -c "
from backend.cache import warmup_cache
warmup_cache()
print('Cache warmup completed')
" || warn "Cache warmup failed or not available"
    
    # Send deployment notification (if configured)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        log "Sending deployment notification..."
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ Causal UI Gym deployed successfully to $DEPLOYMENT_ENV\"}" \
            "$SLACK_WEBHOOK_URL" || warn "Failed to send Slack notification"
    fi
    
    # Cleanup old images
    log "Cleaning up old Docker images..."
    docker image prune -f
    
    success "Post-deployment tasks completed"
}

# Function to display deployment status
display_status() {
    log "Deployment Status:"
    echo "===================="
    
    # Service status
    docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" ps
    
    echo ""
    log "Service URLs:"
    echo "Main Application: http://localhost"
    echo "API Documentation: http://localhost/docs"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000"
    echo "Kibana: http://localhost:5601"
    
    echo ""
    log "To monitor logs: docker-compose -f $SCRIPT_DIR/docker-compose.production.yml logs -f"
    log "To stop services: docker-compose -f $SCRIPT_DIR/docker-compose.production.yml down"
}

# Main deployment function
main() {
    local action="${1:-deploy}"
    
    case $action in
        "deploy"|"update")
            log "Starting production deployment for Causal UI Gym..."
            check_prerequisites
            backup_current_deployment
            deploy_application
            post_deployment_tasks
            display_status
            success "Deployment completed successfully!"
            ;;
        "rollback")
            rollback_deployment
            ;;
        "status")
            display_status
            ;;
        "health")
            health_check
            ;;
        "backup")
            backup_current_deployment
            ;;
        *)
            echo "Usage: $0 {deploy|update|rollback|status|health|backup}"
            echo "  deploy/update: Deploy or update the application"
            echo "  rollback:      Rollback to previous version"
            echo "  status:        Show current deployment status"
            echo "  health:        Run health check"
            echo "  backup:        Create backup of current deployment"
            exit 1
            ;;
    esac
}

# Trap errors and attempt rollback
trap 'error "Deployment script failed"; rollback_deployment' ERR

# Run main function
main "$@"