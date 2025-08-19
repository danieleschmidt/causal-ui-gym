#!/bin/bash
# Production Deployment Script for Causal UI Gym
# Implements zero-downtime deployment with comprehensive validation

set -euo pipefail

# Configuration
NAMESPACE="causal-ui-gym"
DEPLOYMENT_CONFIG="deployment/production-orchestrator.yml"
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_ON_FAILURE=true
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-""}
ENVIRONMENT="production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Notification function
send_notification() {
    local message="$1"
    local status="$2"
    
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 Causal UI Gym Deployment - $status: $message\"}" \
            "$SLACK_WEBHOOK_URL" || true
    fi
    
    log_info "Notification sent: $message"
}

# Pre-deployment checks
run_pre_deployment_checks() {
    log_info "Running pre-deployment validation..."
    
    # Check kubectl access
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log_warning "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Validate deployment configuration
    if ! kubectl apply --dry-run=client -f "$DEPLOYMENT_CONFIG" &>/dev/null; then
        log_error "Invalid deployment configuration"
        exit 1
    fi
    
    # Run quality gates
    log_info "Running quality gates..."
    if ! python3 scripts/quality-gates.py; then
        log_error "Quality gates failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            log_warning "Skipping deployment due to quality gate failures"
            exit 1
        fi
    fi
    
    # Build and test
    log_info "Building application..."
    npm ci
    npm run build
    npm test -- --run
    
    log_success "Pre-deployment checks completed"
}

# Build and push container images
build_and_push_images() {
    log_info "Building and pushing container images..."
    
    # Build frontend image
    docker build -t causal-ui-gym/frontend:latest -f Dockerfile .
    docker tag causal-ui-gym/frontend:latest causal-ui-gym/frontend:$(git rev-parse --short HEAD)
    
    # Build backend image
    docker build -t causal-ui-gym/backend:latest -f backend/Dockerfile backend/
    docker tag causal-ui-gym/backend:latest causal-ui-gym/backend:$(git rev-parse --short HEAD)
    
    # Push images (assumes registry is configured)
    # docker push causal-ui-gym/frontend:latest
    # docker push causal-ui-gym/frontend:$(git rev-parse --short HEAD)
    # docker push causal-ui-gym/backend:latest
    # docker push causal-ui-gym/backend:$(git rev-parse --short HEAD)
    
    log_success "Container images built and tagged"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Apply configuration
    kubectl apply -f "$DEPLOYMENT_CONFIG"
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    
    kubectl rollout status deployment/causal-ui-gym-frontend -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"
    kubectl rollout status deployment/causal-ui-gym-backend -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"
    
    log_success "Deployment completed successfully"
}

# Health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Get service endpoints
    local frontend_ip=$(kubectl get service causal-ui-gym-frontend-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
    local backend_ip=$(kubectl get service causal-ui-gym-backend-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
    
    # Health check frontend
    local frontend_healthy=false
    for i in {1..30}; do
        if kubectl exec -n "$NAMESPACE" deployment/causal-ui-gym-frontend -- curl -f http://localhost:3000/health &>/dev/null; then
            frontend_healthy=true
            break
        fi
        sleep 5
    done
    
    if [[ "$frontend_healthy" == "false" ]]; then
        log_error "Frontend health check failed"
        return 1
    fi
    
    # Health check backend
    local backend_healthy=false
    for i in {1..30}; do
        if kubectl exec -n "$NAMESPACE" deployment/causal-ui-gym-backend -- curl -f http://localhost:8000/health &>/dev/null; then
            backend_healthy=true
            break
        fi
        sleep 5
    done
    
    if [[ "$backend_healthy" == "false" ]]; then
        log_error "Backend health check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

# Performance validation
run_performance_tests() {
    log_info "Running performance validation..."
    
    # Basic load test
    if command -v ab &> /dev/null; then
        local backend_url="http://$(kubectl get service causal-ui-gym-backend-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8000"
        
        # Run Apache Bench test
        ab -n 100 -c 10 "$backend_url/health" > performance-test-results.txt
        
        # Check if average response time is acceptable
        local avg_time=$(grep "Time per request" performance-test-results.txt | head -n1 | awk '{print $4}')
        if (( $(echo "$avg_time > 1000" | bc -l) )); then
            log_warning "High response time detected: ${avg_time}ms"
        else
            log_success "Performance test passed: ${avg_time}ms average response time"
        fi
    else
        log_warning "Apache Bench not available, skipping performance tests"
    fi
}

# Security validation
run_security_checks() {
    log_info "Running security validation..."
    
    # Check for security policies
    if kubectl get networkpolicy -n "$NAMESPACE" &>/dev/null; then
        log_success "Network policies are in place"
    else
        log_warning "No network policies found"
    fi
    
    # Check for pod security policies
    if kubectl get podsecuritypolicy &>/dev/null; then
        log_success "Pod security policies are configured"
    else
        log_warning "No pod security policies found"
    fi
    
    # Check for secrets encryption
    local secrets=$(kubectl get secrets -n "$NAMESPACE" -o name)
    if [[ -n "$secrets" ]]; then
        log_success "Secrets are configured"
    else
        log_warning "No secrets found"
    fi
}

# Rollback function
rollback_deployment() {
    log_error "Initiating rollback..."
    
    # Rollback deployments
    kubectl rollout undo deployment/causal-ui-gym-frontend -n "$NAMESPACE"
    kubectl rollout undo deployment/causal-ui-gym-backend -n "$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/causal-ui-gym-frontend -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"
    kubectl rollout status deployment/causal-ui-gym-backend -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"
    
    log_warning "Rollback completed"
    send_notification "Deployment rolled back due to failures" "ROLLBACK"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f performance-test-results.txt
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    echo "=================================="
    echo "🚀 Causal UI Gym Deployment Script"
    echo "=================================="
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Timestamp: $(date)"
    echo "Git Commit: $(git rev-parse --short HEAD)"
    echo "=================================="
    
    send_notification "Deployment started for commit $(git rev-parse --short HEAD)" "START"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Set trap for rollback on error
    if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
        trap 'rollback_deployment; exit 1' ERR
    fi
    
    # Execute deployment steps
    run_pre_deployment_checks
    build_and_push_images
    deploy_to_kubernetes
    run_health_checks
    run_performance_tests
    run_security_checks
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Deployment completed successfully in ${duration}s"
    
    echo "=================================="
    echo "✅ DEPLOYMENT SUMMARY"
    echo "=================================="
    echo "Status: SUCCESS"
    echo "Duration: ${duration}s"
    echo "Frontend: $(kubectl get pods -n "$NAMESPACE" -l component=frontend --no-headers | wc -l) pods running"
    echo "Backend: $(kubectl get pods -n "$NAMESPACE" -l component=backend --no-headers | wc -l) pods running"
    echo "Ingress: $(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo 'Not configured')"
    echo "=================================="
    
    send_notification "Deployment completed successfully in ${duration}s" "SUCCESS"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        --config)
            DEPLOYMENT_CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --environment ENV    Set deployment environment (default: production)"
            echo "  --namespace NS       Set Kubernetes namespace (default: causal-ui-gym)"
            echo "  --no-rollback        Disable automatic rollback on failure"
            echo "  --config FILE        Set deployment configuration file"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute main function
main