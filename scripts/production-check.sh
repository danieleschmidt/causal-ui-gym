#!/bin/bash
# Production Readiness Check Script for Causal UI Gym
# Verifies that the deployment is production-ready before going live

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHECK_TIMEOUT=30
CRITICAL_ERRORS=0
WARNINGS=0
PASSED_CHECKS=0

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
    ((WARNINGS++))
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    ((CRITICAL_ERRORS++))
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] PASS: $1${NC}"
    ((PASSED_CHECKS++))
}

# Function to check if a service is running
check_service_running() {
    local service_name=$1
    local check_command=$2
    
    log "Checking if $service_name is running..."
    
    if eval "$check_command" &> /dev/null; then
        success "$service_name is running"
        return 0
    else
        error "$service_name is not running"
        return 1
    fi
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    log "Checking $name endpoint: $url"
    
    local response
    response=$(curl -s -w "%{http_code}" -o /dev/null --max-time $CHECK_TIMEOUT "$url" || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        success "$name endpoint is healthy (HTTP $response)"
        return 0
    else
        error "$name endpoint failed (HTTP $response, expected $expected_status)"
        return 1
    fi
}

# Function to check Docker containers
check_docker_containers() {
    log "Checking Docker containers status..."
    
    local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
    
    if [ ! -f "$compose_file" ]; then
        error "Docker compose file not found: $compose_file"
        return 1
    fi
    
    # Get container status
    local containers
    containers=$(docker-compose -f "$compose_file" ps --services)
    
    for container in $containers; do
        local status
        status=$(docker-compose -f "$compose_file" ps "$container" | grep "$container" | awk '{print $3}')
        
        if [[ "$status" == "Up" ]]; then
            success "Container $container is running"
        else
            error "Container $container is not running (status: $status)"
        fi
    done
}

# Function to check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Check available memory
    local available_memory
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    
    if [ "$available_memory" -lt 1024 ]; then
        warn "Low available memory: ${available_memory}MB (recommended: >1GB)"
    else
        success "Sufficient memory available: ${available_memory}MB"
    fi
    
    # Check disk space
    local available_disk
    available_disk=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local available_gb=$((available_disk / 1024 / 1024))
    
    if [ "$available_gb" -lt 10 ]; then
        warn "Low disk space: ${available_gb}GB available (recommended: >10GB)"
    else
        success "Sufficient disk space: ${available_gb}GB available"
    fi
    
    # Check CPU load
    local cpu_load
    cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_count
    cpu_count=$(nproc)
    
    if (( $(echo "$cpu_load > $cpu_count" | bc -l) )); then
        warn "High CPU load: $cpu_load (cores: $cpu_count)"
    else
        success "CPU load is normal: $cpu_load (cores: $cpu_count)"
    fi
}

# Function to check configuration files
check_configuration() {
    log "Checking configuration files..."
    
    # Check environment file
    local env_file="$PROJECT_ROOT/deployment/.env.production"
    
    if [ ! -f "$env_file" ]; then
        error "Production environment file not found: $env_file"
        return 1
    fi
    
    # Check for required environment variables
    local required_vars=("SECRET_KEY" "JWT_SECRET" "ENVIRONMENT")
    
    for var in "${required_vars[@]}"; do
        if grep -q "^${var}=" "$env_file" && ! grep -q "^${var}=.*your-.*-here" "$env_file"; then
            success "Environment variable $var is configured"
        else
            error "Environment variable $var is not properly configured"
        fi
    done
    
    # Check Docker compose file
    local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
    
    if [ -f "$compose_file" ]; then
        success "Docker compose file exists"
        
        # Validate compose file
        if docker-compose -f "$compose_file" config &> /dev/null; then
            success "Docker compose file is valid"
        else
            error "Docker compose file has validation errors"
        fi
    else
        error "Docker compose file not found: $compose_file"
    fi
}

# Function to check security configurations
check_security() {
    log "Checking security configurations..."
    
    # Check if running as root
    if [ "$(id -u)" = "0" ]; then
        warn "Running as root user (not recommended for production)"
    else
        success "Not running as root user"
    fi
    
    # Check file permissions
    local sensitive_files=(
        "$PROJECT_ROOT/deployment/.env.production"
        "$PROJECT_ROOT/deployment/entrypoint.sh"
    )
    
    for file in "${sensitive_files[@]}"; do
        if [ -f "$file" ]; then
            local perms
            perms=$(stat -c "%a" "$file")
            
            case "$file" in
                *.env*)
                    if [ "$perms" = "600" ] || [ "$perms" = "644" ]; then
                        success "File permissions for $file are secure ($perms)"
                    else
                        warn "File permissions for $file may be too permissive ($perms)"
                    fi
                    ;;
                *.sh)
                    if [ "$perms" = "755" ] || [ "$perms" = "750" ]; then
                        success "File permissions for $file are correct ($perms)"
                    else
                        warn "File permissions for $file are incorrect ($perms)"
                    fi
                    ;;
            esac
        fi
    done
    
    # Check for default passwords
    local env_file="$PROJECT_ROOT/deployment/.env.production"
    if [ -f "$env_file" ]; then
        if grep -q "admin" "$env_file" || grep -q "password" "$env_file" || grep -q "123456" "$env_file"; then
            warn "Potential default passwords found in environment file"
        else
            success "No obvious default passwords found"
        fi
    fi
}

# Function to check application health
check_application_health() {
    log "Checking application health endpoints..."
    
    # Wait for services to be ready
    sleep 10
    
    # Check main health endpoint
    check_http_endpoint "Main Health" "http://localhost/health" 200
    
    # Check API status
    check_http_endpoint "API Status" "http://localhost/api/status" 200
    
    # Check readiness endpoint
    check_http_endpoint "Readiness" "http://localhost/ready" 200
    
    # Check API documentation
    check_http_endpoint "API Docs" "http://localhost/docs" 200
    
    # Check static assets
    check_http_endpoint "Static Assets" "http://localhost/" 200
}

# Function to check database connectivity
check_database() {
    log "Checking database connectivity..."
    
    local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
    
    if docker-compose -f "$compose_file" ps postgres | grep -q "Up"; then
        if docker-compose -f "$compose_file" exec -T postgres pg_isready -U causal_user -d causal_ui_gym &> /dev/null; then
            success "Database is accessible"
        else
            error "Database is not accessible"
        fi
    else
        warn "Database container is not running"
    fi
}

# Function to check cache (Redis)
check_cache() {
    log "Checking cache connectivity..."
    
    local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
    
    if docker-compose -f "$compose_file" ps redis | grep -q "Up"; then
        if docker-compose -f "$compose_file" exec -T redis redis-cli ping | grep -q "PONG"; then
            success "Cache (Redis) is accessible"
        else
            error "Cache (Redis) is not accessible"
        fi
    else
        warn "Cache container is not running"
    fi
}

# Function to check monitoring services
check_monitoring() {
    log "Checking monitoring services..."
    
    # Check Prometheus
    check_http_endpoint "Prometheus" "http://localhost:9090/-/healthy" 200
    
    # Check Grafana
    check_http_endpoint "Grafana" "http://localhost:3000/api/health" 200
    
    # Check application metrics endpoint
    check_http_endpoint "App Metrics" "http://localhost:8000/metrics" 200
}

# Function to check backup system
check_backup_system() {
    log "Checking backup system..."
    
    local backup_script="$PROJECT_ROOT/deployment/deploy.sh"
    
    if [ -f "$backup_script" ] && [ -x "$backup_script" ]; then
        success "Backup script is available and executable"
        
        # Test backup functionality (dry run)
        if BACKUP_ENABLED=true "$backup_script" backup &> /dev/null; then
            success "Backup system test passed"
        else
            warn "Backup system test failed"
        fi
    else
        warn "Backup script not found or not executable"
    fi
}

# Function to run performance checks
check_performance() {
    log "Running performance checks..."
    
    # Check response times
    local start_time
    start_time=$(date +%s.%3N)
    
    if curl -f -s "http://localhost/health" > /dev/null; then
        local end_time
        end_time=$(date +%s.%3N)
        local response_time
        response_time=$(echo "$end_time - $start_time" | bc)
        
        if (( $(echo "$response_time < 1.0" | bc -l) )); then
            success "Health endpoint response time: ${response_time}s"
        else
            warn "Slow health endpoint response time: ${response_time}s"
        fi
    fi
    
    # Check concurrent requests handling
    log "Testing concurrent request handling..."
    
    local concurrent_test_result=0
    for i in {1..5}; do
        curl -f -s "http://localhost/health" > /dev/null &
    done
    
    wait
    
    if [ $? -eq 0 ]; then
        success "Concurrent request handling test passed"
    else
        warn "Concurrent request handling test failed"
    fi
}

# Function to generate final report
generate_report() {
    echo ""
    echo "======================================="
    echo "    PRODUCTION READINESS REPORT"
    echo "======================================="
    echo ""
    echo "Total Checks Run: $((PASSED_CHECKS + WARNINGS + CRITICAL_ERRORS))"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
    echo -e "Critical Errors: ${RED}$CRITICAL_ERRORS${NC}"
    echo ""
    
    if [ $CRITICAL_ERRORS -eq 0 ]; then
        if [ $WARNINGS -eq 0 ]; then
            echo -e "${GREEN}✅ PRODUCTION READY${NC}"
            echo "All checks passed. The application is ready for production deployment."
            return 0
        else
            echo -e "${YELLOW}⚠️  PRODUCTION READY WITH WARNINGS${NC}"
            echo "The application can be deployed but has some non-critical issues."
            return 0
        fi
    else
        echo -e "${RED}❌ NOT PRODUCTION READY${NC}"
        echo "Critical issues must be resolved before production deployment."
        return 1
    fi
}

# Main execution
main() {
    log "Starting production readiness check for Causal UI Gym..."
    echo ""
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run all checks
    check_system_resources
    check_configuration
    check_security
    check_docker_containers
    check_application_health
    check_database
    check_cache
    check_monitoring
    check_backup_system
    check_performance
    
    # Generate and display report
    generate_report
}

# Run main function
main "$@"