#!/bin/bash

# Build script for Causal UI Gym
# Handles multi-stage builds for different environments

set -euo pipefail

# Configuration
PROJECT_NAME="causal-ui-gym"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/danieleschmidt}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
COMMIT_SHA=$(git rev-parse HEAD)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

show_help() {
    cat << EOF
Build script for Causal UI Gym

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Build environment (dev|staging|production) [default: dev]
    -t, --tag TAG           Docker image tag [default: git short SHA]
    -p, --push              Push images to registry
    -c, --clean             Clean build artifacts before building
    -m, --multi-arch        Build multi-architecture images
    -v, --verbose           Verbose output
    -h, --help              Show this help message

Examples:
    $0                      # Build for development
    $0 -e production -p     # Build and push production image
    $0 -e staging -t v1.0.0 # Build staging with specific tag
    $0 -c -m -p             # Clean, multi-arch build and push

EOF
}

cleanup() {
    log_info "Cleaning up build artifacts..."
    rm -rf dist/ build/ .tmp/
    docker system prune -f --filter "label=project=${PROJECT_NAME}" || true
}

build_frontend() {
    local env=$1
    log_info "Building frontend for ${env} environment..."
    
    # Set environment variables
    export NODE_ENV=$env
    export VITE_BUILD_DATE=$BUILD_DATE
    export VITE_VERSION=$VERSION
    export VITE_COMMIT_SHA=$COMMIT_SHA
    
    # Install dependencies
    npm ci --prefer-offline --no-audit
    
    # Run linting and type checking
    npm run lint
    npm run typecheck
    
    # Build frontend
    npm run build
    
    # Build Storybook for documentation
    if [[ "$env" == "production" ]]; then
        npm run build-storybook
    fi
    
    log_success "Frontend build completed"
}

build_docker_image() {
    local env=$1
    local tag=$2
    local push=$3
    local multi_arch=$4
    
    log_info "Building Docker image for ${env} environment..."
    
    local image_name="${DOCKER_REGISTRY}/${PROJECT_NAME}"
    local full_tag="${image_name}:${tag}"
    local latest_tag="${image_name}:latest"
    
    # Build arguments
    local build_args=(
        --build-arg BUILD_DATE="$BUILD_DATE"
        --build-arg VERSION="$VERSION"
        --build-arg COMMIT_SHA="$COMMIT_SHA"
        --build-arg BRANCH_NAME="$BRANCH_NAME"
        --label "project=${PROJECT_NAME}"
        --label "version=${VERSION}"
        --label "build-date=${BUILD_DATE}"
        --label "commit-sha=${COMMIT_SHA}"
    )
    
    # Environment-specific configurations
    case $env in
        "dev"|"development")
            build_args+=(--target development)
            ;;
        "staging")
            build_args+=(--target production)
            ;;
        "production")
            build_args+=(--target production)
            build_args+=(--build-arg NODE_ENV=production)
            ;;
    esac
    
    # Multi-architecture build
    if [[ "$multi_arch" == "true" ]]; then
        log_info "Building multi-architecture image..."
        docker buildx build \
            "${build_args[@]}" \
            --platform linux/amd64,linux/arm64 \
            --tag "$full_tag" \
            --tag "$latest_tag" \
            ${push:+--push} \
            .
    else
        # Single architecture build
        docker build \
            "${build_args[@]}" \
            --tag "$full_tag" \
            --tag "$latest_tag" \
            .
        
        if [[ "$push" == "true" ]]; then
            log_info "Pushing images to registry..."
            docker push "$full_tag"
            docker push "$latest_tag"
        fi
    fi
    
    log_success "Docker image built: $full_tag"
}

run_tests() {
    local env=$1
    log_info "Running tests for ${env} environment..."
    
    # Unit tests
    npm run test:coverage
    
    # Integration tests (if not dev)
    if [[ "$env" != "dev" ]]; then
        npm run test:integration
    fi
    
    # E2E tests (if production)
    if [[ "$env" == "production" ]]; then
        npm run test:e2e
    fi
    
    # Security checks
    npm run security-check
    
    log_success "All tests passed"
}

validate_environment() {
    log_info "Validating build environment..."
    
    # Check required tools
    command -v node >/dev/null 2>&1 || { log_error "Node.js is required"; exit 1; }
    command -v npm >/dev/null 2>&1 || { log_error "npm is required"; exit 1; }
    command -v docker >/dev/null 2>&1 || { log_error "Docker is required"; exit 1; }
    command -v git >/dev/null 2>&1 || { log_error "Git is required"; exit 1; }
    
    # Check Docker daemon
    docker info >/dev/null 2>&1 || { log_error "Docker daemon is not running"; exit 1; }
    
    # Check git repository
    git rev-parse --git-dir >/dev/null 2>&1 || { log_error "Not in a git repository"; exit 1; }
    
    log_success "Environment validation passed"
}

generate_build_info() {
    local env=$1
    log_info "Generating build information..."
    
    cat > build-info.json << EOF
{
    "project": "${PROJECT_NAME}",
    "version": "${VERSION}",
    "environment": "${env}",
    "buildDate": "${BUILD_DATE}",
    "commitSha": "${COMMIT_SHA}",
    "branchName": "${BRANCH_NAME}",
    "dockerImage": "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
}
EOF
    
    log_success "Build info generated: build-info.json"
}

# Main execution
main() {
    local environment="dev"
    local tag="$VERSION"
    local push="false"
    local clean="false"
    local multi_arch="false"
    local verbose="false"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                environment="$2"
                shift 2
                ;;
            -t|--tag)
                tag="$2"
                shift 2
                ;;
            -p|--push)
                push="true"
                shift
                ;;
            -c|--clean)
                clean="true"
                shift
                ;;
            -m|--multi-arch)
                multi_arch="true"
                shift
                ;;
            -v|--verbose)
                verbose="true"
                set -x
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate environment argument
    case $environment in
        dev|development|staging|production)
            ;;
        *)
            log_error "Invalid environment: $environment"
            log_error "Valid environments: dev, development, staging, production"
            exit 1
            ;;
    esac
    
    log_info "Starting build process..."
    log_info "Environment: $environment"
    log_info "Tag: $tag"
    log_info "Push: $push"
    log_info "Multi-arch: $multi_arch"
    
    # Clean if requested
    if [[ "$clean" == "true" ]]; then
        cleanup
    fi
    
    # Validate environment
    validate_environment
    
    # Generate build info
    generate_build_info "$environment"
    
    # Build frontend
    build_frontend "$environment"
    
    # Run tests
    run_tests "$environment"
    
    # Build Docker image
    build_docker_image "$environment" "$tag" "$push" "$multi_arch"
    
    log_success "Build completed successfully!"
    log_info "Build artifacts:"
    log_info "  Frontend: ./dist/"
    log_info "  Docker image: ${DOCKER_REGISTRY}/${PROJECT_NAME}:${tag}"
    log_info "  Build info: ./build-info.json"
}

# Handle script interruption
trap 'log_error "Build interrupted"; exit 130' INT

# Execute main function
main "$@"