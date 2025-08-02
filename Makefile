# Makefile for Causal UI Gym

# Variables
PROJECT_NAME = causal-ui-gym
DOCKER_REGISTRY = ghcr.io/danieleschmidt
VERSION ?= $(shell git rev-parse --short HEAD)
IMAGE_TAG = $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(VERSION)
LATEST_TAG = $(DOCKER_REGISTRY)/$(PROJECT_NAME):latest

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Causal UI Gym - Build and Development Commands"
	@echo "=============================================="
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Development commands
.PHONY: install
install: ## Install all dependencies
	npm install
	pip install -r requirements.txt

.PHONY: dev
dev: ## Start development servers
	docker-compose up frontend backend postgres redis

.PHONY: dev-full
dev-full: ## Start all services including monitoring
	docker-compose --profile monitoring up

.PHONY: stop
stop: ## Stop all services
	docker-compose down

.PHONY: clean
clean: ## Clean up containers, volumes, and build artifacts
	docker-compose down -v --remove-orphans
	docker system prune -f
	rm -rf dist/ build/ coverage/ .nyc_output/
	npm run clean

# Testing commands
.PHONY: test
test: ## Run all tests
	npm run test
	npm run test:e2e
	python -m pytest tests/performance/

.PHONY: test-unit
test-unit: ## Run unit tests only
	npm run test:unit

.PHONY: test-integration
test-integration: ## Run integration tests only
	npm run test:integration

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	npm run test:e2e

.PHONY: test-performance
test-performance: ## Run performance tests
	python -m pytest tests/performance/ -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	npm run test:coverage

# Code quality commands
.PHONY: lint
lint: ## Run linting
	npm run lint
	python -m flake8 .
	python -m mypy .

.PHONY: lint-fix
lint-fix: ## Fix linting issues
	npm run lint:fix
	python -m black .
	python -m isort .

.PHONY: format
format: ## Format code
	npm run format
	python -m black .
	python -m isort .

.PHONY: typecheck
typecheck: ## Run type checking
	npm run typecheck
	python -m mypy .

.PHONY: security-check
security-check: ## Run security checks
	npm audit --audit-level moderate
	python -m bandit -r . -f json
	python -m safety check

.PHONY: validate
validate: ## Run full validation (lint, typecheck, test, security)
	make lint
	make typecheck
	make test
	make security-check

# Build commands
.PHONY: build-frontend
build-frontend: ## Build frontend for production
	npm run build

.PHONY: build-storybook
build-storybook: ## Build Storybook
	npm run build-storybook

.PHONY: build-docker
build-docker: ## Build Docker image
	docker build -t $(IMAGE_TAG) -t $(LATEST_TAG) .

.PHONY: build-docker-dev
build-docker-dev: ## Build Docker image for development
	docker build --target frontend-builder -t $(PROJECT_NAME):dev .

.PHONY: build-multi-arch
build-multi-arch: ## Build multi-architecture Docker image
	docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE_TAG) -t $(LATEST_TAG) --push .

# Deployment commands
.PHONY: deploy-staging
deploy-staging: ## Deploy to staging environment
	docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

.PHONY: deploy-production
deploy-production: ## Deploy to production environment
	docker-compose -f docker-compose.yml -f docker-compose.production.yml --profile production up -d

.PHONY: deploy-monitoring
deploy-monitoring: ## Deploy monitoring stack
	docker-compose --profile monitoring up -d prometheus grafana jaeger

# Database commands
.PHONY: db-migrate
db-migrate: ## Run database migrations
	docker-compose exec backend python -m alembic upgrade head

.PHONY: db-seed
db-seed: ## Seed database with test data
	docker-compose exec backend python scripts/seed_database.py

.PHONY: db-reset
db-reset: ## Reset database (drop and recreate)
	docker-compose down postgres
	docker volume rm causal-ui-gym_postgres_data
	docker-compose up -d postgres
	sleep 5
	make db-migrate

.PHONY: db-backup
db-backup: ## Backup database
	mkdir -p backups
	docker-compose exec postgres pg_dump -U user causal_ui_gym > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

.PHONY: db-restore
db-restore: ## Restore database from backup (requires BACKUP_FILE variable)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Usage: make db-restore BACKUP_FILE=backup_file.sql"; exit 1; fi
	docker-compose exec -T postgres psql -U user causal_ui_gym < $(BACKUP_FILE)

# Monitoring and debugging commands
.PHONY: logs
logs: ## Show logs for all services
	docker-compose logs -f

.PHONY: logs-backend
logs-backend: ## Show backend logs
	docker-compose logs -f backend

.PHONY: logs-frontend
logs-frontend: ## Show frontend logs
	docker-compose logs -f frontend

.PHONY: shell-backend
shell-backend: ## Open shell in backend container
	docker-compose exec backend /bin/bash

.PHONY: shell-frontend
shell-frontend: ## Open shell in frontend container
	docker-compose exec frontend /bin/sh

.PHONY: stats
stats: ## Show container resource usage
	docker stats

.PHONY: health
health: ## Check health of all services
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health || echo "Backend: UNHEALTHY"
	@curl -f http://localhost:5173 || echo "Frontend: UNHEALTHY"
	@docker-compose ps

# Performance and profiling commands
.PHONY: profile-backend
profile-backend: ## Profile backend performance
	docker-compose exec backend python -m cProfile -o profile.stats -m causal_ui_gym.server

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	docker-compose exec backend python performance-benchmarks.py
	npm run test:performance

.PHONY: load-test
load-test: ## Run load tests with k6
	docker run --rm -i --network host grafana/k6 run - < tests/performance/k6-load-test.js

# Documentation commands
.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@if command -v mkdocs > /dev/null; then \
		mkdocs serve; \
	else \
		echo "mkdocs not found. Install with: pip install mkdocs mkdocs-material"; \
	fi

.PHONY: docs-build
docs-build: ## Build documentation
	@if command -v mkdocs > /dev/null; then \
		mkdocs build; \
	else \
		echo "mkdocs not found. Install with: pip install mkdocs mkdocs-material"; \
	fi

# Utility commands
.PHONY: ps
ps: ## Show running containers
	docker-compose ps

.PHONY: images
images: ## Show Docker images
	docker images | grep $(PROJECT_NAME)

.PHONY: prune
prune: ## Clean up unused Docker resources
	docker system prune -af --volumes

.PHONY: env-check
env-check: ## Check environment setup
	@echo "Checking environment..."
	@command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed"; exit 1; }
	@command -v python >/dev/null 2>&1 || { echo "Python is required but not installed"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed"; exit 1; }
	@echo "Environment check passed ✓"

.PHONY: setup
setup: env-check install ## Complete development setup
	@echo "Setting up development environment..."
	cp .env.example .env
	docker-compose pull
	@echo "Setup completed! Run 'make dev' to start development servers."

# CI/CD commands
.PHONY: ci-install
ci-install: ## Install dependencies in CI environment
	npm ci
	pip install -r requirements.txt

.PHONY: ci-test
ci-test: ## Run tests in CI environment
	npm run test:coverage
	npm run test:e2e
	python -m pytest tests/performance/ --maxfail=1

.PHONY: ci-build
ci-build: ## Build for CI/CD pipeline
	npm run build
	docker build -t $(IMAGE_TAG) .

.PHONY: ci-deploy
ci-deploy: ## Deploy in CI/CD pipeline
	@echo "Deploying $(IMAGE_TAG)..."
	docker push $(IMAGE_TAG)
	docker push $(LATEST_TAG)

# Release commands
.PHONY: release-patch
release-patch: ## Create patch release
	npm version patch
	git push --tags

.PHONY: release-minor
release-minor: ## Create minor release
	npm version minor
	git push --tags

.PHONY: release-major
release-major: ## Create major release
	npm version major
	git push --tags

# Maintenance commands
.PHONY: update-deps
update-deps: ## Update dependencies
	npm update
	pip-review --auto

.PHONY: outdated
outdated: ## Check for outdated dependencies
	npm outdated
	pip list --outdated

.PHONY: audit
audit: ## Security audit
	npm audit
	pip-audit

# Special targets
.PHONY: all
all: validate build-docker ## Run validation and build

.PHONY: quick
quick: lint-fix test-unit ## Quick development check

.DEFAULT_GOAL := help