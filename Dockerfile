# Multi-stage Dockerfile for Causal UI Gym Production Deployment
# Optimized for security, performance, and container scanning compliance

# Stage 1: Frontend Build
FROM node:24-alpine AS frontend-builder

# Install security updates
RUN apk update && apk upgrade && apk add --no-cache dumb-init

# Set working directory
WORKDIR /app

# Copy package files for dependency installation
COPY package*.json ./
COPY tsconfig.json vite.config.ts ./

# Install dependencies with audit
RUN npm ci --only=production --audit && npm cache clean --force

# Copy source code
COPY src/ ./src/
COPY public/ ./public/
COPY index.html ./

# Set production environment variables
ENV NODE_ENV=production
ENV REACT_APP_API_BASE_URL=/api
ENV REACT_APP_WS_URL=wss://localhost/ws
ENV VITE_BUILD_SOURCEMAP=false

# Build optimized frontend
RUN npm run build && npm run build:analyze

# Stage 2: Python Backend Build
FROM python:3.13.7-slim AS backend-builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=random

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt pyproject.toml ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Stage 3: Production Runtime
FROM python:3.13.7-slim AS production

# Metadata labels for container scanning
LABEL maintainer="Terragon Labs <security@terragonlabs.com>" \
      version="1.0.0" \
      description="Causal UI Gym - Production Ready Container" \
      org.opencontainers.image.title="causal-ui-gym" \
      org.opencontainers.image.description="React + JAX Causal Inference Platform" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/terragonlabs/causal-ui-gym"

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y \
    tini \
    curl \
    nginx \
    supervisor \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create non-root user with restricted permissions
RUN groupadd -r -g 10001 causalapp && \
    useradd -r -u 10001 -g causalapp -d /app -s /sbin/nologin causalapp

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    NODE_ENV=production \
    ENVIRONMENT=production \
    PYTHONPATH=/app/backend \
    PORT=8000

# Set working directory
WORKDIR /app

# Copy Python virtual environment from builder
COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy built frontend assets
COPY --from=frontend-builder /app/dist ./frontend/dist

# Copy backend application
COPY --from=backend-builder /app/backend ./backend

# Create directory structure and copy configuration files
RUN mkdir -p /app/logs /app/tmp /var/log/nginx
COPY deployment/nginx.conf /etc/nginx/nginx.conf
COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY deployment/entrypoint.sh /app/entrypoint.sh

# Set proper file permissions
RUN chown -R causalapp:causalapp /app /var/log/nginx && \
    chmod +x /app/entrypoint.sh && \
    chmod 644 /etc/nginx/nginx.conf && \
    chmod 644 /etc/supervisor/conf.d/supervisord.conf

# Create production environment template
COPY deployment/.env.production.template /app/.env.template

# Advanced health check with multiple endpoints
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost/health && \
        curl -f http://localhost/api/status && \
        curl -f http://localhost/ready || exit 1

# Expose HTTP and application ports
EXPOSE 80 8000

# Switch to non-root user for security
USER causalapp

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]

# Default command runs supervisor to manage multiple processes
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf", "-n"]