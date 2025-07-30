#!/bin/bash

echo "🌅 Starting development environment..."

# Check if services are running
echo "🔍 Checking service status..."

# Check PostgreSQL
if pg_isready -h postgres -p 5432 -U user > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "⚠️  PostgreSQL is not ready - some features may not work"
fi

# Check Redis
if redis-cli -h redis ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "⚠️  Redis is not ready - caching may not work"
fi

# Update PATH if needed
if [[ ":$PATH:" != *":/workspace/node_modules/.bin:"* ]]; then
    echo "export PATH=\"/workspace/node_modules/.bin:$PATH\"" >> ~/.zshrc
fi

# Set up helpful aliases
cat >> ~/.zshrc << 'EOF'

# Causal UI Gym development aliases
alias dev-frontend="npm run dev"
alias dev-backend="python -m causal_ui_gym.server"
alias test-frontend="npm run test"
alias test-backend="pytest"
alias lint-all="npm run lint && black . && isort ."
alias format-all="prettier --write . && black . && isort ."
alias start-services="docker-compose up -d postgres redis"
alias stop-services="docker-compose down"
alias logs-backend="tail -f logs/app.log"
alias db-shell="PGPASSWORD=password psql -h postgres -U user -d causal_ui_gym"
alias redis-shell="redis-cli -h redis"

# Quick project navigation
alias cdp="cd /workspace"
alias cddocs="cd /workspace/docs"
alias cdsrc="cd /workspace/src"
alias cdtests="cd /workspace/tests"

# Development helpers
alias fresh-install="rm -rf node_modules package-lock.json && npm install"
alias clean-python="find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null"
alias rebuild-container="docker-compose down && docker-compose build --no-cache"

EOF

# Source the updated zshrc
source ~/.zshrc

# Display development tips
echo ""
echo "💡 Development tips:"
echo "  - Use 'code .' to open the project in VS Code"
echo "  - The container auto-forwards ports 3000, 5173, and 8000"
echo "  - Extensions are pre-configured for optimal development"
echo "  - Pre-commit hooks will run automatically on git commit"
echo ""
echo "🐛 Debugging:"
echo "  - Frontend: http://localhost:5173"
echo "  - Backend: http://localhost:8000"
echo "  - API docs: http://localhost:8000/docs"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo ""

echo "🎉 Development environment is ready!"