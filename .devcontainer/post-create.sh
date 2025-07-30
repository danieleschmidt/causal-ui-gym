#!/bin/bash

echo "🚀 Setting up Causal UI Gym development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
pip install -e .

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install --install-hooks

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data logs .secrets

# Set up Git configuration (if not already configured)
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Please set up your Git configuration:"
    echo "git config --global user.name 'Your Name'"
    echo "git config --global user.email 'your.email@example.com'"
fi

# Initialize secrets baseline
echo "🔐 Initializing secrets baseline..."
detect-secrets scan --baseline .secrets.baseline

# Set up development database
echo "🗄️  Setting up development database..."
# Wait for postgres to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! pg_isready -h postgres -p 5432 -U user; do
  sleep 1
done

# Create test database if it doesn't exist
PGPASSWORD=password psql -h postgres -U user -d causal_ui_gym -c "SELECT 1;" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Creating test database..."
    PGPASSWORD=password createdb -h postgres -U user causal_ui_gym_test
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  npm run dev          # Start frontend development server"
echo "  python -m causal_ui_gym.server  # Start backend API server"
echo "  npm run test         # Run frontend tests"
echo "  pytest               # Run backend tests"
echo "  npm run storybook    # Start Storybook"
echo ""
echo "🔍 Useful development commands:"
echo "  npm run lint         # Lint frontend code"
echo "  black .              # Format Python code"
echo "  pre-commit run --all-files  # Run all pre-commit hooks"
echo "  docker-compose up -d postgres redis  # Start supporting services"
echo ""
echo "📚 Documentation:"
echo "  README.md            # Project overview"
echo "  docs/DEVELOPMENT.md  # Development guide"
echo "  docs/ARCHITECTURE.md # Architecture overview"